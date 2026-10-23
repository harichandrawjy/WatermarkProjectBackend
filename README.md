# Watermark API

FastAPI backend for **Aegis** — bridges the React frontend in [`../../Website/WatermarkProjectWebsite`](../../Website/WatermarkProjectWebsite) with the LWT/DCT/SVD watermark engine bundled here under [`engine/`](engine/). Handles auth (Supabase), file storage, and per-user encode history.

## Setup

From this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your Supabase project credentials:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=<anon-public-key>
SUPABASE_SERVICE_KEY=<service-role-key>
```

Never expose `SUPABASE_SERVICE_KEY` to the frontend — it bypasses Row Level Security.

Optional: `ALLOWED_ORIGINS` (comma-separated) to allowlist extra origins for CORS (e.g. a Vercel deployment URL).

Optional: `SITE_URL` — where password-recovery emails send the user back to. Defaults to `http://localhost:5173`. Set this to your deployed frontend URL in production, **and** add the same URL under Supabase Dashboard → Authentication → URL Configuration → Redirect URLs, or Supabase will refuse to redirect there and the reset link will fail.

The watermark engine in `engine/` is imported directly via `sys.path` — no `pip install`.

## Database

Run the migrations once each, in order, via Supabase Dashboard → SQL Editor:

1. [`migrations/001_add_user_id.sql`](migrations/001_add_user_id.sql) — adds `user_id UUID` and a `created_at` default to the `watermarks` table so per-user isolation and history sorting work.
2. [`migrations/002_add_watermark_keys.sql`](migrations/002_add_watermark_keys.sql) — adds `owner_key` / `media_key` (the byte-truncated, lowercased form of `owner` / `media` that the watermark actually carries), backfills them for existing rows, and adds the `UNIQUE (owner_key, media_key)` index that `/encode` relies on as its race-proof collision backstop.
3. [`migrations/003_add_profiles_short_id.sql`](migrations/003_add_profiles_short_id.sql) — adds a `profiles` table holding each account's assigned 8-character `short_id`, which is what now gets embedded as the watermark's owner.

All three are idempotent — re-running them is safe. `/encode` and `/lookup` both fail without 002, since they read and write `owner_key` / `media_key` directly; `/encode` fails without 003, since it allocates a `short_id` per user.

## Run

```powershell
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/> to see the service info JSON.

## Endpoints

### Auth (Supabase passthrough)

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/auth/register` | `{ email, password }` | `{ user, access_token \| null, refresh_token \| null, needs_confirmation }` |
| POST | `/auth/login` | `{ email, password }` | `{ user, access_token, refresh_token }` |
| POST | `/auth/refresh` | `{ refresh_token }` | Fresh token pair — called by the frontend on 401 |
| GET  | `/auth/me` | Bearer | `{ id, email }` |
| POST | `/auth/forgot-password` | `{ email }` | `{ sent: true, message }` — **always**, see below |
| POST | `/auth/reset-password` | `{ access_token, new_password }` | `{ ok: true }` |

`/auth/forgot-password` reports success for every address, including ones with no account and even when Supabase itself errors. Returning "no such user" would make the endpoint an account-enumeration oracle — anyone could probe which emails are registered. That matters here because `/lookup` already exposes owner identities publicly.

`/auth/reset-password` calls GoTrue's REST API directly with the recovery token as a Bearer header rather than using `supabase_auth.auth.update_user()`, which acts on the process-wide client's stored session. A concurrent request could swap that session underneath the call — unacceptable when changing a password. The direct call is stateless and unambiguous about whose password changes.

Supabase's built-in email sender is rate-limited to a handful of messages per hour on the free tier and often lands in spam. Configure custom SMTP (Dashboard → Project Settings → Auth → SMTP) before relying on password reset or email confirmation in a demo.

### Encode / verify / lookup

| Method | Path | Auth | Body / Params | Returns |
|---|---|---|---|---|
| POST | `/encode` | Bearer | multipart: `file`, `media_id` | `{ id, kind, watermarked_url, metadata_url, metadata, psnr_db }` |
| POST | `/verify` | — | multipart: `file`, `metadata` (JSON string) | `AnalysisResult`-shaped JSON |
| POST | `/lookup` | — | multipart: `file` | Blind-extracts owner + media IDs from the watermark, then returns the matching DB row |
| GET | `/metadata/{id}` | — | — | Metadata JSON for a prior encode |
| GET | `/files/{name}` | — | — | The watermarked output file (served via StaticFiles) |
| GET | `/me/media` | Bearer | — | `{ items: WatermarkRow[] }` — the user's own encodes, newest first |
| GET | `/me/media/{id}/metadata` | Bearer | — | Metadata JSON for that record (scoped to the caller) |
| DELETE | `/me/media/{id}` | Bearer | — | Deletes the record, freeing its `media_id` for reuse |

`owner` on `/encode` is **not** a form field — it's tied to the caller's JWT so it can't be spoofed.

Two distinct values are involved, and the difference matters:

| | what it is | where it lives |
|---|---|---|
| `owner` | the account's full email | `watermarks.owner`, returned by `/lookup` — **never embedded** |
| `short_id` | an assigned 8-char tag | `profiles.short_id`, copied to `watermarks.owner_key` — **this is what goes into the pixels** |

The engine's owner slot is `OWNER_ID_BYTES = 8`, so embedding the email directly would truncate it to the first 8 characters and merge every account sharing that prefix into one owner (`john.smith@gmail.com` and `john.smigielski@x.com` both become `john.smi`). Widening the slot isn't viable — at 512×512 the LL-family capacity is 768 bits, and a 12-byte owner id leaves zero room for the majority-vote repetition copies. So the 8 bytes now hold an **assigned** identifier (unique by constraint) instead of a **derived** prefix (unique by luck).

`/lookup` matches on `owner_key` and returns the row's full email, so the user-facing output is unchanged.

## Storage

**No user media is retained.** The server keeps metadata; it does not keep files.

- `uploads/` — deleted as soon as the request finishes, including on error paths
- `outputs/` — written so the encode response can serve a preview and a download, then left to the host's disk lifecycle. **No retention is promised.**

This is a design decision, not a limitation to work around. Encoding is
**deterministic**: the same original + owner + `media_id` produces a
byte-identical file every time, because the signature, chain tag and QIM
lattice all derive from those inputs via SHA-256. The watermarked file is
therefore a pure function of things the user already has, and it is handed to
them at encode time. What genuinely cannot be regenerated is the metadata —
which is why that lives in Postgres (`watermarks.metadata`).

Keeping user media would also contradict what the Verify page promises about
uploads. Keeping nothing makes one policy true everywhere.

Practical consequences:

- On hosts with an ephemeral filesystem (Render's free tier, most container
  platforms) `/files/<name>` works while the container is warm and 404s after a
  deploy or a wake from sleep. The Dashboard treats a miss as expected and says so.
- Because `/encode` rejects a reused `media_id` (the UNIQUE index from migration
  002), losing a downloaded file would otherwise burn that id permanently.
  `DELETE /me/media/{id}` frees it, so re-encoding under the same name is possible.

If you *do* want durable media, add object storage (Supabase Storage is the
natural fit) rather than relying on the local disk — but note a lossless FFV1
video is ~22 MB, so a 1 GB free tier holds roughly 45 of them.

## Output format (locked-in by design)

- Images → **PNG** regardless of upload format. JPEG would destroy the watermark.
- Videos → **MKV** with **FFV1** (lossless). MP4 would destroy the watermark.

Downstream re-encoding to a lossy format is correctly flagged as tampered — that is the algorithm working as designed.

## Screenshots and rescaled copies

A screenshot of a watermarked image used to fail verification completely — not
because the watermark was gone, but because it was no longer being read from
where it was written. The decoder reads fixed 32×32 pixel tiles; a screenshot
moves that grid two ways at once:

- **scale** — the image on screen is almost never 1:1 with its own pixels
  (display DPI scaling, browser zoom, fit-to-container)
- **offset** — the capture rectangle does not start on a multiple of 32

Measured, a **one pixel** crop was already enough to destroy every extracted
bit. [`engine/resync.py`](engine/resync.py) recovers from both by searching for
the transform that puts the grid back, then re-running the ordinary extraction.
Nothing about embedding changed, so existing watermarked files stay valid and
the eval numbers above are unaffected.

It runs **only after normal extraction has failed**. On `/verify` the
`width`/`height` now stored in metadata collapse the whole realistic screenshot
space to one 64-candidate offset scan, and the best agreement wins rather than
the first candidate over the threshold — a near-miss accepted greedily verifies
worse than the exact geometry it was standing next to.

On `/lookup` it runs later still: only once the database has come back empty, so
a merely recompressed file resolves through the normal path and never pays for a
search. There is deliberately no cheap "has the grid moved?" pre-test. One was
built and measured, and it does not work — an aligned frame scored as low as
**0.35x** its own wrong-phase null while a 2x upscale reached **4.39x**, because
an integer upscale leaves every other block boundary still on the grid. The two
populations overlap completely, so the search itself is the only honest test.

### It cannot return a wrong owner

The search tries thousands of geometries, so its accept test carries the
weight. Every candidate goes through the payload's 2-byte SHA-256 integrity
tag, and nothing else is allowed to confirm a hit — in particular the
majority-vote fallback is deliberately unreachable from the search, because it
has no such tag and on a misaligned image returns the *media* id in the owner
slot. Measured over 8 never-watermarked photos: 11,874 candidate geometries
scored, 372 decoded, **0 accepted**.

### What it recovers

Across the eval corpus, matched against what is even theoretically recoverable
(i.e. what still decodes when handed its exact true geometry):

| screenshot | recovered | typical time |
|---|---|---|
| crop ≤ 8px | **8/8** | 0.04 – 6.2 s |
| DPI 125% / 150% / 200% | **7/8** | 1.8 – 4.8 s |
| fit-to-container 0.8x | **7/8** | 1.7 – 4.4 s |
| rescaled **and** cropped | **7/8** | 1.8 – 4.8 s |

The search reaches the ceiling in every case: the one image it misses does not
decode even when handed its exact true geometry, so nothing could have
recovered it. That is a robustness limit of the watermark, not of the search —
fine detail shifts the LL-family singular values past what QIM can hold.

**Robustness here is content-dependent, and that matters more than the table
suggests.** A heavily textured photograph carries a much weaker watermark: one
1280×959 landscape encoded by this service fails JPEG q95 and sigma=2 noise
outright, so its owner ID cannot be read back from a screenshot no matter how
perfectly the geometry is restored. Its media ID still comes back. Run
[`diagnose_screenshot.py`](diagnose_screenshot.py) on a file to see which limit
it hit.

Out of range by construction:

- downscaling past ~0.6x — the QIM lattice itself is gone, measured no decode
  even at the true geometry
- crops larger than 8px — those blocks are lost, and only the unvalidated vote
  path could guess them
- distortions that are not scale-plus-crop — rotation, perspective, or an
  aspect ratio that changed

### Captures that caught the page around the image

A real screenshot is rarely cropped to the pixel; it carries a strip of window
chrome or page background. That strip is not a crop the offset scan can undo —
the content becomes a sub-rectangle, so every block index *and the row stride
itself* are wrong. Measured, a **2 pixel** border was enough to turn a clean
recovery into no match at all.

`/verify` now locates the content rectangle (scanning in from each edge while
the row stays flat) and scans inside it as well, keeping whichever of the two
fits better. Both are always tried, rather than trimming only when the first
attempt comes back empty — a framed upload does not come back empty, it comes
back with a near-miss that clears the threshold. On that 2px border the
untrimmed scan offered 0.945 and the trimmed one was exact; stopping at the
first acceptable answer took the worse of the two.

Measured on a 2000×2509 encode displayed at 1594×2000, borders of 1, 2, 12 and
40px all recover both ids exactly.

### Why the scan is fast

Restoring a whole 2000×2509 frame to read a payload that lives in its first
block rows cost 481 ms per candidate, which made a 64-candidate scan take 25
seconds. Only the strip the payload occupies is restored now — Haar has no
overlap beyond its 4×4 level-2 support, so those block rows decompose
identically to the same rows of the full image. The same scan is **0.7 s**.

A note on the ranking signal, because it is the part most likely to mislead
someone extending this: `lattice_score` is what makes the *blind* sweep
affordable, but it measures whether the QIM lattice survived — not whether the
payload can be read. Those diverge. On a 2000×2509 image downscaled to 1594px
the true geometry ranked **1760th of 2154** by lattice score while decoding
perfectly, because resampling scatters S[0] without flipping the parity that
carries the bit. Verification does not rank by it at all: with the dimensions
known it scans offsets directly and scores each by how far the recovered bits
agree with the ids it already holds, which measures the right thing.

### It restores geometry, not the verdict

A recovered screenshot still reports honestly. Resampling is a real
modification, so the file may still come back `tampered` — what changes is that
the report names the actual reason instead of claiming the owner ID could not
be matched, which is indistinguishable from a forgery and the wrong thing to
tell someone about their own file. When a restoration happened, the response
carries a `resynced` object (`width`, `height`, `offsetX`, `offsetY`) and leads
the `reasons` list with a plain-language note.

## Watermark engine (in `engine/`)

- [`watermark_engine.py`](engine/watermark_engine.py) — image path. Embeds bits with LWT + 8×8 DCT + SVD + QIM on the Y channel; BCH(15,7) error-correction on the payload; per-sub-block SHA-256 parity in Cb LSBs for spatial tamper localization.
- [`video_watermark.py`](engine/video_watermark.py) — per-frame embed with frame-ID + chain-tag bits so deletions, duplicates, and reorders surface as discrete temporal events instead of cascading errors.
- [`resync.py`](engine/resync.py) — decode-side only. Finds the geometry of a rescaled or cropped copy so the other two can read it; see [Screenshots and rescaled copies](#screenshots-and-rescaled-copies).

Constants exposed by the engine (`OWNER_ID_BITS`, `MEDIA_ID_BITS`, `FRAME_ID_BITS`, `CHAIN_TAG_BITS`, `SPATIAL_BLOCK`, `META_PAYLOAD_BYTES`, BCH `_BCH_N` / `_BCH_K`) are imported by `main.py` to keep encode-side and verify-side in lockstep.

## Auth architecture (`auth.py`, `db.py`)

Two Supabase clients:

- `supabase` — created with **service role key**, bypasses RLS. Used for every `.table(...)` operation. Authorization is enforced in endpoint code (`get_current_user` + per-row `user_id` filters), not by RLS.
- `supabase_auth` — created with the **anon key**. Used for `sign_up`, `sign_in_with_password`, `refresh_session`, `get_user(token)`. Kept separate so a session stored after sign-in never leaks into data operations on the trusted client.

`get_current_user` (in `auth.py`) is a FastAPI dependency: reads `Authorization: Bearer <jwt>`, resolves the user via `supabase_auth.auth.get_user(token)`, injects `{ id, email, token }` into the endpoint.

The backend **never** touches passwords — Supabase handles hashing, JWT signing, and email confirmation.

## Quick test (PowerShell)

```powershell
# 1. Register (or login) to get an access token
$body = @{ email = "you@example.com"; password = "hunter2" } | ConvertTo-Json
$auth = curl.exe -s -X POST http://localhost:8000/auth/login `
  -H "Content-Type: application/json" -d $body | ConvertFrom-Json
$tok  = $auth.access_token

# 2. Encode
curl.exe -X POST http://localhost:8000/encode `
  -H "Authorization: Bearer $tok" `
  -F "file=@./engine/sample.jpg" `
  -F "media_id=img_001"

# 3. Verify (no auth needed — anyone with the metadata can verify)
curl.exe -X POST http://localhost:8000/verify `
  -F "file=@./storage/outputs/<watermarked>.png" `
  -F "metadata=<contents of metadata.json>"
```

## Connecting the frontend

The frontend defaults to `http://localhost:8000`, overridable via `VITE_API_URL` in a `.env` at the Vite project root. `Encode.tsx`, `Verify.tsx`, and `Dashboard.tsx` already send `FormData` to these endpoints and attach `Authorization: Bearer <token>` via the `authedFetch` helper from [`src/context/auth.tsx`](../../Website/WatermarkProjectWebsite/src/context/auth.tsx).
