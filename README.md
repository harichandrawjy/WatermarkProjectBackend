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

`owner` on `/encode` is **not** a form field — it's tied to the caller's JWT so it can't be spoofed.

Two distinct values are involved, and the difference matters:

| | what it is | where it lives |
|---|---|---|
| `owner` | the account's full email | `watermarks.owner`, returned by `/lookup` — **never embedded** |
| `short_id` | an assigned 8-char tag | `profiles.short_id`, copied to `watermarks.owner_key` — **this is what goes into the pixels** |

The engine's owner slot is `OWNER_ID_BYTES = 8`, so embedding the email directly would truncate it to the first 8 characters and merge every account sharing that prefix into one owner (`john.smith@gmail.com` and `john.smigielski@x.com` both become `john.smi`). Widening the slot isn't viable — at 512×512 the LL-family capacity is 768 bits, and a 12-byte owner id leaves zero room for the majority-vote repetition copies. So the 8 bytes now hold an **assigned** identifier (unique by constraint) instead of a **derived** prefix (unique by luck).

`/lookup` matches on `owner_key` and returns the row's full email, so the user-facing output is unchanged.

## Storage

Files land in `./storage/`:

- `uploads/` — incoming uploads
- `outputs/` — watermarked files (served at `/files/<name>`)

Metadata JSONs live in Postgres (`watermarks.metadata`), not on disk. For production, swap the local `storage/` for S3/MinIO.

## Output format (locked-in by design)

- Images → **PNG** regardless of upload format. JPEG would destroy the watermark.
- Videos → **MKV** with **FFV1** (lossless). MP4 would destroy the watermark.

Downstream re-encoding to a lossy format is correctly flagged as tampered — that is the algorithm working as designed.

## Watermark engine (in `engine/`)

- [`watermark_engine.py`](engine/watermark_engine.py) — image path. Embeds bits with LWT + 8×8 DCT + SVD + QIM on the Y channel; BCH(15,7) error-correction on the payload; per-sub-block SHA-256 parity in Cb LSBs for spatial tamper localization.
- [`video_watermark.py`](engine/video_watermark.py) — per-frame embed with frame-ID + chain-tag bits so deletions, duplicates, and reorders surface as discrete temporal events instead of cascading errors.

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
