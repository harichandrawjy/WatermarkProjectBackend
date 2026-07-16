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

The watermark engine in `engine/` is imported directly via `sys.path` — no `pip install`.

## Database

Run the SQL in [`migrations/001_add_user_id.sql`](migrations/001_add_user_id.sql) once via Supabase Dashboard → SQL Editor. It adds `user_id UUID` and a `created_at` default to the `watermarks` table so per-user isolation and history sorting work.

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

`owner` on `/encode` is **not** a form field — it's derived from the JWT's email so it can't be spoofed. The engine truncates the owner string to `OWNER_ID_BYTES = 8` when embedding bits; the full email is kept in the DB so `/lookup` still works after compression.

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
