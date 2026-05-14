# Watermark API

FastAPI bridge between the React frontend (`../Website`) and the watermark engine (`../kenji2.0`).

## Setup

From this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The engine in `../kenji2.0` is imported directly — no need to install it as a package.

## Run

```
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/> to confirm the service is up. You should see a JSON status response.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/` | — | service info |
| POST | `/encode` | multipart: `file`, `owner`, `media_id` | `{ id, watermarked_url, metadata_url, metadata, psnr_db }` |
| POST | `/verify` | multipart: `file`, `metadata` (JSON string) | `AnalysisResult`-shaped JSON |
| GET  | `/metadata/{id}` | — | metadata JSON for a previously encoded file |
| GET  | `/files/{name}` | — | the watermarked output file |

## Quick test (PowerShell)

```powershell
# Encode
curl.exe -X POST http://localhost:8000/encode `
  -F "file=@../kenji2.0/kauniape.jpg" `
  -F "owner=Reuters" `
  -F "media_id=img_001"

# The response contains a `watermarked_url` and `metadata` — save the metadata
# JSON locally and point /verify at the watermarked file you just got back.
```

## Storage

Files land in `./storage/`:

- `uploads/` — incoming uploads
- `outputs/` — watermarked files (served at `/files/<name>`)
- `metadata/` — sidecar JSONs (queryable at `/metadata/<id>`)

This folder is local-only. For production, replace with S3/MinIO or whatever you wire into Diagram 1's "File Storage" block.

## Output format choices (locked-in by design)

- Images are saved as **PNG** regardless of the upload format. JPEG would destroy the watermark.
- Videos are saved as **MKV** with FFV1 (lossless). MP4 would destroy the watermark.

These match the constraints documented in `../kenji2.0/CLAUDE.md`. If you re-encode the watermarked output to a lossy format downstream, verification will correctly flag it as tampered — that is the algorithm working as designed.

## Connecting the frontend

In the website's `Encode.tsx` and `Verify.tsx`, replace the mock simulation with `fetch("http://localhost:8000/encode", ...)` and `fetch("http://localhost:8000/verify", ...)` as multipart `FormData`. The verify response shape already matches the `AnalysisResult` interface in `Website/src/types.ts` (or wherever it's defined).