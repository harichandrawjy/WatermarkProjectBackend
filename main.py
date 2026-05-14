"""
Watermark API
=============
FastAPI bridge between the React frontend (../Website) and the
LWT/DCT/SVD watermark engine (./engine).

Endpoints:
  POST /encode  — multipart upload + owner/media_id form fields.
                  Returns a watermarked file URL plus metadata JSON.
  POST /verify  — multipart upload + metadata JSON string.
                  Returns an AnalysisResult-shaped JSON for the
                  frontend's Results page.
  GET  /files/{name}  — serves watermarked output files.
"""

import sys
import os
import uuid
import json
import base64
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ENGINE_DIR = (Path(__file__).resolve().parent / "engine").resolve()
if not ENGINE_DIR.exists():
    raise RuntimeError(f"Cannot find watermark engine at {ENGINE_DIR}")
sys.path.insert(0, str(ENGINE_DIR))

from watermark_engine import embed_image, verify_image, SPATIAL_BLOCK
from video_watermark import embed_video, verify_video

# ── Storage layout ──
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
UPLOAD_DIR  = STORAGE_DIR / "uploads"
OUTPUT_DIR  = STORAGE_DIR / "outputs"
META_DIR    = STORAGE_DIR / "metadata"
for d in (STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


app = FastAPI(title="Watermark API", version="1.0")

_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
# Set ALLOWED_ORIGINS in your host's env vars (comma-separated) to add
# production domains, e.g. https://watermark-project-website.vercel.app/
_extra = os.getenv("ALLOWED_ORIGINS", "")
_origins = _DEFAULT_ORIGINS + [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve watermarked outputs at /files/<name>
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise HTTPException(400, f"Unsupported file type: {ext}")


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / bytes types to JSON-friendly forms."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    return obj


def _spatial_to_regions(spatial_map):
    """Convert a Hb x Wb bool grid into pixel-rect regions matching the
    website's TamperedRegion interface."""
    if spatial_map is None:
        return []
    arr = np.asarray(spatial_map)
    if arr.ndim != 2:
        return []
    regions = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j]:
                regions.append({
                    "x":     int(j * SPATIAL_BLOCK),
                    "y":     int(i * SPATIAL_BLOCK),
                    "w":     int(SPATIAL_BLOCK),
                    "h":     int(SPATIAL_BLOCK),
                    "label": "modified_block",
                })
    return regions


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":    "watermark-api",
        "engine_dir": str(ENGINE_DIR),
        "endpoints":  ["/encode", "/verify", "/files/{name}"],
    }


@app.post("/encode")
async def encode(
    file:     UploadFile = File(...),
    owner:    str        = Form(...),
    media_id: str        = Form(...),
):
    kind    = _detect_kind(file.filename)
    file_id = uuid.uuid4().hex[:12]
    in_ext  = os.path.splitext(file.filename)[1].lower()
    in_path = UPLOAD_DIR / f"{file_id}_input{in_ext}"

    in_path.write_bytes(await file.read())

    if kind == "image":
        # Force lossless PNG output so the watermark survives.
        out_path = OUTPUT_DIR / f"{file_id}_wm.png"
        img      = np.array(Image.open(in_path).convert("RGB"))
        wm, meta = embed_image(img, owner, media_id)
        Image.fromarray(wm).save(out_path)
    else:
        # Force lossless MKV (FFV1) output so the watermark survives.
        out_path = OUTPUT_DIR / f"{file_id}_wm.mkv"
        meta     = embed_video(str(in_path), str(out_path), owner, media_id)

    meta_path = META_DIR / f"{file_id}_meta.json"
    meta_jsonable = _to_jsonable(meta)
    meta_path.write_text(json.dumps(meta_jsonable, indent=2))

    return {
        "id":              file_id,
        "kind":            kind,
        "watermarked_url": f"/files/{out_path.name}",
        "metadata_url":    f"/metadata/{file_id}",
        "metadata":        meta_jsonable,
        "psnr_db":         meta.get("psnr_db"),
    }


@app.get("/metadata/{file_id}")
def get_metadata(file_id: str):
    meta_path = META_DIR / f"{file_id}_meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Metadata not found")
    return json.loads(meta_path.read_text())


@app.post("/verify")
async def verify(
    file:     UploadFile = File(...),
    metadata: str        = Form(...),
):
    kind    = _detect_kind(file.filename)
    file_id = uuid.uuid4().hex[:12]
    in_ext  = os.path.splitext(file.filename)[1].lower()
    in_path = UPLOAD_DIR / f"{file_id}_verify{in_ext}"

    in_path.write_bytes(await file.read())

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}")

    if kind == "image":
        img        = np.array(Image.open(in_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        verdict    = verify_image(img, meta)
        return _shape_image_response(verdict, file.filename, img_w, img_h)
    else:
        verdict = verify_video(str(in_path), meta, sample_frames=30)
        return _shape_video_response(verdict, file.filename)


def _shape_image_response(v: dict, filename: str, width: int, height: int) -> dict:
    return {
        "status":          "tampered" if v["tampered"] else "authentic",
        "confidence":      max(0.0, min(1.0, 1.0 - float(v["ber"]))),
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - float(v["ber"]))),
        "ber":             float(v["ber"]),
        "fileType":        "image",
        "fileName":        filename,
        "imageWidth":      int(width),
        "imageHeight":     int(height),
        "tamperedRegions": _spatial_to_regions(v.get("spatial_map")),
        "watermarkFound":  bool(v["watermark_found"]),
        "ownerMatch":      bool(v["owner_match"]),
        "mediaMatch":      bool(v["media_match"]),
        "owner":           v.get("owner"),
        "media":           v.get("media"),
        "blocksTampered":  int(v["n_blocks_tampered"]),
        "blocksTotal":     int(v["n_blocks_total"]),
    }


def _shape_video_response(v: dict, filename: str) -> dict:
    temporal = v.get("temporal_tamper_map")
    if hasattr(temporal, "tolist"):
        temporal = temporal.tolist()
    elif temporal is None:
        temporal = []

    frame_results = [
        {
            "frame":      i,
            "status":     "tampered" if t else "authentic",
            "confidence": 0.5 if t else 0.95,
        }
        for i, t in enumerate(temporal)
    ]

    return {
        "status":          "tampered" if v["TAMPERED"] else "authentic",
        "confidence":      max(0.0, min(1.0, 1.0 - float(v.get("frame_tamper_rate", 0.0)))),
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - float(v["average_ber"]))),
        "ber":             float(v["average_ber"]),
        "fileType":        "video",
        "fileName":        filename,
        "tamperedRegions": [],
        "frameResults":    frame_results,
        "framesChecked":   int(v["frames_checked"]),
        "framesTampered":  int(v["frames_tampered"]),
        "frameTamperRate": float(v.get("frame_tamper_rate", 0.0)),
        "chainBreaks":     int(v["frames_chain_break"]),
        "idMismatches":    int(v["frames_id_mismatch"]),
    }