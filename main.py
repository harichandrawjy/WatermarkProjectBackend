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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr

from auth import get_current_user

ENGINE_DIR = (Path(__file__).resolve().parent / "engine").resolve()
if not ENGINE_DIR.exists():
    raise RuntimeError(f"Cannot find watermark engine at {ENGINE_DIR}")
sys.path.insert(0, str(ENGINE_DIR))

from watermark_engine import (
    embed_image, verify_image, SPATIAL_BLOCK,
    _verify_channel, _parse_meta_payload,
    bch_decode_bits, array_to_bytes,
    to_float_ycbcr, META_PAYLOAD_BYTES,
    _BCH_N, _BCH_K,
    _majority_vote, _decode_id_fixed,
    OWNER_ID_BITS, MEDIA_ID_BITS, FRAME_ID_BITS, CHAIN_TAG_BITS,
    OWNER_ID_BYTES, MEDIA_ID_BYTES,
    OWNER_REPEATS, MEDIA_REPEATS, FRAME_REPEATS, CHAIN_REPEATS,
)
from video_watermark import embed_video, verify_video

from db import supabase

# ── Storage layout ──
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
UPLOAD_DIR  = STORAGE_DIR / "uploads"
OUTPUT_DIR  = STORAGE_DIR / "outputs"
for d in (STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR):
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
        "endpoints":  [
            "/encode", "/verify", "/lookup", "/files/{name}",
            "/auth/register", "/auth/login", "/auth/me",
            "/me/media", "/me/media/{id}/metadata",
        ],
    }


# ──────────────────────────────────────────────────────────────
# Auth (thin wrapper around Supabase Auth)
# ──────────────────────────────────────────────────────────────

class AuthBody(BaseModel):
    email:    EmailStr
    password: str


@app.post("/auth/register")
def auth_register(body: AuthBody):
    try:
        res = supabase.auth.sign_up({"email": body.email, "password": body.password})
    except Exception as e:
        raise HTTPException(400, f"Registration failed: {e}")

    user = getattr(res, "user", None)
    if not user:
        raise HTTPException(400, "Could not create user")

    session = getattr(res, "session", None)
    # When email-confirmation is on in Supabase, `session` is None until the
    # user clicks the confirmation link. We surface that to the frontend so
    # it can show a "check your inbox" screen.
    return {
        "user":               {"id": user.id, "email": user.email},
        "access_token":       session.access_token if session else None,
        "needs_confirmation": session is None,
    }


@app.post("/auth/login")
def auth_login(body: AuthBody):
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(401, f"Login failed: {e}")

    session = getattr(res, "session", None)
    user    = getattr(res, "user", None)
    if not session or not user:
        raise HTTPException(401, "Invalid credentials")

    return {
        "user":         {"id": user.id, "email": user.email},
        "access_token": session.access_token,
    }


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return {"id": user["id"], "email": user["email"]}


# ──────────────────────────────────────────────────────────────
# Encode / verify / lookup
# ──────────────────────────────────────────────────────────────

@app.post("/encode")
async def encode(
    file:     UploadFile = File(...),
    media_id: str        = Form(...),
    user:     dict       = Depends(get_current_user),
):
    # Owner is derived from the authenticated user — no longer free-text.
    # The watermark engine truncates to OWNER_ID_BYTES (8), but the DB
    # keeps the full email so /lookup still works after compression.
    owner = user["email"]

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

    meta_jsonable = _to_jsonable(meta)
    supabase.table("watermarks").insert({
        "id":       file_id,
        "owner":    owner,
        "media":    media_id,
        "kind":     kind,
        "metadata": meta_jsonable,
        "user_id":  user["id"],
    }).execute()

    return {
        "id":              file_id,
        "kind":            kind,
        "watermarked_url": f"/files/{out_path.name}",
        "metadata_url":    f"/metadata/{file_id}",
        "metadata":        meta_jsonable,
        "psnr_db":         meta.get("psnr_db") or meta.get("psnr_y_mean_db"),
    }


# ──────────────────────────────────────────────────────────────
# Dashboard: a user's own encoded media
# ──────────────────────────────────────────────────────────────

@app.get("/me/media")
def list_my_media(user: dict = Depends(get_current_user)):
    res = (
        supabase.table("watermarks")
        .select("*")
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .execute()
    )
    return {"items": res.data or []}


@app.get("/me/media/{file_id}/metadata")
def get_my_metadata(file_id: str, user: dict = Depends(get_current_user)):
    res = (
        supabase.table("watermarks")
        .select("metadata")
        .eq("id", file_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Not found or not yours")
    return res.data[0]["metadata"]


@app.get("/metadata/{file_id}")
def get_metadata(file_id: str):
    res = supabase.table("watermarks").select("metadata").eq("id", file_id).execute()
    if not res.data:
        raise HTTPException(404, "Metadata not found")
    return res.data[0]["metadata"]


def _blind_extract_ids(img_array: np.ndarray) -> dict | None:
    """Extract owner_id and media_id from a watermarked image without metadata.

    Tries BCH decode first (works on lossless files).  When compression
    corrupts BCH, falls back to majority-vote on the repetition copies —
    the same recovery path that verify_image uses.
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        return None

    Y, _, _ = to_float_ycbcr(img_array[:, :, :3])

    meta_raw_bits = META_PAYLOAD_BYTES * 8                          # 192
    meta_padded   = ((_BCH_K - meta_raw_bits % _BCH_K) % _BCH_K) + meta_raw_bits
    n_meta_coded  = (meta_padded // _BCH_K) * _BCH_N               # 420

    # Compute how many extra repetition bits the subbands can hold.
    H, W = Y.shape
    h2, w2 = H // 4, W // 4
    blocks_per_sub = (h2 // 8) * (w2 // 8)
    total_capacity = blocks_per_sub * 3                             # LLLL+LLLH+LLHL
    max_extras = (FRAME_REPEATS * FRAME_ID_BITS +
                  OWNER_REPEATS * OWNER_ID_BITS +
                  MEDIA_REPEATS * MEDIA_ID_BITS +
                  CHAIN_REPEATS * CHAIN_TAG_BITS)
    n_extra = min(max(0, total_capacity - n_meta_coded), max_extras)

    res = _verify_channel(Y, n_meta_coded, b"\x00" * 32, 0,
                          n_extra_llll_bits=n_extra)

    # ── Try 1: BCH decode ──
    decoded, _ = bch_decode_bits(res["meta_bits_coded"])
    meta_bytes = array_to_bytes(decoded[:meta_raw_bits])
    parsed = _parse_meta_payload(meta_bytes)
    if parsed:
        return parsed

    # ── Try 2: majority-vote fallback on repetition copies ──
    # Layout matches embed order: frame(5x) → owner(3x) → media(3x) → chain(3x)
    extra = res["extra_llll_bits"]
    p = 0
    p += min(FRAME_REPEATS * FRAME_ID_BITS, len(extra))            # skip frame_id

    n_owner = min(OWNER_REPEATS * OWNER_ID_BITS, max(0, len(extra) - p))
    owner_section = extra[p:p + n_owner]
    p += n_owner

    n_media = min(MEDIA_REPEATS * MEDIA_ID_BITS, max(0, len(extra) - p))
    media_section = extra[p:p + n_media]

    owner_voted = (_majority_vote(owner_section, OWNER_ID_BITS)
                   if len(owner_section) >= OWNER_ID_BITS * 2 else None)
    media_voted = (_majority_vote(media_section, MEDIA_ID_BITS)
                   if len(media_section) >= MEDIA_ID_BITS * 2 else None)

    if owner_voted is None and media_voted is None:
        return None

    owner_str = (_decode_id_fixed(array_to_bytes(owner_voted)[:OWNER_ID_BYTES])
                 if owner_voted is not None else "")
    media_str = (_decode_id_fixed(array_to_bytes(media_voted)[:MEDIA_ID_BYTES])
                 if media_voted is not None else "")

    if not owner_str and not media_str:
        return None

    return {"owner_id": owner_str, "media_id": media_str}


@app.post("/lookup")
async def lookup(file: UploadFile = File(...)):
    """Blind-extract owner + media from the watermark, then find the
    matching record in the database."""
    kind   = _detect_kind(file.filename)
    data   = await file.read()

    import io

    parsed = None
    if kind == "image":
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        parsed = _blind_extract_ids(img)
    else:
        import cv2
        file_id_tmp = uuid.uuid4().hex[:12]
        in_ext = os.path.splitext(file.filename)[1].lower()
        tmp_path = UPLOAD_DIR / f"{file_id_tmp}_lookup{in_ext}"
        tmp_path.write_bytes(data)
        try:
            cap = cv2.VideoCapture(str(tmp_path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(str(tmp_path))
            for _ in range(20):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                parsed = _blind_extract_ids(rgb)
                if parsed:
                    break
            cap.release()
        finally:
            tmp_path.unlink(missing_ok=True)

    if not parsed:
        raise HTTPException(404, "Could not extract watermark from this file")

    owner_id = parsed["owner_id"]
    media_id = parsed["media_id"]

    # Watermark stores only the first 8 bytes of owner/media, but the
    # database keeps the full original string.  Use case-insensitive prefix
    # matching so "alice@st" (extracted) matches "alice@studio.com" (stored).
    res = (
        supabase.table("watermarks")
        .select("*")
        .ilike("owner", f"{owner_id}%")
        .ilike("media", f"{media_id}%")
        .execute()
    )

    if not res.data:
        raise HTTPException(
            404,
            f"No database record for owner={owner_id!r} media={media_id!r}. "
            f"Extracted from watermark — check these match your encode inputs."
        )

    record = res.data[0]
    return {
        "id":       record["id"],
        "owner":    record["owner"],
        "media":    record["media"],
        "kind":     record["kind"],
        "metadata": record["metadata"],
    }


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
        verdict = verify_video(str(in_path), meta, sample_frames=None)
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

    spatial_arr = v.get("spatial_tamper_map")
    per_frame   = v.get("per_frame", []) or []

    frame_results = []
    for i, t in enumerate(temporal):
        per_frame_map = None
        if spatial_arr is not None and hasattr(spatial_arr, "shape") \
                and spatial_arr.ndim == 3 and i < spatial_arr.shape[0]:
            per_frame_map = spatial_arr[i]
        true_frame_idx = per_frame[i].get("frame_idx", i) if i < len(per_frame) else i
        frame_results.append({
            "frame":           int(true_frame_idx),
            "status":          "tampered" if t else "authentic",
            "confidence":      0.5 if t else 0.95,
            "tamperedRegions": _spatial_to_regions(per_frame_map),
        })

    # Pixel dimensions derived from the spatial block grid so the frontend
    # heatmap scales correctly per video.
    video_w = video_h = None
    if spatial_arr is not None and hasattr(spatial_arr, "shape") and spatial_arr.ndim == 3:
        _, Hb, Wb = spatial_arr.shape
        video_h = int(Hb * SPATIAL_BLOCK)
        video_w = int(Wb * SPATIAL_BLOCK)

    return {
        "status":          "tampered" if v["TAMPERED"] else "authentic",
        "confidence":      max(0.0, min(1.0, 1.0 - float(v.get("frame_tamper_rate", 0.0)))),
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - float(v["average_ber"]))),
        "ber":             float(v["average_ber"]),
        "fileType":        "video",
        "fileName":        filename,
        "tamperedRegions": [],
        "frameResults":    frame_results,
        "imageWidth":      video_w,
        "imageHeight":     video_h,
        "framesChecked":   int(v["frames_checked"]),
        "framesTampered":  int(v["frames_tampered"]),
        "frameTamperRate": float(v.get("frame_tamper_rate", 0.0)),
        "chainBreaks":     int(v["frames_chain_break"]),
        "idMismatches":    int(v["frames_id_mismatch"]),
    }