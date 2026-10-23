"""
Watermark API
=============
FastAPI bridge between the React frontend (../Website) and the
LWT/SVD watermark engine (./engine).

Endpoints:
  POST /encode  — multipart upload + owner/media_id form fields.
                  Returns a watermarked file URL plus metadata JSON.
  POST /verify  — multipart upload + metadata JSON string.
                  Returns an AnalysisResult-shaped JSON for the
                  frontend's Results page.
  GET  /files/{name}  — serves watermarked output files.
"""

import sys
import time
import os
import io
import re
import uuid
import json
import base64
import secrets
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from PIL import Image, features

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

from auth import get_current_user, get_optional_user

ENGINE_DIR = (Path(__file__).resolve().parent / "engine").resolve()
if not ENGINE_DIR.exists():
    raise RuntimeError(f"Cannot find watermark engine at {ENGINE_DIR}")
sys.path.insert(0, str(ENGINE_DIR))

from watermark_engine import (
    embed_image, verify_image, SPATIAL_BLOCK, LSB_SUB_BLOCK_SIZE,
    _verify_channel, _parse_meta_payload,
    bch_decode_bits, array_to_bytes, bits_to_array,
    to_float_ycbcr, META_PAYLOAD_BYTES,
    _BCH_N, _BCH_K,
    _majority_vote, _decode_id_fixed, _encode_id_fixed,
    _llfamily_block_capacity, _plan_repeat_copies,
    OWNER_ID_BITS, MEDIA_ID_BITS, FRAME_ID_BITS, CHAIN_TAG_BITS,
    OWNER_ID_BYTES, MEDIA_ID_BYTES,
    OWNER_REPEATS, MEDIA_REPEATS, FRAME_REPEATS, CHAIN_REPEATS,
)
from video_watermark import embed_video, verify_video
from resync import (resync_extract, apply_transform, content_box,
                    MAX_OFFSET as RESYNC_MAX_OFFSET)

import httpx

from db import supabase, supabase_auth, SUPABASE_URL, SUPABASE_ANON_KEY

# ── Storage layout ──
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
UPLOAD_DIR  = STORAGE_DIR / "uploads"
OUTPUT_DIR  = STORAGE_DIR / "outputs"
for d in (STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Accepted INPUT extensions.  Decoding itself is format-agnostic — every path
# goes through Image.open(...).convert("RGB") — so this set is purely a gate,
# and anything Pillow can read is safe to list.
#
# .webp is REQUIRED here, not optional: IMAGE_FORMATS below offers WEBP as an
# output format, so leaving it out meant the app handed back a file it would
# then refuse on /verify and /lookup — the round trip was broken for every
# WEBP encode.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# AVIF decoding only ships with Pillow >= 11.3 (libavif).  Advertise it only
# when this build can actually decode it, so an older Pillow returns a clean
# 400 "unsupported" instead of a 500 from inside the decoder.
if features.check("avif"):
    IMAGE_EXTS.add(".avif")

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Selectable OUTPUT formats, keyed by media kind.  Deliberately restricted to
# LOSSLESS containers only: the invisible semi-fragile watermark lives in the
# SVD + chroma-LSB layers, and any lossy codec (JPEG, H.264, VP8...)
# perturbs those coefficients past the BCH + majority-vote recovery margin, so
# a lossy re-encode would hand back a file that fails its own verification.
#   image  → PNG / lossless WEBP
#   video  → MKV / AVI, both written with the FFV1 lossless codec (see
#            video_watermark.embed_video, which already routes .mkv/.avi
#            through cv2.VideoWriter's FFV1 path).
IMAGE_FORMATS = {"png": ".png", "webp": ".webp"}
VIDEO_FORMATS = {"mkv": ".mkv", "avi": ".avi"}



app = FastAPI(title="Watermark API", version="1.0")

_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
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
    # Response headers are hidden from cross-origin JS unless named here, so
    # /media/{id} could set X-Output-Filename and the frontend would still read
    # null. Only safelisted headers are visible by default.
    expose_headers=["X-Output-Filename"],
)

# Serve watermarked outputs at /files/<name>
# Serve watermarked outputs at /files/<name>.
#
# RETENTION: none is promised.  These files live on the container's local disk,
# which on most hosts (including Render's free tier) is ephemeral — a deploy or
# a wake-from-sleep gives the process an empty directory back.
#
# That is deliberate rather than tolerated.  Encoding is DETERMINISTIC: the same
# original + owner + media_id yields a byte-identical file every time, because
# the signature, chain tag and QIM lattice all derive from those inputs via
# SHA-256.  So the watermarked file is not unrecoverable data — it is a pure
# function of things the user already has, and it was handed to them at encode
# time.  What genuinely cannot be regenerated is the metadata, and that lives
# in Postgres.
#
# Storing user media would also contradict what the Verify page promises about
# uploads.  Keeping nothing makes one policy true everywhere.
#
# The practical effect: /files/<name> works while the container is warm and
# 404s afterwards.  The frontend treats a miss as expected and says so.
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


def _embedded_pattern_data_url(img_shape, owner_id: str, media_id: str, frame_id: int = 0):
    """The fragile watermark pattern that gets embedded, as a PNG data URL.

    Regenerated rather than returned by the engine, because it is fully
    determined by (owner, media, frame): the same SHA-256 derivation both the
    encoder and the decoder run. That is the point worth showing a user — the
    watermark is not stored anywhere, it is *computed* from the identity, which
    is why the verifier can rebuild it and compare.

    The grid is one cell per LSB_SUB_BLOCK_SIZE (16) pixels of the chroma
    channel, matching what Verify renders after extraction so the two images
    are directly comparable.
    """
    try:
        from watermark_engine import (          # noqa: PLC0415
            _master_signature, _compute_lsb_target_pattern, LSB_SUB_BLOCK_SIZE,
        )
        h, w = int(img_shape[0]), int(img_shape[1])
        n_sub_h, n_sub_w = h // LSB_SUB_BLOCK_SIZE, w // LSB_SUB_BLOCK_SIZE
        if n_sub_h < 1 or n_sub_w < 1:
            return None
        sig = _master_signature(owner_id, media_id, frame_id)
        pattern = _compute_lsb_target_pattern(n_sub_h, n_sub_w, sig, frame_id)
        return _pattern_to_data_url(pattern)
    except Exception as e:
        # Illustrative only — never fail an encode over a preview image.
        print(f"[encode] watermark pattern preview unavailable: {e}")
        return None


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


def _pattern_to_data_url(grid, mismatch=None, scale: int = 10):
    """Render a 2D 0/1 bit grid as an upscaled PNG data URL (black=0, white=1).

    When `mismatch` (a same-shape bool array) is supplied, cells that differ
    from the expected pattern are painted rose-red — so the extracted fragile
    watermark visually shows exactly where tampering corrupted it.  Nearest-
    neighbour upscaling keeps each bit a crisp square.
    """
    if grid is None:
        return None
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return None
    h, w = g.shape
    val = ((g.astype(np.uint8) & 1) * 255).astype(np.uint8)
    rgb = np.stack([val, val, val], axis=-1)        # white=1, black=0
    if mismatch is not None:
        m = np.asarray(mismatch).astype(bool)
        if m.shape == g.shape:
            rgb[m] = (244, 63, 94)                   # rose-500 — tampered bits
    img = Image.fromarray(rgb, mode="RGB").resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_data_url(mask, scale: int = 10):
    """Render a 2D bool mismatch grid as a TRANSPARENT PNG, red where flagged.

    The sibling `_pattern_to_data_url` paints red onto the opaque black/white
    pattern, which is right for the side-by-side comparison but useless over a
    photo — the noise texture would hide the image entirely.  Here only the
    flagged cells are drawn, so the overlay localises tampering on the real
    picture and nothing else.

    Returns None when nothing is flagged: an all-transparent PNG is bytes spent
    to draw nothing, and the caller can branch on its absence.
    """
    if mask is None:
        return None
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2 or m.size == 0 or not m.any():
        return None
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[m] = (244, 63, 94, 255)                     # rose-500, fully opaque
    img = Image.fromarray(rgba, mode="RGBA").resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Cells within this many cells of each other are treated as one edit. At
# LSB_SUB_BLOCK_SIZE=16 that is a 48px gap tolerance, chosen by measurement:
# strict adjacency (radius 0) splits even a single spliced block into 2 boxes
# and a handwritten annotation into 4, while radius 3 reports both correctly
# and still keeps two separate splices apart. Over-merging is the safer error —
# it under-counts distinct edits, whereas under-merging invents them.
CLUSTER_GROUP_RADIUS = 3


def _mask_to_clusters(mask, min_cells: int = 3, radius: int = CLUSTER_GROUP_RADIUS):
    """Group flagged cells into edits and box each one, in image pixels.

    The layer flags a conservative SUBSET of an edited region — measured
    precision ~1.0 but recall ~0.4 — so the raw cells read as scattered
    confetti rather than as one tampered area. Grouping says "these belong to
    the same edit" without colouring in a cell that did not actually fail its
    parity check.

    Proximity, not adjacency, decides membership: a thin edit (handwriting, a
    drawn line) flags cells in a dashed trail with gaps, so requiring cells to
    touch fragments one annotation into several boxes or, when every fragment
    lands under `min_cells`, into none at all.

    Dilation only decides WHICH cells group together — each box is then the
    extent of the real flagged cells inside its group, never of the dilated
    region. The box is therefore a statement about what was detected, and
    because recall is well under 1 the true edit routinely runs past it; the
    UI has to label it as such rather than as the edited region.

    Groups under `min_cells` are dropped: an isolated flipped cell with nothing
    near it is the quantization noise floor MIN_TAMPER_BLOCKS exists to reject,
    and boxing it would give it more weight than the evidence supports.
    """
    if mask is None:
        return []
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2 or m.size == 0 or not m.any():
        return []
    k = 2 * radius + 1
    grown = ndimage.binary_dilation(m, np.ones((k, k), dtype=bool))
    # 8-connectivity on top of the dilation, so diagonal contact also joins.
    labels, n = ndimage.label(grown, structure=np.ones((3, 3), dtype=int))
    out = []
    for i in range(1, n + 1):
        cells_mask = m & (labels == i)
        cells = int(cells_mask.sum())
        if cells < min_cells:
            continue
        ys, xs = np.nonzero(cells_mask)
        out.append({
            "x":     int(xs.min() * LSB_SUB_BLOCK_SIZE),
            "y":     int(ys.min() * LSB_SUB_BLOCK_SIZE),
            "w":     int((xs.max() - xs.min() + 1) * LSB_SUB_BLOCK_SIZE),
            "h":     int((ys.max() - ys.min() + 1) * LSB_SUB_BLOCK_SIZE),
            "cells": cells,
        })
    # Biggest first, so the UI can label the dominant edit without re-sorting.
    out.sort(key=lambda c: c["cells"], reverse=True)
    return out


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
        res = supabase_auth.auth.sign_up({"email": body.email, "password": body.password})
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
        "access_token":       session.access_token  if session else None,
        "refresh_token":      session.refresh_token if session else None,
        "needs_confirmation": session is None,
    }


@app.post("/auth/login")
def auth_login(body: AuthBody):
    try:
        res = supabase_auth.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(401, f"Login failed: {e}")

    session = getattr(res, "session", None)
    user    = getattr(res, "user", None)
    if not session or not user:
        raise HTTPException(401, "Invalid credentials")

    return {
        "user":          {"id": user.id, "email": user.email},
        "access_token":  session.access_token,
        "refresh_token": session.refresh_token,
    }


class RefreshBody(BaseModel):
    refresh_token: str


@app.post("/auth/refresh")
def auth_refresh(body: RefreshBody):
    """Exchange a refresh_token for a fresh access_token + refresh_token pair.

    Supabase access tokens expire after ~1 hour. The frontend calls this
    whenever it gets a 401 on an authed request, so users don't get
    silently logged out mid-session.
    """
    try:
        res = supabase_auth.auth.refresh_session(body.refresh_token)
    except Exception as e:
        raise HTTPException(401, f"Refresh failed: {e}")

    session = getattr(res, "session", None)
    user    = getattr(res, "user", None)
    if not session or not user:
        raise HTTPException(401, "Refresh token rejected")

    return {
        "user":          {"id": user.id, "email": user.email},
        "access_token":  session.access_token,
        "refresh_token": session.refresh_token,
    }


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    # Includes the short_id so the UI can tell the user which identifier is
    # actually embedded in their files. Without it the Results page shows an
    # 8-character tag the user has never been told about, which reads as a
    # decoding error rather than as their own owner id.
    #
    # Allocates on first call rather than waiting for first encode — knowing
    # the id before you use it is more useful than discovering it afterwards.
    try:
        short_id = _get_or_create_short_id(user)
    except Exception:
        short_id = None
    return {"id": user["id"], "email": user["email"], "short_id": short_id}


class ForgotBody(BaseModel):
    email: EmailStr


@app.post("/auth/forgot-password")
def auth_forgot_password(body: ForgotBody):
    """Send a password-recovery email.

    Supabase mails a link to `<SITE_URL>/#access_token=...&type=recovery`; the
    frontend picks that hash up and shows the "set a new password" form.

    Always reports success — even for an address with no account, and even if
    Supabase itself errors.  Reporting "no such user" would turn this endpoint
    into an account-enumeration oracle: anyone could probe which emails are
    registered.  That matters here because /lookup already exposes owner
    identities publicly, so confirming an address exists is a real leak.
    """
    site_url = os.getenv("SITE_URL", "http://localhost:5173")
    try:
        supabase_auth.auth.reset_password_email(
            body.email, {"redirect_to": site_url}
        )
    except Exception as e:
        # Logged, never returned — the caller must not be able to tell the
        # difference between "sent" and "no such account".
        print(f"[forgot-password] suppressed for {body.email!r}: {e}")

    return {
        "sent": True,
        "message": "If that email has an account, a reset link is on its way.",
    }


class ResetBody(BaseModel):
    access_token: str
    new_password: str


@app.post("/auth/reset-password")
def auth_reset_password(body: ResetBody):
    """Set a new password using the recovery token from the emailed link.

    Calls GoTrue's REST endpoint directly rather than
    `supabase_auth.auth.update_user()`, which would act on the shared global
    client's stored session.  That client is process-wide, so a concurrent
    request could have swapped the session underneath us — an acceptable risk
    almost nowhere, and certainly not when changing a password.  Passing the
    recovery token as an explicit Bearer header keeps the operation stateless
    and unambiguous about whose password is being changed.
    """
    token = body.access_token.strip()
    if not token:
        raise HTTPException(400, "Missing recovery token")
    if not body.new_password:
        raise HTTPException(400, "New password is required")

    try:
        res = httpx.put(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey":        SUPABASE_ANON_KEY,
                "Content-Type":  "application/json",
            },
            json={"password": body.new_password},
            timeout=15,
        )
    except Exception as e:
        raise HTTPException(503, f"Could not reach the auth service: {e}")

    if res.status_code >= 400:
        # Surface Supabase's own wording — it owns the password policy, so it
        # is the only thing that knows why a password was rejected (too short,
        # previously used, link expired, ...).  Duplicating those rules here
        # would just create a second source of truth that drifts.
        detail = ""
        try:
            j = res.json()
            detail = j.get("msg") or j.get("message") or j.get("error_description") or ""
        except Exception:
            detail = res.text[:200]
        if res.status_code in (401, 403):
            raise HTTPException(
                401,
                detail or "This reset link has expired or was already used. "
                          "Request a new one.",
            )
        if res.status_code >= 500:
            # Upstream is down (Supabase returns Cloudflare 5xx when the auth
            # origin is unreachable).  Reporting that as a 400 would blame the
            # user's input for an outage and send them re-requesting links
            # that were never going to work.
            raise HTTPException(
                503,
                "The authentication service is temporarily unavailable. "
                "Please try again in a few minutes.",
            )
        raise HTTPException(400, detail or f"Could not update password ({res.status_code})")

    return {"ok": True}


# ──────────────────────────────────────────────────────────────
# Encode / verify / lookup
# ──────────────────────────────────────────────────────────────

def _id_key(s: str, n_bytes: int) -> str:
    """The exact lowercased 8-byte key that gets embedded for `s` — i.e. the
    UTF-8 byte-truncated, null-stripped form, lowercased.  Encode stores this
    in owner_key/media_key and lookup matches it exactly, so uniqueness and
    resolution both operate on what the watermark actually carries."""
    return _decode_id_fixed(_encode_id_fixed(s, n_bytes)).lower()


# Look-alike characters (i/l/1, o/0) are excluded so a short_id can be read
# aloud or retyped without ambiguity.
_SHORT_ID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
_SHORT_ID_LEN      = OWNER_ID_BYTES          # 8 — fills the embed slot exactly


def _new_short_id() -> str:
    return "".join(secrets.choice(_SHORT_ID_ALPHABET) for _ in range(_SHORT_ID_LEN))


def _get_or_create_short_id(user: dict) -> str:
    """The 8-character identifier actually embedded in this user's watermarks.

    The engine truncates the owner string to OWNER_ID_BYTES (8), so embedding
    an email would collapse every account sharing its first 8 characters into a
    single owner — `john.smith@gmail.com` and `john.smigielski@x.com` both
    become `john.smi`.  Widening the field isn't possible (at 512x512 a 12-byte
    owner id leaves no room for the majority-vote repetition copies), so the
    fix is to make those 8 bytes an ASSIGNED value rather than a DERIVED one:
    short_ids carry a UNIQUE constraint, arbitrary email prefixes cannot.

    Allocated lazily on first encode and stable forever after — it is baked
    into every file the user has already distributed.
    """
    res = (
        supabase.table("profiles")
        .select("short_id")
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["short_id"]

    # Retry on short_id collision.  The alphabet gives 31^8 ~= 8.5e11 values,
    # so a retry is rare — but the UNIQUE constraint, not luck, is what makes
    # uniqueness actual rather than probable.
    for _ in range(5):
        sid = _new_short_id()
        try:
            supabase.table("profiles").insert({
                "user_id": user["id"],
                "short_id": sid,
                "email":    user["email"],
            }).execute()
            return sid
        except Exception:
            # Either short_id collided (retry with a new one) or a concurrent
            # request already created this user's row (re-read and use theirs).
            res = (
                supabase.table("profiles")
                .select("short_id")
                .eq("user_id", user["id"])
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]["short_id"]

    raise HTTPException(500, "Could not allocate a watermark ID — please retry.")


@app.post("/encode")
async def encode(
    file:       UploadFile = File(...),
    media_id:   str        = Form(...),
    # Output container/format, validated against the media kind below.  Empty
    # falls back to the lossless default (PNG for images, MKV for video).
    out_format: str        = Form("", alias="format"),
    user:       dict       = Depends(get_current_user),
):
    # Two different strings, deliberately:
    #   owner_email — the human-readable account, kept in the DB and returned
    #                 by /lookup.  Never embedded.
    #   owner       — the assigned 8-char short_id that IS embedded.  Using the
    #                 email here would truncate to its first 8 bytes and merge
    #                 distinct accounts into one owner (see migration 003).
    # The engine embeds media_id into a fixed MEDIA_ID_BYTES slot, so anything
    # longer is silently truncated — "foto_jokowi" would become "foto_jok",
    # and the verify screen would then report an "extracted" value that looks
    # wrong to the user even though it is correct. Reject it instead, so what
    # they type is what lands in the pixels.
    #
    # Measured in BYTES, not characters: one non-ASCII character can occupy up
    # to four of them, so a 4-character CJK id would overflow an 8-byte slot.
    # The frontend's maxLength is a convenience; this is the actual guarantee.
    media_id = media_id.strip()
    if not media_id:
        raise HTTPException(400, "Media ID is required.")
    n_bytes = len(media_id.encode("utf-8"))
    if n_bytes > MEDIA_ID_BYTES:
        raise HTTPException(
            400,
            f"Media ID is {n_bytes} bytes; the watermark carries at most "
            f"{MEDIA_ID_BYTES}. Use a shorter id so it embeds exactly as typed."
        )

    owner_email = user["email"]
    owner       = _get_or_create_short_id(user)

    # Uniqueness guard: an owner cannot reuse a media_id whose embedded 8-byte
    # key collides with one they already have — otherwise blind /lookup could
    # resolve to the wrong record.  Checked BEFORE embedding so we don't waste
    # work or leave an orphan file; the DB unique index is the race-proof
    # backstop (see migration 002).
    #
    # Scoped to user_id as well as owner_key: short_ids make owner_key unique
    # per account already, but legacy rows still hold email-prefix keys that
    # two accounts can share — without this filter such a row would block a
    # different user and quote THEIR media_id back in the error below.
    owner_key = _id_key(owner,    OWNER_ID_BYTES)
    media_key = _id_key(media_id, MEDIA_ID_BYTES)
    dup = (
        supabase.table("watermarks")
        .select("id, media")
        .eq("user_id", user["id"])
        .eq("owner_key", owner_key)
        .eq("media_key", media_key)
        .limit(1)
        .execute()
    )
    if dup.data:
        raise HTTPException(
            409,
            f"You already have media_id '{dup.data[0].get('media')}' whose "
            f"watermark key ('{media_key}') collides with '{media_id}'. "
            f"Choose a media_id with a different first 8 characters."
        )

    kind = _detect_kind(file.filename)

    # ── Resolve the requested output format ──
    # Pick the format table for this media kind, so an image can't request a
    # video container (or vice-versa) and a lossy pick can't slip through.
    fmt_table = IMAGE_FORMATS if kind == "image" else VIDEO_FORMATS
    fmt = (out_format or "").strip().lower()
    if not fmt:
        fmt = "png" if kind == "image" else "mkv"          # lossless defaults
    if fmt not in fmt_table:
        raise HTTPException(
            400,
            f"Unsupported {kind} format '{fmt}'. "
            f"Choose one of: {', '.join(fmt_table)} (lossless only — the "
            f"invisible watermark cannot survive a lossy re-encode)."
        )
    out_ext = fmt_table[fmt]

    file_id = uuid.uuid4().hex[:12]
    in_ext  = os.path.splitext(file.filename)[1].lower()
    in_path = UPLOAD_DIR / f"{file_id}_input{in_ext}"

    in_path.write_bytes(await file.read())

    # The fragile watermark pattern that gets embedded, rendered as an image so
    # the user can SEE what a watermark actually is rather than being told one
    # exists.  Verify already shows this pattern next to the one recovered from
    # a file; showing it at encode time gives the "before" half of that
    # comparison, which is where the concept has to land first.
    wm_pattern_url = None

    if kind == "image":
        out_path = OUTPUT_DIR / f"{file_id}_wm{out_ext}"
        img      = np.array(Image.open(in_path).convert("RGB"))
        wm, meta = embed_image(img, owner, media_id)
        wm_pattern_url = _embedded_pattern_data_url(img.shape, owner, media_id)
        # WEBP must be told to stay lossless; PNG is lossless by definition.
        if fmt == "webp":
            Image.fromarray(wm).save(out_path, format="WEBP", lossless=True)
        else:
            Image.fromarray(wm).save(out_path)
    else:
        # MKV/AVI both route through the FFV1 lossless codec in embed_video.
        out_path = OUTPUT_DIR / f"{file_id}_wm{out_ext}"
        meta     = embed_video(str(in_path), str(out_path), owner, media_id)
        res = meta.get("resolution", "")
        if "x" in res:
            w_s, h_s = res.split("x")[:2]
            # Frame 0's pattern — each frame is keyed by its own frame_id, so
            # this is representative rather than universal.
            wm_pattern_url = _embedded_pattern_data_url(
                (int(h_s), int(w_s), 3), owner, media_id, frame_id=0)

    # The original upload has served its purpose — embedding (and, for video,
    # the audio mux) is finished.  Drop it now rather than leaving user media
    # sitting in storage/uploads forever.  out_path is deliberately KEPT: it is
    # the watermarked file served at /files/<name> and re-downloaded from the
    # Dashboard.
    in_path.unlink(missing_ok=True)

    meta_jsonable = _to_jsonable(meta)
    # The engine records owner_id = the embedded short_id.  Carry the readable
    # address alongside it so the Verify/Results UI can name the owner from a
    # metadata sidecar alone, without a /lookup round-trip.
    meta_jsonable["owner_email"] = owner_email

    try:
        supabase.table("watermarks").insert({
            "id":        file_id,
            "owner":     owner_email,
            "media":     media_id,
            "owner_key": owner_key,
            "media_key": media_key,
            "kind":      kind,
            "metadata":  meta_jsonable,
            "user_id":   user["id"],
            # Display only — the media id is the identifier. Kept because it is
            # what a user actually recognises a row by; `file.filename` was
            # previously read for the extension and then thrown away.
            "file_name": file.filename,
        }).execute()
    except Exception as e:
        # Race-proof backstop for the unique index — clean up the orphan file
        # and surface a clear 409 instead of a raw 500.
        out_path.unlink(missing_ok=True)
        in_path.unlink(missing_ok=True)
        if "duplicate" in str(e).lower() or "23505" in str(e):
            raise HTTPException(
                409,
                f"media_id '{media_id}' (key '{media_key}') is already used by this owner. "
                f"Choose a media_id with a different first 8 characters."
            )
        raise

    return {
        "id":              file_id,
        "kind":            kind,
        "watermarked_url": f"/files/{out_path.name}",
        "metadata_url":    f"/metadata/{file_id}",
        "metadata":        meta_jsonable,
        "psnr_db":         meta.get("psnr_db") or meta.get("psnr_y_mean_db"),
        # PNG data URL of the fragile pattern now living in the file's chroma
        # channel — the same image Verify compares against after extraction.
        "watermark_pattern": wm_pattern_url,
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


@app.delete("/me/media/{file_id}")
def delete_my_media(file_id: str, user: dict = Depends(get_current_user)):
    """Delete one of the caller's encode records, freeing its media_id.

    Needed because encoding is deterministic but the app forbids repeats: the
    same original + owner + media_id always produces a byte-identical file, yet
    /encode rejects a reused media_id with a 409 (the UNIQUE index from
    migration 002 enforces it).  Without a way to remove the record, a user who
    loses their downloaded file has permanently burned that media_id — they can
    neither re-download it nor regenerate it under the same name.

    Scoped by user_id so one account cannot delete another's record, and the
    row is fetched first so a missing id is a clean 404 rather than a silent
    no-op reported as success.

    The watermark already embedded in copies out in the world is unaffected —
    deleting the record only means /lookup can no longer resolve it.
    """
    res = (
        supabase.table("watermarks")
        .select("id, kind, media")
        .eq("id", file_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Not found or not yours")
    row = res.data[0]

    supabase.table("watermarks").delete().eq("id", file_id).eq("user_id", user["id"]).execute()

    # Drop the local copy too if this container still happens to hold one.
    # The output extension now varies with the chosen format (png/webp/mkv/avi),
    # so remove whichever "<id>_wm.*" file exists rather than assuming one.
    for p in OUTPUT_DIR.glob(f"{file_id}_wm.*"):
        p.unlink(missing_ok=True)

    return {"deleted": file_id, "media": row.get("media")}


@app.get("/media/{file_id}")
def get_media_by_id(file_id: str):
    """Serve a watermarked output by encode id, whatever its extension is.

    The output format became user-selectable (png/webp for images, mkv/avi for
    video) but is not stored on the row, so a caller holding only the id cannot
    know the filename.  The Dashboard used to guess it from `kind` — always
    `.png` or `.mkv` — which silently 404s for anything encoded as WEBP or AVI
    even while the file sits in the directory.

    Resolving server-side by glob avoids a migration and fixes existing rows
    too.  Deliberately NOT under /files: that path is a StaticFiles mount, which
    would match this route first and try to serve a file literally named
    "by-id/...".

    Public, matching the /files mount it complements — no access regression.
    """
    # `file_id` reaches a glob pattern, so a traversal like "../../etc/passwd"
    # would escape OUTPUT_DIR.  Ids are hex from uuid4, so anything outside
    # this character class is already invalid.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", file_id):
        raise HTTPException(400, "Invalid id")

    matches = sorted(OUTPUT_DIR.glob(f"{file_id}_wm.*"))
    if not matches:
        raise HTTPException(
            404,
            "That file is no longer on the server. Watermarked media is not "
            "retained — use the copy downloaded at encode time, or re-encode."
        )

    path = matches[0]
    return FileResponse(
        path,
        filename=path.name,
        # Lets the browser-side download name the file with the right extension
        # without the frontend having to guess it back.
        headers={"X-Output-Filename": path.name},
    )


@app.get("/metadata/{file_id}")
def get_metadata(file_id: str):
    res = supabase.table("watermarks").select("metadata").eq("id", file_id).execute()
    if not res.data:
        raise HTTPException(404, "Metadata not found")
    return res.data[0]["metadata"]


# An unwatermarked upload looks exactly like a misaligned one until the search
# has finished failing — the search is what tells them apart — so this bounds
# what that costs.  Sized at roughly twice the slowest recovery measured end to
# end (5.4s, a fit-to-container rescale, which is the case that has to reach
# the exhaustive scale sweep) to leave headroom on a slower box.
RESYNC_BUDGET_S = 12.0


def _bch_extract_ids(img_array: np.ndarray) -> dict | None:
    """Extract the payload via BCH alone, gated on its integrity tag.

    Split out from `_blind_extract_ids` because this is the only form safe to
    hand to a geometry search: `_parse_meta_payload` rejects anything whose
    SHA-256 tag does not match, so a wrong transform returns None rather than a
    confident wrong owner.  The majority-vote path below has no such tag, and
    on a misaligned image its sections are themselves shifted — measured
    returning the media id in the owner slot.  Searching with it would
    eventually confirm a wrong geometry.
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        return None
    if img_array.shape[0] < 64 or img_array.shape[1] < 64:
        return None

    Y, _, _ = to_float_ycbcr(img_array[:, :, :3])

    meta_raw_bits = META_PAYLOAD_BYTES * 8                          # 192
    meta_padded   = ((_BCH_K - meta_raw_bits % _BCH_K) % _BCH_K) + meta_raw_bits
    n_meta_coded  = (meta_padded // _BCH_K) * _BCH_N               # 420

    res = _verify_channel(Y, n_meta_coded, b"\x00" * 32, 0, n_extra_llll_bits=0)
    decoded, _ = bch_decode_bits(res["meta_bits_coded"])
    return _parse_meta_payload(array_to_bytes(decoded[:meta_raw_bits]))


def _blind_extract_ids(img_array: np.ndarray,
                       frame_repeats: int = 1) -> dict | None:
    """Extract owner_id and media_id from a watermarked image without metadata.

    Tries BCH decode first (works on lossless files), then falls back to
    majority-vote on the repetition copies — the same recovery path
    verify_image uses.  Both read the grid where it is; recovering a grid that
    MOVED is `_resync_extract_ids`, which /lookup reaches for only once a
    lookup on these ids has actually come back empty.

    `frame_repeats` selects which encoder's layout to read back, and it has to
    match: `embed_image` writes a single frame_id copy (a still image's frame
    is always 0 and the BCH meta already carries it, so the space goes to the
    provenance ids instead), while `embed_video` writes FRAME_REPEATS of them.
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        return None

    parsed = _bch_extract_ids(img_array)
    if parsed:
        return parsed

    Y, _, _ = to_float_ycbcr(img_array[:, :, :3])

    meta_raw_bits = META_PAYLOAD_BYTES * 8                          # 192
    meta_padded   = ((_BCH_K - meta_raw_bits % _BCH_K) % _BCH_K) + meta_raw_bits
    n_meta_coded  = (meta_padded // _BCH_K) * _BCH_N               # 420

    # Rebuild the encoder's own repeat plan rather than assuming the maxima.
    # The copies that fit depend on the image's capacity, so the section
    # boundaries move with it — and reading them at fixed offsets silently
    # lands mid-section.  This was wrong for every still image: embed_image
    # writes ONE frame_id copy, so skipping FRAME_REPEATS (5) of them
    # overshoots by 128 bits and starts reading inside the owner section,
    # which is why blind lookup returned the media id in the owner slot.
    n_frame_copies, n_owner_copies, n_media_copies, n_chain_copies = \
        _plan_repeat_copies(_llfamily_block_capacity(Y.shape) - n_meta_coded,
                            frame_repeats=frame_repeats)
    n_extra = (n_frame_copies * FRAME_ID_BITS +
               n_owner_copies * OWNER_ID_BITS +
               n_media_copies * MEDIA_ID_BITS +
               n_chain_copies * CHAIN_TAG_BITS)

    res = _verify_channel(Y, n_meta_coded, b"\x00" * 32, 0,
                          n_extra_llll_bits=n_extra)

    # ── Try 3: majority-vote fallback on repetition copies ──
    # Layout matches embed order: frame → owner → media → chain.
    extra = res["extra_llll_bits"]
    p = min(n_frame_copies * FRAME_ID_BITS, len(extra))             # skip frame_id

    n_owner = min(n_owner_copies * OWNER_ID_BITS, max(0, len(extra) - p))
    owner_section = extra[p:p + n_owner]
    p += n_owner

    n_media = min(n_media_copies * MEDIA_ID_BITS, max(0, len(extra) - p))
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


def _resync_extract_ids(img_array: np.ndarray) -> dict | None:
    """Find the ids in a copy whose pixel grid moved — a screenshot, usually.

    Kept apart from `_blind_extract_ids`, and reached only after a lookup on
    that function's answer has come back empty, because there is no cheap test
    for "the grid moved".  One was tried: score the image against its own
    wrong-phase samples and search only when it fails to stand out.  Measured
    across the corpus the two populations overlap completely — an aligned frame
    scored as low as 0.35x its null while a 2x upscale reached 4.39x, because
    an integer upscale leaves every other block boundary still on the grid.  No
    threshold separates them, so the only honest test is the search itself.

    Letting the database decide keeps that cost off the common path: a file
    that was merely recompressed resolves through the vote path and never gets
    here, and only a file nothing matches pays for the search.
    """
    recovered, _info = resync_extract(img_array, _bch_extract_ids,
                                      time_budget=RESYNC_BUDGET_S)
    return recovered


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
                # embed_video lays out FRAME_REPEATS frame_id copies, unlike
                # the single copy embed_image writes — see _blind_extract_ids.
                parsed = _blind_extract_ids(rgb, frame_repeats=FRAME_REPEATS)
                if parsed:
                    break
            cap.release()
        finally:
            tmp_path.unlink(missing_ok=True)

    if not parsed:
        raise HTTPException(404, "Could not extract watermark from this file")

    owner_id = parsed["owner_id"]
    media_id = parsed["media_id"]

    # Match on the exact embedded 8-byte key (lowercased), unique per owner via
    # the index from migration 002.  This is what the watermark actually
    # carries, so resolution is unambiguous — no prefix over-matching where one
    # id is a prefix of another.  The readable owner/media strings come back
    # from the row itself.
    #
    # For files encoded after migration 003 the owner key is an assigned
    # short_id, so it identifies exactly one account.  Files encoded before it
    # carry an email prefix and remain resolvable through their original row —
    # which is why 003 does not rewrite historic owner_keys.
    owner_key = owner_id.lower()
    media_key = media_id.lower()
    res = (
        supabase.table("watermarks")
        .select("*")
        .eq("owner_key", owner_key)
        .eq("media_key", media_key)
        .execute()
    )

    if not res.data and kind == "image":
        # Nothing matched, so the ids read out of the pixels are suspect — most
        # often because the upload is a screenshot and the grid they came from
        # has moved.  Search for the real geometry and ask once more.
        retry = _resync_extract_ids(img)
        if retry and (retry["owner_id"].lower(), retry["media_id"].lower())                 != (owner_key, media_key):
            owner_id, media_id = retry["owner_id"], retry["media_id"]
            owner_key, media_key = owner_id.lower(), media_id.lower()
            res = (
                supabase.table("watermarks")
                .select("*")
                .eq("owner_key", owner_key)
                .eq("media_key", media_key)
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


def _record_verification(user: dict | None, resp: dict, meta: dict, kind: str) -> None:
    """Append this run to the caller's verification history.

    Best-effort and deliberately silent: /verify works without an account and
    without the database, and adding history must not introduce a new way for
    it to fail.  A logging problem should never cost the user their report, so
    every error here is swallowed.

    No-op for anonymous callers — /verify is public, so there is frequently no
    one to attribute the run to.
    """
    if not user:
        # Logged, not silent.  An anonymous call and an expired token produce
        # the same empty history list in the UI, and without this line there is
        # no way to tell them apart — the early return left no trace anywhere.
        print("[verify] history SKIPPED: no authenticated caller "
              "(token absent, malformed, or expired)")
        return
    try:
        claimed_owner = meta.get("owner_id")
        claimed_media = meta.get("media_id")

        # Best-effort link back to the encode record, via the same 8-byte keys
        # /lookup matches on.  Nullable by design: users can verify files they
        # never encoded.
        watermark_id = None
        if claimed_owner and claimed_media:
            try:
                hit = (
                    supabase.table("watermarks")
                    .select("id")
                    .eq("owner_key", _id_key(str(claimed_owner), OWNER_ID_BYTES))
                    .eq("media_key", _id_key(str(claimed_media), MEDIA_ID_BYTES))
                    .limit(1)
                    .execute()
                )
                if hit.data:
                    watermark_id = hit.data[0]["id"]
            except Exception:
                pass

        supabase.table("verifications").insert({
            "id":               uuid.uuid4().hex[:12],
            "user_id":          user["id"],
            "file_name":        resp.get("fileName"),
            "kind":             kind,
            "status":           resp.get("status"),
            "reasons":          resp.get("reasons") or [],
            "ber":              resp.get("ber"),
            "wm_accuracy":      resp.get("wmAccuracy"),
            "block_flag_ratio": resp.get("blockFlagRatio"),
            "blocks_tampered":  resp.get("blocksTampered"),
            "blocks_total":     resp.get("blocksTotal"),
            "claimed_owner":    meta.get("owner_email") or claimed_owner,
            "claimed_media":    claimed_media,
            "recovered_owner":  resp.get("ownerId"),
            "recovered_media":  resp.get("mediaId"),
            "watermark_id":     watermark_id,
        }).execute()
        print(f"[verify] history RECORDED for user {user.get('id')}")
    except Exception as e:
        # Still swallowed — a logging failure must never cost the user their
        # report — but named, so the cause is one line away instead of unknown.
        print(f"[verify] history FAILED for user {user.get('id')}: "
              f"{type(e).__name__}: {e}")


@app.get("/me/verifications")
def list_my_verifications(user: dict = Depends(get_current_user)):
    res = (
        supabase.table("verifications")
        .select("*")
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    return {"items": res.data or []}


@app.delete("/me/verifications/{check_id}")
def delete_my_verification(check_id: str, user: dict = Depends(get_current_user)):
    """Remove one entry from the caller's verification history.

    Unlike deleting an encode record, this has no side effects: a history row
    is only a summary of a past report, and nothing else reads it.  Scoped by
    user_id so one account cannot delete another's row, and fetched first so a
    missing id is a clean 404 rather than a silent no-op reported as success.
    """
    res = (
        supabase.table("verifications")
        .select("id")
        .eq("id", check_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Not found or not yours")

    supabase.table("verifications").delete().eq("id", check_id).eq("user_id", user["id"]).execute()
    return {"deleted": check_id}


@app.delete("/me/verifications")
def clear_my_verifications(user: dict = Depends(get_current_user)):
    """Clear the caller's whole verification history.

    Deletes every row for this user, not just the 100 the list endpoint shows,
    so "clear" leaves nothing behind to reappear later.
    """
    res = supabase.table("verifications").delete().eq("user_id", user["id"]).execute()
    return {"deleted": len(res.data or [])}


# Bit agreement against the expected ids that marks a candidate geometry as the
# right one.  A wrong grid yields uncorrelated bits and lands near 0.5; the true
# grid clears 0.9 even on content whose watermark is too damaged for BCH.  At
# 0.75 over 128 bits a false accept is ~1e-8 per candidate, and it only decides
# WHERE to read - verify_image still reports what it actually finds there.
GEOMETRY_BIT_AGREEMENT = 0.75


def _make_geometry_oracle(meta: dict):
    """An oracle that finds the grid using what `/verify` already knows.

    `_bch_extract_ids` is the right oracle for a blind lookup, where the
    payload has to authenticate itself.  Verification is not blind: the caller
    supplied the ids, so a candidate can be judged by how well the bits under
    it agree with those — and that survives damage BCH cannot correct.  It is
    also cheaper, since agreement needs no BCH decode at all (~2ms against
    ~66ms), which matters across a search of thousands of candidates.

    Self-validating for the same reason as the integrity tag: agreement this
    high against a *specific* 128-bit expectation cannot arise from a grid that
    is merely plausible.
    """
    exp = np.concatenate([
        bits_to_array(_encode_id_fixed(meta["owner_id"], OWNER_ID_BYTES)),
        bits_to_array(_encode_id_fixed(meta["media_id"], MEDIA_ID_BYTES)),
    ]).astype(np.uint8)

    n_frame = int(meta.get("frame_repeat_copies", 0))
    n_owner = int(meta.get("owner_repeat_copies", 0))
    n_media = int(meta.get("media_repeat_copies", 0))
    n_chain = int(meta.get("chain_repeat_copies", 0))
    n_extra = (n_frame * FRAME_ID_BITS + n_owner * OWNER_ID_BITS +
               n_media * MEDIA_ID_BITS + n_chain * CHAIN_TAG_BITS)
    n_meta_coded = int(meta["meta_bits_coded"])

    if n_owner < 2 or n_media < 2:
        return None          # too few copies to vote on; caller falls back

    def rows_needed(width: int) -> int:
        """Pixel rows the payload spans at this width, so the restore can stop
        there.  One bit per 8x8 block of a level-2 subband is one bit per 32x32
        pixel tile, and the reader fills LLLL row-major before spilling into
        the next subband — so covering these rows covers the whole payload
        without the spill ever happening."""
        blocks_per_row = max(1, width // 32)
        return -(-(n_meta_coded + n_extra) // blocks_per_row) * 32

    def oracle(img: np.ndarray) -> dict | None:
        if img.ndim < 3 or img.shape[2] < 3 or min(img.shape[:2]) < 64:
            return None
        Y, _, _ = to_float_ycbcr(img[:, :, :3])
        res = _verify_channel(Y, n_meta_coded, b"\x00" * 32, 0,
                              n_extra_llll_bits=n_extra)
        extra = res["extra_llll_bits"]
        if len(extra) < n_extra:
            return None

        p = n_frame * FRAME_ID_BITS
        owner_voted = _majority_vote(extra[p:p + n_owner * OWNER_ID_BITS],
                                     OWNER_ID_BITS)
        p += n_owner * OWNER_ID_BITS
        media_voted = _majority_vote(extra[p:p + n_media * MEDIA_ID_BITS],
                                     MEDIA_ID_BITS)
        if owner_voted is None or media_voted is None:
            return None

        got = np.concatenate([owner_voted, media_voted]).astype(np.uint8)
        if got.size != exp.size:
            return None
        agreement = float(np.mean(got == exp))
        return {"agreement": agreement} if agreement >= GEOMETRY_BIT_AGREEMENT else None

    oracle.rows_needed = rows_needed          # type: ignore[attr-defined]
    return oracle


def _best_known_geometry(img: np.ndarray, oracle, known: tuple[int, int],
                         deadline: float) -> dict | None:
    """Scan crop offsets at the known original size; return the best fit.

    A screenshot of a file we hold metadata for has only two unknowns left, and
    one of them is answered: the size is on record, so what remains is the
    phase the capture rectangle started on.  That is a 64-candidate scan, small
    enough to evaluate exhaustively and take the maximum rather than settle for
    the first candidate over the threshold — near-misses clear the threshold
    too, and verify worse than the exact geometry sitting beside them.

    Only the strip the payload occupies is restored per candidate.  Restoring
    the whole frame measured 481ms on a 2000x2509 upload, which turned this
    scan into 25 seconds; the strip is the same arithmetic on a fraction of the
    pixels.
    """
    rows = getattr(oracle, "rows_needed", None)
    max_rows = rows(known[0]) if rows else None

    best: dict | None = None
    best_agreement = 0.0
    for dy in range(RESYNC_MAX_OFFSET):
        for dx in range(RESYNC_MAX_OFFSET):
            if time.monotonic() > deadline:
                return best
            cand = {"width": known[0], "height": known[1], "dy": dy, "dx": dx}
            got = oracle(apply_transform(img, cand, max_rows=max_rows))
            if got and got["agreement"] > best_agreement:
                best_agreement = got["agreement"]
                best = dict(cand, score=round(got["agreement"], 4))
                if best_agreement >= 1.0:
                    return best          # exact; nothing can beat it
    return best


def _verify_image_resynced(img: np.ndarray, meta: dict) -> tuple[dict, dict | None]:
    """`verify_image`, but recovering first if the copy was rescaled or cropped.

    Verification reads the watermark from fixed block positions, so a copy
    whose pixel grid moved — a screenshot, most often — fails every field at
    once: not because the watermark is gone, but because it is no longer being
    read from where it was written.  Reported plainly that comes out as "the
    owner ID could not be matched", which is indistinguishable from a genuine
    forgery and is the wrong thing to tell someone about their own file.

    So when nothing matches AND the grid looks displaced, the geometry is
    searched for and verification is re-run on the restored image.

    The search is driven by the cheap BCH oracle rather than by `verify_image`
    itself: the oracle costs ~66ms against a full verification's chroma
    analysis, and the search may try hundreds of candidates before it lands.
    Only the winner is verified properly.

    Note what this does NOT do: it restores geometry, never the verdict.  A
    resampled image has lost the fragile chroma layer, so it can still come
    back tampered — correctly, since it genuinely is a re-encoded copy.  What
    changes is that the report now names the real reason instead of blaming the
    owner ID.
    """
    verdict = verify_image(img, meta)
    if verdict.get("owner_match") or verdict.get("media_match"):
        return verdict, None                      # grid was where we expected

    if img.ndim < 3 or img.shape[2] < 3:
        return verdict, None

    # No gate here on whether the grid "looks" displaced: the cheap test for
    # that does not work.  See `_resync_extract_ids` — an aligned frame and a
    # 2x upscale land in the same range, so the search is the only real test.
    # It stays bounded instead: `_best_known_geometry` is a fixed 64-candidate
    # scan whenever the metadata records the original size.

    # Metadata written since the dimensions were added turns the scale search
    # into a single candidate; older metadata just falls back to searching.
    known = None
    if meta.get("width") and meta.get("height"):
        known = (int(meta["width"]), int(meta["height"]))

    # The agreement oracle reads the grid through damage that defeats BCH,
    # which is the difference between recovering a heavily textured photo and
    # not.  It needs the repeat-count fields, so metadata predating them falls
    # back to BCH.
    oracle = _make_geometry_oracle(meta)

    # One deadline for the whole recovery, shared by every stage below.  The
    # known-size scan and the fallback search used to receive RESYNC_BUDGET_S
    # each, so an upload that matched nothing paid for both: measured 14-19s
    # against the 12s this constant claims.  That is precisely the path an
    # unwatermarked upload takes on a public endpoint, so it is the one that
    # most needs a real bound.
    deadline = time.monotonic() + RESYNC_BUDGET_S

    # With the original dimensions known, the entire realistic screenshot space
    # — rescaled, cropped, or both — collapses to one scan over the offsets, so
    # scan it directly and take the BEST agreement rather than the first one
    # over the line.  That distinction matters: the oracle accepts a merely
    # good geometry, and a near-miss accepted greedily verifies worse than the
    # exact one it was standing in front of.  Measured, greedy acceptance
    # turned a 3px crop that had matched both ids into a match on neither.
    if oracle and known:
        best_src, best = img, _best_known_geometry(img, oracle, known, deadline)

        # A real screenshot usually caught some of the page around the image,
        # and that is not a crop the offset scan can undo — the content is a
        # sub-rectangle, so the row stride itself is wrong.  Scan inside the
        # content rectangle too, and keep whichever fits better.
        #
        # Both are scanned, rather than trimming only when the first attempt
        # comes back empty, because a framed upload does not come back empty:
        # it comes back with a near-miss that clears the threshold.  Measured
        # on a 2px border, the untrimmed scan offered 0.945 and the trimmed one
        # was exact, so stopping at the first acceptable answer took the worse
        # of the two and matched neither id.
        if not best or best["score"] < 1.0:
            box = content_box(img)
            if box:
                top, bottom, left, right = box
                inner = img[top:bottom, left:right]
                alt = _best_known_geometry(inner, oracle, known, deadline)
                if alt and (not best or alt["score"] > best["score"]):
                    best_src = inner
                    best = dict(alt, trimmed=[left, top, right, bottom])

        if best:
            return verify_image(apply_transform(best_src, best), meta), best

    # Whatever the known-size scan left.  Nothing left means the answer is
    # already in: the scan exhausted the budget without a fit.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return verdict, None

    found, info = resync_extract(
        img, oracle or _bch_extract_ids, known_dims=known,
        time_budget=remaining)
    transform = info.get("transform")
    if not found or not transform:
        return verdict, None

    restored = apply_transform(img, transform)
    return verify_image(restored, meta), transform


@app.post("/verify")
async def verify(
    file:     UploadFile = File(...),
    metadata: str        = Form(...),
    user:     dict | None = Depends(get_optional_user),
):
    kind    = _detect_kind(file.filename)
    file_id = uuid.uuid4().hex[:12]
    in_ext  = os.path.splitext(file.filename)[1].lower()
    in_path = UPLOAD_DIR / f"{file_id}_verify{in_ext}"

    in_path.write_bytes(await file.read())

    # `finally` so the upload is removed on the error paths too — a bad
    # metadata JSON or an unreadable file must not leave media behind.  The
    # Verify page tells users their upload is deleted once the report is
    # generated; this is what makes that true.
    try:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid metadata JSON: {e}")

        if kind == "image":
            img        = np.array(Image.open(in_path).convert("RGB"))
            img_h, img_w = img.shape[:2]
            verdict, resync_info = _verify_image_resynced(img, meta)
            resp       = _shape_image_response(verdict, file.filename, img_w, img_h,
                                               resync_info)
        else:
            verdict = verify_video(str(in_path), meta, sample_frames=None)
            resp    = _shape_video_response(verdict, file.filename)

        _record_verification(user, resp, meta, kind)
        return resp
    finally:
        in_path.unlink(missing_ok=True)


def _tamper_reasons(v: dict) -> list:
    """Why a verdict came out `tampered`, in the user's words.

    The verdict is a single boolean OR'd from unrelated signals, so on its own
    it can't say whether the file was edited, re-encoded past recovery, or had
    frames removed.  Previously the UI compensated with a "confidence" score
    that measured cleanliness — which inverts on a tampered verdict and could
    read "TAMPERED — 100% confidence".  Naming the actual trigger is both
    truthful and more useful than any percentage.
    """
    reasons = []
    if v.get("content_tampered"):
        n = int(v.get("n_blocks_tampered", 0))
        reasons.append(f"{n} image region{'s' if n != 1 else ''} were modified")
    if not v.get("owner_match"):
        reasons.append("the owner ID could not be matched")
    if not v.get("media_match"):
        reasons.append("the media ID could not be matched")
    if v.get("frame_match") is False and v.get("frame_recovery") != "none":
        reasons.append("the frame number did not match")
    if v.get("chain_match") is False and v.get("chain_recovery") != "none":
        reasons.append("the chain tag did not match")
    return reasons


def _resync_reason(resync: dict, width: int, height: int) -> str:
    """Describe, in the user's terms, what had to be undone to read the file."""
    parts = []
    if (int(resync["width"]), int(resync["height"])) != (int(width), int(height)):
        parts.append(f"rescaled from {width}x{height} to "
                     f"{int(resync['width'])}x{int(resync['height'])}")
    if resync.get("dx") or resync.get("dy"):
        parts.append(f"cropped by {int(resync['dx'])}px left, "
                     f"{int(resync['dy'])}px top")
    what = " and ".join(parts) if parts else "geometrically altered"
    return (f"this copy was {what} - typical of a screenshot; "
            "the watermark was read after restoring the original geometry")


def _shape_image_response(v: dict, filename: str, width: int, height: int,
                          resync: dict | None = None) -> dict:
    # A genuine bit error rate: the fraction of fragile-watermark parity bits
    # that flipped.  The old `ber` was `spatial.mean()` — the fraction of
    # flagged 32x32 BLOCKS — which is a localisation ratio, not a bit rate, and
    # was mislabelled "Bit Error Rate" in the UI.  Both are reported now, each
    # under its own name.
    ber_bits = float(v.get("ber_lsb_sub", v.get("ber", 0.0)))

    # `spatial_map` (and the watermark comparison grids) were computed on
    # whichever image `verify_image()` actually ran on: the geometrically
    # restored image when resync fired, the raw upload otherwise. Report
    # THAT size as imageWidth/imageHeight so the frontend's percentage-based
    # overlay lines up with the map that produced it, instead of always the
    # raw upload's dimensions (which can differ after a screenshot's scale
    # or crop was undone). Mirrors how the video path derives its dims
    # straight from spatial_tamper_map's shape, below.
    map_width, map_height = (
        (int(resync["width"]), int(resync["height"])) if resync else (width, height)
    )

    reasons = _tamper_reasons(v) if v["tampered"] else []
    if resync:
        # Say this first.  Everything else in the report describes a file that
        # had to be geometrically restored before it could be read at all, and
        # without that context a "tampered" verdict on someone's own screenshot
        # reads as an accusation rather than as a description of resampling.
        reasons.insert(0, _resync_reason(resync, width, height))

    return {
        "status":          "tampered" if v["tampered"] else "authentic",
        # Measured, not synthesised: what fraction of the fragile watermark's
        # bits survived. No "confidence" score — see _tamper_reasons.
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - ber_bits)),
        "ber":             ber_bits,
        "blockFlagRatio":  float(v["ber"]),
        "reasons":         reasons,
        # Present only when the upload's pixel grid had to be restored before
        # the watermark could be read — the UI can use it to explain why a
        # screenshot verifies at all, and why it may still read as modified.
        "resynced":        ({"width":  int(resync["width"]),
                             "height": int(resync["height"]),
                             "offsetX": int(resync["dx"]),
                             "offsetY": int(resync["dy"])} if resync else None),
        "fileType":        "image",
        "fileName":        filename,
        "imageWidth":      int(map_width),
        "imageHeight":     int(map_height),
        # Only surface flagged regions on a real content edit — below the
        # MIN_TAMPER_BLOCKS floor the flagged blocks are quantization noise, so
        # we report none (consistent with the clean watermark comparison).
        "tamperedRegions": _spatial_to_regions(v.get("spatial_map")) if v.get("content_tampered") else [],
        "watermarkFound":  bool(v["watermark_found"]),
        "ownerMatch":      bool(v["owner_match"]),
        "mediaMatch":      bool(v["media_match"]),
        "owner":           v.get("owner"),
        "media":           v.get("media"),
        # The RAW strings pulled out of the pixels — present even on a
        # mismatch, which is exactly when they matter: the UI can show
        # "expected 7nnzgbne, extracted 7nnzg8ne" instead of echoing back the
        # expected value with a "Mismatch" badge stuck on it.
        "ownerId":         v.get("owner_id_recovered"),
        "mediaId":         v.get("media_id_recovered"),
        "blocksTampered":  int(v["n_blocks_tampered"]),
        "blocksTotal":     int(v["n_blocks_total"]),
        # Fragile-watermark comparison images (PNG data URLs): the expected
        # pattern vs. the one extracted from this file (tampered bits in red).
        "watermarkOriginal":  _pattern_to_data_url(v.get("wm_target_pattern")),
        "watermarkExtracted": _pattern_to_data_url(
            v.get("wm_extracted_pattern"), v.get("wm_display_mismatch")),
        # The same flagged cells again, but transparent — for laying over the
        # source photo client-side.  Images only: video frames already carry two
        # base64 PNGs each, and a third per frame would bloat those responses.
        "watermarkMask": _mask_to_data_url(v.get("wm_display_mismatch")),
        # Bounding boxes grouping those cells into edits. A lower bound on each
        # edited region, not its true extent — see _mask_to_clusters.
        "watermarkClusters": _mask_to_clusters(v.get("wm_display_mismatch")),
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
        pf = per_frame[i] if i < len(per_frame) else {}
        # Only surface flagged regions for a real content edit on this frame —
        # below the MIN_TAMPER_BLOCKS floor the blocks are quantization noise.
        per_frame_map = None
        if pf.get("content_tampered") and spatial_arr is not None \
                and hasattr(spatial_arr, "shape") \
                and spatial_arr.ndim == 3 and i < spatial_arr.shape[0]:
            per_frame_map = spatial_arr[i]
        # Prefer the recovered embedded frame_id (the original frame number);
        # fall back to playback position when it couldn't be recovered.
        rec_fid = pf.get("rec_fid")
        true_frame_idx = rec_fid if rec_fid is not None else pf.get("frame_idx", i)
        # Per-frame numbers are measured, not assigned.  This used to be
        # `0.5 if t else 0.95` — two invented constants surfaced to the user as
        # a percentage.  `ber` here is the frame's real fragile-layer bit error
        # rate; `blocksTampered` is how much of it was flagged.
        frame_results.append({
            "frame":           int(true_frame_idx),
            "status":          "tampered" if t else "authentic",
            "ber":             float(pf.get("ber_lsb_sub", 0.0)),
            "blocksTampered":  int(pf.get("n_blocks_tampered", 0)),
            "blocksTotal":     int(pf.get("n_blocks_total", 0)),
            "tamperedRegions": _spatial_to_regions(per_frame_map),
            # Per-frame fragile-watermark comparison (present for every frame
            # so the user can inspect whichever frame they select).
            "watermarkOriginal":  _pattern_to_data_url(pf.get("wm_target_pattern")),
            "watermarkExtracted": _pattern_to_data_url(
                pf.get("wm_extracted_pattern"), pf.get("wm_display_mismatch")),
        })

    # Re-insert deleted frames as red placeholders at their original position
    # so the timeline still shows the full frame count (e.g. frame 5 deleted →
    # 1-4 + [5 deleted] + 6-10), instead of collapsing to the survivors.  Their
    # "original" watermark is shown next to a deleted placeholder on the front.
    for mp in v.get("missing_frame_patterns", []) or []:
        frame_results.append({
            "frame":              int(mp["frame_id"]),
            "status":             "deleted",
            "ber":                None,   # nothing to measure — the frame is gone
            "blocksTampered":     0,
            "blocksTotal":        0,
            "tamperedRegions":    [],
            "watermarkOriginal":  _pattern_to_data_url(mp.get("target_pattern")),
            "watermarkExtracted": None,   # nothing to extract — the frame is gone
        })
    frame_results.sort(key=lambda r: r["frame"])

    # Pixel dimensions derived from the spatial block grid so the frontend
    # heatmap scales correctly per video.
    video_w = video_h = None
    if spatial_arr is not None and hasattr(spatial_arr, "shape") and spatial_arr.ndim == 3:
        _, Hb, Wb = spatial_arr.shape
        video_h = int(Hb * SPATIAL_BLOCK)
        video_w = int(Wb * SPATIAL_BLOCK)

    # Identity fields — previously omitted entirely for video, which left the
    # Results page with no Match/Mismatch badges and forced it to guess whether
    # a watermark had been found at all.  Derived from the per-frame results:
    # a frame's recovered id is the majority value across every frame that
    # recovered one.
    def _modal(key: str):
        vals = [r.get(key) for r in per_frame if r.get(key)]
        return max(set(vals), key=vals.count) if vals else None

    owner_recovered = _modal("owner_id_recovered")
    media_recovered = _modal("media_id_recovered")
    n_checked       = max(len(per_frame), 1)
    owner_match_all = v.get("owner_match_frames", 0) > n_checked / 2
    media_match_all = v.get("media_match_frames", 0) > n_checked / 2

    # Same treatment as the image path: a real bit error rate (mean fragile-layer
    # BER across verified frames) rather than the block-flag ratio that used to
    # be reported under that name.
    per_frame_bers = [float(r.get("ber_lsb_sub", 0.0)) for r in per_frame]
    ber_bits = float(sum(per_frame_bers) / len(per_frame_bers)) if per_frame_bers else 0.0

    # Why the clip was flagged. Temporal edits are the case the old "confidence"
    # number handled worst: deleting a frame leaves every survivor authentic, so
    # frame_tamper_rate stayed 0 and the UI announced "TAMPERED — 100%".
    reasons = []
    n_content = int(v.get("frames_content_tampered", 0))
    if n_content:
        reasons.append(f"{n_content} frame{'s' if n_content != 1 else ''} had modified regions")
    n_ident = int(v.get("frames_id_mismatch", 0))
    if n_ident:
        reasons.append(f"{n_ident} frame{'s' if n_ident != 1 else ''} failed the identity check")
    n_chain = int(v.get("frames_chain_break", 0))
    if n_chain:
        reasons.append(f"{n_chain} frame{'s' if n_chain != 1 else ''} had an inconsistent chain tag")
    n_del = len(v.get("missing_frame_ids", []) or [])
    if n_del:
        reasons.append(f"{n_del} frame{'s' if n_del != 1 else ''} deleted")
    if v.get("reordered"):
        reasons.append("frames were reordered")
    n_dup = len(v.get("duplicate_frame_ids", []) or [])
    if n_dup:
        reasons.append(f"{n_dup} frame{'s' if n_dup != 1 else ''} duplicated")
    if int(v.get("frames_truncated", 0)) > 2:
        reasons.append(f"{int(v['frames_truncated'])} frames truncated from the end")

    return {
        "status":          "tampered" if v["TAMPERED"] else "authentic",
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - ber_bits)),
        "ber":             ber_bits,
        "blockFlagRatio":  float(v["average_ber"]),
        "reasons":         reasons if v["TAMPERED"] else [],
        "fileType":        "video",
        "fileName":        filename,
        "tamperedRegions": [],
        "watermarkFound":  bool(owner_recovered or media_recovered),
        "ownerMatch":      bool(owner_match_all),
        "mediaMatch":      bool(media_match_all),
        "owner":           v.get("owner"),
        "media":           v.get("media"),
        "ownerId":         owner_recovered,
        "mediaId":         media_recovered,
        "frameResults":    frame_results,
        "imageWidth":      video_w,
        "imageHeight":     video_h,
        "framesChecked":   int(v["frames_checked"]),
        "framesTampered":  int(v["frames_tampered"]),
        "frameTamperRate": float(v.get("frame_tamper_rate", 0.0)),
        "chainBreaks":     int(v["frames_chain_break"]),
        "idMismatches":    int(v["frames_id_mismatch"]),
        # Temporal localization — deletions/reorders surfaced as discrete
        # events instead of cascading across every later frame.
        "missingFrames":   [int(f) for f in v.get("missing_frame_ids", [])],
        "framesDeleted":   int(v.get("frames_deleted", 0)),
        "reordered":       bool(v.get("reordered", False)),
        "reorderAt":       [int(x) for x in v.get("reorder_at", [])],
        "duplicateFrames": [int(f) for f in v.get("duplicate_frame_ids", [])],
        "framesTruncated": int(v.get("frames_truncated", 0)),
    }