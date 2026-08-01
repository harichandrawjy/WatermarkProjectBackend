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
import io
import uuid
import json
import base64
import secrets
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
    _majority_vote, _decode_id_fixed, _encode_id_fixed,
    OWNER_ID_BITS, MEDIA_ID_BITS, FRAME_ID_BITS, CHAIN_TAG_BITS,
    OWNER_ID_BYTES, MEDIA_ID_BYTES,
    OWNER_REPEATS, MEDIA_REPEATS, FRAME_REPEATS, CHAIN_REPEATS,
)
from video_watermark import embed_video, verify_video

import httpx

from db import supabase, supabase_auth, SUPABASE_URL, SUPABASE_ANON_KEY

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
    return {"id": user["id"], "email": user["email"]}


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
    file:     UploadFile = File(...),
    media_id: str        = Form(...),
    user:     dict       = Depends(get_current_user),
):
    # Two different strings, deliberately:
    #   owner_email — the human-readable account, kept in the DB and returned
    #                 by /lookup.  Never embedded.
    #   owner       — the assigned 8-char short_id that IS embedded.  Using the
    #                 email here would truncate to its first 8 bytes and merge
    #                 distinct accounts into one owner (see migration 003).
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
            verdict    = verify_image(img, meta)
            return _shape_image_response(verdict, file.filename, img_w, img_h)
        else:
            verdict = verify_video(str(in_path), meta, sample_frames=None)
            return _shape_video_response(verdict, file.filename)
    finally:
        in_path.unlink(missing_ok=True)


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
        frame_results.append({
            "frame":           int(true_frame_idx),
            "status":          "tampered" if t else "authentic",
            "confidence":      0.5 if t else 0.95,
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
            "confidence":         0.0,
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

    return {
        "status":          "tampered" if v["TAMPERED"] else "authentic",
        "confidence":      max(0.0, min(1.0, 1.0 - float(v.get("frame_tamper_rate", 0.0)))),
        "wmAccuracy":      max(0.0, min(1.0, 1.0 - float(v["average_ber"]))),
        "ber":             float(v["average_ber"]),
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