"""
Video Watermarking
==================
Applies the semi-fragile watermark per frame and produces both a SPATIAL
tamper map (which 32x32 cover blocks were modified, per frame) and a
TEMPORAL tamper map (which frames were modified) — matching the
"Spatial localization map" → "Temporal localization map" → "Integrity
report + tamper map" outputs in decoder.png.

Temporal integrity is enforced by the BCH-protected frame_id in the LLLL
meta payload of every embedded frame.  At verify time the decoder reads
the frame_id back from each frame and compares it to the iteration index
in the stream — any deletion / insertion / reorder shifts which frame_id
lands at which index, so the comparison fails and the frame is flagged
as a chain break.  Per-block fingerprints stay purely spatial (no chain
hash mixed in), so a temporal edit no longer also inflates the
content-tampering BER.
"""

import cv2
import numpy as np
import hashlib
import json
import os
import shutil
import subprocess
from typing import Optional, Dict, List

from watermark_engine import (
    bch_encode_bits, bch_decode_bits,
    bits_to_array, array_to_bytes,
    to_float_ycbcr, from_float_ycbcr,
    _embed_into_channel, _verify_channel,
    _embed_lsb_chroma, _verify_lsb_chroma,

    _build_meta_payload, _parse_meta_payload,
    _master_signature, _meta_owner_media_id,
    _owner_id_bits_array, _media_id_bits_array, _majority_vote,
    _frame_id_bits_array, _chain_tag_bits_array,
    _encode_id_fixed, _decode_id_fixed,
    _chain_genesis, _frame_chain_hash,

    OWNER_ID_BYTES, MEDIA_ID_BYTES,
    OWNER_ID_BITS, MEDIA_ID_BITS,
    OWNER_REPEATS, MEDIA_REPEATS, FRAME_REPEATS, CHAIN_REPEATS,
    FRAME_ID_BITS, CHAIN_TAG_BITS,

    BER_TAMPER_THRESHOLD, MIN_TAMPER_BLOCKS,
)
import struct


# ──────────────────────────────────────────────────────────────
# FFmpeg subprocess writer  (lossy MP4 path — OpenCV's avc1 is unreliable
# on Windows because of openh264 DLL version mismatches.  We pipe raw
# BGR frames into ffmpeg directly with libx264 instead.)
# ──────────────────────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable.

    Prefers the bundled binary from `imageio-ffmpeg` (pip-installable,
    no system setup) and falls back to system PATH.  Returns None if
    neither is available.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


class _FFmpegWriter:
    """Minimal subset of cv2.VideoWriter's API backed by an ffmpeg subprocess.

    Pipes raw BGR24 frames into `ffmpeg -c:v libx264 -crf 18` so the resulting
    MP4 actually uses H.264 (not OpenCV's broken openh264 path or the legacy
    mp4v codec).  CRF=18 is "visually lossless" — the best chance the
    semi-fragile watermark survives.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 ffmpeg_exe: str, crf: int = 18, preset: str = "medium",
                 audio_source_path: Optional[str] = None,
                 codec: str = "libx264", pix_fmt: str = "yuv444p"):
        self._proc: Optional[subprocess.Popen] = None
        self._opened = False
        self._output_path = output_path
        # yuv444p (no chroma subsampling) — required for the watermark's
        # ownership/LLLL bits to survive the BGR↔YUV↔BGR round-trip.  yuv420p
        # reconstructs chroma by upsampling, which perturbs RGB → recomputed
        # Y differs from what was embedded → BCH + majority-vote both fail
        # even at CRF=0 (verified empirically — 0/20 frames recover owner
        # under lossless yuv420p, 20/20 under lossless yuv444p).  Cost: the
        # file uses the High 4:4:4 Predictive profile which some hardware
        # decoders / browsers don't support; software players (VLC, ffplay)
        # play it fine.
        cmd = [
            ffmpeg_exe, "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
        ]
        if audio_source_path is not None:
            # Second input: the original file, used only to copy its audio
            # stream into the output.  `1:a?` makes the audio map optional —
            # a silent source still encodes cleanly.  `-c:a copy` avoids any
            # re-encode of the audio (bit-identical, no extra deps).
            cmd.extend([
                "-i", audio_source_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:a", "copy",
            ])
        else:
            cmd.append("-an")
        vid_opts = ["-c:v", codec]
        if codec != "ffv1":
            vid_opts.extend(["-preset", preset, "-crf", str(crf)])
        vid_opts.extend(["-pix_fmt", pix_fmt, output_path])
        cmd.extend(vid_opts)
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self._opened = True
        except (OSError, ValueError):
            self._opened = False

    def isOpened(self) -> bool:
        return self._opened and self._proc is not None and self._proc.poll() is None

    def write(self, bgr: np.ndarray) -> None:
        if not self.isOpened():
            raise IOError("ffmpeg subprocess is not running")
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def release(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        # Wait for ffmpeg to finish flushing.  Raise if it failed so the
        # caller doesn't end up with a half-written file silently.
        rc = self._proc.wait()
        err = b""
        if self._proc.stderr is not None:
            try:
                err = self._proc.stderr.read()
            except Exception:
                pass
            self._proc.stderr.close()
        self._proc = None
        self._opened = False
        if rc != 0:
            raise IOError(
                f"ffmpeg exited with code {rc} writing {self._output_path}.  "
                f"stderr:\n{err.decode('utf-8', errors='replace')}"
            )


# ──────────────────────────────────────────────────────────────
# EMBED VIDEO
# ──────────────────────────────────────────────────────────────

def embed_video(
    input_path: str,
    output_path: str,
    owner_id: str,
    media_id: str,
    embed_every_n_frames: int = 1,
    progress_callback=None,
) -> Dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ext = os.path.splitext(output_path)[1].lower()
    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe is None:
        raise IOError(
            f"Cannot encode '{output_path}': no ffmpeg found.  "
            "Install it via `pip install imageio-ffmpeg`, or put ffmpeg on PATH."
        )

    # cv2.VideoWriter's FFV1 produces pixel-exact frames that the watermark
    # verifier can always decode.  It cannot carry audio, so we write a
    # video-only temp file first and mux audio from the original input
    # after the frame loop.
    ffv1_temp_path: Optional[str] = None
    if ext in ('.mkv', '.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        base, ext_clean = os.path.splitext(output_path)
        ffv1_temp_path = f"{base}.video_only{ext_clean}"
        out = cv2.VideoWriter(ffv1_temp_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"Cannot open VideoWriter for {output_path} (FFV1 unavailable)")
    else:
        out = _FFmpegWriter(output_path, fps=fps, width=width, height=height,
                            ffmpeg_exe=ffmpeg_exe, audio_source_path=input_path)
        if not out.isOpened():
            raise IOError(f"Cannot start ffmpeg subprocess for {output_path}")
        print(f"[embed] Lossy '{ext}' output via ffmpeg+libx264 (CRF=18).  "
              "Verification may report tampered if the codec perturbs LLLL "
              "beyond the BCH + majority-vote margin.")

    embedded_indices: List[int] = []
    n_meta_bits_raw = None
    n_meta_bits_coded = None
    n_owner_repeat_bits = None
    n_owner_repeat_copies = None
    n_media_repeat_bits = None
    n_media_repeat_copies = None

    # Owner + media repeats are constant across the clip — build once.
    # frame_id and chain_tag change per frame, so they're built inside the
    # loop and PREPENDED (frame_id leads the stream — highest priority,
    # lives in LLLL).  Priority order matches embed_image:
    #   frame_id → owner_id → media_id → chain_tag.
    owner_repeated = np.tile(_owner_id_bits_array(owner_id), OWNER_REPEATS)
    media_repeated = np.tile(_media_id_bits_array(media_id), MEDIA_REPEATS)
    constant_extras = np.concatenate([owner_repeated, media_repeated])

    # Per-frame chain state: chain_i = SHA256(media_hash || chain_{i-1} || frame_id_i).
    # First 2 bytes embedded as chain_tag in LLLL.  Catches reorder / replay
    # attacks that frame_id alone misses.
    media_hash_bytes = hashlib.sha256(media_id.encode()).digest()[:4]
    chain_state      = _chain_genesis(media_hash_bytes)

    n_frame_repeat_bits   = None
    n_frame_repeat_copies = None
    n_chain_repeat_bits   = None
    n_chain_repeat_copies = None

    # Per-frame PSNR (pre-encoding — measures the watermark's perturbation
    # only, NOT the codec's compression noise that follows).
    psnr_y_per_frame:   List[float] = []
    psnr_rgb_per_frame: List[float] = []

    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % embed_every_n_frames == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            signature   = _master_signature(owner_id, media_id, frame_idx)

            # Advance chain BEFORE building the payload — this frame's
            # chain_tag commits to frame_idx + all prior chain state.
            chain_state = _frame_chain_hash(media_hash_bytes, chain_state, frame_idx)
            chain_tag   = chain_state[:2]

            meta_raw   = _build_meta_payload(owner_id, media_id, frame_idx,
                                             chain_tag=chain_tag)
            meta_bits  = bits_to_array(meta_raw)
            meta_coded = bch_encode_bits(meta_bits)

            if n_meta_bits_raw is None:
                n_meta_bits_raw   = len(meta_bits)
                n_meta_bits_coded = len(meta_coded)

            # Per-frame extras: frame_id (LEADS the stream — highest priority,
            # lives in LLLL with 5x repeats per user spec), then constant
            # owner+media block, then chain_tag.  Order MUST match
            # embed_image's stream layout or verify drifts.
            frame_id_repeated  = np.tile(_frame_id_bits_array(frame_idx),  FRAME_REPEATS)
            chain_tag_repeated = np.tile(_chain_tag_bits_array(chain_tag), CHAIN_REPEATS)
            extras             = np.concatenate([frame_id_repeated,
                                                 constant_extras,
                                                 chain_tag_repeated])

            Y, Cb, Cr = to_float_ycbcr(rgb)
            Y_wm, info = _embed_into_channel(Y, meta_coded, signature, frame_idx, extras)

            # LSB layer on chroma — independent of Y's DCT/SVD watermark.
            Cb_wm = _embed_lsb_chroma(Cb, signature, frame_idx)

            if info["n_meta_bits"] < len(meta_coded):
                # Mirror embed_image: a frame too small to hold the BCH meta
                # cannot carry the watermark.  Fail loudly instead of writing
                # a raw frame the verifier would later flag as tampered with
                # no record of the embed-time skip.
                cap.release()
                out.release()
                raise ValueError(
                    f"Frame {frame_idx}: LLLL too small "
                    f"({info['n_meta_bits']}/{len(meta_coded)} BCH-coded meta bits). "
                    f"Use a larger video or shorter owner/media IDs."
                )

            if n_owner_repeat_copies is None:
                # First successful frame defines how many full copies of each
                # field fit in this video's LL-family capacity; verify uses
                # the same counts.  Priority order matches the extras stream:
                #   frame_id (5x) → owner_id (3x) → media_id (3x) → chain_tag (3x).
                n_extra_emb = info["n_extra_llll_bits"]
                n_frame_repeat_copies = min(FRAME_REPEATS, n_extra_emb // FRAME_ID_BITS)
                remaining             = max(0, n_extra_emb - n_frame_repeat_copies * FRAME_ID_BITS)
                n_owner_repeat_copies = min(OWNER_REPEATS, remaining // OWNER_ID_BITS)
                remaining            -= n_owner_repeat_copies * OWNER_ID_BITS
                n_media_repeat_copies = min(MEDIA_REPEATS, remaining // MEDIA_ID_BITS)
                remaining            -= n_media_repeat_copies * MEDIA_ID_BITS
                n_chain_repeat_copies = min(CHAIN_REPEATS, remaining // CHAIN_TAG_BITS)
                n_frame_repeat_bits   = n_frame_repeat_copies * FRAME_ID_BITS
                n_owner_repeat_bits   = n_owner_repeat_copies * OWNER_ID_BITS
                n_media_repeat_bits   = n_media_repeat_copies * MEDIA_ID_BITS
                n_chain_repeat_bits   = n_chain_repeat_copies * CHAIN_TAG_BITS
                if (n_frame_repeat_copies < FRAME_REPEATS
                        or n_owner_repeat_copies < OWNER_REPEATS
                        or n_media_repeat_copies < MEDIA_REPEATS
                        or n_chain_repeat_copies < CHAIN_REPEATS):
                    print(f"[embed_video] note: fit "
                          f"{n_frame_repeat_copies}/{FRAME_REPEATS} frame_id + "
                          f"{n_owner_repeat_copies}/{OWNER_REPEATS} owner_id + "
                          f"{n_media_repeat_copies}/{MEDIA_REPEATS} media_id + "
                          f"{n_chain_repeat_copies}/{CHAIN_REPEATS} chain_tag "
                          f"majority-vote copies per frame.")

            rgb_wm = from_float_ycbcr(Y_wm, Cb_wm, Cr)
            bgr_wm = cv2.cvtColor(rgb_wm, cv2.COLOR_RGB2BGR)
            out.write(bgr_wm)

            # Pre-encoding PSNR — measures watermark perturbation only, not
            # codec degradation that comes after.  Y-channel PSNR is the
            # standard reporting metric for video watermarking; RGB is the
            # broader perceptual proxy.
            mse_y   = float(np.mean((Y - Y_wm.astype(np.float64)) ** 2))
            mse_rgb = float(np.mean((rgb.astype(np.float64) - rgb_wm.astype(np.float64)) ** 2))
            psnr_y_per_frame.append(
                float(10.0 * np.log10(255.0 ** 2 / mse_y)) if mse_y > 0 else float("inf"))
            psnr_rgb_per_frame.append(
                float(10.0 * np.log10(255.0 ** 2 / mse_rgb)) if mse_rgb > 0 else float("inf"))

            embedded_indices.append(frame_idx)
        else:
            out.write(bgr)

        if progress_callback:
            progress_callback(frame_idx, total)
        frame_idx += 1

    cap.release()
    out.release()

    # FFV1 path: cv2.VideoWriter wrote a video-only temp file.  Mux the
    # audio stream from the original input into the final output via ffmpeg.
    # ffmpeg_exe is guaranteed non-None (checked at the top of this function).
    if ffv1_temp_path is not None:
        mux_cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-i", ffv1_temp_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(mux_cmd, capture_output=True)
        if result.returncode == 0:
            os.remove(ffv1_temp_path)
        else:
            err = result.stderr.decode("utf-8", errors="replace")
            print(f"[embed_video] WARN: audio mux failed (exit {result.returncode}):\n{err}")
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(ffv1_temp_path, output_path)

    # Aggregate PSNR stats (over embedded frames only; non-embedded frames
    # are written unchanged so they contribute infinite PSNR and aren't
    # interesting to report).  Ignore inf when computing min/max so a single
    # zero-MSE frame doesn't hide the worst-case perturbation.
    def _finite(xs): return [x for x in xs if np.isfinite(x)]
    psnr_y_finite   = _finite(psnr_y_per_frame)
    psnr_rgb_finite = _finite(psnr_rgb_per_frame)
    psnr_y_mean   = float(np.mean(psnr_y_finite))   if psnr_y_finite   else float("inf")
    psnr_y_min    = float(np.min(psnr_y_finite))    if psnr_y_finite   else float("inf")
    psnr_y_max    = float(np.max(psnr_y_finite))    if psnr_y_finite   else float("inf")
    psnr_rgb_mean = float(np.mean(psnr_rgb_finite)) if psnr_rgb_finite else float("inf")
    psnr_rgb_min  = float(np.min(psnr_rgb_finite))  if psnr_rgb_finite else float("inf")
    psnr_rgb_max  = float(np.max(psnr_rgb_finite))  if psnr_rgb_finite else float("inf")

    metadata = {
        "version":               "v4",
        "owner_id":              owner_id,
        "media_id":              media_id,
        "total_frames":          frame_idx,
        "embedded_frames":       len(embedded_indices),
        "embed_every_n_frames":  embed_every_n_frames,
        "fps":                   fps,
        "resolution":            f"{width}x{height}",
        "n_meta_bits_raw":       n_meta_bits_raw,
        "n_meta_bits_coded":     n_meta_bits_coded,
        "frame_repeat_bits":     int(n_frame_repeat_bits or 0),
        "frame_repeat_copies":   int(n_frame_repeat_copies or 0),
        "owner_repeat_bits":     int(n_owner_repeat_bits or 0),
        "owner_repeat_copies":   int(n_owner_repeat_copies or 0),
        "media_repeat_bits":     int(n_media_repeat_bits or 0),
        "media_repeat_copies":   int(n_media_repeat_copies or 0),
        "chain_repeat_bits":     int(n_chain_repeat_bits or 0),
        "chain_repeat_copies":   int(n_chain_repeat_copies or 0),
        "owner_id_bits":         int(OWNER_ID_BITS),
        "media_id_bits":         int(MEDIA_ID_BITS),
        "frame_id_bits":         int(FRAME_ID_BITS),
        "chain_tag_bits":        int(CHAIN_TAG_BITS),
        # Back-compat alias (kept for sidecars produced before the rename).
        "owner_hash_bits":       int(OWNER_ID_BITS),
        "ber_threshold":         BER_TAMPER_THRESHOLD,
        # Pre-encoding PSNR — perturbation quality of the watermark itself
        # (Y-channel and RGB).  Codec degradation that follows is separate.
        "psnr_y_mean_db":        psnr_y_mean,
        "psnr_y_min_db":         psnr_y_min,
        "psnr_y_max_db":         psnr_y_max,
        "psnr_rgb_mean_db":      psnr_rgb_mean,
        "psnr_rgb_min_db":       psnr_rgb_min,
        "psnr_rgb_max_db":       psnr_rgb_max,
    }
    # Fail fast if anything in the metadata is not JSON-serialisable —
    # the verifier round-trips this through a .json sidecar file.
    json.dumps(metadata)
    return metadata


# ──────────────────────────────────────────────────────────────
# VERIFY VIDEO
# ──────────────────────────────────────────────────────────────

def verify_video(
    video_path: str,
    metadata: Dict,
    sample_frames: Optional[int] = 30,
    progress_callback=None,
) -> Dict:
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    owner_id            = metadata["owner_id"]
    media_id            = metadata["media_id"]
    embed_every_n       = metadata.get("embed_every_n_frames", 1)
    n_meta_bits_raw     = metadata["n_meta_bits_raw"]
    n_meta_bits_coded   = metadata["n_meta_bits_coded"]
    n_owner_repeat      = int(metadata.get("owner_repeat_bits", 0))
    n_media_repeat      = int(metadata.get("media_repeat_bits", 0))
    n_frame_repeat      = int(metadata.get("frame_repeat_bits", 0))
    n_chain_repeat      = int(metadata.get("chain_repeat_bits", 0))
    n_extra_total       = n_owner_repeat + n_media_repeat + n_frame_repeat + n_chain_repeat
    # Round-trip the IDs through encode→decode so comparisons match what the
    # decoder will actually see after the fixed-length truncation/null-pad.
    expected_owner_id, expected_media_id = _meta_owner_media_id(owner_id, media_id)

    # Use the embed-time frame count when available; container reports
    # are unreliable on some codecs.
    total = int(metadata.get("total_frames") or cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample uniformly across the *embedded* frame indices, not just the
    # first N — sequential sampling would let an attacker tamper after
    # frame N undetected.
    embedded_indices = list(range(0, total, embed_every_n))
    if sample_frames and sample_frames < len(embedded_indices):
        pick = np.linspace(0, len(embedded_indices) - 1, sample_frames, dtype=int)
        indices_to_verify = {int(embedded_indices[i]) for i in pick}
    else:
        indices_to_verify = set(embedded_indices)

    per_frame: List[Dict] = []
    spatial_maps: List[np.ndarray] = []
    temporal_map: List[bool] = []

    # Pre-build the expected chain state for each embedded iteration.  The
    # chain depends on the *sequence* of embedded frame_ids, so we evolve it
    # alongside the frame walk and snapshot the per-iteration value into a
    # lookup so sampled (non-sequential) frames can fetch their expected
    # chain_tag in O(1).
    media_hash_bytes = hashlib.sha256(media_id.encode()).digest()[:4]
    _chain_running   = _chain_genesis(media_hash_bytes)
    expected_chain_tags: Dict[int, str] = {}
    for emb_idx in embedded_indices:
        _chain_running = _frame_chain_hash(media_hash_bytes, _chain_running, emb_idx)
        expected_chain_tags[emb_idx] = _chain_running[:2].hex()

    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % embed_every_n == 0:
            if frame_idx in indices_to_verify:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                Y, Cb, _ = to_float_ycbcr(rgb)
                signature = _master_signature(owner_id, media_id, frame_idx)

                res = _verify_channel(
                    Y, n_meta_bits_coded, signature, frame_idx,
                    n_extra_llll_bits=n_extra_total,
                )

                # Spatial map: chroma LSB block-mean parity (compression-robust).
                lsb_res          = _verify_lsb_chroma(Cb, signature, frame_idx)
                combined_spatial = lsb_res["lsb_spatial_map"]

                # Layout matches embed_video: frame_id → owner_id → media_id → chain_tag.
                extra_bits       = res["extra_llll_bits"]
                p = 0
                frame_repeat_ext = extra_bits[p:p + n_frame_repeat]; p += n_frame_repeat
                owner_repeat_ext = extra_bits[p:p + n_owner_repeat]; p += n_owner_repeat
                media_repeat_ext = extra_bits[p:p + n_media_repeat]; p += n_media_repeat
                chain_repeat_ext = extra_bits[p:p + n_chain_repeat]

                meta_bits, n_corr = bch_decode_bits(res["meta_bits_coded"])
                meta_bytes = array_to_bytes(meta_bits[:n_meta_bits_raw])
                parsed     = _parse_meta_payload(meta_bytes)


                owner_match = bool(parsed and parsed.get("owner_id") == expected_owner_id)
                media_match = bool(parsed and parsed.get("media_id") == expected_media_id)
                frame_match = bool(parsed and parsed.get("frame") == frame_idx)
                # Chain tag check: catches reorder / replay attacks where
                # frame_id matches its iteration index but the frame actually
                # came from a different position in the original clip.
                expected_chain_tag_hex = expected_chain_tags.get(frame_idx)
                chain_match  = bool(
                    parsed
                    and expected_chain_tag_hex is not None
                    and parsed.get("chain_tag") == expected_chain_tag_hex
                )
                owner_id_out    = parsed.get("owner_id") if parsed else None
                media_id_out    = parsed.get("media_id") if parsed else None
                recovery_method = "bch" if parsed is not None else "none"

                # Majority-vote fallback for owner_id (raw UTF-8 string).
                # Recovers the actual identifier under compression — not a hash.
                if (not owner_match
                        and n_owner_repeat > 0
                        and len(owner_repeat_ext) >= OWNER_ID_BITS * 2):
                    voted = _majority_vote(owner_repeat_ext, OWNER_ID_BITS)
                    if voted is not None:
                        voted_str = _decode_id_fixed(array_to_bytes(voted)[:OWNER_ID_BYTES])
                        if voted_str == expected_owner_id:
                            owner_match     = True
                            owner_id_out    = voted_str
                            recovery_method = "majority_vote"
                        elif owner_id_out is None:
                            owner_id_out = voted_str

                # Majority-vote fallback for media_id (raw UTF-8 string).
                if (not media_match
                        and n_media_repeat > 0
                        and len(media_repeat_ext) >= MEDIA_ID_BITS * 2):
                    voted = _majority_vote(media_repeat_ext, MEDIA_ID_BITS)
                    if voted is not None:
                        voted_str = _decode_id_fixed(array_to_bytes(voted)[:MEDIA_ID_BYTES])
                        if voted_str == expected_media_id:
                            media_match     = True
                            media_id_out    = voted_str
                            recovery_method = "majority_vote"
                        elif media_id_out is None:
                            media_id_out = voted_str

                # Majority-vote fallback for frame_id (temporal integrity).
                # Expected frame_id at this iteration is just frame_idx.
                expected_frame_id_hex = struct.pack(">I", frame_idx).hex()
                frame_recovery_local  = "bch" if frame_match else "none"
                if (not frame_match
                        and n_frame_repeat > 0
                        and len(frame_repeat_ext) >= FRAME_ID_BITS * 2):
                    voted = _majority_vote(frame_repeat_ext, FRAME_ID_BITS)
                    if voted is not None:
                        voted_hex = array_to_bytes(voted).hex()
                        if voted_hex == expected_frame_id_hex:
                            frame_match          = True
                            frame_recovery_local = "majority_vote"
                            recovery_method      = "majority_vote"

                # Majority-vote fallback for chain_tag.
                chain_recovery_local = "bch" if chain_match else "none"
                if (not chain_match
                        and n_chain_repeat > 0
                        and expected_chain_tag_hex is not None
                        and len(chain_repeat_ext) >= CHAIN_TAG_BITS * 2):
                    voted = _majority_vote(chain_repeat_ext, CHAIN_TAG_BITS)
                    if voted is not None:
                        voted_hex = array_to_bytes(voted).hex()
                        if voted_hex == expected_chain_tag_hex:
                            chain_match          = True
                            chain_recovery_local = "majority_vote"
                            recovery_method      = "majority_vote"

                # Temporal integrity: TWO independent checks.
                #   (a) frame_id mismatch — embedded frame_id != iteration
                #       index.  Catches deletion / insertion / unwatermarked-
                #       splice.  Flagged via `frame_match`.
                #   (b) chain_tag mismatch — embedded chain_tag != expected
                #       chain_tag for this iteration.  Catches reorder /
                #       replay attacks where two frames from the SAME media
                #       were swapped (both have valid frame_ids individually
                #       but the chain depends on the original sequence).
                #       Flagged via `chain_match`.
                # Only counted when BCH actually parsed (parsed is not None);
                # codec noise that fails BCH is handled by majority-vote on
                # the owner/media copies, not by these signals.
                chain_ok = frame_match and (chain_match or parsed is None)

                explicit_frame_mismatch = bool(
                    parsed is not None and parsed.get("frame") != frame_idx
                )
                explicit_chain_mismatch = bool(parsed is not None and not chain_match)
                explicit_owner_mismatch = bool(parsed is not None and not owner_match)
                explicit_media_mismatch = bool(parsed is not None and not media_match)

                # Content tamper: enough blocks flagged in the LSB-only spatial
                # map to clear the noise floor (see MIN_TAMPER_BLOCKS).  The
                # heatmap still shows every flagged block individually — this
                # only gates the per-frame verdict.
                content_tampered = bool(int(combined_spatial.sum()) >= MIN_TAMPER_BLOCKS)
                frame_tampered   = (
                    content_tampered
                    or not (owner_match and media_match)
                    or not chain_ok
                )

                spatial_maps.append(combined_spatial)

                temporal_map.append(frame_tampered)

                per_frame.append({
                    "frame_idx":               frame_idx,
                    # Recovered RAW IDs (truncated UTF-8) — present even on
                    # mismatch so the caller sees what came out of the watermark.
                    "owner_id_recovered":      owner_id_out,
                    "media_id_recovered":      media_id_out,
                    # Back-compat keys (older callers read the "_hash" names).
                    "owner_hash":              owner_id_out,
                    "media_hash":              media_id_out,
                    "reported_frame":          parsed.get("frame") if parsed else None,
                    "reported_chain_tag":      parsed.get("chain_tag") if parsed else None,
                    "expected_chain_tag":      expected_chain_tag_hex,
                    "owner_match":             owner_match,
                    "media_match":             media_match,
                    "frame_match":             frame_match,
                    "chain_match":             chain_match,
                    "chain_ok":                chain_ok,
                    "explicit_frame_mismatch": explicit_frame_mismatch,
                    "explicit_chain_mismatch": explicit_chain_mismatch,
                    "explicit_owner_mismatch": explicit_owner_mismatch,
                    "explicit_media_mismatch": explicit_media_mismatch,
                    "recovery_method":         recovery_method,
                    "frame_recovery":          frame_recovery_local,
                    "chain_recovery":          chain_recovery_local,
                    "ber":                     float(combined_spatial.mean()) if combined_spatial.size else 0.0,
                    "ber_lsb_sub":             lsb_res["ber_lsb_sub"],
                    "ber_lsb_block":           lsb_res["ber_lsb_block"],
                    "bch_corrections":         int(n_corr),
                    "content_tampered":        bool(content_tampered),
                    "frame_tampered":          bool(frame_tampered),
                                        "n_blocks_tampered":       int(combined_spatial.sum()),
                    "n_blocks_total":          int(combined_spatial.size),

                })

        if progress_callback:
            progress_callback(frame_idx, total)
        frame_idx += 1

    cap.release()

    # End-truncation check.  Compare the number of frames actually read
    # against `metadata["total_frames"]` recorded at embed time.  Deleting
    # the last K frames is invisible to the per-frame chain/frame_id checks
    # (the remaining frames still chain validly), so we have to compare
    # totals to catch it.
    observed_total_frames = frame_idx
    expected_total_frames = int(metadata.get("total_frames") or 0)
    frames_truncated      = max(0, expected_total_frames - observed_total_frames)
    # Tolerate up to 2 frames of difference: audio muxing can shift MKV
    # container metadata so cv2 reads slightly fewer frames than were
    # written.  Real truncation attacks remove many more frames.
    truncated             = bool(expected_total_frames > 0 and frames_truncated > 2)

    spatial_arr  = np.stack(spatial_maps, axis=0) if spatial_maps else np.empty((0, 0, 0), bool)
    temporal_arr = np.array(temporal_map, dtype=bool)

    n_tampered_frames   = int(temporal_arr.sum())
    n_chain_breaks      = sum(1 for r in per_frame if not r["chain_ok"])
    n_content_tampered  = sum(1 for r in per_frame if r["content_tampered"])
    n_id_mismatch       = sum(1 for r in per_frame if not (r["owner_match"] and r["media_match"]))
    n_owner_match       = sum(1 for r in per_frame if r["owner_match"])
    n_media_match       = sum(1 for r in per_frame if r["media_match"])
    n_explicit_frame_mismatch = sum(1 for r in per_frame if r["explicit_frame_mismatch"])
    n_explicit_chain_mismatch = sum(1 for r in per_frame if r["explicit_chain_mismatch"])
    n_explicit_owner_mismatch = sum(1 for r in per_frame if r["explicit_owner_mismatch"])
    n_explicit_media_mismatch = sum(1 for r in per_frame if r["explicit_media_mismatch"])
    avg_ber             = float(np.mean([r["ber"] for r in per_frame])) if per_frame else 0.0

    majority = max(len(per_frame) / 2, 1)
    owner_out = owner_id if n_owner_match > majority else None
    media_out = media_id if n_media_match > majority else None

    frame_tamper_rate = n_tampered_frames / max(len(per_frame), 1)

    # Video tamper rule (user spec): "if the block is tampered or there is
    # frame changed, then it is tampered".  Any frame with content_tampered
    # (block flagged) or chain_break (frame_id mismatch) or identity_mismatch
    # marks the whole video as tampered.
    explicit_tamper = bool(
        n_explicit_frame_mismatch
        or n_explicit_chain_mismatch
        or n_explicit_owner_mismatch
        or n_explicit_media_mismatch
    )
    tampered = explicit_tamper or n_tampered_frames > 0 or truncated

    return {
        "video_path":                  video_path,
        "frames_checked":              len(per_frame),
        "owner":                       owner_out,
        "media":                       media_out,
        "owner_match_frames":          n_owner_match,
        "media_match_frames":          n_media_match,
        "frames_tampered":             n_tampered_frames,
        "frames_chain_break":          n_chain_breaks,
        "frames_content_tampered":     n_content_tampered,
        "frames_id_mismatch":          n_id_mismatch,
        "frames_explicit_frame_mismatch": n_explicit_frame_mismatch,
        "frames_explicit_chain_mismatch": n_explicit_chain_mismatch,
        "frames_explicit_owner_mismatch": n_explicit_owner_mismatch,
        "frames_explicit_media_mismatch": n_explicit_media_mismatch,
        "observed_total_frames":       observed_total_frames,
        "expected_total_frames":       expected_total_frames,
        "frames_truncated":            frames_truncated,
        "truncated":                   truncated,
        "average_ber":                 avg_ber,
        "ber_threshold":               BER_TAMPER_THRESHOLD,
        "frame_tamper_rate":           frame_tamper_rate,
        "TAMPERED":                    bool(tampered),
        "spatial_tamper_map":          spatial_arr,    # F x Hb x Wb bool
        "temporal_tamper_map":         temporal_arr,   # F bool
        "per_frame":                   per_frame,
    }