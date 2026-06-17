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
    _compute_lsb_target_pattern, _wm_display_patterns,
    _llfamily_block_capacity, _plan_repeat_copies,

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

    # Fair repeat allocation, computed once for the clip (capacity is constant
    # per resolution).  owner_id and media_id get EQUAL copies so media is no
    # longer starved.  frame_id keeps its full repeats here (temporal integrity
    # / deletion localization needs to recover frame_id under compression).
    # Layout order: frame_id → owner_id → media_id → chain_tag.
    _meta_len_bits = len(bch_encode_bits(
        bits_to_array(_build_meta_payload(owner_id, media_id, 0, b"\x00\x00"))))
    capacity_bits  = _llfamily_block_capacity((height, width))
    n_frame_copies, n_owner_copies, n_media_copies, n_chain_copies = \
        _plan_repeat_copies(capacity_bits - _meta_len_bits, frame_repeats=FRAME_REPEATS)
    if n_owner_copies < OWNER_REPEATS or n_media_copies < MEDIA_REPEATS:
        print(f"[embed_video] note: fit {n_frame_copies} frame_id + "
              f"{n_owner_copies}/{OWNER_REPEATS} owner_id + "
              f"{n_media_copies}/{MEDIA_REPEATS} media_id + "
              f"{n_chain_copies}/{CHAIN_REPEATS} chain_tag copies/frame "
              f"(LL-family capacity {capacity_bits} bits).")

    # Owner + media repeats are constant across the clip — build once.
    owner_repeated = np.tile(_owner_id_bits_array(owner_id), n_owner_copies)
    media_repeated = np.tile(_media_id_bits_array(media_id), n_media_copies)
    constant_extras = np.concatenate([owner_repeated, media_repeated]).astype(np.uint8)

    # Bit/copy tallies recorded in the metadata (verify slices the extra stream
    # by these counts, in frame → owner → media → chain order).
    n_frame_repeat_copies = n_frame_copies
    n_owner_repeat_copies = n_owner_copies
    n_media_repeat_copies = n_media_copies
    n_chain_repeat_copies = n_chain_copies
    n_frame_repeat_bits   = n_frame_copies * FRAME_ID_BITS
    n_owner_repeat_bits   = n_owner_copies * OWNER_ID_BITS
    n_media_repeat_bits   = n_media_copies * MEDIA_ID_BITS
    n_chain_repeat_bits   = n_chain_copies * CHAIN_TAG_BITS

    # Per-frame chain state: chain_i = SHA256(media_hash || chain_{i-1} || frame_id_i).
    # First 2 bytes embedded as chain_tag in LLLL.  Catches reorder / replay
    # attacks that frame_id alone misses.
    media_hash_bytes = hashlib.sha256(media_id.encode()).digest()[:4]
    chain_state      = _chain_genesis(media_hash_bytes)

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

            # Per-frame extras: frame_id LEADS the stream (highest priority,
            # lives in LLLL), then the constant owner+media block, then
            # chain_tag.  Copy counts come from the fair plan above; order MUST
            # match the verify-side slicing (frame → owner → media → chain).
            frame_id_repeated  = np.tile(_frame_id_bits_array(frame_idx),  n_frame_copies)
            chain_tag_repeated = np.tile(_chain_tag_bits_array(chain_tag), n_chain_copies)
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
    # The full set of frame_ids that were embedded at encode time.  Temporal
    # anomalies (deletion / reorder / replay) are derived by comparing the
    # SEQUENCE of recovered frame_ids against this set after the walk — never
    # by comparing a frame's recovered id to its playback index, which would
    # cascade (deleting one frame shifts every later index by one).
    expected_fids_set = set(embedded_indices)

    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % embed_every_n == 0:
            if frame_idx in indices_to_verify:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                Y, Cb, _ = to_float_ycbcr(rgb)

                # _verify_channel ignores signature/frame_id for bit extraction,
                # so we can decode the stream BEFORE knowing which frame this
                # actually is.  The fragile LSB layer IS keyed by the frame id,
                # so it's verified further down — once rec_fid is recovered.
                res = _verify_channel(
                    Y, n_meta_bits_coded, b"", frame_idx,
                    n_extra_llll_bits=n_extra_total,
                )

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

                # ── Recover this frame's embedded frame_id (source of truth
                # for ordering) ──────────────────────────────────────────────
                # BCH first, then majority-vote on the 5× frame_id repeats.
                # rec_fid is what the frame CLAIMS to be; temporal anomalies
                # are derived from the SEQUENCE of rec_fids after the walk —
                # never by comparing rec_fid to the playback index.
                rec_fid = parsed.get("frame") if parsed else None
                frame_recovery_local = "bch" if rec_fid is not None else "none"
                if (rec_fid is None
                        and n_frame_repeat > 0
                        and len(frame_repeat_ext) >= FRAME_ID_BITS * 2):
                    voted = _majority_vote(frame_repeat_ext, FRAME_ID_BITS)
                    if voted is not None:
                        try:
                            rec_fid = struct.unpack(">I", array_to_bytes(voted)[:4])[0]
                            frame_recovery_local = "majority_vote"
                        except Exception:
                            rec_fid = None

                # ── Fragile LSB layer, keyed by the frame's OWN id ──
                # The chroma LSB pattern derives from signature(owner|media|
                # frame_id).  Verifying with the RECOVERED id (not the playback
                # index) means a deletion that shifts later frames' positions no
                # longer desyncs the pattern → those survivors stay clean.  Fall
                # back to the playback index when the id couldn't be recovered.
                sig_fid          = rec_fid if rec_fid is not None else frame_idx
                signature        = _master_signature(owner_id, media_id, sig_fid)
                lsb_res          = _verify_lsb_chroma(Cb, signature, sig_fid)
                combined_spatial = lsb_res["lsb_spatial_map"]

                # Chain tag is validated against the expected chain for THIS
                # frame's OWN recovered frame_id — so a survivor keeps matching
                # even when other frames around it were deleted (the chain is
                # keyed by frame_id, not by playback position).
                expected_chain_tag_hex = (
                    expected_chain_tags.get(rec_fid) if rec_fid is not None else None)
                chain_match = bool(
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

                # Majority-vote fallback for chain_tag, keyed by rec_fid.
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

                # frame_id is "valid" when recovered AND it is one of the
                # originally-embedded ids.  Whether it sits at the right
                # POSITION is a sequence-level question handled after the loop.
                frame_id_valid = bool(rec_fid is not None and rec_fid in expected_fids_set)

                # ── Per-frame integrity (NO position/index comparison here) ──
                # A frame is tampered if its content blocks are flagged, its
                # identity is broken, or its chain_tag is inconsistent with its
                # OWN frame_id (a forged / content-altered frame).  Deleting
                # OTHER frames leaves all of these clean for the survivors, so
                # there is no cascade.
                content_tampered = bool(int(combined_spatial.sum()) >= MIN_TAMPER_BLOCKS)
                self_chain_bad   = bool(parsed is not None and frame_id_valid and not chain_match)
                id_mismatch      = not (owner_match and media_match)
                frame_tampered   = bool(content_tampered or id_mismatch or self_chain_bad)

                # Noise-suppressed watermark visualisation: red only on real
                # content tamper (clean frames show NO red at all).
                wm_ext_disp, wm_mask_disp = _wm_display_patterns(lsb_res, content_tampered)

                spatial_maps.append(combined_spatial)

                temporal_map.append(frame_tampered)

                per_frame.append({
                    "frame_idx":          frame_idx,     # playback position in this file
                    "rec_fid":            rec_fid,       # recovered embedded frame_id
                    # Recovered RAW IDs (truncated UTF-8) — present even on
                    # mismatch so the caller sees what came out of the watermark.
                    "owner_id_recovered": owner_id_out,
                    "media_id_recovered": media_id_out,
                    # Back-compat keys (older callers read the "_hash" names).
                    "owner_hash":         owner_id_out,
                    "media_hash":         media_id_out,
                    "reported_frame":     rec_fid,
                    "reported_chain_tag": parsed.get("chain_tag") if parsed else None,
                    "expected_chain_tag": expected_chain_tag_hex,
                    "owner_match":        owner_match,
                    "media_match":        media_match,
                    "chain_match":        chain_match,
                    "frame_id_valid":     frame_id_valid,
                    "recovery_method":    recovery_method,
                    "frame_recovery":     frame_recovery_local,
                    "chain_recovery":     chain_recovery_local,
                    "ber":                float(combined_spatial.mean()) if combined_spatial.size else 0.0,
                    "ber_lsb_sub":        lsb_res["ber_lsb_sub"],
                    "ber_lsb_block":      lsb_res["ber_lsb_block"],
                    "bch_corrections":    int(n_corr),
                    "content_tampered":   bool(content_tampered),
                    "self_chain_bad":     bool(self_chain_bad),
                    "id_mismatch":        bool(id_mismatch),
                    "frame_tampered":     bool(frame_tampered),
                    "n_blocks_tampered":  int(combined_spatial.sum()),
                    "n_blocks_total":     int(combined_spatial.size),
                    # Fragile-watermark grids for every verified frame so the
                    # frontend can show the original-vs-extracted comparison on
                    # whichever frame is selected (rendered to PNG in main.py).
                    # The extracted grid + mask are noise-suppressed: clean
                    # frames show NO red, only real tamper does.
                    "wm_target_pattern":    lsb_res["lsb_target_pattern"],
                    "wm_extracted_pattern": wm_ext_disp,
                    "wm_display_mismatch":  wm_mask_disp,
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
    n_self_chain_bad    = sum(1 for r in per_frame if r["self_chain_bad"])
    n_content_tampered  = sum(1 for r in per_frame if r["content_tampered"])
    n_id_mismatch       = sum(1 for r in per_frame if r["id_mismatch"])
    n_owner_match       = sum(1 for r in per_frame if r["owner_match"])
    n_media_match       = sum(1 for r in per_frame if r["media_match"])
    avg_ber             = float(np.mean([r["ber"] for r in per_frame])) if per_frame else 0.0

    # ── Temporal sequence analysis (deletion / reorder / replay) ──────────
    # Localize by the SEQUENCE of recovered frame_ids rather than per-frame
    # index comparison — so deleting one frame surfaces as ONE missing id,
    # not a cascade across every later frame.
    playback_fids = [r["rec_fid"] for r in per_frame]
    known_fids    = [f for f in playback_fids if f is not None]
    recovered_set = set(known_fids)
    # Frame_ids that were embedded but never showed up → deleted (or, under
    # heavy compression, unrecoverable).  This is the localized deletion list.
    missing_frame_ids = [int(f) for f in embedded_indices if f not in recovered_set]

    # Expected fragile-watermark pattern for each deleted frame_id, so the
    # frontend can still show its "original" watermark next to a "frame
    # deleted — nothing to extract" placeholder.  Deterministic from the
    # signature; the grid shape matches the verified frames' patterns.
    pattern_shape = None
    for r in per_frame:
        tp = r.get("wm_target_pattern")
        if tp is not None and hasattr(tp, "shape"):
            pattern_shape = tp.shape
            break
    missing_frame_patterns: List[Dict] = []
    if pattern_shape is not None:
        n_sub_h, n_sub_w = pattern_shape
        for fid in missing_frame_ids:
            sig = _master_signature(owner_id, media_id, fid)
            target = _compute_lsb_target_pattern(n_sub_h, n_sub_w, sig, fid)
            missing_frame_patterns.append({
                "frame_id":       fid,
                "target_pattern": target.astype(np.uint8),
            })
    # Reorder / replay: recovered ids must strictly increase in playback order.
    reorder_at: List[int] = []
    duplicate_frame_ids   = sorted(int(f) for f in recovered_set if known_fids.count(f) > 1)
    _prev = None
    for r in per_frame:
        f = r["rec_fid"]
        if f is None:
            continue
        if _prev is not None and f <= _prev:
            reorder_at.append(int(r["frame_idx"]))
        _prev = f
    reordered = bool(reorder_at)

    majority = max(len(per_frame) / 2, 1)
    owner_out = owner_id if n_owner_match > majority else None
    media_out = media_id if n_media_match > majority else None

    frame_tamper_rate = n_tampered_frames / max(len(per_frame), 1)

    # Video tamper rule: a frame's own content/identity/chain broke, OR the
    # temporal sequence was altered (deletion / reorder / replay / truncation).
    # Crucially, deleting one frame flags only the missing id — the surviving
    # frames stay authentic.
    content_or_id_tamper = bool(n_content_tampered or n_id_mismatch or n_self_chain_bad)
    temporal_tamper      = bool(missing_frame_ids or reordered or duplicate_frame_ids or truncated)
    tampered             = content_or_id_tamper or temporal_tamper

    return {
        "video_path":                  video_path,
        "frames_checked":              len(per_frame),
        "owner":                       owner_out,
        "media":                       media_out,
        "owner_match_frames":          n_owner_match,
        "media_match_frames":          n_media_match,
        "frames_tampered":             n_tampered_frames,
        # Per-frame chain breaks now mean "this frame's own chain is bad", not
        # the old index-shift cascade.  Kept under the same response key.
        "frames_chain_break":          n_self_chain_bad,
        "frames_content_tampered":     n_content_tampered,
        "frames_id_mismatch":          n_id_mismatch,
        # Temporal localization (the headline fix): which frame_ids went
        # missing / got reordered / duplicated, derived from the id sequence.
        "missing_frame_ids":           missing_frame_ids,
        "missing_frame_patterns":      missing_frame_patterns,
        "frames_deleted":              len(missing_frame_ids),
        "reordered":                   reordered,
        "reorder_at":                  reorder_at,
        "duplicate_frame_ids":         duplicate_frame_ids,
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