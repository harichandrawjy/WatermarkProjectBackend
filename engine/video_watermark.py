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
import json
import os
from typing import Optional, Dict, List

from watermark_engine import (
    bch_encode_bits, bch_decode_bits,
    bits_to_array, array_to_bytes,
    to_float_ycbcr, from_float_ycbcr,
    _embed_into_channel, _verify_channel,
        _embed_lsb_chroma, _verify_lsb_chroma,

    _build_meta_payload, _parse_meta_payload,
    _master_signature, _meta_owner_media_hash,
    _owner_hash_bits_array, _media_hash_bits_array, _majority_vote,
        
    OWNER_HASH_BITS, OWNER_HASH_REPEATS,

    BER_TAMPER_THRESHOLD,
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
    if ext in ('.mkv', '.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')   # lossless — watermark survives
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # lossy — semi-fragile WM will read as tampered
        print(f"[embed] WARNING: '{ext}' uses lossy compression; use .mkv for lossless output "
              "if you want the watermark to verify cleanly.")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot open VideoWriter for {output_path} (codec '{fourcc}' unavailable)")

    embedded_indices: List[int] = []
    fp_grid_shape = None
    n_meta_bits_raw = None
    n_meta_bits_coded = None
    n_owner_repeat_bits = None
    n_owner_repeat_copies = None
    n_media_repeat_bits = None
    n_media_repeat_copies = None

    # Same owner+media across the whole clip → repeated patterns identical
    # for every frame; build once.  Owner first (priority field).
    owner_repeated = np.tile(_owner_hash_bits_array(owner_id), OWNER_HASH_REPEATS)
    media_repeated = np.tile(_media_hash_bits_array(media_id), OWNER_HASH_REPEATS)
    extras         = np.concatenate([owner_repeated, media_repeated])

    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % embed_every_n_frames == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            signature  = _master_signature(owner_id, media_id, frame_idx)

            meta_raw   = _build_meta_payload(owner_id, media_id, frame_idx)
            meta_bits  = bits_to_array(meta_raw)
            meta_coded = bch_encode_bits(meta_bits)

            if n_meta_bits_raw is None:
                n_meta_bits_raw   = len(meta_bits)
                n_meta_bits_coded = len(meta_coded)

            Y, Cb, Cr = to_float_ycbcr(rgb)
            Y_wm, info = _embed_into_channel(Y, meta_coded, signature, frame_idx, extras)

                        # LSB layer on chroma — independent of Y's DCT/SVD watermark.
            Cb_wm = _embed_lsb_chroma(Cb, signature, frame_idx)


            if info["n_meta_bits"] < len(meta_coded):
                # Mirror embed_image: a frame too small to hold the BCH meta
                # cannot carry the watermark.  Silently writing the raw frame
                # would de-sync the verifier (it would read no watermark and
                # flag the frame as tampered, with no way to know it was
                # an embed-time skip).  Fail loudly instead.
                cap.release()
                out.release()
                raise ValueError(
                    f"Frame {frame_idx}: LLLL too small "
                    f"({info['n_meta_bits']}/{len(meta_coded)} BCH-coded meta bits). "
                    f"Use a larger video or shorter owner/media IDs."
                )

            if n_owner_repeat_copies is None:
                # First successful frame defines how many full owner+media
                # copies fit in this video's LLLL; verify uses the same counts.
                n_extra_emb = info["n_extra_llll_bits"]
                n_owner_repeat_copies = min(OWNER_HASH_REPEATS, n_extra_emb // OWNER_HASH_BITS)
                remaining             = max(0, n_extra_emb - n_owner_repeat_copies * OWNER_HASH_BITS)
                n_media_repeat_copies = min(OWNER_HASH_REPEATS, remaining // OWNER_HASH_BITS)
                n_owner_repeat_bits   = n_owner_repeat_copies * OWNER_HASH_BITS
                n_media_repeat_bits   = n_media_repeat_copies * OWNER_HASH_BITS
                if n_owner_repeat_copies < OWNER_HASH_REPEATS or n_media_repeat_copies < OWNER_HASH_REPEATS:
                    print(f"[embed_video] note: fit {n_owner_repeat_copies}/{OWNER_HASH_REPEATS} owner + "
                          f"{n_media_repeat_copies}/{OWNER_HASH_REPEATS} media hash copies per frame "
                          f"— frame too small for full repetition margin.")

            if fp_grid_shape is None:
                fp_grid_shape = (info["fp_grid_h"], info["fp_grid_w"])

            rgb_wm = from_float_ycbcr(Y_wm, Cb_wm, Cr)
            bgr_wm = cv2.cvtColor(rgb_wm, cv2.COLOR_RGB2BGR)
            out.write(bgr_wm)

            embedded_indices.append(frame_idx)
        else:
            out.write(bgr)

        if progress_callback:
            progress_callback(frame_idx, total)
        frame_idx += 1

    cap.release()
    out.release()

    metadata = {
        "version":               "v3",
        "owner_id":              owner_id,
        "media_id":              media_id,
        "total_frames":          frame_idx,
        "embedded_frames":       len(embedded_indices),
        "embed_every_n_frames":  embed_every_n_frames,
        "fps":                   fps,
        "resolution":            f"{width}x{height}",
        "n_meta_bits_raw":       n_meta_bits_raw,
        "n_meta_bits_coded":     n_meta_bits_coded,
        "owner_repeat_bits":     int(n_owner_repeat_bits or 0),
        "owner_repeat_copies":   int(n_owner_repeat_copies or 0),
        "media_repeat_bits":     int(n_media_repeat_bits or 0),
        "media_repeat_copies":   int(n_media_repeat_copies or 0),
        "owner_hash_bits":       int(OWNER_HASH_BITS),
        # Tuple → list so json.dump round-trips exactly as we read it back.
        "fp_grid_shape":         list(fp_grid_shape) if fp_grid_shape else None,
        "ber_threshold":         BER_TAMPER_THRESHOLD,
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
    expected_owner_h_hex, expected_media_h_hex = _meta_owner_media_hash(owner_id, media_id)
    expected_owner_h_bytes = bytes.fromhex(expected_owner_h_hex)
    expected_media_h_bytes = bytes.fromhex(expected_media_h_hex)

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
                    n_extra_llll_bits=n_owner_repeat + n_media_repeat,
                )

                                # AND the DCT/SVD spatial map with the chroma-LSB map to
                # suppress compression false positives in the localisation.
                lsb_res = _verify_lsb_chroma(Cb, signature, frame_idx,
                                             target_shape=res["spatial_map"].shape)
                dct_map = res["spatial_map"]
                lsb_map = lsb_res["lsb_spatial_map"]
                h_min = min(dct_map.shape[0], lsb_map.shape[0])
                w_min = min(dct_map.shape[1], lsb_map.shape[1])
                combined_spatial = dct_map[:h_min, :w_min] & lsb_map[:h_min, :w_min]

                extra_bits       = res["extra_llll_bits"]
                owner_repeat_ext = extra_bits[:n_owner_repeat]
                media_repeat_ext = extra_bits[n_owner_repeat:n_owner_repeat + n_media_repeat]

                meta_bits, n_corr = bch_decode_bits(res["meta_bits_coded"])
                meta_bytes = array_to_bytes(meta_bits[:n_meta_bits_raw])
                parsed     = _parse_meta_payload(
                    meta_bytes, expected_owner_h_bytes, expected_media_h_bytes)


                owner_match = bool(parsed and parsed.get("owner_hash") == expected_owner_h_hex)
                media_match = bool(parsed and parsed.get("media_hash") == expected_media_h_hex)
                frame_match = bool(parsed and parsed.get("frame") == frame_idx)
                owner_hash_out  = parsed.get("owner_hash") if parsed else None
                media_hash_out  = parsed.get("media_hash") if parsed else None
                recovery_method = "bch" if parsed is not None else "none"

                # Majority-vote fallback for owner_hash when BCH miscorrected.
                if (not owner_match
                        and n_owner_repeat > 0
                        and len(owner_repeat_ext) >= OWNER_HASH_BITS * 2):
                    voted = _majority_vote(owner_repeat_ext, OWNER_HASH_BITS)
                    if voted is not None:
                        voted_hex = array_to_bytes(voted).hex()
                        if voted_hex == expected_owner_h_hex:
                            owner_match     = True
                            owner_hash_out  = voted_hex
                            recovery_method = "majority_vote"
                        elif owner_hash_out is None:
                            owner_hash_out = voted_hex

                # Majority-vote fallback for media_hash.
                if (not media_match
                        and n_media_repeat > 0
                        and len(media_repeat_ext) >= OWNER_HASH_BITS * 2):
                    voted = _majority_vote(media_repeat_ext, OWNER_HASH_BITS)
                    if voted is not None:
                        voted_hex = array_to_bytes(voted).hex()
                        if voted_hex == expected_media_h_hex:
                            media_match     = True
                            media_hash_out  = voted_hex
                            recovery_method = "majority_vote"
                        elif media_hash_out is None:
                            media_hash_out = voted_hex

                # Temporal integrity: the BCH-protected frame_id in LLLL
                # must match this iteration index.  A deletion / insertion
                # / reorder shifts which frame_id lands at which index, so
                # frame_match goes False for every disturbed frame.
                chain_ok = frame_match

                # An *explicit* mismatch is one where BCH+parity validated
                # (parsed is not None) but the field disagrees with the
                # expected value.  The 16-bit parity tag inside the meta
                # payload makes this signal cryptographically strong:
                # codec noise that flips two bits in a Hamming codeword
                # produces a random body whose parity tag passes only at
                # ~1/65536 — so explicit mismatches don't false-positive
                # on lossy compression, while a single occurrence is a
                # definite temporal/identity tamper.
                explicit_frame_mismatch = bool(
                    parsed is not None and parsed.get("frame") != frame_idx
                )
                explicit_owner_mismatch = bool(parsed is not None and not owner_match)
                explicit_media_mismatch = bool(parsed is not None and not media_match)

                content_tampered = res["tampered"]
                frame_tampered   = (
                    content_tampered
                    or not (owner_match and media_match)
                    or parsed is None
                    or not chain_ok
                )

                spatial_maps.append(combined_spatial)

                temporal_map.append(frame_tampered)

                per_frame.append({
                    "frame_idx":               frame_idx,
                    "owner_hash":              owner_hash_out,
                    "media_hash":              media_hash_out,
                    "reported_frame":          parsed.get("frame") if parsed else None,
                    "owner_match":             owner_match,
                    "media_match":             media_match,
                    "chain_ok":                chain_ok,
                    "explicit_frame_mismatch": explicit_frame_mismatch,
                    "explicit_owner_mismatch": explicit_owner_mismatch,
                    "explicit_media_mismatch": explicit_media_mismatch,
                    "recovery_method":         recovery_method,
                                        "ber":                     float(combined_spatial.mean()) if combined_spatial.size else 0.0,
                    "ber_dct":                 res["ber"],

                    "ber_LLLH":                res["ber_A"],
                    "ber_LLHL":                res["ber_B"],
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

    spatial_arr  = np.stack(spatial_maps, axis=0) if spatial_maps else np.empty((0, 0, 0), bool)
    temporal_arr = np.array(temporal_map, dtype=bool)

    n_tampered_frames   = int(temporal_arr.sum())
    n_chain_breaks      = sum(1 for r in per_frame if not r["chain_ok"])
    n_content_tampered  = sum(1 for r in per_frame if r["content_tampered"])
    n_id_mismatch       = sum(1 for r in per_frame if not (r["owner_match"] and r["media_match"]))
    n_owner_match       = sum(1 for r in per_frame if r["owner_match"])
    n_media_match       = sum(1 for r in per_frame if r["media_match"])
    n_explicit_frame_mismatch = sum(1 for r in per_frame if r["explicit_frame_mismatch"])
    n_explicit_owner_mismatch = sum(1 for r in per_frame if r["explicit_owner_mismatch"])
    n_explicit_media_mismatch = sum(1 for r in per_frame if r["explicit_media_mismatch"])
    avg_ber             = float(np.mean([r["ber"] for r in per_frame])) if per_frame else 0.0

    majority = max(len(per_frame) / 2, 1)
    owner_out = owner_id if n_owner_match > majority else None
    media_out = media_id if n_media_match > majority else None

    frame_tamper_rate = n_tampered_frames / max(len(per_frame), 1)

    # Two paths to TAMPERED:
    #   • Any *explicit* cryptographic mismatch — BCH+parity validated,
    #     but the field is wrong.  Single occurrence is sufficient: the
    #     16-bit parity tag makes accidental matches ~1/65536 per frame,
    #     so codec noise won't false-positive here.  Catches small-scale
    #     temporal attacks (frame swap, short reorder) that don't move
    #     enough frames to clear the bulk-rate threshold.
    #   • Bulk failure rate above the per-block BER threshold — covers
    #     content tampering and codec damage that produces unparseable
    #     payloads on many frames at once.
    explicit_tamper = bool(
        n_explicit_frame_mismatch
        or n_explicit_owner_mismatch
        or n_explicit_media_mismatch
    )
    tampered = explicit_tamper or frame_tamper_rate > BER_TAMPER_THRESHOLD

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
        "frames_explicit_owner_mismatch": n_explicit_owner_mismatch,
        "frames_explicit_media_mismatch": n_explicit_media_mismatch,
        "average_ber":                 avg_ber,
        "ber_threshold":               BER_TAMPER_THRESHOLD,
        "frame_tamper_rate":           frame_tamper_rate,
        "TAMPERED":                    bool(tampered),
        "spatial_tamper_map":          spatial_arr,    # F x Hb x Wb bool
        "temporal_tamper_map":         temporal_arr,   # F bool
        "per_frame":                   per_frame,
    }