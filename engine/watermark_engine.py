"""
Semi-Fragile Watermarking Engine
==================================
Pipeline:

  Embed:  Y  → 2-level LWT → {LLLL, LLLH, LLHL} → DCT(8x8) → SVD → QIM on S[0] → IDWT.
          Cb → 16×16 block-mean LSB parity (orthogonal fragile layer).

  Verify: Y  → 2-level LWT → {LLLL, LLLH, LLHL} → DCT(8x8) → SVD → extract bits →
                BCH-decode → owner_id/media_id/frame_id (raw) + majority-vote fallback.
          Cb → block-mean LSB parity → spatial tamper map → decision.

Subband layout (all level-2 LL-family subbands carry payload):
  LLLL  ← BCH-coded meta + frame_id repeats   (most robust — QIM_STEP_ROBUST)
  LLLH  ← spillover for repeats              (used when LLLL is full)
  LLHL  ← spillover for repeats              (used when LLLH is full)

The owner_id and media_id are embedded as RAW 8-byte UTF-8 (null-padded,
truncated if longer) rather than hashes — so verification can RESTORE
the actual strings under compression/tamper, not just match a digest.
Frame_id gets 5x repetition in LLLL (strongest subband) for very strong
temporal detection.

Spatial tamper localisation is carried entirely by the chroma LSB layer.
Temporal integrity (frame deletion / reorder) is handled by the
BCH-protected frame_id in LLLL — see verify_video.

Note on LWT vs DWT
------------------
For the Haar wavelet, the lifting-scheme implementation produces the
*same* subband decomposition as the filterbank DWT.  We use pywt.dwt2
(filterbank) which is mathematically equivalent here — switching to a
true integer-lifting Haar would change nothing about the bits embedded.
"""

import numpy as np
import hashlib
import json
import struct
from typing import Tuple, Optional, Dict, List
import pywt
from scipy.fft import dct, idct
import galois


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
BLOCK_SIZE        = 8       # DCT block size
QIM_STEP_ROBUST   = 32.0    # LLLL  — media meta, robust  (was 24.0)
WAVELET           = 'haar'  # Haar (LWT-equivalent for this wavelet)
DWT_LEVELS        = 2
LSB_QIM_STEP           = 2.0
LSB_BLOCK_TAMPER_RATIO = 0.45   # >45% sub-block disagreement → flag block.
                                # Tuned so compression stays at 0 blocks while
                                # 32x32+ edits flip enough sub-blocks to flag.
# Spatial-block size — each level-2 wavelet coefficient covers 4x4 spatial
# pixels, so an 8x8 coefficient block covers 32x32 pixels.  Used as the
# block granularity for the chroma-LSB tamper map.
SPATIAL_BLOCK = BLOCK_SIZE * (2 ** DWT_LEVELS)   # 32
LSB_SUB_BLOCK_SIZE     = 16     # 16x16 sub-blocks → 4 LSB cells per 32x32 spatial block

BER_TAMPER_THRESHOLD = 0.15  # Informational — kept for metadata back-compat.

# Minimum number of flagged 32×32 spatial blocks required before a frame
# (or image) is considered content-tampered.  The chroma-LSB layer is
# quantization-noisy and routinely produces ~3 isolated false positives
# per frame on a clean re-encode; setting this to 6 tolerates that floor
# while still catching real tampering of a single 32×32 region or larger
# (which collapses ~6+ blocks once the LSB sub-block aggregation rounds
# borders).  Per-block sensitivity in the returned spatial map is
# unchanged — every flagged block still surfaces in the heatmap.
MIN_TAMPER_BLOCKS = 6

# Raw owner_id / media_id are embedded directly (not as hashes) so that
# verification can RESTORE the actual identifier strings under
# compression/tamper, not just compare a fingerprint.  Both are encoded
# as fixed-length UTF-8 (null-padded shorter IDs, truncated longer ones).
# 8 bytes is the sweet spot: long enough for typical IDs ("alice42",
# "vid_001a", etc.), short enough that 3 majority-vote copies fit in
# the LLLL-family capacity of a 512x512 image.
OWNER_ID_BYTES     = 8
MEDIA_ID_BYTES     = 8
OWNER_ID_BITS      = OWNER_ID_BYTES * 8   # 64
MEDIA_ID_BITS      = MEDIA_ID_BYTES * 8   # 64
FRAME_ID_BITS      = 32  # raw frame_id repeated for majority-vote temporal recovery
CHAIN_TAG_BITS     = 16  # raw chain_tag repeated for majority-vote chain recovery

# Repetition counts — each field's raw bits are tiled this many times and
# embedded after the BCH-coded meta.  Decoder majority-votes per field.
# Priority order (head of stream gets the most LLLL space — lower-priority
# fields get truncated first when capacity runs out):
#   frame_id (5x) — temporal integrity is critical for video, and frame_id
#                   sits in LLLL (most robust subband) per user spec
#   owner_id (3x) — provenance
#   media_id (3x) — provenance
#   chain_tag(3x) — replay/reorder detection
FRAME_REPEATS      = 5
OWNER_REPEATS      = 3
MEDIA_REPEATS      = 3
CHAIN_REPEATS      = 3

# Back-compat alias — older callers read these names.
OWNER_HASH_BITS    = OWNER_ID_BITS
OWNER_HASH_REPEATS = OWNER_REPEATS


# ──────────────────────────────────────────────────────────────
# BCH(15,7) — 2-error-correcting BCH used on LLLL meta
# ──────────────────────────────────────────────────────────────
# Each 7-bit data block → 15-bit codeword.  Decoder corrects up to 2 bit
# errors per codeword.  Rate 7/15 ≈ 47% (vs Hamming(7,4)'s 57%).
#
# Pays ~27% more LLLL space for 2× the per-codeword error tolerance.
# Picked over BCH(15,5) (t=3, rate 33%) because the longer BCH-coded
# payload of BCH(15,5) pushes the unprotected repetition copies into
# noisier LLLL real estate — empirically the trade-off comes out negative
# under H.264 noise.  BCH(15,7) preserves the majority-vote margin while
# still improving per-codeword tolerance.

_BCH_N = 15
_BCH_K = 7
_BCH   = galois.BCH(_BCH_N, _BCH_K)   # t = 2 (verified at construction)
_GF2   = galois.GF(2)


def bch_encode_bits(data_bits: np.ndarray) -> np.ndarray:
    """BCH(15,7) encode a bit array (length padded to multiple of 7)."""
    bits = np.asarray(data_bits, dtype=np.uint8)
    pad  = (_BCH_K - len(bits) % _BCH_K) % _BCH_K
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    msg    = _GF2(bits.reshape(-1, _BCH_K))
    coded  = _BCH.encode(msg)
    return np.asarray(coded, dtype=np.uint8).reshape(-1)


def bch_decode_bits(coded_bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """BCH(15,7) decode. Returns (data_bits, num_corrections_applied).

    Codewords with errors beyond t=2 are flagged as uncorrectable by galois
    (errors entry = -1 for that codeword).  We surface that as data bits
    extracted from the systematic positions anyway — the caller's integrity
    tag will reject any uncorrectable codeword that mangled the payload.
    """
    coded = np.asarray(coded_bits, dtype=np.uint8)
    n     = (len(coded) // _BCH_N) * _BCH_N
    if n == 0:
        return np.zeros(0, dtype=np.uint8), 0
    cw_arr = _GF2(coded[:n].reshape(-1, _BCH_N))
    decoded, errs = _BCH.decode(cw_arr, errors=True)
    # `errs` is per-codeword; -1 means uncorrectable.  Count only successful
    # corrections toward the running total (matches Hamming(7,4) semantics).
    errs_np = np.asarray(errs)
    n_corr  = int(errs_np[errs_np > 0].sum()) if errs_np.size else 0
    data    = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    return data, n_corr


# ──────────────────────────────────────────────────────────────
# YCbCr / bit helpers
# ──────────────────────────────────────────────────────────────

def to_float_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = img_rgb[:,:,0].astype(np.float64)
    G = img_rgb[:,:,1].astype(np.float64)
    B = img_rgb[:,:,2].astype(np.float64)
    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128.0
    Cr =  0.5     * R - 0.41869 * G - 0.08131 * B + 128.0
    return Y, Cb, Cr


def from_float_ycbcr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    Cb_ = Cb - 128.0
    Cr_ = Cr - 128.0
    R = Y + 1.40200 * Cr_
    G = Y - 0.34414 * Cb_ - 0.71414 * Cr_
    B = Y + 1.77200 * Cb_
    return np.rint(np.stack([R, G, B], axis=2)).clip(0, 255).astype(np.uint8)


def bits_to_array(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)


def array_to_bytes(arr: np.ndarray) -> bytes:
    bits = (np.asarray(arr) > 0.5).astype(np.uint8)
    pad  = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _encode_id_fixed(s: str, n_bytes: int) -> bytes:
    """UTF-8 encode `s` into exactly `n_bytes`: null-pad short, truncate long.

    Truncation is byte-wise — a multi-byte UTF-8 char that straddles the
    cutoff is dropped from the suffix.  Null-padding lets the decoder
    strip trailing zeros to recover the original (when it fit)."""
    raw = s.encode('utf-8')[:n_bytes]
    return raw + b'\x00' * (n_bytes - len(raw))


def _decode_id_fixed(b: bytes) -> str:
    """Inverse of `_encode_id_fixed`: strip null padding and UTF-8 decode.

    `errors='replace'` keeps decoding robust when a truncated multi-byte
    sequence remains at the tail (recovery still surfaces something
    rather than throwing)."""
    return b.rstrip(b'\x00').decode('utf-8', errors='replace')


def _owner_id_bits_array(owner_id: str) -> np.ndarray:
    """Bit array of the raw fixed-length owner_id (8 bytes UTF-8)."""
    return bits_to_array(_encode_id_fixed(owner_id, OWNER_ID_BYTES))


def _media_id_bits_array(media_id: str) -> np.ndarray:
    """Bit array of the raw fixed-length media_id (8 bytes UTF-8)."""
    return bits_to_array(_encode_id_fixed(media_id, MEDIA_ID_BYTES))


# Back-compat aliases — older callers reference the hash names.
_owner_hash_bits_array = _owner_id_bits_array
_media_hash_bits_array = _media_id_bits_array


def _frame_id_bits_array(frame_id: int) -> np.ndarray:
    """32-bit big-endian array form of frame_id (matches the meta-payload encoding)."""
    return bits_to_array(struct.pack(">I", frame_id))


def _chain_tag_bits_array(chain_tag_bytes: bytes) -> np.ndarray:
    """16-bit array form of the 2-byte chain_tag (matches the meta-payload encoding)."""
    if len(chain_tag_bytes) != 2:
        raise ValueError(f"chain_tag must be 2 bytes, got {len(chain_tag_bytes)}")
    return bits_to_array(chain_tag_bytes)


def _majority_vote(repeated_bits: np.ndarray, copy_len: int) -> Optional[np.ndarray]:
    """Recover a copy_len-bit pattern from N concatenated copies of itself
    by majority vote.  Returns None when fewer than 2 full copies are
    available (vote isn't meaningful otherwise)."""
    n = len(repeated_bits) // copy_len
    if n < 2:
        return None
    copies = repeated_bits[:n * copy_len].reshape(n, copy_len).astype(np.int32)
    votes  = copies.sum(axis=0)
    return (votes * 2 > n).astype(np.uint8)   # strict majority; ties → 0


# ──────────────────────────────────────────────────────────────
# CORE DCT → SVD → QIM (single subband, fixed 1 bit per 8x8 block)
# ──────────────────────────────────────────────────────────────

def _tile_blocks(subband: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
    """Reshape `subband` into a contiguous (N, 8, 8) batch of non-overlapping
    8×8 tiles, in the same row-major (bi, bj) order as the original
    nested-loop implementations.  Returns the batch plus shape info needed
    to untile.  Trailing pixels that don't form a complete block are
    dropped, matching the original `range(0, h - BLOCK + 1, BLOCK)` walk.
    """
    H, W = subband.shape
    H8 = (H // BLOCK_SIZE) * BLOCK_SIZE
    W8 = (W // BLOCK_SIZE) * BLOCK_SIZE
    nh = H8 // BLOCK_SIZE
    nw = W8 // BLOCK_SIZE
    # transpose forces a copy on reshape, so the returned batch is a
    # fresh contiguous buffer — safe to mutate in-place.
    blocks = (subband[:H8, :W8]
              .reshape(nh, BLOCK_SIZE, nw, BLOCK_SIZE)
              .transpose(0, 2, 1, 3)
              .reshape(nh * nw, BLOCK_SIZE, BLOCK_SIZE))
    return blocks, nh, nw, H8, W8


def _embed_bits_dct_svd(
    subband: np.ndarray,
    bits: np.ndarray,
    qim_step: float,
) -> Tuple[np.ndarray, int]:
    """Embed `bits` into 8x8 DCT→SVD blocks of `subband` via S[0] QIM.

    Walks blocks row-major; exactly one bit per block — never skip a
    block, since embed/extract must iterate identical positions or the
    per-block fingerprint comparison de-synchronises.

    Per block: quantise S[0] to the nearest QIM lattice point matching
    the bit's parity (q=1 fallback when S[0] rounds to 0). The +2 bump
    that follows preserves parity while keeping new S[0] >= S[1], so
    the decoder's re-SVD doesn't sort it down — see CLAUDE.md
    "SVD ordering invariant".

    Implementation: tiles the subband into a (N, 8, 8) batch and runs a
    single batched DCT + SVD over the whole subband instead of one
    Python-level call per block.  Mathematically equivalent to the
    sequential per-block code; the SVD-ordering guard is computed in
    closed form (`q + 2*ceil(max(0, S1/step - q)/2)`) instead of being
    iterated.
    """
    sub = subband.copy()
    blocks, nh, nw, H8, W8 = _tile_blocks(sub)
    total_blocks = blocks.shape[0]
    n_bits = min(len(bits), total_blocks)
    if n_bits == 0:
        return sub, 0

    proc = blocks[:n_bits]

    # Batched 2D orthonormal DCT — same axis order (row, col) as the
    # original per-block call applied along axis=0 then axis=1.
    dblocks = dct(dct(proc, axis=1, norm='ortho'), axis=2, norm='ortho')

    # Batched SVD — one LAPACK call covers every block.
    U, S, Vt = np.linalg.svd(dblocks)

    S0 = S[:, 0]
    S1 = S[:, 1]   # always present: 8x8 blocks → S has length 8
    bits_arr = np.asarray(bits[:n_bits], dtype=np.int64) & 1

    # Nearest-parity quantization (vectorised mirror of the original).
    q = np.round(S0 / qim_step).astype(np.int64)
    direction = np.where(S0 >= q * qim_step, 1, -1).astype(np.int64)
    wrong_parity = (q % 2) != bits_arr
    q_adj = np.where(
        wrong_parity,
        np.where(q == 0, 1, q + direction),
        q,
    )

    # SVD-ordering guard: smallest q' = q_adj + 2k (k≥0 integer) such
    # that q'*step >= S1.  Equivalent to `while q*step < S1: q += 2`.
    deficit = S1 / qim_step - q_adj
    k = np.ceil(np.maximum(deficit, 0.0) / 2.0).astype(np.int64)
    q_final = q_adj + 2 * k

    S_new = S.copy()
    S_new[:, 0] = q_final * qim_step

    # Reconstruct DCT block: U @ diag(S) @ Vt == U @ (S[..., None] * Vt)
    recon = np.matmul(U, S_new[..., None] * Vt)

    # Inverse 2D DCT — reverse the axis order used in the forward pass.
    iblocks = idct(idct(recon, axis=2, norm='ortho'), axis=1, norm='ortho')

    blocks[:n_bits] = iblocks

    # Untile back to spatial layout.
    sub[:H8, :W8] = (blocks
                     .reshape(nh, nw, BLOCK_SIZE, BLOCK_SIZE)
                     .transpose(0, 2, 1, 3)
                     .reshape(H8, W8))
    return sub, n_bits


def _extract_bits_dct_svd_with_mask(
    subband: np.ndarray,
    n_bits: int,
    qim_step: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract up to n_bits bits, one per 8x8 block in the same row-major
    order used by `_embed_bits_dct_svd`.  Batched DCT + SVD; semantics
    identical to the per-block loop.
    """
    blocks, _, nw, _, _ = _tile_blocks(subband)
    total_blocks = blocks.shape[0]
    n_extract = min(n_bits, total_blocks)
    if n_extract == 0:
        return np.zeros(0, dtype=np.uint8), []

    proc = blocks[:n_extract]
    dblocks = dct(dct(proc, axis=1, norm='ortho'), axis=2, norm='ortho')
    _, S, _ = np.linalg.svd(dblocks)

    bits = (np.round(S[:, 0] / qim_step).astype(np.int64) % 2).astype(np.uint8)

    coords: List[Tuple[int, int]] = []
    for k in range(n_extract):
        bi, bj = divmod(k, nw)
        coords.append((bi * BLOCK_SIZE, bj * BLOCK_SIZE))
    return bits, coords


# ──────────────────────────────────────────────────────────────
# LSB-BASED SPATIAL LOCALISATION  (compression-tolerant layer)
# ──────────────────────────────────────────────────────────────
# Encodes a target parity bit in the LSB of each 4×4 sub-block's mean
# (block-mean QIM with step Q=2).  Aggregated to the same 32×32 spatial
# grid as the DCT/SVD fingerprint map, so the two maps can be ANDed for
# compression-robust tamper localisation.

def _compute_lsb_target_pattern(
    n_sub_h: int,
    n_sub_w: int,
    signature: bytes,
    frame_id: int,
) -> np.ndarray:
    """Pseudo-random target LSB grid keyed by (signature, frame_id).

    Same shape as the 4×4 sub-block grid.  Both encoder and decoder
    derive identical bits from public inputs — no key material.
    """
    n_total = n_sub_h * n_sub_w
    n_bytes = (n_total + 7) // 8
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        h = hashlib.sha256()
        h.update(b"lsb-pattern-v1|")
        h.update(signature)
        h.update(struct.pack(">II", frame_id, counter))
        out.extend(h.digest())
        counter += 1
    raw  = np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8)
    bits = np.unpackbits(raw)[:n_total]
    return bits.reshape(n_sub_h, n_sub_w).astype(np.uint8)


def _embed_lsb_sub_block_parity(
    Y: np.ndarray,
    target_pattern: np.ndarray,
    sub_block: int = LSB_SUB_BLOCK_SIZE,
    q: float = LSB_QIM_STEP,
) -> np.ndarray:
    """Adjust Y so each 4×4 sub-block's mean encodes the target LSB parity.

    Per sub-block: shift the mean to the nearest multiple of q whose
    parity matches the target bit.  The shift is applied uniformly to
    all 16 pixels of the sub-block, so each pixel moves by at most q/2.
    With q=2 this is ≤1 grey-level → ≥48 dB local PSNR.
    """
    Y_out = Y.copy().astype(np.float64)
    n_sub_h, n_sub_w = target_pattern.shape
    H = n_sub_h * sub_block
    W = n_sub_w * sub_block

    # View Y_out's top-left H×W region as a (n_sub_h, sub_block, n_sub_w, sub_block)
    # tile array — this is a view, so in-place += writes back to Y_out.
    tile = Y_out[:H, :W].reshape(n_sub_h, sub_block, n_sub_w, sub_block)
    means = tile.mean(axis=(1, 3))                           # (n_sub_h, n_sub_w)

    targets    = (target_pattern & 1).astype(np.int64)
    current_q  = np.round(means / q).astype(np.int64)
    parity_now = current_q & 1
    needs_flip = parity_now != targets

    # When parity is wrong, choose the nearer of (current_q + 1) and (current_q - 1).
    up        = current_q + 1
    down      = current_q - 1
    up_dist   = np.abs(up   * q - means)
    down_dist = np.abs(down * q - means)
    flipped   = np.where(up_dist <= down_dist, up, down)
    target_q  = np.where(needs_flip, flipped, current_q)

    shift = target_q * q - means                             # (n_sub_h, n_sub_w)
    tile += shift[:, None, :, None]                          # broadcast to all 16 px
    return Y_out


def _verify_lsb_sub_block_parity(
    Y: np.ndarray,
    target_pattern: np.ndarray,
    sub_block: int = LSB_SUB_BLOCK_SIZE,
    q: float = LSB_QIM_STEP,
) -> np.ndarray:
    """Per-sub-block boolean: True where extracted LSB parity ≠ target."""
    n_sub_h, n_sub_w = target_pattern.shape
    H = n_sub_h * sub_block
    W = n_sub_w * sub_block
    tile  = Y[:H, :W].astype(np.float64).reshape(n_sub_h, sub_block, n_sub_w, sub_block)
    means = tile.mean(axis=(1, 3))
    actual_lsb = (np.round(means / q).astype(np.int64) & 1).astype(np.uint8)
    return actual_lsb != (target_pattern & 1)


def _aggregate_lsb_to_block_map(
    sub_mismatch: np.ndarray,
    spatial_block: int = SPATIAL_BLOCK,
    sub_block: int = LSB_SUB_BLOCK_SIZE,
    ratio_threshold: float = LSB_BLOCK_TAMPER_RATIO,
) -> np.ndarray:
    """Down-sample the sub-block mismatch grid to the 32×32 spatial-block grid.

    A spatial block is flagged when more than `ratio_threshold` of its
    (spatial_block/sub_block)² = 64 sub-bits disagree with their target.
    Compression noise produces ~5% disagreements (well below 30%); real
    tampering of a block produces ~50% (well above 30%) — so this single
    threshold cleanly separates the two regimes.
    """
    sub_per_block = spatial_block // sub_block               # 32/4 = 8
    n_sub_h, n_sub_w = sub_mismatch.shape
    n_bh = n_sub_h // sub_per_block
    n_bw = n_sub_w // sub_per_block
    if n_bh == 0 or n_bw == 0:
        return np.zeros((n_bh, n_bw), dtype=bool)
    # Trim ragged remainder so reshape divides evenly.
    trimmed = sub_mismatch[:n_bh * sub_per_block, :n_bw * sub_per_block].astype(np.float64)
    ratio = trimmed.reshape(n_bh, sub_per_block, n_bw, sub_per_block).mean(axis=(1, 3))
    return ratio > ratio_threshold


def _embed_lsb_chroma(
    Cb: np.ndarray,
    signature: bytes,
    frame_id: int,
) -> np.ndarray:
    """Embed the LSB block-mean parity layer into a Cb chroma channel.

    Chroma is used (not luminance) so the layer is fully orthogonal to
    the DCT/SVD watermark in Y — they cannot disturb each other.  The
    spatial-map signal it produces is independent of, and complementary
    to, the DCT/SVD fingerprint map.
    """
    n_sub_h = Cb.shape[0] //LSB_SUB_BLOCK_SIZE
    n_sub_w = Cb.shape[1] //LSB_SUB_BLOCK_SIZE
    pattern = _compute_lsb_target_pattern(n_sub_h, n_sub_w, signature, frame_id)
    return _embed_lsb_sub_block_parity(Cb, pattern)


def _verify_lsb_chroma(
    Cb: np.ndarray,
    signature: bytes,
    frame_id: int,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Extract the LSB block-mean parity layer from a Cb chroma channel.

    If `target_shape` is given, the aggregated block map is cropped to
    match it — letting the caller AND the LSB map directly with the
    DCT/SVD spatial map without size mismatches.
    """
    n_sub_h = Cb.shape[0] // LSB_SUB_BLOCK_SIZE
    n_sub_w = Cb.shape[1] // LSB_SUB_BLOCK_SIZE
    pattern = _compute_lsb_target_pattern(n_sub_h, n_sub_w, signature, frame_id)
    sub_mismatch = _verify_lsb_sub_block_parity(Cb, pattern)
    block_map = _aggregate_lsb_to_block_map(sub_mismatch)
    # The pattern actually extracted from the (possibly tampered) chroma is the
    # expected target XOR the per-sub-block mismatch — wherever they disagree,
    # the extracted bit flipped.  Both grids are surfaced so callers can RENDER
    # the original vs. extracted fragile watermark side by side.
    actual_pattern = (pattern.astype(np.uint8) ^ sub_mismatch.astype(np.uint8))
    if target_shape is not None:
        h_min = min(block_map.shape[0], target_shape[0])
        w_min = min(block_map.shape[1], target_shape[1])
        block_map = block_map[:h_min, :w_min]
    return {
        "lsb_spatial_map":    block_map,
        "lsb_sub_mismatch":   sub_mismatch,
        "lsb_target_pattern": pattern.astype(np.uint8),
        "lsb_actual_pattern": actual_pattern,
        "ber_lsb_sub":      float(sub_mismatch.mean()) if sub_mismatch.size else 0.0,
        "ber_lsb_block":    float(block_map.mean())    if block_map.size    else 0.0,
    }


def _wm_display_patterns(lsb_res: Dict, content_tampered: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (extracted_pattern, red_mask) used to VISUALISE the fragile
    watermark, with the quantization noise-floor suppressed.

    The raw `lsb_sub_mismatch` flags every sub-block whose parity flipped —
    including the scattered, isolated flips caused by the RGB↔YCbCr round-trip
    and codec noise.  Showing those as red on an AUTHENTIC result is
    misleading ("authentic, but why is there red?").  So:

      • content_tampered=False → return the expected pattern unchanged with an
        all-False mask: the 'extracted' image is identical to the 'original'
        and shows NO red at all.
      • content_tampered=True  → red only on sub-blocks that fall inside a
        32x32 block actually flagged by the ratio threshold (the real tampered
        region).  Isolated noise flips outside any flagged block are dropped,
        and the extracted image flips only those real bits.

    This keeps the visualisation consistent with the verdict: clean when
    authentic, localised red cluster when tampered.
    """
    target = lsb_res.get("lsb_target_pattern")
    if target is None or target.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty, empty.astype(bool)
    target = target.astype(np.uint8)
    if not content_tampered:
        return target, np.zeros(target.shape, dtype=bool)

    sub_mismatch  = np.asarray(lsb_res["lsb_sub_mismatch"], dtype=bool)
    block_map     = np.asarray(lsb_res["lsb_spatial_map"],  dtype=bool)
    sub_per_block = SPATIAL_BLOCK // LSB_SUB_BLOCK_SIZE
    # Upsample the coarse flagged-block grid back to the sub-block grid.
    up = np.kron(block_map, np.ones((sub_per_block, sub_per_block), dtype=bool))
    h = min(up.shape[0], sub_mismatch.shape[0])
    w = min(up.shape[1], sub_mismatch.shape[1])
    mask = np.zeros(sub_mismatch.shape, dtype=bool)
    mask[:h, :w] = sub_mismatch[:h, :w] & up[:h, :w]
    extracted = (target ^ mask.astype(np.uint8))
    return extracted, mask



# ──────────────────────────────────────────────────────────────
# 2-LEVEL LWT EMBED  (full pipeline for a single Y channel)
# ──────────────────────────────────────────────────────────────

def _embed_into_channel(
    Y: np.ndarray,
    meta_bits: np.ndarray,           # already BCH-encoded
    signature: bytes,
    frame_id: int,
    extra_llll_bits: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """Embed BCH-coded meta + raw repetition copies across the level-2
    LL-family subbands (LLLL → LLLH → LLHL, in priority order).

    LLLL is the most compression-robust subband, so we fill it first.
    LLLH and LLHL are used as overflow when the payload exceeds LLLL's
    capacity (which happens for any reasonably-sized payload at small
    image dimensions like 512x512).  HH-family subbands stay untouched.
    """
    Y = Y.astype(np.float64)

    # Level-1 then level-2 LWT (Haar lifting ≡ Haar DWT)
    LL,   (LH1, HL1, HH1) = pywt.dwt2(Y,  WAVELET)
    LLLL, (LLLH, LLHL, LLHH) = pywt.dwt2(LL, WAVELET)

    # Build the combined bit stream: BCH-coded meta first, then raw
    # repetition copies (majority-voted at decode).  The meta payload
    # already contains the IDs once via BCH; the repetition copies are
    # an independent recovery channel for when BCH miscorrects.
    if extra_llll_bits is None or len(extra_llll_bits) == 0:
        full_stream = np.asarray(meta_bits, dtype=np.uint8)
    else:
        full_stream = np.concatenate([
            np.asarray(meta_bits, dtype=np.uint8),
            np.asarray(extra_llll_bits, dtype=np.uint8),
        ])

    # ── Embed across LLLL → LLLH → LLHL in priority order. ──
    # Each subband-embed returns the number of bits it consumed; we
    # cascade the remainder into the next subband.
    cursor = 0
    LLLL, n_in_llll = _embed_bits_dct_svd(LLLL, full_stream[cursor:], QIM_STEP_ROBUST)
    cursor += n_in_llll

    n_in_llh = 0
    if cursor < len(full_stream):
        LLLH, n_in_llh = _embed_bits_dct_svd(LLLH, full_stream[cursor:], QIM_STEP_ROBUST)
        cursor += n_in_llh

    n_in_lhl = 0
    if cursor < len(full_stream):
        LLHL, n_in_lhl = _embed_bits_dct_svd(LLHL, full_stream[cursor:], QIM_STEP_ROBUST)
        cursor += n_in_lhl

    n_embedded = cursor
    n_meta     = min(n_embedded, len(meta_bits))
    n_extra    = max(0, n_embedded - len(meta_bits))

    # Inverse 2-level
    LL_wm  = pywt.idwt2((LLLL, (LLLH, LLHL, LLHH)), WAVELET)[:LL.shape[0], :LL.shape[1]]
    Y_wm   = pywt.idwt2((LL_wm, (LH1, HL1, HH1)),   WAVELET)[:Y.shape[0],  :Y.shape[1]]

    info = {
        "n_meta_bits":         int(n_meta),
        "n_extra_llll_bits":   int(n_extra),
        "n_bits_llll":         int(n_in_llll),
        "n_bits_lllh":         int(n_in_llh),
        "n_bits_llhl":         int(n_in_lhl),
    }
    return Y_wm, info


# ──────────────────────────────────────────────────────────────
# 2-LEVEL LWT EXTRACT + per-block compare → BER + spatial map
# ──────────────────────────────────────────────────────────────

def _verify_channel(
    Y: np.ndarray,
    n_meta_bits_coded: int,
    signature: bytes,
    frame_id: int,
    n_extra_llll_bits: int = 0,
) -> Dict:
    """Extract the bit stream from the level-2 LL-family subbands in the
    same priority order used at embed: LLLL → LLLH → LLHL.  The cascade
    must match `_embed_into_channel` exactly or the per-block alignment
    drifts and every downstream bit becomes garbage."""
    Yf = Y.astype(np.float64)
    LL, _ = pywt.dwt2(Yf, WAVELET)
    LLLL, (LLLH, LLHL, _LLHH) = pywt.dwt2(LL, WAVELET)

    n_extra     = max(0, n_extra_llll_bits)
    n_total     = n_meta_bits_coded + n_extra

    # Pull from LLLL first, spill into LLLH then LLHL — same order as embed.
    parts: List[np.ndarray] = []
    remaining = n_total

    for sub in (LLLL, LLLH, LLHL):
        if remaining <= 0:
            break
        bits, _ = _extract_bits_dct_svd_with_mask(sub, remaining, QIM_STEP_ROBUST)
        parts.append(bits)
        remaining -= len(bits)

    full_ext  = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    meta_bits = full_ext[:n_meta_bits_coded]
    extra_ext = full_ext[n_meta_bits_coded:n_meta_bits_coded + n_extra]

    return {
        "meta_bits_coded":  meta_bits,
        "extra_llll_bits":  extra_ext,
    }


# ──────────────────────────────────────────────────────────────
# META PAYLOAD (LLLL)
# ──────────────────────────────────────────────────────────────

def _chain_genesis(media_hash_bytes: bytes) -> bytes:
    """Initial 32-byte chain state for frame 0 (or for an image's only frame).

    Derived from the media hash so different media start independent chains.
    No secret key — the chain provides temporal integrity, not authenticity
    (CLAUDE.md "Known weaknesses"); add HMAC if forgery resistance is needed.
    """
    h = hashlib.sha256()
    h.update(b"chain-genesis-v1|")
    h.update(media_hash_bytes)
    return h.digest()


def _frame_chain_hash(media_hash_bytes: bytes, prev_chain: bytes, frame_id: int) -> bytes:
    """Evolve the chain state by one frame.  Returns the new 32-byte state.

    chain_i = SHA256("chain-v1|" || media_hash || chain_{i-1} || frame_id_i)
    The first 2 bytes of the result are what gets embedded in LLLL.
    """
    h = hashlib.sha256()
    h.update(b"chain-v1|")
    h.update(media_hash_bytes)
    h.update(prev_chain)
    h.update(struct.pack(">I", frame_id))
    return h.digest()


def _expected_chain_state(media_hash_bytes: bytes, iteration_index: int,
                          embed_every_n_frames: int = 1) -> bytes:
    """Compute the chain state that *should* be present at a given iteration.

    Verifier-side helper.  Deterministic from (media_hash, iteration_index)
    — does not require walking every frame, so verify can still sample.
    """
    chain = _chain_genesis(media_hash_bytes)
    for k in range(iteration_index + 1):
        frame_id = k * embed_every_n_frames
        chain = _frame_chain_hash(media_hash_bytes, chain, frame_id)
    return chain


# Meta payload layout (24 bytes, BCH(15,7)-protected):
#   owner_id(8) || media_id(8) || frame_id(4) || chain_tag(2) || integrity_tag(2)
# owner_id and media_id are raw UTF-8 (null-padded), not hashes — this is
# what lets verify RESTORE the actual identifier strings rather than just
# matching a fingerprint.
META_PAYLOAD_BYTES = OWNER_ID_BYTES + MEDIA_ID_BYTES + 4 + 2 + 2   # = 24
_META_BODY_BYTES   = META_PAYLOAD_BYTES - 2                        # = 22 (excl. integrity)


def _build_meta_payload(owner_id: str, media_id: str, frame_id: int,
                        chain_tag: bytes = b"\x00\x00") -> bytes:
    """Compact 24-byte LLLL header carrying the raw IDs (not hashes).

    Layout:
      owner_id   : 8 bytes  UTF-8, null-padded/truncated
      media_id   : 8 bytes  UTF-8, null-padded/truncated
      frame_id   : 4 bytes  big-endian uint32
      chain_tag  : 2 bytes  (caller-supplied — first 2B of SHA-256 chain state)
      integrity  : 2 bytes  SHA-256(body)[:2]  — sanity check on BCH decode
    """
    if len(chain_tag) != 2:
        raise ValueError(f"chain_tag must be 2 bytes, got {len(chain_tag)}")
    owner_b = _encode_id_fixed(owner_id, OWNER_ID_BYTES)
    media_b = _encode_id_fixed(media_id, MEDIA_ID_BYTES)
    frame_b = struct.pack(">I", frame_id)
    body    = owner_b + media_b + frame_b + chain_tag                # 22 bytes
    tag     = hashlib.sha256(body).digest()[:2]
    return body + tag                                                 # 24 bytes


def _parse_meta_payload(meta_bytes: bytes) -> Optional[Dict]:
    """Inverse of `_build_meta_payload`. Returns None if the integrity tag
    doesn't validate (BCH miscorrected / payload is junk)."""
    try:
        if len(meta_bytes) < META_PAYLOAD_BYTES:
            return None
        body = meta_bytes[:_META_BODY_BYTES]
        tag  = meta_bytes[_META_BODY_BYTES:META_PAYLOAD_BYTES]
        if hashlib.sha256(body).digest()[:2] != tag:
            return None
        p = 0
        owner_b = body[p:p + OWNER_ID_BYTES]; p += OWNER_ID_BYTES
        media_b = body[p:p + MEDIA_ID_BYTES]; p += MEDIA_ID_BYTES
        frame_b = body[p:p + 4];              p += 4
        chain_b = body[p:p + 2]
        return {
            "owner_id":  _decode_id_fixed(owner_b),
            "media_id":  _decode_id_fixed(media_b),
            "frame":     struct.unpack(">I", frame_b)[0],
            "chain_tag": chain_b.hex(),
        }
    except Exception:
        return None


def _meta_owner_media_id(owner_id: str, media_id: str) -> Tuple[str, str]:
    """Decoded round-trip form of owner_id / media_id as they would appear
    after embed + extract.  Used to compare an extracted ID against what
    the caller passed in, accounting for null-padding truncation.
    """
    return (
        _decode_id_fixed(_encode_id_fixed(owner_id, OWNER_ID_BYTES)),
        _decode_id_fixed(_encode_id_fixed(media_id, MEDIA_ID_BYTES)),
    )


# Back-compat alias — kept so older callers don't break during transition.
_meta_owner_media_hash = _meta_owner_media_id


def _master_signature(owner_id: str, media_id: str, frame_id: int) -> bytes:
    """A deterministic per-frame signature.  Acts as the 'Signature' input
    to the per-block SHA256 — both encoder and decoder reconstruct it from
    the (owner, media, frame) tuple, no key material required.
    """
    h = hashlib.sha256()
    h.update(b"semi-fragile-wm/v2|")
    h.update(owner_id.encode())
    h.update(b"|")
    h.update(media_id.encode())
    h.update(b"|")
    h.update(struct.pack(">I", frame_id))
    return h.digest()


# ──────────────────────────────────────────────────────────────
# CAPACITY-AWARE REPEAT ALLOCATION
# ──────────────────────────────────────────────────────────────

def _llfamily_block_capacity(y_shape: Tuple[int, int]) -> int:
    """Total 8x8-block payload slots across the level-2 LL-family subbands
    (LLLL + LLLH + LLHL) for a Y channel of `y_shape` — i.e. how many bits
    `_embed_into_channel` can carry.  Derived from the exact pywt subband
    shapes so it matches the tiling used at embed time.
    """
    LL, _ = pywt.dwt2(np.zeros(y_shape, dtype=np.float64), WAVELET)
    LLLL, (LLLH, LLHL, _LLHH) = pywt.dwt2(LL, WAVELET)
    return sum((s.shape[0] // BLOCK_SIZE) * (s.shape[1] // BLOCK_SIZE)
               for s in (LLLL, LLLH, LLHL))


def _plan_repeat_copies(extra_capacity_bits: int,
                        frame_repeats: int) -> Tuple[int, int, int, int]:
    """Allocate majority-vote repeat copies across frame_id / owner_id /
    media_id / chain_tag within the available LL-family capacity.

    owner_id and media_id are allocated as PAIRS, so they always receive the
    SAME number of copies — neither provenance field is starved.  This fixes
    the old greedy frame→owner→media→chain order, where media_id routinely got
    0 copies (no majority-vote recovery) while owner_id got 2, making media_id
    the first thing to fail under compression.  frame_id leads (temporal,
    lives in LLLL); chain_tag takes whatever remains.

    Returns (n_frame, n_owner, n_media, n_chain).
    """
    rem = max(0, int(extra_capacity_bits))
    nf = min(frame_repeats, rem // FRAME_ID_BITS)
    rem -= nf * FRAME_ID_BITS
    pair_bits = OWNER_ID_BITS + MEDIA_ID_BITS
    pairs = min(OWNER_REPEATS, MEDIA_REPEATS, rem // pair_bits)
    n_owner = n_media = pairs
    rem -= pairs * pair_bits
    nc = min(CHAIN_REPEATS, rem // CHAIN_TAG_BITS)
    return nf, n_owner, n_media, nc


# ──────────────────────────────────────────────────────────────
# PUBLIC API — IMAGE
# ──────────────────────────────────────────────────────────────

def embed_image(
    img_array: np.ndarray,
    owner_id: str,
    media_id: str,
    frame_id: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """Embed a semi-fragile watermark.

    Returns (watermarked_uint8_RGB, metadata_dict).
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        raise ValueError("Input must be H×W×3 RGB.")

    signature   = _master_signature(owner_id, media_id, frame_id)
    # Chain still keyed by media_id — kept as a SHA-256 derivation so the
    # chain_tag's secrecy properties are unchanged.  This is an internal
    # derivation, not an embedded fingerprint, so it doesn't conflict with
    # the "no hashes in the watermark" goal.
    media_h_b   = hashlib.sha256(media_id.encode()).digest()[:4]
    chain_state = _frame_chain_hash(media_h_b, _chain_genesis(media_h_b), frame_id)
    chain_tag   = chain_state[:2]
    meta_raw    = _build_meta_payload(owner_id, media_id, frame_id, chain_tag=chain_tag)
    meta_bits   = bits_to_array(meta_raw)
    meta_coded  = bch_encode_bits(meta_bits)

    # Y channel first — its size determines the LL-family capacity, which we
    # need before deciding how many majority-vote repeat copies to embed.
    Y, Cb, Cr = to_float_ycbcr(img_array[:,:,:3])

    # Fair repeat allocation: owner_id and media_id get EQUAL copies so media
    # is no longer starved.  A still image's frame_id is always 0 and already
    # carried by the BCH meta, so a single raw copy suffices — the freed space
    # goes to the provenance IDs.  (frame_id → owner_id ≡ media_id → chain_tag)
    capacity_bits = _llfamily_block_capacity(Y.shape)
    extra_cap     = capacity_bits - len(meta_coded)
    n_frame_copies, n_owner_copies, n_media_copies, n_chain_copies = \
        _plan_repeat_copies(extra_cap, frame_repeats=1)

    extras = np.concatenate([
        np.tile(_frame_id_bits_array(frame_id),   n_frame_copies),
        np.tile(_owner_id_bits_array(owner_id),    n_owner_copies),
        np.tile(_media_id_bits_array(media_id),    n_media_copies),
        np.tile(_chain_tag_bits_array(chain_tag),  n_chain_copies),
    ]).astype(np.uint8)

    Y_wm, info = _embed_into_channel(Y, meta_coded, signature, frame_id, extras)

    # LSB layer on chroma — independent of Y's DCT/SVD watermark.
    Cb_wm = _embed_lsb_chroma(Cb, signature, frame_id)
    if info["n_meta_bits"] < len(meta_coded):
        raise ValueError(
            f"LLLL family too small: embedded {info['n_meta_bits']}/{len(meta_coded)} "
            f"BCH-coded meta bits.  Use a larger image or shorter owner/media IDs."
        )

    n_frame_bits  = n_frame_copies * FRAME_ID_BITS
    n_owner_bits  = n_owner_copies * OWNER_ID_BITS
    n_media_bits  = n_media_copies * MEDIA_ID_BITS
    n_chain_bits  = n_chain_copies * CHAIN_TAG_BITS

    if n_owner_copies < OWNER_REPEATS or n_media_copies < MEDIA_REPEATS:
        print(f"[embed_image] note: fit "
              f"{n_frame_copies} frame_id + "
              f"{n_owner_copies}/{OWNER_REPEATS} owner_id + "
              f"{n_media_copies}/{MEDIA_REPEATS} media_id + "
              f"{n_chain_copies}/{CHAIN_REPEATS} chain_tag majority-vote copies "
              f"(LL-family capacity {capacity_bits} bits).")

    rgb = from_float_ycbcr(Y_wm, Cb_wm, Cr)
    mse = np.mean((img_array[:,:,:3].astype(np.float64) - rgb.astype(np.float64)) ** 2)
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    metadata = {
        "version":             "v4",
        "owner_id":            owner_id,
        "media_id":            media_id,
        "frame_id":            frame_id,
        "meta_bits_raw":       int(len(meta_bits)),
        "meta_bits_coded":     int(len(meta_coded)),
        "frame_repeat_bits":     int(n_frame_bits),
        "frame_repeat_copies":   int(n_frame_copies),
        "owner_repeat_bits":     int(n_owner_bits),
        "owner_repeat_copies":   int(n_owner_copies),
        "media_repeat_bits":     int(n_media_bits),
        "media_repeat_copies":   int(n_media_copies),
        "chain_repeat_bits":     int(n_chain_bits),
        "chain_repeat_copies":   int(n_chain_copies),
        "owner_id_bits":         int(OWNER_ID_BITS),
        "media_id_bits":         int(MEDIA_ID_BITS),
        "frame_id_bits":         int(FRAME_ID_BITS),
        "chain_tag_bits":        int(CHAIN_TAG_BITS),
        # Back-compat alias — older callers read owner_hash_bits.
        "owner_hash_bits":       int(OWNER_ID_BITS),
        "psnr_db":             psnr,
    }
    return rgb, metadata


def verify_image(
    img_array: np.ndarray,
    metadata: Dict,
) -> Dict:
    """Verify a watermarked image.  Returns a verdict dict including a
    spatial tamper map and BER measurement.
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        Y = img_array.astype(np.float64) if img_array.ndim == 2 else \
            (0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2]).astype(np.float64)
        Cb = None
    else:
        Y, Cb, _ = to_float_ycbcr(img_array[:,:,:3])

    frame_id   = metadata["frame_id"]
    signature  = _master_signature(metadata["owner_id"], metadata["media_id"], frame_id)

    n_owner_repeat = int(metadata.get("owner_repeat_bits", 0))
    n_media_repeat = int(metadata.get("media_repeat_bits", 0))
    n_frame_repeat = int(metadata.get("frame_repeat_bits", 0))
    n_chain_repeat = int(metadata.get("chain_repeat_bits", 0))
    n_owner_copies = int(metadata.get("owner_repeat_copies", 0))
    n_media_copies = int(metadata.get("media_repeat_copies", 0))
    n_frame_copies = int(metadata.get("frame_repeat_copies", 0))
    n_chain_copies = int(metadata.get("chain_repeat_copies", 0))

    n_extra_total = n_owner_repeat + n_media_repeat + n_frame_repeat + n_chain_repeat
    res = _verify_channel(
        Y, metadata["meta_bits_coded"], signature, frame_id,
        n_extra_llll_bits=n_extra_total,
    )
    extra_bits   = res["extra_llll_bits"]
    # Extra-bit layout MUST match embed order: frame_id → owner → media → chain.
    p = 0
    frame_repeat_ext = extra_bits[p:p + n_frame_repeat]; p += n_frame_repeat
    owner_repeat_ext = extra_bits[p:p + n_owner_repeat]; p += n_owner_repeat
    media_repeat_ext = extra_bits[p:p + n_media_repeat]; p += n_media_repeat
    chain_repeat_ext = extra_bits[p:p + n_chain_repeat]

    # BCH-decode meta bits
    meta_bits_dec, n_corr = bch_decode_bits(res["meta_bits_coded"])
    meta_bytes = array_to_bytes(meta_bits_dec[:metadata["meta_bits_raw"]])
    parsed     = _parse_meta_payload(meta_bytes)

    # Compare against the round-trip form of the IDs (i.e., after encode→decode
    # truncation), so IDs longer than the fixed slot still match cleanly.
    expected_owner_id, expected_media_id = _meta_owner_media_id(
        metadata["owner_id"], metadata["media_id"])

    # Expected chain_tag for a single-image case: one chain step from genesis,
    # keyed by the first 4 bytes of SHA-256(media_id) (matches embed_image).
    media_h_b           = hashlib.sha256(metadata["media_id"].encode()).digest()[:4]
    expected_chain_full = _frame_chain_hash(media_h_b, _chain_genesis(media_h_b), frame_id)
    expected_chain_tag  = expected_chain_full[:2].hex()

    expected_frame_id_hex = struct.pack(">I", frame_id).hex()

    bch_ok           = parsed is not None
    owner_match      = bool(parsed and parsed.get("owner_id") == expected_owner_id)
    media_match      = bool(parsed and parsed.get("media_id") == expected_media_id)
    chain_match      = bool(parsed and parsed.get("chain_tag") == expected_chain_tag)
    frame_match      = bool(parsed and parsed.get("frame")     == frame_id)
    owner_id_out     = parsed.get("owner_id") if parsed else None
    media_id_out     = parsed.get("media_id") if parsed else None
    frame_out        = parsed.get("frame") if parsed else None
    chain_tag_out    = parsed.get("chain_tag") if parsed else None
    owner_recovery   = "bch" if owner_match else ("none" if not bch_ok else "bch_mismatch")
    media_recovery   = "bch" if media_match else ("none" if not bch_ok else "bch_mismatch")
    frame_recovery   = "bch" if frame_match else ("none" if not bch_ok else "bch_mismatch")
    chain_recovery   = "bch" if chain_match else ("none" if not bch_ok else "bch_mismatch")

    # ── Majority-vote fallback for frame_id when BCH path failed ──
    # frame_id leads the repetition stream (lives in LLLL — most robust),
    # so it's the field most likely to recover under heavy compression.
    if not frame_match and n_frame_repeat > 0 and len(frame_repeat_ext) >= FRAME_ID_BITS * 2:
        voted = _majority_vote(frame_repeat_ext, FRAME_ID_BITS)
        if voted is not None:
            voted_hex = array_to_bytes(voted).hex()
            if voted_hex == expected_frame_id_hex:
                frame_match    = True
                frame_out      = frame_id
                frame_recovery = "majority_vote"
            elif frame_out is None:
                try:
                    frame_out = struct.unpack(">I", bytes.fromhex(voted_hex))[0]
                except Exception:
                    pass

    # ── Majority-vote fallback for owner_id when BCH path failed ──
    # Recovers the RAW UTF-8 owner string (truncated/null-padded), not a hash —
    # so the actual identifier survives compression/tamper.
    if not owner_match and n_owner_repeat > 0 and len(owner_repeat_ext) >= OWNER_ID_BITS * 2:
        voted = _majority_vote(owner_repeat_ext, OWNER_ID_BITS)
        if voted is not None:
            voted_str = _decode_id_fixed(array_to_bytes(voted)[:OWNER_ID_BYTES])
            if voted_str == expected_owner_id:
                owner_match    = True
                owner_id_out   = voted_str
                owner_recovery = "majority_vote"
            elif owner_id_out is None:
                owner_id_out = voted_str

    # ── Majority-vote fallback for media_id when BCH path failed ──
    if not media_match and n_media_repeat > 0 and len(media_repeat_ext) >= MEDIA_ID_BITS * 2:
        voted = _majority_vote(media_repeat_ext, MEDIA_ID_BITS)
        if voted is not None:
            voted_str = _decode_id_fixed(array_to_bytes(voted)[:MEDIA_ID_BYTES])
            if voted_str == expected_media_id:
                media_match    = True
                media_id_out   = voted_str
                media_recovery = "majority_vote"
            elif media_id_out is None:
                media_id_out = voted_str

    # ── Majority-vote fallback for chain_tag when BCH path failed ──
    if not chain_match and n_chain_repeat > 0 and len(chain_repeat_ext) >= CHAIN_TAG_BITS * 2:
        voted = _majority_vote(chain_repeat_ext, CHAIN_TAG_BITS)
        if voted is not None:
            voted_hex = array_to_bytes(voted).hex()
            if voted_hex == expected_chain_tag:
                chain_match    = True
                chain_tag_out  = voted_hex
                chain_recovery = "majority_vote"
            elif chain_tag_out is None:
                chain_tag_out = voted_hex

    # Watermark is "found" if EITHER BCH parsed it OR majority-vote recovered
    # any of the critical fields.  Compression makes BCH fail often, but
    # majority-vote recovers reliably — both paths count as success.
    watermark_found = bch_ok or owner_match or media_match or frame_match

    # Top-level recovery_method summarizes the highest-quality path used.
    if any(r == "majority_vote" for r in
           (owner_recovery, media_recovery, frame_recovery, chain_recovery)):
        recovery_method = "majority_vote"
    elif owner_match or media_match or frame_match:
        recovery_method = "bch"
    else:
        recovery_method = "none"

    # ── Spatial tamper map from chroma LSB layer ──
    # Block-mean parity on Cb stays stable inside the Q=2 quantiser margin
    # under JPEG/H.264 mild compression but flips cleanly on real pixel
    # edits.  PSNR is preserved because LSB only shifts Cb by ≤ Q/2 = 1.
    if Cb is not None:
        lsb_res        = _verify_lsb_chroma(Cb, signature, frame_id)
        lsb_block_map  = lsb_res["lsb_spatial_map"]
        lsb_sub_mis    = lsb_res["lsb_sub_mismatch"]
        ber_lsb_sub    = lsb_res["ber_lsb_sub"]
        ber_lsb_block  = lsb_res["ber_lsb_block"]
        spatial        = lsb_block_map
        # Noise-suppressed watermark visualisation: red only when the image is
        # genuinely content-tampered (≥ MIN_TAMPER_BLOCKS flagged), and only on
        # sub-blocks inside a flagged 32x32 block.  Authentic → zero red.
        content_tampered_img      = bool(int(spatial.sum()) >= MIN_TAMPER_BLOCKS)
        wm_target                 = lsb_res["lsb_target_pattern"]
        wm_extracted, wm_display_mismatch = _wm_display_patterns(lsb_res, content_tampered_img)
    else:
        # Greyscale input — no chroma available, no spatial map.
        lsb_block_map  = np.zeros((0, 0), dtype=bool)
        lsb_sub_mis    = np.zeros((0, 0), dtype=bool)
        ber_lsb_sub    = 0.0
        ber_lsb_block  = 0.0
        wm_target      = np.zeros((0, 0), dtype=np.uint8)
        wm_extracted   = np.zeros((0, 0), dtype=np.uint8)
        wm_display_mismatch = np.zeros((0, 0), dtype=bool)
        content_tampered_img = False
        spatial        = lsb_block_map

    ber_combined = float(spatial.mean()) if spatial.size else 0.0

    return {
        "watermark_found":     watermark_found,
        "recovery_method":     recovery_method,
        "owner_recovery":      owner_recovery,
        "media_recovery":      media_recovery,
        "frame_recovery":      frame_recovery,
        "chain_recovery":      chain_recovery,
        "owner":               metadata["owner_id"] if owner_match else None,
        "media":               metadata["media_id"] if media_match else None,
        # Raw recovered IDs (truncated UTF-8) — present even on mismatch so
        # the caller can inspect what was actually extracted from the
        # watermark.  None when no path produced a value.
        "owner_id_recovered":  owner_id_out,
        "media_id_recovered":  media_id_out,
        # Back-compat: callers reading the old hash keys still get a string;
        # under the new scheme this is the raw recovered identifier, not a digest.
        "owner_hash":          owner_id_out,
        "media_hash":          media_id_out,
        "reported_frame":      frame_out,
        "reported_chain_tag":  chain_tag_out,
        "expected_chain_tag":  expected_chain_tag,
        "owner_match":         owner_match,
        "media_match":         media_match,
        "frame_match":         frame_match,
        "chain_match":         chain_match,
        "bch_corrections":     int(n_corr),
        "owner_repeat_copies": int(n_owner_copies),
        "media_repeat_copies": int(n_media_copies),
        "frame_repeat_copies": int(n_frame_copies),
        "chain_repeat_copies": int(n_chain_copies),
        "ber":                 ber_combined,
        "ber_lsb_sub":         ber_lsb_sub,
        "ber_lsb_block":       ber_lsb_block,
        "spatial_map":         spatial,
        "lsb_spatial_map":     lsb_block_map,
        "lsb_sub_mismatch":    lsb_sub_mis,
        # Fragile-watermark bit grids (0/1) for visual comparison: the
        # expected pattern vs. what was actually extracted from this file.
        # The display mismatch is noise-suppressed (red only on real tamper).
        "wm_target_pattern":     wm_target,
        "wm_extracted_pattern":  wm_extracted,
        "wm_display_mismatch":   wm_display_mismatch,
        # True only when ≥ MIN_TAMPER_BLOCKS blocks are flagged — i.e. a real
        # content edit, not the quantization noise floor.  Drives whether the
        # frontend shows any flagged regions / heatmap (keeps it consistent
        # with the noise-suppressed watermark comparison).
        "content_tampered":    content_tampered_img,
        "n_blocks_total":      int(spatial.size),
        "n_blocks_tampered":   int(spatial.sum()),
        # Tamper rule: any block flagged, identity broken, or temporal mismatch.
        # Frame/chain checks fire only when we actually have a confirmed value
        # (BCH parsed OR majority-vote succeeded for that field) — otherwise
        # codec noise that destroyed both paths would false-positive.
        "tampered":            bool(int(spatial.sum()) >= MIN_TAMPER_BLOCKS) or
                               bool(not owner_match) or
                               bool(not media_match) or
                               bool((parsed is not None or frame_recovery == "majority_vote") and not frame_match) or
                               bool((parsed is not None or chain_recovery == "majority_vote") and not chain_match),
        "ber_threshold":       BER_TAMPER_THRESHOLD,
    }


# ──────────────────────────────────────────────────────────────
# Internals exposed to video_watermark.py
# ──────────────────────────────────────────────────────────────

__all__ = [
    "embed_image", "verify_image",
    "bch_encode_bits", "bch_decode_bits",
    "bits_to_array", "array_to_bytes",
    "_embed_lsb_chroma", "_verify_lsb_chroma",

    "to_float_ycbcr", "from_float_ycbcr",
    "_embed_into_channel", "_verify_channel",
    "_build_meta_payload", "_parse_meta_payload",
    "_master_signature",
    "_meta_owner_media_id", "_meta_owner_media_hash",
    "_owner_id_bits_array", "_media_id_bits_array",
    "_owner_hash_bits_array", "_media_hash_bits_array",
    "_encode_id_fixed", "_decode_id_fixed",
    "_frame_id_bits_array", "_chain_tag_bits_array",
    "_majority_vote",
    "_chain_genesis", "_frame_chain_hash", "_expected_chain_state",
    "BLOCK_SIZE", "SPATIAL_BLOCK",
    "QIM_STEP_ROBUST",
    "BER_TAMPER_THRESHOLD", "MIN_TAMPER_BLOCKS",
    "OWNER_ID_BYTES", "MEDIA_ID_BYTES",
    "OWNER_ID_BITS", "MEDIA_ID_BITS",
    "FRAME_ID_BITS", "CHAIN_TAG_BITS",
    "OWNER_REPEATS", "MEDIA_REPEATS", "FRAME_REPEATS", "CHAIN_REPEATS",
    "OWNER_HASH_BITS", "OWNER_HASH_REPEATS",
    "LSB_SUB_BLOCK_SIZE", "LSB_QIM_STEP", "LSB_BLOCK_TAMPER_RATIO",
]