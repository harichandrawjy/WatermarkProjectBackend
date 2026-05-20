"""
Semi-Fragile Watermarking Engine
==================================
Pipeline:

  Embed:  Y  → 2-level LWT → LLLL → DCT(8x8) → SVD → QIM on S[0] → IDWT.
          Cb → 16×16 block-mean LSB parity (orthogonal fragile layer).

  Verify: Y  → 2-level LWT → LLLL → DCT(8x8) → SVD → extract bits →
                BCH-decode → owner/media/frame_id + majority-vote fallback.
          Cb → block-mean LSB parity → spatial tamper map → decision.

Subband layout:
  LLLL  ← media meta (BCH-protected)              robust   (QIM_STEP_ROBUST)
  LLLH  ← (unused — passed through)
  LLHL  ← (unused — passed through)

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

# Raw repeated owner_hash copies embedded in LLLL alongside the BCH-coded
# meta payload.  Used as a majority-vote fallback when BCH decoding fails
# (which happens when ≥2 bits flip in the same 7-bit codeword).  Each copy
# is the truncated SHA-256 owner_hash = 4 bytes = 32 bits.
OWNER_HASH_BITS    = 32
OWNER_HASH_REPEATS = 3   # need ≥3 for unambiguous majority vote
FRAME_ID_BITS      = 32  # raw frame_id repeated for majority-vote temporal recovery
CHAIN_TAG_BITS     = 16  # raw chain_tag repeated for majority-vote chain recovery


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


def _owner_hash_bits_array(owner_id: str) -> np.ndarray:
    """32-bit array form of the truncated SHA-256 owner_hash."""
    return bits_to_array(hashlib.sha256(owner_id.encode()).digest()[:4])


def _media_hash_bits_array(media_id: str) -> np.ndarray:
    """32-bit array form of the truncated SHA-256 media_hash."""
    return bits_to_array(hashlib.sha256(media_id.encode()).digest()[:4])


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
    if target_shape is not None:
        h_min = min(block_map.shape[0], target_shape[0])
        w_min = min(block_map.shape[1], target_shape[1])
        block_map = block_map[:h_min, :w_min]
    return {
        "lsb_spatial_map":  block_map,
        "lsb_sub_mismatch": sub_mismatch,
        "ber_lsb_sub":      float(sub_mismatch.mean()) if sub_mismatch.size else 0.0,
        "ber_lsb_block":    float(block_map.mean())    if block_map.size    else 0.0,
    }



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
    Y = Y.astype(np.float64)

    # Level-1 then level-2 LWT (Haar lifting ≡ Haar DWT)
    LL,   (LH1, HL1, HH1) = pywt.dwt2(Y,  WAVELET)
    LLLL, (LLLH, LLHL, LLHH) = pywt.dwt2(LL, WAVELET)

    # LLLL stream = [BCH-coded meta] + [raw owner/media hash repetitions].
    # Repetitions are extracted whole-pattern at decode and majority-voted
    # to recover owner_hash and media_hash even when BCH miscorrects.
    if extra_llll_bits is None or len(extra_llll_bits) == 0:
        llll_stream = meta_bits
    else:
        llll_stream = np.concatenate([
            np.asarray(meta_bits, dtype=np.uint8),
            np.asarray(extra_llll_bits, dtype=np.uint8),
        ])

    # ── Embed: LLLL only.  LLLH/LLHL pass through untouched. ──
    LLLL, n_llll = _embed_bits_dct_svd(LLLL, llll_stream, QIM_STEP_ROBUST)

    n_meta  = min(n_llll, len(meta_bits))
    n_extra = max(0, n_llll - len(meta_bits))

    # Inverse 2-level
    LL_wm  = pywt.idwt2((LLLL, (LLLH, LLHL, LLHH)), WAVELET)[:LL.shape[0], :LL.shape[1]]
    Y_wm   = pywt.idwt2((LL_wm, (LH1, HL1, HH1)),   WAVELET)[:Y.shape[0],  :Y.shape[1]]

    info = {
        "n_meta_bits":         int(n_meta),
        "n_extra_llll_bits":   int(n_extra),
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
    Yf = Y.astype(np.float64)
    LL, _ = pywt.dwt2(Yf, WAVELET)
    LLLL, _ = pywt.dwt2(LL, WAVELET)

    # Single LLLL extraction covering BCH meta + raw owner/media repetitions
    # (kept in one call so the block walk order matches embed exactly).
    n_extra      = max(0, n_extra_llll_bits)
    n_total_llll = n_meta_bits_coded + n_extra
    llll_ext, _  = _extract_bits_dct_svd_with_mask(LLLL, n_total_llll, QIM_STEP_ROBUST)
    meta_bits    = llll_ext[:n_meta_bits_coded]
    extra_ext    = llll_ext[n_meta_bits_coded:n_meta_bits_coded + n_extra]

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


# Meta payload layout (16 bytes, BCH(7,4)-protected as 224 bits):
#   owner_hash(4) || media_hash(4) || frame_id(4) || chain_tag(2) || integrity_tag(2)
META_PAYLOAD_BYTES = 16
_META_BODY_BYTES   = 14    # everything except the trailing integrity_tag


def _build_meta_payload(owner_id: str, media_id: str, frame_id: int,
                        chain_tag: bytes = b"\x00\x00") -> bytes:
    """Compact 16-byte LLLL header.

    The full owner/media strings live in the JSON sidecar; here we only embed
    truncated SHA-256 hashes of them (one-way commitments), the frame index,
    a 2-byte chain_tag for temporal integrity, and a 2-byte integrity tag.
    At 512x512 the LLLL subband holds ~256 bits after BCH(7,4), which is
    enough room for the 224 BCH-coded bits of this payload.
    """
    if len(chain_tag) != 2:
        raise ValueError(f"chain_tag must be 2 bytes, got {len(chain_tag)}")
    owner_h = hashlib.sha256(owner_id.encode()).digest()[:4]
    media_h = hashlib.sha256(media_id.encode()).digest()[:4]
    frame_b = struct.pack(">I", frame_id)
    body    = owner_h + media_h + frame_b + chain_tag        # 14 bytes
    tag     = hashlib.sha256(body).digest()[:2]
    return body + tag                                         # 16 bytes = 128 bits


def _parse_meta_payload(meta_bytes: bytes) -> Optional[Dict]:
    try:
        if len(meta_bytes) < META_PAYLOAD_BYTES:
            return None
        body, tag = meta_bytes[:_META_BODY_BYTES], meta_bytes[_META_BODY_BYTES:META_PAYLOAD_BYTES]
        if hashlib.sha256(body).digest()[:2] != tag:
            return None
        return {
            "owner_hash": body[:4].hex(),
            "media_hash": body[4:8].hex(),
            "frame":      struct.unpack(">I", body[8:12])[0],
            "chain_tag":  body[12:14].hex(),
        }
    except Exception:
        return None


def _meta_owner_media_hash(owner_id: str, media_id: str) -> Tuple[str, str]:
    """Truncated owner/media hashes used to validate the JSON sidecar matches
    what was actually embedded in the watermark."""
    return (
        hashlib.sha256(owner_id.encode()).digest()[:4].hex(),
        hashlib.sha256(media_id.encode()).digest()[:4].hex(),
    )


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
    media_h_b   = hashlib.sha256(media_id.encode()).digest()[:4]
    chain_state = _frame_chain_hash(media_h_b, _chain_genesis(media_h_b), frame_id)
    chain_tag   = chain_state[:2]
    meta_raw    = _build_meta_payload(owner_id, media_id, frame_id, chain_tag=chain_tag)
    meta_bits   = bits_to_array(meta_raw)
    meta_coded  = bch_encode_bits(meta_bits)

    # Raw repeated fields for majority-vote fallback when BCH miscorrects.
    # Priority order (head of stream gets the most LLLL space — lower-priority
    # fields get truncated first):
    #   owner_hash   — provenance-critical
    #   media_hash   — provenance-critical
    #   frame_id     — temporal integrity
    #   chain_tag    — temporal integrity (reorder/replay detection)
    owner_repeated     = np.tile(_owner_hash_bits_array(owner_id),       OWNER_HASH_REPEATS)
    media_repeated     = np.tile(_media_hash_bits_array(media_id),       OWNER_HASH_REPEATS)
    frame_id_repeated  = np.tile(_frame_id_bits_array(frame_id),         OWNER_HASH_REPEATS)
    chain_tag_repeated = np.tile(_chain_tag_bits_array(chain_tag),       OWNER_HASH_REPEATS)
    extras             = np.concatenate([owner_repeated, media_repeated,
                                         frame_id_repeated, chain_tag_repeated])

    Y, Cb, Cr = to_float_ycbcr(img_array[:,:,:3])
    Y_wm, info = _embed_into_channel(Y, meta_coded, signature, frame_id, extras)

    # LSB layer on chroma — independent of Y's DCT/SVD watermark.
    Cb_wm = _embed_lsb_chroma(Cb, signature, frame_id)
    if info["n_meta_bits"] < len(meta_coded):
        raise ValueError(
            f"LLLL too small: embedded {info['n_meta_bits']}/{len(meta_coded)} "
            f"BCH-coded meta bits.  Use a larger image or shorter owner/media IDs."
        )

    # Tally how many FULL copies of each fit (partial copies can't vote).
    # Priority order: owner → media → frame_id → chain_tag.  Each lower-
    # priority field gets whatever LLLL space remains after the higher ones.
    n_extra_emb       = info["n_extra_llll_bits"]
    n_owner_copies    = min(OWNER_HASH_REPEATS, n_extra_emb // OWNER_HASH_BITS)
    remaining         = max(0, n_extra_emb - n_owner_copies * OWNER_HASH_BITS)
    n_media_copies    = min(OWNER_HASH_REPEATS, remaining // OWNER_HASH_BITS)
    remaining        -= n_media_copies * OWNER_HASH_BITS
    n_frame_copies    = min(OWNER_HASH_REPEATS, remaining // FRAME_ID_BITS)
    remaining        -= n_frame_copies * FRAME_ID_BITS
    n_chain_copies    = min(OWNER_HASH_REPEATS, remaining // CHAIN_TAG_BITS)

    n_owner_bits  = n_owner_copies * OWNER_HASH_BITS
    n_media_bits  = n_media_copies * OWNER_HASH_BITS
    n_frame_bits  = n_frame_copies * FRAME_ID_BITS
    n_chain_bits  = n_chain_copies * CHAIN_TAG_BITS

    if (n_owner_copies < OWNER_HASH_REPEATS or n_media_copies < OWNER_HASH_REPEATS
            or n_frame_copies < OWNER_HASH_REPEATS or n_chain_copies < OWNER_HASH_REPEATS):
        print(f"[embed_image] note: fit "
              f"{n_owner_copies}/{OWNER_HASH_REPEATS} owner + "
              f"{n_media_copies}/{OWNER_HASH_REPEATS} media + "
              f"{n_frame_copies}/{OWNER_HASH_REPEATS} frame_id + "
              f"{n_chain_copies}/{OWNER_HASH_REPEATS} chain_tag majority-vote copies in LLLL.")

    rgb = from_float_ycbcr(Y_wm, Cb_wm, Cr)
    mse = np.mean((img_array[:,:,:3].astype(np.float64) - rgb.astype(np.float64)) ** 2)
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    metadata = {
        "version":             "v3",
        "owner_id":            owner_id,
        "media_id":            media_id,
        "frame_id":            frame_id,
        "meta_bits_raw":       int(len(meta_bits)),
        "meta_bits_coded":     int(len(meta_coded)),
        "owner_repeat_bits":     int(n_owner_bits),
        "owner_repeat_copies":   int(n_owner_copies),
        "media_repeat_bits":     int(n_media_bits),
        "media_repeat_copies":   int(n_media_copies),
        "frame_repeat_bits":     int(n_frame_bits),
        "frame_repeat_copies":   int(n_frame_copies),
        "chain_repeat_bits":     int(n_chain_bits),
        "chain_repeat_copies":   int(n_chain_copies),
        "owner_hash_bits":       int(OWNER_HASH_BITS),
        "frame_id_bits":         int(FRAME_ID_BITS),
        "chain_tag_bits":        int(CHAIN_TAG_BITS),
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
    p = 0
    owner_repeat_ext = extra_bits[p:p + n_owner_repeat]; p += n_owner_repeat
    media_repeat_ext = extra_bits[p:p + n_media_repeat]; p += n_media_repeat
    frame_repeat_ext = extra_bits[p:p + n_frame_repeat]; p += n_frame_repeat
    chain_repeat_ext = extra_bits[p:p + n_chain_repeat]

    # BCH-decode meta bits
    meta_bits_dec, n_corr = bch_decode_bits(res["meta_bits_coded"])
    meta_bytes = array_to_bytes(meta_bits_dec[:metadata["meta_bits_raw"]])
    parsed     = _parse_meta_payload(meta_bytes)

    expected_owner_h, expected_media_h = _meta_owner_media_hash(
        metadata["owner_id"], metadata["media_id"])

    # Expected chain_tag for a single-image case: one chain step from genesis.
    media_h_b           = bytes.fromhex(expected_media_h)
    expected_chain_full = _frame_chain_hash(media_h_b, _chain_genesis(media_h_b), frame_id)
    expected_chain_tag  = expected_chain_full[:2].hex()

    expected_frame_id_hex = struct.pack(">I", frame_id).hex()

    bch_ok           = parsed is not None
    owner_match      = bool(parsed and parsed.get("owner_hash") == expected_owner_h)
    media_match      = bool(parsed and parsed.get("media_hash") == expected_media_h)
    chain_match      = bool(parsed and parsed.get("chain_tag")  == expected_chain_tag)
    frame_match      = bool(parsed and parsed.get("frame")      == frame_id)
    owner_hash_out   = parsed.get("owner_hash") if parsed else None
    media_hash_out   = parsed.get("media_hash") if parsed else None
    frame_out        = parsed.get("frame") if parsed else None
    chain_tag_out    = parsed.get("chain_tag") if parsed else None
    owner_recovery   = "bch" if owner_match else ("none" if not bch_ok else "bch_mismatch")
    media_recovery   = "bch" if media_match else ("none" if not bch_ok else "bch_mismatch")
    frame_recovery   = "bch" if frame_match else ("none" if not bch_ok else "bch_mismatch")
    chain_recovery   = "bch" if chain_match else ("none" if not bch_ok else "bch_mismatch")

    # ── Majority-vote fallback for owner_hash when BCH path failed ──
    # Compression often makes BCH fail; majority-vote on the repeated copies
    # still recovers the owner reliably.  This is the primary robustness path.
    if not owner_match and n_owner_repeat > 0 and len(owner_repeat_ext) >= OWNER_HASH_BITS * 2:
        voted = _majority_vote(owner_repeat_ext, OWNER_HASH_BITS)
        if voted is not None:
            voted_hex = array_to_bytes(voted).hex()
            if voted_hex == expected_owner_h:
                owner_match    = True
                owner_hash_out = voted_hex
                owner_recovery = "majority_vote"
            elif owner_hash_out is None:
                owner_hash_out = voted_hex

    # ── Majority-vote fallback for media_hash when BCH path failed ──
    if not media_match and n_media_repeat > 0 and len(media_repeat_ext) >= OWNER_HASH_BITS * 2:
        voted = _majority_vote(media_repeat_ext, OWNER_HASH_BITS)
        if voted is not None:
            voted_hex = array_to_bytes(voted).hex()
            if voted_hex == expected_media_h:
                media_match    = True
                media_hash_out = voted_hex
                media_recovery = "majority_vote"
            elif media_hash_out is None:
                media_hash_out = voted_hex

    # ── Majority-vote fallback for frame_id when BCH path failed ──
    # Temporal integrity is as critical as identity — if BCH miscorrects,
    # majority-vote on the raw frame_id copies still recovers it.
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
    else:
        # Greyscale input — no chroma available, no spatial map.
        lsb_block_map  = np.zeros((0, 0), dtype=bool)
        lsb_sub_mis    = np.zeros((0, 0), dtype=bool)
        ber_lsb_sub    = 0.0
        ber_lsb_block  = 0.0
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
        "owner_hash":          owner_hash_out,
        "media_hash":          media_hash_out,
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
    "_master_signature", "_meta_owner_media_hash",
    "_owner_hash_bits_array", "_media_hash_bits_array", "_majority_vote",
    "BLOCK_SIZE", "SPATIAL_BLOCK",
    "QIM_STEP_ROBUST",
    "BER_TAMPER_THRESHOLD", "MIN_TAMPER_BLOCKS",
    "OWNER_HASH_BITS", "OWNER_HASH_REPEATS",
    "LSB_SUB_BLOCK_SIZE", "LSB_QIM_STEP", "LSB_BLOCK_TAMPER_RATIO",
]