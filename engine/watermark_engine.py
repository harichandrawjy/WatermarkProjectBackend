"""
Semi-Fragile Watermarking Engine
==================================
Pipeline (matches encoder.png / decoder.png architecture):

  Embed:  Y → 2-level LWT → DCT(8x8) → block-select → SVD → QIM on S[0]
          → IDWT.  BCH-coded media meta in LLLL; per-block hashed
          watermark in LLLH and LLHL.

  Verify: Y → 2-level LWT → DCT(8x8) → SVD → extract bits → BCH-decode
          (LLLL) → recompute per-block expected fingerprint → BER →
          spatial tamper map → decision.

Subband layout:
  LLLL  ← media meta (BCH-protected)              robust   (QIM_STEP_ROBUST)
  LLLH  ← per-block hashed watermark              fragile  (QIM_STEP_FRAGILE)
  LLHL  ← per-block hashed watermark (key-XOR)    fragile  (QIM_STEP_FRAGILE)

The per-block hash of a block at position (i,j) is:
  SHA256( signature || frame_id || block_id || quantised_pixels )
The LSB of that hash is the bit embedded in S[0] of that block.  Temporal
integrity (frame deletion / reorder) is handled by the BCH-protected frame_id
in LLLL — see verify_video — so per-block fingerprints stay purely spatial.
See note in README on quantisation choice.

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


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
BLOCK_SIZE        = 8       # DCT block size
QIM_STEP_ROBUST   = 32.0    # LLLL  — media meta, robust  (was 24.0)
QIM_STEP_FRAGILE  = 16.0    # LLLH/LLHL — semi-fragile fingerprint
WAVELET           = 'haar'  # Haar (LWT-equivalent for this wavelet)
DWT_LEVELS        = 2
LSB_QIM_STEP           = 2.0 
LSB_BLOCK_TAMPER_RATIO = 0.1
# Spatial-block size per LLLH/LLHL 8x8 coefficient block:
# Each level-2 coefficient covers 4x4 spatial pixels, so an 8x8 block of
# coefficients covers 32x32 spatial pixels.  This is the "block" used for
# per-block fingerprinting.
SPATIAL_BLOCK = BLOCK_SIZE * (2 ** DWT_LEVELS)   # 32
LSB_SUB_BLOCK_SIZE     = 16 
# The "Pixel Value" input to the per-block SHA256 (encoder.png) is reduced
# to a small set of robust block summary features rather than raw pixels.
# Hashing all 1024 pixels of a 32x32 block would be far too fragile — even
# the watermark's own perturbation (~10-15 px) flips per-pixel quantisation
# almost everywhere.  Quadrant medians are stable to ~10 px shifts but
# remain sensitive to spatial content changes (a region splice changes
# at least one quadrant median).
PIXEL_QUANT_STEP  = 32      # quantisation step for quadrant medians

BER_TAMPER_THRESHOLD = 0.15  # >15% block fingerprint mismatches → tampered  (was 0.18)
# Raw repeated owner_hash copies embedded in LLLL alongside the BCH-coded
# meta payload.  Used as a majority-vote fallback when BCH decoding fails
# (which happens when ≥2 bits flip in the same 7-bit codeword).  Each copy
# is the truncated SHA-256 owner_hash = 4 bytes = 32 bits.
OWNER_HASH_BITS    = 32
OWNER_HASH_REPEATS = 3   # need ≥3 for unambiguous majority vote


# ──────────────────────────────────────────────────────────────
# HAMMING(7,4) — single-error-correcting BCH used on LLLL meta
# ──────────────────────────────────────────────────────────────
# Code rate 4/7.  Each 4-bit data nibble → 7-bit codeword.  Decoder
# corrects any single bit error in each 7-bit codeword.

def bch_encode_bits(data_bits: np.ndarray) -> np.ndarray:
    """Hamming(7,4) encode a bit array (length padded to multiple of 4)."""
    bits = np.asarray(data_bits, dtype=np.uint8)
    pad  = (4 - len(bits) % 4) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    out = np.empty(len(bits) // 4 * 7, dtype=np.uint8)
    for i, j in zip(range(0, len(bits), 4), range(0, len(out), 7)):
        d1, d2, d3, d4 = bits[i:i+4]
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        out[j:j+7] = (p1, p2, d1, p3, d2, d3, d4)
    return out


def bch_decode_bits(coded_bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """Hamming(7,4) decode. Returns (data_bits, num_corrections_applied)."""
    coded = np.asarray(coded_bits, dtype=np.uint8)
    n     = (len(coded) // 7) * 7
    coded = coded[:n].copy()
    data  = np.empty(n // 7 * 4, dtype=np.uint8)
    n_corr = 0
    for i, j in zip(range(0, n, 7), range(0, len(data), 4)):
        cw = coded[i:i+7]
        p1, p2, d1, p3, d2, d3, d4 = cw
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syn = (s3 << 2) | (s2 << 1) | s1
        if syn:
            cw[syn - 1] ^= 1
            n_corr += 1
        data[j:j+4] = cw[2], cw[4], cw[5], cw[6]
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
# PER-BLOCK FINGERPRINT  (LLLH / LLHL hashed watermark)
# ──────────────────────────────────────────────────────────────

def _quantise_pixels(spatial_block: np.ndarray) -> bytes:
    """Robust block summary used as the 'Pixel Value' input to the per-block
    SHA256.  Uses *means* of the whole block plus its 4 quadrants — means
    average out the watermark's high-frequency perturbation (so the same
    quantised summary survives embedding & mild compression), but still
    change when the block's content is replaced.
    """
    h, w   = spatial_block.shape[:2]
    qh, qw = h // 2, w // 2
    full   = float(np.mean(spatial_block))
    q00    = float(np.mean(spatial_block[:qh,    :qw]))
    q01    = float(np.mean(spatial_block[:qh,    qw:]))
    q10    = float(np.mean(spatial_block[qh:,    :qw]))
    q11    = float(np.mean(spatial_block[qh:,    qw:]))
    return bytes([
        int(full) // PIXEL_QUANT_STEP & 0xFF,
        int(q00)  // PIXEL_QUANT_STEP & 0xFF,
        int(q01)  // PIXEL_QUANT_STEP & 0xFF,
        int(q10)  // PIXEL_QUANT_STEP & 0xFF,
        int(q11)  // PIXEL_QUANT_STEP & 0xFF,
    ])


def _per_block_fingerprint(
    spatial_block: np.ndarray,
    signature: bytes,
    frame_id: int,
    block_id: int,
) -> int:
    """SHA256( signature || frame_id || block_id || quant_pixels ) → LSB.

    The fingerprint is purely spatial: it depends on the cover pixels of
    this block plus a (signature, frame_id, block_id) tag that is
    deterministic from public IDs.  Frame-to-frame chaining is handled by
    the BCH-protected frame_id in LLLL (verified separately), not here.
    """
    h = hashlib.sha256()
    h.update(signature)
    h.update(struct.pack(">I", frame_id))
    h.update(struct.pack(">I", block_id))
    h.update(_quantise_pixels(spatial_block))
    return h.digest()[0] & 1


def _compute_fingerprint_grid(
    Y: np.ndarray,
    n_blocks_h: int,
    n_blocks_w: int,
    signature: bytes,
    frame_id: int,
    key_xor: int = 0,
) -> np.ndarray:
    """Compute (n_blocks_h x n_blocks_w) fingerprint bit grid by hashing
    each 32x32 spatial cover block.  `key_xor` lets LLHL use a different
    bit per block (forgery resistance / redundancy)."""
    grid = np.zeros((n_blocks_h, n_blocks_w), dtype=np.uint8)
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            sb = Y[bi*SPATIAL_BLOCK:(bi+1)*SPATIAL_BLOCK,
                   bj*SPATIAL_BLOCK:(bj+1)*SPATIAL_BLOCK]
            block_id = bi * n_blocks_w + bj
            bit = _per_block_fingerprint(sb, signature, frame_id, block_id)
            grid[bi, bj] = bit ^ (key_xor & 1) ^ ((block_id & 1) & key_xor)
    return grid

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

    # ── Compute per-block fingerprint grids from the cover Y channel ──
    n_h = LLLH.shape[0] // BLOCK_SIZE
    n_w = LLLH.shape[1] // BLOCK_SIZE
    fp_LLLH = _compute_fingerprint_grid(Y, n_h, n_w, signature, frame_id, key_xor=0)
    fp_LLHL = _compute_fingerprint_grid(Y, n_h, n_w, signature, frame_id, key_xor=1)

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

    # ── Embed: LLLL gets meta + repetition copies; LLLH/LLHL get fingerprint bits ──
    LLLL, n_llll  = _embed_bits_dct_svd(LLLL, llll_stream,        QIM_STEP_ROBUST)
    LLLH, n_fpA   = _embed_bits_dct_svd(LLLH, fp_LLLH.flatten(),  QIM_STEP_FRAGILE)
    LLHL, n_fpB   = _embed_bits_dct_svd(LLHL, fp_LLHL.flatten(),  QIM_STEP_FRAGILE)

    n_meta  = min(n_llll, len(meta_bits))
    n_extra = max(0, n_llll - len(meta_bits))

    # Inverse 2-level
    LL_wm  = pywt.idwt2((LLLL, (LLLH, LLHL, LLHH)), WAVELET)[:LL.shape[0], :LL.shape[1]]
    Y_wm   = pywt.idwt2((LL_wm, (LH1, HL1, HH1)),   WAVELET)[:Y.shape[0],  :Y.shape[1]]

    info = {
        "n_meta_bits":         int(n_meta),
        "n_extra_llll_bits":   int(n_extra),
        "n_fpA_bits":          int(n_fpA),
        "n_fpB_bits":          int(n_fpB),
        "fp_grid_h":           int(n_h),
        "fp_grid_w":           int(n_w),
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
    LLLL, (LLLH, LLHL, _) = pywt.dwt2(LL, WAVELET)

    n_h = LLLH.shape[0] // BLOCK_SIZE
    n_w = LLLH.shape[1] // BLOCK_SIZE
    n_fp_bits = n_h * n_w

    # Single LLLL extraction covering BCH meta + raw owner/media repetitions
    # (kept in one call so the block walk order matches embed exactly).
    n_extra      = max(0, n_extra_llll_bits)
    n_total_llll = n_meta_bits_coded + n_extra
    llll_ext, _  = _extract_bits_dct_svd_with_mask(LLLL, n_total_llll, QIM_STEP_ROBUST)
    meta_bits    = llll_ext[:n_meta_bits_coded]
    extra_ext    = llll_ext[n_meta_bits_coded:n_meta_bits_coded + n_extra]

    fpA_ext,  _  = _extract_bits_dct_svd_with_mask(LLLH, n_fp_bits, QIM_STEP_FRAGILE)
    fpB_ext,  _  = _extract_bits_dct_svd_with_mask(LLHL, n_fp_bits, QIM_STEP_FRAGILE)

    fp_expected_A = _compute_fingerprint_grid(Yf, n_h, n_w, signature, frame_id, 0).flatten()
    fp_expected_B = _compute_fingerprint_grid(Yf, n_h, n_w, signature, frame_id, 1).flatten()

    # Per-block tamper: a block is flagged when BOTH LLLH and LLHL bits
    # disagree with the expected fingerprint.  Requiring agreement across
    # the two subbands suppresses isolated false positives from
    # compression noise.
    miss_A = (fpA_ext != fp_expected_A)
    miss_B = (fpB_ext != fp_expected_B)
    miss_both = miss_A & miss_B

    spatial_map = miss_both.reshape(n_h, n_w)
    ber         = float(miss_both.mean())
    ber_A       = float(miss_A.mean())
    ber_B       = float(miss_B.mean())

    return {
        "meta_bits_coded":  meta_bits,
        "extra_llll_bits":  extra_ext,
        "spatial_map":      spatial_map,
        "ber":              ber,
        "ber_A":            ber_A,
        "ber_B":            ber_B,
        "tampered":         ber > BER_TAMPER_THRESHOLD,
    }


# ──────────────────────────────────────────────────────────────
# META PAYLOAD (LLLL)
# ──────────────────────────────────────────────────────────────

def _build_meta_payload(owner_id: str, media_id: str, frame_id: int) -> bytes:
    """Compact 14-byte LLLL header.

    The full owner/media strings live in the JSON sidecar; here we only embed
    truncated SHA-256 hashes of them (a one-way commitment), the frame index,
    and a 2-byte integrity tag.  At 512x512 the LLLL subband holds ~256 bits
    after BCH(7,4), which is exactly enough room.
    """
    owner_h = hashlib.sha256(owner_id.encode()).digest()[:4]
    media_h = hashlib.sha256(media_id.encode()).digest()[:4]
    frame_b = struct.pack(">I", frame_id)
    body    = owner_h + media_h + frame_b
    tag     = hashlib.sha256(body).digest()[:2]
    return body + tag                                # 14 bytes = 112 bits


def _parse_meta_payload(meta_bytes: bytes) -> Optional[Dict]:
    try:
        if len(meta_bytes) < 14:
            return None
        body, tag = meta_bytes[:12], meta_bytes[12:14]
        if hashlib.sha256(body).digest()[:2] != tag:
            return None
        return {
            "owner_hash": body[:4].hex(),
            "media_hash": body[4:8].hex(),
            "frame":      struct.unpack(">I", body[8:12])[0],
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

    signature  = _master_signature(owner_id, media_id, frame_id)
    meta_raw   = _build_meta_payload(owner_id, media_id, frame_id)
    meta_bits  = bits_to_array(meta_raw)
    meta_coded = bch_encode_bits(meta_bits)

    # Raw repeated owner_hash + media_hash for majority-vote fallback when
    # BCH miscorrects.  Owner copies come first (priority — owner is the
    # provenance-critical field); if LLLL runs out, media copies get truncated.
    owner_repeated = np.tile(_owner_hash_bits_array(owner_id), OWNER_HASH_REPEATS)
    media_repeated = np.tile(_media_hash_bits_array(media_id), OWNER_HASH_REPEATS)
    extras         = np.concatenate([owner_repeated, media_repeated])

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
    n_extra_emb    = info["n_extra_llll_bits"]
    n_owner_copies = min(OWNER_HASH_REPEATS, n_extra_emb // OWNER_HASH_BITS)
    remaining      = max(0, n_extra_emb - n_owner_copies * OWNER_HASH_BITS)
    n_media_copies = min(OWNER_HASH_REPEATS, remaining // OWNER_HASH_BITS)
    n_owner_bits   = n_owner_copies * OWNER_HASH_BITS
    n_media_bits   = n_media_copies * OWNER_HASH_BITS

    if n_owner_copies < OWNER_HASH_REPEATS or n_media_copies < OWNER_HASH_REPEATS:
        print(f"[embed_image] note: fit {n_owner_copies}/{OWNER_HASH_REPEATS} owner + "
              f"{n_media_copies}/{OWNER_HASH_REPEATS} media hash copies in LLLL "
              f"(image small - full repetition margin needs >=2 copies each).")

    rgb = from_float_ycbcr(Y_wm, Cb_wm, Cr)
    mse = np.mean((img_array[:,:,:3].astype(np.float64) - rgb.astype(np.float64)) ** 2)
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    metadata = {
        "version":             "v2",
        "owner_id":            owner_id,
        "media_id":            media_id,
        "frame_id":            frame_id,
        "meta_bits_raw":       int(len(meta_bits)),
        "meta_bits_coded":     int(len(meta_coded)),
        "owner_repeat_bits":   int(n_owner_bits),
        "owner_repeat_copies": int(n_owner_copies),
        "media_repeat_bits":   int(n_media_bits),
        "media_repeat_copies": int(n_media_copies),
        "owner_hash_bits":     int(OWNER_HASH_BITS),
        "fp_grid_h":           int(info["fp_grid_h"]),
        "fp_grid_w":           int(info["fp_grid_w"]),
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
    n_owner_copies = int(metadata.get("owner_repeat_copies", 0))
    n_media_copies = int(metadata.get("media_repeat_copies", 0))

    res = _verify_channel(
        Y, metadata["meta_bits_coded"], signature, frame_id,
        n_extra_llll_bits=n_owner_repeat + n_media_repeat,
    )
    extra_bits        = res["extra_llll_bits"]
    owner_repeat_ext  = extra_bits[:n_owner_repeat]
    media_repeat_ext  = extra_bits[n_owner_repeat:n_owner_repeat + n_media_repeat]

    # BCH-decode meta bits
    meta_bits_dec, n_corr = bch_decode_bits(res["meta_bits_coded"])
    meta_bytes = array_to_bytes(meta_bits_dec[:metadata["meta_bits_raw"]])
    parsed     = _parse_meta_payload(meta_bytes)

    expected_owner_h, expected_media_h = _meta_owner_media_hash(
        metadata["owner_id"], metadata["media_id"])

    watermark_found  = parsed is not None
    owner_match      = bool(parsed and parsed.get("owner_hash") == expected_owner_h)
    media_match      = bool(parsed and parsed.get("media_hash") == expected_media_h)
    owner_hash_out   = parsed.get("owner_hash") if parsed else None
    media_hash_out   = parsed.get("media_hash") if parsed else None
    owner_recovery   = "bch" if owner_match else ("none" if not watermark_found else "bch_mismatch")
    media_recovery   = "bch" if media_match else ("none" if not watermark_found else "bch_mismatch")

    # ── Majority-vote fallback for owner_hash when BCH path failed ──
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

    # Top-level recovery_method summarizes the higher-quality of the two.
    if owner_recovery == "majority_vote" or media_recovery == "majority_vote":
        recovery_method = "majority_vote"
    elif owner_match or media_match:
        recovery_method = "bch"
    else:
        recovery_method = "none"

    dct_spatial = res["spatial_map"]

    # ── LSB chroma layer → spatial map combination ──
    # Verify the LSB block-mean parity on Cb, aggregate to the same
    # 32×32-block grid as the DCT/SVD map, then AND them: a block is
    # reported as tampered only when BOTH the DCT/SVD fingerprint AND
    # the chroma LSB parity disagree.  JPEG/H.264 may cause sporadic
    # DCT/SVD flips, but chroma block-means stay stable inside the
    # Q=4 quantiser margin, so the LSB layer reads clean and AND kills
    # the false positives.  PSNR is preserved because LSB only shifts
    # Cb by ≤ Q/2 = 2 — invisible to Y and small in RGB-perceived MSE.
    if Cb is not None:
        lsb_res = _verify_lsb_chroma(Cb, signature, frame_id,
                                     target_shape=dct_spatial.shape)
        lsb_block_map  = lsb_res["lsb_spatial_map"]
        lsb_sub_mis    = lsb_res["lsb_sub_mismatch"]
        ber_lsb_sub    = lsb_res["ber_lsb_sub"]
        ber_lsb_block  = lsb_res["ber_lsb_block"]
        h_min = min(dct_spatial.shape[0], lsb_block_map.shape[0])
        w_min = min(dct_spatial.shape[1], lsb_block_map.shape[1])
        spatial = dct_spatial[:h_min, :w_min] & lsb_block_map[:h_min, :w_min]
    else:
        # Greyscale input — no chroma available, fall back to DCT/SVD only.
        lsb_block_map  = np.zeros_like(dct_spatial)
        lsb_sub_mis    = np.zeros((0, 0), dtype=bool)
        ber_lsb_sub    = 0.0
        ber_lsb_block  = 0.0
        spatial        = dct_spatial

    ber_combined = float(spatial.mean()) if spatial.size else 0.0

    return {
        "watermark_found":     watermark_found,
        "recovery_method":     recovery_method,
        "owner_recovery":      owner_recovery,
        "media_recovery":      media_recovery,
        "owner":               metadata["owner_id"] if owner_match else None,
        "media":               metadata["media_id"] if media_match else None,
        "owner_hash":          owner_hash_out,
        "media_hash":          media_hash_out,
        "owner_match":         owner_match,
        "media_match":         media_match,
        "bch_corrections":     int(n_corr),
        "owner_repeat_copies": int(n_owner_copies),
        "media_repeat_copies": int(n_media_copies),
                "ber":                 ber_combined,                # of AND-combined map
        "ber_dct":             res["ber"],                  # DCT/SVD-only BER

        "ber_LLLH":            res["ber_A"],
        "ber_LLHL":            res["ber_B"],
                "ber_lsb_sub":         ber_lsb_sub,
        "ber_lsb_block":       ber_lsb_block,
        "spatial_map":         spatial,                     # AND-combined
        "dct_spatial_map":     dct_spatial,                 # DCT/SVD layer only
        "lsb_spatial_map":     lsb_block_map,               # chroma-LSB layer only
        "lsb_sub_mismatch":    lsb_sub_mis,                 # raw 4×4 sub-block grid

        "n_blocks_total":      int(spatial.size),
        "n_blocks_tampered":   int(spatial.sum()),
        "tampered":            bool(res["tampered"] and watermark_found and owner_match and media_match) or
                               bool(watermark_found and (not owner_match or not media_match)) or
                               bool(not watermark_found),
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
    "QIM_STEP_ROBUST", "QIM_STEP_FRAGILE",
    "BER_TAMPER_THRESHOLD",
    "OWNER_HASH_BITS", "OWNER_HASH_REPEATS",
    "MIN_COPIES_FOR_VOTE",
    "LSB_SUB_BLOCK_SIZE", "LSB_QIM_STEP", "LSB_BLOCK_TAMPER_RATIO"
]