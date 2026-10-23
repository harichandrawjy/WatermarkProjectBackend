"""
Geometric Resynchronisation
===========================
Recovers the watermark from a screenshot — or from any copy that was rescaled
or lightly cropped on its way here.

Why this module exists
----------------------
The embedder writes one bit per 8x8 block of the level-2 LL-family subbands.
A level-2 coefficient covers 4x4 pixels, so a block covers a 32x32 pixel tile,
and `_verify_channel` reads those tiles back at fixed positions in row-major
order.  Nothing in that path searches for the grid — it assumes the pixels it
is handed sit on the same lattice the embedder wrote to.

A screenshot breaks that assumption two ways at once:

  scale   the image on screen is almost never 1:1 with its own pixels (display
          DPI scaling, browser zoom, fit-to-container).  A 32px tile lands on
          32*s screenshot pixels, so every block boundary moves.
  offset  the capture rectangle does not start on a multiple of 32, so even at
          exactly 1:1 the whole grid is phase-shifted.  Measured: a ONE pixel
          crop is already enough to destroy every extracted bit.

Both are pure geometry — the watermark energy is still in the pixels.  So this
module changes nothing about what is embedded.  It searches for the transform
that puts the lattice back where the decoder expects it, then hands the
restored image to the ordinary extraction path.

Order of operations
-------------------
A candidate transform pads first and resizes second, and that order is not
arbitrary.  The crop happened in screenshot pixels, so the offset that undoes
it is a whole number of screenshot pixels; undoing the scale first would turn
it into a fractional one.  Measured on a 1.25x screenshot cropped by (3, 2):
resize-then-pad tops out at 0.425 and never decodes, because the true offset
lands at 2.4 restored pixels and no integer padding reaches it.  Pad-then-resize
scores 0.946 at exactly (3, 2) and decodes cleanly.

How the search is made affordable
---------------------------------
A full extraction attempt costs ~66 ms, and 64 ms of that is `bch_decode_bits`
— far too slow to run across a candidate space of any size.  But a correctly
aligned image has a property that is much cheaper to test than to decode: the
embedder forces S[0] of every block onto a multiple of QIM_STEP_ROBUST, so the
S[0] population clusters hard on the QIM lattice.  Misalign the grid and the
blocks are arbitrary again, so S[0] spreads out.

`lattice_score` measures that concentration with one batched SVD and no BCH, in
~1.4 ms, and the search decodes only the candidates that rank highest.

The score is RELATIVE, and that is not a detail
-----------------------------------------------
Its floor is set by image content, not by alignment: a flat or dark frame has
singular values bunched together to begin with, so even a wrong grid scores
respectably.  Measured across the eval corpus, correct geometry ranged
0.27 - 0.97 and wrong geometry 0.01 - 0.47 — ranges that OVERLAP between
images, one photo's correct 0.268 sitting below another's wrong 0.474.  So
candidates are ranked against each other within a pass; nothing is compared to
a fixed number except HIGH_CONFIDENCE, which is set where no image has produced
a false positive.

The same finding killed a cheap "is the grid even displaced?" pre-test that
once guarded this module.  Comparing an image against its own wrong-phase
samples looked principled, but measured across the corpus an aligned frame
scored as low as 0.35x its own null while a 2x upscale reached 4.39x — an
integer upscale leaves every other block boundary still on the grid.  No
threshold separates them, so callers now run the search itself and let a real
answer, or the absence of one, be the test.

The score is also blind to damage the caller may not be
-------------------------------------------------------
It measures whether the QIM lattice survived, which is not the same question as
whether the payload can be read.  On heavily textured content the lattice is
destroyed by resampling while three redundant copies still vote out the right
bits: measured, a photo whose true geometry scored 0.021 — below several wrong
candidates — validated at 0.977 agreement.  A caller whose validator is both
cheap and more sensitive than the score should therefore pass prefilter=False
and judge every candidate itself.

Why a wrong answer is not possible
----------------------------------
Whatever ranks candidates, only the validator accepts one, and every validator
used here is self-checking.  `_parse_meta_payload` validates a 2-byte SHA-256
tag over the payload body, so a blind candidate is accepted only when the
recovered bytes carry a matching digest — a 1-in-65536 accident, which the
caller's chain-tag check narrows further.  Verification, which already knows
the expected ids, instead requires the recovered bits to agree with them far
beyond chance.  Either way a bad transform yields None, never a plausible
owner id.

What a validator must NOT be is the bare majority-vote path: it carries no
integrity tag, and on a misaligned image its sections are themselves shifted —
measured returning the media id in the owner slot.  Across thousands of
candidates an unvalidated path would eventually confirm a wrong geometry, which
is worse than reporting nothing.

What it recovers, and what it does not
--------------------------------------
  rescaled 0.66x - 4x       recovered
  cropped up to 8px         recovered
  both together             recovered
  downscaled past ~0.6x     not recoverable — the QIM lattice itself is gone by
                            then (measured score 0.40, no decode even when
                            handed the exact true geometry), so this is an
                            information limit, not a search limit
  cropped by more           not recovered.  Those blocks are gone, and the
                            repetition copies that could rebuild them are only
                            readable through the unvalidated vote path.
  image inside a larger     recovered, but only by a caller that asks: use
  frame                     `content_box` to find the sub-rectangle and search
                            inside it too, keeping whichever fits better.  The
                            search itself does not trim, because a frame is not
                            a crop — the content's row stride is wrong, not
                            just its phase.

On padding a crop back
----------------------
A crop must be put back where it came from before the row-major block sequence
lines up again — but the padding itself lands inside the border blocks and
corrupts their bits.  Flat grey padding corrupts them badly enough that BCH
fails even at the true offset; replicating the edge pixels keeps those blocks
close enough to their original values that BCH still corrects them.  Measured
at the true offset, crop (5,3): grey padding fails, edge padding decodes.
"""

import math
import time
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pywt
from PIL import Image

from watermark_engine import (
    META_PAYLOAD_BYTES,
    QIM_STEP_ROBUST,
    WAVELET,
    _BCH_K,
    _BCH_N,
    _tile_blocks,
    to_float_ycbcr,
)

# Blocks the BCH-coded meta payload occupies, counted the same way
# `_blind_extract_ids` does: one bit per block, so this is also a bit count.
# Every watermarked image carries at least these, whatever its size.
_META_RAW_BITS    = META_PAYLOAD_BYTES * 8
_META_CODED_BITS  = ((((_BCH_K - _META_RAW_BITS % _BCH_K) % _BCH_K)
                      + _META_RAW_BITS) // _BCH_K) * _BCH_N


# ──────────────────────────────────────────────────────────────
# TUNING
# ──────────────────────────────────────────────────────────────
# The score is a RELATIVE signal, not an absolute one, because its floor is set
# by image content: a flat or dark frame has singular values bunched together
# to begin with, so even a wrong grid scores respectably.  Measured across the
# eval corpus, correct geometry ranges 0.27 - 0.97 while wrong geometry ranges
# 0.01 - 0.47 — and the ranges OVERLAP between images (one photo's correct
# 0.268 sits below another's wrong 0.474).  Any fixed cutoff therefore either
# misses real recoveries or floods the search with false decodes; an early
# version at 0.45 did both, recovering 4 of 8 corpus images.
#
# So candidates are ranked against each other within a pass instead.  Only a
# score high enough to be unambiguous on any image decodes on sight:
HIGH_CONFIDENCE = 0.80

# ...everything else competes, and only the best few of each pass are decoded.
# This is what makes the search content-independent — the true geometry has to
# out-score its neighbours, not clear a number.
DECODES_PER_PASS = 12

# Floor for entering that competition at all.  Purely a memory bound: below
# this nothing has ever decoded, so there is no point ranking it.
MIN_PLAUSIBLE = 0.10

# Side length, in restored pixels, of the window the cheap score looks at.
# Discrimination by window size (aligned : misaligned) measured 96px 4.6x,
# 256px 14.8x, 384px 35.3x.  256 still separates by an order of magnitude while
# costing roughly half of 384 — and this cost is paid by every candidate in the
# sweep, so it is what decides whether the search finishes inside the budget.
SCORE_WINDOW = 256

# Denominator bound for the window's rational scale approximation; see
# `_exact_window`.  Large enough that the residual ratio error stays far below
# a pixel across a SCORE_WINDOW-sized window, small enough that the window
# stays proportional to SCORE_WINDOW rather than to the whole upload.
_RATIO_DENOMINATOR = 512

# How far the offset search goes.  A crop is only recoverable while the edge
# padding stays small enough for BCH to correct the border blocks it disturbs;
# measured clean at 5px and unreliable by 7px without rescaling.  8 covers the
# working range without making the joint sweep quadratically expensive.
MAX_OFFSET = 8

# Scale bounds for the linear sweep, as a multiple of the screenshot's width.
# Upper: recovering a shot that was downscaled to the 0.66x floor needs a
# target 1.5x its width.  Lower: a 4x DPI upscale needs a target a quarter of
# it.  Going wider only adds candidates that could not decode anyway.
SWEEP_MIN_FACTOR = 0.25
SWEEP_MAX_FACTOR = 1.6

# Display scales worth trying before the linear sweep: Windows and macOS DPI
# steps, common browser zoom levels, and a few fit-to-container ratios.
# Ordered by how often they show up in a real screenshot.
COMMON_SCALES = (
    1.0, 1.25, 1.5, 2.0, 1.1, 1.2, 1.75, 0.9, 0.8, 0.75,
    1.33, 0.67, 3.0, 0.5, 2.5, 1.4, 1.6, 0.85, 0.7, 4.0,
)

# How many of those the joint crop+scale pass tries.  That pass multiplies the
# list by 63 offsets, so each extra scale costs 63 candidates; the head of the
# list covers the DPI and zoom steps that actually produce a cropped
# screenshot, while the tail is rare enough not to be worth the budget it would
# take away from the exhaustive sweep that follows.
JOINT_PASS_SCALES = 6

DEFAULT_TIME_BUDGET = 12.0


# ──────────────────────────────────────────────────────────────
# CHEAP ALIGNMENT SCORE
# ──────────────────────────────────────────────────────────────

def lattice_score(Y: np.ndarray, step: float = QIM_STEP_ROBUST,
                  max_blocks: int = _META_CODED_BITS) -> float:
    """How tightly the block S[0] population sits on the QIM lattice, in [0, 1].

    Mirrors the subband cascade `_verify_channel` reads from (LLLL, LLLH,
    LLHL), because the transform that aligns those three is exactly the one
    that makes the decoder's fixed block positions correct again.

    The measure is circular: each S[0] becomes an angle 2*pi*S0/step, so values
    on consecutive lattice points reinforce instead of cancelling, and the
    magnitude of the mean resultant is the concentration.  This is the first
    circular moment of the quantiser residual — scale-free, so the threshold
    does not need retuning per image.
    """
    LL, _ = pywt.dwt2(Y.astype(np.float64), WAVELET)
    LLLL, (LLLH, LLHL, _LLHH) = pywt.dwt2(LL, WAVELET)

    # Only the blocks the payload actually reaches, walked in the same
    # LLLL -> LLLH -> LLHL order `_verify_channel` reads them.
    #
    # Scoring every block instead makes the measure depend on image size, which
    # is not a property of alignment at all: the payload is a fixed ~420 blocks
    # whatever the resolution, so a large frame dilutes it with thousands of
    # untouched blocks.  Measured, the same correct geometry scored 0.97 on a
    # 512x512 image and 0.12 on a 2000x2509 one — the latter decoding perfectly
    # while ranking below candidates that were simply wrong.
    remaining = max(1, int(max_blocks))
    parts: List[np.ndarray] = []
    for sub in (LLLL, LLLH, LLHL):
        if remaining <= 0:
            break
        blocks, _, _, _, _ = _tile_blocks(sub)
        if not blocks.shape[0]:
            continue
        take = blocks[:remaining]
        remaining -= take.shape[0]
        # compute_uv=False — only the singular values matter here, and skipping
        # U/Vt is most of the cost of the batched SVD.
        parts.append(np.linalg.svd(take, compute_uv=False)[:, 0])

    if not parts:
        return 0.0
    S0 = np.concatenate(parts)
    if S0.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(2j * np.pi * S0 / step))))


def apply_transform(img_array: np.ndarray, transform: Dict,
                    max_rows: Optional[int] = None) -> np.ndarray:
    """Rebuild the restored image from a transform reported in `info`.

    Lets a caller run a cheap oracle during the search and its expensive
    analysis just once, on the winner — rather than paying for the expensive
    one on every candidate the search tries.

    `max_rows` restores only the top strip of that many rows, which is what
    makes a per-candidate oracle affordable on a large upload: the payload
    lives in the first block rows, so restoring a 2000x2509 frame to read it
    wastes most of the resample.  The strip is exact rather than approximate —
    Haar has no overlap beyond its 4x4 level-2 support, so the block rows
    inside it decompose identically to the same rows of the full image, and the
    caller is responsible for asking for enough of them that the payload never
    has to spill into the next subband.
    """
    src = np.ascontiguousarray(img_array[:, :, :3])
    dy, dx = int(transform["dy"]), int(transform["dx"])
    tw, th = int(transform["width"]), int(transform["height"])

    if not max_rows or max_rows >= th:
        return _apply(src, dy, dx, tw, th)

    padded = _pad_edge(src, dy, dx)
    ph = padded.shape[0]
    need_h, out_h = _exact_window(ph, th, max_rows)
    if need_h >= ph:
        return _resize(padded, tw, th)
    return _resize(padded[:need_h], tw, out_h)


def content_box(img_array: np.ndarray, tol: int = 12
                ) -> Optional[Tuple[int, int, int, int]]:
    """Locate the image inside a capture that caught the page around it.

    A real screenshot is rarely cropped to the pixel: it carries a strip of
    window chrome or page background, and that strip is fatal here.  It is not
    a crop the offset search can undo — the content is a sub-rectangle, so
    every block index and the row stride itself are wrong.  Measured, a 2px
    border was enough to turn a clean recovery into no match at all.

    Detection scans in from each edge while the row or column stays flat, which
    is what a solid background looks like beside a photograph.  Returns None
    when nothing was trimmed, so the caller can skip the retry.
    """
    if img_array.ndim < 3 or img_array.shape[2] < 3:
        return None
    g = np.asarray(Image.fromarray(img_array[:, :, :3]).convert("L"),
                   dtype=np.int16)
    h, w = g.shape
    top, bottom, left, right = 0, h, 0, w
    while top < bottom - 1 and np.ptp(g[top]) <= tol:
        top += 1
    while bottom > top + 1 and np.ptp(g[bottom - 1]) <= tol:
        bottom -= 1
    while left < right - 1 and np.ptp(g[:, left]) <= tol:
        left += 1
    while right > left + 1 and np.ptp(g[:, right - 1]) <= tol:
        right -= 1

    if (top, bottom, left, right) == (0, h, 0, w):
        return None
    if (bottom - top) < 64 or (right - left) < 64:
        return None
    return top, bottom, left, right


def _score_rgb(img: np.ndarray) -> float:
    if img.shape[0] < 32 or img.shape[1] < 32:
        return 0.0
    Y, _, _ = to_float_ycbcr(img[:, :, :3])
    return lattice_score(Y)


# ──────────────────────────────────────────────────────────────
# CANDIDATE TRANSFORMS
# ──────────────────────────────────────────────────────────────

def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """LANCZOS resample.  The engine's own eval uses LANCZOS for its resize
    attacks, so reconstruction stays consistent with how robustness was
    measured."""
    return np.array(Image.fromarray(img).resize((max(1, int(w)), max(1, int(h))),
                                                Image.LANCZOS))


def _pad_edge(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Put a crop back at offset (dy, dx) by replicating the edge pixels.

    Only the top-left is padded: that is where a lost offset has to be restored
    for the block grid to start in the right phase.  Whatever is missing at the
    bottom-right only costs the trailing blocks, which the payload does not
    reach.
    """
    if dy == 0 and dx == 0:
        return img
    return np.pad(img, ((dy, 0), (dx, 0), (0, 0)), mode="edge")


def _apply(shot: np.ndarray, dy: int, dx: int,
           target_w: int, target_h: int) -> np.ndarray:
    """The full candidate transform: pad in screenshot space, then rescale."""
    padded = _pad_edge(shot, dy, dx)
    if (padded.shape[1], padded.shape[0]) == (target_w, target_h):
        return padded
    return _resize(padded, target_w, target_h)


def _exact_window(src_len: int, dst_len: int, min_out: int) -> Tuple[int, int]:
    """A (source, destination) window length that resamples at the same ratio
    as src_len -> dst_len, and is at least `min_out` long.

    Scoring a corner instead of the whole frame is what keeps the search cost
    flat across upload sizes, but the corner has to be resampled at the *same*
    ratio as the full restore or the comparison is meaningless.  Rounding the
    window to a convenient size is not good enough: shot 337 -> 512 restores at
    1.5193, while its 253px corner -> 384 is 1.5178, and that 0.1% drifts the
    lattice by ~0.37px across the window.  Measured, that alone dropped a
    decodable candidate from 0.56 to 0.26 — below threshold, so the search
    discarded a geometry that would have decoded.

    Only multiples of the reduced denominator map to whole destination pixels.
    Taking that denominator exactly (src_len/gcd) is correct but degenerates to
    the whole frame whenever the two lengths are near-coprime — which, sweeping
    a width one pixel at a time, is most of them.  So the ratio is replaced by
    its best rational approximation with a bounded denominator: the error of
    such an approximation is under 1/(q*D), which across a window this size
    moves the lattice by well under a hundredth of a pixel, while keeping the
    window itself proportional to min_out instead of to the upload.
    """
    if src_len <= 0 or dst_len <= 0:
        return max(1, src_len), max(1, dst_len)

    ratio = Fraction(dst_len, src_len).limit_denominator(_RATIO_DENOMINATOR)
    p, q = ratio.numerator, ratio.denominator
    if p <= 0:
        return src_len, dst_len

    k = max(1, int(math.ceil(min_out / p)))
    need, out = q * k, p * k
    if need >= src_len:
        return src_len, dst_len
    return need, out


class _Scorer:
    """Scores candidate geometries, reusing everything that does not change.

    A naive scorer costs ~11-18 ms per candidate, which is an order of
    magnitude above the 1.4 ms the lattice score itself needs — the rest goes
    on numpy->PIL->numpy round trips and on resampling three colour channels to
    produce a Y the caller then recomputes.  Over a sweep of a thousand
    candidates that overhead is the whole time budget.

    Two things remove it.  The screenshot is reduced to luma once: the score
    only ever looks at Y, and PIL's "L" conversion is the same ITU-R 601-2
    combination `to_float_ycbcr` applies, so resampling one channel instead of
    three is both cheaper and equivalent (resampling and the luma combination
    are both linear, so their order does not matter beyond rounding).  And the
    padded image is cached per offset rather than rebuilt per size, since the
    search tries many sizes against each offset.

    Cropping is then handed to PIL's `box=` argument, which resamples a
    sub-rectangle in place — no intermediate array for the corner at all.
    """

    def __init__(self, shot: np.ndarray) -> None:
        self._luma = np.asarray(Image.fromarray(shot).convert("L"))
        self._key: Optional[Tuple[int, int]] = None
        self._img: Optional[Image.Image] = None

    def _padded(self, dy: int, dx: int) -> Image.Image:
        if self._key != (dy, dx):
            arr = (self._luma if (dy == 0 and dx == 0)
                   else np.pad(self._luma, ((dy, 0), (dx, 0)), mode="edge"))
            self._img = Image.fromarray(arr)
            self._key = (dy, dx)
        return self._img                                   # type: ignore[return-value]

    def score(self, dy: int, dx: int, target_w: int, target_h: int) -> float:
        """Score one candidate, looking only at a top-left window of it.

        Anchoring the window at the top-left keeps its block phase identical to
        the full image's, so the score means the same thing it would at full
        size — while the cost stays flat no matter how big the upload is.
        """
        sh, sw = self._luma.shape[:2]
        ph, pw = sh + dy, sw + dx
        if pw <= 0 or ph <= 0 or target_w <= 0 or target_h <= 0:
            return 0.0

        need_w, out_w = _exact_window(pw, target_w, SCORE_WINDOW)
        need_h, out_h = _exact_window(ph, target_h, SCORE_WINDOW)
        if min(need_w, need_h, out_w, out_h) < 32:
            return 0.0

        window = self._padded(dy, dx).resize(
            (out_w, out_h), Image.LANCZOS, box=(0, 0, need_w, need_h))
        return lattice_score(np.asarray(window, dtype=np.float64))


def _size_candidates(padded_shape: Tuple[int, int],
                     known_dims: Optional[Tuple[int, int]],
                     full_sweep: bool,
                     n_scales: Optional[int] = None) -> List[Tuple[int, int]]:
    """Candidate (width, height) to restore to, best first.

    The scale peak is a spike, not a slope — measured 0.947 at the true width
    and <= 0.11 two pixels either side — so this cannot be a coarse-to-fine
    descent.  Every width has to be tried on its own, which is why the ordering
    matters: the known size first, then the display scales that actually occur,
    and only then a linear sweep for the arbitrary fit-to-container case.

    `n_scales` trims COMMON_SCALES for callers that multiply this list by the
    offset grid, where the tail of rare scales costs 63 candidates apiece.
    """
    ph, pw = padded_shape
    aspect = ph / pw if pw else 1.0
    seen: set = set()
    out: List[Tuple[int, int]] = []

    def add(w, h=None) -> None:
        w = int(round(w))
        h = int(round(h if h is not None else w * aspect))
        if w >= 64 and h >= 64 and (w, h) not in seen:
            seen.add((w, h))
            out.append((w, h))

    if known_dims:
        add(known_dims[0], known_dims[1])
    add(pw, ph)                                  # identity — no rescale at all
    for s in COMMON_SCALES[:n_scales] if n_scales else COMMON_SCALES:
        add(pw / s)

    if full_sweep:
        # Fit-to-container gives arbitrary ratios, so sweep every integer width
        # in the recoverable band, outward from the shot's own width — the
        # further a guess sits from 1:1, the less likely it is.
        lo = max(64, int(pw * SWEEP_MIN_FACTOR))
        hi = int(pw * SWEEP_MAX_FACTOR)
        for delta in range(1, max(pw - lo, hi - pw) + 1):
            if pw - delta >= lo:
                add(pw - delta)
            if pw + delta <= hi:
                add(pw + delta)

    return out


# ──────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────

def resync_extract(
    img_array: np.ndarray,
    validate: Callable[[np.ndarray], Optional[Dict]],
    known_dims: Optional[Tuple[int, int]] = None,
    time_budget: float = DEFAULT_TIME_BUDGET,
    prefilter: bool = True,
) -> Tuple[Optional[Dict], Dict]:
    """Search for the geometry that makes `img_array` decodable again.

    `validate` must be self-validating — it is called on candidate images and
    must return None unless the payload passes its integrity tag.  Anything
    weaker (the majority-vote path in particular) will eventually confirm a
    wrong transform; see the module docstring.

    `prefilter` ranks candidates by lattice score and only validates the
    promising ones, which is what makes an expensive validator affordable
    across a wide search.  Turn it OFF when the validator is cheap AND more
    sensitive than the score — the two are not measured on the same thing, and
    a validator that reads through damage the lattice score cannot see will be
    starved by it.  Measured on a heavily textured photo: the true geometry
    scored 0.021, below both the cutoff and several wrong candidates, while the
    validator agreed with it at 0.977.  Prefiltering there discards the answer.

    Returns (result, info).  `result` is whatever `validate` returned for the
    winning candidate, or None.  `info` always reports what the search did, so
    a caller can tell "not watermarked" apart from "ran out of budget".
    """
    started = time.perf_counter()
    info: Dict = {
        "attempted": True,
        "candidates_scored": 0,
        "candidates_decoded": 0,
        "transform": None,
        "best_score": 0.0,
        "timed_out": False,
        "elapsed_s": 0.0,
    }

    if img_array.ndim < 3 or img_array.shape[2] < 3:
        info["attempted"] = False
        return None, info

    shot = np.ascontiguousarray(img_array[:, :, :3])
    sh, sw = shot.shape[:2]
    if sh < 64 or sw < 64:
        info["attempted"] = False
        return None, info

    def out_of_budget() -> bool:
        if (time.perf_counter() - started) >= time_budget:
            info["timed_out"] = True
            return True
        return False

    scorer = _Scorer(shot)

    def decode(dy: int, dx: int, tw: int, th: int, score: float) -> Optional[Dict]:
        info["candidates_decoded"] += 1
        result = validate(_apply(shot, dy, dx, tw, th))
        if result:
            info["transform"] = {"width": tw, "height": th, "dy": dy, "dx": dx,
                                 "score": round(score, 4)}
        return result

    # The passes run cheapest-and-likeliest first, so an ordinary screenshot
    # resolves long before the exhaustive ones are reached.  Each yields
    # (dy, dx, sizes) groups.
    offsets = [(dy, dx)
               for dy in range(MAX_OFFSET) for dx in range(MAX_OFFSET)
               if dy or dx]

    def passes():
        """Each pass is one *complete* group of candidates, not one candidate.

        The grouping matters: scores are ranked within a pass, so a pass has to
        be wide enough for a ranking to mean something.  Yielding one offset at
        a time would make every offset its own pass and every plausible score
        an immediate decode — 63 decodes of 64 ms on a flat image, whose wrong
        phases all score respectably.
        """
        # 1. Uncropped, common display scales — the usual screenshot.
        yield [(0, 0, tw, th)
               for tw, th in _size_candidates((sh, sw), known_dims, False)]

        # 2. Cropped at 1:1 — no rescale, just the lost phase.
        yield [(dy, dx, sw + dx, sh + dy) for dy, dx in offsets]

        # 3. Cropped and rescaled together, over the head of COMMON_SCALES —
        #    the DPI and zoom steps that actually produce a cropped screenshot.
        #    Ahead of the linear sweep because it is bounded (63 offsets x a
        #    handful of scales) where the sweep is not, so putting the sweep
        #    first makes every cropped-and-rescaled upload pay for it: measured
        #    end to end, 11-14s that way against 1-2s this way.  Offset is the
        #    outer loop so the scorer's padded image is built once per offset.
        yield [(dy, dx, tw, th)
               for dy, dx in offsets
               for tw, th in _size_candidates((sh + dy, sw + dx), known_dims,
                                              False, n_scales=JOINT_PASS_SCALES)]

        # 4. Uncropped, arbitrary scale — fit-to-container, which lands on
        #    ratios no fixed list can enumerate.  Last: it is the only
        #    unbounded pass, so it is the one that should absorb whatever
        #    budget the bounded passes left.
        yield [(0, 0, tw, th)
               for tw, th in _size_candidates((sh, sw), known_dims, True)]

    def finish(result: Optional[Dict]) -> Tuple[Optional[Dict], Dict]:
        info["elapsed_s"] = round(time.perf_counter() - started, 2)
        return result, info

    # Each pass scores its candidates, decoding outright only the ones no image
    # could produce by accident, and holding the rest back to compete.  The
    # runners-up are decoded best-first once the pass has established what a
    # good score looks like *for this image* — which is the whole point, since
    # the same number means different things on a flat frame and a busy one.
    for candidates in passes():
        contenders: List[Tuple[float, int, int, int, int]] = []

        for dy, dx, tw, th in candidates:
            if out_of_budget():
                return finish(None)

            if not prefilter:
                result = decode(dy, dx, tw, th, 0.0)
                if result:
                    return finish(result)
                continue

            score = scorer.score(dy, dx, tw, th)
            info["candidates_scored"] += 1
            if score > info["best_score"]:
                info["best_score"] = round(score, 4)

            if score >= HIGH_CONFIDENCE:
                result = decode(dy, dx, tw, th, score)
                if result:
                    return finish(result)
            elif score >= MIN_PLAUSIBLE:
                contenders.append((score, dy, dx, tw, th))

        contenders.sort(key=lambda c: c[0], reverse=True)
        for score, cdy, cdx, ctw, cth in contenders[:DECODES_PER_PASS]:
            if out_of_budget():
                return finish(None)
            result = decode(cdy, cdx, ctw, cth, score)
            if result:
                return finish(result)

    return finish(None)
