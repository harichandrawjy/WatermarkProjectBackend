"""Robustness / fragility evaluation for the Aegis semi-fragile watermark.

Read-only with respect to the engine: this script imports `embed_image` and
`verify_image` and calls them, exactly as main.py does.  It changes nothing in
watermark_engine.py, video_watermark.py or the app.  Delete this folder and the
project is byte-for-byte unchanged.

What it measures
----------------
1. Imperceptibility   — PSNR and SSIM of the watermarked image vs the original.
2. Robustness         — does a CONTENT-PRESERVING operation (recompression,
                        noise, resize, brightness) still verify as authentic?
                        A semi-fragile scheme is supposed to survive these.
3. Fragility          — is a CONTENT-ALTERING edit (splice, blanking, copy-move)
                        detected?  It is supposed to break on these.
4. Localisation       — when an edit IS detected, does the flagged region match
                        the region actually edited?  Reported as precision /
                        recall / F1 against a ground-truth mask, which is known
                        exactly because this script performs the edits itself.
5. Threshold choice   — LSB_BLOCK_TAMPER_RATIO and MIN_TAMPER_BLOCKS once rested
                        on "tuned empirically".  The sweep re-derives the verdict
                        at other values from data verify_image already returns, so
                        the chosen pair is justified from measurements rather than
                        asserted.  MIN_TAMPER_BLOCKS was moved 6 -> 3 on this
                        evidence; see the constant's comment in the engine.

                        NOTE: the sweep counts `content_tampered` — the spatial-
                        block criterion ALONE.  The shipped verdict is that OR a
                        broken owner/media identity, so sweep detection counts
                        read lower than section 2's per-operation table.  They
                        answer different questions; do not compare them directly.

Run:  python eval/robustness_eval.py
Out:  eval/results/{attacks.csv, threshold_sweep.csv, summary.md}
"""

import csv
import hashlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND / "engine"))

from watermark_engine import (            # noqa: E402
    embed_image, verify_image,
    _aggregate_lsb_to_block_map,
    SPATIAL_BLOCK, LSB_BLOCK_TAMPER_RATIO, MIN_TAMPER_BLOCKS,
)

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

OWNER = "7nnzgbne"          # a realistic assigned short_id
MEDIA = "evaltest"
MAX_DIM = 1024              # downscale huge sources so a full run stays minutes


# ──────────────────────────────────────────────────────────────
# Quality metrics
# ──────────────────────────────────────────────────────────────

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on the luma channel, Gaussian window sigma=1.5.

    Implemented here rather than pulled from scikit-image so the evaluation
    adds no dependency to a project that does not otherwise need one.  Follows
    Wang et al. 2004: C1 = (0.01*L)^2, C2 = (0.03*L)^2 with L = 255.
    """
    def luma(x):
        x = x.astype(np.float64)
        return 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]

    x, y = luma(a), luma(b)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda v: gaussian_filter(v, sigma=1.5)          # noqa: E731

    mx, my = g(x), g(y)
    mxx, myy, mxy = mx * mx, my * my, mx * my
    sx = g(x * x) - mxx
    sy = g(y * y) - myy
    sxy = g(x * y) - mxy

    num = (2 * mxy + C1) * (2 * sxy + C2)
    den = (mxx + myy + C1) * (sx + sy + C2)
    return float(np.mean(num / den))


# ──────────────────────────────────────────────────────────────
# Attacks
#
# Each returns (image, edited_rects).  `edited_rects` is None for a
# content-preserving operation and a list of (y0, y1, x0, x1) for an edit, which
# is what makes localisation measurable: the ground truth is known exactly
# because this script is the one doing the editing.
# ──────────────────────────────────────────────────────────────

def _jpeg(a, q):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=q)
    return np.array(Image.open(io.BytesIO(b.getvalue())).convert("RGB")), None


def _png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
    return np.array(Image.open(io.BytesIO(b.getvalue())).convert("RGB")), None


def _noise(a, sigma, seed=0):
    rng = np.random.default_rng(seed)
    f = np.clip(a.astype(np.float64) + rng.normal(0, sigma, a.shape), 0, 255)
    # rint, not astype: truncation would inject a systematic -0.5 level shift
    # and be measured as noise sensitivity that is really a rounding bug.
    return np.rint(f).astype(np.uint8), None


def _resize(a, scale):
    h, w = a.shape[:2]
    im = Image.fromarray(a)
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.array(small.resize((w, h), Image.LANCZOS)), None


def _brightness(a, delta):
    return np.clip(a.astype(np.int16) + delta, 0, 255).astype(np.uint8), None


def _contrast(a, factor):
    return np.clip(np.rint(a.astype(np.float64) * factor), 0, 255).astype(np.uint8), None


def _blank(a, frac=0.14):
    """Inpainting-style edit: flatten a region to a constant."""
    o = a.copy(); h, w = a.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y0, x0 = h // 3, w // 3
    o[y0:y0 + bh, x0:x0 + bw] = 128
    return o, [(y0, y0 + bh, x0, x0 + bw)]


def _splice(a, frac=0.14):
    """Paste content from elsewhere in the same image (copy-move)."""
    o = a.copy(); h, w = a.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    sy, sx = h // 2, w // 2
    dy, dx = h // 6, w // 6
    o[dy:dy + bh, dx:dx + bw] = a[sy:sy + bh, sx:sx + bw]
    return o, [(dy, dy + bh, dx, dx + bw)]


def _small_edit(a):
    """A single 32x32 block — the smallest unit the design claims to localise."""
    o = a.copy(); h, w = a.shape[:2]
    y0, x0 = h // 2, w // 2
    o[y0:y0 + 32, x0:x0 + 32] = 255 - o[y0:y0 + 32, x0:x0 + 32]
    return o, [(y0, y0 + 32, x0, x0 + 32)]


# (label, fn, expectation)  expectation: "authentic" | "tampered"
ATTACKS = [
    ("none (control)",        lambda a: (a, None),        "authentic"),
    ("PNG re-save",           _png,                       "authentic"),
    ("JPEG q95",              lambda a: _jpeg(a, 95),     "authentic"),
    ("JPEG q90",              lambda a: _jpeg(a, 90),     "authentic"),
    ("JPEG q80",              lambda a: _jpeg(a, 80),     "authentic"),
    ("JPEG q70",              lambda a: _jpeg(a, 70),     "authentic"),
    ("JPEG q60",              lambda a: _jpeg(a, 60),     "authentic"),
    ("JPEG q50",              lambda a: _jpeg(a, 50),     "authentic"),
    ("noise sigma=1",         lambda a: _noise(a, 1.0),   "authentic"),
    ("noise sigma=3",         lambda a: _noise(a, 3.0),   "authentic"),
    ("noise sigma=5",         lambda a: _noise(a, 5.0),   "authentic"),
    ("resize 0.75x",          lambda a: _resize(a, 0.75), "authentic"),
    ("resize 0.5x",           lambda a: _resize(a, 0.5),  "authentic"),
    ("brightness +1",         lambda a: _brightness(a, 1),   "authentic"),
    ("brightness +5",         lambda a: _brightness(a, 5),   "authentic"),
    ("contrast x1.01",        lambda a: _contrast(a, 1.01),  "authentic"),
    ("splice / copy-move",    _splice,                    "tampered"),
    ("blank region",          _blank,                     "tampered"),
    ("single 32x32 edit",     _small_edit,                "tampered"),
]


# ──────────────────────────────────────────────────────────────
# Localisation ground truth
# ──────────────────────────────────────────────────────────────

def truth_mask(shape_blocks, rects, img_shape):
    """Blocks that overlap an edited rectangle.

    The spatial map is one cell per SPATIAL_BLOCK (32) pixels, so a block is
    'truly tampered' when the edit touches any of its pixels.
    """
    hb, wb = shape_blocks
    m = np.zeros((hb, wb), dtype=bool)
    for (y0, y1, x0, x1) in rects or []:
        b_y0, b_y1 = y0 // SPATIAL_BLOCK, min(hb - 1, (y1 - 1) // SPATIAL_BLOCK)
        b_x0, b_x1 = x0 // SPATIAL_BLOCK, min(wb - 1, (x1 - 1) // SPATIAL_BLOCK)
        m[b_y0:b_y1 + 1, b_x0:b_x1 + 1] = True
    return m


def localisation(pred: np.ndarray, truth: np.ndarray):
    """Precision / recall / F1 of the flagged blocks against ground truth."""
    if pred.shape != truth.shape:
        h = min(pred.shape[0], truth.shape[0]); w = min(pred.shape[1], truth.shape[1])
        pred, truth = pred[:h, :w], truth[:h, :w]
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec) else float("nan")
    return prec, rec, f1, tp, fp, fn


# ──────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────

def _already_watermarked(arr: np.ndarray) -> bool:
    """True if this image already carries a recoverable watermark.

    Sources must be clean originals. Embedding over an existing watermark makes
    every downstream number meaningless, and it is an easy mistake to make since
    the app's own storage is full of watermarked output. The check uses the same
    blind extractor /lookup uses; a clean image yields either nothing or
    undecodable bytes, so we accept only a result that decodes to printable text.
    """
    try:
        from main import _blind_extract_ids           # noqa: PLC0415
    except Exception:
        return False                                  # cannot check; do not block
    try:
        got = _blind_extract_ids(arr)
    except Exception:
        return False
    if not got:
        return False
    for v in (got.get("owner_id") or "", got.get("media_id") or ""):
        v = v.strip()
        if len(v) >= 4 and all(32 <= ord(c) < 127 for c in v):
            return True                               # decoded cleanly => watermarked
    return False


def load_sources(include_synthetic=True):
    """Load the operator-supplied test set from eval/images/.

    Three guards, each covering a way the sample can be quietly invalidated:
      - already-watermarked files are refused outright
      - byte-identical duplicates are dropped, since the same photo counted
        twice silently weights every average toward it
      - anything too small to hold a meaningful block grid is skipped

    Synthetic images are appended as a supplement, not a substitute: a flat
    gradient and a random texture bracket the extremes of image content, which
    is what shows whether behaviour depends on local detail. They are tagged so
    the summary can report real photos separately.
    """
    out, seen_hashes, rejected = [], set(), []
    folder = HERE / "images"

    for f in sorted(folder.glob("*")):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".tif", ".tiff"):
            continue
        try:
            im = Image.open(f).convert("RGB")
        except Exception as e:
            rejected.append((f.name, f"unreadable ({type(e).__name__})")); continue

        if min(im.size) < 512:
            rejected.append((f.name, f"too small ({im.size[0]}x{im.size[1]}, need 512 short side)")); continue

        if max(im.size) > MAX_DIM:
            r = MAX_DIM / max(im.size)
            im = im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS)

        # Crop to whole 32-px blocks so ground-truth rectangles map exactly.
        w = (im.size[0] // SPATIAL_BLOCK) * SPATIAL_BLOCK
        h = (im.size[1] // SPATIAL_BLOCK) * SPATIAL_BLOCK
        arr = np.array(im.crop((0, 0, w, h)))

        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest in seen_hashes:
            rejected.append((f.name, "duplicate of an earlier image")); continue

        if _already_watermarked(arr):
            rejected.append((f.name, "ALREADY WATERMARKED - use a clean original")); continue

        seen_hashes.add(digest)
        out.append((f.stem[:20], arr, "photo"))

    if rejected:
        print("rejected sources:")
        for name, why in rejected:
            print(f"   {name:34s} {why}")
        print()

    if include_synthetic:
        rng = np.random.default_rng(7)
        gx, gy = np.meshgrid(np.linspace(0, 255, 512), np.linspace(0, 255, 512))
        out.append(("synth-gradient",
                    np.stack([gx, gy, (gx + gy) / 2], axis=-1).astype(np.uint8), "synthetic"))
        out.append(("synth-texture",
                    np.clip(rng.normal(128, 45, (512, 512, 3)), 0, 255).astype(np.uint8), "synthetic"))

    if not any(k == "photo" for _, _, k in out):
        print("No usable photographs found in eval/images/.")
        print("See eval/images/README.md for what the set should contain.")
        if not include_synthetic:
            sys.exit(1)
    return out


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    sources = load_sources()
    n_photo = sum(1 for _, _, k in sources if k == "photo")
    print(f"accepted sources: {len(sources)}  ({n_photo} photographs, "
          f"{len(sources)-n_photo} synthetic)")
    for n, a, kind in sources:
        print(f"   {n:22s} {a.shape[1]}x{a.shape[0]:<6d} {kind}")
    print(f"\nattacks per image: {len(ATTACKS)}   total verifies: {len(sources)*len(ATTACKS)}\n")

    rows, sweep_rows, quality = [], [], []
    t_start = time.time()

    for name, orig, kind in sources:
        wm, meta = embed_image(orig, OWNER, MEDIA)
        quality.append({
            "image": name, "kind": kind,
            "width": orig.shape[1], "height": orig.shape[0],
            "psnr_db": round(psnr(orig, wm), 2),
            "ssim": round(ssim(orig, wm), 5),
        })
        print(f"[{name}] embedded  PSNR {quality[-1]['psnr_db']} dB  SSIM {quality[-1]['ssim']}")

        for label, fn, expect in ATTACKS:
            attacked, rects = fn(wm)
            v = verify_image(attacked, meta)

            pred = np.asarray(v["spatial_map"], dtype=bool)
            got = "tampered" if v["tampered"] else "authentic"
            correct = (got == expect)

            prec = rec = f1 = float("nan")
            if rects:
                t = truth_mask(pred.shape, rects, attacked.shape)
                prec, rec, f1, *_ = localisation(pred, t)

            rows.append({
                "image": name, "kind": kind,
                "attack": label, "expected": expect, "verdict": got,
                "correct": int(correct),
                "owner_match": int(v["owner_match"]), "media_match": int(v["media_match"]),
                "ber_bits": round(float(v.get("ber_lsb_sub", 0.0)), 5),
                "blocks_flagged": int(v["n_blocks_tampered"]),
                "blocks_total": int(v["n_blocks_total"]),
                "loc_precision": "" if np.isnan(prec) else round(prec, 4),
                "loc_recall":    "" if np.isnan(rec) else round(rec, 4),
                "loc_f1":        "" if np.isnan(f1) else round(f1, 4),
            })

            # Threshold sweep, recomputed from the sub-block grid the verifier
            # already returned — no engine constants are modified.
            sub = np.asarray(v["lsb_sub_mismatch"], dtype=bool)
            for ratio in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
                bm = _aggregate_lsb_to_block_map(sub, ratio_threshold=ratio)
                flagged = int(bm.sum())
                for min_blocks in (1, 2, 3, 4, 6, 8, 12):
                    sweep_rows.append({
                        "image": name, "attack": label, "expected": expect,
                        "ratio": ratio, "min_blocks": min_blocks,
                        "flagged": flagged,
                        "content_tampered": int(flagged >= min_blocks),
                    })

            mark = "ok " if correct else "MISS"
            print(f"    {mark} {label:22s} {got:9s} blocks={int(v['n_blocks_tampered']):>4d}"
                  + (f"  F1={f1:.3f}" if rects and not np.isnan(f1) else ""))

    _write(rows, sweep_rows, quality)
    print(f"\ndone in {time.time()-t_start:.0f}s -> {RESULTS}")


def _write(rows, sweep_rows, quality):
    with open(RESULTS / "attacks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(RESULTS / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys())); w.writeheader(); w.writerows(sweep_rows)
    # Emitted so make_charts.py can compute the imperceptibility headline from
    # measurements. It used to hard-code PSNR/SSIM, which silently went stale
    # the moment the image set or the engine changed.
    with open(RESULTS / "quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(quality[0].keys())); w.writeheader(); w.writerows(quality)

    n_img = len(quality)
    n_photo = sum(1 for q in quality if q["kind"] == "photo")
    lines = ["# Watermark evaluation", "",
             f"{n_photo} photographs + {n_img - n_photo} synthetic images "
             f"x {len(ATTACKS)} operations = {len(rows)} verifications.", "",
             "Sources are checked before use: anything already carrying a watermark is",
             "refused, and byte-identical duplicates are dropped, so no image can be",
             "double-watermarked or silently counted twice.", "",
             "## 1. Imperceptibility", "",
             "| image | type | size | PSNR (dB) | SSIM |", "|---|---|---|---|---|"]
    for q in quality:
        lines.append(f"| {q['image']} | {q['kind']} | {q['width']}x{q['height']} | "
                     f"{q['psnr_db']} | {q['ssim']} |")
    ph = [q for q in quality if q["kind"] == "photo"]
    if ph:
        lines.append(f"| **mean (photographs)** | | | "
                     f"**{np.mean([q['psnr_db'] for q in ph]):.2f}** | "
                     f"**{np.mean([q['ssim'] for q in ph]):.5f}** |")
    lines += [f"| **mean (all)** | | | **{np.mean([q['psnr_db'] for q in quality]):.2f}** | "
              f"**{np.mean([q['ssim'] for q in quality]):.5f}** |", ""]

    lines += ["## 2. Robustness and fragility", "",
              "`correct` = the verdict matched what the operation should produce.",
              "Photographs are reported separately, since synthetic extremes are there to",
              "probe the limits rather than to represent typical input.", "",
              "| operation | should be | correct (photos) | correct (all) | mean BER | mean blocks flagged |",
              "|---|---|---|---|---|---|"]
    for label, _fn, expect in ATTACKS:
        sel = [r for r in rows if r["attack"] == label]
        selp = [r for r in sel if r["kind"] == "photo"]
        okp = sum(r["correct"] for r in selp)
        ok = sum(r["correct"] for r in sel)
        lines.append(f"| {label} | {expect} | {okp}/{len(selp)} | {ok}/{len(sel)} | "
                     f"{np.mean([r['ber_bits'] for r in sel]):.4f} | "
                     f"{np.mean([r['blocks_flagged'] for r in sel]):.1f} |")

    lines += ["", "## 3. Localisation accuracy", "",
              "Flagged blocks vs the region actually edited.", "",
              "| edit | precision | recall | F1 |", "|---|---|---|---|"]
    for label, _fn, expect in ATTACKS:
        if expect != "tampered":
            continue
        sel = [r for r in rows if r["attack"] == label and r["loc_f1"] != ""]
        if not sel:
            continue
        lines.append(f"| {label} | {np.mean([r['loc_precision'] for r in sel]):.3f} | "
                     f"{np.mean([r['loc_recall'] for r in sel]):.3f} | "
                     f"{np.mean([r['loc_f1'] for r in sel]):.3f} |")

    lines += ["", "## 4. Threshold sweep", "",
              f"Current settings: LSB_BLOCK_TAMPER_RATIO = {LSB_BLOCK_TAMPER_RATIO}, "
              f"MIN_TAMPER_BLOCKS = {MIN_TAMPER_BLOCKS}.", "",
              "Detection = content-altering edits correctly flagged. ",
              "False alarms = content-preserving operations wrongly flagged.", "",
              "| ratio | min blocks | detection | false alarms |", "|---|---|---|---|"]
    for ratio in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for mb in (1, 2, 3, 4, 6, 8, 12):
            sel = [s for s in sweep_rows if s["ratio"] == ratio and s["min_blocks"] == mb]
            tam = [s for s in sel if s["expected"] == "tampered"]
            cln = [s for s in sel if s["expected"] == "authentic"]
            det = sum(s["content_tampered"] for s in tam)
            fa = sum(s["content_tampered"] for s in cln)
            star = "  <- current" if (ratio == LSB_BLOCK_TAMPER_RATIO and mb == MIN_TAMPER_BLOCKS) else ""
            lines.append(f"| {ratio} | {mb} | {det}/{len(tam)} | {fa}/{len(cln)} |{star}")

    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
