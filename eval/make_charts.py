"""Render the evaluation results as slide-ready SVG charts.

SVG rather than PNG because PowerPoint imports SVG as a true vector (Insert →
Pictures), so the charts stay sharp at any projector resolution and their text
stays selectable.  It also means no plotting dependency has to be added to a
project that does not otherwise need one.

Colours are the validated categorical slots 1-2 (#2a78d6 blue, #eb6834 orange)
and the fixed status steps; the pairing passes CVD separation, the normal-vision
floor and 3:1 contrast on a light surface.

Reads  eval/results/{attacks.csv, threshold_sweep.csv}
Writes eval/results/charts/*.svg
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# The shipped thresholds are read from the engine, never restated here — a
# hard-coded copy silently mislabels the chart the moment a constant moves.
sys.path.insert(0, str(HERE.parent / "engine"))
from watermark_engine import (            # noqa: E402
    LSB_BLOCK_TAMPER_RATIO, MIN_TAMPER_BLOCKS,
)

INK      = "#0b0b0b"
INK_SOFT = "#52514e"
GRID     = "#e4e3df"
SURFACE  = "#ffffff"
BLUE     = "#2a78d6"
ORANGE   = "#eb6834"
GOOD     = "#0ca30c"
CRIT     = "#d03b3b"
FONT     = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>{body}</svg>')


def _txt(x, y, s, size=13, fill=INK, anchor="start", weight="400"):
    esc = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{esc}</text>')


def load(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ──────────────────────────────────────────────────────────────
# 1. Outcome per operation — horizontal bars, split by intent
# ──────────────────────────────────────────────────────────────

def chart_outcomes(rows):
    order, meta = [], {}
    for r in rows:
        if r["attack"] not in meta:
            order.append(r["attack"]); meta[r["attack"]] = {"exp": r["expected"], "ok": 0, "n": 0}
        meta[r["attack"]]["ok"] += int(r["correct"]); meta[r["attack"]]["n"] += 1

    preserve = [a for a in order if meta[a]["exp"] == "authentic"]
    alter    = [a for a in order if meta[a]["exp"] == "tampered"]
    total    = meta[order[0]]["n"]

    row_h, gap, pad_l = 26, 8, 210
    plot_w = 470
    groups = [("Should survive — content preserving", preserve),
              ("Should be detected — content altering", alter)]
    h = 96 + sum(len(g[1]) * row_h + 46 for g in groups)
    w = pad_l + plot_w + 90

    b = [_txt(24, 34, "Verdict correctness by operation", 19, INK, weight="500"),
         _txt(24, 56, f"{total} images per operation · bar shows how many were judged correctly",
              13, INK_SOFT)]

    y = 92
    for title, items in groups:
        b.append(_txt(24, y, title, 13, INK_SOFT, weight="500"))
        y += 22
        for a in items:
            ok, n = meta[a]["ok"], meta[a]["n"]
            frac = ok / n
            bw = max(2, plot_w * frac)
            col = GOOD if frac == 1 else (ORANGE if frac >= 0.5 else CRIT)
            b.append(_txt(pad_l - 12, y + 15, a, 13, INK, anchor="end"))
            b.append(f'<rect x="{pad_l}" y="{y+4}" width="{plot_w}" height="{row_h-10}" '
                     f'rx="4" fill="{GRID}"/>')
            b.append(f'<rect x="{pad_l}" y="{y+4}" width="{bw:.1f}" height="{row_h-10}" '
                     f'rx="4" fill="{col}"/>')
            b.append(_txt(pad_l + plot_w + 12, y + 15, f"{ok}/{n}", 13,
                          INK if frac == 1 else INK_SOFT, weight="500"))
            y += row_h
        y += 24

    return _svg(w, h, "".join(b))


# ──────────────────────────────────────────────────────────────
# 2. JPEG quality — the degradation curve
# ──────────────────────────────────────────────────────────────

def chart_jpeg(rows):
    qs = [95, 90, 80, 70, 60, 50]
    vals, bers = [], []
    for q in qs:
        sel = [r for r in rows if r["attack"] == f"JPEG q{q}"]
        vals.append(sum(int(r["correct"]) for r in sel))
        bers.append(sum(float(r["ber_bits"]) for r in sel) / len(sel))
    n = len([r for r in rows if r["attack"] == "JPEG q95"])

    w, h = 720, 400
    L, R, T, B = 70, 40, 92, 62
    pw, ph = w - L - R, h - T - B

    def X(i): return L + pw * i / (len(qs) - 1)
    def Y(v): return T + ph * (1 - v / n)

    b = [_txt(24, 34, "Survival of JPEG recompression", 19, INK, weight="500"),
         _txt(24, 56, f"How many of {n} images still verified as authentic", 13, INK_SOFT)]

    for k in range(n + 1):
        if k % 2: continue
        yy = Y(k)
        b.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        b.append(_txt(L - 12, yy + 4, k, 12, INK_SOFT, anchor="end"))

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    for i, v in enumerate(vals):
        b.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="{BLUE}" '
                 f'stroke="{SURFACE}" stroke-width="2"/>')
        b.append(_txt(X(i), Y(v) - 14, v, 12, INK, anchor="middle", weight="500"))
        b.append(_txt(X(i), T + ph + 24, f"q{qs[i]}", 12, INK_SOFT, anchor="middle"))
        b.append(_txt(X(i), T + ph + 42, f"BER {bers[i]:.3f}", 10, INK_SOFT, anchor="middle"))

    b.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{INK_SOFT}" stroke-width="1"/>')
    b.append(_txt(24, h - 12, "Lower JPEG quality → more images wrongly flagged as tampered",
                  12, INK_SOFT))
    return _svg(w, h, "".join(b))


# ──────────────────────────────────────────────────────────────
# 3. Threshold trade-off
# ──────────────────────────────────────────────────────────────

def chart_threshold(sweep, ratio=LSB_BLOCK_TAMPER_RATIO):
    mbs = sorted({int(s["min_blocks"]) for s in sweep})
    det, fa, n_t, n_c = [], [], 0, 0
    for mb in mbs:
        sel = [s for s in sweep if float(s["ratio"]) == ratio and int(s["min_blocks"]) == mb]
        t = [s for s in sel if s["expected"] == "tampered"]
        c = [s for s in sel if s["expected"] == "authentic"]
        n_t, n_c = len(t), len(c)
        det.append(100 * sum(int(s["content_tampered"]) for s in t) / max(len(t), 1))
        fa.append(100 * sum(int(s["content_tampered"]) for s in c) / max(len(c), 1))

    w, h = 720, 440
    L, R, T, B = 62, 150, 116, 62
    pw, ph = w - L - R, h - T - B

    def X(i): return L + pw * i / (len(mbs) - 1)
    def Y(p): return T + ph * (1 - p / 100)

    # This chart plots `content_tampered` — the spatial-block criterion ALONE.
    # The shipped verdict is that OR a broken owner/media identity, and most
    # false alarms come from the identity half (resize/brightness/contrast
    # destroy the robust watermark while flagging zero blocks).  Unlabelled,
    # this chart reads ~5% while the end-to-end figure is 25%, and the two look
    # like they contradict each other.
    b = [_txt(24, 34, "Choosing MIN_TAMPER_BLOCKS", 19, INK, weight="500"),
         _txt(24, 56, f"At ratio {ratio} · {n_t} content-altering edits, "
                      f"{n_c} content-preserving operations", 13, INK_SOFT),
         _txt(24, 76, "Spatial-block criterion only. Excludes false alarms from the robust "
                      "identity layer,", 12, INK_SOFT),
         _txt(24, 92, "which no threshold affects — so these rates are lower than the "
                      "end-to-end verdict.", 12, INK_SOFT)]

    for p in (0, 25, 50, 75, 100):
        yy = Y(p)
        b.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        b.append(_txt(L - 10, yy + 4, f"{p}%", 12, INK_SOFT, anchor="end"))

    labels = []
    for series, col, label in ((det, BLUE, "Edits detected"), (fa, ORANGE, "False alarms")):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
        for i, v in enumerate(series):
            b.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{col}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
        labels.append([Y(series[-1]) + 4, label, col])

    # The two series can converge at the right edge (they do whenever a high
    # threshold suppresses both), which draws the legend text on top of itself.
    # Push them apart around their midpoint when they are closer than one line.
    labels.sort(key=lambda t: t[0])
    if labels[1][0] - labels[0][0] < 15:
        mid = (labels[0][0] + labels[1][0]) / 2
        labels[0][0], labels[1][0] = mid - 8, mid + 8
    for y, label, col in labels:
        b.append(_txt(L + pw + 14, y, label, 12, col, weight="500"))

    for i, mb in enumerate(mbs):
        b.append(_txt(X(i), T + ph + 24, mb, 12, INK_SOFT, anchor="middle"))
    b.append(_txt(L + pw / 2, T + ph + 48, "MIN_TAMPER_BLOCKS", 12, INK_SOFT, anchor="middle"))

    if MIN_TAMPER_BLOCKS in mbs:
        i = mbs.index(MIN_TAMPER_BLOCKS); x = X(i)
        b.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" stroke="{INK_SOFT}" '
                 f'stroke-width="1" stroke-dasharray="4 4"/>')
        b.append(_txt(x, T - 10, f"current ({MIN_TAMPER_BLOCKS})", 11, INK_SOFT, anchor="middle"))
    else:
        # Loud rather than silent: a chart with no marker looks fine and is wrong.
        raise SystemExit(
            f"MIN_TAMPER_BLOCKS={MIN_TAMPER_BLOCKS} is not in the swept values {mbs}. "
            f"Add it to the min_blocks grid in robustness_eval.py and re-run the harness.")

    b.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{INK_SOFT}" stroke-width="1"/>')
    return _svg(w, h, "".join(b))


# ──────────────────────────────────────────────────────────────
# 4. Headline figures — a stat row, not a chart
# ──────────────────────────────────────────────────────────────

def chart_stats(rows, quality):
    imgs = sorted({r["image"] for r in rows})
    loc = [r for r in rows if r["loc_precision"] not in ("", None)]
    prec = sum(float(r["loc_precision"]) for r in loc) / len(loc)
    rec = sum(float(r["loc_recall"]) for r in loc) / len(loc)

    # Averaged over every image, matching the "N images" subtitle below.
    # Photographs alone score higher; if you quote that number instead, change
    # the subtitle too, or the tile and the caption describe different samples.
    psnr = sum(float(q["psnr_db"]) for q in quality) / len(quality)
    ssim = sum(float(q["ssim"]) for q in quality) / len(quality)

    stats = [(f"{psnr:.1f} dB", "mean PSNR", "imperceptibility"),
             (f"{ssim:.3f}", "mean SSIM", "structural similarity"),
             (f"{prec:.2f}", "localisation precision", "flagged blocks that were right"),
             (f"{rec:.2f}", "localisation recall", "of the edited area found")]

    cw, gap = 250, 16
    w = len(stats) * cw + (len(stats) - 1) * gap + 48
    h = 190
    b = [_txt(24, 34, "Headline results", 19, INK, weight="500"),
         _txt(24, 56, f"{len(imgs)} images · {len(rows)} verifications", 13, INK_SOFT)]
    x = 24
    for val, label, sub in stats:
        b.append(f'<rect x="{x}" y="80" width="{cw}" height="86" rx="10" fill="#f6f6f4"/>')
        b.append(_txt(x + 18, 122, val, 30, INK, weight="500"))
        b.append(_txt(x + 18, 143, label, 13, INK, weight="500"))
        b.append(_txt(x + 18, 160, sub, 11, INK_SOFT))
        x += cw + gap
    return _svg(w, h, "".join(b))


# ──────────────────────────────────────────────────────────────
# 5. Before / after the quantiser-step fix — dumbbell comparison
# ──────────────────────────────────────────────────────────────

def chart_before_after(before_rows, after_rows, before_q, after_q):
    """Paired before/after per operation, read from two harness runs.

    Only operations that MOVED get a row; the count of unchanged ones goes in
    the subtitle rather than being silently dropped.
    """
    def agg(rows):
        d = {}
        for r in rows:
            k = r["attack"]
            d.setdefault(k, {"exp": r["expected"], "ok": 0, "n": 0})
            d[k]["ok"] += int(r["correct"]); d[k]["n"] += 1
        return d

    A, B = agg(before_rows), agg(after_rows)
    order = [k for k in A if k in B]
    moved = [k for k in order if A[k]["ok"] != B[k]["ok"]]
    unchanged = len(order) - len(moved)
    n_img = A[order[0]]["n"]

    def totals(rows):
        ed = [r for r in rows if r["expected"] == "tampered"]
        cl = [r for r in rows if r["expected"] == "authentic"]
        return (sum(int(r["correct"]) for r in rows), len(rows),
                sum(int(r["correct"]) for r in ed), len(ed),
                sum(1 for r in cl if int(r["correct"]) == 0), len(cl))

    tb, tn, eb, en, fb, fn = totals(before_rows)
    ta, _,  ea, _,  fa_, _ = totals(after_rows)

    row_h, pad_l, plot_w = 26, 200, 330
    head = 190
    w = pad_l + plot_w + 120
    h = head + len(moved) * row_h + 40

    b = [_txt(24, 34, "Effect of the quantiser-step fix", 19, INK, weight="500"),
         _txt(24, 56, f"LSB_QIM_STEP {before_q} → {after_q}, MIN_TAMPER_BLOCKS re-derived · "
                      f"{n_img} images × {len(order)} operations", 13, INK_SOFT),
         _txt(24, 74, f"{len(moved)} operations moved, {unchanged} unchanged · "
                      f"hollow = before, filled = after", 12, INK_SOFT)]

    # headline stat pairs
    x = 24
    for label, lo, hi, good_up in (("total correct", f"{tb}/{tn}", f"{ta}/{tn}", True),
                                   ("edits detected", f"{eb}/{en}", f"{ea}/{en}", True),
                                   ("false alarms", f"{fb}/{fn}", f"{fa_}/{fn}", False)):
            improved = (ta > tb) if good_up else (fa_ < fb)
            col = GOOD if improved else CRIT
            b.append(f'<rect x="{x}" y="96" width="200" height="62" rx="10" fill="#f6f6f4"/>')
            b.append(_txt(x + 16, 124, f"{lo}  →  {hi}", 19, col, weight="500"))
            b.append(_txt(x + 16, 145, label, 12, INK_SOFT))
            x += 212

    def X(v): return pad_l + plot_w * v / n_img

    y = head
    for k in moved:
        before, after = A[k]["ok"], B[k]["ok"]
        col = GOOD if after > before else CRIT
        cy = y + 11
        b.append(_txt(pad_l - 12, cy + 4, k, 13, INK, anchor="end"))
        b.append(f'<line x1="{pad_l}" y1="{cy}" x2="{pad_l+plot_w}" y2="{cy}" '
                 f'stroke="{GRID}" stroke-width="6" stroke-linecap="round"/>')
        b.append(f'<line x1="{X(before):.1f}" y1="{cy}" x2="{X(after):.1f}" y2="{cy}" '
                 f'stroke="{col}" stroke-width="6" stroke-linecap="round"/>')
        b.append(f'<circle cx="{X(before):.1f}" cy="{cy}" r="5.5" fill="{SURFACE}" '
                 f'stroke="{INK_SOFT}" stroke-width="2"/>')
        b.append(f'<circle cx="{X(after):.1f}" cy="{cy}" r="5.5" fill="{col}"/>')
        b.append(_txt(pad_l + plot_w + 14, cy + 4, f"{before} → {after}", 13, col, weight="500"))
        y += row_h

    for v in (0, n_img):
        b.append(_txt(X(v), y + 16, v, 11, INK_SOFT, anchor="middle"))
    b.append(_txt(pad_l + plot_w / 2, y + 16, f"images judged correctly (of {n_img})",
                  11, INK_SOFT, anchor="middle"))
    return _svg(w, h, "".join(b))


def main():
    rows = load("attacks.csv")
    sweep = load("threshold_sweep.csv")
    quality = load("quality.csv")
    files = {
        "1_headline.svg":   chart_stats(rows, quality),
        "2_outcomes.svg":   chart_outcomes(rows),
        "3_jpeg.svg":       chart_jpeg(rows),
        "4_threshold.svg":  chart_threshold(sweep),
    }

    # Optional: the before/after slide, only when a baseline run is kept around.
    # Skipped rather than faked if the folder is absent, so a fresh clone still
    # produces the four core charts.
    baseline = RES.parent / "results_T6_backup" / "attacks.csv"
    if baseline.exists():
        with open(baseline, encoding="utf-8") as f:
            before = list(csv.DictReader(f))
        files["5_before_after.svg"] = chart_before_after(before, rows, "2", "3")
    else:
        print(f"  (skipping 5_before_after.svg — no baseline at {baseline})")
    for name, svg in files.items():
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"  {name}  ({len(svg)//1024} KB)")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
