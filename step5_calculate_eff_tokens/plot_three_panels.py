#!/usr/bin/env python
"""task10: precision, homogeneity and pass-rate-by-decile as one three-panel row.

Nothing is recomputed here -- the panels are read straight from the CSVs the
two producing scripts already wrote (plot_precision_heatmap.py --pair and
plot_pass_by_decile.py), so the numbers cannot drift from the standalone
figures.  No colorbars: every cell carries its own number.

The page is laid out to an exact size (no bbox_inches), with the margins
derived from what the labels actually occupy in points.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pass_by_decile import RAMPS as _R2, INK, INK_MUTED   # blue, red, green
# The orange ramp used to come from plot_precision_heatmap, which is now kept
# only in step4_apply_threshold (as step4a_threshold_heatmap.py).  It was the
# single thing this file took from there -- that module's blue and red were
# overridden by _R2 anyway -- so it is inlined rather than imported across
# stages.  Light -> dark, ending on #5c220d.
_ORANGE = ["#fdf1ea", "#fbe2d2", "#f9cfb4", "#f7b894", "#f4a076", "#f18c5a",
           "#ee7845", "#eb6834", "#dd5c2a", "#c94f22", "#ad421b", "#8f3615",
           "#742b11", "#5c220d"]
# one more single-hue sequential ramp, light -> dark, ending on #007979
# keys in their original order: blue, orange, red came from the old
# _R1 import, green from _R2.  blue and red take _R2's values, as they
# did when _R2 was merged over _R1.
RAMPS = {"blue": _R2["blue"], "orange": _ORANGE, "red": _R2["red"],
         **_R2,
         "teal": ["#f2f8f8", "#e0efef", "#cce4e4", "#b3d7d7", "#99c9c9",
                  "#80bcbc", "#66afaf", "#4da1a1", "#339494", "#1f8989",
                  "#0f8181", "#007979"],
         # light -> dark ending on #eb7d00
         "amber": ["#fef9f2", "#fdefe0", "#fbe5cc", "#f9d8b3", "#f7cb99",
                   "#f5be80", "#f3b166", "#f1a44d", "#ef9733", "#ed8d1f",
                   "#ec850f", "#eb7d00"]}


def _lum(rgba):
    """relative luminance of a drawn cell, for picking the label ink"""
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
         for v in rgba[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

_SANS = next((f for f in ("Helvetica", "Nimbus Sans", "Arial")
              if any(x.name == f for x in
                     matplotlib.font_manager.fontManager.ttflist)),
             "DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": [_SANS, "DejaVu Sans"],
                     "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": _SANS, "mathtext.it": f"{_SANS}:italic",
                     "mathtext.bf": f"{_SANS}:bold", "mathtext.cal": _SANS,
                     "mathtext.sf": _SANS})


def fmt_count(v):
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    metavar="CSV::CMAP::YLABEL::TITLE::KIND",
                    help="one panel, left to right; KIND is count or frac")
    ap.add_argument("--size", default="8.3,1.2", help="page W,H in inches")
    ap.add_argument("--raster-cells", action="store_true",
                    help="draw the grid with imshow instead of one vector rectangle "
                         "per cell. matplotlib then writes a 4-bit INDEXED image "
                         "into the PDF, which some tools render washed out when the "
                         "figure is embedded, so vector cells are the default")
    ap.add_argument("--split", action="store_true",
                    help="write each panel as its own file, keeping the geometry "
                         "it has in the combined page")
    ap.add_argument("--fill", action="store_true",
                    help="like --square but the row pitch is stretched to use the "
                         "full requested HEIGHT instead of being padded, so cells "
                         "come out a little taller than wide")
    ap.add_argument("--square", action="store_true",
                    help="square cells: the page keeps the requested WIDTH, each "
                         "panel is as tall as its own row count needs, and the page "
                         "HEIGHT is solved for (the H in --size becomes a cap)")
    ap.add_argument("--fontsize", type=float, default=6.5)
    ap.add_argument("--tick-fontsize", type=float, default=0.0)
    ap.add_argument("--cell-fontsize", type=float, default=0.0)
    ap.add_argument("--frac-vmax", default="1",
                    help="top of the colour ramp for frac panels: 1 (the share "
                         "scale as such), data (the panel's own maximum, for "
                         "contrast when every value is small), or a number")
    ap.add_argument("--xlabel", default="metacell size $k$")
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--out", required=True, help="output path (.pdf and/or .png)")
    args = ap.parse_args()

    panels = []
    for spec in args.csv:
        bits = spec.split("::")
        assert len(bits) == 5, f"need CSV::CMAP::YLABEL::TITLE::KIND, got {spec}"
        path, cmap, ylab, title, kind = bits
        df = pd.read_csv(path, index_col=0)
        panels.append(dict(M=df.to_numpy(float), rows=list(df.index),
                           ks=list(df.columns), cmap=cmap, ylab=ylab,
                           title=title, kind=kind))
        print(f"{os.path.basename(path)}: {df.shape[0]} x {df.shape[1]}")

    W, H = (float(v) for v in args.size.split(","))
    fs = args.fontsize
    tfs = args.tick_fontsize or fs * 0.85
    cfs = args.cell_fontsize or fs * 0.62
    n = len(panels)

    # margins from the labels' own metrics: a y tick column plus a rotated
    # ylabel on the left of every panel, x ticks plus one xlabel underneath
    ywid = max(max(len(str(r)) for r in p["rows"]) for p in panels)
    gap_in = (ywid * tfs * 0.58 + fs * 1.3 + 4) / 72
    right_in = 2 / 72
    bot_in = (tfs * 1.25 + fs * 1.35 + 5) / 72
    top_in = 0 if args.no_titles else (fs * 1.5 + 2) / 72
    panel_w = (W - gap_in * n - right_in) / n

    def draw(ax, p):
        M = p["M"]
        if p["kind"] != "frac":
            top = M.max()
        elif args.frac_vmax == "data":
            top = M.max()
        else:
            top = float(args.frac_vmax)
        cm = LinearSegmentedColormap.from_list(p["cmap"], RAMPS[p["cmap"]])
        nr, nc = M.shape
        if args.raster_cells:
            ax.imshow(M, cmap=cm, aspect="auto", vmin=0, vmax=top)
        else:
            for i in range(nr):
                for j in range(nc):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                           facecolor=cm(M[i, j] / top if top
                                                        else 0.0)))
            ax.set_xlim(-0.5, nc - 0.5)
            ax.set_ylim(nr - 0.5, -0.5)
            ax.set_axisbelow(False)          # the white rules sit ON the cells
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = f"{v:.0%}" if p["kind"] == "frac" else fmt_count(v)
                # the ramps do not all darken at the same rate, so the label ink
                # follows the luminance of the colour actually drawn
                ax.text(j, i, txt, ha="center", va="center", fontsize=cfs,
                        color="white" if _lum(cm(v / top if top else 0)) < 0.30
                        else INK)
        ax.set_xticks(range(len(p["ks"])))
        ax.set_xticklabels(p["ks"], fontsize=tfs)
        ax.set_yticks(range(len(p["rows"])))
        ax.set_yticklabels(p["rows"], fontsize=tfs)
        ax.set_xlabel(args.xlabel, fontsize=fs)
        ax.set_ylabel(p["ylab"], fontsize=fs)
        if not args.no_titles:
            # panels sit on one baseline, so a shorter one would carry its title
            # lower than the rest; lift it to the top of the tallest panel
            tall = max(q["M"].shape[0] for q in panels)
            y = (1 + (tall - M.shape[0]) / M.shape[0]
                 if args.square or args.fill else 1.0)
            ax.set_title(p["title"], fontsize=fs, color=INK, pad=2, y=y)
        ax.set_xticks(np.arange(len(p["ks"]) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(p["rows"]) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="both", length=0, colors=INK_MUTED, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def geometry():
        """the cell pitch and vertical placement every panel shares"""
        cell = panel_w / max(len(q["ks"]) for q in panels)
        cell_h = cell
        if args.fill:
            # keep the column pitch set by the width, but stretch the row pitch
            # so the tallest panel reaches the requested height; the shorter
            # panel still uses that same row pitch, so it stays shorter
            cell_h = (H - bot_in - top_in) / max(q["M"].shape[0] for q in panels)
        need = cell_h * max(q["M"].shape[0] for q in panels) + bot_in + top_in
        # the --size height is a floor as well as a cap: square cells at this
        # width may need less than it, and the rest becomes padding split
        # evenly above and below so the drawn block sits centred on the page
        return cell, cell_h, max(0.0, H - need) / 2, max(H, need)

    def report_axes(fig, pairs, Wp, Hp):
        fig.canvas.draw()
        for ax, q in pairs:
            b = ax.get_position()
            print(f"{q['title']}: heatmap {b.width * Wp:.3f} x "
                  f"{b.height * Hp:.3f} in, cell "
                  f"{b.width * Wp / q['M'].shape[1]:.4f} x "
                  f"{b.height * Hp / q['M'].shape[0]:.4f} in")

    if args.split:
        # one file per panel, at the geometry it has inside the combined page:
        # same cell pitch, same margins, same height, so the pieces still line
        # up if they are placed beside each other later
        cell, cell_h, pad, H = geometry()
        Wp = gap_in + panel_w + right_in
        exts = [os.path.splitext(x)[1] for x in args.out.split(",")]
        stem = os.path.splitext(args.out.split(",")[0])[0]
        for q in panels:
            fig = plt.figure(figsize=(Wp, H))
            ax = fig.add_axes([gap_in / Wp, (bot_in + pad) / H,
                               panel_w / Wp, cell_h * q["M"].shape[0] / H])
            draw(ax, q)
            slug = "_".join("".join(c if c.isalnum() else " "
                                    for c in q["title"].lower()).split())
            report_axes(fig, [(ax, q)], Wp, H)
            for e in exts:
                out = f"{stem}_{slug}{e}"
                fig.savefig(out, dpi=600, facecolor="white")
                print("wrote", out, f"at {Wp:.3g} x {H:.3g} in")
            plt.close(fig)
        return

    if args.square or args.fill:
        cell, cell_h, pad, H = geometry()
        fig = plt.figure(figsize=(W, H))
        axes = [fig.add_axes([(gap_in * (i + 1) + panel_w * i) / W,
                              (bot_in + pad) / H,
                              panel_w / W, cell_h * p["M"].shape[0] / H])
                for i, p in enumerate(panels)]
        if abs(cell_h - cell) > 1e-4:
            print(f"cell {cell:.4f} x {cell_h:.4f} in "
                  f"(height / width = {cell_h / cell:.2f})")
        if pad > 0.004:
            print(f"padding to {H:g} in ({pad:.3g} in above and below)")
    else:
        fig, axes = plt.subplots(1, n, figsize=(W, H))
        axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        draw(ax, p)

    if not (args.square or args.fill):
        fig.subplots_adjust(left=gap_in / W, right=1 - right_in / W,
                            bottom=bot_in / H, top=1 - top_in / H,
                            wspace=gap_in / panel_w)
    report_axes(fig, list(zip(axes, panels)), W, H)
    for out in args.out.split(","):
        fig.savefig(out, dpi=600, facecolor="white")
        print("wrote", out, f"at {W:g} x {H:.3g} in")
    plt.close(fig)


if __name__ == "__main__":
    main()
