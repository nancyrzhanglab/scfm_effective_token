#!/usr/bin/env python
"""Genes passing BOTH criteria -- a total-count line over the per-decile grid.

Same table as step4d_pass_by_decile.py (a gene passes at size k when its median CV
across metacells is <= --cv-cut AND its within/total share is <= --within-cut),
drawn as two stacked panels on one x axis:

    top     how many genes pass in all, at each k
    bottom  where those genes sit, by expression decile -- the cell carries the
            COUNT and the colour encodes the same count, so the two panels are
            one quantity at two granularities and the column sums of the grid
            reproduce the line exactly (asserted).

Cells are drawn as vector rectangles rather than with imshow: matplotlib writes
an imshow grid into the PDF as a 4-bit INDEXED image, which some viewers render
washed out once the figure is embedded in a document.

The page is laid out to an exact size in inches (no bbox_inches), with every
margin stated in inches and converted to figure fractions, so the width can be
matched to a sibling figure to the point.
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
from step4d_pass_by_decile import RAMPS, INK, INK_MUTED

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


def _lum(rgba):
    """relative luminance of a drawn cell, for picking the label ink"""
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
         for v in rgba[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--label", default="Leiden (scGPT)")
    ap.add_argument("--cv-cut", type=float, default=0.25)
    ap.add_argument("--within-cut", type=float, default=0.25)
    ap.add_argument("--lump-below", type=int, default=8)
    ap.add_argument("--cmap", choices=sorted(RAMPS), default="blue")
    ap.add_argument("--size", default="2.71833,2.5", help="page W,H in inches")
    ap.add_argument("--fontsize", type=float, default=7.0)
    ap.add_argument("--cell-fontsize", type=float, default=0.0)
    ap.add_argument("--line-ratio", type=float, default=0.5,
                    help="height of the line panel as a multiple of the GRID's "
                         "height.  The grid keeps square cells, so the page "
                         "height is solved from this rather than taken from "
                         "--size, whose H then only reports what was asked.")
    ap.add_argument("--solve-height", action="store_true",
                    help="ignore the H in --size and solve it instead, so the "
                         "grid cells come out square")
    ap.add_argument("--diag", default="",
                    help="cv_ltv_diag_*.csv, for the metacell count per rung")
    ap.add_argument("--metacells", action="store_true",
                    help="draw the metacell count behind the gene line on a "
                         "RIGHT-hand log axis.  The two series share no units "
                         "and span six orders of magnitude, so the second axis "
                         "is a reference backdrop, not a comparison: its spine, "
                         "ticks and line all carry one grey so the eye does not "
                         "read the crossing points as meaningful.")
    ap.add_argument("--metacell-color", default="#d2d5d9")
    ap.add_argument("--metacell-style", choices=["bar", "line"], default="bar")
    ap.add_argument("--suffix", default="_with_total",
                    help="output tag, so a variant can be written beside the "
                         "original instead of replacing it")
    ap.add_argument("--out-prefix", required=True)
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_prefix), exist_ok=True)

    # ---- the table, computed exactly as step4d_pass_by_decile.py does ---------
    d = pd.read_csv(A.curves)
    # within/total from the UNWEIGHTED split (w_c = 1, share_between1), as in
    # step5, so the genes counted here are the ones step5 banks; only
    # cv_ltv_weighting_compare.py curves (*_wcmp*) carry it
    assert "share_between1" in d, \
        f"{A.curves} has no share_between1 -- use a cv_ltv_weighting_compare.py curves file"
    d["within"] = 1.0 - d["share_between1"]
    ks = sorted(d.k.unique())
    base = d.groupby("gene")["lam_bar"].first()
    dec = pd.qcut(base.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    grp = dec.where(dec >= A.lump_below, 0)
    names = [f"1–{A.lump_below - 1}"] + [str(i) for i in range(A.lump_below, 11)]
    codes = [0] + list(range(A.lump_below, 11))
    d["grp"] = d.gene.map(grp)

    M = np.zeros((len(codes), len(ks)), int)
    for j, k in enumerate(ks):
        s = d[d.k == k]
        ok = (s.median_CV <= A.cv_cut) & (s["within"] <= A.within_cut)
        for i, c in enumerate(codes):
            M[i, j] = int((ok & (s.grp == c)).sum())
    M = M[::-1]                                       # decile 10 on top
    rows = names[::-1]
    sizes = {n: int((grp == c).sum()) for n, c in zip(names, codes)}
    total = M.sum(0)
    # the line is the grid's column sum by construction; say so and check it
    assert (total == M.sum(0)).all()

    tab = pd.DataFrame(M, index=rows, columns=ks)
    tab.index.name = "expression_decile"
    tab.loc["total"] = total
    tab.to_csv(f"{A.out_prefix}{A.suffix}.csv")
    print(f"genes per group: {sizes}")
    print(tab.to_string())

    SEQ = LinearSegmentedColormap.from_list(A.cmap, RAMPS[A.cmap])
    vmax = int(M.max())
    cfs = A.cell_fontsize or A.fontsize * 0.85

    def render(path, dpi, ml=None):
        W, H = (float(v) for v in A.size.split(","))
        fs = A.fontsize
        # margins in inches.  ml is a guess on the first pass and the measured
        # value on the second, so the grid geometry is solved against the real
        # margin rather than being stretched after the fact.
        if ml is None:
            ml = 0.055 + (3 * fs * 0.80 * 0.60 + 3) / 72 + (fs * 1.5) / 72
        # the right axis needs room for its tick labels and its own ylabel
        mr = (0.045 if not (A.metacells and A.diag)
              else 0.045 + (3 * fs * 0.80 * 0.60 + 4) / 72 + (fs * 1.45) / 72)
        mb = (fs * 0.80 * 1.25 + fs * 1.35 + 5) / 72                  # xticks + xlabel
        mt = 0.045
        gap = (fs * 0.9) / 72                                         # between panels
        band_w = W - ml - mr
        cell = band_w / len(ks)
        if A.solve_height:
            heat_h = cell * len(rows)          # square cells set the page
            line_h = A.line_ratio * heat_h
            H = mb + heat_h + gap + line_h + mt
        else:
            # the page is what was asked for; the ratio is held and the cells
            # take whatever aspect is left over
            band_h = H - mb - mt - gap
            heat_h = band_h / (1.0 + A.line_ratio)
            line_h = band_h - heat_h
        fig = plt.figure(figsize=(W, H))
        ax_h = fig.add_axes([ml / W, mb / H, band_w / W, heat_h / H])
        ax_l = fig.add_axes([ml / W, (mb + heat_h + gap) / H,
                             band_w / W, line_h / H])

        # ---- the grid, one vector rectangle per cell -----------------------
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                rgba = SEQ(v / vmax if vmax else 0.0)
                ax_h.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=rgba, edgecolor="white",
                                         lw=0.8, zorder=1))
                ax_h.text(j, i, f"{v:,}", ha="center", va="center",
                          fontsize=cfs, zorder=2,
                          color="white" if _lum(rgba) < 0.45 else INK)
        ax_h.set_xlim(-0.5, len(ks) - 0.5)
        ax_h.set_ylim(len(rows) - 0.5, -0.5)
        ax_h.set_xticks(range(len(ks)))
        ax_h.set_xticklabels(ks, fontsize=fs * 0.80)
        ax_h.set_yticks(range(len(rows)))
        ax_h.set_yticklabels(rows, fontsize=fs * 0.80)
        ax_h.set_xlabel("metacell size $k$", fontsize=fs)
        ax_h.set_ylabel("decile", fontsize=fs)
        ax_h.tick_params(which="both", length=0, colors=INK_MUTED, pad=1.5)
        for sp in ax_h.spines.values():
            sp.set_visible(False)

        # ---- the total, on the same categorical x --------------------------
        c_line = RAMPS[A.cmap][-3] if isinstance(RAMPS[A.cmap][0], str) else INK
        x = np.arange(len(ks))
        ax_l.plot(x, total, "-o", color=c_line, lw=1.3, ms=2.8,
                  markerfacecolor=c_line, clip_on=False, zorder=3)
        # no per-point numbers: the grid below already carries every count,
        # and the column sums reproduce this line exactly
        ax_l.set_xlim(-0.5, len(ks) - 0.5)
        ax_l.set_ylim(0, total.max() * 1.10)
        ax_l.set_xticks(range(len(ks))); ax_l.set_xticklabels([])
        yt = [0, int(round(total.max() / 50.0) * 50)]
        ax_l.set_yticks(yt); ax_l.set_yticklabels([f"{v:,}" for v in yt],
                                                  fontsize=fs * 0.80)
        ax_l.set_ylabel("genes", fontsize=fs)
        ax_l.tick_params(length=2, colors=INK_MUTED, pad=1.5)
        if A.metacells and A.diag:
            dg = pd.read_csv(A.diag).set_index("k")
            npool = [int(dg.loc[k, "n_pools"]) for k in ks]
            ax2 = ax_l.twinx()
            ax2.set_yscale("log")
            # a common base for the bars; on a log axis the LENGTHS are not
            # proportional to the counts, so these read as "how many, roughly"
            # rather than as a quantity to measure off the page
            ax2.set_ylim(10 ** 3, max(npool) * 3)
            if A.metacell_style == "bar":
                ax2.bar(x, npool, width=0.72, color=A.metacell_color, lw=0,
                        zorder=1)
            else:
                ax2.plot(x, npool, "-", color=A.metacell_color, lw=1.2,
                         zorder=1)
            ax2.set_ylabel("metacells", fontsize=fs, color=A.metacell_color)
            ax2.tick_params(axis="y", labelsize=fs * 0.80,
                            colors=A.metacell_color, length=2, pad=1.5)
            ax2.set_yticks([1e4, 1e6])
            ax2.set_yticklabels(["10$^4$", "10$^6$"])
            ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax2.spines[["top", "left"]].set_visible(False)
            ax2.spines["right"].set_color(A.metacell_color)
            ax2.spines["right"].set_linewidth(0.6)
            ax2.spines["bottom"].set_visible(False)
            # the gene line must sit ON TOP of the backdrop, which twinx would
            # otherwise paint over
            ax_l.set_zorder(ax2.get_zorder() + 1)
            ax_l.patch.set_visible(False)
            # ax_l keeps its own right spine unless told otherwise, and it
            # would draw in ink on top of the grey one
            ax_l.spines[["top", "right"]].set_visible(False)
        else:
            ax_l.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax_l.spines[sp].set_color(INK_MUTED)
            ax_l.spines[sp].set_linewidth(0.6)

        fig.canvas.draw()
        need = min(ax_h.get_tightbbox().x0, ax_l.get_tightbbox().x0) / fig.dpi
        if path is None:                      # measuring pass
            plt.close(fig)
            return ml + (0.055 - need)        # inches; + means it was too tight
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        print(f"wrote {path} at {W:g} x {H:g} in  (left margin {ml:.3f} in)")
        return ml

    ml = render(None, 0)                      # measure, then lay out for real
    render(f"{A.out_prefix}{A.suffix}.png", 600, ml)
    render(f"{A.out_prefix}{A.suffix}.pdf", 600, ml)


if __name__ == "__main__":
    main()
