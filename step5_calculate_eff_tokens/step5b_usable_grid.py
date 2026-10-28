#!/usr/bin/env python
"""The cumulative usable-measurement total, over a grid of CV x within cuts.

One cell = one (CV cut, within/total cut) pair, holding the SAME banking rule
step5a_usable_measurements.py --cumulative uses, per gene:

    k_min(g) = the smallest k with median_CV <= cv AND within/total <= w
    cell     = sum over genes of n_metacells(k_min(g))

so a gene is counted once, at the finest rung where it qualifies, and valued at
that rung's metacell count; a gene that never qualifies adds 0.  A gene that
passes at one rung and fails at the next (the within/total criterion tightens
with k) is still banked at its first passing rung.

within/total is the UNWEIGHTED law-of-total-variance split (w_c = 1,
share_between1), as in step5a, so the curves must come from
cv_ltv_weighting_compare.py (*_wcmp*).

Cells are vector rectangles, not imshow: matplotlib writes an imshow grid into a
PDF as a 4-bit INDEXED image, which some viewers render washed out once the
figure is embedded in a document.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
matplotlib.use("Agg")

# The palette of step4_apply_threshold_and_characterize/step4d_pass_by_decile.py, inlined rather
# than imported across stages (this script used to sit next to that one).
# Sequential, one hue, light -> dark.
RAMPS = {
    "green": ["#f3faf6", "#ddf1e6", "#c2e7d3", "#a3dabd", "#81cca5", "#5dbd8c",
              "#38b07f", "#1baf7a", "#189a6b", "#14855c", "#10704d", "#0b5a3e",
              "#084530", "#063324"],
    "blue": ["#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"],
    # positional stops: the anchor #e84b4b is pinned at 0.80 of the ramp
    "red": [(0.00, "#fdf0ef"), (0.10, "#fbdedd"), (0.20, "#f8c9c8"),
            (0.30, "#f5b2b1"), (0.40, "#f29b9a"), (0.50, "#f08685"),
            (0.60, "#ee716f"), (0.70, "#eb5d5b"), (0.80, "#e84b4b"),
            (0.90, "#c73c3c"), (1.00, "#9e2e2e")],
}
INK, INK_MUTED = "#1a1a19", "#6b6a66"

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
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
         for v in rgba[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def cumulative_series(d, diag, cv_cut, w_cut):
    """The banking rule of step5a_usable_measurements.py --cumulative.

    Returns (k, running total, genes banked at k) at every rung; the grid cell
    is the last running total, and the bar panel is the whole series -- the
    same numbers, un-collapsed."""
    passes = (d.median_CV <= cv_cut) & (d["within"] <= w_cut)
    k_min = d[passes].groupby("gene").k.min()           # first passing rung per gene
    banked = k_min.map(diag["n_pools"].astype(int))     # metacells counted per gene
    return [(int(k), int(banked[k_min <= k].sum()), int((k_min == k).sum()))
            for k in sorted(d.k.unique())]


def cumulative(d, diag, cv_cut, w_cut):
    return cumulative_series(d, diag, cv_cut, w_cut)[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--diag", required=True, help="for the metacell count per k")
    ap.add_argument("--label", default="Leiden (scGPT)")
    ap.add_argument("--cuts", default="0.05,0.15,0.25,0.35,0.45",
                    help="the same ladder on both axes")
    ap.add_argument("--cv-cuts", default="", help="override for the CV axis")
    ap.add_argument("--w-cuts", default="", help="override for the within axis")
    ap.add_argument("--cmap", choices=sorted(RAMPS), default="blue")
    ap.add_argument("--size", default="3.2,2.6", help="page W,H in inches")
    ap.add_argument("--fontsize", type=float, default=8.0)
    ap.add_argument("--cell-fontsize", type=float, default=0.0)
    ap.add_argument("--highlight", default="0.25,0.25", metavar="CV,W",
                    help="ring this cell in red; empty for none")
    ap.add_argument("--highlight-color", default="#c0392b")
    ap.add_argument("--relative", action="store_true",
                    help="colour and label each cell by its ratio to the "
                         "--highlight cell, on a diverging scale centred at 1x. "
                         "The absolute total is monotone in BOTH cutoffs by "
                         "construction, so its maximum is always the loose "
                         "corner and the map invites the reading that looser is "
                         "better.  A ratio to the chosen operating point asks "
                         "the answerable question instead: what does moving the "
                         "threshold cost or buy?")
    ap.add_argument("--nominal", type=float, default=0.0,
                    metavar="N",
                    help="nominal token count; each cell then also shows the "
                         "effective/nominal share.  For a scGPT-style corpus "
                         "over this cohort that is 5,353,450 cells x 1200 input "
                         "genes = 6.4242e9 -- i.e. what the corpus would be "
                         "worth if every cell x gene slot carried signal.")
    ap.add_argument("--ramp-max", type=float, default=1.0,
                    help="use only the first fraction of the colour ramp, so "
                         "the darkest cell lands mid-ramp and the whole map "
                         "reads lighter.  The label ink still flips on measured "
                         "luminance, so contrast stays correct.")
    ap.add_argument("--square", action="store_true",
                    help="square cells: the axes box is cut to the smaller of "
                         "its two sides and centred, so the page keeps the "
                         "requested size and the spare room becomes margin")
    ap.add_argument("--marginal", action="store_true",
                    help="cells become the fold gain from loosening the CV cut "
                         "ONE step, so the grid shows diminishing returns "
                         "rather than a quantity that is monotone by "
                         "construction.  The within steps are omitted: around "
                         "the operating point they are all 1.00-1.03x.")
    ap.add_argument("--annotate-cost", action="store_true",
                    help="a second line on each CV tick: the pooled count the "
                         "cut demands (Lambda = 1/CV^2)")
    ap.add_argument("--bars", action="store_true",
                    help="also write a 5x5 panel of the per-rung accumulation "
                         "that each grid cell collapses into one number")
    ap.add_argument("--bars-size", default="7.6,5.6")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--out-prefix", required=True)
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_prefix), exist_ok=True)

    d = pd.read_csv(A.curves)
    # within/total from the UNWEIGHTED split (w_c = 1, share_between1), as in
    # step5a; only cv_ltv_weighting_compare.py curves (*_wcmp*) carry it
    assert "share_between1" in d, \
        f"{A.curves} has no share_between1 -- use a cv_ltv_weighting_compare.py curves file"
    d["within"] = 1.0 - d["share_between1"]
    diag = pd.read_csv(A.diag).set_index("k")
    cvs = [float(v) for v in (A.cv_cuts or A.cuts).split(",")]
    ws = [float(v) for v in (A.w_cuts or A.cuts).split(",")]

    # rows: within/total cut, descending so the most permissive sits at the top
    ws_plot = sorted(ws, reverse=True)
    M = np.array([[cumulative(d, diag, cv, w) for cv in cvs] for w in ws_plot],
                 dtype=float)
    tab = pd.DataFrame(M.astype(np.int64), index=ws_plot, columns=cvs)
    tab.index.name = "within_cut"; tab.columns.name = "cv_cut"
    tab.to_csv(f"{A.out_prefix}_usable_grid.csv")
    print("cumulative usable measurements (genes x metacells, banked once)")
    print((tab / 1e6).round(1).to_string(), "\n(millions)")

    SEQ = LinearSegmentedColormap.from_list(A.cmap, RAMPS[A.cmap])
    vmax = float(M.max())
    REL = ref = None
    if A.relative:
        hcv, hw = (float(v) for v in A.highlight.split(","))
        ref = float(tab.loc[hw, hcv])
        # log2 of the ratio, so halving and doubling sit equally far from the
        # neutral midpoint; TwoSlopeNorm keeps 1x on the grey centre even though
        # the two tails have very different lengths
        with np.errstate(divide="ignore"):
            REL = np.log2(np.where(M > 0, M / ref, np.nan))
        print(f"\nratio to the {hcv:g}/{hw:g} cell ({ref/1e6:.1f} M)")
        print(pd.DataFrame(M / ref, index=tab.index,
                           columns=tab.columns).round(2).to_string())
    MARG = mcols = None
    if A.marginal:
        MARG = M[:, 1:] / np.where(M[:, :-1] > 0, M[:, :-1], np.nan)
        mcols = [f"{a:g}\u2192{b:g}" for a, b in zip(cvs[:-1], cvs[1:])]
        print("\nfold gain from loosening the CV cut one step")
        print(pd.DataFrame(MARG, index=tab.index,
                           columns=mcols).round(2).to_string())

    ylabs = [f"{w:g}" for w in ws_plot]
    cfs = A.cell_fontsize or A.fontsize * 0.85
    W, H = (float(v) for v in A.size.split(","))
    fs = A.fontsize

    def render(path, ml=None):
        if ml is None:
            ml = 0.055 + (4 * fs * 0.80 * 0.60 + 3) / 72 + (fs * 1.5) / 72
        mr, mt = 0.05, (fs * 1.5 + 3) / 72
        mb = (fs * 0.80 * 1.3 + fs * 1.45 + 5) / 72
        fig = plt.figure(figsize=(W, H))
        bw, bh = W - ml - mr, H - mb - mt
        if A.square:
            ncol = len(cvs) - 1 if A.marginal else len(cvs)
            # equal cell SIDE, not an equal box: the grid may not be square
            side = min(bw / ncol, bh / len(ws_plot))
            bw2, bh2 = side * ncol, side * len(ws_plot)
            x0, y0 = ml + (bw - bw2) / 2, mb + (bh - bh2) / 2
            ax = fig.add_axes([x0 / W, y0 / H, bw2 / W, bh2 / H])
        else:
            ax = fig.add_axes([ml / W, mb / H, bw / W, bh / H])
        if REL is not None:
            lim = float(np.nanmax(np.abs(REL)))
            DIV = LinearSegmentedColormap.from_list(
                "div", ["#9e2e2e", "#e8a0a0", "#efefed", "#9ec5f4", "#104281"])
            nrm = matplotlib.colors.TwoSlopeNorm(
                vcenter=0.0, vmin=-lim, vmax=max(float(np.nanmax(REL)), 0.05))
        G = MARG if MARG is not None else M
        gmax = float(np.nanmax(G))
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                v = G[i, j]
                if MARG is not None:
                    # gains span 1.3x to 9.6x: a linear ramp would leave every
                    # column but the first at the pale end and hide the knee
                    f_ = np.log(v) / np.log(gmax) if np.isfinite(v) else np.nan
                    rgba = (SEQ(max(f_, 0.0) * A.ramp_max) if np.isfinite(f_)
                            else (0.93, 0.93, 0.92, 1.0))
                elif REL is not None:
                    rgba = (DIV(nrm(REL[i, j])) if np.isfinite(REL[i, j])
                            else (0.93, 0.93, 0.92, 1.0))
                else:
                    rgba = SEQ((v / vmax if vmax else 0.0) * A.ramp_max)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=rgba,
                                       edgecolor="white", lw=0.8, zorder=1))
                if REL is not None and MARG is None:
                    r = v / ref if ref else float("nan")
                    txt = ("\u2014" if not np.isfinite(r) or r == 0 else
                           f"{r:.2f}\u00d7" if r < 1 else f"{r:.1f}\u00d7")
                elif MARG is not None:
                    txt = "\u2014" if not np.isfinite(v) else f"{v:.2f}\u00d7"
                else:
                    txt = f"{v/1e6:.0f}" if v >= 1e7 else f"{v/1e6:.1f}"
                ink = "white" if _lum(rgba) < 0.45 else INK
                if A.nominal and REL is None and MARG is None:
                    # value above, effective/nominal share below
                    pct = 100.0 * v / A.nominal
                    ax.text(j, i - 0.17, txt, ha="center", va="center",
                            fontsize=cfs, zorder=2, color=ink)
                    ax.text(j, i + 0.20, f"{pct:.2f}%",
                            ha="center", va="center", fontsize=cfs * 0.66,
                            zorder=2, color=ink, alpha=0.62)
                else:
                    ax.text(j, i, txt, ha="center", va="center", fontsize=cfs,
                            zorder=2, color=ink)
        if A.highlight and MARG is None:
            hcv, hw = (float(v) for v in A.highlight.split(","))
            if hcv in cvs and hw in ws_plot:
                j, i = cvs.index(hcv), ws_plot.index(hw)
                # drawn inset by half a line width so the ring sits inside the
                # cell rather than straddling its white separator
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=A.highlight_color, lw=1.6,
                                       joinstyle="miter", zorder=4))
            else:
                print(f"  --highlight {A.highlight}: not on the grid, skipped")
        ax.set_xlim(-0.5, G.shape[1] - 0.5)
        ax.set_ylim(len(ws_plot) - 0.5, -0.5)
        ax.set_xticks(range(G.shape[1]))
        if MARG is not None:
            xl = mcols
        elif A.annotate_cost:
            # CV <= c demands Lambda >= 1/c^2 counts in the pooled metacell
            xl = [f"{c:g}\n({1/c**2:.0f})" for c in cvs]
        else:
            xl = [f"{c:g}" for c in cvs]
        ax.set_xticklabels(xl, fontsize=fs * 0.80)
        ax.set_yticks(range(len(ws_plot)))
        ax.set_yticklabels(ylabs, fontsize=fs * 0.80)
        ax.set_xlabel("loosening the median CV cutoff" if MARG is not None
                      else "median CV cutoff" + ("  (pooled counts needed)"
                                                 if A.annotate_cost else ""),
                      fontsize=fs)
        ax.set_ylabel("within / total cutoff", fontsize=fs)
        hl = A.highlight.split(",")
        if MARG is not None:
            ttl = "Fold gain per CV step — diminishing returns"
        elif REL is not None:
            ttl = ("Usable measurements, relative to "
                   f"CV≤{float(hl[0]):g} / w≤{float(hl[1]):g}")
        else:
            ttl = f"Cumulative usable measurements (M) — {A.label}"
        ax.set_title(ttl, fontsize=fs, color=INK, pad=3)
        ax.tick_params(which="both", length=0, colors=INK_MUTED, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.canvas.draw()
        need = ax.get_tightbbox().x0 / fig.dpi
        if path is None:
            plt.close(fig); return ml + (0.055 - need)
        fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)
        print(f"wrote {path} at {W:g} x {H:g} in (left margin {ml:.3f} in)")
        return ml

    ml = render(None)
    tag = ("_usable_grid_marginal" if A.marginal else
           "_usable_grid_rel" if A.relative else "_usable_grid")
    render(f"{A.out_prefix}{tag}.png", ml)
    if A.pdf:
        render(f"{A.out_prefix}{tag}.pdf", ml)

    if A.bars:
        BW, BH = (float(v) for v in A.bars_size.split(","))
        ser = {(w, cv): cumulative_series(d, diag, cv, w)
               for w in ws_plot for cv in cvs}
        ks = [r[0] for r in ser[(ws_plot[0], cvs[0])]]
        x = np.arange(len(ks))
        # one shared y axis across all 25 panels: the point of the figure is
        # that the cells are comparable, and per-panel scaling would hide it
        top = max(r[1] for v in ser.values() for r in v) / 1e6 * 1.14
        fig, axes = plt.subplots(len(ws_plot), len(cvs), figsize=(BW, BH),
                                 sharex=True, sharey=True)
        for i, w in enumerate(ws_plot):
            for j, cv in enumerate(cvs):
                ax = axes[i, j]
                y = [r[1] / 1e6 for r in ser[(w, cv)]]
                hit = (A.highlight and (cv, w) ==
                       tuple(float(v) for v in A.highlight.split(",")))
                ax.bar(x, y, width=0.82,
                       color=RAMPS[A.cmap][8], lw=0, zorder=2)
                ax.text(0.97, 0.90, f"{y[-1]:,.0f}M", ha="right", va="top",
                        transform=ax.transAxes, fontsize=fs * 0.85,
                        color=INK_MUTED)
                if hit:
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color(A.highlight_color)
                        sp.set_linewidth(1.5)
                else:
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
                    for sp in ("left", "bottom"):
                        ax.spines[sp].set_color(INK_MUTED)
                        ax.spines[sp].set_linewidth(0.6)
                ax.set_ylim(0, top)
                ax.set_xticks([0, 6, len(ks) - 1])
                ax.set_xticklabels([ks[0], ks[6], ks[-1]], fontsize=fs * 0.75)
                ax.tick_params(length=2, colors=INK_MUTED, pad=1.5,
                               labelsize=fs * 0.75)
                if i == 0:
                    ax.set_title(f"CV $\\leq$ {cv:g}", fontsize=fs, color=INK,
                                 pad=3)
                if j == 0:
                    ax.set_ylabel(f"w $\\leq$ {w:g}", fontsize=fs, color=INK)
        fig.supxlabel("metacell size $k$", fontsize=fs, y=0.015)
        fig.supylabel("cumulative usable measurements (millions)",
                      fontsize=fs, x=0.012)
        fig.tight_layout(rect=(0.025, 0.03, 1, 1))
        out = f"{A.out_prefix}_usable_grid_bars.png"
        fig.savefig(out, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"wrote {out} at {BW:g} x {BH:g} in")


if __name__ == "__main__":
    main()
