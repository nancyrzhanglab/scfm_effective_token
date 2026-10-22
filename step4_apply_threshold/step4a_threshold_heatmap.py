#!/usr/bin/env python
"""task10: how many genes reach a given precision at each metacell size.

Cell (t, k) = number of genes whose median CV across metacells at size k is at
or below the precision threshold t. Both axes are ordinal and the value is a
magnitude, so this is a one-hue sequential heatmap.

Genes whose CV is undefined at a given k (no metacell with a non-zero rate)
simply fail the threshold, which is the right behaviour: they are not precise.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("Agg")

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

# sequential: one hue, light -> dark. Two sequential contexts appear in this
# series, so the second takes the next categorical slot's hue (orange), each as
# its own one-hue ramp -- never a rainbow, never a hue at a midpoint.
RAMPS = {
    "blue": ["#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"],
    "orange": ["#fdf1ea", "#fbe2d2", "#f9cfb4", "#f7b894", "#f4a076", "#f18c5a",
               "#ee7845", "#eb6834", "#dd5c2a", "#c94f22", "#ad421b", "#8f3615",
               "#742b11", "#5c220d"],
    # positional stops: the anchor #e84b4b is pinned at 0.80 of the ramp
    "red": [(0.00, "#fdf0ef"), (0.10, "#fbdedd"), (0.20, "#f8c9c8"),
            (0.30, "#f5b2b1"), (0.40, "#f29b9a"), (0.50, "#f08685"),
            (0.60, "#ee716f"), (0.70, "#eb5d5b"), (0.80, "#e84b4b"),
            (0.90, "#c73c3c"), (1.00, "#9e2e2e")],
}
INK, INK_MUTED = "#1a1a19", "#6b6a66"



# step3a used to write these as median_Lambda / frac_pools_Lambda0.  The name
# changed with the vocabulary -- lambda is the RATE, mu the mean expression, and
# this column is a pooled COUNT, sum_{c in j} mu_cg -- so older CSVs are mapped
# on load and both generations stay readable.
_OLD2NEW = {"median_Lambda": "median_Mu", "frac_pools_Lambda0": "frac_pools_Mu0"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--label", default="Leiden (scGPT)")
    ap.add_argument("--metric", choices=["median_CV", "share_within"],
                    default="median_CV",
                    help="median_CV counts precise genes; share_within counts "
                         "genes homogeneous inside metacells (within/total)")
    ap.add_argument("--cmap", choices=sorted(RAMPS), default="blue")
    ap.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,"
                                            "0.35,0.40,0.45,0.50")
    ap.add_argument("--row-order", choices=["ascending", "descending"],
                    default="descending",
                    help="descending puts the loosest threshold at the top, so "
                         "the counts read downward as the target tightens")
    ap.add_argument("--min-mu", type=float, default=0.0,
                    help="drop genes whose median POOLED COUNT at the given k is "
                         "below this.  Mu_j = sum_{c in j} mu_cg is a count, not "
                         "a rate -- CV <= 0.25 is exactly Mu >= 16 -- and the "
                         "panel carries pseudogenes whose expression is ~0.")
    ap.add_argument("--pair", action="store_true",
                    help="render the CV (precision) and within/total "
                         "(homogeneity) heatmaps side by side, each on its own "
                         "one-hue ramp and its own colour scale")
    ap.add_argument("--pair-cmaps", default="blue,red")
    ap.add_argument("--annot", choices=["auto", "on", "off"], default="auto",
                    help="in-cell counts; auto drops them when a panel is too "
                         "narrow to read them")
    ap.add_argument("--pdf", default="", help="also write a PDF at W,H inches")
    ap.add_argument("--cell-size", type=float, default=0.0,
                    help="inches per heatmap cell; the PDF page is then solved for "
                         "(width AND height), so two figures drawn with the same "
                         "value have identical cell geometry. Implies --square")
    ap.add_argument("--square", action="store_true",
                    help="draw square cells: the PDF keeps the requested WIDTH and its "
                         "HEIGHT is recomputed so the cell grid is exactly square with no "
                         "slack above or below the heatmaps")
    ap.add_argument("--pdf-fontsize", type=float, default=10)
    ap.add_argument("--cell-format", choices=["count", "compact"],
                    default="count",
                    help="compact renders 2,967 as 3.0k so four-digit counts "
                         "fit a narrow cell")
    ap.add_argument("--cell-fontsize", type=float, default=0.0,
                    help="font size of the in-cell counts; 0 derives it from "
                         "the figure font size. Below 6 pt the thousands "
                         "separator is dropped so 4-digit counts fit.")
    ap.add_argument("--axis-fontsize", type=float, default=0.0,
                    help="font size of the x/y axis labels; 0 follows the "
                         "figure font size. Large values use short axis labels, "
                         "which is the only way they fit a small panel.")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    d = pd.read_csv(args.curves).rename(columns=_OLD2NEW)
    d["share_within"] = 1.0 - d["share_between"]
    SEQ = LinearSegmentedColormap.from_list(args.cmap, RAMPS[args.cmap])
    print(f"ramp '{args.cmap}': 0.80 -> "
          + matplotlib.colors.to_hex(SEQ(0.80)))
    ks = sorted(d.k.unique())
    ts = [float(x) for x in args.thresholds.split(",")]
    n_genes = d.gene.nunique()

    M = np.zeros((len(ts), len(ks)), int)
    for j, k in enumerate(ks):
        sub = d[d.k == k]
        if args.min_mu > 0:
            sub = sub[sub.median_Mu >= args.min_mu]
        for i, t in enumerate(ts):
            M[i, j] = int((sub[args.metric] <= t).sum())
    if args.row_order == "descending":
        ts = ts[::-1]
        M = M[::-1]
    tab = pd.DataFrame(M, index=[f"{t:g}" for t in ts], columns=ks)
    tab.index.name = "CV_threshold" if args.metric == "median_CV" else "within_over_total_threshold"
    tag = "precision" if args.metric == "median_CV" else "homogeneity"
    tab.to_csv(f"{args.out_prefix}_{tag}_counts.csv")
    print(tab.to_string())

    is_cv = args.metric == "median_CV"

    def counts_for(metric):
        out = np.zeros((len(ts), len(ks)), int)
        for j, k in enumerate(ks):
            sub = d[d.k == k]
            if args.min_mu > 0:
                sub = sub[sub.median_Mu >= args.min_mu]
            for i, t in enumerate(ts):
                out[i, j] = int((sub[metric] <= t).sum())
        # ts is already in display order by the time this runs -- flipping
        # here as well would invert the data against its own row labels
        return out

    if args.pair:
        # the paired figure shows both metrics, so both count tables are
        # written -- otherwise the one not named by --metric goes stale
        for mt, tg, nm in (("median_CV", "precision", "CV_threshold"),
                           ("share_within", "homogeneity",
                            "within_over_total_threshold")):
            tb = pd.DataFrame(counts_for(mt), index=[f"{t:g}" for t in ts],
                              columns=ks)
            tb.index.name = nm
            tb.to_csv(f"{args.out_prefix}_{tg}_counts.csv")

    def render_pair(path, figsize, fs, tight):
        afs = args.axis_fontsize or fs
        short = afs > fs * 1.15
        mats = [("median_CV", counts_for("median_CV"),
                 "median CV $\\leq$" if short
                 else "precision   (median CV $\\leq$)", "Precision"),
                ("share_within", counts_for("share_within"),
                 "within / total $\\leq$" if short
                 else "homogeneity   (within / total $\\leq$)", "Homogeneity")]
        cmaps = args.pair_cmaps.split(",")
        cfs = args.cell_fontsize or fs * 0.62
        annot = (args.annot == "on" or
                 (args.annot == "auto" and figsize[0] / 2 >= 3.2))
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        bars = []
        for ax, (metric, Mx, ylab, head), cname in zip(axes, mats, cmaps):
            cm = LinearSegmentedColormap.from_list(cname, RAMPS[cname])
            im = ax.imshow(Mx, cmap=cm, aspect=("equal" if args.square else "auto"),
                           vmin=0, vmax=Mx.max())
            if annot:
                for i in range(Mx.shape[0]):
                    for j in range(Mx.shape[1]):
                        v = Mx[i, j]
                        if args.cell_format == "compact" and v >= 1000:
                            txt = f"{v / 1000:.1f}k"
                        else:
                            txt = f"{v:,}" if cfs >= 6 else f"{v}"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=cfs,
                                color="white" if v > 0.55 * Mx.max() else INK)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels(ks, fontsize=fs * 0.8, rotation=90)
            ax.set_yticks(range(len(ts)))
            ax.set_yticklabels([f"{t:g}" for t in ts], fontsize=fs * 0.8)
            ax.set_xlabel("metacell size $k$", fontsize=afs)
            ax.set_ylabel(ylab, fontsize=afs)
            ax.set_title(head, fontsize=afs, color=INK)
            ax.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(ts) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", lw=0.8)
            ax.tick_params(which="both", length=0, colors=INK_MUTED)
            for sp in ax.spines.values():
                sp.set_visible(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            cb.ax.tick_params(labelsize=fs * 0.7, length=2, colors=INK_MUTED)
            cb.outline.set_visible(False)
            bars.append((ax, cb))

        def fit_bars():
            # aspect="equal" shrinks the image axes but leaves the colorbar at
            # the height of the slot it was carved from, so it has to be put
            # back onto the axes it belongs to.
            fig.canvas.draw()
            for a, c in bars:
                p, q = a.get_position(), c.ax.get_position()
                c.ax.set_position([q.x0, p.y0, q.width, p.height])
        fig.suptitle(f"Genes meeting a target \u2014 {args.label}", fontsize=afs)
        if tight:
            fig.tight_layout()
            if args.square:
                # square cells leave the slot taller than the axes drawn in it;
                # take that slack out of the page so the title does not float
                fig.canvas.draw()
                w_in, h_in = fig.get_size_inches()
                slot = axes[0].get_position(original=True).height * h_in
                drawn = axes[0].get_position().height * h_in
                fig.set_size_inches(w_in, h_in - (slot - drawn))
                fig.tight_layout()
                fit_bars()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            # margins from what the labels actually occupy: the y tick labels
            # ("0.45" at the tick size) plus one rotated axis label, converted
            # from points to a fraction of the page
            w_in, h_in = figsize
            nr, nc = Mx.shape
            bot_in = (4 * fs * 0.8 * 0.62 + afs * 1.25 + 6) / 72
            top_in = (2 * afs * 1.45 + 4) / 72

            def place(w, h):
                # lay the page out at (w, h) and report the heatmap width the
                # left panel actually ends up with, in inches -- the colorbar
                # has already taken its share out of it by then
                fig.set_size_inches(w, h)
                left = (4 * fs * 0.8 * 0.58 + afs * 1.25 + 6) / 72 / w
                fig.subplots_adjust(left=left, right=0.965, bottom=bot_in / h,
                                    top=1 - top_in / h, wspace=0.55)
                fig.canvas.draw()
                return axes[0].get_position().width * w

            ax_w = place(w_in, h_in)
            if args.cell_size:
                # Heatmap width is affine in page width (the left margin is
                # fixed in points, the right one is a fraction), so two probes
                # and a secant step land on the requested cell pitch exactly.
                tgt = args.cell_size * nc
                w1 = w_in + (tgt - ax_w) * 1.25
                a1 = place(w1, h_in)
                if abs(a1 - ax_w) > 1e-9:
                    w_in = w1 + (tgt - a1) * (w1 - w_in) / (a1 - ax_w)
                ax_w = place(w_in, h_in)
            if args.square or args.cell_size:
                # the width now fixes the column pitch; the height is free, so
                # set it to the one that leaves no band above or below
                h_in = ax_w * nr / nc + bot_in + top_in
                ax_w = place(w_in, h_in)
                fit_bars()
            p0 = axes[0].get_position()
            print(f"wrote paired PDF at {w_in:.3g} x {h_in:.3g} in; heatmap "
                  f"{p0.width * w_in:.4f} x {p0.height * h_in:.4f} in, cell "
                  f"{p0.width * w_in / nc:.4f} x {p0.height * h_in / nr:.4f} in")
            fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)

    def render(path, figsize, fs, tight):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=M.max())
        for i in range(len(ts)):
            for j in range(len(ks)):
                v = M[i, j]
                ax.text(j, i, f"{v:,}" if v else "0", ha="center", va="center",
                        fontsize=fs * 0.72,
                        color="white" if v > 0.55 * M.max() else INK)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, fontsize=fs * 0.85)
        ax.set_yticks(range(len(ts)))
        ax.set_yticklabels([f"{t:g}" for t in ts], fontsize=fs * 0.85)
        ax.set_xlabel("metacell size $k$ (cells per metacell)", fontsize=fs)
        ax.set_ylabel("precision threshold (median CV $\\leq$)" if is_cv
                      else "homogeneity threshold (within / total $\\leq$)",
                      fontsize=fs)
        head = ("Genes reaching a precision target" if is_cv
                else "Genes homogeneous within metacells")
        sub = f"of {n_genes:,} genes" + (
            f", pooled count $\\geq$ {args.min_mu:g}" if args.min_mu > 0 else "")
        ax.set_title(f"{head} \u2014 {args.label}\n{sub}", fontsize=fs, color=INK)
        ax.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ts) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="both", length=0, colors=INK_MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("number of genes", fontsize=fs * 0.85, color=INK_MUTED)
        cb.ax.tick_params(labelsize=fs * 0.75, length=2, colors=INK_MUTED)
        cb.outline.set_visible(False)
        if tight:
            fig.tight_layout()
            if args.square:
                # square cells leave the slot taller than the axes drawn in it;
                # take that slack out of the page so the title does not float
                fig.canvas.draw()
                w_in, h_in = fig.get_size_inches()
                slot = axes[0].get_position(original=True).height * h_in
                drawn = axes[0].get_position().height * h_in
                fig.set_size_inches(w_in, h_in - (slot - drawn))
                fig.tight_layout()
                fit_bars()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.82)
            fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)

    if args.pair:
        render_pair(f"{args.out_prefix}_precision_homogeneity.png",
                    (12.0, 5.2), 11, True)
        if args.pdf:
            w, h = (float(v) for v in args.pdf.split(","))
            render_pair(f"{args.out_prefix}_precision_homogeneity.pdf",
                        (w, h), args.pdf_fontsize, False)
    else:
        render(f"{args.out_prefix}_{tag}_heatmap.png", (9.0, 5.0), 11, True)
        if args.pdf:
            w, h = (float(v) for v in args.pdf.split(","))
            render(f"{args.out_prefix}_{tag}_heatmap.pdf", (w, h),
                   args.pdf_fontsize, False)
            print(f"wrote PDF at {w} x {h} in")
    print("done")


if __name__ == "__main__":
    main()
