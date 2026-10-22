#!/usr/bin/env python
"""task10: genes passing BOTH a precision and a homogeneity target, by
expression decile and metacell size.

A gene passes at size k when its median CV across metacells is at or below
--cv-cut AND its within-metacell share of total variance is at or below
--within-cut. Genes are binned by mean expression rate (lam_bar, which the
pipeline records per gene and which does not depend on the partition), with the
low deciles lumped because almost nothing down there ever passes.
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

# sequential, one hue, light -> dark; one ramp per criterion so the figures in
# this series stay distinguishable at a glance
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--label", default="Leiden (scGPT)")
    ap.add_argument("--cv-cut", type=float, default=0.5,
                    help="set very large to drop the precision criterion and "
                         "count on homogeneity alone")
    ap.add_argument("--within-cut", type=float, default=0.5,
                    help="set to 1 (or more) to drop the homogeneity criterion "
                         "and count on precision alone")
    ap.add_argument("--cmap", choices=sorted(RAMPS), default="green")
    ap.add_argument("--fraction", action="store_true",
                    help="plot the share of each decile that passes rather "
                         "than the raw count, so rows of different size are "
                         "comparable")
    ap.add_argument("--suffix", default="",
                    help="appended to the output names, so variants of this "
                         "figure do not overwrite each other")
    ap.add_argument("--lump-below", type=int, default=8,
                    help="deciles strictly below this are shown as one row")
    ap.add_argument("--pdf", default="")
    ap.add_argument("--square", action="store_true",
                    help="draw square cells: the page keeps the requested WIDTH and its "
                         "HEIGHT is recomputed so the grid is square with no slack")
    ap.add_argument("--pdf-fontsize", type=float, default=8,
                    help="axis label / tick size on the PDF")
    ap.add_argument("--pdf-cell-fontsize", type=float, default=0.0,
                    help="in-cell size on the PDF; 0 derives it from --pdf-fontsize")
    ap.add_argument("--pdf-bare", action="store_true", default=True,
                    help="on the PDF, drop the title, the per-row gene counts "
                         "and the colourbar -- at a small size they cost more "
                         "room than they carry")
    ap.add_argument("--pdf-keep-chrome", dest="pdf_bare", action="store_false")
    ap.add_argument("--pdf-no-colorbar", action="store_true",
                    help="drop the colourbar but keep the title; every cell "
                         "already carries its value")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    d = pd.read_csv(args.curves)
    d["within"] = 1.0 - d["share_between"]
    ks = sorted(d.k.unique())

    # expression deciles from the per-gene mean rate (constant across k)
    base = d.groupby("gene")["lam_bar"].first()
    dec = pd.qcut(base.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    grp = dec.where(dec >= args.lump_below, 0)          # 0 == the lumped rows
    names = [f"1–{args.lump_below - 1}"] + \
            [str(i) for i in range(args.lump_below, 11)]
    codes = [0] + list(range(args.lump_below, 11))
    d["grp"] = d.gene.map(grp)

    M = np.zeros((len(codes), len(ks)), int)
    for j, k in enumerate(ks):
        s = d[d.k == k]
        ok = (s.median_CV <= args.cv_cut) & (s["within"] <= args.within_cut)
        for i, c in enumerate(codes):
            M[i, j] = int((ok & (s.grp == c)).sum())
    M = M[::-1]                                          # decile 10 on top
    rows = names[::-1]
    sizes = {n: int((grp == c).sum()) for n, c in zip(names, codes)}
    if args.fraction:
        M = M / np.array([sizes[r] for r in rows])[:, None]
    rng = base.groupby(grp).agg(["min", "max"])

    tab = pd.DataFrame(M.round(4) if args.fraction else M,
                       index=rows, columns=ks)
    tab.index.name = "expression_decile"
    cv_on, w_on = args.cv_cut < 1e6, args.within_cut < 1.0
    sfx = args.suffix or ("" if (cv_on and w_on)
                          else "_cv_only" if cv_on else "_within_only")
    tab.to_csv(f"{args.out_prefix}_pass_by_decile{sfx}.csv")
    terms = ([f"median CV <= {args.cv_cut:g}"] if cv_on else []) + \
            ([f"within/total <= {args.within_cut:g}"] if w_on else [])
    print("pass = " + " AND ".join(terms) + f"\ngenes per group: {sizes}\n")
    print("mean-rate range per group:")
    print(rng.rename(index={0: names[0]}).to_string(), "\n")
    print(tab.to_string())

    SEQ = LinearSegmentedColormap.from_list(args.cmap, RAMPS[args.cmap])

    def render(path, figsize, fs, tight, cfs=None):
        bare = (not tight) and args.pdf_bare
        show_cb = tight or not (bare or args.pdf_no_colorbar)
        cfs = cfs or fs * 0.8
        fig, ax = plt.subplots(figsize=figsize)
        top = 1.0 if args.fraction else M.max()
        im = ax.imshow(M, cmap=SEQ, aspect=("equal" if args.square else "auto"),
                       vmin=0, vmax=top)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = f"{v:.0%}" if args.fraction else f"{v:,}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=cfs,
                        color="white" if v > 0.55 * top else INK)
        # the x tick labels must fit one column; "1024" at the axis size does
        # not, so cap them by the width actually available
        right = 0.87 if show_cb and not tight else 0.99
        w_in = figsize[0] * (right - 0.13)
        tfs = min(fs * 0.9, 72 * w_in / len(ks) / 2.9)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=tfs)
        ax.set_yticks(range(len(rows)))
        long_rows = tight
        ax.set_yticklabels([f"{r}  ({sizes[r]:,})" for r in rows] if long_rows
                           else rows, fontsize=tfs if not tight else fs * 0.9)
        ax.set_xlabel("metacell size $k$ (cells per metacell)", fontsize=fs)
        ax.set_ylabel("expression decile  (genes)" if tight else "decile",
                      fontsize=fs)
        bits = ([f"CV $\\leq$ {args.cv_cut:g}"] if cv_on else []) + \
               ([f"within/total $\\leq$ {args.within_cut:g}"] if w_on else [])
        crit = " and ".join(bits)
        if not bare:
            # on a fixed page the one-line title overruns; wrap it and size it
            # to whichever line is longer
            t1, t2 = f"Genes passing {crit}", f"\u2014 {args.label}"
            tt = (f"{t1}\n{t2}" if tight else f"{t1}\n{t2}")
            tfs_t = fs if tight else min(
                fs, 72 * figsize[0] * 0.95 / max(len(t1), len(t2)) / 0.56)
            ax.set_title(tt, fontsize=tfs_t, color=INK, pad=3)
        ax.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0, colors=INK_MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if show_cb:
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("share of the decile" if args.fraction
                         else "number of genes", fontsize=fs * 0.85,
                         color=INK_MUTED)
            cb.ax.tick_params(labelsize=fs * 0.8, length=2, colors=INK_MUTED)
            cb.outline.set_visible(False)

        def fit_bar():
            # aspect="equal" shrinks the image axes but leaves the colorbar at
            # the height of the slot it came out of
            fig.canvas.draw()
            if show_cb:
                pa, q = ax.get_position(), cb.ax.get_position()
                cb.ax.set_position([q.x0, pa.y0, q.width, pa.height])
        if tight:
            fig.tight_layout()
            if args.square:
                # square cells leave the slot taller than the axes drawn in it;
                # take that slack out of the page
                fig.canvas.draw()
                fw, fh = fig.get_size_inches()
                slack = (ax.get_position(original=True).height
                         - ax.get_position().height) * fh
                fig.set_size_inches(fw, fh - slack)
                fig.tight_layout()
                fit_bar()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            w_in, h_in = figsize
            labs = rows
            left = (max(len(x) for x in labs) * tfs * 0.60 + fs * 1.25 + 4) \
                / 72 / w_in
            bot_in = (tfs * 1.25 + fs * 1.25 + 4) / 72
            top_in = 0.015 * h_in if bare else (2 * tfs_t * 1.35 + 6) / 72
            fig.subplots_adjust(left=left, right=right, bottom=bot_in / h_in,
                                top=1 - top_in / h_in)
            if args.square:
                # width is fixed, so the column pitch is; solve for the height
                fig.canvas.draw()
                ax_w = ax.get_position().width * w_in
                h_in = ax_w * M.shape[0] / M.shape[1] + bot_in + top_in
                fig.set_size_inches(w_in, h_in)
                fig.subplots_adjust(left=left, right=right, bottom=bot_in / h_in,
                                    top=1 - top_in / h_in)
                fit_bar()
            fig.canvas.draw()
            pa = ax.get_position()
            print(f"wrote PDF at {w_in:g} x {h_in:.3g} in; heatmap "
                  f"{pa.width * w_in:.4f} x {pa.height * h_in:.4f} in, cell "
                  f"{pa.width * w_in / M.shape[1]:.4f} x "
                  f"{pa.height * h_in / M.shape[0]:.4f} in")
            fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)

    render(f"{args.out_prefix}_pass_by_decile{sfx}.png", (9.2, 0.55 * len(rows) + 2.2),
           11, True)
    if args.pdf:
        w, h = (float(v) for v in args.pdf.split(","))
        render(f"{args.out_prefix}_pass_by_decile{sfx}.pdf", (w, h),
               args.pdf_fontsize, False,
               cfs=args.pdf_cell_fontsize or None)
    print("done")


if __name__ == "__main__":
    main()
