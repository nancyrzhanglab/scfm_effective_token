#!/usr/bin/env python
"""task10: usable measurements against metacell size.

For each k: (number of metacells) x (number of genes that pass both the
precision and the homogeneity target at that k). Pooling trades the first
factor against the second -- metacells halve with every step up the ladder,
while the set of well-measured genes grows -- and this is the product.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
BAR, LINE = "#2a78d6", "#1a1a19"
INK, INK_MUTED = "#1a1a19", "#6b6a66"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--diag", required=True,
                    help="cv_ltv_diag_*.csv, for the metacell count per k")
    ap.add_argument("--label", default="Leiden (scGPT)")
    ap.add_argument("--cv-cut", type=float, default=0.5)
    ap.add_argument("--within-cut", type=float, default=0.5)
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--genes-panel", action="store_true", default=True,
                    help="add the passing-gene count as a second panel sharing "
                         "the x axis. The two series are 6 orders of magnitude "
                         "apart and in different units, so they get one axis "
                         "each rather than a second y axis on one plot.")
    ap.add_argument("--no-genes-panel", dest="genes_panel", action="store_false")
    ap.add_argument("--bar-color", default=BAR,
                    help="bar fill; the line stays ink")
    ap.add_argument("--pdf", default="")
    ap.add_argument("--pdf-fontsize", type=float, default=8)
    ap.add_argument("--pdf-value-fontsize", type=float, default=0.0,
                    help="size of the numbers on the bars and points; 0 derives "
                         "it from the axis size")
    ap.add_argument("--cumulative", action="store_true",
                    help="bank each gene ONCE, at the first rung where it passes, "
                         "and count it at that rung's metacell number: "
                         "usable(k) = usable(k-1) + n_metacells(k) * |genes newly "
                         "passing at k|.  The default instead re-counts every "
                         "passing gene at every rung, which asks a different "
                         "question -- what one rung alone would give you.")
    ap.add_argument("--suffix", default="",
                    help="appended to the output names so threshold variants do "
                         "not overwrite each other")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    d = pd.read_csv(args.curves)
    d["within"] = 1.0 - d["share_between"]
    diag = pd.read_csv(args.diag).set_index("k")
    rows = []
    banked = set()          # genes already counted at a finer rung
    cum = 0
    for k in sorted(d.k.unique()):
        s = d[d.k == k]
        ok = (s.median_CV <= args.cv_cut) & (s["within"] <= args.within_cut)
        passing = set(s.gene[ok])
        npass = len(passing)
        npool = int(diag.loc[k, "n_pools"])
        # "new" is set-based, not a difference of counts: a gene can pass at one
        # rung and fail at the next (the within/total criterion tightens with k),
        # so a count difference would go negative and double-count on the way
        # back up.  A gene is banked at the FIRST rung that delivers it.
        new = passing - banked
        banked |= passing
        cum += npool * len(new)
        rows.append(dict(k=k, n_metacells=npool, n_genes_pass=npass,
                         n_genes_new=len(new), usable_at_k=npool * npass,
                         usable_cumulative=cum))
    for r in rows:
        r["usable"] = r["usable_cumulative"] if args.cumulative else r["usable_at_k"]
    t = pd.DataFrame(rows)
    sfx = args.suffix
    t.to_csv(f"{args.out_prefix}_usable_measurements{sfx}.csv", index=False)
    print(t.to_string(index=False))

    def render(path, figsize, fs, tight, vfs=None):
        vfs = vfs or fs * 0.75
        if args.genes_panel:
            fig, (ax, ax2) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.12))
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax2 = None
        x = np.arange(len(t))
        ax.bar(x, t.usable / 1e6, color=args.bar_color, width=0.74,
               edgecolor="white", linewidth=0.5)
        for xi, v in zip(x, t.usable / 1e6):
            ax.text(xi, v, f"{v:,.0f}" if v >= 10 else f"{v:,.1f}",
                    ha="center", va="bottom", fontsize=vfs, color=INK_MUTED)
        if args.yscale == "log":
            ax.set_yscale("log")
        # at a small fixed size the two-line y labels are taller than their
        # own panels, so the compact page gets short ones
        ax.set_ylabel("usable measurements (millions)\n"
                      "metacells $\\times$ passing genes" if tight
                      else "usable (M)", fontsize=fs)
        t1 = (f"Metacells $\\times$ genes passing CV $\\leq$ {args.cv_cut:g}"
              f" and within/total $\\leq$ {args.within_cut:g}")
        t2 = f"\u2014 {args.label}"
        # on a fixed page the one-line title overruns; wrap it and size it down
        tfs = fs if tight else min(fs, 72 * figsize[0] * 0.97 /
                                   max(len(t1), len(t2)) / 0.54)
        ax.set_title(f"{t1}\n{t2}" if not tight else f"{t1} {t2}",
                     fontsize=tfs, color=INK, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.12)

        bottom = ax2 if ax2 is not None else ax
        if ax2 is not None:
            ax2.plot(x, t.n_genes_pass, "o-", color=LINE, lw=1.8, ms=4.5)
            for xi, v in zip(x, t.n_genes_pass):
                ax2.text(xi, v, f"{v}", ha="center", va="bottom",
                         fontsize=vfs, color=INK_MUTED)
            ax2.set_ylabel("passing genes" if tight else "genes", fontsize=fs)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.margins(y=0.22)
            ax2.tick_params(axis="y", labelsize=fs * 0.85)
            ax.tick_params(axis="x", length=0)
        if tight:
            keep = np.ones(len(t), bool)
        else:   # 512 and 1024 collide at this width; label every other rung
            keep = np.array([i % 2 == 0 for i in range(len(t))])
        bottom.set_xticks(x[keep])
        bottom.set_xticklabels(t.k[keep], fontsize=fs * 0.85)
        bottom.set_xlabel("metacell size $k$ (cells per metacell)" if tight
                          else "metacell size $k$", fontsize=fs)
        for a_ in ([ax] if ax2 is None else [ax, ax2]):
            a_.tick_params(axis="y", labelsize=fs * 0.85, colors=INK_MUTED,
                           length=3)
        if tight:
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            w_in, h_in = figsize
            left = (5 * fs * 0.62 + 2 * fs * 1.25 + 4) / 72 / w_in
            fig.subplots_adjust(
                left=left, right=0.99,
                bottom=(fs * 0.85 * 1.4 + fs * 1.5 + 8) / 72 / h_in,
                top=1 - (2 * tfs * 1.4 + 4) / 72 / h_in, hspace=0.14)
            # that left margin is an upper bound (5 tick characters, a two-line
            # ylabel); the PDF draws one short line and 2-3 digits, so measure
            # what the labels really occupy and give the rest back to the axes
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            x0 = min(a_.get_tightbbox(r).x0 for a_ in fig.axes) / fig.dpi
            left = max(2 / 72, left * w_in - x0 + 2 / 72) / w_in
            fig.subplots_adjust(left=left)
            fig.savefig(path, dpi=600, facecolor="white")
            print(f"left margin {left * w_in:.3f} in")
        plt.close(fig)

    render(f"{args.out_prefix}_usable_measurements{sfx}.png", (7.6, 5.4), 11, True)
    if args.pdf:
        w, h = (float(v) for v in args.pdf.split(","))
        render(f"{args.out_prefix}_usable_measurements{sfx}.pdf", (w, h),
               args.pdf_fontsize, False,
               vfs=args.pdf_value_fontsize or None)
        print(f"wrote PDF at {w} x {h} in")
    print("done")


if __name__ == "__main__":
    main()
