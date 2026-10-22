#!/usr/bin/env python
"""task10: CV and LTV vs k for a whole expression decile, not one gene.

A single gene's curve is noisy and, as GZMK vs MKI67 showed, the ranking of
methods is gene-specific.  This aggregates over the genes of one expression
decile: the line is the MEDIAN across those genes and the band is their IQR, so
the spread shown is BETWEEN GENES -- a different quantity from the per-gene
figures, where the band was across metacells.

Deciles are cut on lam_bar, the cohort mean rate, which is a property of the
cells and is identical at every k and for every method.  The decile is taken
from ONE reference curve set and the same gene list is then applied to every
method, so the comparison is on a fixed gene set rather than on each method's
own idea of what the top decile is.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

_SANS = next((f for f in ("Helvetica", "Nimbus Sans", "Arial")
              if any(x.name == f for x in
                     matplotlib.font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": [_SANS, "DejaVu Sans"],
                     "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": _SANS, "mathtext.it": f"{_SANS}:italic"})
INK, INK_MUTED = "#1a1a19", "#6b6a66"
METHOD_COLORS = {"Leiden": "#8B2626", "SEACells": "#525EA7",
                 "MetaCell-1": "#5FACD3"}
MARKERS = {"Leiden": "o", "SEACells": "s", "MetaCell-1": "^"}
LABELS = {"median_CV": "median CV across metacells",
          "share_within": "within-metacell / total variance"}
TITLES = {"median_CV": "Poisson CV", "share_within": "Law of total variance"}
# on a narrow stacked page the long title spans the whole panel and crowds the
# upper panel's tick labels, so the fixed-size renders use these instead
TITLES_SHORT = {"median_CV": "Poisson CV", "share_within": "LTV"}
LABELS_SHORT = {"median_CV": "median CV", "share_within": "within / total"}


def split_label(lab):
    if "(" in lab and lab.rstrip().endswith(")"):
        m, e = lab.split("(", 1)
        return m.strip(), e.rstrip(")").strip()
    return lab.strip(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="append", required=True,
                    metavar="LABEL::CSV")
    ap.add_argument("--reference", default="",
                    help="label whose lam_bar defines the deciles; default the first")
    ap.add_argument("--decile", type=int, default=10)
    ap.add_argument("--n-deciles", type=int, default=10)
    ap.add_argument("--metrics", default="median_CV,share_within")
    ap.add_argument("--yscale", default="log,linear")
    ap.add_argument("--hlines", default="0.1,0.25,0.5")
    ap.add_argument("--yticks", default="",
                    help="the only y ticks drawn, so this figure can share an "
                         "axis with the per-gene one: panels separated by '|', "
                         "ticks within a panel by ','.  Empty = matplotlib's "
                         "own, which on a log axis gives 10^0 / 10^-1.")
    ap.add_argument("--floor", default="", metavar="LABEL::CSV",
                    help="a simulated Poisson null to overlay.  Its share is "
                         "computed against the REFERENCE curve's total, not its "
                         "own: the simulation has ~9%% less total variance "
                         "(removing within-metacell biology shrinks the spread "
                         "of per-cell rates), and dividing each by its own total "
                         "inflates the floor by a factor that grows with k.")
    ap.add_argument("--band", choices=["iqr", "none"], default="iqr")
    ap.add_argument("--stack", action="store_true",
                    help="panels one above the other instead of side by side; "
                         "only the bottom panel carries the x label")
    ap.add_argument("--pdf", default="")
    ap.add_argument("--pdf-fontsize", type=float, default=9)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    series = []
    for spec in args.curves:
        label, path = spec.split("::")
        d = pd.read_csv(path)
        # the curve files carry within/between/total and share_between; the
        # ratio this project reads is within/total, so derive it once here
        if "share_within" not in d:
            d["share_within"] = np.where(d.total > 0, d.within / d.total, np.nan)
        series.append((label, d))
        print(f"{label}: {d.gene.nunique():,} genes x {d.k.nunique()} rungs")

    ref = args.reference or series[0][0]
    rd = dict(series)[ref]
    lb = rd.groupby("gene").lam_bar.first()
    lb = lb[lb > 0]
    q = np.quantile(lb, np.linspace(0, 1, args.n_deciles + 1)[1:-1])
    dec = np.digitize(lb, q) + 1
    genes = set(lb.index[dec == args.decile])
    print(f"\ndecile {args.decile} from {ref}: {len(genes):,} genes, "
          f"lam_bar {lb[list(genes)].min():.2e} - {lb[list(genes)].max():.2e}")

    floor = None
    if args.floor:
        flab, fpath = args.floor.split("::")
        fd = pd.read_csv(fpath)
        if "share_within" not in fd:
            fd["share_within"] = np.where(fd.total > 0, fd.within / fd.total, np.nan)
        floor = (flab, fd)
        print(f"{flab}: {fd.gene.nunique():,} genes x {fd.k.nunique()} rungs "
              f"(plotted on the {ref} denominator)")

    metrics = args.metrics.split(",")
    yscales = args.yscale.split(",")
    rows = []
    for label, d in series:
        sub = d[d.gene.isin(genes)]
        for k, g in sub.groupby("k"):
            r = dict(method=label, k=int(k), n_genes=len(g))
            for m in metrics:
                v = g[m].to_numpy(float)
                v = v[np.isfinite(v)]
                r[f"{m}_q25"], r[f"{m}_med"], r[f"{m}_q75"] = np.percentile(
                    v, [25, 50, 75])
            rows.append(r)
    if floor is not None:
        flab, fd = floor
        rd_idx = rd.set_index(["gene", "k"])
        for k, g in fd[fd.gene.isin(genes)].groupby("k"):
            g = g.set_index("gene")
            den = rd_idx.reindex([(i, k) for i in g.index]).total.to_numpy()
            r = dict(method=flab, k=int(k), n_genes=len(g))
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(den > 0, g.within.to_numpy() / den, np.nan)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            r["share_within_q25"], r["share_within_med"], r["share_within_q75"] = \
                np.percentile(v, [25, 50, 75])
            for m in metrics:
                if m == "share_within":
                    continue
                vv = g[m].to_numpy(float); vv = vv[np.isfinite(vv)]
                r[f"{m}_q25"], r[f"{m}_med"], r[f"{m}_q75"] = (
                    np.percentile(vv, [25, 50, 75]) if len(vv) else (np.nan,)*3)
            rows.append(r)

    tab = pd.DataFrame(rows).sort_values(["method", "k"])
    tab.to_csv(f"{args.out_prefix}_decile{args.decile}_curves.csv", index=False)
    for m in metrics:
        print(f"\n{m}: median over the decile")
        print(tab.pivot(index="k", columns="method",
                        values=f"{m}_med").round(3).to_string())

    def render(path, figsize, fs, tight):
        nr, nc = (len(metrics), 1) if args.stack else (1, len(metrics))
        fig, axes = plt.subplots(nr, nc, figsize=figsize)
        axes = np.atleast_1d(axes)
        for ax, metric, ysc in zip(axes, metrics, yscales):
            for label in ([floor[0]] if floor is not None else []):
                d = tab[tab.method == label].sort_values("k")
                if d.empty or d[f"{metric}_med"].isna().all():
                    continue
                if args.band == "iqr":
                    ax.fill_between(d.k, d[f"{metric}_q25"], d[f"{metric}_q75"],
                                    color="0.55", alpha=0.12, lw=0)
                ax.plot(d.k, d[f"{metric}_med"], ls=(0, (5, 2)), color="0.35",
                        marker="o", lw=1.6 if tight else 1.2,
                        ms=4 if tight else 2.5, label=label, zorder=5)
            for label, _ in series:
                d = tab[tab.method == label].sort_values("k")
                meth, emb = split_label(label)
                c = METHOD_COLORS.get(meth, "0.3")
                ls = ":" if emb.lower().startswith("harmony") else "-"
                if args.band == "iqr":
                    ax.fill_between(d.k, d[f"{metric}_q25"], d[f"{metric}_q75"],
                                    color=c, alpha=0.13, lw=0)
                ax.plot(d.k, d[f"{metric}_med"], ls=ls, color=c,
                        marker=MARKERS.get(meth, "D"),
                        lw=1.6 if tight else 1.2, ms=5 if tight else 3,
                        alpha=0.9, label=label)
            for v in (float(x) for x in args.hlines.split(",") if x.strip()):
                ax.axhline(v, ls=(0, (4, 4)), lw=0.9, color="0.65", zorder=0)
            ax.set_yscale(ysc)
            if args.yticks:
                sets = args.yticks.split("|")
                yts = sets[min(list(metrics).index(metric), len(sets) - 1)]
                yt = [float(v) for v in yts.split(",") if v.strip()]
                lo, hi = ax.get_ylim()
                if ysc == "log":
                    yt = [v for v in yt if lo <= v <= hi]
                ax.set_yticks(yt)
                ax.set_yticklabels([f"{v:g}" for v in yt])
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_ylim(lo, hi)
            ks_all = sorted(set(tab.k))
            ax.set_xticks([1] + [k for k in ks_all if k >= 64])
            last = (not args.stack) or ax is axes[-1]
            ax.set_xlabel("metacell size $k$" if last else "", fontsize=fs)
            SH = args.stack and not tight
            ax.set_ylabel((LABELS_SHORT if SH else LABELS).get(metric, metric),
                          fontsize=fs)
            ax.set_title((TITLES_SHORT if SH else TITLES).get(metric, metric),
                         fontsize=fs, color=INK)
            ax.tick_params(labelsize=fs * 0.85, colors=INK_MUTED)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK_MUTED)
        axes[0].legend(frameon=False, fontsize=fs * 0.8, loc="best")
        if tight:
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            w_in, h_in = figsize
            L = (4 * fs * 0.6 + fs * 1.4 + 4) / 72
            B = (fs * 0.85 * 1.3 + fs * 1.4 + 5) / 72
            T = (fs * 1.5 + 3) / 72
            kw = dict(left=L / w_in, right=0.985,
                      bottom=B / h_in, top=1 - T / h_in)
            # a stacked pair needs room between the panels for the upper one's
            # tick labels; hspace is a fraction of the MEAN axes height
            # the gap must hold the upper panel's tick labels AND the lower
            # panel's title, with a little air between them
            gap = (fs * 0.85 * 1.3 + fs * 1.5 + 9) / 72
            kw["hspace" if args.stack else "wspace"] = (
                gap / ((h_in - B - T) / len(metrics)) if args.stack else 0.3)
            fig.subplots_adjust(**kw)
            bb = axes[0].get_position()
            print(f"  axes box {bb.width*figsize[0]:.3f} x "
                  f"{bb.height*figsize[1]:.3f} in  (aspect "
                  f"{bb.width*figsize[0]/(bb.height*figsize[1]):.4f})")
            fig.savefig(path, dpi=600, facecolor="white")
            print(f"wrote PDF at {w_in:g} x {h_in:g} in")
        plt.close(fig)

    out = f"{args.out_prefix}_decile{args.decile}_cv_ltv.png"
    render(out, (5.4, 4.4 * len(metrics)) if args.stack
                else (5.4 * len(metrics), 4.4), 12, True)
    print("\nwrote", out)
    if args.pdf:
        w, h = (float(v) for v in args.pdf.split(","))
        render(out.replace(".png", ".pdf"), (w, h), args.pdf_fontsize, False)


if __name__ == "__main__":
    main()
