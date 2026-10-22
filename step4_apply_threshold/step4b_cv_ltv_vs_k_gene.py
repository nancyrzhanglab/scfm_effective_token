#!/usr/bin/env python
"""task10: Poisson CV vs metacell size k for one gene, across construction methods.

Curves are median CV over metacells at each k, with the inter-quartile band.
Pure Poisson sampling would fall as k^-1/2; the guide line shows that slope so
departures from it are readable.

Not every curve is computed on the same reconstruction -- the source file for
each is printed and carried into the legend, because a method scored on a
different main-effect matrix is not directly comparable to the others.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

_SANS = next((f for f in ("Helvetica", "Nimbus Sans", "Arial")
              if any(x.name == f for x in
                     matplotlib.font_manager.fontManager.ttflist)),
             "DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": [_SANS, "DejaVu Sans"],
                     "pdf.fonttype": 42,
                     # without these the italic k in "$k$" comes from mathtext's
                     # default serif/DejaVu and the PDF embeds a second typeface
                     "mathtext.fontset": "custom",
                     "mathtext.rm": _SANS,
                     "mathtext.it": f"{_SANS}:italic",
                     "mathtext.bf": f"{_SANS}:bold",
                     "mathtext.cal": _SANS,
                     "mathtext.sf": _SANS})

# colour identifies the metacell method; line style identifies the embedding /
# reconstruction the curve was scored on
METHOD_COLORS = {"Leiden": "#8B2626", "SEACells": "#525EA7",
                 "MetaCell-1": "#5FACD3"}
FALLBACK = ["#8c6bb1", "#b03a48", "#7f7f7f"]
# marker shape also tracks the method, so two curves that coincide stay legible
MARKERS = {"Leiden": "o", "SEACells": "s", "MetaCell-1": "^"}


def split_label(label):
    """'MetaCell-1 (Harmony)' -> ('MetaCell-1', 'Harmony')."""
    if "(" in label:
        m, e = label.split("(", 1)
        return m.strip(), e.rstrip(")").strip()
    return label.strip(), ""



# step3a used to write these as median_Lambda / frac_pools_Lambda0.  The name
# changed with the vocabulary -- lambda is the RATE, mu the mean expression, and
# this column is a pooled COUNT, sum_{c in j} mu_cg -- so older CSVs are mapped
# on load and both generations stay readable.
_OLD2NEW = {"median_Lambda": "median_Mu", "frac_pools_Lambda0": "frac_pools_Mu0"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", default="GZMK")
    ap.add_argument("--metric", default="median_CV",
                    help="one metric, or several comma-separated for side-by-"
                         "side panels: median_CV, share_between (between/total) "
                         "or share_within (within/total)")
    ap.add_argument("--h5ad", required=True, help="for the gene-name -> index map")
    ap.add_argument("--curve", action="append", required=True,
                    metavar="LABEL::CURVES_CSV[::META_JSON]",
                    help="repeatable; '::' separated so labels may contain '='")
    ap.add_argument("--xscale", choices=["linear", "log2"], default="linear")
    ap.add_argument("--yscale", default="linear",
                    help="one per panel, comma-separated: linear or log")
    ap.add_argument("--yticks", default="0,0.25,0.5,0.75,1,1.25",
                    help="the only ticks drawn; panels separated by '|', ticks "
                         "within a panel by ','")
    ap.add_argument("--hlines", default="0.1,0.25,0.5",
                    help="CV levels to mark with a dashed rule (empty for none)")
    ap.add_argument("--floor", default="", metavar="LABEL::CSV",
                    help="a simulated Poisson null to overlay.  Its within/total "
                         "uses the FIRST --curve's total as the denominator: the "
                         "simulation has ~9%% less total variance, so dividing by "
                         "its own would inflate the floor by a factor growing with k.")
    ap.add_argument("--band", action="store_true", default=True)
    ap.add_argument("--no-band", dest="band", action="store_false")
    ap.add_argument("--stack", action="store_true",
                    help="panels one above the other instead of side by side; "
                         "only the bottom panel carries the x label")
    ap.add_argument("--pdf", default="",
                    help="also write a PDF at exactly this width,height in "
                         "inches, with shortened labels to fit")
    ap.add_argument("--pdf-fontsize", type=float, default=10)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    f = h5py.File(args.h5ad, "r")
    g = f["var/feature_name"]
    cats = np.array([c.decode() for c in g["categories"][:]])
    genes = cats[g["codes"][:]]
    f.close()
    hit = np.flatnonzero(genes == args.gene)
    if hit.size == 0:
        raise SystemExit(f"{args.gene} is not in the panel")
    gi = int(hit[0])
    print(f"{args.gene}: panel index {gi}")

    series, rows = [], []
    for spec in args.curve:
        parts = spec.split("::")
        label, path = parts[0], parts[1]
        meta = json.load(open(parts[2])) if len(parts) > 2 else {}
        d = pd.read_csv(path).rename(columns=_OLD2NEW)
        d = d[d["gene"] == gi].sort_values("k")
        if d.empty:
            raise SystemExit(f"{label}: no rows for gene {gi}")
        src = os.path.basename(str(meta.get("counts", "?")))
        series.append((label, d, src))
        d = d.assign(share_within=1.0 - d["share_between"])
        for _, r in d.iterrows():
            rows.append(dict(method=label, k=int(r.k), median_CV=r.median_CV,
                             q25_CV=r.q25_CV, q75_CV=r.q75_CV,
                             share_between=r.share_between,
                             share_within=1.0 - r.share_between,
                             within=r.within, between=r["between"], total=r.total,
                             median_Mu=r.median_Mu, counts=src))
        print(f"  {label:22s} k={list(d.k)}  counts={src}")
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{args.out_prefix}_{args.gene}_curves.csv", index=False)

    floor = None
    if args.floor:
        flab, fpath = args.floor.split("::")
        fd = pd.read_csv(fpath).rename(columns=_OLD2NEW)
        fd = fd[fd["gene"] == gi].sort_values("k")
        ref = series[0][1].set_index("k")          # the first curve is the yardstick
        den = ref.total.reindex(fd.k.to_numpy()).to_numpy()
        fd = fd.assign(share_within=np.where(den > 0,
                                             fd.within.to_numpy() / den, np.nan),
                       share_between=np.nan)
        floor = (flab, fd)
        print(f"  {flab:22s} k={sorted(fd.k.tolist())} "
            f"(share on the {series[0][0]} denominator)")

    metrics = args.metric.split(",")
    yscales = (args.yscale.split(",") + [args.yscale.split(",")[-1]] * 4)[:len(metrics)]
    ytick_sets = (args.yticks.split("|") + [args.yticks.split("|")[-1]] * 4)[:len(metrics)]
    multi = len({s_ for _, _, s_ in series}) > 1
    # the raw components are variances of the count-scale reconstruction, so
    # they are only comparable within one embedding: the scGPT and Harmony
    # reconstructions carry different total variance for the same gene
    YL = {"components": f"{args.gene}  variance of the count-scale expression",
          "within": f"{args.gene}  within-metacell variance",
          "between": f"{args.gene}  between-metacell variance",
          "total": f"{args.gene}  total variance",
          "median_CV": f"{args.gene}  median CV across metacells",
          "share_between": f"{args.gene}  between-metacell / total variance",
          "share_within": f"{args.gene}  within-metacell / total variance"}
    YL_SHORT = {"components": "variance", "within": "within var", "between": "between var",
                "total": "total var",
                "median_CV": "median CV", "share_between": "between / total",
                "share_within": "within / total"}
    TT = {"components": "Variance components vs $k$",
          "within": "Within-metacell variance vs $k$",
          "between": "Between-metacell variance vs $k$",
          "total": "Total variance vs $k$",
          "median_CV": "Poisson CV vs $k$", "share_between":
          "Law of total variance vs $k$", "share_within":
          "Law of total variance vs $k$"}
    TT_SHORT = {"components": "components", "within": "within", "between": "between", "total": "total",
                "median_CV": "Poisson CV", "share_between": "LTV",
                "share_within": "LTV"}

    def render(path, figsize, fs, tick_fs, short, tight, margins=None):
        nr, nc = (len(metrics), 1) if args.stack else (1, len(metrics))
        fig, axes = plt.subplots(nr, nc, figsize=figsize)
        axes = np.atleast_1d(axes)
        seen = {}
        tot_seen = []
        for ax, metric, ysc, yts in zip(axes, metrics, yscales, ytick_sets):
            # "components" overlays within and between on one axis and marks
            # each embedding's (k-invariant) total with a flat rule
            comp = metric == "components"
            lw_ = 1.8 if not short else 1.2
            ms_ = 5 if not short else 3
            if comp:
                for label, d, _ in series:
                    emb = split_label(label)[1] or "total"
                    t_ = float(d.total.iloc[0])
                    if t_ not in [round(v, 18) for v in tot_seen]:
                        tot_seen.append(t_)
                        ax.axhline(t_, color="0.55", lw=lw_ * 0.7,
                                   ls=":" if emb.lower().startswith("harmony")
                                   else "-", zorder=0)
                        ax.annotate(f"total ({emb})", xy=(1.0, t_),
                                    xycoords=("axes fraction", "data"),
                                    xytext=(-3, 3), textcoords="offset points",
                                    fontsize=tick_fs * 0.85, color="0.45",
                                    ha="right", va="bottom")
            if floor is not None and metric in ("share_within", "within",
                                                "between", "total",
                                                "components"):
                fl = floor[1]
                ax.plot(fl.k, fl["within" if comp else metric],
                        ls=(0, (5, 2)), color="0.35",
                        marker="o", lw=lw_, ms=ms_, zorder=5, label=floor[0])
            for label, d, src in series:
                meth, emb = split_label(label)
                if meth not in seen:
                    seen[meth] = METHOD_COLORS.get(
                        meth, FALLBACK[len(seen) % len(FALLBACK)])
                c_ = seen[meth]
                ls = ":" if emb.lower().startswith("harmony") else "-"
                if args.band and metric == "median_CV":
                    ax.fill_between(d.k, d.q25_CV, d.q75_CV, color=c_,
                                    alpha=0.10, lw=0)
                if comp:
                    # filled marker = within, hollow = between, same hue/style
                    for mkey, mfc in (("within", c_), ("between", "white")):
                        yy = pd.Series(d[mkey].to_numpy(), index=d.index)
                        if ysc == "log":
                            yy = yy.mask(yy.to_numpy() <= 0)
                        ax.plot(d.k, yy, marker=MARKERS.get(meth, "D"), ls=ls,
                                color=c_, lw=lw_, ms=ms_, alpha=0.9,
                                markerfacecolor=mfc, markeredgecolor=c_,
                                label=label if mkey == "within" else None)
                    continue
                y = d[metric] if metric in d else 1.0 - d["share_between"]
                if ysc == "log":
                    # within is exactly 0 at k=1; log10(0) draws a spike to the
                    # axis floor, so leave the point out instead
                    y = pd.Series(y).mask(pd.Series(y).to_numpy() <= 0)
                ax.plot(d.k, y, marker=MARKERS.get(meth, "D"), ls=ls, color=c_,
                        lw=1.8 if not short else 1.2,
                        ms=5 if not short else 3, alpha=0.9, label=label)
            ks_all = sorted(set(tab.k))
            if args.xscale == "log2":
                ax.set_xscale("log", base=2)
                ax.set_xticks(ks_all); ax.set_xticklabels(ks_all)
            else:
                ax.set_xticks([1] + [k for k in ks_all if k >= 64])
            if ysc == "log":
                ax.set_yscale("log")
            yt = [float(v) for v in yts.split(",")]
            if ysc == "log":
                # a FixedLocator makes the autoscaler pad out to the tick range,
                # so pin the limits to the data and keep only the ticks inside
                col = (pd.concat([tab["within"], tab["between"], tab["total"]])
                       if comp else tab[metric])
                pos = col[col > 0]
                lo, hi = float(pos.min()) / 1.35, float(pos.max()) * 1.35
                ax.set_ylim(lo, hi)
                yt = [v for v in yt if lo <= v <= hi]
            ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if ysc == "linear":
                top = float((tab[["within", "between", "total"]].max().max()
                             if metric == "components"
                             else tab[metric].max()))
                ax.set_ylim(min(yt), max(max(yt), top * 1.04))
            for v in [float(x) for x in args.hlines.split(",") if x.strip()]:
                lo, hi = ax.get_ylim()
                if lo <= v <= hi:
                    ax.axhline(v, color="0.6", lw=0.9 if not short else 0.6,
                               ls="--", zorder=0)
                    if not short:
                        ax.annotate(f"{v:g}", xy=(1.0, v),
                                    xycoords=("axes fraction", "data"),
                                    xytext=(3, 0), textcoords="offset points",
                                    fontsize=7, color="0.45", va="center",
                                    annotation_clip=False)
            ax.tick_params(labelsize=tick_fs, length=2.5, pad=1.5)
            # 64 and 128 sit only 64 units apart on a linear 1..1024 axis, so
            # the x labels need to be a shade smaller than the y labels to fit
            ax.tick_params(axis="x", labelsize=tick_fs * (0.8 if short else 1.0))
            last = (not args.stack) or ax is axes[-1]
            ax.set_xlabel(("metacell size $k$" if short
                           else "metacell size $k$ (cells per metacell)")
                          if last else "", fontsize=fs)
            ax.set_ylabel((YL_SHORT if short else YL)[metric], fontsize=fs)
            ax.set_title((TT_SHORT if short else TT)[metric]
                         + ("" if short else f" \u2014 {args.gene}"), fontsize=fs)
            ax.spines[["top", "right"]].set_visible(False)
        h, l = axes[0].get_legend_handles_labels()
        if "components" in metrics:
            proxy = matplotlib.lines.Line2D
            h += [proxy([], [], ls="none", marker="o", color="0.35",
                        markerfacecolor="0.35", ms=5 if not short else 3),
                  proxy([], [], ls="none", marker="o", color="0.35",
                        markerfacecolor="white", ms=5 if not short else 3)]
            l += ["filled: within", "hollow: between"]
        kw = {}
        if metrics == ["components"] and tight:
            # every corner of a single components panel is occupied, so the key
            # goes outside the axes (bbox_inches="tight" keeps it in the file)
            kw = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        axes[0].legend(h, l, fontsize=tick_fs, frameon=False,
                       handlelength=1.4, labelspacing=0.3, borderpad=0.1, **kw)
        if multi and not short:
            fig.text(0.5, -0.01, "solid: scGPT reconstruction    "
                     "dotted: Harmony reconstruction",
                     ha="center", fontsize=7.5, color="0.35")
        if tight:
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.subplots_adjust(**margins)
            bb = axes[0].get_position()
            print(f"  axes box {bb.width*figsize[0]:.3f} x "
                  f"{bb.height*figsize[1]:.3f} in  (aspect "
                  f"{bb.width*figsize[0]/(bb.height*figsize[1]):.4f})")
            fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)

    stem = f"{args.out_prefix}_{args.gene}_{args.metric.replace(',', '_')}_vs_k"
    # one panel carrying both components needs a wider axis or 512 and 1024
    # collide on the log2 x axis
    _w = 6.6 if metrics == ["components"] else 5.4 * len(metrics)
    render(stem + ".png", (_w, 4.2), 11, 9, False, True)
    if args.pdf:
        w, h = (float(v) for v in args.pdf.split(","))
        fs = args.pdf_fontsize
        # margins are stated in INCHES and converted, so the same numbers hold
        # whatever page size is asked for
        if args.stack:
            m = dict(left=(fs * 0.80 * 3 * 0.60 + 4) / 72 + (fs * 1.45) / 72,
                     right=0.04, bottom=(fs * 0.80 * 1.3 + fs * 1.45 + 4) / 72,
                     top=(fs * 1.5) / 72)
            mg = dict(left=m["left"] / w, right=1 - m["right"] / w,
                      bottom=m["bottom"] / h, top=1 - m["top"] / h,
                      hspace=(m["bottom"] + (fs * 0.5) / 72) /
                             ((h - m["bottom"] - m["top"]) / len(metrics)))
        else:
            mg = dict(left=0.105, right=0.985, bottom=0.215, top=0.88,
                      wspace=0.42)
        render(stem + ".pdf", (w, h), fs, fs * 0.8, True, False, margins=mg)
        print(f"wrote PDF at {w} x {h} in, {fs} pt")
    print("done")


if __name__ == "__main__":
    main()
