"""Download every CD8 T cell in the CELLxGENE census, restricted to HVGs
reselected on that same population.

Phase 1  select 3000 HVGs over all 5.4M CD8 cells x all genes (seurat_v3,
         batched by dataset_id so genes that merely separate datasets do not win)
Phase 2  download the cells x HVG raw-count matrix in restartable blocks
Phase 3  concatenate blocks on disk into one h5ad

Both phases checkpoint, so a rerun skips completed work.
"""
import json, os, sys, time, argparse
import numpy as np, pandas as pd
import cellxgene_census, tiledbsoma
from cellxgene_census.experimental.pp import highly_variable_genes

t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective/results/poisson/task8_scGPTemb2gex/all_CD8T")
ap.add_argument("--census-version", default="2025-11-08",
                help="PIN THE BUILD.  open_soma() with no version opens whatever "
                     "build is current, so the cohort silently changes with the "
                     "release; 2025-11-08 is the build this project was built on "
                     "(5,381,464 CD8 cells, 110 datasets).  'latest' restores the "
                     "old behaviour.")
ap.add_argument("--terms", default="/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective/results/poisson/task8_scGPTemb2gex/cd8_ontology_terms.json")
ap.add_argument("--n-top-genes", type=int, default=3000)
ap.add_argument("--batch-key", default="dataset_id")
ap.add_argument("--block", type=int, default=400_000)
ap.add_argument("--cell-set", default="primary", choices=["primary", "unique", "all"],
                help="primary: is_primary_data==True (canonical copies only). "
                     "unique: every distinct cell, deduped on observation_joinid, "
                     "preferring the primary copy. all: every row, duplicates included.")
ap.add_argument("--prefix", default="cd8", help="output filename prefix")
ap.add_argument("--blocks-dir", default="",
                help="where the per-block h5ads go (default: <out>/blocks).  Set this to "
                     "keep a second gene selection's blocks beside the first instead of "
                     "mixing them in one directory.")
ap.add_argument("--var-suffix", default="",
                help="appended to hvg_var.csv / hvg_feature_ids.txt.  Without it a second "
                     "run in the same --out overwrites the first selection's gene list.")
ap.add_argument("--phase", default="all", choices=["all", "hvg", "download", "concat"])
ap.add_argument("--only-block", type=int, default=None,
                help="download just this block index and exit (for array jobs)")
a = ap.parse_args()

OUT = a.out; BLK = a.blocks_dir or os.path.join(OUT, "blocks")
os.makedirs(BLK, exist_ok=True)
terms = json.load(open(a.terms))
vf = "cell_type_ontology_term_id in [" + ",".join(f"'{t}'" for t in terms) + "]"
if a.cell_set == "primary":
    vf += " and is_primary_data == True"
log(f"{len(terms)} terms; cell-set={a.cell_set}; filter: {vf[:90]}...")

HVG_CSV = os.path.join(OUT, f"hvg_selection_{a.prefix}.csv")
IDS_TXT = os.path.join(OUT, f"hvg_feature_ids{a.var_suffix}.txt")

_soma = ({} if a.census_version == "latest"
         else dict(census_version=a.census_version))
with cellxgene_census.open_soma(**_soma) as census:
    build = (census["census_info"]["summary"].read().concat().to_pandas()
             .set_index("label").loc["census_build_date", "value"])
    log(f"census build {build}")
    if a.census_version != "latest":
        assert build == a.census_version, (
            f"asked for census {a.census_version}, got {build}")
    exp = census["census_data"]["homo_sapiens"]

    # ---------------- cell set ----------------
    # Resolved BEFORE the HVG pass so that pass sees exactly the cells that get
    # downloaded.  In "unique" mode the selection is row-level (dedupe), which a
    # value_filter cannot express, so the query is driven by coords instead.
    OBS_PQ = os.path.join(OUT, f"{a.prefix}_obs.parquet")
    if not os.path.exists(OBS_PQ):
        obs_all = exp.obs.read(value_filter=vf).concat().to_pandas()
        log(f"query returns {len(obs_all):,} rows")
        if a.cell_set == "unique":
            n0 = len(obs_all)
            obs_all = (obs_all.sort_values(["observation_joinid", "is_primary_data"],
                                           ascending=[True, False])
                       .drop_duplicates("observation_joinid", keep="first")
                       .sort_values("soma_joinid"))
            npr = int(obs_all.is_primary_data.sum())
            log(f"deduped {n0:,} rows -> {len(obs_all):,} distinct cells "
                f"({npr:,} with a primary copy, {len(obs_all)-npr:,} without)")
        obs_all.to_parquet(OBS_PQ, index=False)
        del obs_all
    ids = np.sort(pd.read_parquet(OBS_PQ, columns=["soma_joinid"])
                  .soma_joinid.values.astype(np.int64))
    log(f"cell set: {len(ids):,} cells")

    # ---------------- phase 1: HVG over the full population ----------------
    if a.phase in ("all", "hvg") and not os.path.exists(HVG_CSV):
        log("phase 1: streaming HVG over ALL genes -- this is the expensive pass")
        with exp.axis_query("RNA", obs_query=tiledbsoma.AxisQuery(coords=(ids,))) as q:
            n_obs, n_var = q.n_obs, q.n_vars
            log(f"  query matches {n_obs:,} cells x {n_var:,} genes")
            hvg = highly_variable_genes(q, n_top_genes=a.n_top_genes,
                                        batch_key=a.batch_key, flavor="seurat_v3")
        hvg.to_csv(HVG_CSV)
        log(f"  wrote {HVG_CSV}  ({int(hvg.highly_variable.sum())} HVGs flagged)")
    else:
        log("phase 1: HVG table already present, skipping")

    hvg = pd.read_csv(HVG_CSV, index_col=0)
    sel = hvg.index[hvg.highly_variable.values].to_numpy().astype(np.int64)
    sel.sort()
    assert len(sel) == a.n_top_genes, f"expected {a.n_top_genes} HVGs, got {len(sel)}"
    log(f"selected {len(sel)} HVG soma_joinids")

    var = (exp.ms["RNA"].var.read(coords=(sel,),
           column_names=["soma_joinid", "feature_id", "feature_name", "feature_length"])
           .concat().to_pandas())
    var.to_csv(os.path.join(OUT, f"hvg_var{a.var_suffix}.csv"), index=False)
    np.savetxt(IDS_TXT, var.feature_id.values, fmt="%s")
    MARK = ["CD8A","CD8B","GZMK","GZMB","NKG7","PRF1","IL7R","CCR7","TCF7",
            "SELL","KLRG1","TOX","LEF1","CCL5","PDCD1","ITGAE","ZNF683"]
    have = set(var.feature_name)
    log("marker check -> present: " + ",".join(m for m in MARK if m in have))
    log("           -> absent : " + ",".join(m for m in MARK if m not in have))

    # ---------------- phase 2: blocked download ----------------
    log(f"phase 2: {len(ids):,} cells -> {int(np.ceil(len(ids)/a.block))} blocks of {a.block:,}")

    paths = []
    if a.phase in ("all", "download", "concat"):
        for bi, lo in enumerate(range(0, len(ids), a.block)):
            hi = min(lo + a.block, len(ids))
            p = os.path.join(BLK, f"{a.prefix}_{lo:09d}_{hi:09d}.h5ad")
            paths.append(p)
            if a.only_block is not None and bi != a.only_block:
                continue
            if os.path.exists(p):
                log(f"  block {bi}: exists, skip"); continue
            if a.phase == "concat":
                raise SystemExit(f"missing block {p}")
            ad = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", X_name="raw",
                obs_coords=ids[lo:hi], var_coords=sel)
            assert ad.shape == (hi - lo, len(sel)), ad.shape
            X = ad.X
            assert np.all(X.data == np.floor(X.data)), "raw layer is not integer counts"
            tmp = f"{p}.tmp{os.getpid()}"
            ad.write_h5ad(tmp)                # write then rename: never a partial block
            os.replace(tmp, p)
            log(f"  block {bi}: {ad.shape[0]:,} cells, nnz={X.nnz:,}, "
                f"{os.path.getsize(p)/1e9:.2f} GB")
            del ad, X

if a.only_block is not None:
    log(f"only-block {a.only_block} done"); raise SystemExit(0)

# ---------------- phase 3: concatenate ----------------
FINAL = os.path.join(OUT, f"{a.prefix}_hvg.h5ad")
if a.phase in ("all", "concat") and not os.path.exists(FINAL):
    from anndata.experimental import concat_on_disk
    log(f"phase 3: concat {len(paths)} blocks -> {FINAL}")
    import h5py
    ref = None
    for q in paths:                      # every block must carry the same 3000 genes
        with h5py.File(q, "r") as h:
            v = h["var"]["_index"][:] if "_index" in h["var"] else h["var"][list(h["var"])[0]][:]
        if ref is None: ref = v
        else: assert np.array_equal(ref, v), f"var mismatch in {q}"
    log(f"  var identical across {len(paths)} blocks ({len(ref)} genes) -> inner == outer")
    TMP = FINAL[:-5] + ".tmp.h5ad"        # must still end .h5ad
    concat_on_disk(paths, TMP, axis=0, join="inner")
    os.replace(TMP, FINAL)
    # every block came back with a positional 0..N-1 obs index, so the merged
    # file would carry 14 duplicate sets of names; soma_joinid is the census's
    # own unique per-cell key
    import subprocess
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "fix_obs_index.py"), "--h5ad", FINAL], check=True)
    log(f"  {FINAL}  {os.path.getsize(FINAL)/1e9:.2f} GB")

import anndata as adm
A = adm.read_h5ad(FINAL, backed="r")
log(f"FINAL {A.shape}  nnz={A.X.nnz if hasattr(A.X,'nnz') else '?'}")
json.dump({"census_build": build, "n_cells": int(A.shape[0]), "n_genes": int(A.shape[1]),
           "n_top_genes": a.n_top_genes, "batch_key": a.batch_key,
           "flavor": "seurat_v3", "layer": "raw counts",
           "filter": vf, "n_terms": len(terms),
           "cell_set": a.cell_set},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
log("OK")
