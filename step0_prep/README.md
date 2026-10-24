# step0 — the CD8 cohort and its scGPT embedding


## a. Download — `step0a_data_download.py`

Queries the Census for every CD8 T cell and downloads a cells x 3000-HVG raw
count matrix in three checkpointed phases (`--phase hvg|download|concat|all`):

* **HVG** — 3,000 genes by `seurat_v3` over all 5.4 M CD8 cells x all genes,
  batched by `dataset_id` so genes that merely separate datasets cannot win.
* **download** — the cells x HVG raw counts, in restartable blocks of 400 k.
* **concat** — the blocks are joined on disk into one h5ad.

The cell set is `cell_type_ontology_term_id in <terms> and is_primary_data ==
True` (`--cell-set primary`, the default).  The terms come from
`--terms .../cd8_ontology_terms.json`, a fixed list of 17 CD8 ontology IDs
(`CL:0000625` CD8-positive alpha-beta T cell and its descendants).  That file is an INPUT, not something any script generates; a copy now sits
beside the scripts as `cd8_ontology_terms.json`.

**The build is pinned.**  `--census-version` defaults to `2025-11-08`, the build
this project used, and the script asserts the build it opened matches.  Without
that pin `open_soma()` takes whatever build is current and the cohort changes
silently with each Census release.  `--census-version latest` restores the old
behaviour.

Output: `CD8T_all_hvg.h5ad`, `hvg_var.csv`, `hvg_feature_ids.txt`,
`cd8_obs.parquet` — **5,381,464 cells, 110 datasets**.

## b. UMI filter — `step0b_umi_filter.py`

Two filters, applied in this order, and the order matters:

1. drop the assays that are not UMI whole-transcriptome:
   **Smart-seq2** (reads, not molecules — every PCR duplicate counted again,
   median library 678 k vs 3.8 k) and **BD Rhapsody Targeted mRNA** (a few
   hundred genes by design, median 85 genes detected).
2. then recount dataset sizes and drop datasets with **< 1000 cells**, so a
   dataset that falls below the threshold only after losing its Smart-seq2
   cells is also dropped.

Writes `allcd8_filtered_rowidx.npy` (the row index into the parent h5ad) and
`allcd8_filter.json`:

```
n_start 5,381,464   ->   n_kept 5,353,450      (28,014 cells dropped, 0.52%)
datasets_start 110  ->   datasets_kept 94
```

The script prints a step-by-step table of cells / datasets / donors / assays /
cell types / tissues surviving each step.

**Why these two assays:** the library-size distribution is bimodal, and the
discriminator is counts per gene, not genes detected

## c. Materialise — `step0c_extract_filtered.py`

Slices the parent h5ad by `allcd8_filtered_rowidx.npy` into
`CD8T_all_hvg_filtered.h5ad`, keeping the same 3,000 gene columns so every
parent-derived array (PCA scores, embeddings) stays sliceable by the same index.


## d. scGPT input — `step0d_scgpt_prep.py`

Turns the HVG count matrix into what scGPT tokenises.

Cells with NO expressed usable gene cannot be tokenised; they are dropped and
their positions recorded in `allcd8_scgpt_rowidx.npy`, so embedding rows can
always be mapped back to h5ad row order instead of silently misaligning.

Outputs: `allcd8_counts_scgpt.npz`, `allcd8_genes_scgpt.txt`,
`allcd8_scgpt_rowidx.npy`, `allcd8_scgpt_input.json`.

## e. The embedding — `step0e_scgpt_embed.py`

Runs the 51 M-parameter scGPT checkpoint over the cells and writes one 512-d
vector per cell. 

Output: shards under `data/scGPT_related/scGPT_emb/CD8T_all/`.

## f. PCA of the embedding — `step0f_scgpt_pca.py`

The 512-d embedding is 11 GB, so it is reduced in two streaming passes over the
shards: pass 1 accumulates the sum and X^T X (a 512 x 512 covariance), pass 2
centres and projects onto the top 50 eigenvectors.  Covariance PCA, centred,
unscaled.  Eigenvector signs are fixed (largest-|value| entry made positive) so
reruns are bit-comparable.

Run as step 1 of `task9_scGPTemb/cellanova_scgptpca.sub`:

```
--emb-dir data/scGPT_related/scGPT_emb/CD8T_all \
--out-dir results/poisson/task9_scGPTemb --prefix allcd8_scgpt \
--n-pcs 50 --expect-n 5381464
```

Outputs `allcd8_scgpt_pca50.npy` (5,381,464 x 50 float32, 1.08 GB) plus
`allcd8_scgpt_pca_mean.npy` and the loadings — this 50-d latent is THE scGPT
embedding every later step uses, and the basis the CellANOVA main effect
(`Mmat`) is fitted in.


## g. PCA of the counts — `step0g_pca_counts.py`

The input Harmony corrects, and the comparison baseline for the scGPT latent.
The z-scored 5.35 M x 3000 matrix would be ~60 GiB dense, so it is never
materialised: `log1p(0) = 0` means the lognorm layer has exactly the counts'
sparsity, so the covariance of the z-scored data follows in closed form from two
sparse accumulations,

    G2 = Ln^T Ln,  s1 = colsum(Ln)  ->  cov = (G2 - n mu mu^T)/(n-1)
    corr = cov / (sd sd^T)

and `corr`'s top eigenvectors ARE the loadings `sc.pp.scale + sc.pp.pca` would
give.  Run with `--normaliser raw_sum`, `--n-comps 50`, `--prefix allcd8f` on
the step-c output: 5,353,450 cells, 94 datasets, 4,707 donors, 61 zero-variance
genes, 10.0% of variance in 50 PCs.

Outputs `<prefix>_pca50.npy`, `<prefix>_pca_loadings.npy`, `<prefix>_pca.json`.
Note `step0g_pca_counts.sub` is the original `prep_allcd8_filtered.sub`: it runs
step c, then this PCA with `--prefix allcd8f`, then submits step h.

## h. Harmony — `step0h_harmony.py`

`harmonypy.run_harmony` on those 50 PCs, batch key `dataset_id`.  `donor_id`
(2,871 levels) is available but is biological as well as technical, and the
per-level correction is ill-posed when many levels hold a handful of cells.

Outputs `<prefix>_harmony_dataset_id.npy` (N x 50 float32),
`..._eta2.csv`, `..._json`.
