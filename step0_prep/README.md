# step0 — the CD8 cohort and its scGPT embedding

Three scripts, copied verbatim from `scripts/poisson/task8_scGPTemb2gex/`, in
the order they were run.  The names of the originals are given so the provenance
is traceable; nothing in the code was changed.

| run order | file here | original |
|---|---|---|
| a | `step0a_data_download.py` | `task8_scGPTemb2gex/download_all_cd8.py` |
| b | `step0b_umi_filter.py` | `task8_scGPTemb2gex/filter_impact_allcd8.py` |
| c | `step0c_extract_filtered.py` | `task8_scGPTemb2gex/extract_rows_h5ad.py` |
| d | `step0d_scgpt_prep.py` + `.sub` | `task8_scGPTemb2gex/prep_scgpt_allcd8.py` |
| e | `step0e_scgpt_embed.py` + `.sub` | `task8_scGPTemb2gex/embed_scgpt_task8.py` |
| f | `step0f_scgpt_pca.py` | `task9_scGPTemb/pca_scgpt_allcd8.py` |
| g | `step0g_pca_counts.py` + `.sub` | `task8_scGPTemb2gex/pca_cd8.py` |
| h | `step0h_harmony.py` + `.sub` | `task8_scGPTemb2gex/harmony_cd8.py` |

Steps d-f are the scGPT cell embedding; g-h are the Harmony one, the baseline
it is compared against.  The two branches are independent after step c.

Steps d-f are the scGPT cell embedding.  Note they run on the UNFILTERED
`CD8T_all_hvg.h5ad` (all 5,381,464 cells), not on the step-c output: the
embedding covers the whole cohort and the UMI filter is applied later, by row
index, wherever a measurement needs it.

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
discriminator is counts per gene, not genes detected — the three regimes are
separated in `plot_cd8_assay_groups.py` (not copied here) and recorded in
`cd8_assay_regimes.json`.  A Poisson count model needs molecules, so read-based
and panel-based cells have to go.

## c. Materialise — `step0c_extract_filtered.py`

Slices the parent h5ad by `allcd8_filtered_rowidx.npy` into
`CD8T_all_hvg_filtered.h5ad`, keeping the same 3,000 gene columns so every
parent-derived array (PCA scores, embeddings) stays sliceable by the same index.

The original driver `prep_allcd8_filtered.sub` then ran PCA
(`pca_cd8.py --normaliser raw_sum`) and Harmony on the filtered file; those are
later steps, not part of step0.

## Paths

All three scripts carry absolute paths to
`results/poisson/task8_scGPTemb2gex/` as defaults.  `step0a_data_download.py` takes
`--out`, but the other two hard-code `D = .../task8_scGPTemb2gex`; edit that
line if this stage is ever re-run into a different directory.


## d. scGPT input — `step0d_scgpt_prep.py`

Turns the HVG count matrix into what scGPT tokenises: the subset of genes whose
symbols are in the model vocabulary (`data/scGPT_related/models/full/vocab.json`),
written as a CSR npz plus a gene list.  Two-pass streaming (count nonzeros, then
fill preallocated arrays) because holding the block list and the final matrix at
once would need ~7 GB twice.

Cells with NO expressed usable gene cannot be tokenised; they are dropped and
their positions recorded in `allcd8_scgpt_rowidx.npy`, so embedding rows can
always be mapped back to h5ad row order instead of silently misaligning.

Outputs: `allcd8_counts_scgpt.npz`, `allcd8_genes_scgpt.txt`,
`allcd8_scgpt_rowidx.npy`, `allcd8_scgpt_input.json`.
The `.sub` launches step e on a clean exit, so the array never starts from a
half-written prep.

## e. The embedding — `step0e_scgpt_embed.py`

Runs the 51 M-parameter scGPT checkpoint over the cells and writes one 512-d
vector per cell.  Submitted as a 32-task array (`-t 1-32 -tc 8`, 8 slots x 1.5 G
on hpc3.q, CPU); each task shards further into fixed global blocks named by their
cell range, so a finished block is skipped on restart and hitting the wall clock
costs at most one partial block.  The model is built and loaded once per job and
reused across blocks.

Determinism: scGPT's collator uses `sampling=True`, so a cell with more expressed
genes than `max_length` would get a RANDOM subset.  On the HVG matrix no cell
comes close, and the script ASSERTS that per block rather than assuming it.

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

What the script MEASURES rather than asserts: per PC, the fraction of variance
explained by the batch factor (one-way ANOVA eta^2) before and after correction,
with cell-type eta^2 on the same axes as a guard against over-correction. For
the UMI-based run (`allcd8f`, 5,353,450 cells, 94 batches, 50 comps, 10 iters,
seed 0, 152 min):

```
batch eta^2   0.2542 -> 0.0073      (what Harmony exists to shrink)
cell-type     0.0675 -> 0.0309      (should not collapse, and did not)
```

Outputs `<prefix>_harmony_dataset_id.npy` (N x 50 float32),
`..._eta2.csv`, `..._json`.

**Two runs exist.**  `step0h_harmony.sub` is the UMI-based one
(`--prefix allcd8f`, 4 x 13 G on mem.q, sized after the unfiltered run peaked at
43.8 G of 48 G).  An unfiltered `--prefix allcd8` run over all 5,381,464 cells
also exists (`harmony_all_cd8.sub` in the original directory, 3 x 16 G,
batch eta^2 0.2606 -> 0.0069 in 212 min); it is the one that matches the scGPT
embedding's coverage cell-for-cell, so it is what a cross-embedding comparison
on the full cohort uses.

**Harmony preserves the PCA axes.** `run_harmony` returns `Z_corr` in the same
50 coordinates it was given — it subtracts per-cluster batch offsets and does
not rotate the basis.  That is why a stored main-effect matrix fitted on these
axes stays valid across re-runs, and why `H_obs - P_obs` is exactly the
per-cell correction.
