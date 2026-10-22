# step1 — CellANOVA: embedding → cell × gene counts

Recovers a count-scale cell × gene matrix from a cell embedding, by taking
CellANOVA's **main effect** — the part of expression the embedding explains once
the per-batch offsets are averaged out.

**The same four scripts serve both embeddings.** scGPT and Harmony differ only
in `--embedding`, `--mmat` and `--tag`; nothing in the code branches on which
one it is.  That is the point: the comparison between the two latents is a
comparison of latents, not of pipelines.

Copied verbatim from `scripts/poisson/task8_scGPTemb2gex/` (a–d, and the
Harmony drivers) and `scripts/poisson/task9_scGPTemb/` (the scGPT drivers).

| | file here | original |
|---|---|---|
| a | `step1a_make_clipgroups.py` | `t8/make_clipgroups.py` |
| b | `step1b_cellanova_fitM.py` | `t8/cellanova_fitM_stream.py` |
| c | `step1c_cellanova_counts.py` | `t8/cellanova_counts_shard.py` |
| d | `step1d_verify_counts.py` + `.sub` | `t8/verify_counts_matrix.py` |
| e | `step1e_observed_gene_sums.py` | (new -- see below) |

Drivers, one pair per embedding:

| embedding | fit | counts |
|---|---|---|
| Harmony | `step1_harmony_fitM_and_init.sub` | `step1_harmony_counts_array.sub` |
| scGPT | `step1_scgpt_pca_and_fitM.sub` | `step1_scgpt_counts_array.sub` |

## The model

    lnr    = (C @ Mmat) * sd_g + mu_g        main effect, lognorm scale
    clip   -> [0, ceil[group, gene]]         per cell-type ceiling
    counts = expm1(lnr) * L_c / target_sum   back to counts
    renorm -> each cell's HVG total matched to its OBSERVED HVG total

`C` is the (N × d) embedding, `Mmat` the (d × G) main-effect matrix, `mu_g`/`sd_g`
the per-gene lognorm moments, `L_c` the cell's library size.

## a. Clip groups — `step1a_make_clipgroups.py`

Integer codes of `obs.cell_type`, one per cell.  The ceiling in step c is taken
per (cell type, gene), so a gene can be reconstructed as high as the most
extreme cell of its OWN type but no higher — a global ceiling would let a rare
high-expressing type raise the cap for every other type.  Built inside the fitM
job rather than interactively, because the interactive shell is capped at 2 G.

## b. fit_M — `step1b_cellanova_fitM.py`

CellANOVA's `fit_M` by streaming accumulation, instead of writing a 64.6 GB
`layers['scale']` matrix to disk.  Per integration level x (here `dataset_id`),

    B_x  = lstsq(C_x, S_x) = pinv(C_x^T C_x) @ (C_x^T S_x)
    Mmat = mean over good levels of B_x

and both `C_x^T C_x` (d × d) and `C_x^T S_x` (d × G) are sums over cells, so they
accumulate blockwise: 110 levels × (50×50 + 50×3000) floats ≈ 133 MB in RAM
against a 64.6 GB memmap.

The same single pass also collects everything step c needs — `obs_tot` (observed
HVG row sums) and `ceil_tab` (per clip-group × gene max of the observed lognorm)
— so the matrix is read once, not three times.

Outputs: `<prefix>_<tag>_Mmat_dataset_id.npy`, `<prefix>_ceiling_table.npy`,
`observed_hvg_libsize.npy`.

**Note for anything that resamples the data:** fit_M regresses REAL expression on
the latent.  Refitting `Mmat` against a noisier (e.g. simulated) latent shrinks
the coefficients, so a simulation must reuse the stored `Mmat`, not refit it.

## c. Counts — `step1c_cellanova_counts.py`

Pass 2 is row-independent given `Mmat`, the ceiling table and `obs_tot`, so it
runs as a 16-task array writing DISJOINT slices of one preallocated `.npy` — no
concatenation, no per-shard files.  `--init` preallocates the output once before
the array starts.  Only the renormalised matrix is kept.

**Do not revert the write path to mmap.**  An earlier version mapped the shared
output and let each task write its slice; on this cluster a cross-node shared
mmap read back as zeros while every per-shard self-check still passed.  The
current code uses positional write + fsync.

## d. Verify — `step1d_verify_counts.py`

A whole-matrix check in a separate process after every task has exited, with
ordinary reads over all N rows.  Deliberately not the per-shard self-check: the
failure it exists to catch (job 1423187) passed the per-shard check on two tasks
while the file on disk was empty.

## e. Observed per-gene totals — `step1e_observed_gene_sums.py`

NOT a copy: nothing in the repo produced `observed_hvg_genesum.npy`, which step3
requires as `--gene-obs` (the denominator of its calibration line).  The file on
disk had been made ad hoc, so step3 could not be re-run from source.  This
rebuilds it:

    out[g] = sum over all cells of X[c, g]       X = the h5ad's raw counts

streamed over the CSR's `data`/`indices` with `bincount` -- `indptr` is never
needed, since a column sum does not care which row a value came from.  Its total
must equal `sum(observed_hvg_libsize.npy)` (the same sum along the other axis),
which `--check` asserts.

Verified against the file that was already on disk: identical for all 3,000
genes, max abs diff 0.0, 44 s over 6.28e8 nonzeros.

```
python step1e_observed_gene_sums.py \
  --h5ad  <MAT>/CD8T_all_hvg.h5ad \
  --out   <T8>/observed_hvg_genesum.npy \
  --check <T8>/observed_hvg_libsize.npy
```

Run it after step1b (which writes the libsize file) and before any step3 driver.

## Outputs

| embedding | Mmat | counts |
|---|---|---|
| Harmony | `all_CD8T/allcd8_harmony_Mmat_dataset_id.npy` | `all_CD8T/main_effect_counts_perCTclip.npy` |
| scGPT | `task9_scGPTemb/allcd8_scgptpca_Mmat_dataset_id.npy` | `task9_scGPTemb/main_effect_counts_scgptpca_perCTclip.npy` |

Both are over the full 5,381,464-cell cohort; the UMI filter
(`allcd8_filtered_rowidx.npy`, step0b) is applied downstream by row index.

`step1_scgpt_pca_and_fitM.sub` runs two things: step f of step0_prep (PCA of the
scGPT embedding) and then fit_M on it, which is why that file appears in both
stages.
