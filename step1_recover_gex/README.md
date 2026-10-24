# step1 — CellANOVA: embedding → cell × gene counts

Recovers a count-scale cell × gene matrix from a cell embedding, by taking
CellANOVA's **main effect** — the part of expression the embedding explains once
the per-batch offsets are averaged out.

**The same four scripts serve both embeddings.** scGPT and Harmony differ only
in `--embedding`, `--mmat` and `--tag`

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
extreme cell of its own type but no higher — a global ceiling would let a rare
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

Outputs: `<prefix>_<tag>_Mmat_dataset_id.npy`, `<prefix>_ceiling_table.npy`,
`observed_hvg_libsize.npy`.

**Note for anything that resamples the data:** fit_M regresses REAL expression on
the latent.  Refitting `Mmat` against a noisier (e.g. simulated) latent shrinks
the coefficients, so a simulation must reuse the stored `Mmat`, not refit it.

## c. Counts — `step1c_cellanova_counts.py`

Pass 2 is row-independent given `Mmat`, the ceiling table and `obs_tot`, so it
runs as a 16-task array writing DISJOINT slices of one preallocated `.npy` — no
concatenation, no per-shard files. 

## d. Verify — `step1d_verify_counts.py`


## e. Observed per-gene totals — `step1e_observed_gene_sums.py`

This rebuilds:

    out[g] = sum over all cells of X[c, g]       X = the h5ad's raw counts
