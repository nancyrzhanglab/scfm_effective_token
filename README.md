# Effective tokens in single-cell foundation-model corpora

The pipeline behind the question *"how many of a scRNA-seq corpus's nominal
tokens actually carry measurable, non-redundant signal?"*, run on 5.38 M CD8
T cells from the CELLxGENE Census.

The short version: a cell × gene count is a **usable measurement** only if it is
measured precisely enough (Poisson `CV <= 0.25`, equivalently a pooled count
`Mu >= 16`) **and** sits in a metacell homogeneous enough for pooling to be
honest (`within/total <= 0.25`). Pooling cells into metacells trades the first
against the second, so the usable total is maximised at an interior metacell
size, not at either extreme.

## Stages

| | what it does |
|---|---|
| `step0_prep/` | pull every CD8 T cell from the Census, keep the UMI-based ones, embed with scGPT, and with PCA+Harmony as the comparison |
| `step1_recover_gex/` | CellANOVA: turn a cell embedding back into a count-scale cell × gene matrix |
| `step2_construct_metacell/` | nested metacell ladders by three methods (recursive Leiden, SEACells, MetaCell-1) on both embeddings |
| `step3_estimate_noise_and_mixing/` | per gene and per metacell size: Poisson CV, and the law-of-total-variance split into within / between |
| `step4_apply_threshold/` | how each criterion behaves against metacell size, one criterion at a time |
| `step5_calculate_eff_tokens/` | apply both criteria and bank the usable measurements |

Each stage has its own README with the reasoning that is not recoverable from
the code — why a step is a fork of an earlier one, which failure a guard exists
to catch, and what the measured numbers were.

## The cohort

| | cells | datasets |
|---|---|---|
| CD8 T cells, Census build 2025-11-08, `is_primary_data` | 5,381,464 | 110 |
| after dropping Smart-seq2 and BD Rhapsody Targeted, then datasets < 1000 cells | **5,353,450** | 94 |

All tissues, not blood only — 44 `tissue_general` levels, blood 76.7%. The
17 CD8 cell-type ontology terms that define the cohort are in
`step0_prep/cd8_ontology_terms.json`.

## Running it

**Paths are cluster-specific.** Every script defaults to absolute paths under
`/home/mnt/weka/nzh/nzhanglab/project/emiliac/FM_effective/`. Most take
`--in-dir` / `--out-dir` / `--out-prefix`; two in `step0_prep` hard-code the
output directory as a module constant and need editing. The per-stage READMEs
say which.

**Three conda environments**, captured in `env/` as both a conda export and a
full package listing. They are on three different Python versions and three
numpy lines, which is load-bearing rather than accidental — `env/README.md`
explains why a single merged environment is not a drop-in replacement. Nothing
is passed between them except `.npy` / `.h5ad` / `.csv` files.

**The scGPT checkpoint** is not in this repo (205 MB). `env/` pins it by md5
with its provenance and a verification script.

**The batch drivers are not here either.** The jobs ran on a UGE cluster; the
`.sub` files, with their memory sizing and shard counts, live beside the
original scripts in the lab tree.

## A caveat worth reading before quoting a number

The usable-token total is **monotone in both cutoffs by construction**, so any
view of it is maximised at the loosest thresholds. That is a property of the
definition, not a finding. `step5b_usable_grid.py` has options
(`--relative`, `--marginal`, `--annotate-cost`) that exist specifically to stop
the grid being read as "looser is better".
