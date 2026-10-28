# Effective tokens in single-cell foundation-model corpora

The pipeline behind the question *"how many of a scRNA-seq corpus's nominal
tokens actually carry measurable, non-redundant signal?"*, run on 5.35 M CD8
T cells from the CELLxGENE Census.


## Stages

| | what it does |
|---|---|
| `step0_prep/` | pull every CD8 T cell from the Census, keep the UMI-based ones, embed with scGPT, and with PCA+Harmony as the comparison |
| `step1_recover_gex/` | CellANOVA: turn a cell embedding back into a count-scale cell × gene matrix |
| `step2_construct_metacell/` | nested metacell ladders by three methods (recursive Leiden, SEACells, MetaCell-1) on both embeddings |
| `step3_estimate_noise_and_mixing/` | per gene and per metacell size: Poisson CV to estimate measurement noise, and the law-of-total-variance split into within / between to measure mixing |
| `step4_apply_threshold_and_characterize/` | how each criterion behaves against metacell size, one criterion at a time, and characterize the passing genes at one threshold pair |
| `step5_calculate_eff_tokens/` | apply both criteria and bank the usable measurements |


## The cohort

| | cells | datasets |
|---|---|---|
| CD8 T cells, Census build 2025-11-08, `is_primary_data`, and retain UMI-based assay and datasets >= 1000 cells | **5,353,450** | 94 |

The 17 CD8 cell-type ontology terms that define the cohort are in
`step0_prep/cd8_ontology_terms.json`.
