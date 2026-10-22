# Environments and external artefacts

Three conda environments, captured from the ones the pipeline actually ran in
on 2026-10-21.  Each driver hard-codes an absolute interpreter path under
`/home/stat/nzh/team/emiliac/miniconda3/envs/<env>/bin/python`; on another
machine those paths have to be rewritten.

| env | used by | key versions |
|---|---|---|
| `cxgcensus` | step0a-f, step2 (piles, Ward, MetaCell-1), step3-5 | python 3.11.16, cellxgene-census 1.18.0, tiledbsoma 2.3.0, scanpy 1.10.1, anndata 0.10.7, numpy 1.26.4, scipy 1.13.0, pandas 2.2.2, SEACells 0.3.3, leidenalg 0.10.2, igraph 0.11.4, statsmodels 0.14.2, gseapy 1.1.2 |
| `cellanova` | step0g-h, step1, step2 Leiden | python 3.9.23, harmonypy 0.0.10, scanpy 1.10.3, anndata 0.10.8, numpy 1.25.2, scipy 1.13.1, torch 2.0.1, leidenalg 0.10.2, igraph 0.11.8 |
| `scgpt` | step0e (the embedding only) | python 3.10.21, scgpt 0.2.4, torch 2.3.0+cu121, numpy 2.1.3, scanpy 1.11.5, anndata 0.11.4, torchtext 0.18.0 |

Two capture formats per env, because neither is complete on its own:

* `<env>.conda-explicit.txt` — `conda list --export`, the conda-installed half.
* `<env>.pip-freeze.txt` — every installed distribution with its version,
  including the pip-only ones that matter most here (cellxgene-census,
  harmonypy, SEACells, scgpt).

`install_scgpt_env.sh` and `README_env.md` are the original build recipe for the
`scgpt` env, copied from `scripts/scgpt/`.

**The three envs are on three python versions** (3.9.23 / 3.11.16 / 3.10.21)
and three numpy lines (1.25 / 1.26 / 2.1).  That is not an
accident to be normalised away: scgpt 0.2.4 needs the numpy-2 line while
cellanova's torch 2.0.1 does not tolerate it.  Nothing is passed between the
envs except `.npy` / `.h5ad` / `.csv` files, so the split is safe, but a single
merged environment is not a drop-in replacement.

## The scGPT checkpoint

Not copied here — `best_model.pt` is 205 MB.  It lives at
`data/scGPT_related/models/full/` and is pinned by hash:

```
9922ec94305126e6e4f9c1575cf493ae  best_model.pt      (205,385,258 bytes)
8efa7f3ca6949e1facdaa47e539e9855  vocab.json         (1,317,639 bytes)
```

`scgpt_checkpoint.md5` holds those, `scgpt_checkpoint_args.json` the model
config, and `scgpt_checkpoint_provenance.md` where it came from (the
"whole-human (recommended)" link on the scGPT repo, downloaded 2026-09-01) plus
what it contains: 163 tensors, 51.3 M params, 12 encoder layers, gene embedding
(60697, 512), vocab 60697 tokens.  `verify_scgpt_checkpoint.py` re-checks all of
that against a downloaded copy.

Note `args.json` sets `fast_transformer: true` — the released weights were
trained on the FlashAttention path, and scGPT falls back to
`nn.TransformerEncoder` when flash-attn is unavailable.  step0e ran on CPU, i.e.
on the fallback path.

## The Census build

`step0a_data_download.py` now pins `--census-version 2025-11-08` and ASSERTS the
build it opened matches.  Without the pin, `open_soma()` takes whatever build is
current and the cohort changes silently with each Census release — the
5,381,464 cells / 110 datasets in every number this project reports are specific
to that build.  `--census-version latest` restores the old behaviour.
