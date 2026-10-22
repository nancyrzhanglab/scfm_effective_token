# `scgpt` conda env

Built 2026-09-01 by `install_scgpt_env.sh` (submitted via `install_scgpt_env.sub`,
job 1420133).  Log: `scripts/log/scgpt_install_env.log`.

    conda activate scgpt          # /home/stat/nzh/team/emiliac/miniconda3/envs/scgpt

| package | version | why this version |
|---|---|---|
| python   | 3.10        | |
| torch    | 2.3.0+cu121 | newest torch `torchtext` still builds against |
| torchtext| 0.18.0      | EOL; hard-pins `torch==2.3.0`.  scgpt needs its C++ `Vocab` |
| scgpt    | 0.2.4       | |
| scanpy   | 1.11.5      | |
| anndata  | 0.11.4      | matches the rest of the project's h5ad files |
| numpy    | 2.1.3       | |
| scib     | 1.1.7       | used by `scgpt/trainer.py`, `scgpt/utils/util.py` |
| setuptools | <81       | 81 dropped `pkg_resources`, which the older deps import |

`pip check` -> **No broken requirements found.**

## Known issues, and why they are left alone

**`import scvi` fails** (`cannot import name 'SparseDataset' from anndata`).
scgpt pins `scvi-tools<1.0` and pip resolved 0.20.3 (2023), which predates
anndata's removal of `SparseDataset` in 0.11.  Left broken deliberately:
`grep -rn "import scvi" site-packages/scgpt/` returns **nothing** -- scgpt
declares scvi-tools and never imports it.  Fixing it would force anndata <0.10,
breaking scanpy 1.11 and the project's h5ad files.

**No flash-attn.**  `args.json` has `fast_transformer: true`, so the released
weights were trained on the FlashAttention path.  flash-attn needs sm75+; the
only GPUs this cluster can burst are V100 (sm70), so build the model with
`use_fast_transformer=False`.  See the attention-layout note below -- this is
not a free switch.

## Two traps when loading the whole-human checkpoint

Both are caught by `verify_scgpt_checkpoint.py` (job 1420142, `VERIFY OK`).

**1. Fused attention weights.**  The checkpoint stores attention as
`self_attn.Wqkv.{weight,bias}` (flash-attn's fused layout).  `nn.
MultiheadAttention` calls the same tensor `self_attn.in_proj_{weight,bias}`.
A plain `load_state_dict(..., strict=False)` therefore leaves **all 12 attention
layers randomly initialised while reporting success**.  `scgpt.utils.
load_pretrained` does the rename when `use_fast_transformer=False`, but it is
non-strict and silently drops non-matching keys, so it cannot tell you if the
rename ever stops working.  Always assert afterwards that
`model.transformer_encoder.layers[0].self_attn.in_proj_weight` equals the
checkpoint tensor.

**2. `cls_decoder` is random, not pretrained.**  `args.json` has `no_cls: true`
/ `USE_CLS: false`, so there is no classifier in the checkpoint, but
`TransformerModel` always constructs one: 10 tensors, 0.53 M params, randomly
initialised.  Read only when `forward(..., CLS=True)`.  Verified by filling it
with NaN and confirming `_encode` and `forward(CLS=False)` are unchanged and
finite.  **Fine-tune before using as a classifier.**

`flag_encoder.weight` in the checkpoint has no home in `TransformerModel`; it
belongs to the generative pretraining objective (`USE_GENERATIVE_TRAINING: true`)
and is not used for encoding.

## Model flags that reproduce the checkpoint

Inferred from the checkpoint's own key names, not from documentation:

    do_mvc=True, do_dab=False, use_batch_labels=False,
    explicit_zero_prob=False, domain_spec_batchnorm=False, n_cls=1

with `d_model=512, nhead=8, d_hid=512, nlayers=12, nlayers_cls=3, dropout=0.2,
pad_token="<pad>", pad_value=-2, input_emb_style="continuous", n_input_bins=51`
and `ntoken=60697`.

## GPUs

**Expect hours of queue wait.**  `gpu.q` is open to `stat_nzh_team` (`user_lists NONE`, `projects NONE`) and is a
**burst** queue (`complex_values burst=gpu`) -- nodes launch on demand, so
`qstat -g c` shows 0 total slots even though the queue works.  The submit hook
forces total RAM to 58G and caps walltime at **4 hours**; do not set
`m_mem_free`.  Request with `-q gpu.q -l gpu=1`.  `gpuv100.q` is ACL-denied to
this project.  See `gpu_probe.sub` for what hardware actually arrives.

The JSV rewrites GPU requests: it injects `burst=gpu` and `-pe openmp 4`, caps
`h_rt` at 14400 s, and overrides `m_mem_free` to 14G per slot (4 x 14G = 56G).
A job observed on 2026-09-01 waited 2.5 h in `qw` before its instance came up,
so a pending `cannot run in PE openmp` line in `qstat -j` is normal, not an error.
