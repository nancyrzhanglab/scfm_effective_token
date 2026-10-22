# scGPT whole-human checkpoint

Downloaded 2026-09-01 from the "whole-human (recommended)" link in
https://github.com/bowang-lab/scGPT#pretrained-scGPT-checkpoints
Google Drive folder: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
(Drive folder is named `scGPT_human`; contents flattened into this directory.)

| file | bytes | md5 |
|---|---|---|
| args.json      |      1300 | (text) |
| best_model.pt  | 205385258 | 9922ec94305126e6e4f9c1575cf493ae |
| vocab.json     |   1317639 | 8efa7f3ca6949e1facdaa47e539e9855 |

Verified with torch 2.5.1: state dict loads, 163 tensors, 51.3 M params,
12 transformer encoder layers, gene embedding (60697, 512) — consistent with
args.json (nlayers 12, embsize 512, nheads 8, d_hid 512).
Vocab = 60697 tokens incl. <pad>, <cls>, <eoc>.

Training config of note: binned input (n_bins 51), max_seq_len 1200,
pretrained on CELLxGENE census human (May 2023), fp16, MVC + generative training.
`args.json` sets `fast_transformer: true`, i.e. the released weights were trained
with the FlashAttention path; scGPT falls back to nn.TransformerEncoder if
flash-attn is unavailable.
