# 2026-03-27 Better Quantization Allocation

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Goal:

- improve artifact quality under the same 16 MB budget by allocating precision per tensor more intelligently
- keep the first pass export-only so training dynamics stay comparable to the March 23 baseline
- add exact MLP hidden-channel equalization before quantization, exploiting the winner's `LeakyReLU(x)^2` MLP so float behavior is preserved

What This Branch Established:

- The built-in `late_layers_int8` heuristic was not worth it. It pushed the artifact over cap without improving final bpb.
- The sensitivity profiler was useful: fragile groups were mostly MLPs, not the late-layer heuristic targets.
- The next natural extension is bank-aware late QAT, because the profiler's fragile tensors are bank-derived MLP slices while the winner's existing late QAT only touches `CastedLinear` weights.
- This branch became the base for the later dated export experiments:
  - `2026-03-28_MLPEqualizedExport`
  - `2026-03-28_PermutationAwareExport`

Main knobs:

- `QUANT_KEEP_LATE_LAYERS_INT8=0`
- `QUANT_FORCE_FP16_PATTERNS=...`
- `QUANT_FORCE_INT8_PATTERNS=...`
- `QUANT_FORCE_INT6_PATTERNS=...`
- `QUANT_MLP_EQUALIZE_ENABLED=1`
- `QUANT_MLP_EQUALIZE_ALPHA=0.5`
- `QUANT_MLP_EQUALIZE_METRIC=rms`
- `QUANT_MLP_EQUALIZE_MIN_SCALE=0.25`
- `QUANT_MLP_EQUALIZE_MAX_SCALE=4.0`
- `QUANT_SENSITIVITY_ENABLED=1`
- `QUANT_SENSITIVITY_TOKENS=1048576`
- `QUANT_SENSITIVITY_MAX_GROUPS=12`
- `BANK_QAT_ENABLED=1`
- `BANK_QAT_BLOCKS=0,1,2,3`
- `BANK_QAT_INCLUDE_MLP=1`
- `BANK_QAT_INCLUDE_ATTN=0`
- `BANK_QAT_THRESHOLD=-1`

Suggested profiler run from repo root:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QUANT_KEEP_LATE_LAYERS_INT8=0 \
QUANT_MLP_EQUALIZE_ENABLED=1 QUANT_MLP_EQUALIZE_ALPHA=0.5 \
QUANT_MLP_EQUALIZE_METRIC=rms QUANT_MLP_EQUALIZE_MIN_SCALE=0.25 QUANT_MLP_EQUALIZE_MAX_SCALE=4.0 \
QUANT_SENSITIVITY_ENABLED=1 QUANT_SENSITIVITY_TOKENS=1048576 QUANT_SENSITIVITY_MAX_GROUPS=8 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-27_BetterQuantAllocation/train_gpt.py
```

Status:

- Useful as a profiler / staging branch.
- The newest thing to try from this folder is bank-aware late QAT on top of export equalization.

Suggested bank-aware late-QAT run from repo root:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QUANT_KEEP_LATE_LAYERS_INT8=0 \
QUANT_MLP_EQUALIZE_ENABLED=1 QUANT_MLP_EQUALIZE_ALPHA=0.5 \
QUANT_MLP_EQUALIZE_METRIC=rms QUANT_MLP_EQUALIZE_MIN_SCALE=0.25 QUANT_MLP_EQUALIZE_MAX_SCALE=4.0 \
BANK_QAT_ENABLED=1 BANK_QAT_BLOCKS=0,1,2,3 BANK_QAT_INCLUDE_MLP=1 BANK_QAT_INCLUDE_ATTN=0 \
QUANT_SENSITIVITY_ENABLED=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-27_BetterQuantAllocation/train_gpt.py
```
