# 2026-03-28 MLP Equalized Export

Goal:

- keep a dated snapshot of the current export-only MLP equalization path
- preserve the March 23 training and legal TTT recipe while testing exact pre-quantization MLP rescaling

Base:

- `records/track_10min_16mb/2026-03-27_BetterQuantAllocation/train_gpt.py`

Observed Result:

- Essentially a tie with the March 23 baseline, but very slightly worse on final score.
- `legal_ttt_exact`: `1.11932544`
- March 23 reference: `1.11922988`
- Artifact was slightly smaller than March 23:
  - this branch: `15923586` total bytes
  - March 23: `15952368` total bytes

Conclusion:

- Exact MLP equalization slightly improved compressibility.
- It did not produce a real bpb win by itself.
- This branch is kept as the clean control for export-side exact rescaling.

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
- `QUANT_SENSITIVITY_ENABLED=0` for the final paid comparison run

Validated run from repo root:

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
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-28_MLPEqualizedExport/train_gpt.py
```

Notes:

- This is the cleanest control for exact export-only MLP rescaling.
- It is safe to keep around as a reference, but it did not beat the March 23 winner.
