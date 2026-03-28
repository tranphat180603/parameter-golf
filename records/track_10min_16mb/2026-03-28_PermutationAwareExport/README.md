# 2026-03-28 Permutation-Aware Export

Goal:

- keep the current MLP equalization exporter
- add exact MLP hidden-channel permutations after equalization to improve compressibility with zero functional change

Observed Result:

- Negative result.
- Per-layer permutation search increased the compressed model size instead of reducing it.
- Example run:
  - `Serialized model int6+lzma: 16012396`
  - `Total submission size int6+lzma: 16131670`
  - `final_int6_roundtrip_exact val_bpb: 1.14418017`
  - calibration base was already over the byte budget, so auto-allocation had no slack to spend:
    - `quant_auto_allocate:done no byte budget slack`

Conclusion:

- The local per-layer permutation heuristic destroyed enough global regularity that `lzma` compressed worse.
- This branch was abandoned after the export-stage result was already worse than the equalization control.

Implemented export stack:

- `QUANT_KEEP_LATE_LAYERS_INT8=0`
- `QUANT_FORCE_FP16_PATTERNS=...`
- `QUANT_FORCE_INT8_PATTERNS=...`
- `QUANT_FORCE_INT6_PATTERNS=...`
- `QUANT_MLP_EQUALIZE_ENABLED=1`
- `QUANT_MLP_EQUALIZE_ALPHA=0.5`
- `QUANT_MLP_EQUALIZE_METRIC=rms`
- `QUANT_MLP_EQUALIZE_MIN_SCALE=0.25`
- `QUANT_MLP_EQUALIZE_MAX_SCALE=4.0`
- `QUANT_MLP_PERMUTE_ENABLED=1`
- `QUANT_MLP_PERMUTE_METRIC=rms`
- `QUANT_MLP_PERMUTE_SEARCH=1`
- `QUANT_AUTO_ALLOCATE_ENABLED=1`
- `QUANT_AUTO_ALLOCATE_TOKENS=262144`
- `QUANT_AUTO_ALLOCATE_MAX_GROUPS=8`
- `QUANT_TARGET_TOTAL_BYTES=16000000`
- `QUANT_SENSITIVITY_ENABLED=0` for the paid run; auto-allocation handled the calibration pass instead

Run that produced the negative result:

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
QUANT_MLP_PERMUTE_ENABLED=1 QUANT_MLP_PERMUTE_METRIC=rms QUANT_MLP_PERMUTE_SEARCH=1 \
QUANT_AUTO_ALLOCATE_ENABLED=1 QUANT_AUTO_ALLOCATE_TOKENS=262144 QUANT_AUTO_ALLOCATE_MAX_GROUPS=8 \
QUANT_TARGET_TOTAL_BYTES=16000000 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-28_PermutationAwareExport/train_gpt.py
```

Status:

- Kept as a documented negative result.
- Not recommended for further paid runs in its current form.
