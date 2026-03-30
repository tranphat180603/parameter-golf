# Late Projection Int8 on March 23 Baseline

This branch starts from the March 23 record stack and changes exactly one exporter policy:

- keep selected late-layer projection tensors at int8 instead of int6

Training, legal TTT, architecture, optimizer, and the rest of the quantization path are unchanged.

## Export Policy

The exporter keeps the last `QUANT_KEEP_LATE_LAYERS_INT8` layers' most sensitive projection-like tensors at int8:

- `blocks.{i}.attn.c_k.weight`
- `blocks.{i}.attn.proj.weight`
- `blocks.{i}.mlp.proj.weight`

All earlier large MLP/attention tensors still use the normal March 23 int6 path, and embeddings/control tensors keep the baseline handling.

Default:

```bash
QUANT_KEEP_LATE_LAYERS_INT8=2
```

## Why This Exists

From first principles, late projection errors are more likely to survive into the final logits than early expansion errors. This branch tests that hypothesis in the narrowest possible way:

- no retraining changes
- no packing changes
- no calibration changes
- no auto-allocation

## Run Command

Use the exact March 23 config, plus the late-projection int8 knob:

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
QUANT_KEEP_LATE_LAYERS_INT8=2 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-30_LateProjInt8/train_gpt.py
```

## What To Watch

- `Serialized model int6+lzma`
- `Total submission size int6+lzma`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `legal_ttt_exact`

Success means:

- dequantized metrics improve enough to justify the extra bytes

Failure means:

- late projection precision was not the real export bottleneck
- or the byte cost of int8 on these tensors outweighs the quality gain
