# True Packed Int6 Export on March 23 Baseline

This branch starts from the March 23 record stack and changes exactly one thing:

- replace the existing "int6 stored in int8 tensors" export path with true packed 6-bit storage for int6 tensors

Training, legal TTT, architecture, optimizer, and quantization policy are otherwise unchanged.

## Why This Exists

The March 23 exporter quantizes many large MLP and attention tensors to int6, but stores the quantized values in `torch.int8` tensors before LZMA compression. This branch tests whether packing those int6 codes into a real 6-bit payload improves compressed artifact size enough to matter.

This is a pure exporter ablation:

- no retraining changes
- no new mixed-precision policy
- no activation-aware calibration
- no architecture changes

## Export Change

For tensors classified as int6:

1. quantize per row exactly as before
2. remap signed codes `[-31, 31]` to unsigned codes `[0, 62]`
3. pack 4 values into 3 bytes
4. store packed bytes plus the original shape metadata
5. unpack during dequantization before final eval

Int8 tensors and passthrough tensors remain unchanged.

## Run Command

Use the exact March 23 config, only with the new script path:

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
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-30_TruePackedInt6Export/train_gpt.py
```

## What To Watch

- `Serialized quant payload raw`
- `Serialized model int6+lzma`
- `Total submission size int6+lzma`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `legal_ttt_exact`

Success means:

- compressed artifact bytes go down materially
- and final dequantized metrics stay flat or improve

Failure means:

- bit-packing reduced byte-alignment/compressibility enough that LZMA erased the theoretical 6-bit saving
- or metadata overhead canceled the gain
