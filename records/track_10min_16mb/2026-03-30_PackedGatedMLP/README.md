# Packed Gated MLP on March 23 Baseline

This branch starts from the March 23 record stack and changes exactly one core modeling component:

- replace the `LeakyReLU(0.5)^2` MLP with a packed gated MLP
- keep the March 23 optimizer, training schedule, quantization path, and legal TTT config unchanged

## MLP Change

Old MLP:

```python
x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5)
y = F.linear(x.square(), down_w)
```

New packed gated MLP:

```python
vg = F.linear(x, vg_w)
v, g = vg.chunk(2, dim=-1)
g = 2.0 * torch.sigmoid(g)
y = F.linear(v * g, down_w)
```

To keep the MLP budget matched to the March 23 recipe, the hidden width is:

```python
mlp_hidden_dim = int((2.0 / 3.0) * mlp_mult * model_dim)
```

With `MLP_MULT=3.0` and `MODEL_DIM=512`, that gives `1024`, so the MLP stays at roughly the same parameter count and matmul budget as the original 3x MLP.

## Bank Layout

The 4 bank structure is preserved:

- `qo_bank`
- `kv_bank`
- `mlp_vg_bank`
- `mlp_down_bank`

`mlp_vg_bank[i]` stores the packed value+gate projection for layer `i`, and export/reload splits it into:

- `blocks.{i}.mlp.value.weight`
- `blocks.{i}.mlp.gate.weight`
- `blocks.{i}.mlp.proj.weight`

This keeps the quantization/export path classifying them as MLP tensors.

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
records/track_10min_16mb/2026-03-30_PackedGatedMLP/train_gpt.py
```

## What To Watch

- `step_avg`
- `DIAGNOSTIC post_ema val_bpb`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `legal_ttt_exact`

This branch is a pure MLP-core ablation. If it wins, it means the March 23 stack was bottlenecked by the MLP block itself rather than by evaluation or quantization tricks.
