# 2026-03-28 Middle-Heavy FFN

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Goal:

- keep the March 23 training recipe, exporter, and legal TTT protocol
- move MLP capacity toward the middle layers instead of using a uniform `MLP_MULT=3.0`
- keep the average MLP width close to the baseline while capping the widest layer at `3.5x` to limit step-time blowup

Default schedule in this branch:

```text
2.75,2.75,3.0,3.0,3.25,3.5,3.25,3.0,3.0,2.75,2.75
```

For `MODEL_DIM=512`, that gives active per-layer MLP widths:

```text
1408,1408,1536,1536,1664,1792,1664,1536,1536,1408,1408
```

Implementation notes:

- Training groups layers by shared MLP width and stores one bank tensor per width bucket.
- The default aligned schedule collapses into buckets `1408x4`, `1536x4`, `1664x2`, `1792x1`.
- Export only serializes the active per-layer MLP slices.
- Eval reconstructs the grouped banks and uses the same active-width schedule.

Why this branch exists:

- The failed doc-aware / LoRA / permutation lanes suggest the next credible improvement has to come from model quality, not more eval tricks.
- The quant profiler kept pointing at MLP tensors as the fragile/high-value part of the model.
- This is the cleanest architecture change that targets that observation directly.

Main knobs:

- `MLP_MULTS=...`
- `MLP_MULT=3.0` as a fallback when `MLP_MULTS` is empty
- `MLP_DIM_ALIGN=128` to keep widths tensor-core friendly

Suggested full run from repo root:

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
MLP_MULTS=2.75,2.75,3.0,3.0,3.25,3.5,3.25,3.0,3.0,2.75,2.75 \
MLP_DIM_ALIGN=128 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-28_MiddleHeavyFFN/train_gpt.py
```

What to watch in the logs:

- `mlp_schedule ...`
- `step_avg:...ms`
- `final_int6_sliding_window_exact ...`
- `legal_ttt_exact ...`

Status:

- New architecture branch.
- Not validated yet.
