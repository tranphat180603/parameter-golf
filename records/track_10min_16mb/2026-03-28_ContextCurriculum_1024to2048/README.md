# 2026-03-28 Context Curriculum 1024->2048

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Goal:

- keep the March 23 winner intact
- improve checkpoint quality per 600 seconds by training cheaper early and longer late
- compare against the same legal TTT and export path as March 23

Why this branch exists:

- repo timing data says `seq_len=2048` is the best fixed point, but `1024` is much cheaper early
- `1024 -> 2048` looks materially better than spending training time at `4096`
- this is a train-time-only change, not another eval trick

Curriculum:

- stage 1: `CURRICULUM_STAGE1_SEQ_LEN=1024`
- stage 2: `CURRICULUM_STAGE2_SEQ_LEN=2048`
- switch at `CURRICULUM_SWITCH_FRACTION=0.25` of the 600-second budget
- warmup prewarms both sequence lengths, then restores weights and optimizer state before timed training

Main knobs:

- `CURRICULUM_ENABLED=1`
- `CURRICULUM_STAGE1_SEQ_LEN=1024`
- `CURRICULUM_STAGE2_SEQ_LEN=2048`
- `CURRICULUM_SWITCH_FRACTION=0.25`

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
CURRICULUM_ENABLED=1 CURRICULUM_STAGE1_SEQ_LEN=1024 CURRICULUM_STAGE2_SEQ_LEN=2048 \
CURRICULUM_SWITCH_FRACTION=0.25 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-28_ContextCurriculum_1024to2048/train_gpt.py
```

What to watch in the logs:

- `curriculum enabled=1 ...`
- `warmup_step:... seq_len:...`
- `curriculum_switch step:... train_seq_len:...`
- `step_avg:...ms`
- `final_int6_sliding_window_exact ...`
- `legal_ttt_exact ...`

Status:

- New training-curriculum branch.
- Not validated yet.
