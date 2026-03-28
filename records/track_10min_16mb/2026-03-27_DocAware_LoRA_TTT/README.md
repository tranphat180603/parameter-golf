# 2026-03-27 Doc-Isolated Eval

Work-in-progress experiment based on:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Main change:

- replace fixed 32K-token score-first SGD TTT with document-aware, score-only evaluation
- keep the original chunk-SGD TTT path behind `TTT_MODE=sliding_sgd`
- use BOS-based document boundaries and per-document windows with no adaptation

Default TTT mode in this folder:

```bash
TTT_ENABLED=1
TTT_MODE=doc_isolated
TTT_DOC_CHUNK_TOKENS=256
TTT_DOC_EVAL_SEQ_LEN=2048
TTT_DOC_BATCH_SEQS=32
```

Run from the repository root with:

```bash
TTT_ENABLED=1 \
TTT_MODE=doc_isolated \
TTT_DOC_CHUNK_TOKENS=256 \
TTT_DOC_EVAL_SEQ_LEN=2048 \
TTT_DOC_BATCH_SEQS=32 \
python records/track_10min_16mb/2026-03-27_DocAware_LoRA_TTT/train_gpt.py
```

For the March 23 apples-to-apples recipe plus doc-isolated eval:

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
TTT_ENABLED=1 TTT_MODE=doc_isolated \
TTT_DOC_CHUNK_TOKENS=256 TTT_DOC_EVAL_SEQ_LEN=2048 TTT_DOC_BATCH_SEQS=32 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-27_DocAware_LoRA_TTT/train_gpt.py
```

The tokenizer BOS id is taken from SentencePiece automatically unless overridden with:

```bash
TTT_BOS_ID=<id>
```
