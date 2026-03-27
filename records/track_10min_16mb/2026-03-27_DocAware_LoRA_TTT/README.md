# 2026-03-27 Doc-Aware LoRA TTT

Work-in-progress experiment based on:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Main change:

- replace fixed 32K-token score-first SGD TTT with document-aware, score-first LoRA TTT
- reset adapters per document
- adapt only lightweight LoRA deltas on `q`, `v`, and logits
- keep the original chunk-SGD TTT path behind `TTT_MODE=sliding_sgd`

Default TTT mode in this folder:

```bash
TTT_ENABLED=1
TTT_MODE=doc_lora
TTT_LORA_RANK=8
TTT_LORA_LR=0.01
TTT_DOC_CHUNK_TOKENS=256
TTT_DOC_EVAL_SEQ_LEN=2048
TTT_DOC_BATCH_SEQS=32
```

Run from the repository root with:

```bash
TTT_ENABLED=1 \
TTT_MODE=doc_lora \
TTT_LORA_RANK=8 \
TTT_LORA_LR=0.01 \
TTT_DOC_CHUNK_TOKENS=256 \
TTT_DOC_EVAL_SEQ_LEN=2048 \
TTT_DOC_BATCH_SEQS=32 \
python records/track_10min_16mb/2026-03-27_DocAware_LoRA_TTT/train_gpt.py
```

The tokenizer BOS id is taken from SentencePiece automatically unless overridden with:

```bash
TTT_BOS_ID=<id>
```
