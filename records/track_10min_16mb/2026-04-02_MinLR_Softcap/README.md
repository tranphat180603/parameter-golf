# 2026-04-02 Min LR + Lower Softcap

Base: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`

## Changes

### 1. Non-zero minimum LR in warmdown (MIN_LR_FRAC=0.1)
The parent decays LR linearly to 0.0 during warmdown, wasting the final steps.
We clamp the warmdown fraction to a floor of 0.1, so the model never stops learning.
Based on modded-nanogpt speedrun Record #19 ("lr decay to 0.1 instead of 0.0").

### 2. Lower logit softcap (30 -> 15)
The parent uses LOGIT_SOFTCAP=30, allowing logits in [-30, +30].
We lower to 15, producing tighter logit distributions and better-calibrated predictions.
Based on modded-nanogpt speedrun Record #18.

## New environment knobs

- `MIN_LR_FRAC` (default 0.1) — warmdown floor as fraction of peak LR
- `LOGIT_SOFTCAP` (default changed from 30.0 to 15.0)

## Run

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 MIN_LR_FRAC=0.1 LOGIT_SOFTCAP=15.0 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
