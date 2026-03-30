# Hybrid Local Mixer + Attention

March 23 baseline with one architectural change:

- layers `0-4`: cheap causal multi-shift local mixer
- layers `5-10`: original full GQA attention

The local mixer is intentionally optimized for wallclock:

- depthwise sparse causal conv over fixed offsets `0,1,2,4,8,16`
- one learned output projection bank per local layer
- no Q/K/V/out attention stack in the lower half
- no pre-stack `SmearGate` when local mixing is active
- MLP, skip connections, Muon, EMA/SWA, export path, and legal TTT all kept

The goal is simple: stop paying full attention cost in early layers where most signal is local anyway, and buy more useful steps inside the same 600s budget.

## Implementation Notes

- New bank: `local_mix_bank`
- New exported tensor names: `blocks.{i}.tmix.proj.weight`
- New small control params live in `blocks.{i}.temporal_mix.mix_logits`
- The local-mix bank is quantized with the same int6 exporter path as the other large banks

## Expected Behavior

- `step_avg` should come down relative to March 23
- raw pre-export BPB should be the first thing to judge
- if speed improves but BPB does not, the local mixer is too weak
- if BPB improves without speed loss, this is a real architecture win

## Run Command

This keeps the March 23 config and adds only the local-mix knobs:

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
LOCAL_MIX_LAYERS=5 LOCAL_MIX_OFFSETS=0,1,2,4,8,16 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-30_HybridLocalMix/train_gpt.py
```

## What To Watch

- `local_mix layers:... offsets:...`
- `smear_active:False`
- `step_avg:...ms`
- `DIAGNOSTIC post_ema ...`
- `final_int6_roundtrip_exact ...`
- `legal_ttt_exact ...`
