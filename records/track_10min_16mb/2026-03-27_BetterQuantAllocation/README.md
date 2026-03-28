# 2026-03-27 Better Quantization Allocation

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Goal:

- improve artifact quality under the same 16 MB budget by allocating precision per tensor more intelligently
- keep the first pass export-only so training dynamics stay comparable to the March 23 baseline

Planned first changes:

- add per-name precision overrides for `fp16`, `int8`, and `int6`
- add a small sensitivity profiler using held-out validation slices
- rank tensors by `delta_bpb / byte_saved` instead of only using broad category rules

Current knobs:

- `QUANT_KEEP_LATE_LAYERS_INT8=2`
- `QUANT_FORCE_FP16_PATTERNS=...`
- `QUANT_FORCE_INT8_PATTERNS=...`
- `QUANT_FORCE_INT6_PATTERNS=...`
- `QUANT_SENSITIVITY_ENABLED=1`
- `QUANT_SENSITIVITY_TOKENS=1048576`
- `QUANT_SENSITIVITY_MAX_GROUPS=12`

Suggested first run from repo root:

```bash
QUANT_KEEP_LATE_LAYERS_INT8=2 \
QUANT_SENSITIVITY_ENABLED=1 \
QUANT_SENSITIVITY_TOKENS=1048576 \
QUANT_SENSITIVITY_MAX_GROUPS=8 \
python records/track_10min_16mb/2026-03-27_BetterQuantAllocation/train_gpt.py
```
