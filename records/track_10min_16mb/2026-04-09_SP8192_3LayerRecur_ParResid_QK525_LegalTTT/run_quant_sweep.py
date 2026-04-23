#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ATTN_CANDIDATES = (
    "blocks.0.attn.c_q.weight",
    "blocks.0.attn.c_v.weight",
    "blocks.0.attn.proj.weight",
    "blocks.9.attn.c_q.weight",
    "blocks.9.attn.c_v.weight",
    "blocks.9.attn.proj.weight",
    "blocks.10.attn.c_q.weight",
    "blocks.10.attn.c_v.weight",
    "blocks.10.attn.proj.weight",
)

MLP_CANDIDATES = (
    "blocks.0.mlp.fc.weight",
    "blocks.0.mlp.proj.weight",
    "blocks.10.mlp.fc.weight",
    "blocks.10.mlp.proj.weight",
)

CONFIG_PRESETS = {
    "blocks.10.mlp.proj.weight_grid": (
        ("b7", {"bits": 7}),
        ("b7_cs11p5", {"bits": 7, "clip_sigmas": 11.5}),
        ("b7_cs10p5", {"bits": 7, "clip_sigmas": 10.5}),
        ("b7_cs9p5", {"bits": 7, "clip_sigmas": 9.5}),
        ("b6", {"bits": 6}),
        ("b6_cs13p5", {"bits": 6, "clip_sigmas": 13.5}),
        ("b6_cs14p5", {"bits": 6, "clip_sigmas": 14.5}),
        ("b6_cs15p5", {"bits": 6, "clip_sigmas": 15.5}),
    ),
}


def _candidate_list(family: str) -> tuple[str, ...]:
    if family == "attn":
        return ATTN_CANDIDATES
    if family == "mlp":
        return MLP_CANDIDATES
    raise ValueError(f"Unknown family: {family}")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "records").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root above {start}")


def _tensor_slug(name: str) -> str:
    return name.replace("/", "_")


def _write_override(path: Path, tensor_name: str, override: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {tensor_name: override}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_chunk(items: list[tuple[str, dict[str, float | int]]], num_shards: int, shard_index: int):
    chunk_size = (len(items) + num_shards - 1) // num_shards
    start = shard_index * chunk_size
    end = min(len(items), start + chunk_size)
    return items[start:end]


def _single_tensor_runs(
    tensor_name: str,
    config_preset: str | None,
    bits: int,
    clip_sigmas: float | None,
) -> list[tuple[str, dict[str, float | int]]]:
    if config_preset:
        return [(label, dict(override)) for (label, override) in CONFIG_PRESETS[config_preset]]
    override = {"bits": bits}
    if clip_sigmas is not None:
        override["clip_sigmas"] = clip_sigmas
    label = f"b{bits}" if clip_sigmas is None else f"b{bits}_cs{str(clip_sigmas).replace('.', 'p')}"
    return [(label, override)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantization sweeps.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--family", choices=("attn", "mlp"), help="Sweep a built-in family candidate list.")
    mode.add_argument("--tensor", help="Sweep a single tensor across one or more override configs.")
    parser.add_argument("--bits", type=int, default=7, help="Bit width for family sweeps or single-config tensor sweeps.")
    parser.add_argument("--clip-sigmas", type=float, help="Clip sigmas for family sweeps or single-config tensor sweeps.")
    parser.add_argument("--config-preset", choices=tuple(CONFIG_PRESETS), help="Named config grid for --tensor mode.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split a tensor config grid across multiple machines.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index within --num-shards.")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    if args.family and args.config_preset:
        raise SystemExit("--config-preset only applies to --tensor mode")

    record_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(record_dir)
    train_script = record_dir / "train_gpt_human.py"
    score_script = record_dir / "score_quant_sweeps.py"

    if args.family:
        root = record_dir / "sweeps" / args.family / args.tag
        run_specs = []
        override = {"bits": args.bits}
        if args.clip_sigmas is not None:
            override["clip_sigmas"] = args.clip_sigmas
        elif args.family in ("attn", "mlp"):
            override["clip_sigmas"] = 11.5
        for tensor_name in _candidate_list(args.family):
            run_specs.append((tensor_name, tensor_name, dict(override)))
        mode_label = f"family={args.family}"
    else:
        tensor_slug = _tensor_slug(args.tensor)
        root = record_dir / "sweeps" / "custom" / tensor_slug / args.tag
        all_configs = _single_tensor_runs(args.tensor, args.config_preset, args.bits, args.clip_sigmas)
        selected_configs = _split_chunk(all_configs, args.num_shards, args.shard_index)
        if args.num_shards > 1:
            root = root / f"shard_{args.shard_index + 1}_of_{args.num_shards}"
        run_specs = [(args.tensor, label, override) for (label, override) in selected_configs]
        mode_label = f"tensor={args.tensor}"

    override_dir = root / "overrides"
    log_dir = root / "logs"
    model_dir = root / "models"
    artifact_dir = root / "artifacts"
    for path in (override_dir, log_dir, model_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(mode_label)
    print(f"Output root: {root}")
    if args.tensor:
        print(f"Shard: {args.shard_index + 1}/{args.num_shards}")
    if not run_specs:
        print("No configs selected for this shard.")
        return 0

    for tensor_name, label, override in run_specs:
        stem = label if args.tensor else tensor_name
        override_path = override_dir / f"{stem}.json"
        log_path = log_dir / f"{stem}.log"
        model_path = model_dir / f"{stem}.pt"
        artifact_path = artifact_dir / f"{stem}.ptz"
        _write_override(override_path, tensor_name, override)
        env = os.environ.copy()
        env["RUN_ID"] = stem
        env["QUANT_OVERRIDE_PATH"] = str(override_path)
        env["LOGFILE"] = str(log_path)
        env["MODEL_PATH"] = str(model_path)
        env["QUANTIZED_MODEL_PATH"] = str(artifact_path)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print()
        print(f"=== {stem} ===")
        print(f"tensor: {tensor_name}")
        print(f"override: {json.dumps(override, sort_keys=True)}")
        print(f"log: {log_path}")
        subprocess.run([sys.executable, str(train_script)], cwd=repo_root, env=env, check=True)

    print()
    print("Sweep complete.")
    print(f"Logs: {log_dir}")
    print("Score command:")
    print(f"  python3 {score_script} --baseline /path/to/baseline.log {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
