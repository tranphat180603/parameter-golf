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


def _candidate_list(family: str) -> tuple[str, ...]:
    if family == "attn":
        return ATTN_CANDIDATES
    if family == "mlp":
        return MLP_CANDIDATES
    raise ValueError(f"Unknown family: {family}")


def _write_override(path: Path, tensor_name: str, bits: int, clip_sigmas: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {tensor_name: {"bits": bits, "clip_sigmas": clip_sigmas}}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "records").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root above {start}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-GPU quantization sweep family.")
    parser.add_argument("--family", choices=("attn", "mlp"), required=True)
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--clip-sigmas", type=float, default=11.5)
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    record_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(record_dir)
    train_script = record_dir / "train_gpt_human.py"
    score_script = record_dir / "score_quant_sweeps.py"
    root = record_dir / "sweeps" / args.family / args.tag
    override_dir = root / "overrides"
    log_dir = root / "logs"
    model_dir = root / "models"
    artifact_dir = root / "artifacts"
    for path in (override_dir, log_dir, model_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"Family: {args.family}")
    print(f"Output root: {root}")
    print(f"Bits: {args.bits}")
    print(f"Clip sigmas: {args.clip_sigmas}")

    for tensor_name in _candidate_list(args.family):
        override_path = override_dir / f"{tensor_name}.json"
        log_path = log_dir / f"{tensor_name}.log"
        model_path = model_dir / f"{tensor_name}.pt"
        artifact_path = artifact_dir / f"{tensor_name}.ptz"
        _write_override(override_path, tensor_name, args.bits, args.clip_sigmas)
        env = os.environ.copy()
        env["RUN_ID"] = f"{args.family}_{tensor_name}"
        env["QUANT_OVERRIDE_PATH"] = str(override_path)
        env["LOGFILE"] = str(log_path)
        env["MODEL_PATH"] = str(model_path)
        env["QUANTIZED_MODEL_PATH"] = str(artifact_path)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print()
        print(f"=== {tensor_name} ===")
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
