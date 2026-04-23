#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MLP_SWEEP_TARGETS = (
    "blocks.0.mlp.fc.weight",
    "blocks.0.mlp.proj.weight",
    "blocks.10.mlp.fc.weight",
    "blocks.10.mlp.proj.weight",
)


@dataclass
class RunMetrics:
    path: Path
    overrides: tuple[str, ...]
    family: str
    quantized: float | None
    sliding: float | None
    ttt: float | None
    artifact_bytes: int | None

    @property
    def name(self) -> str:
        if self.overrides:
            return "+".join(self.overrides)
        return self.path.stem

    @property
    def complete(self) -> bool:
        return None not in (self.quantized, self.sliding, self.ttt, self.artifact_bytes)


def _collect_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() not in (".log", ".txt"):
                    continue
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _extract_metric(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}([0-9]+\.[0-9]+)", text)
    return float(match.group(1)) if match else None


def _extract_artifact_bytes(text: str) -> int | None:
    patterns = (
        r"Serialized model quantized\+ans:\s*([0-9]+) bytes",
        r"Serialized model quantized\+brotli:\s*([0-9]+) bytes",
        r"Serialized model quantized\+lzma:\s*([0-9]+) bytes",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_overrides(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    overrides: list[str] = []
    in_block = False
    for line in lines:
        if line.strip() == "Quantization overrides:":
            in_block = True
            continue
        if not in_block:
            continue
        if not line.startswith("  "):
            break
        name = line.strip().split(":", 1)[0]
        if name:
            overrides.append(name)
    return tuple(sorted(overrides))


def _family_for_overrides(overrides: tuple[str, ...]) -> str:
    if not overrides:
        return "baseline"
    if all(".attn." in name for name in overrides):
        return "attn"
    if all(".mlp." in name for name in overrides):
        return "mlp"
    return "mixed"


def parse_run(path: Path) -> RunMetrics:
    text = path.read_text(encoding="utf-8", errors="replace")
    overrides = _extract_overrides(text)
    return RunMetrics(
        path=path,
        overrides=overrides,
        family=_family_for_overrides(overrides),
        quantized=_extract_metric(text, "quantized val_loss:"),
        sliding=_extract_metric(text, "quantized_sliding_window val_loss:"),
        ttt=_extract_metric(text, "quantized_ttt val_loss:"),
        artifact_bytes=_extract_artifact_bytes(text),
    )


def _find_baseline(runs: list[RunMetrics], baseline_arg: str | None) -> RunMetrics:
    if baseline_arg is not None:
        baseline_path = Path(baseline_arg).resolve()
        for run in runs:
            if run.path == baseline_path:
                return run
        raise SystemExit(f"Baseline log not found in inputs: {baseline_path}")
    candidates = [run for run in runs if run.family == "baseline"]
    if len(candidates) != 1:
        raise SystemExit("Pass --baseline explicitly when inputs do not contain exactly one no-override baseline log.")
    return candidates[0]


def _score_run(
    baseline: RunMetrics,
    run: RunMetrics,
    loss_unit: float,
    byte_unit: int,
    wq: float,
    wsw: float,
    wttt: float,
    wbytes: float,
) -> dict[str, float | int | str]:
    d_q = baseline.quantized - run.quantized
    d_sw = baseline.sliding - run.sliding
    d_ttt = baseline.ttt - run.ttt
    d_bytes = run.artifact_bytes - baseline.artifact_bytes
    score = (
        wq * (d_q / loss_unit)
        + wsw * (d_sw / loss_unit)
        + wttt * (d_ttt / loss_unit)
        - wbytes * (d_bytes / byte_unit)
    )
    return {
        "family": run.family,
        "name": run.name,
        "score": score,
        "d_q": d_q,
        "d_sw": d_sw,
        "d_ttt": d_ttt,
        "d_bytes": d_bytes,
        "artifact_bytes": run.artifact_bytes,
        "path": str(run.path),
    }


def _print_table(title: str, rows: list[dict[str, float | int | str]]) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    headers = ("score", "d_q", "d_sw", "d_ttt", "d_bytes", "artifact_bytes", "name")
    widths = {header: len(header) for header in headers}
    rendered: list[dict[str, str]] = []
    for row in rows:
        out = {
            "score": f"{row['score']:.2f}",
            "d_q": f"{row['d_q']:+.6f}",
            "d_sw": f"{row['d_sw']:+.6f}",
            "d_ttt": f"{row['d_ttt']:+.6f}",
            "d_bytes": f"{int(row['d_bytes']):+d}",
            "artifact_bytes": str(int(row["artifact_bytes"])),
            "name": str(row["name"]),
        }
        rendered.append(out)
        for header, value in out.items():
            widths[header] = max(widths[header], len(value))
    print("  " + "  ".join(header.ljust(widths[header]) for header in headers))
    for row in rendered:
        print("  " + "  ".join(row[header].ljust(widths[header]) for header in headers))


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = ("family", "name", "score", "d_q", "d_sw", "d_ttt", "d_bytes", "artifact_bytes", "path")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score quantization sweep logs against a baseline log.")
    parser.add_argument("inputs", nargs="+", help="Log files and/or directories containing logs.")
    parser.add_argument("--baseline", help="Baseline log path. If omitted, exactly one no-override log must be present in inputs.")
    parser.add_argument("--csv", help="Optional path to write CSV output.")
    parser.add_argument("--loss-unit", type=float, default=0.001, help="Loss delta that equals one score unit.")
    parser.add_argument("--byte-unit", type=int, default=16384, help="Byte delta that equals one score penalty unit.")
    parser.add_argument("--weight-quantized", type=float, default=1.0)
    parser.add_argument("--weight-sliding", type=float, default=1.5)
    parser.add_argument("--weight-ttt", type=float, default=2.0)
    parser.add_argument("--weight-bytes", type=float, default=1.0)
    args = parser.parse_args()

    runs = [parse_run(path) for path in _collect_paths(args.inputs)]
    if not runs:
        raise SystemExit("No logs found.")

    baseline = _find_baseline(runs, args.baseline)
    if not baseline.complete:
        raise SystemExit(f"Baseline is incomplete and cannot be scored: {baseline.path}")

    scored: list[dict[str, float | int | str]] = []
    incomplete: list[RunMetrics] = []
    for run in runs:
        if run.path == baseline.path:
            continue
        if not run.complete:
            incomplete.append(run)
            continue
        scored.append(
            _score_run(
                baseline,
                run,
                args.loss_unit,
                args.byte_unit,
                args.weight_quantized,
                args.weight_sliding,
                args.weight_ttt,
                args.weight_bytes,
            )
        )

    scored.sort(key=lambda row: (-float(row["score"]), str(row["name"])))
    attn_rows = [row for row in scored if row["family"] == "attn"]
    mlp_rows = [row for row in scored if row["family"] == "mlp"]
    mixed_rows = [row for row in scored if row["family"] == "mixed"]

    print(f"Baseline: {baseline.path}")
    print(f"  quantized={baseline.quantized:.6f}  sliding={baseline.sliding:.6f}  ttt={baseline.ttt:.6f}  artifact_bytes={baseline.artifact_bytes}")
    _print_table("Attn leaderboard", attn_rows)
    _print_table("MLP leaderboard", mlp_rows)
    if mixed_rows:
        _print_table("Mixed leaderboard", mixed_rows)

    seen_mlp_singletons = {row["name"] for row in mlp_rows if "+" not in str(row["name"])}
    missing_mlp = [name for name in MLP_SWEEP_TARGETS if name not in seen_mlp_singletons]
    print("MLP targets")
    for name in MLP_SWEEP_TARGETS:
        marker = "done" if name in seen_mlp_singletons else "missing"
        print(f"  {marker}: {name}")

    if incomplete:
        print("Incomplete logs")
        for run in incomplete:
            missing = []
            if run.quantized is None:
                missing.append("quantized")
            if run.sliding is None:
                missing.append("sliding")
            if run.ttt is None:
                missing.append("ttt")
            if run.artifact_bytes is None:
                missing.append("artifact_bytes")
            print(f"  {run.path}: missing {', '.join(missing)}")

    if args.csv:
        _write_csv(Path(args.csv), scored)
        print(f"CSV written: {Path(args.csv).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
