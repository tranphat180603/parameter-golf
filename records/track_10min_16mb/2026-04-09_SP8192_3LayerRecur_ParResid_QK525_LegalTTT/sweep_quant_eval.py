#!/usr/bin/env python3
import argparse
import importlib.util
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist


THIS_DIR = Path(__file__).resolve().parent
IMPL_PATH = THIS_DIR / "train_gpt_human.py"
PACKED_SCRIPT_PATH = THIS_DIR / "train_gpt.py"


def _load_impl(path: Path):
    spec = importlib.util.spec_from_file_location("train_gpt_human_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fmt_value(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse an existing checkpoint, sweep MATRIX_CLIP_SIGMAS and "
            "EMBED_CLIP_SIGMAS, and report quantized size plus eval metrics."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to an existing fp checkpoint to quantize repeatedly.",
    )
    parser.add_argument(
        "--matrix-clip-sigmas",
        type=float,
        nargs="+",
        required=True,
        help="One or more MATRIX_CLIP_SIGMAS values to test.",
    )
    parser.add_argument(
        "--embed-clip-sigmas",
        type=float,
        nargs="+",
        required=True,
        help="One or more EMBED_CLIP_SIGMAS values to test.",
    )
    parser.add_argument(
        "--gptq-calibration-batches",
        type=int,
        default=8,
        help="Calibration batches per quantization run. Use 64 for final numbers.",
    )
    parser.add_argument(
        "--eval",
        choices=["none", "quantized", "sliding", "all"],
        default="sliding",
        help="Which post-quant evaluation to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/quant_eval_sweep",
        help="Where to write temporary checkpoints, artifacts, logs, and summary.",
    )
    parser.add_argument(
        "--label",
        default="quant_eval",
        help="Prefix for per-run log/artifact filenames.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to use. Run with 1 visible GPU.",
    )
    parser.add_argument(
        "--script-path",
        default=str(PACKED_SCRIPT_PATH),
        help="Script whose source length should be counted toward total size.",
    )
    return parser


def _setup_cuda():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False


def _setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    return distributed, world_size, local_rank, device


def _make_hparams(
    impl,
    args,
    matrix_cs: float,
    embed_cs: float,
    output_dir: Path,
    distributed: bool,
    world_size: int,
    local_rank: int,
):
    h = impl.Hyperparameters()
    tag = f"{args.label}_m{_fmt_value(matrix_cs)}_e{_fmt_value(embed_cs)}"
    h.distributed = distributed
    h.rank = int(os.environ.get("RANK", "0"))
    h.world_size = world_size
    h.local_rank = local_rank
    h.is_main_process = h.rank == 0
    h.grad_accum_steps = 8 // world_size
    h.quantize_only = True
    h.gptq_calibration_batches = args.gptq_calibration_batches
    h.matrix_clip_sigmas = matrix_cs
    h.embed_clip_sigmas = embed_cs
    h.model_path = str(output_dir / f"{tag}.pt")
    h.quantized_model_path = str(output_dir / f"{tag}.ptz")
    h.logfile = str(output_dir / f"{tag}.log")
    h.ttt_enabled = False
    h.etlb_enabled = False
    h.sliding_window_enabled = args.eval in ("sliding", "all")
    return h, tag


def _write_summary(output_dir: Path, rows):
    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as f:
        headers = [
            "tag",
            "matrix_clip_sigmas",
            "embed_clip_sigmas",
            "artifact_bytes",
            "total_bytes",
            "quantized_val_loss",
            "quantized_val_bpb",
            "sliding_val_loss",
            "sliding_val_bpb",
        ]
        print("\t".join(headers), file=f)
        for row in rows:
            print(
                "\t".join(
                    [
                        row["tag"],
                        str(row["matrix_clip_sigmas"]),
                        str(row["embed_clip_sigmas"]),
                        str(row["artifact_bytes"]),
                        str(row["total_bytes"]),
                        "" if row["quantized_val_loss"] is None else f"{row['quantized_val_loss']:.8f}",
                        "" if row["quantized_val_bpb"] is None else f"{row['quantized_val_bpb']:.8f}",
                        "" if row["sliding_val_loss"] is None else f"{row['sliding_val_loss']:.8f}",
                        "" if row["sliding_val_bpb"] is None else f"{row['sliding_val_bpb']:.8f}",
                    ]
                ),
                file=f,
            )
    return summary_path


def main():
    args = _build_parser().parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    script_path = Path(args.script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for code-size accounting: {script_path}")
    distributed, world_size, local_rank, device = _setup_distributed()
    _setup_cuda()

    impl = _load_impl(IMPL_PATH)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code = script_path.read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))
    checkpoint_state = torch.load(model_path, map_location="cpu")

    rows = []
    val_data_cache = {}
    for matrix_cs, embed_cs in itertools.product(args.matrix_clip_sigmas, args.embed_clip_sigmas):
        h, tag = _make_hparams(
            impl, args, matrix_cs, embed_cs, output_dir, distributed, world_size, local_rank
        )
        impl.set_logging_hparams(h)
        impl.log(f"sweep_start tag:{tag} matrix_clip_sigmas:{matrix_cs} embed_clip_sigmas:{embed_cs}")

        val_key = (h.tokenizer_path, h.val_files, h.eval_seq_len, h.vocab_size)
        if args.eval != "none" and val_key not in val_data_cache:
            val_data_cache[val_key] = impl.ValidationData(h, device)
        val_data = val_data_cache.get(val_key)

        base_model = impl.GPT(h).to(device).bfloat16()
        impl.restore_fp32_params(base_model)
        base_model.load_state_dict(checkpoint_state, strict=True)

        impl.serialize(h, base_model, code)
        if distributed:
            dist.barrier()
        artifact_bytes = Path(h.quantized_model_path).stat().st_size
        total_bytes = artifact_bytes + code_bytes

        quantized_val_loss = None
        quantized_val_bpb = None
        sliding_val_loss = None
        sliding_val_bpb = None

        if args.eval != "none":
            eval_model = impl.deserialize(h, device)
            if h.num_loops > 0:
                eval_model.looping_active = True

            if args.eval in ("quantized", "all"):
                compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
                quantized_val_loss, quantized_val_bpb = impl.timed_eval(
                    "quantized", impl.eval_val, h, device, val_data, compiled_model
                )
                del compiled_model
                torch._dynamo.reset()

            if args.eval in ("sliding", "all"):
                sliding_val_loss, sliding_val_bpb = impl.timed_eval(
                    "quantized_sliding_window", impl.eval_val_sliding, h, device, val_data, eval_model
                )

            del eval_model

        row = {
            "tag": tag,
            "matrix_clip_sigmas": matrix_cs,
            "embed_clip_sigmas": embed_cs,
            "artifact_bytes": artifact_bytes,
            "total_bytes": total_bytes,
            "quantized_val_loss": quantized_val_loss,
            "quantized_val_bpb": quantized_val_bpb,
            "sliding_val_loss": sliding_val_loss,
            "sliding_val_bpb": sliding_val_bpb,
        }
        rows.append(row)
        if h.is_main_process:
            print(
                f"{tag}: artifact={artifact_bytes} total={total_bytes}"
                + (
                    ""
                    if sliding_val_loss is None
                    else f" sliding_val_loss={sliding_val_loss:.8f} sliding_val_bpb={sliding_val_bpb:.8f}"
                )
                + (
                    ""
                    if quantized_val_loss is None
                    else f" quantized_val_loss={quantized_val_loss:.8f} quantized_val_bpb={quantized_val_bpb:.8f}"
                )
            )

        del base_model
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if int(os.environ.get("RANK", "0")) == 0:
        summary_path = _write_summary(output_dir, rows)
        print(f"summary_tsv: {summary_path}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
