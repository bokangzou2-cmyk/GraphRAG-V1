from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import OUT_DIR, ROOT


DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_DIR = OUT_DIR / "query_rewriter_lora" / "final_dataset"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_rewriter_lora" / "qwen2_5_1_5b_rewriter_lora"


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Query Rewriter-only LoRA with interruption-friendly defaults.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=896)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    resume_from = args.resume_from_checkpoint
    if args.auto_resume and resume_from is None:
        resume_from = latest_checkpoint(args.output_dir)
        if resume_from:
            print(f"Auto-resuming from {resume_from}")
        else:
            print(f"No checkpoint found under {args.output_dir}; starting a new run.")

    command = [
        sys.executable,
        str(ROOT / "processing" / "train_router_lora.py"),
        "--model",
        str(args.model),
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(args.output_dir),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--logging-steps",
        str(args.logging_steps),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if resume_from:
        command.extend(["--resume-from-checkpoint", str(resume_from)])

    print("Running:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
