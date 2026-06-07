from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from common import OUT_DIR
from app.llm.query_router import VALID_LABELS


DEFAULT_DATASET_DIR = OUT_DIR / "router_lora" / "dataset_v2_lora_processed"
DEFAULT_ADAPTER_DIR = OUT_DIR / "router_lora" / "qwen2_5_1_5b_router_lora_v2_lora_processed"
DEFAULT_REPORT = OUT_DIR / "router_lora" / "router_lora_v2_lora_processed_eval_report.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def require_eval_imports():
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing eval dependencies. Install them with:\n"
            "  pip install -e .[router-train]"
        ) from exc
    return {"PeftModel": PeftModel, "AutoModelForCausalLM": AutoModelForCausalLM, "AutoTokenizer": AutoTokenizer}


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def predict(row: dict, model: Any, tokenizer: Any, max_new_tokens: int) -> tuple[str | None, str]:
    messages = row["messages"][:-1] if row.get("messages") else [
        {"role": "user", "content": row.get("router_input") or row.get("question") or row.get("raw_question") or ""}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    payload = extract_json(text)
    label = payload.get("label") if isinstance(payload, dict) else None
    if label not in VALID_LABELS:
        label = None
    return label, text


def predict_batch(rows: list[dict], model: Any, tokenizer: Any, max_new_tokens: int) -> list[tuple[str | None, str]]:
    prompts = []
    for row in rows:
        messages = row["messages"][:-1] if row.get("messages") else [
            {"role": "user", "content": row.get("router_input") or row.get("question") or row.get("raw_question") or ""}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for index in range(len(rows)):
        generated = output_ids[index][prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = extract_json(text)
        label = payload.get("label") if isinstance(payload, dict) else None
        if label not in VALID_LABELS:
            label = None
        results.append((label, text))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained router LoRA adapter.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR / "test.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    imports = require_eval_imports()
    tokenizer = imports["AutoTokenizer"].from_pretrained(args.adapter_dir if args.adapter_dir.exists() else args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    base_model = imports["AutoModelForCausalLM"].from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = imports["PeftModel"].from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    correct = 0
    invalid = 0
    labels = sorted(VALID_LABELS)
    confusion: dict[str, Counter] = defaultdict(Counter)
    failures = []
    for start in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[start : start + max(1, args.batch_size)]
        predictions = predict_batch(batch, model, tokenizer, args.max_new_tokens)
        for row, (predicted, raw) in zip(batch, predictions):
            expected = row["label"]
            if predicted is None:
                invalid += 1
            if predicted == expected:
                correct += 1
            else:
                failures.append(
                    {
                        "id": row["id"],
                        "question": row.get("raw_question") or row.get("question") or row.get("router_input"),
                        "expected": expected,
                        "predicted": predicted,
                        "raw": raw[:500],
                    }
                )
            confusion[expected][predicted or "<invalid>"] += 1
        print(f"\r{min(start + len(batch), len(rows))}/{len(rows)}", end="", flush=True)
    print()

    per_label = {}
    for label in labels:
        total = sum(confusion[label].values())
        per_label[label] = {
            "total": total,
            "correct": confusion[label][label],
            "accuracy": (confusion[label][label] / total) if total else None,
        }

    report = {
        "dataset": str(args.dataset),
        "adapter_dir": str(args.adapter_dir),
        "rows": len(rows),
        "accuracy": (correct / len(rows)) if rows else 0,
        "invalid_outputs": invalid,
        "per_label": per_label,
        "confusion": {label: dict(counter) for label, counter in confusion.items()},
        "failures_sample": failures[:100],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "accuracy", "invalid_outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
