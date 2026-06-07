from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from common import OUT_DIR, ROOT, iter_jsonl

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm.query_rewriter import complete_rewriter_payload, rewrite_query, rule_rewrite  # noqa: E402
from query_rewriter_common import extract_json, normalize_rewriter_answer, query_rewriter_prompt  # noqa: E402
from processing.build_router_lora_v2_dataset import (  # noqa: E402
    ANSWER_BY_LABEL,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    answer_for,
    build_messages,
    load_old_router_rows,
    router_input_from_rewriter,
    validate_answer,
)


DEFAULT_GPT_SHARDS_DIR = OUT_DIR / "router_lora" / "gpt_authored_v2_shards"
DEFAULT_OUTPUT_DIR = OUT_DIR / "router_lora" / "dataset_v2_lora_processed"
PROCESSED_ROWS_NAME = "processed_rows.jsonl"
ERROR_ROWS_NAME = "error_rows.jsonl"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and normalize_text(message.get("content")):
            return normalize_text(message["content"])
    return ""


def expected_targets(label: str) -> list[str]:
    return list(ANSWER_BY_LABEL[label]["retrieval_targets"])


def normalize_source_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("source_messages")
    if not isinstance(messages, list) or not messages:
        question = normalize_text(row.get("question") or row.get("raw_question") or "")
        messages = [{"role": "user", "content": question}]
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = normalize_text(message.get("content"))
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    if not normalized or normalized[-1]["role"] != "user":
        raise ValueError("source_messages_must_end_with_user")
    return normalized


def validate_raw_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = row.get("label")
    if label not in ANSWER_BY_LABEL:
        errors.append(f"unknown_label:{label}")
    try:
        normalize_source_messages(row)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def load_gpt_raw_rows(shards_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not shards_dir.exists():
        raise SystemExit(f"missing GPT shards dir: {shards_dir}")
    for path in sorted(shards_dir.glob("*.jsonl")):
        for index, row in enumerate(iter_jsonl(path), start=1):
            row = dict(row)
            row.setdefault("id", f"{path.stem}_{index:05d}")
            row.setdefault("source", "gpt_5_4_direct_router_v2")
            row.setdefault("source_method", path.stem)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def convert_old_router_rows(old_router_dir: Path, cap_per_label: int, seed: int) -> list[dict[str, Any]]:
    if cap_per_label <= 0:
        return []
    converted: list[dict[str, Any]] = []
    for row in load_old_router_rows(old_router_dir, cap_per_label=cap_per_label, seed=seed):
        converted.append(
            {
                "id": f"old_router_{row['id']}",
                "source_messages": row["source_messages"],
                "label": row["label"],
                "retrieval_targets": expected_targets(row["label"]),
                "tags": ["old_router_vetted"],
                "source": "old_router_lora_dataset",
                "source_method": "old_router_v1_vetted_label_lora_reprocessed",
            }
        )
    return converted


def read_processed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                ids.add(str(row["id"]))
    return ids


def render_progress(done: int, total: int, *, started_at: float, prefix: str = "rewriting") -> str:
    width = 30
    ratio = done / total if total else 1.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.1, time.time() - started_at)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else 0.0
    return f"\r{prefix} [{bar}] {done}/{total} {ratio:6.2%} {rate:5.2f} rows/s ETA {remaining:6.0f}s"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_processed_row(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw["label"])
    messages = normalize_source_messages(raw)
    rewrite = rewrite_query(messages)
    raw_question = latest_user_query(messages)
    rewriter_payload = dict(rewrite.payload)
    rewriter_guarded = False
    non_retrieval_labels = {
        "general_no_legal_retrieval",
        "non_criminal_legal",
        "unclear_need_clarification",
        "low_quality_or_incomplete",
    }
    criminal_terms = re.compile(r"刑法|刑事|犯罪|罪名|构成要件|量刑|判刑|盗窃|诈骗|抢劫|故意伤害|故意杀人|过失致人死亡|帮信|判决书|案")
    if label in non_retrieval_labels and not criminal_terms.search(raw_question):
        standalone = str(rewriter_payload.get("standalone_query") or "")
        introduced_criminal_terms = criminal_terms.search(standalone) or rewriter_payload.get("case_name_mentions") or rewriter_payload.get("resolved_case_refs") or rewriter_payload.get("crime_mentions")
        if introduced_criminal_terms:
            rewriter_payload = {
                "standalone_query": raw_question,
                "resolved_case_refs": [],
                "case_name_mentions": [],
                "crime_mentions": [],
                "amount_constraints": [],
                "field_intents": [],
                "rewrite_required": False,
                "confidence": min(float(rewriter_payload.get("confidence") or 0.75), 0.75),
                "warnings": ["rewriter_guard_preserved_non_retrieval_input"],
            }
            rewriter_guarded = True
    router_input = router_input_from_rewriter(rewriter_payload, raw_question)
    answer = answer_for(label)
    errors = validate_answer(answer)
    if errors:
        raise ValueError(f"invalid_answer:{','.join(errors)}")
    row_id = str(raw["id"])
    return {
        "id": row_id,
        "raw_question": raw_question,
        "source_messages": messages,
        "rewriter": rewriter_payload,
        "rewriter_source": rewrite.source,
        "rewriter_reason": rewrite.reason,
        "rewriter_guarded": rewriter_guarded,
        "router_input": router_input,
        "label": label,
        "answer": answer,
        "messages": build_messages(router_input, answer),
        "source": raw.get("source") or "unknown",
        "source_method": raw.get("source_method") or "unknown",
        "tags": list(raw.get("tags") or []),
        "text_hash": stable_hash(router_input),
    }


class BatchedQueryRewriterLora:
    def __init__(self, max_new_tokens: int) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(settings.query_rewriter_adapter_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.query_rewriter_base_model,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model = PeftModel.from_pretrained(base_model, settings.query_rewriter_adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def rewrite_batch(self, message_batches: list[list[dict[str, Any]]]) -> list[tuple[dict[str, Any], str]]:
        prompts = [
            self.tokenizer.apply_chat_template(query_rewriter_prompt(messages), tokenize=False, add_generation_prompt=True)
            for messages in message_batches
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        results: list[tuple[dict[str, Any], str]] = []
        prompt_len = inputs["input_ids"].shape[1]
        for index in range(len(message_batches)):
            generated = output_ids[index][prompt_len:]
            raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            payload = normalize_rewriter_answer(extract_json(raw_output))
            payload = complete_rewriter_payload(payload, message_batches[index])
            results.append((payload, raw_output))
        return results


def guard_rewriter_payload(label: str, raw_question: str, rewriter_payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    rewriter_payload = dict(rewriter_payload)
    non_retrieval_labels = {
        "general_no_legal_retrieval",
        "non_criminal_legal",
        "unclear_need_clarification",
        "low_quality_or_incomplete",
    }
    criminal_terms = re.compile(r"刑法|刑事|犯罪|罪名|构成要件|量刑|判刑|盗窃|诈骗|抢劫|故意伤害|故意杀人|过失致人死亡|帮信|判决书|案")
    if label in non_retrieval_labels and not criminal_terms.search(raw_question):
        standalone = str(rewriter_payload.get("standalone_query") or "")
        introduced_criminal_terms = criminal_terms.search(standalone) or rewriter_payload.get("case_name_mentions") or rewriter_payload.get("resolved_case_refs") or rewriter_payload.get("crime_mentions")
        if introduced_criminal_terms:
            return (
                {
                    "standalone_query": raw_question,
                    "resolved_case_refs": [],
                    "case_name_mentions": [],
                    "crime_mentions": [],
                    "amount_constraints": [],
                    "field_intents": [],
                    "rewrite_required": False,
                    "confidence": min(float(rewriter_payload.get("confidence") or 0.75), 0.75),
                    "warnings": ["rewriter_guard_preserved_non_retrieval_input"],
                },
                True,
            )
    return rewriter_payload, False


def build_processed_row_from_rewrite(raw: dict[str, Any], rewriter_payload: dict[str, Any], rewrite_source: str, rewrite_reason: str) -> dict[str, Any]:
    label = str(raw["label"])
    messages = normalize_source_messages(raw)
    raw_question = latest_user_query(messages)
    rewriter_payload, rewriter_guarded = guard_rewriter_payload(label, raw_question, rewriter_payload)
    router_input = router_input_from_rewriter(rewriter_payload)
    answer = answer_for(label)
    errors = validate_answer(answer)
    if errors:
        raise ValueError(f"invalid_answer:{','.join(errors)}")
    row_id = str(raw["id"])
    return {
        "id": row_id,
        "raw_question": raw_question,
        "source_messages": messages,
        "rewriter": rewriter_payload,
        "rewriter_source": rewrite_source,
        "rewriter_reason": rewrite_reason,
        "rewriter_guarded": rewriter_guarded,
        "router_input": router_input,
        "label": label,
        "answer": answer,
        "messages": build_messages(router_input, answer),
        "source": raw.get("source") or "unknown",
        "source_method": raw.get("source_method") or "unknown",
        "tags": list(raw.get("tags") or []),
        "text_hash": stable_hash(router_input),
    }


def build_processed_row_with_rule_fallback(raw: dict[str, Any], reason: str) -> dict[str, Any]:
    messages = normalize_source_messages(raw)
    payload = complete_rewriter_payload(rule_rewrite(messages, reason=reason), messages)
    row = build_processed_row_from_rewrite(raw, payload, "fallback", reason)
    warnings = row["rewriter"].setdefault("warnings", [])
    if reason not in warnings:
        warnings.append(reason)
    return row


def stratified_split(rows: list[dict[str, Any]], train_ratio: float, validation_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_n = int(total * train_ratio)
        validation_n = int(total * validation_ratio)
        if total >= 3:
            train_n = max(1, min(train_n, total - 2))
            validation_n = max(1, min(validation_n, total - train_n - 1))
        splits["train"].extend(label_rows[:train_n])
        splits["validation"].extend(label_rows[train_n : train_n + validation_n])
        splits["test"].extend(label_rows[train_n + validation_n :])
    for split, split_rows in splits.items():
        rng.shuffle(split_rows)
        for row in split_rows:
            row["split"] = split
    return splits


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_processed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def validate_processed_rows(rows: list[dict[str, Any]], min_count: int) -> dict[str, Any]:
    invalid: list[dict[str, Any]] = []
    text_hash_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        errors: list[str] = []
        answer = row.get("answer") or {}
        errors.extend(validate_answer(answer))
        assistant = json.loads(row["messages"][-1]["content"])
        forbidden = {"standalone_query", "resolved_case_refs", "amount_constraints", "field_intents", "crime_mentions"} & set(assistant)
        if forbidden:
            errors.append(f"forbidden_assistant_fields:{','.join(sorted(forbidden))}")
        if not row.get("router_input"):
            errors.append("empty_router_input")
        if errors:
            invalid.append({"id": row.get("id"), "errors": errors})
        text_hash_labels[row["text_hash"]].add(row["label"])
    conflicts = {key: sorted(value) for key, value in text_hash_labels.items() if len(value) > 1}
    if len(rows) < min_count:
        raise SystemExit(f"processed_dataset_too_small:{len(rows)} < {min_count}")
    if invalid:
        raise SystemExit(f"invalid_processed_rows:{invalid[:5]}")
    if conflicts:
        raise SystemExit(f"conflicting_processed_labels:{list(conflicts.items())[:5]}")
    return {
        "invalid_rows": 0,
        "conflict_count": 0,
        "unique_router_inputs": len({row["router_input"] for row in rows}),
        "unique_raw_questions": len({row["raw_question"] for row in rows}),
    }


def refresh_processed_router_inputs(processed_rows_path: Path) -> None:
    rows = load_processed_rows(processed_rows_path)
    if not rows:
        return
    tmp_path = processed_rows_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            raw_question = str(row.get("raw_question") or "")
            router_input = router_input_from_rewriter(row.get("rewriter") or {}, raw_question)
            row["router_input"] = router_input
            row["messages"] = build_messages(router_input, row["answer"])
            row["text_hash"] = stable_hash(router_input)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(processed_rows_path)


def finalize_dataset(output_dir: Path, processed_rows_path: Path, args: argparse.Namespace) -> None:
    refresh_processed_router_inputs(processed_rows_path)
    rows = load_processed_rows(processed_rows_path)
    validation = validate_processed_rows(rows, args.min_count)
    splits = stratified_split(rows, args.train_ratio, args.validation_ratio, args.seed)
    counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for split, split_rows in splits.items():
        counts[split] = write_jsonl(output_dir / f"{split}.jsonl", split_rows)
        label_counts[split] = dict(Counter(row["label"] for row in split_rows))
        source_counts[split] = dict(Counter(row["source"] for row in split_rows))
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}-lora-materialized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "gpt_shards_dir": str(args.gpt_shards_dir),
        "old_router_dir": str(args.old_router_dir),
        "processed_rows": str(processed_rows_path),
        "rewrite_mode": "lora",
        "query_rewriter_adapter_dir": str(settings.query_rewriter_adapter_dir),
        "seed": args.seed,
        "total_count": len(rows),
        "counts": counts,
        "label_counts": label_counts,
        "all_label_counts": dict(Counter(row["label"] for row in rows)),
        "source_counts": source_counts,
        "validation": validation,
        "system_prompt": SYSTEM_PROMPT,
        "notes": [
            "Router-only dataset materialized by running the current Query Rewriter LoRA for every raw row.",
            "processed_rows.jsonl is append-only and supports resume after interruption.",
            "Assistant answers contain only Router fields.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "total_count": len(rows), "counts": counts, "validation": validation}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Router v2 data by running Query Rewriter LoRA row by row with resumable progress.")
    parser.add_argument("--gpt-shards-dir", type=Path, default=DEFAULT_GPT_SHARDS_DIR)
    parser.add_argument("--old-router-dir", type=Path, default=OUT_DIR / "router_lora" / "dataset")
    parser.add_argument("--old-cap-per-label", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-count", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.82)
    parser.add_argument("--validation-ratio", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Optional processing cap for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of rows to rewrite per model.generate call.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for each Rewriter JSON output.")
    parser.add_argument("--validate-raw-only", action="store_true", help="Validate and summarize raw rows without loading/running the Rewriter LoRA.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help="Skip rewriting and only split existing processed_rows.jsonl.")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch by deleting processed/error JSONL files first.")
    args = parser.parse_args()

    if settings.query_rewriter_mode != "lora":
        raise SystemExit(f"QUERY_REWRITER_MODE must be lora for this script; current={settings.query_rewriter_mode}")
    if not settings.query_rewriter_adapter_dir.exists():
        raise SystemExit(f"missing query rewriter adapter: {settings.query_rewriter_adapter_dir}")
    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise SystemExit("--train-ratio and --validation-ratio must be positive and sum to less than 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_rows_path = args.output_dir / PROCESSED_ROWS_NAME
    error_rows_path = args.output_dir / ERROR_ROWS_NAME
    if args.no_resume:
        for path in [processed_rows_path, error_rows_path]:
            if path.exists():
                path.unlink()
    if args.finalize_only:
        finalize_dataset(args.output_dir, processed_rows_path, args)
        return

    raw_rows = load_gpt_raw_rows(args.gpt_shards_dir)
    raw_rows.extend(convert_old_router_rows(args.old_router_dir, args.old_cap_per_label, args.seed))
    if args.limit:
        raw_rows = raw_rows[: args.limit]

    valid_raw: list[dict[str, Any]] = []
    for row in raw_rows:
        errors = validate_raw_row(row)
        if errors:
            append_jsonl(error_rows_path, {"id": row.get("id"), "stage": "raw_validation", "errors": errors, "row": row})
            continue
        valid_raw.append(row)
    if args.validate_raw_only:
        print(
            json.dumps(
                {
                    "raw_rows": len(raw_rows),
                    "valid_raw": len(valid_raw),
                    "invalid_raw": len(raw_rows) - len(valid_raw),
                    "label_counts": dict(Counter(row["label"] for row in valid_raw)),
                    "source_counts": dict(Counter(row.get("source") for row in valid_raw)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    processed_ids = read_processed_ids(processed_rows_path)
    pending = [row for row in valid_raw if str(row["id"]) not in processed_ids]
    done = len(processed_ids)
    total = len(valid_raw)
    started_at = time.time()
    print(f"raw_rows={len(raw_rows)} valid_raw={len(valid_raw)} already_processed={done} pending={len(pending)}")
    batch_runner = BatchedQueryRewriterLora(max_new_tokens=args.max_new_tokens)
    try:
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start : start + max(1, args.batch_size)]
            try:
                message_batches = [normalize_source_messages(row) for row in batch]
                rewrites = batch_runner.rewrite_batch(message_batches)
                for row, (rewriter_payload, _raw_output) in zip(batch, rewrites):
                    processed = build_processed_row_from_rewrite(row, rewriter_payload, "lora", "query_rewriter_lora_batched")
                    append_jsonl(processed_rows_path, processed)
                    done += 1
            except Exception as exc:  # noqa: BLE001 - row-level fault isolation is required for long batch jobs.
                if not args.continue_on_error:
                    raise
                for row in batch:
                    try:
                        rewriter_payload, _raw_output = batch_runner.rewrite_batch([normalize_source_messages(row)])[0]
                        processed = build_processed_row_from_rewrite(row, rewriter_payload, "lora", "query_rewriter_lora_batched_single_fallback")
                        append_jsonl(processed_rows_path, processed)
                        done += 1
                    except Exception as row_exc:  # noqa: BLE001 - keep long jobs resumable.
                        try:
                            processed = build_processed_row_with_rule_fallback(row, reason=f"query_rewriter_lora_failed:{type(row_exc).__name__}")
                            append_jsonl(processed_rows_path, processed)
                            done += 1
                            append_jsonl(error_rows_path, {"id": row.get("id"), "stage": "rewrite_fallback", "error": f"{type(row_exc).__name__}:{row_exc}", "batch_error": f"{type(exc).__name__}:{exc}", "row": row})
                        except Exception as fallback_exc:  # noqa: BLE001 - record unrecoverable row.
                            append_jsonl(error_rows_path, {"id": row.get("id"), "stage": "rewrite", "error": f"{type(row_exc).__name__}:{row_exc}", "batch_error": f"{type(exc).__name__}:{exc}", "fallback_error": f"{type(fallback_exc).__name__}:{fallback_exc}", "row": row})
            if args.progress_every and (done % args.progress_every == 0 or done == total):
                print(render_progress(done, total, started_at=started_at), end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume from processed_rows.jsonl.")
        raise SystemExit(130)
    print()
    finalize_dataset(args.output_dir, processed_rows_path, args)


if __name__ == "__main__":
    main()
