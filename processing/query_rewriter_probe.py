from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from common import OUT_DIR, ROOT
from query_rewriter_common import extract_json, query_rewriter_prompt, validate_rewriter_answer, normalize_rewriter_answer
from app.llm.query_rewriter import complete_rewriter_payload


DEFAULT_EVAL_PATH = OUT_DIR / "query_rewriter_lora" / "rewriter_probe_eval_sample.json"
DEFAULT_JSON_REPORT = OUT_DIR / "query_rewriter_lora" / "rewriter_probe_report.json"
DEFAULT_TEXT_REPORT = OUT_DIR / "query_rewriter_lora" / "rewriter_probe_report.txt"
DEFAULT_BASE_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = OUT_DIR / "query_rewriter_lora" / "qwen2_5_1_5b_rewriter_lora"


def default_eval_cases() -> list[dict[str, Any]]:
    history = [
        {"role": "user", "content": "我朋友入户盗窃40000元，有案例参考吗"},
        {
            "role": "assistant",
            "content": (
                "可以看两个参考案例。第一个参考案例是《王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书》，"
                "简称王士东案，金额约43852元。第二个参考案例是《刘方祥、刘礼芬盗窃一审刑事判决书》，简称刘方祥案，金额约39136元。"
            ),
        },
    ]
    return [
        {
            "id": "rw_first_case_fact",
            "category": "multi_turn_case_reference",
            "messages": history + [{"role": "user", "content": "第一个参考案例具体情况是什么"}],
            "expected": {
                "standalone_contains_all": ["王士东", "案情"],
                "resolved_case_title_contains_any": ["王士东"],
                "field_intents_any": ["court_found_facts_text"],
            },
        },
        {
            "id": "rw_second_case_reasoning",
            "category": "multi_turn_case_reference",
            "messages": history + [{"role": "user", "content": "第二个案例为什么判得轻一些"}],
            "expected": {
                "standalone_contains_all": ["刘方祥", "量刑"],
                "resolved_case_title_contains_any": ["刘方祥"],
                "field_intents_any": ["reasoning_text", "judgment_text"],
            },
        },
        {
            "id": "rw_two_case_compare",
            "category": "multi_case_compare",
            "messages": history + [{"role": "user", "content": "前面两个案例量刑差异在哪里"}],
            "expected": {
                "standalone_contains_all": ["王士东", "刘方祥", "差异"],
                "field_intents_any": ["reasoning_text", "judgment_text"],
            },
        },
        {
            "id": "rw_colloquial_crime",
            "category": "colloquial_crime_alias",
            "messages": [{"role": "user", "content": "我朋友过失杀人主动投案，会不会判死刑"}],
            "expected": {
                "standalone_contains_all": ["过失致人死亡罪"],
                "crime_mentions_any": ["过失致人死亡罪"],
            },
        },
        {
            "id": "rw_amount_theft",
            "category": "amount_constraint",
            "messages": [{"role": "user", "content": "入室盗窃40000元一般怎么判，有案例吗"}],
            "expected": {
                "standalone_contains_all": ["入户盗窃", "盗窃罪"],
                "crime_mentions_any": ["盗窃罪"],
                "amount_value": 40000,
            },
        },
    ]


def ensure_eval_cases(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cases = default_eval_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases


class LoraRunner:
    def __init__(self, model_dir: Path, adapter_dir: Path, max_new_tokens: int) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def rewrite(self, messages: list[dict[str, Any]], *, apply_rule_completion: bool) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(query_rewriter_prompt(messages), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = normalize_rewriter_answer(extract_json(raw))
        if apply_rule_completion:
            payload = complete_rewriter_payload(payload, messages)
        payload["_raw_output"] = raw
        return payload


def evaluate(case: dict[str, Any], payload: dict[str, Any], error: str | None) -> dict[str, Any]:
    checks = []
    checks.append({"name": "no_error", "passed": error is None, "actual": error})
    checks.append({"name": "schema_valid", "passed": not validate_rewriter_answer(payload), "actual": validate_rewriter_answer(payload)})
    expected = case.get("expected", {})
    standalone = str(payload.get("standalone_query") or "")
    if expected.get("standalone_contains_all"):
        checks.append({
            "name": "standalone_contains_all",
            "passed": all(item in standalone for item in expected["standalone_contains_all"]),
            "expected": expected["standalone_contains_all"],
            "actual": standalone,
        })
    if expected.get("resolved_case_title_contains_any"):
        titles = " ".join(str(item.get("case_title") or "") for item in payload.get("resolved_case_refs", []))
        checks.append({
            "name": "resolved_case_title_contains_any",
            "passed": any(item in titles for item in expected["resolved_case_title_contains_any"]),
            "expected": expected["resolved_case_title_contains_any"],
            "actual": titles,
        })
    if expected.get("field_intents_any"):
        actual = payload.get("field_intents", [])
        checks.append({
            "name": "field_intents_any",
            "passed": any(item in actual for item in expected["field_intents_any"]),
            "expected": expected["field_intents_any"],
            "actual": actual,
        })
    if expected.get("crime_mentions_any"):
        actual = payload.get("crime_mentions", [])
        checks.append({
            "name": "crime_mentions_any",
            "passed": any(item in actual for item in expected["crime_mentions_any"]),
            "expected": expected["crime_mentions_any"],
            "actual": actual,
        })
    if expected.get("amount_value") is not None:
        values = [item.get("value") for item in payload.get("amount_constraints", [])]
        checks.append({"name": "amount_value", "passed": expected["amount_value"] in values, "expected": expected["amount_value"], "actual": values})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a trained Query Rewriter LoRA.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-rule-completion", action="store_true", help="Evaluate the raw LoRA output without deterministic Rewriter completion.")
    args = parser.parse_args()

    cases = ensure_eval_cases(args.eval_path)
    runner = LoraRunner(args.model_dir, args.adapter_dir, args.max_new_tokens)
    rows = []
    for case in cases:
        error = None
        try:
            payload = runner.rewrite(case["messages"], apply_rule_completion=not args.no_rule_completion)
        except Exception as exc:  # noqa: BLE001 - exact failure is part of the report.
            payload = {}
            error = f"{type(exc).__name__}:{exc}"
        evaluation = evaluate(case, payload, error)
        rows.append({"id": case["id"], "category": case.get("category"), "passed": evaluation["passed"], "payload": payload, "checks": evaluation["checks"]})
    passed = sum(1 for row in rows if row["passed"])
    report = {
        "summary": {"total": len(rows), "passed": passed, "failed": len(rows) - passed, "pass_rate": round(passed / len(rows), 4) if rows else 0.0},
        "results": rows,
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Query Rewriter Probe Report", f"total: {len(rows)}", f"passed: {passed}", f"pass_rate: {report['summary']['pass_rate']:.2%}", "", "failed cases:"]
    for row in rows:
        if not row["passed"]:
            failed = ", ".join(check["name"] for check in row["checks"] if not check["passed"])
            lines.append(f"- {row['id']} [{row['category']}]: {failed}")
    args.text_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json_report: {args.json_report}")
    print(f"text_report: {args.text_report}")


if __name__ == "__main__":
    main()
