from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from hybrid_search_sample import search as hybrid_search
from langchain_orchestration_sample import run_orchestration


DEFAULT_EVAL_PATH = OUT_DIR / "langchain_orchestration_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "langchain_orchestration_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "langchain_orchestration_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def check_equal(name: str, actual: Any, expected: Any) -> dict:
    return {"name": name, "passed": actual == expected, "expected": expected, "actual": actual}


def check_contains_all(name: str, actual: list, expected: list) -> dict:
    return {
        "name": name,
        "passed": all(item in actual for item in expected),
        "expected": expected,
        "actual": actual,
    }


def retrieval_ids(rows: list[dict]) -> list[str]:
    return [row.get("retrieval_id") for row in rows if row.get("retrieval_id")]


def logs_by_event(logs: list[dict]) -> dict[str, dict]:
    return {str(item.get("event")): item.get("payload", {}) for item in logs}


def check_required_log_payloads(result: dict) -> list[dict]:
    logs = logs_by_event(result.get("orchestration_logs", []))
    checks = []
    if "router_result" in logs:
        payload = logs["router_result"]
        required = ["label", "needs_retrieval", "retrieval_targets", "enabled_retrievers", "source_types"]
        checks.append({
            "name": "router_result_log_payload_fields",
            "passed": all(field in payload for field in required),
            "expected": required,
            "actual": sorted(payload),
        })
    if "hybrid_merge_rerank" in logs:
        payload = logs["hybrid_merge_rerank"]
        required = [
            "router_result",
            "enabled_retrievers",
            "lexical_count",
            "graph_count",
            "vector_count",
            "final_context_count",
            "confidence",
            "warnings",
        ]
        checks.append({
            "name": "hybrid_merge_rerank_log_payload_fields",
            "passed": all(field in payload for field in required),
            "expected": required,
            "actual": sorted(payload),
        })
        rows = payload.get("selected_context") or []
        if rows:
            first = rows[0]
            required_row = ["retrieval_id", "retrieval_sources", "ranking_adjustments", "graph_path_count"]
            checks.append({
                "name": "hybrid_log_context_diagnostics",
                "passed": all(field in first for field in required_row),
                "expected": required_row,
                "actual": sorted(first),
            })
    if "answer_contract_prepared" in logs:
        payload = logs["answer_contract_prepared"]
        required = ["confidence", "citation_count", "graph_path_count", "warnings", "citation_ids", "llm_message_role_counts"]
        checks.append({
            "name": "answer_contract_log_payload_fields",
            "passed": all(field in payload for field in required),
            "expected": required,
            "actual": sorted(payload),
        })
    if "mock_llm_output" in logs:
        payload = logs["mock_llm_output"]
        required = ["mode", "raw_preview"]
        checks.append({
            "name": "mock_llm_log_payload_fields",
            "passed": all(field in payload for field in required),
            "expected": required,
            "actual": sorted(payload),
        })
    if "final_response" in logs:
        payload = logs["final_response"]
        required = ["answer_type", "warning_count", "citation_count", "refusal_reason"]
        checks.append({
            "name": "final_response_log_payload_fields",
            "passed": all(field in payload for field in required),
            "expected": required,
            "actual": sorted(payload),
        })
    return checks


def check_case_result(result: dict, direct_result: dict | None, expected: dict) -> list[dict]:
    checks = []
    if expected.get("answer_type"):
        checks.append(check_equal("answer_type", result.get("answer_type"), expected["answer_type"]))
    if "refusal_reason" in expected:
        checks.append(check_equal("refusal_reason", result.get("refusal_reason"), expected["refusal_reason"]))
    if "citations_min" in expected:
        actual = len(result.get("citations", []))
        checks.append({"name": "citations_min", "passed": actual >= int(expected["citations_min"]), "expected": expected["citations_min"], "actual": actual})
    if "citations_exact" in expected:
        actual = len(result.get("citations", []))
        checks.append({"name": "citations_exact", "passed": actual == int(expected["citations_exact"]), "expected": expected["citations_exact"], "actual": actual})
    if expected.get("warnings"):
        checks.append(check_contains_all("warnings", result.get("warnings", []), expected["warnings"]))
    if expected.get("warnings_absent"):
        warnings = result.get("warnings", [])
        checks.append({
            "name": "warnings_absent",
            "passed": all(item not in warnings for item in expected["warnings_absent"]),
            "expected": expected["warnings_absent"],
            "actual": warnings,
        })
    if expected.get("log_events"):
        events = [item.get("event") for item in result.get("orchestration_logs", [])]
        checks.append(check_contains_all("log_events", events, expected["log_events"]))
    if expected.get("same_retrieval_as_hybrid"):
        direct_ids = retrieval_ids((direct_result or {}).get("selected_context", []))
        wrapped_ids = retrieval_ids(result.get("retrieval_context", []))
        checks.append(check_equal("same_retrieval_as_hybrid", wrapped_ids, direct_ids))
        checks.append(check_contains_all("direct_warnings_preserved", result.get("warnings", []), (direct_result or {}).get("warnings", [])))
    allowed_ids = {item.get("retrieval_id") for item in result.get("retrieval_context", []) if item.get("retrieval_id")}
    citation_ids = [item.get("retrieval_id") for item in result.get("citations", []) if item.get("retrieval_id")]
    checks.append({
        "name": "citations_in_retrieval_context",
        "passed": all(item in allowed_ids for item in citation_ids),
        "expected": "all citation retrieval_id values are present in retrieval_context",
        "actual": citation_ids,
    })
    if result.get("answer_type") == "no_answer":
        checks.append(check_equal("no_answer_has_empty_citations", len(result.get("citations", [])), 0))
    checks.extend(check_required_log_payloads(result))
    if expected.get("assistant_history_excluded_from_prompt"):
        prepared = logs_by_event(result.get("orchestration_logs", [])).get("answer_contract_prepared", {})
        role_counts = prepared.get("llm_message_role_counts", {})
        checks.append(check_equal("assistant_history_excluded_from_prompt", role_counts.get("assistant", 0), 0))
    return checks


def evaluate_case(case: dict) -> dict:
    top_k = int(case.get("top_k", 10))
    messages = case.get("messages") or [{"role": "user", "content": case["query"]}]
    result = run_orchestration(messages, top_k=top_k, mock_llm_mode=case.get("mock_llm_mode", "grounded"))
    direct_result = None
    if case.get("expected", {}).get("same_retrieval_as_hybrid") and case["query"].strip():
        source_types = (result.get("confidence") or {}).get("router", {}).get("source_types")
        direct_result = hybrid_search(case["query"], top_k=top_k, source_types=source_types)
    checks = check_case_result(result, direct_result, case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "mock_llm_mode": case.get("mock_llm_mode", "grounded"),
        "status": "PASS" if not failed else "WARN",
        "checks": checks,
        "answer_type": result.get("answer_type"),
        "warnings": result.get("warnings", []),
        "confidence": result.get("confidence"),
        "refusal_reason": result.get("refusal_reason"),
        "citations": result.get("citations", []),
        "orchestration_logs": result.get("orchestration_logs", []),
        "selected_context": result.get("retrieval_context", []),
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "LangChain orchestration eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']} mode={item['mock_llm_mode']}")
        lines.append(f"  answer_type={item['answer_type']} confidence={item['confidence'].get('level')} warnings={','.join(item['warnings']) if item['warnings'] else '-'}")
        lines.append(f"  citations={len(item['citations'])} logs={','.join(log.get('event', '') for log in item['orchestration_logs'])}")
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)} actual={json.dumps(check['actual'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain orchestration checks with mock LLM output.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    results = [evaluate_case(case) for case in load_eval_cases(args.eval_file)]
    warn_count = sum(1 for item in results if item["status"] != "PASS")
    report = {
        "status": "PASS" if warn_count == 0 else "WARN",
        "case_count": len(results),
        "pass_count": len(results) - warn_count,
        "warn_count": warn_count,
        "eval_file": str(args.eval_file),
        "results": results,
    }
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_text_report(args.text_report, report)
    print(f"status: {report['status']}")
    print(f"cases: {report['case_count']} pass: {report['pass_count']} warn: {report['warn_count']}")
    print(f"wrote {args.json_report}")
    print(f"wrote {args.text_report}")
    if args.fail_on_warn and warn_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
