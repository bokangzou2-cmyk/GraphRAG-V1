from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import OUT_DIR
from app.llm.chat_service import answer_chat
from grounding_validator_sample import validate_grounding
from langchain_orchestration_sample import run_orchestration_with_mode


DEFAULT_JSON_REPORT = OUT_DIR / "chat_vs_orchestration_eval_report.json"
DEFAULT_TEXT_REPORT = OUT_DIR / "chat_vs_orchestration_eval_report.txt"
DEFAULT_PROVIDER_JSON_REPORT = OUT_DIR / "chat_vs_orchestration_provider_eval_report.json"
DEFAULT_PROVIDER_TEXT_REPORT = OUT_DIR / "chat_vs_orchestration_provider_eval_report.txt"
DEFAULT_PROVIDER_GROUNDING_JSON_REPORT = OUT_DIR / "provider_grounding_eval_report.json"
DEFAULT_PROVIDER_GROUNDING_TEXT_REPORT = OUT_DIR / "provider_grounding_eval_report.txt"
DEFAULT_CASES_PATH = OUT_DIR / "chat_vs_orchestration_eval_cases.json"

EVAL_CASES = [
    {"id": "general_weather", "query": "今天天气怎么样？", "top_k": 5, "category": "general"},
    {"id": "general_poem", "query": "帮我写一首五言绝句", "top_k": 5, "category": "general"},
    {"id": "non_criminal_divorce", "query": "离婚财产怎么分？", "top_k": 5, "category": "non_criminal_legal"},
    {"id": "law_only_theft", "query": "盗窃罪怎么判？", "top_k": 5, "category": "criminal_law_law_only"},
    {"id": "law_only_attempt", "query": "犯罪未遂的构成特征是什么？", "top_k": 5, "category": "criminal_law_law_only"},
    {"id": "case_only_sentence", "query": "吴必定案判了多久", "top_k": 5, "category": "criminal_law_case"},
    {"id": "case_only_fact", "query": "吴必定案案情是什么？", "top_k": 5, "category": "criminal_law_case"},
    {"id": "case_and_law_basis", "query": "吴必定案判了多久，依据什么法条？", "top_k": 8, "category": "criminal_law_case_and_law"},
    {"id": "unknown_case", "query": "不存在案判了多久", "top_k": 5, "category": "criminal_law_case"},
    {"id": "empty_query", "query": "   ", "top_k": 5, "category": "low_quality"},
]


def retrieval_ids(response: dict) -> list[str]:
    return [str(row.get("retrieval_id")) for row in response.get("retrieval_context", []) if row.get("retrieval_id")]


def citation_count(response: dict) -> int:
    return len(response.get("citations", []))


def source_types(response: dict) -> list[str]:
    return sorted({str(row.get("source_type")) for row in response.get("retrieval_context", []) if row.get("source_type")})


def graph_path_count(response: dict) -> int:
    return len(response.get("graph_paths", []))


def confidence_summary(response: dict) -> dict:
    confidence = response.get("confidence") or {}
    return {
        "level": confidence.get("level"),
        "score": confidence.get("score"),
        "reason": confidence.get("reason"),
        "query_type": confidence.get("query_type"),
        "router_label": (confidence.get("router") or {}).get("label"),
    }


def answer_preview(response: dict, limit: int = 220) -> str:
    return str(response.get("answer") or "")[:limit]


def sorted_warnings(response: dict) -> list[str]:
    return sorted(response.get("warnings") or [])


def provider_errors(response: dict) -> list[dict]:
    errors = []
    if response.get("provider_error"):
        errors.append({"source": "provider_error_field", **response["provider_error"]})
    if "llm_call_failed" in set(response.get("warnings") or []):
        errors.append(
            {
                "source": "response",
                "refusal_reason": response.get("refusal_reason"),
                "answer_preview": answer_preview(response),
            }
        )
    for log in response.get("orchestration_logs", []) or []:
        if log.get("event") == "provider_llm_failed":
            errors.append({"source": "orchestration_log", **(log.get("payload") or {})})
    return errors


def hallucination_risk(response: dict) -> dict:
    warnings = set(response.get("warnings") or [])
    answer_type = response.get("answer_type")
    context_count = len(response.get("retrieval_context") or [])
    citations = citation_count(response)
    confidence = response.get("confidence") or {}
    reasons = []
    if answer_type == "llm_grounded" and citations == 0:
        reasons.append("grounded_answer_without_citations")
    if answer_type == "llm_grounded" and context_count == 0:
        reasons.append("grounded_answer_without_context")
    if "llm_invalid_json" in warnings:
        reasons.append("llm_invalid_json")
    if "llm_call_failed" in warnings:
        reasons.append("llm_call_failed")
    if confidence.get("level") == "low" and answer_type == "llm_grounded":
        reasons.append("grounded_answer_low_confidence")
    if reasons:
        return {"level": "high" if any("without" in item for item in reasons) else "medium", "reasons": reasons}
    return {"level": "low", "reasons": []}


def risk_rank(level: str | None) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(level), 0)


def compare_lists(left: list[str], right: list[str]) -> dict:
    left_set = set(left)
    right_set = set(right)
    return {
        "same_order": left == right,
        "same_set": left_set == right_set,
        "left_only": [item for item in left if item not in right_set],
        "right_only": [item for item in right if item not in left_set],
    }


def classify_difference(stable: dict, orchestration: dict, retrieval_diff: dict, issues: list[dict]) -> tuple[str, list[dict]]:
    if not issues and retrieval_diff["same_order"]:
        return "完全一致", issues

    risk_reasons = []
    stable_provider_failed = bool(provider_errors(stable))
    orchestration_provider_failed = bool(provider_errors(orchestration))
    if stable.get("answer_type") != orchestration.get("answer_type"):
        stable_type = stable.get("answer_type")
        orch_type = orchestration.get("answer_type")
        if not stable_provider_failed and stable_type in {"llm_grounded", "retrieval_fallback", "low_confidence"} and orch_type == "no_answer":
            risk_reasons.append("orchestration_misrefused")
        elif not stable_provider_failed and stable_type == "llm_grounded" and orch_type != "llm_grounded":
            risk_reasons.append("orchestration_answer_type_downgrade")
    if not stable_provider_failed and citation_count(orchestration) < citation_count(stable):
        risk_reasons.append("orchestration_citation_count_decreased")
    if stable.get("retrieval_context") and not orchestration.get("retrieval_context"):
        risk_reasons.append("orchestration_missing_context")
    if graph_path_count(stable) > 0 and graph_path_count(orchestration) == 0:
        risk_reasons.append("orchestration_lost_graph_paths")
    if retrieval_diff["left_only"]:
        risk_reasons.append("orchestration_retrieval_ids_missing")
    stable_hallucination = hallucination_risk(stable)
    orchestration_hallucination = hallucination_risk(orchestration)
    if risk_rank(orchestration_hallucination["level"]) > risk_rank(stable_hallucination["level"]):
        risk_reasons.append("orchestration_hallucination_risk_increased")
    if orchestration_provider_failed:
        risk_reasons.append("orchestration_provider_call_failed")

    for reason in risk_reasons:
        issues.append({"severity": "risk", "reason": reason})
    if risk_reasons:
        return "风险差异", issues
    return "可接受差异", issues


def compare_case(case: dict, stable: dict, orchestration: dict) -> dict:
    issues: list[dict] = []
    stable_ids = retrieval_ids(stable)
    orchestration_ids = retrieval_ids(orchestration)
    retrieval_diff = compare_lists(stable_ids, orchestration_ids)
    stable_grounding = validate_grounding(case["query"], stable)
    orchestration_grounding = validate_grounding(case["query"], orchestration, stable)

    if stable.get("answer_type") != orchestration.get("answer_type"):
        issues.append({
            "severity": "info",
            "reason": "answer_type_diff",
            "stable": stable.get("answer_type"),
            "orchestration": orchestration.get("answer_type"),
        })
    if citation_count(stable) != citation_count(orchestration):
        issues.append({
            "severity": "info",
            "reason": "citation_count_diff",
            "stable": citation_count(stable),
            "orchestration": citation_count(orchestration),
        })
    if sorted_warnings(stable) != sorted_warnings(orchestration):
        issues.append({
            "severity": "info",
            "reason": "warnings_diff",
            "stable": sorted_warnings(stable),
            "orchestration": sorted_warnings(orchestration),
        })
    if confidence_summary(stable) != confidence_summary(orchestration):
        issues.append({
            "severity": "info",
            "reason": "confidence_diff",
            "stable": confidence_summary(stable),
            "orchestration": confidence_summary(orchestration),
        })
    if not retrieval_diff["same_order"]:
        issues.append({
            "severity": "info",
            "reason": "retrieval_id_diff",
            "stable_count": len(stable_ids),
            "orchestration_count": len(orchestration_ids),
            "stable_only": retrieval_diff["left_only"][:10],
            "orchestration_only": retrieval_diff["right_only"][:10],
        })
    if graph_path_count(stable) != graph_path_count(orchestration):
        issues.append({
            "severity": "info",
            "reason": "graph_path_count_diff",
            "stable": graph_path_count(stable),
            "orchestration": graph_path_count(orchestration),
        })

    classification, issues = classify_difference(stable, orchestration, retrieval_diff, issues)
    return {
        "id": case["id"],
        "query": case["query"],
        "category": case.get("category"),
        "classification": classification,
        "issues": issues,
        "stable": {
            "answer_type": stable.get("answer_type"),
            "citations_count": citation_count(stable),
            "warnings": sorted_warnings(stable),
            "confidence": confidence_summary(stable),
            "hallucination_risk": hallucination_risk(stable),
            "grounding": stable_grounding,
            "provider_errors": provider_errors(stable),
            "retrieval_ids": stable_ids,
            "source_types": source_types(stable),
            "graph_paths_count": graph_path_count(stable),
            "refusal_reason": stable.get("refusal_reason"),
            "answer_preview": answer_preview(stable),
        },
        "orchestration": {
            "answer_type": orchestration.get("answer_type"),
            "citations_count": citation_count(orchestration),
            "warnings": sorted_warnings(orchestration),
            "confidence": confidence_summary(orchestration),
            "hallucination_risk": hallucination_risk(orchestration),
            "grounding": orchestration_grounding,
            "provider_errors": provider_errors(orchestration),
            "retrieval_ids": orchestration_ids,
            "source_types": source_types(orchestration),
            "graph_paths_count": graph_path_count(orchestration),
            "refusal_reason": orchestration.get("refusal_reason"),
            "answer_preview": answer_preview(orchestration),
            "log_events": [item.get("event") for item in orchestration.get("orchestration_logs", [])],
        },
    }


def evaluate_case(case: dict, orchestration_generation_mode: str) -> dict:
    messages = case.get("messages") or [{"role": "user", "content": case["query"]}]
    top_k = int(case.get("top_k", 5))
    stable = asyncio.run(answer_chat(messages, top_k=top_k))
    orchestration = run_orchestration_with_mode(
        messages,
        top_k=top_k,
        mock_llm_mode=case.get("mock_llm_mode", "grounded"),
        generation_mode=orchestration_generation_mode,
    )
    return compare_case(case, stable, orchestration)


def average_score(results: list[dict], side: str, key: str) -> float:
    values = [float(item[side].get("grounding", {}).get(key, 0.0)) for item in results]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def grounding_summary(results: list[dict]) -> dict:
    return {
        "stable": {
            "grounded_score_avg": average_score(results, "stable", "grounded_score"),
            "citation_coverage_score_avg": average_score(results, "stable", "citation_coverage_score"),
            "hallucination_risk_score_avg": average_score(results, "stable", "hallucination_risk_score"),
            "refusal_quality_score_avg": average_score(results, "stable", "refusal_quality_score"),
        },
        "orchestration": {
            "grounded_score_avg": average_score(results, "orchestration", "grounded_score"),
            "citation_coverage_score_avg": average_score(results, "orchestration", "citation_coverage_score"),
            "hallucination_risk_score_avg": average_score(results, "orchestration", "hallucination_risk_score"),
            "refusal_quality_score_avg": average_score(results, "orchestration", "refusal_quality_score"),
            "high_hallucination_risk_count": sum(
                1
                for item in results
                if float(item["orchestration"].get("grounding", {}).get("hallucination_risk_score", 0.0)) >= 0.5
            ),
            "low_citation_coverage_count": sum(
                1
                for item in results
                if float(item["orchestration"].get("grounding", {}).get("citation_coverage_score", 1.0)) < 0.8
            ),
        },
    }


def evaluate(cases: list[dict], orchestration_generation_mode: str) -> dict:
    results = []
    for case in cases:
        results.append(evaluate_case(case, orchestration_generation_mode))
    counts = Counter(item["classification"] for item in results)
    risk_samples = [item for item in results if item["classification"] == "风险差异"]
    issue_counts = Counter(issue["reason"] for item in results for issue in item["issues"] if issue.get("severity") == "risk")
    provider_error_samples = [
        item
        for item in results
        if item["stable"].get("provider_errors") or item["orchestration"].get("provider_errors")
    ]
    return {
        "schema_version": "chat-vs-orchestration-eval-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "orchestration_generation_mode": orchestration_generation_mode,
        "case_count": len(results),
        "classification_counts": counts,
        "risk_issue_counts": issue_counts,
        "provider_error_count": len(provider_error_samples),
        "provider_error_samples": provider_error_samples,
        "provider_grounding": grounding_summary(results),
        "risk_samples": risk_samples,
        "results": results,
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Chat vs LangChain orchestration eval report",
        f"created_at_utc: {report['created_at_utc']}",
        f"orchestration_generation_mode: {report['orchestration_generation_mode']}",
        f"cases: {report['case_count']}",
        "classification_counts: " + json.dumps(report["classification_counts"], ensure_ascii=False),
        "risk_issue_counts: " + json.dumps(report["risk_issue_counts"], ensure_ascii=False),
        f"provider_error_count: {report.get('provider_error_count', 0)}",
        "provider_grounding: " + json.dumps(report.get("provider_grounding", {}), ensure_ascii=False),
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['classification']}] {item['id']} ({item['category']}): {item['query']!r}")
        lines.append(
            "  answer_type: stable={stable} orchestration={orch}".format(
                stable=item["stable"]["answer_type"],
                orch=item["orchestration"]["answer_type"],
            )
        )
        lines.append(
            "  citations: stable={stable} orchestration={orch}; graph_paths: stable={sg} orchestration={og}".format(
                stable=item["stable"]["citations_count"],
                orch=item["orchestration"]["citations_count"],
                sg=item["stable"]["graph_paths_count"],
                og=item["orchestration"]["graph_paths_count"],
            )
        )
        lines.append(
            "  confidence: stable={stable} orchestration={orch}".format(
                stable=json.dumps(item["stable"]["confidence"], ensure_ascii=False),
                orch=json.dumps(item["orchestration"]["confidence"], ensure_ascii=False),
            )
        )
        lines.append(
            "  hallucination_risk: stable={stable} orchestration={orch}".format(
                stable=json.dumps(item["stable"]["hallucination_risk"], ensure_ascii=False),
                orch=json.dumps(item["orchestration"]["hallucination_risk"], ensure_ascii=False),
            )
        )
        lines.append(
            "  grounding: stable={stable} orchestration={orch}".format(
                stable=json.dumps(item["stable"].get("grounding", {}), ensure_ascii=False),
                orch=json.dumps(item["orchestration"].get("grounding", {}), ensure_ascii=False),
            )
        )
        if item["stable"].get("provider_errors") or item["orchestration"].get("provider_errors"):
            lines.append(
                "  provider_errors: stable={stable} orchestration={orch}".format(
                    stable=json.dumps(item["stable"].get("provider_errors"), ensure_ascii=False),
                    orch=json.dumps(item["orchestration"].get("provider_errors"), ensure_ascii=False),
                )
            )
        if item["issues"]:
            for issue in item["issues"]:
                prefix = "  risk" if issue.get("severity") == "risk" else "  diff"
                lines.append(f"{prefix}: {json.dumps(issue, ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_provider_grounding_reports(report: dict) -> None:
    grounding_report = {
        "schema_version": "provider-grounding-eval-v1",
        "created_at_utc": report["created_at_utc"],
        "orchestration_generation_mode": report["orchestration_generation_mode"],
        "case_count": report["case_count"],
        "classification_counts": report["classification_counts"],
        "risk_issue_counts": report["risk_issue_counts"],
        "provider_error_count": report.get("provider_error_count", 0),
        "provider_grounding": report.get("provider_grounding", {}),
        "high_risk_samples": [
            item
            for item in report["results"]
            if item["classification"] == "风险差异"
            or float(item["orchestration"].get("grounding", {}).get("hallucination_risk_score", 0.0)) >= 0.5
        ],
        "results": [
            {
                "id": item["id"],
                "query": item["query"],
                "category": item.get("category"),
                "classification": item["classification"],
                "stable_grounding": item["stable"].get("grounding", {}),
                "orchestration_grounding": item["orchestration"].get("grounding", {}),
                "stable_citations_count": item["stable"].get("citations_count"),
                "orchestration_citations_count": item["orchestration"].get("citations_count"),
                "stable_warnings": item["stable"].get("warnings"),
                "orchestration_warnings": item["orchestration"].get("warnings"),
                "issues": item.get("issues", []),
            }
            for item in report["results"]
        ],
    }
    DEFAULT_PROVIDER_GROUNDING_JSON_REPORT.write_text(
        json.dumps(grounding_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "Provider grounding eval report",
        f"created_at_utc: {grounding_report['created_at_utc']}",
        f"cases: {grounding_report['case_count']}",
        "classification_counts: " + json.dumps(grounding_report["classification_counts"], ensure_ascii=False),
        "risk_issue_counts: " + json.dumps(grounding_report["risk_issue_counts"], ensure_ascii=False),
        "provider_grounding: " + json.dumps(grounding_report["provider_grounding"], ensure_ascii=False),
        f"high_risk_samples: {len(grounding_report['high_risk_samples'])}",
        "",
    ]
    for item in grounding_report["results"]:
        lines.append(f"[{item['classification']}] {item['id']}: {item['query']!r}")
        lines.append("  stable_grounding: " + json.dumps(item["stable_grounding"], ensure_ascii=False))
        lines.append("  orchestration_grounding: " + json.dumps(item["orchestration_grounding"], ensure_ascii=False))
        if item["issues"]:
            for issue in item["issues"]:
                lines.append("  issue: " + json.dumps(issue, ensure_ascii=False))
        lines.append("")
    DEFAULT_PROVIDER_GROUNDING_TEXT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def load_cases(path: Path | None) -> list[dict]:
    if not path:
        path = DEFAULT_CASES_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("eval cases file must contain a JSON list")
        return data
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("eval cases file must contain a JSON list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy answer_chat behavior with the current LangChain orchestration behavior.")
    parser.add_argument("--cases", type=Path, help=f"Optional JSON list of cases. Defaults to {DEFAULT_CASES_PATH}.")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--text-report", type=Path)
    parser.add_argument("--generation-mode", "--orchestration-generation-mode", dest="generation_mode", choices=["mock", "provider"], default="mock")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N cases.")
    parser.add_argument("--fail-on-risk", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    json_report = args.json_report or (DEFAULT_PROVIDER_JSON_REPORT if args.generation_mode == "provider" else DEFAULT_JSON_REPORT)
    text_report = args.text_report or (DEFAULT_PROVIDER_TEXT_REPORT if args.generation_mode == "provider" else DEFAULT_TEXT_REPORT)

    report = evaluate(cases, args.generation_mode)
    json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_text_report(text_report, report)
    if args.generation_mode == "provider":
        write_provider_grounding_reports(report)
    print(f"cases: {report['case_count']}")
    print("classification_counts: " + json.dumps(report["classification_counts"], ensure_ascii=False))
    print("risk_issue_counts: " + json.dumps(report["risk_issue_counts"], ensure_ascii=False))
    print(f"provider_error_count: {report.get('provider_error_count', 0)}")
    print("provider_grounding: " + json.dumps(report.get("provider_grounding", {}), ensure_ascii=False))
    print(f"wrote {json_report}")
    print(f"wrote {text_report}")
    if args.generation_mode == "provider":
        print(f"wrote {DEFAULT_PROVIDER_GROUNDING_JSON_REPORT}")
        print(f"wrote {DEFAULT_PROVIDER_GROUNDING_TEXT_REPORT}")
    if args.fail_on_risk and report["risk_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
