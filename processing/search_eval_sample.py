from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from search_sample import search


DEFAULT_EVAL_PATH = OUT_DIR / "search_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "search_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "search_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def direct_expectation_checks(result: dict, expected: dict) -> list[dict]:
    checks = []
    hits = result.get("direct_hits", [])

    title_values = expected.get("title_contains_any") or []
    if title_values:
        passed = any(contains_any(hit.get("title"), title_values) for hit in hits)
        checks.append({
            "name": "direct_title_contains_any",
            "passed": passed,
            "expected": title_values,
        })

    source_types = expected.get("source_type_any") or []
    if source_types:
        passed = any(hit.get("source_type") in source_types for hit in hits)
        checks.append({
            "name": "direct_source_type_any",
            "passed": passed,
            "expected": source_types,
        })

    fields = expected.get("field_any") or []
    if fields:
        passed = any(hit.get("field") in fields for hit in hits)
        checks.append({
            "name": "direct_field_any",
            "passed": passed,
            "expected": fields,
        })

    article_numbers = expected.get("law_article_no_any") or []
    if article_numbers:
        passed = any(
            hit.get("source_type") == "law_article"
            and (hit.get("law_metadata") or {}).get("article_no") in article_numbers
            for hit in hits
        )
        checks.append({
            "name": "direct_law_article_no_any",
            "passed": passed,
            "expected": article_numbers,
        })

    return checks


def graph_expectation_checks(result: dict, expected_edges: list[dict]) -> list[dict]:
    checks = []
    expansions = result.get("graph_expansions", [])
    for expected in expected_edges:
        passed = any(
            all(
                expansion.get(key) == value
                for key, value in expected.items()
            )
            for expansion in expansions
        )
        checks.append({
            "name": "graph_edge",
            "passed": passed,
            "expected": expected,
        })
    return checks


def compact_hit(hit: dict) -> dict:
    law_metadata = hit.get("law_metadata") or {}
    return {
        "score": hit.get("score"),
        "source_type": hit.get("source_type"),
        "field": hit.get("field"),
        "title": hit.get("title"),
        "retrieval_id": hit.get("retrieval_id"),
        "case_id": hit.get("case_id"),
        "article_no": law_metadata.get("article_no"),
        "law_title": law_metadata.get("title"),
        "reasons": hit.get("reasons", [])[:8],
        "text_preview": hit.get("text_preview", ""),
    }


def compact_expansion(expansion: dict) -> dict:
    law_chunk = expansion.get("law_chunk") or {}
    return {
        "case_id": expansion.get("case_id"),
        "edge_type": expansion.get("edge_type"),
        "article": expansion.get("article"),
        "article_no": expansion.get("article_no"),
        "law_name": expansion.get("law_name"),
        "law_version": expansion.get("law_version"),
        "resolution": expansion.get("resolution"),
        "law_chunk_title": law_chunk.get("title"),
        "law_chunk_id": law_chunk.get("retrieval_id"),
    }


def evaluate_case(case: dict) -> dict:
    query = case["query"]
    top_k = int(case.get("top_k", 5))
    result = search(
        query,
        top_k=top_k,
        expansion_k=int(case.get("expansion_k", 8)),
        context_k=int(case.get("context_k", 12)),
    )

    checks = []
    if case.get("expected_direct"):
        checks.extend(direct_expectation_checks(result, case["expected_direct"]))
    if case.get("expected_graph"):
        checks.extend(graph_expectation_checks(result, case["expected_graph"]))

    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", query),
        "query": query,
        "status": "PASS" if not failed_checks else "WARN",
        "checks": checks,
        "direct_hits": [compact_hit(hit) for hit in result.get("direct_hits", [])],
        "graph_expansions": [compact_expansion(item) for item in result.get("graph_expansions", [])],
        "final_context": [compact_hit(item) | {"context_source": item.get("context_source")} for item in result.get("final_context", [])],
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Search eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        for hit_idx, hit in enumerate(item["direct_hits"][:3], 1):
            article = f" article_no={hit['article_no']}" if hit.get("article_no") else ""
            lines.append(
                f"  direct {hit_idx}. {hit['source_type']} {hit['field']} {hit['title']}{article} score={hit['score']}"
            )
        for expansion in item["graph_expansions"][:5]:
            lines.append(
                "  graph. "
                f"{expansion['edge_type']} {expansion['article']} "
                f"({expansion['law_version']}) -> {expansion['resolution']}"
            )
        failed = [check for check in item["checks"] if not check["passed"]]
        for check in failed:
            lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed sample queries against search_sample.py.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    eval_cases = load_eval_cases(args.eval_file)
    results = [evaluate_case(case) for case in eval_cases]
    warn_count = sum(1 for item in results if item["status"] != "PASS")
    report = {
        "status": "PASS" if warn_count == 0 else "WARN",
        "case_count": len(results),
        "pass_count": len(results) - warn_count,
        "warn_count": warn_count,
        "eval_file": str(args.eval_file),
        "results": results,
    }

    args.json_report.parent.mkdir(parents=True, exist_ok=True)
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
