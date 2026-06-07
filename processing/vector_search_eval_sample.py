from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from vector_search_sample import search


DEFAULT_EVAL_PATH = OUT_DIR / "vector_search_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "vector_search_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "vector_search_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def check_hits(hits: list[dict], expected: dict) -> list[dict]:
    checks = []
    top = hits[0] if hits else {}
    if expected.get("title_contains_any"):
        values = expected["title_contains_any"]
        checks.append({"name": "title_contains_any", "passed": any(contains_any(hit.get("title"), values) for hit in hits), "expected": values})
    if expected.get("source_type_any"):
        values = expected["source_type_any"]
        checks.append({"name": "source_type_any", "passed": any(hit.get("source_type") in values for hit in hits), "expected": values})
    if expected.get("field_any"):
        values = expected["field_any"]
        checks.append({"name": "field_any", "passed": any(hit.get("field") in values for hit in hits), "expected": values})
    if expected.get("law_article_no_any"):
        values = expected["law_article_no_any"]
        checks.append({"name": "law_article_no_any", "passed": any(hit.get("article_no") in values for hit in hits), "expected": values})
    if expected.get("top1_source_type"):
        checks.append({"name": "top1_source_type", "passed": top.get("source_type") == expected["top1_source_type"], "expected": expected["top1_source_type"]})
    if expected.get("top1_field_any"):
        values = expected["top1_field_any"]
        checks.append({"name": "top1_field_any", "passed": top.get("field") in values, "expected": values})
    if expected.get("top1_law_article_no_any"):
        values = expected["top1_law_article_no_any"]
        checks.append({"name": "top1_law_article_no_any", "passed": top.get("article_no") in values, "expected": values})
    if expected.get("rank_max"):
        rank_max = int(expected["rank_max"])
        values = expected.get("rank_field_any") or expected.get("field_any") or []
        checks.append({
            "name": "rank_max",
            "passed": any(hit.get("rank", 999999) <= rank_max and (not values or hit.get("field") in values) for hit in hits),
            "expected": {"rank_max": rank_max, "field_any": values},
        })
    return checks


def evaluate_case(case: dict) -> dict:
    result = search(case["query"], top_k=int(case.get("top_k", 10)))
    checks = check_hits(result["hits"], case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "status": "PASS" if not failed else "WARN",
        "checks": checks,
        "hits": result["hits"],
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Vector search eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        for hit in item["hits"][:5]:
            article = f" article_no={hit['article_no']}" if hit.get("article_no") else ""
            lines.append(f"  {hit['rank']}. {hit['source_type']} {hit['field']} {hit['title']}{article} score={hit['score']}")
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sample vector retrieval checks.")
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
