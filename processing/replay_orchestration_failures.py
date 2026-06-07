from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from common import OUT_DIR
from app.llm.chat_service import answer_chat
from grounding_validator_sample import validate_grounding
from langchain_orchestration_sample import run_orchestration_with_mode


DEFAULT_FAILURE_CASES = OUT_DIR / "orchestration_failure_cases.json"
DEFAULT_JSON_REPORT = OUT_DIR / "orchestration_failure_replay_report.json"
DEFAULT_TEXT_REPORT = OUT_DIR / "orchestration_failure_replay_report.txt"


def load_cases(path: Path, case_id: str | None) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if case_id:
        rows = [row for row in rows if row.get("id") == case_id or row.get("query") == case_id]
    if not rows:
        raise SystemExit(f"No failure cases matched: {case_id or path}")
    return rows


def replay_case(case: dict, generation_mode: str) -> dict:
    messages = case.get("messages") or [{"role": "user", "content": case["query"]}]
    top_k = int(case.get("top_k", case.get("stable", {}).get("top_k", 8) or 8))
    stable = asyncio.run(answer_chat(messages, top_k=top_k))
    orchestration = run_orchestration_with_mode(messages, top_k=top_k, generation_mode=generation_mode)
    before = {
        "stable_citation_count": case.get("stable_citation_count"),
        "orchestration_citation_count": case.get("orchestration_citation_count"),
        "failure_type": case.get("failure_type"),
        "hallucination_risk": case.get("hallucination_risk"),
        "provider_error": case.get("provider_error"),
    }
    after = {
        "stable_answer_type": stable.get("answer_type"),
        "orchestration_answer_type": orchestration.get("answer_type"),
        "stable_citation_count": len(stable.get("citations", [])),
        "orchestration_citation_count": len(orchestration.get("citations", [])),
        "stable_warnings": stable.get("warnings", []),
        "orchestration_warnings": orchestration.get("warnings", []),
        "stable_graph_paths": len(stable.get("graph_paths", [])),
        "orchestration_graph_paths": len(orchestration.get("graph_paths", [])),
    }
    return {
        "id": case.get("id"),
        "query": case.get("query"),
        "generation_mode": generation_mode,
        "before": before,
        "after": after,
        "citation_delta": after["orchestration_citation_count"] - int(before.get("orchestration_citation_count") or 0),
        "warnings_changed": before.get("failure_type") not in after["orchestration_warnings"],
        "grounding": validate_grounding(case["query"], orchestration, stable),
        "stable_answer_preview": str(stable.get("answer") or "")[:260],
        "orchestration_answer_preview": str(orchestration.get("answer") or "")[:260],
    }


def write_text(path: Path, report: dict) -> None:
    lines = [
        "Orchestration failure replay report",
        f"created_at_utc: {report['created_at_utc']}",
        f"generation_mode: {report['generation_mode']}",
        f"cases: {report['case_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['id']}] {item['query']}")
        lines.append(f"  before: {json.dumps(item['before'], ensure_ascii=False)}")
        lines.append(f"  after: {json.dumps(item['after'], ensure_ascii=False)}")
        lines.append(f"  grounding: {json.dumps(item['grounding'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay orchestration failure cases and compare current behavior to recorded failures.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--id", dest="case_id", help="Replay one case by id or exact query.")
    parser.add_argument("--generation-mode", choices=["mock", "provider"], default="provider")
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT)
    args = parser.parse_args()

    results = [replay_case(case, args.generation_mode) for case in load_cases(args.cases, args.case_id)]
    report = {
        "schema_version": "orchestration-failure-replay-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_mode": args.generation_mode,
        "case_count": len(results),
        "results": results,
    }
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_text(args.text_report, report)
    print(f"cases: {len(results)}")
    print(f"wrote {args.json_report}")
    print(f"wrote {args.text_report}")


if __name__ == "__main__":
    main()
