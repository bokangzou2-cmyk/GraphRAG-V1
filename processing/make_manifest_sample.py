from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from common import OUT_DIR, ROOT, iter_jsonl


OUTPUT = OUT_DIR / "manifest_sample.json"

ARTIFACTS = [
    "law_articles_sample.jsonl",
    "cases_sample.jsonl",
    "cases_enriched_sample.jsonl",
    "case_chunks_sample.jsonl",
    "law_chunks_sample.jsonl",
    "retrieval_chunks_sample.jsonl",
    "graph_sample.jsonl",
    "graph_nodes_sample.jsonl",
    "graph_edges_sample.jsonl",
    "graph_retriever_eval_sample.json",
    "graph_retriever_eval_report_sample.json",
    "graph_retriever_eval_report_sample.txt",
    "data_quality_report_sample.json",
    "sample_report.txt",
]

SCRIPTS = [
    "parse_law_sample.py",
    "parse_cases_sample.py",
    "enrich_cases_sample.py",
    "chunk_cases_sample.py",
    "chunk_law_sample.py",
    "build_retrieval_chunks_sample.py",
    "build_graph_sample.py",
    "graph_retriever_sample.py",
    "graph_retriever_eval_sample.py",
    "validate_sample_outputs.py",
    "make_sample_report.py",
    "make_manifest_sample.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    return sum(1 for _ in iter_jsonl(path))


def artifact_row(name: str) -> dict:
    path = OUT_DIR / name
    row = {
        "name": name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": path.exists(),
    }
    if not path.exists():
        return row
    stat = path.stat()
    row.update({
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
        "record_count": jsonl_count(path),
    })
    return row


def script_row(name: str) -> dict:
    path = ROOT / "processing" / name
    row = {
        "name": name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": path.exists(),
    }
    if path.exists():
        row["sha256"] = sha256_file(path)
    return row


def load_quality_status() -> dict:
    path = OUT_DIR / "data_quality_report_sample.json"
    if not path.exists():
        return {"status": "missing"}
    with path.open("r", encoding="utf-8-sig") as f:
        report = json.load(f)
    return {
        "status": report.get("status"),
        "error_count": len(report.get("errors", [])),
        "warning_count": len(report.get("warnings", [])),
    }


def build_manifest() -> dict:
    return {
        "schema_version": "sample-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "inputs": [
            {"name": "law_text", "path": "raw/刑法.txt", "mutable": False},
            {"name": "sample_cases", "path": "raw/data/data/类案检索/train/train_set/candidates", "mutable": False},
        ],
        "scripts": [script_row(name) for name in SCRIPTS],
        "artifacts": [artifact_row(name) for name in ARTIFACTS],
        "quality": load_quality_status(),
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote manifest -> {OUTPUT}")


if __name__ == "__main__":
    main()
