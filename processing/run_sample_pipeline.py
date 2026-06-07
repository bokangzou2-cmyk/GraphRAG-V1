from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEPS = [
    "parse_law_sample.py",
    "parse_cases_sample.py",
    "enrich_cases_sample.py",
    "extract_amounts_sample.py",
    "chunk_cases_sample.py",
    "chunk_law_sample.py",
    "build_retrieval_chunks_sample.py",
    "build_graph_sample.py",
    "validate_sample_outputs.py",
    "make_sample_report.py",
    "make_manifest_sample.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n== {step} ==")
        subprocess.run([sys.executable, str(HERE / step)], check=True)


if __name__ == "__main__":
    main()
