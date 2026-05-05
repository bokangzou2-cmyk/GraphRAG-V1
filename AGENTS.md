# GraphRAG-V1 Codex Rules

This project is rebuilding the data processing layer for a criminal-law GraphRAG system. The first priority is a reliable data foundation for the law library, case library, and JSONL graph snapshots.

## Collaboration Rules

- For future coding, inspection, and refactoring tasks, prefer using subagents when the task is non-trivial or can be split safely.
- When subagents are used, set the model to GPT-5.4 and use medium speed/reasoning settings.
- Before any code change, inspect the existing files and understand the current data layout.
- Prefer reusing reliable code from the original GraphRAG project when it is general, tested by usage, and not tightly coupled to the old demo flow.
- Rewrite unreliable, demo-oriented, hard-coded, or one-off code when it blocks maintainability or data correctness.
- After every modification, report the changed files, why they changed, and how the change was verified.
- Do not directly delete old code or data unless the reason is explicit and the replacement path is clear.

## Data Processing Rules

- Keep raw data immutable under `raw/`.
- Put processing code under `processing/`.
- Put generated data under `processing_data/`.
- Do not build indexes or graph databases until the data schemas and pipeline plan are approved.
- Prefer JSONL for large generated corpora and graph snapshots.
- Every generated record should include stable IDs, `source_file`, `text_hash`, extraction provenance, and quality flags where applicable.
- Distinguish original text extraction from rule-inferred fields.
- Preserve evidence text for graph edges whenever possible.

## Expected Data Outputs

- `processing_data/law_articles.jsonl`
- `processing_data/cases_raw.jsonl`
- `processing_data/cases_enriched.jsonl`
- `processing_data/graph_nodes.jsonl`
- `processing_data/graph_edges.jsonl`
- `processing_data/data_quality_report.json`
- `processing_data/manifest.json`

## Implementation Guardrails

- Keep scripts small and composable: parse, enrich, validate, build graph, and report should be separate steps.
- Use streaming reads for large JSON/JSONL/case folders.
- Avoid hard-coded demo sample patches.
- Avoid silently swallowing parse errors; record them in quality reports.
- Keep extraction confidence and method fields explicit.
- Make generated IDs deterministic so incremental rebuilds are possible.
