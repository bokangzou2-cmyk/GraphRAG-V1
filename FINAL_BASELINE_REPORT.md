# Final Baseline Report

Original baseline generated at: 2026-05-29T19:11:03
Repository archive synchronized at: 2026-06-07

## Git State

The baseline has been archived in Git and pushed to GitHub.

- Archive commit: `2f32dc3 Archive LangGraph GraphRAG QA baseline`
- Remote: `https://github.com/bokangzou2-cmyk/GraphRAG-V1`
- Current frontend/backend runtime: LangGraph orchestration path.
- `/api/chat/orchestration` is the primary frontend endpoint.
- `/api/chat` is a compatibility endpoint delegating to the same orchestration implementation.

## Schema And Pipeline

- Schema documentation: `docs/schema.md`
- Pipeline entry: `processing/run_sample_pipeline.py`
- Data quality status: `PASS`

## Frozen Metrics

| Metric | Value |
| --- | ---: |
| law articles | 490 |
| cases | 150 |
| case chunks | 4080 |
| law chunks | 490 |
| retrieval chunks | 4570 |
| avg case chunk length | 425.9 |
| short_chunk flags | 104 |
| graph nodes | 5476 |
| graph edges | 7164 |
| unresolved_applied_law_edges | 5 |
| unknown_law_edges | 10 |

## Eval Baseline

| Eval | Result |
| --- | --- |
| lexical eval | PASS (12/12, warn/fail=0) |
| vector eval | PASS (6/6, warn/fail=0) |
| graph eval | PASS (5/5, warn/fail=0) |
| hybrid eval | PASS (10/10, warn/fail=0) |
| answer eval | PASS (36/36, warn/fail=0) |
| adversarial eval | PASS (120/120, warn/fail=0) |
| orchestration mock eval | PASS (9/9, warn/fail=0) |
| backend tests | Historical baseline: PASS (26 passed, 3 warnings) |
| frontend tests | Historical baseline: PASS (3 passed) |

Recent lightweight archive verification:

- `pytest tests\test_langgraph_orchestration.py tests\test_query_rewriter.py -q`: PASS (4 passed)
- `npm test --prefix frontend`: PASS (5 passed)
- `python -m compileall app processing tests`: PASS

## Freeze Note

This report records the sample-level baseline and the later repository archive state. Hidden eval generation and reporting are validation artifacts only; do not rewrite hidden/adversarial result files just to update project status text.
