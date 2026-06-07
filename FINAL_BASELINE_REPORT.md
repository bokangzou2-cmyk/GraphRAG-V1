# Final Baseline Report

Generated at: 2026-05-29T19:11:03

## Git State

Working tree is not clean. Existing modified/untracked files were present before this QA freeze; no source logic changes are made during hidden validation.

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
| backend tests | PASS (26 passed, 3 warnings) |
| frontend tests | PASS (3 passed) |

## Freeze Note

This report freezes the sample-level baseline before hidden anti-overfitting validation. Hidden eval generation and reporting are validation artifacts only.