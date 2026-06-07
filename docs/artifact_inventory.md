# Artifact Inventory

This project keeps large runtime and training artifacts on the local machine instead of committing them to Git. Do not delete these paths unless the replacement path is explicit and verified.

## Local Models

- `D:\GraphRAG-V1\models`
  - Local base models used by Rewriter/Router LoRA training and runtime inference.
  - Ignored by Git because model weights are large and reproducible from the configured model source.

## Raw Data

- `D:\GraphRAG-V1\raw`
  - Immutable source data.
  - Ignored by Git in this workspace; processing outputs and reports under `processing_data` are tracked where practical.

## LoRA Training Artifacts

- `D:\GraphRAG-V1\processing_data\query_rewriter_lora`
  - Query Rewriter datasets, shards, reports, checkpoints, and adapter outputs.
  - Current runtime adapter reference: `qwen2_5_1_5b_rewriter_lora_gpt_v4_conservative_len768`.

- `D:\GraphRAG-V1\processing_data\router_lora`
  - Router datasets, processed datasets, reports, checkpoints, and adapter outputs.

- `D:\GraphRAG-V1\processing_data\query_understanding_lora`
  - Historical unified Query Understanding LoRA datasets and outputs.
  - Not the active frontend runtime path.

- `D:\GraphRAG-V1\processing_data\query_router_lora`
  - Earlier router-related generated data.

These directories are ignored because they can contain multi-GB checkpoints and adapter artifacts. Keep manifest, README, and training command files in `processing/` and `docs/` under version control.

## Tracked Generated Data

The following generated artifacts are intended to stay versioned when present:

- `processing_data/*_sample.json`
- `processing_data/*_sample.jsonl`
- `processing_data/*_report*.json`
- `processing_data/*_report*.txt`
- `processing_data/graph_*_sample.jsonl`
- `processing_data/retrieval_chunks_sample.jsonl`
- `processing_data/vector_index_sample`
- `processing_data/manifest_sample.json`

Runtime-only local state such as `processing_data/chat_sessions.sqlite3` and `processing_data/*.log` remains ignored.
