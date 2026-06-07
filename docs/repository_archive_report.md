# Repository Archive Report

## Scope

This archive pass organizes the current GraphRAG-V1 workspace into a versioned baseline for the criminal-law GraphRAG sample project.

The committed scope includes:

- Backend FastAPI application code under `app/`.
- Frontend React/Vite application code under `frontend/`.
- Processing, retrieval, evaluation, LoRA dataset/training scripts, and orchestration scripts under `processing/`.
- Tests under `tests/`.
- Project documentation and reports under `docs/` and root-level project reports.
- Core generated sample data, graph outputs, vector index, and evaluation reports under `processing_data/`.
- Runtime configuration template `.env.example`.
- Python project metadata `pyproject.toml`.

## Preserved But Ignored Local Artifacts

These paths are intentionally preserved locally and ignored by Git because they are private, large, or machine-specific:

- `.env`
- `raw/`
- `models/`
- `frontend/node_modules/`
- `processing_data/chat_sessions.sqlite3`
- `processing_data/query_rewriter_lora/`
- `processing_data/query_understanding_lora/`
- `processing_data/router_lora/`
- `processing_data/query_router_lora/`

See `docs/artifact_inventory.md` for artifact references and usage notes.

## Deleted As Cache Or Runtime Noise

The following categories were removed because they are generated and safe to rebuild:

- Python `__pycache__/` directories.
- `.pytest_cache/`.
- `criminal_law_graphrag.egg-info/`.
- `frontend/dist/`.
- Runtime logs under `processing_data/*.log`.

No dataset, vector index, graph output, evaluation result, report, model directory, raw data directory, or LoRA artifact directory was deleted.

## Gitignore Changes

`.gitignore` was changed to:

- Ignore Python caches, local build metadata, runtime logs, local SQLite chat history, models, raw source data, frontend dependencies/build output, and large LoRA artifact directories.
- Stop ignoring all of `processing_data/` by default so sample data, graph outputs, vector index, and evaluation reports can be versioned.

## Archive Notes

Large LoRA training outputs are not committed. Their runtime paths and current active adapter references are preserved in `.env.example`, `processing/*TRAINING.md`, and `docs/artifact_inventory.md`.
