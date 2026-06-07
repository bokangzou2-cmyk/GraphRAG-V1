# Router LoRA v2 Training

This dataset is Router-only and assumes the production flow:

1. User question
2. Query Rewriter LoRA
3. Deterministic Rewriter completion
4. Router LoRA decides retrieval policy

## Dataset Options

### Fast preview dataset

Generated dataset:

```powershell
python processing\build_router_lora_v2_dataset.py `
  --output-dir D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_rewriter_processed `
  --rewrite-mode rules `
  --min-count 5000
```

For strict LoRA materialization, use `--rewrite-mode lora`. It is much slower because it runs the local Rewriter adapter for every row.

Current output:

- total: 7435
- train: 6091
- validation: 664
- test: 680
- unique router inputs: 5140

Assistant answers contain only Router fields:

```json
{
  "label": "criminal_law_case_and_law",
  "needs_retrieval": true,
  "retrieval_targets": ["law_articles", "cases"],
  "case_required": true,
  "law_required": true,
  "clarify_required": false,
  "confidence": 0.88,
  "reason": "criminal_law_case_and_law"
}
```

## Train

```powershell
python processing\train_router_lora.py `
  --model D:\GraphRAG-V1\models\Qwen2.5-1.5B-Instruct `
  --dataset-dir D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_rewriter_processed `
  --output-dir D:\GraphRAG-V1\processing_data\router_lora\qwen2_5_1_5b_router_lora_v2_rewriter_processed `
  --gradient-checkpointing
```

The dataset uses GPT-authored scenario templates, GPT-5.4 subagent seed scenarios, and capped old Router rows reprocessed through the current Rewriter completion path.

### Strict LoRA-processed dataset

Raw GPT-authored shards:

```text
D:\GraphRAG-V1\processing_data\router_lora\gpt_authored_v2_shards
```

Materialize the final Router training set by running the Query Rewriter LoRA on every raw row:

```powershell
python processing\process_router_lora_v2_with_rewriter_lora.py `
  --gpt-shards-dir D:\GraphRAG-V1\processing_data\router_lora\gpt_authored_v2_shards `
  --output-dir D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_lora_processed `
  --old-cap-per-label 0 `
  --min-count 5000 `
  --continue-on-error
```

This command uses GPT-authored rows only. The current target is 5000 raw GPT rows. To mix in vetted old Router data, set `--old-cap-per-label` to a positive value such as `80`.

The script writes every completed row immediately to:

```text
D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_lora_processed\processed_rows.jsonl
```

If interrupted, run the same command again. Existing IDs in `processed_rows.jsonl` are skipped. Row-level failures are stored in `error_rows.jsonl`.

For clear non-retrieval rows, the script still calls the Rewriter LoRA but applies a sanity guard if the Rewriter introduces criminal case/crime terms that were not present in the raw question. Guarded rows are marked with `rewriter_guarded: true`.

Validate raw GPT shards without loading the LoRA:

```powershell
python processing\process_router_lora_v2_with_rewriter_lora.py `
  --gpt-shards-dir D:\GraphRAG-V1\processing_data\router_lora\gpt_authored_v2_shards `
  --old-cap-per-label 0 `
  --validate-raw-only
```

Smoke-test the LoRA processing path:

```powershell
python processing\process_router_lora_v2_with_rewriter_lora.py `
  --gpt-shards-dir D:\GraphRAG-V1\processing_data\router_lora\gpt_authored_v2_shards `
  --old-cap-per-label 0 `
  --output-dir D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_lora_processed_smoke `
  --limit 2 `
  --min-count 1 `
  --continue-on-error
```

To rebuild train/validation/test splits from already processed rows:

```powershell
python processing\process_router_lora_v2_with_rewriter_lora.py `
  --output-dir D:\GraphRAG-V1\processing_data\router_lora\dataset_v2_lora_processed `
  --finalize-only
```
