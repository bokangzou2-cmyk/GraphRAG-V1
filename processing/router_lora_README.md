# Router LoRA Training

Status: legacy first-version training flow.

The active frontend runtime uses the Router LoRA after the Query Rewriter LoRA and deterministic rewrite completion. For the current training recipe and processed 5000-row target dataset, prefer `processing/router_lora_v2_TRAINING.md`.

This document is preserved because the first-version router dataset and commands are still useful for comparison and recovery, but it is not the primary current workflow.

## 1. Install training dependencies

```powershell
pip install -e ".[router-train]"
```

## 2. Prepare the SFT dataset

```powershell
python processing\prepare_router_lora_dataset.py
```

Outputs:

- `processing_data\router_lora\dataset\train.jsonl`
- `processing_data\router_lora\dataset\validation.jsonl`
- `processing_data\router_lora\dataset\test.jsonl`
- `processing_data\router_lora\dataset\manifest.json`

## 3. Train Qwen2.5-1.5B-Instruct with LoRA

If direct Hugging Face download is unstable on Windows, download the base model first:

```powershell
python processing\download_qwen_model.py
```

Then pass the local model directory to the trainer:

```powershell
python processing\train_router_lora.py --model D:\GraphRAG-V1\models\Qwen2.5-1.5B-Instruct --gradient-checkpointing
```

GPU training:

```powershell
python processing\train_router_lora.py --gradient-checkpointing
```

Lower-memory GPU training, if bitsandbytes works in the environment:

```powershell
python processing\train_router_lora.py --use-4bit --gradient-checkpointing
```

Output adapter:

- `processing_data\router_lora\qwen2_5_1_5b_router_lora_v2_lora_processed`

## 4. Evaluate the adapter

```powershell
python processing\eval_router_lora.py
```

Output report:

- `processing_data\router_lora\router_lora_eval_report.json`

The target output is strict JSON with these fields:

```json
{
  "label": "criminal_law_law_only",
  "needs_retrieval": true,
  "retrieval_targets": ["law_articles"],
  "case_required": false,
  "law_required": true,
  "clarify_required": false
}
```

## 5. Enable the router in the app

Add these values to `D:\GraphRAG-V1\.env` to use the trained LoRA router in the current LangChain orchestration runtime:

```powershell
ROUTER_MODE=lora
ROUTER_BASE_MODEL=D:\GraphRAG-V1\models\Qwen2.5-1.5B-Instruct
ROUTER_ADAPTER_DIR=D:\GraphRAG-V1\processing_data\router_lora\qwen2_5_1_5b_router_lora_v2_lora_processed
ROUTER_MAX_NEW_TOKENS=128
```

`ROUTER_MODE=rules` is still available for debugging when the local LoRA model cannot be loaded, but it is not the frontend runtime default.
