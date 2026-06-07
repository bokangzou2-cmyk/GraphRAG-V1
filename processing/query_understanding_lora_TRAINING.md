# Query Understanding LoRA Training

This project uses a Query Understanding LoRA to decide whether a user query needs GraphRAG retrieval, how to rewrite multi-turn questions, and which retrieval targets to use.

## Build Dataset

After GPT-authored shards are generated under:

```powershell
processing_data\query_understanding_lora\gpt_direct_general_dataset
```

merge and validate them:

```powershell
python processing\merge_gpt_authored_dataset.py `
  --dataset-dir processing_data\query_understanding_lora\gpt_direct_general_dataset `
  --target-count 300
```

Then rebuild the final training dataset:

```powershell
python processing\build_query_understanding_final_dataset.py `
  --direct-general-dataset processing_data\query_understanding_lora\gpt_direct_general_dataset `
  --output-dir processing_data\query_understanding_lora\final_5000_dataset `
  --target-count 5000
```

## Train

Default training resumes automatically from the newest checkpoint in the output directory:

```powershell
python processing\train_query_understanding_lora.py
```

Useful validation-speed settings:

```powershell
python processing\train_query_understanding_lora.py `
  --epochs 1 `
  --max-length 768 `
  --train-batch-size 2 `
  --gradient-accumulation-steps 4 `
  --save-steps 100 `
  --eval-steps 100 `
  --save-total-limit 3 `
  --auto-resume
```

To restart from scratch, use a new output directory or pass:

```powershell
python processing\train_query_understanding_lora.py --no-auto-resume
```

## Verify

Run the probe after training:

```powershell
python processing\query_understanding_probe.py `
  --mode lora `
  --adapter-dir processing_data\query_understanding_lora\qwen2_5_1_5b_qu_lora_fast
```

The important regression categories are:

- `criminal_law_general_direct`:刑法常识直答，不走样本库检索。
- `law_lookup`:罪名量刑、具体法条查询，走 law_articles。
- `history_case_reference`:多轮追问案名消解。
- `amount_case_law`:金额和类案检索。
- `guardrail`:违法规避和拒答边界。
