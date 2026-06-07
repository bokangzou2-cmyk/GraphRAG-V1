# Query Rewriter LoRA Training

目标：训练一个只负责问题改写和实体抽取的 LoRA，不再判断是否检索。

## Schema

模型只输出以下字段：

```json
{
  "standalone_query": "改写后的独立检索问题",
  "resolved_case_refs": [],
  "case_name_mentions": [],
  "crime_mentions": [],
  "amount_constraints": [],
  "field_intents": [],
  "rewrite_required": true,
  "confidence": 0.9,
  "warnings": []
}
```

禁止输出：

```text
route_label
needs_retrieval
retrieval_targets
case_required
law_required
```

## V1 Dataset

v1 是格式验证版，规模较小：

```powershell
cd D:\GraphRAG-V1
python processing\build_query_rewriter_dataset.py
```

输出：

```text
processing_data\query_rewriter_lora\final_dataset
```

规模：

```text
train: 2159
validation: 237
test: 237
total: 2633
```

## GPT-Authored V2 Dataset

GPT-authored v2 是建议训练版。它减少旧 Query Understanding 粗投影，以 GPT 直接生成样本为主体，重点强化：

- 双案例/多案例比较
- “入室盗窃/过失杀人/故意伤人/骗钱”等生活化表达归一
- 金额角色、字段意图和 standalone_query 规范化
- 只保留一部分旧 QU 中较适合改写的多轮样本

```powershell
cd D:\GraphRAG-V1
python processing\build_query_rewriter_gpt_dataset.py --old-cap 700
```

输出：

```text
processing_data\query_rewriter_lora\final_gpt_dataset_v2
```

当前 GPT-authored v2 规模：

```text
train: 2176
validation: 239
test: 239
total: 2654
direct_gpt_rows_final: 1954
selected_old_rows_final: 700
invalid_rows: 0
```

GPT shard 位置：

```text
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_001.jsonl
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_002.jsonl
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_003.jsonl
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_004.jsonl
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_005.jsonl
processing_data\query_rewriter_lora\gpt_authored_v2_shards\shard_006.jsonl
```

## Train V2

首次训练使用新输出目录，并关闭自动续训：

```powershell
cd D:\GraphRAG-V1

python processing\train_query_rewriter_lora.py `
  --dataset-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\final_gpt_dataset_v2 `
  --output-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v2 `
  --no-auto-resume
```

中断后续训：

```powershell
python processing\train_query_rewriter_lora.py `
  --dataset-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\final_gpt_dataset_v2 `
  --output-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v2
```

## Probe After Training

基础 probe：

```powershell
python processing\query_rewriter_probe.py `
  --adapter-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v2
```

可选：继续使用 v2 hidden eval：

```powershell
python processing\query_rewriter_probe.py `
  --adapter-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v2 `
  --eval-path D:\GraphRAG-V1\processing_data\query_rewriter_lora\rewriter_hidden_eval_v2_sample.json `
  --json-report D:\GraphRAG-V1\processing_data\query_rewriter_lora\rewriter_hidden_eval_v2_report.json `
  --text-report D:\GraphRAG-V1\processing_data\query_rewriter_lora\rewriter_hidden_eval_v2_report.txt
```

## Runtime Use

这个 LoRA 只适合放在 Router 之前：

```text
用户问题 + 历史
↓
Query Rewriter LoRA
↓
规则/Router LoRA 判断是否检索和检索范围
↓
Retrieval
↓
DeepSeek Answer
```

不要直接用它的输出决定是否检索。

## GPT-Authored V4 Conservative Dataset

v4 在 v3 基础上补充保守改写样本，目标是降低 Rewriter 过度改写风险：

- 原问题已经清楚时，`standalone_query` 尽量保持原文。
- 泛化刑法常识问题不补具体罪名、案名、金额、法条或检索关键词。
- 只有用户明确表达具体犯罪行为或罪名时，才抽取/规范化 `crime_mentions`。
- 显式约束如“不要查案例/别展开法条”必须保留。

构建数据集：

```powershell
cd D:\GraphRAG-V1

python processing\build_query_rewriter_v4_conservative_dataset.py
```

输出：

```text
D:\GraphRAG-V1\processing_data\query_rewriter_lora\final_gpt_dataset_v4_conservative
```

当前规模：

```text
total: 6262
train: 5134
validation: 564
test: 564
invalid_rows: 0
base_v3_rows: 4749
extra_v4_conservative_rows_after_dedupe: 1513
```

训练 v4：

```powershell
cd D:\GraphRAG-V1

python processing\train_query_rewriter_lora.py `
  --dataset-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\final_gpt_dataset_v4_conservative `
  --output-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v4_conservative_len768 `
  --max-length 768 `
  --eval-steps 1000 `
  --save-steps 500 `
  --no-auto-resume
```

中断后续训：

```powershell
python processing\train_query_rewriter_lora.py `
  --dataset-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\final_gpt_dataset_v4_conservative `
  --output-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v4_conservative_len768 `
  --max-length 768 `
  --eval-steps 1000 `
  --save-steps 500
```

训练完成后探针：

```powershell
python processing\query_rewriter_probe.py `
  --adapter-dir D:\GraphRAG-V1\processing_data\query_rewriter_lora\qwen2_5_1_5b_rewriter_lora_gpt_v4_conservative_len768 `
  --json-report D:\GraphRAG-V1\processing_data\query_rewriter_lora\rewriter_probe_gpt_v4_conservative_len768_report.json `
  --text-report D:\GraphRAG-V1\processing_data\query_rewriter_lora\rewriter_probe_gpt_v4_conservative_len768_report.txt
```
