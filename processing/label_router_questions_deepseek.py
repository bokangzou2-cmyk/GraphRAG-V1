from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.minimax_client import MinimaxClient
from common import OUT_DIR, iter_jsonl


QUESTION_DIR = OUT_DIR / "router_training_questions"
INPUT = QUESTION_DIR / "all_questions_unlabeled.jsonl"
LABELED_OUT = QUESTION_DIR / "router_questions_labeled_deepseek.jsonl"
FAILED_OUT = QUESTION_DIR / "router_questions_label_failures.jsonl"
PROGRESS_OUT = QUESTION_DIR / "labeling_progress.json"
LOCK_PATH = QUESTION_DIR / "labeling.lock"

VALID_LABELS = {
    "general_no_legal_retrieval",
    "criminal_law_law_only",
    "criminal_law_case_only",
    "criminal_law_case_and_law",
    "criminal_law_multi_case",
    "criminal_law_advice",
    "non_criminal_legal",
    "unclear_need_clarification",
    "low_quality_or_incomplete",
}
VALID_TARGETS = {"none", "law_articles", "cases", "graph"}

SYSTEM_PROMPT = """你是中文问题路由数据标注员。你的任务是判断一个用户问题是否属于刑法相关问题，以及后续回答应检索什么资料。

只输出一个 JSON 对象，不要输出解释性文字。JSON schema:
{
  "label": string,
  "domain": "general|criminal_law|non_criminal_legal|unclear|low_quality",
  "needs_legal_retrieval": boolean,
  "retrieval_targets": string[],
  "case_required": boolean,
  "law_required": boolean,
  "clarify_required": boolean,
  "quality": "ok|noisy|incomplete",
  "confidence": number,
  "reason": string
}

label 只能是:
- general_no_legal_retrieval: 普通聊天、百科、生活、技术、医疗、财经、写作等，不需要法律检索。
- criminal_law_law_only: 刑法条文、罪名、构成要件、量刑规则、刑法概念；没有指定具体案件。
- criminal_law_case_only: 问具体刑事案件的事实、判决、证据、辩解等；主要需要案例。
- criminal_law_case_and_law: 问具体刑事案件为什么这么判、适用哪些刑法、裁判依据等；需要案例和法条。
- criminal_law_multi_case: 类案、相似案例、多个刑事案例比较。
- criminal_law_advice: 现实个人刑事风险、是否会坐牢、取保候审、报警、判多久等咨询。
- non_criminal_legal: 非刑法法律问题，如民法、合同、婚姻、劳动、行政、公司、知识产权等。
- unclear_need_clarification: 问题太短或指代不明，但不是明显低质量。
- low_quality_or_incomplete: 乱码、拼接错乱、不完整、无法可靠判断。

retrieval_targets 只能包含:
- "none"
- "law_articles"
- "cases"
- "graph"

规则:
1. 刑法相关只包括犯罪、刑罚、刑事责任、刑事诉讼强相关、具体刑事案件、罪名量刑。
2. 其他法律问题不算刑法问题，标为 non_criminal_legal。
3. 普通问题可以由普通 LLM 直接回答，标为 general_no_legal_retrieval，retrieval_targets 为 ["none"]。
4. “盗窃罪怎么判”这类只问罪名规则，标 criminal_law_law_only，不需要案例。
5. “吴必定案判了多久”这类具体案件，标 criminal_law_case_only。
6. “某案适用了哪些刑法规定/为什么这么判”，标 criminal_law_case_and_law。
7. 输出必须是合法 JSON。"""


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if row_id:
                done.add(str(row_id))
    return done


def acquire_lock(force: bool = False) -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force and LOCK_PATH.exists():
        LOCK_PATH.unlink()
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        message = LOCK_PATH.read_text(encoding="utf-8", errors="ignore") if LOCK_PATH.exists() else ""
        raise SystemExit(f"labeling lock exists: {LOCK_PATH}\n{message}\nIf no labeling process is running, rerun with --force-lock.") from exc
    os.write(fd, f"pid={os.getpid()} started_at_utc={datetime.now(timezone.utc).isoformat()}\n".encode("utf-8"))
    return fd


def release_lock(fd: int) -> None:
    os.close(fd)
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("response_not_json")
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response_not_object")
    return data


def normalize_label(data: dict) -> dict:
    label = str(data.get("label") or "").strip()
    if label not in VALID_LABELS:
        raise ValueError(f"invalid_label:{label}")
    targets = data.get("retrieval_targets")
    if not isinstance(targets, list):
        raise ValueError("retrieval_targets_not_list")
    targets = [str(item).strip() for item in targets]
    if not targets:
        targets = ["none"]
    invalid_targets = [item for item in targets if item not in VALID_TARGETS]
    if invalid_targets:
        raise ValueError(f"invalid_targets:{invalid_targets}")
    if label == "general_no_legal_retrieval":
        targets = ["none"]
    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "label": label,
        "domain": str(data.get("domain") or ""),
        "needs_legal_retrieval": bool(data.get("needs_legal_retrieval")),
        "retrieval_targets": targets,
        "case_required": bool(data.get("case_required")),
        "law_required": bool(data.get("law_required")),
        "clarify_required": bool(data.get("clarify_required")),
        "quality": str(data.get("quality") or "ok"),
        "confidence": round(confidence, 3),
        "reason": str(data.get("reason") or "")[:300],
    }


def build_messages(row: dict) -> list[dict]:
    payload = {
        "id": row.get("id"),
        "question": row.get("question"),
        "source_split": row.get("split"),
        "source": row.get("source"),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


async def label_one(client: MinimaxClient, row: dict, retries: int) -> dict:
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            raw = await client.chat(build_messages(row), temperature=0.0, max_tokens=500)
            label = normalize_label(parse_json_object(raw))
            return {
                **row,
                "label_payload": label,
                "label_model": client.model,
                "label_provider": client.provider,
                "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                await asyncio.sleep(min(2 * attempt, 8))
    raise RuntimeError(str(last_error))


def write_progress(total: int, completed: int, failed: int, label_counts: Counter, source_counts: Counter) -> None:
    progress = {
        "schema_version": "router-labeling-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT),
        "output": str(LABELED_OUT),
        "failures": str(FAILED_OUT),
        "total": total,
        "completed": completed,
        "failed": failed,
        "remaining": max(0, total - completed),
        "label_counts": dict(label_counts),
        "source_counts_completed": dict(source_counts),
    }
    PROGRESS_OUT.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(INPUT))
    if args.limit is not None:
        rows = rows[: args.limit]
    client = MinimaxClient()
    if not client.enabled():
        raise SystemExit("LLM_API_KEY is not configured")
    done = load_done_ids(LABELED_OUT)
    if args.retry_failures and FAILED_OUT.exists():
        failed_ids = load_done_ids(FAILED_OUT)
        done -= failed_ids
    pending = [row for row in rows if str(row.get("id")) not in done]
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    if LABELED_OUT.exists():
        for row in iter_jsonl(LABELED_OUT):
            label = ((row.get("label_payload") or {}).get("label"))
            if label:
                label_counts[label] += 1
                source_counts[row.get("source", "")] += 1
    completed = len(done)
    failed = 0
    print(f"provider={client.provider} model={client.model}")
    print(f"total={len(rows)} already_done={len(done)} pending={len(pending)}")
    lock = asyncio.Lock()

    async def process_row(row: dict) -> None:
        nonlocal completed, failed
        row_id = str(row.get("id"))
        try:
            labeled = await label_one(client, row, retries=args.retries)
            async with lock:
                append_jsonl(LABELED_OUT, labeled)
                label = labeled["label_payload"]["label"]
                label_counts[label] += 1
                source_counts[labeled.get("source", "")] += 1
                completed += 1
                if completed % args.progress_every == 0 or completed == len(done) + 1:
                    write_progress(len(rows), completed, failed, label_counts, source_counts)
                    print(f"completed={completed}/{len(rows)} label={label} id={row_id}")
        except Exception as exc:
            failure = {
                **row,
                "error": str(exc)[:500],
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            async with lock:
                failed += 1
                append_jsonl(FAILED_OUT, failure)
                write_progress(len(rows), completed, failed, label_counts, source_counts)
                print(f"failed id={row_id} error={type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

    for start in range(0, len(pending), args.concurrency):
        batch = pending[start : start + args.concurrency]
        await asyncio.gather(*(process_row(row) for row in batch))
    write_progress(len(rows), completed, failed, label_counts, source_counts)
    print(f"done completed={completed} failed_this_run={failed} remaining={max(0, len(rows)-completed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label router training questions with DeepSeek/OpenAI-compatible LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Only label the first N input rows.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    fd = acquire_lock(force=args.force_lock)
    try:
        asyncio.run(main_async(args))
    finally:
        release_lock(fd)


if __name__ == "__main__":
    main()
