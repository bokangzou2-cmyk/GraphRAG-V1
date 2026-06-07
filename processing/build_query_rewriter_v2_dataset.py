from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import OUT_DIR
from query_rewriter_common import build_rewriter_messages, normalize_rewriter_answer, validate_rewriter_answer


DEFAULT_QU_DATASET = OUT_DIR / "query_understanding_lora" / "final_6500_v3_dataset"
DEFAULT_GPT_SEED = OUT_DIR / "query_rewriter_lora" / "gpt_authored_seed.jsonl"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_rewriter_lora" / "final_dataset_v2"
DEFAULT_HIDDEN_EVAL = OUT_DIR / "query_rewriter_lora" / "rewriter_hidden_eval_v2_sample.json"
SCHEMA_VERSION = "query-rewriter-lora-dataset-v2"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}:invalid_json:{exc}") from exc
    return rows


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def answer(
    standalone_query: str,
    *,
    resolved_case_refs: list[dict[str, Any]] | None = None,
    case_name_mentions: list[str] | None = None,
    crime_mentions: list[str] | None = None,
    amount_constraints: list[dict[str, Any]] | None = None,
    field_intents: list[str] | None = None,
    rewrite_required: bool = True,
    confidence: float = 0.9,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return normalize_rewriter_answer(
        {
            "standalone_query": standalone_query,
            "resolved_case_refs": resolved_case_refs or [],
            "case_name_mentions": case_name_mentions or [],
            "crime_mentions": crime_mentions or [],
            "amount_constraints": amount_constraints or [],
            "field_intents": field_intents or [],
            "rewrite_required": rewrite_required,
            "confidence": confidence,
            "warnings": warnings or [],
        }
    )


def row(row_id: str, category: str, source_messages: list[dict[str, str]], ans: dict[str, Any], source_method: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "split": "train",
        "category": category,
        "source_messages": source_messages,
        "answer": ans,
        "messages": build_rewriter_messages(source_messages, ans),
        "source": "query_rewriter_v2_curated_gpt_style",
        "source_method": source_method,
        "text_hash": stable_hash(source_messages),
    }


CASE_BANK = [
    ("王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书", "王士东案", "盗窃罪", 43852, "入户盗窃、多次盗窃、累犯"),
    ("刘方祥、刘礼芬盗窃一审刑事判决书", "刘方祥案", "盗窃罪", 39136, "盗窃数额、退赃、认罪态度"),
    ("周某入户盗窃罪一审刑事判决书", "周某入户盗窃案", "盗窃罪", 3000, "入户盗窃、数额较大"),
    ("吴必定抢劫案", "吴必定案", "抢劫罪", None, "暴力劫取财物、量刑情节"),
    ("张君抢劫、故意杀人案", "张君案", "抢劫罪", None, "抢劫、故意杀人、多起犯罪"),
    ("李某故意伤害罪二审刑事裁定书", "李某故意伤害案", "故意伤害罪", None, "伤情鉴定、赔偿谅解"),
    ("陈某诈骗罪一审刑事判决书", "陈某诈骗案", "诈骗罪", 40000, "虚构事实、非法占有目的"),
    ("何某合同诈骗罪一审刑事判决书", "何某合同诈骗案", "合同诈骗罪", 80000, "合同履行、非法占有目的"),
    ("孙某危险驾驶罪一审刑事判决书", "孙某危险驾驶案", "危险驾驶罪", None, "醉酒驾驶、血醇含量"),
    ("赵某寻衅滋事罪一审刑事判决书", "赵某寻衅滋事案", "寻衅滋事罪", None, "随意殴打、公共秩序"),
]

FIELD_FOLLOWUPS = [
    ("第一个参考案例具体情况是什么", 1, ["court_found_facts_text"], "案情 法院认定事实"),
    ("第二个案例案情和判决结果呢", 2, ["court_found_facts_text", "judgment_text"], "案情 判决结果 量刑"),
    ("刚才那个为什么判这么重", 1, ["reasoning_text", "judgment_text"], "裁判理由 量刑 从重情节"),
    ("第一个案子法院依据哪些证据认定的", 1, ["evidence_text", "court_found_facts_text"], "证据 法院认定事实"),
    ("这个案子被告人怎么辩护的", 1, ["defense_text"], "辩护意见 辩解"),
    ("前面第二个适用了哪些法条", 2, ["legal_basis_text"], "适用法条 法律依据"),
    ("第一个案例有没有自首或者退赃情节", 1, ["reasoning_text", "judgment_text"], "自首 退赃 量刑情节"),
    ("第二个案例法院为什么采信这些证据", 2, ["evidence_text", "reasoning_text"], "证据采信 裁判理由"),
]

COMPARE_FOLLOWUPS = [
    ("前面两个案例量刑差异在哪里", "量刑差异 对比 裁判理由", ["reasoning_text", "judgment_text"]),
    ("这两个案子哪个更接近我朋友的情况", "相似案例 对比 案情 量刑", ["court_found_facts_text", "judgment_text"]),
    ("第一个和第二个案例的主要区别是什么", "案例比较 案情差异 量刑差异", ["court_found_facts_text", "reasoning_text", "judgment_text"]),
    ("有没有类似但判得轻一点的案例", "相似案例 轻判 量刑情节", ["reasoning_text", "judgment_text"]),
    ("这两个案例为什么一个判得更重", "量刑差异 从重从轻情节 裁判理由", ["reasoning_text", "judgment_text"]),
]

ALIASES = [
    ("入室盗窃", "入户盗窃 盗窃罪", "盗窃罪", "theft_amount_total_or_single"),
    ("入室偷东西", "入户盗窃 盗窃罪", "盗窃罪", "theft_amount_total_or_single"),
    ("偷了别人手机", "盗窃罪", "盗窃罪", "theft_amount_total_or_single"),
    ("过失杀人", "过失致人死亡罪", "过失致人死亡罪", "unknown_amount"),
    ("过失让人死亡", "过失致人死亡罪", "过失致人死亡罪", "unknown_amount"),
    ("故意伤人", "故意伤害罪", "故意伤害罪", "unknown_amount"),
    ("把人打成重伤", "故意伤害罪 重伤", "故意伤害罪", "unknown_amount"),
    ("骗钱", "诈骗罪", "诈骗罪", "fraud_amount_total_or_single"),
    ("虚假投资骗钱", "诈骗罪 非法占有目的", "诈骗罪", "fraud_amount_total_or_single"),
    ("持刀抢东西", "抢劫罪", "抢劫罪", "unknown_amount"),
    ("酒驾撞人", "危险驾驶罪 交通肇事风险", "危险驾驶罪", "unknown_amount"),
]


def case_history(case_a: tuple, case_b: tuple, user_prefix: str = "我想找类似案例") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"{user_prefix}，最好有案情、证据和量刑参考。"},
        {
            "role": "assistant",
            "content": (
                f"可以先看两个参考案例。第一个参考案例是《{case_a[0]}》，简称{case_a[1]}，涉及{case_a[2]}"
                f"{'，金额约' + str(case_a[3]) + '元' if case_a[3] else ''}，关注{case_a[4]}。"
                f"第二个参考案例是《{case_b[0]}》，简称{case_b[1]}，涉及{case_b[2]}"
                f"{'，金额约' + str(case_b[3]) + '元' if case_b[3] else ''}，关注{case_b[4]}。"
            ),
        },
    ]


def amount_obj(value: int | None, role: str, raw: str | None = None) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [{"value": value, "role": role, "raw_text": raw or f"{value}元"}]


def generate_multiturn(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    user_prefixes = [
        "我想找类似案例",
        "我朋友的情况想对照几个案例",
        "先帮我看两个参考案例",
        "我在比较这类刑事案件",
        "我想知道哪个案例更接近现实情况",
        "我需要按案情、证据和量刑来比较",
    ]
    focus_suffixes = [
        "",
        "，重点看金额和量刑",
        "，重点看证据和法院认定",
        "，重点看从重从轻情节",
        "，重点看适用法条",
        "，重点看判决结果",
    ]
    for idx in range(count):
        case_a = rng.choice(CASE_BANK)
        case_b = rng.choice([item for item in CASE_BANK if item != case_a])
        if idx % 3 == 0:
            query, suffix, fields = rng.choice(COMPARE_FOLLOWUPS)
            selected = [case_a, case_b]
            refs = [
                {"ref": "前面两个案例", "index": 1, "case_title": case_a[0], "short_title": case_a[1], "source": "assistant_history"},
                {"ref": "前面两个案例", "index": 2, "case_title": case_b[0], "short_title": case_b[1], "source": "assistant_history"},
            ]
        else:
            query, ordinal, fields, suffix = rng.choice(FIELD_FOLLOWUPS)
            selected = [case_a if ordinal == 1 else case_b]
            refs = [
                {
                    "ref": query,
                    "index": ordinal,
                    "case_title": selected[0][0],
                    "short_title": selected[0][1],
                    "source": "assistant_history",
                }
            ]
        focus = rng.choice(focus_suffixes)
        messages = case_history(case_a, case_b, rng.choice(user_prefixes)) + [{"role": "user", "content": f"{query}{focus}"}]
        amounts = []
        for item in selected:
            amounts.extend(amount_obj(item[3], "theft_amount_total_or_single" if item[2] == "盗窃罪" else "unknown_amount"))
        ans = answer(
            " ".join([item[0] for item in selected] + [suffix, focus.replace("，", "")]),
            resolved_case_refs=refs,
            case_name_mentions=[name for item in selected for name in (item[0], item[1])],
            crime_mentions=sorted({item[2] for item in selected}),
            amount_constraints=amounts,
            field_intents=fields,
        )
        rows.append(row(f"rwv2_multi_{idx:06d}", "multi_turn_case_reference" if len(selected) == 1 else "multi_case_compare", messages, ans, "gpt_style_rewriter_multiturn_v2"))
    return rows


def generate_alias_and_amount(count: int, rng: random.Random) -> list[dict[str, Any]]:
    templates = [
        "{alias}{amount}一般怎么判，有案例参考吗",
        "我朋友{alias}了{amount}，大概会怎么处理",
        "{alias}{amount}属于什么罪，量刑看什么",
        "{alias}{amount}有没有类似但判得轻一点的案例",
        "{alias}{amount}对应什么规范罪名和检索关键词",
        "生活里说的{alias}{amount}，法律上应该怎么检索",
        "如果是{alias}{amount}，要查哪些案情和量刑因素",
        "{alias}{amount}和规范罪名怎么对应",
        "{alias}{amount}需要关注数额还是行为方式",
        "请把{alias}{amount}改成适合检索的刑法问题",
    ]
    prefixes = ["", "请问，", "通俗说，", "按刑法检索，", "如果只看样本案例，", "从量刑角度，"]
    tails = ["", "，重点看法条", "，重点看案例", "，重点看金额角色", "，重点看是否入户", "，重点看从轻情节"]
    values = [None, 3000, 4000, 12000, 40000, 43852, 80000, 120000]
    rows = []
    for idx in range(count):
        alias, normalized, crime, role = rng.choice(ALIASES)
        value = rng.choice(values)
        amount_text = "" if value is None else f"{value}元"
        query = f"{rng.choice(prefixes)}{rng.choice(templates).format(alias=alias, amount=amount_text)}{rng.choice(tails)}"
        case_phrase = " 相似案例 量刑" if "案例" in query else " 法律依据 量刑"
        extra = " 入户盗窃" if "入室" in alias else ""
        standalone = f"{normalized} {amount_text}{extra} {case_phrase}".strip()
        fields = ["legal_basis_text", "judgment_text"]
        if "案例" in query or "轻" in query:
            fields.append("reasoning_text")
        ans = answer(
            standalone,
            crime_mentions=[crime],
            amount_constraints=amount_obj(value, role),
            field_intents=fields,
            rewrite_required=alias not in standalone or bool(value),
        )
        rows.append(row(f"rwv2_alias_{idx:06d}", "colloquial_crime_alias" if value is None else "amount_constraint", [{"role": "user", "content": query}], ans, "gpt_style_rewriter_alias_amount_v2"))
    return rows


def generate_field_intents(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    field_templates = [
        ("{short}法院认定的事实是什么", "{short} 案情 法院认定事实", ["court_found_facts_text"]),
        ("{short}判决结果是什么", "{short} 判决结果 量刑", ["judgment_text"]),
        ("{short}有哪些证据", "{short} 证据 法院采信", ["evidence_text"]),
        ("{short}裁判理由是什么", "{short} 裁判理由 量刑", ["reasoning_text"]),
        ("{short}被告人怎么辩护", "{short} 辩护意见 辩解", ["defense_text"]),
        ("{crime}对应哪条法条", "{crime} 适用法条 法律依据", ["legal_basis_text"]),
        ("{crime}量刑时通常看哪些事实", "{crime} 量刑 关键事实 裁判理由", ["court_found_facts_text", "reasoning_text", "judgment_text"]),
        ("{short}法院为什么这么认定", "{short} 法院认定 裁判理由", ["reasoning_text", "court_found_facts_text"]),
        ("{short}证据和判决结果一起看", "{short} 证据 判决结果 量刑", ["evidence_text", "judgment_text"]),
        ("{crime}案件一般有哪些辩护点", "{crime} 辩护意见 争议焦点", ["defense_text"]),
    ]
    prefixes = ["", "请问", "帮我查一下", "我想知道", "检索时重点看"]
    suffixes = ["", "？", "，用于案例检索", "，不要泛泛说", "，要能查到对应字段"]
    for idx in range(count):
        title, short, crime, _amount, _note = rng.choice(CASE_BANK)
        query_template, standalone_template, fields = rng.choice(field_templates)
        query = f"{rng.choice(prefixes)}{query_template.format(short=short, crime=crime)}{rng.choice(suffixes)}"
        standalone = standalone_template.format(short=short, crime=crime)
        case_mentions = [short] if "{short}" in standalone_template else []
        crimes = [crime]
        ans = answer(
            standalone,
            case_name_mentions=case_mentions,
            crime_mentions=crimes,
            field_intents=fields,
            rewrite_required=query != standalone,
            confidence=0.88,
        )
        rows.append(row(f"rwv2_field_{idx:06d}", "field_intent_single_turn", [{"role": "user", "content": query}], ans, "gpt_style_rewriter_field_intent_v2"))
    return rows


def project_good_old_rows(qu_dataset: Path, cap: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    keep_categories = {"multi_turn_rewrite", "history_case_reference", "multi_case_compare", "short_case_name", "amount_case_law"}
    candidates = []
    for split in ["train", "validation", "test"]:
        for old in read_jsonl(qu_dataset / f"{split}.jsonl"):
            if old.get("category") not in keep_categories:
                continue
            old_answer = old.get("answer") or {}
            if not old_answer.get("resolved_case_refs") and old.get("category") == "multi_turn_rewrite":
                continue
            ans = answer(
                old_answer.get("standalone_query") or latest_user_query(old.get("source_messages") or []),
                resolved_case_refs=old_answer.get("resolved_case_refs") or [],
                case_name_mentions=old_answer.get("case_name_mentions") or [],
                crime_mentions=old_answer.get("crime_mentions") or [],
                amount_constraints=old_answer.get("amount_constraints") or [],
                field_intents=old_answer.get("field_intents") or [],
                rewrite_required=True,
                confidence=old_answer.get("confidence", 0.86),
                warnings=old_answer.get("warnings") or [],
            )
            if validate_rewriter_answer(ans):
                continue
            candidates.append(row(f"rwv2_old_{len(candidates):06d}", f"old_good_{old.get('category')}", old.get("source_messages") or [], ans, "selected_old_qu_rewriter_projection_v2"))
    rng.shuffle(candidates)
    rows.extend(candidates[:cap])
    return rows


def load_seed_rows(seed_path: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, old in enumerate(read_jsonl(seed_path)):
        ans = normalize_rewriter_answer(old.get("answer") or {})
        if validate_rewriter_answer(ans):
            continue
        rows.append(row(f"rwv2_seed_{idx:06d}", old.get("category") or "gpt_seed", old.get("source_messages") or [], ans, "gpt_5_5_rewriter_seed_v2"))
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in rows:
        key = stable_hash({"source_messages": item.get("source_messages"), "answer": item.get("answer")})
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def assign_splits(rows: list[dict[str, Any]], seed: int, target_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = dedupe(rows)
    rng.shuffle(rows)
    selected = rows[:target_count]
    train_cut = int(len(selected) * 0.82)
    val_cut = int(len(selected) * 0.91)
    final = []
    for idx, item in enumerate(selected, 1):
        item = dict(item)
        item["source_row_id"] = item.get("id")
        item["id"] = f"rewriter_v2_{idx:06d}"
        item["split"] = "train" if idx <= train_cut else "validation" if idx <= val_cut else "test"
        final.append(item)
    return final


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_hidden_eval(path: Path, rng: random.Random) -> None:
    cases = []
    for idx in range(50):
        case_a = rng.choice(CASE_BANK)
        case_b = rng.choice([item for item in CASE_BANK if item != case_a])
        query, suffix, fields = rng.choice(COMPARE_FOLLOWUPS)
        cases.append(
            {
                "id": f"rw_hidden_compare_{idx:03d}",
                "category": "multi_case_compare",
                "messages": case_history(case_a, case_b) + [{"role": "user", "content": query}],
                "expected": {"standalone_contains_all": [case_a[1].removesuffix("案"), case_b[1].removesuffix("案")], "field_intents_any": fields},
            }
        )
    for idx in range(50):
        alias, normalized, crime, role = rng.choice(ALIASES)
        value = rng.choice([3000, 40000, None])
        amount_text = "" if value is None else f"{value}元"
        cases.append(
            {
                "id": f"rw_hidden_alias_{idx:03d}",
                "category": "colloquial_crime_alias",
                "messages": [{"role": "user", "content": f"{alias}{amount_text}一般怎么判，有案例参考吗"}],
                "expected": {"standalone_contains_all": [normalized.split()[0]], "crime_mentions_any": [crime], **({"amount_value": value} if value else {})},
            }
        )
    for idx in range(40):
        case_a = rng.choice(CASE_BANK)
        case_b = rng.choice([item for item in CASE_BANK if item != case_a])
        query, ordinal, fields, suffix = rng.choice(FIELD_FOLLOWUPS)
        selected = case_a if ordinal == 1 else case_b
        cases.append(
            {
                "id": f"rw_hidden_field_{idx:03d}",
                "category": "field_intent_followup",
                "messages": case_history(case_a, case_b) + [{"role": "user", "content": query}],
                "expected": {"standalone_contains_all": [selected[1].removesuffix("案")], "resolved_case_title_contains_any": [selected[1].removesuffix("案")], "field_intents_any": fields},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-coverage Query Rewriter-only v2 dataset.")
    parser.add_argument("--qu-dataset", type=Path, default=DEFAULT_QU_DATASET)
    parser.add_argument("--gpt-seed", type=Path, default=DEFAULT_GPT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-eval", type=Path, default=DEFAULT_HIDDEN_EVAL)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    rows.extend(generate_multiturn(1900, rng))
    rows.extend(generate_alias_and_amount(1500, rng))
    rows.extend(generate_field_intents(900, rng))
    rows.extend(load_seed_rows(args.gpt_seed))
    rows.extend(project_good_old_rows(args.qu_dataset, 900, rng))
    rows = assign_splits(rows, args.seed, args.target_count)

    invalid = [{"id": item["id"], "errors": validate_rewriter_answer(item["answer"])} for item in rows if validate_rewriter_answer(item["answer"])]
    if invalid:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "validation_report.json").write_text(json.dumps({"invalid": invalid[:50]}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Invalid rows: {len(invalid)}")

    split_rows = {split: [item for item in rows if item["split"] == split] for split in ["train", "validation", "test"]}
    for split, items in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", items)
    build_hidden_eval(args.hidden_eval, rng)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_count": args.target_count,
        "counts": {split: len(items) for split, items in split_rows.items()},
        "total_count": len(rows),
        "category_counts": dict(Counter(item["category"] for item in rows)),
        "source_method_counts": dict(Counter(item["source_method"] for item in rows)),
        "rewrite_required_count": sum(1 for item in rows if item["answer"].get("rewrite_required")),
        "rows_with_case_refs": sum(1 for item in rows if item["answer"].get("resolved_case_refs")),
        "rows_with_amounts": sum(1 for item in rows if item["answer"].get("amount_constraints")),
        "hidden_eval_path": str(args.hidden_eval),
        "hidden_eval_count": 140,
        "invalid_rows": 0,
        "notes": [
            "V2 reduces coarse old QU projection and emphasizes rewriter-only supervision.",
            "The largest blocks target multi-case comparison, colloquial crime normalization, amount roles and field intents.",
            "No router decision fields are present in assistant answers.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
