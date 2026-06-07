from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from common import OUT_DIR, RAW_DIR, iter_jsonl, normalize_text, write_jsonl


QUESTION_DIR = OUT_DIR / "router_training_questions"
CRIMINAL_OUT = QUESTION_DIR / "criminal_law_questions.jsonl"
OTHER_OUT = QUESTION_DIR / "other_questions.jsonl"
ALL_OUT = QUESTION_DIR / "all_questions_unlabeled.jsonl"
REPORT_OUT = QUESTION_DIR / "collection_report.json"

JUDICIAL_EXAM = RAW_DIR / "data" / "data" / "司法考试" / "test_input.json"
CASE_QUERY = RAW_DIR / "data" / "data" / "类案检索" / "query_stage2.json"
LAW_ARTICLES = OUT_DIR / "law_articles_sample.jsonl"

CRIMINAL_RE = re.compile(
    r"刑法|犯罪|罪名|罪责|刑罚|刑事责任|刑事案件|判刑|量刑|判处|有期徒刑|无期徒刑|拘役|管制|罚金|死刑|缓刑|"
    r"自首|累犯|立功|数罪并罚|正当防卫|紧急避险|共同犯罪|未遂|既遂|中止|主犯|从犯|"
    r"故意杀人|杀人|故意伤害|伤害|盗窃|抢劫|诈骗|危险驾驶|交通肇事|贪污|受贿|职务侵占|非法经营|"
    r"寻衅滋事|开设赌场|非法拘禁|强奸|毒品|贩毒|走私|帮信|帮助信息网络犯罪|洗钱|"
    r"敲诈勒索|放火|爆炸|抢夺|侵占罪|非法吸收公众存款|集资诈骗|组织、领导黑社会性质组织|"
    r"逮捕|取保候审|公诉|检察院|引渡"
)
STRICT_CRIMINAL_RE = re.compile(
    r"刑法|犯罪|罪名|罪责|刑罚|刑事责任|刑事案件|构成何罪|构成[^，。？?]{0,12}罪|"
    r"判刑|量刑|判处|有期徒刑|无期徒刑|拘役|管制|罚金|死刑|缓刑|自首|累犯|立功|数罪并罚|"
    r"正当防卫|紧急避险|共同犯罪|未遂|既遂|中止|主犯|从犯|"
    r"故意杀人罪|故意伤害罪|盗窃罪|抢劫罪|诈骗罪|危险驾驶罪|交通肇事罪|贪污罪|受贿罪|"
    r"职务侵占罪|非法经营罪|寻衅滋事罪|开设赌场罪|非法拘禁罪|强奸罪|贩卖毒品罪|走私[^，。？?]{0,8}罪|"
    r"帮助信息网络犯罪|洗钱罪|敲诈勒索罪|放火罪|爆炸罪|抢夺罪|侵占罪|非法吸收公众存款罪|集资诈骗罪"
)
LEGAL_RE = re.compile(
    r"法律|法规|法院|起诉|诉讼|律师|合同|民法|婚姻|离婚|继承|劳动|工伤|行政|仲裁|公司法|"
    r"证券|知识产权|商标|专利|著作权|房产|物业|抚养|赡养|赔偿|借款|债务|欠款|租赁|"
    r"交通事故|消费者|拆迁|征地|户口|社保|医保|公积金"
)
QUESTION_HINT_RE = re.compile(r"[?？]|什么|如何|怎么|是否|能否|可以|哪|谁|为何|为什么|多少|几|吗|呢|现问|下列")
TEMPLATE_FAMILY_PATTERNS = {
    "case_alias": re.compile(r"罪案"),
    "law_article_no": re.compile(r"刑法第\d+条"),
    "crime_general": re.compile(r"罪(一般怎么判|的构成要件|对应刑法|量刑情节|适用缓刑|相近犯罪|立案后|需要哪些)"),
    "exam": re.compile(r"下列|哪一项|哪些选项|正确的是|不正确的是"),
}


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def clean_question(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    text = re.sub(r"^我有一个(?:计算机|信息科学)相关的问题，请用中文回答，", "", text)
    text = re.sub(r"^请在此输入您", "", text)
    text = re.sub(r"的问题[，,。]?祝您的问题早日得到解决[！!。]?$", "", text)
    text = re.sub(r"知道告诉一下下啦~~?", "", text)
    text = re.sub(r"(?i)\bRT\b|如题|求关注|谢邀", "", text)
    text = re.sub(r"\[[^\]]{1,10}\]", "", text)
    text = re.sub(r"问题[:：]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" 　\t\r\n，,。；;：:")
    return text


def acceptable_question(text: str, min_len: int = 6, max_len: int = 220) -> bool:
    if not (min_len <= len(text) <= max_len):
        return False
    if not re.search(r"[\u4e00-\u9fff]", text):
        return False
    if re.search(r"选项[:：]|答案[:：]|从A到D|<blockquote>|XXX|＿|_{2,}|\*{2,}", text):
        return False
    if re.search(r"\?{2,}|？{2,}|�|铱|老\?|<img|<a\s", text):
        return False
    if len(text) > 100 and re.search(r"下列|哪一项|哪些选项|正确的是|不正确的是|如何处理", text):
        return False
    if re.search(r"下列[^？?。；;，,]{0,8}$|说法[^？?。；;，,]{0,4}$|正确的是[:：]?$|不正确的是[:：]?$", text):
        return False
    if text.count("，") + text.count(",") > 12 and "？" not in text and "?" not in text:
        return False
    return bool(QUESTION_HINT_RE.search(text))


def clean_exam_query(text: str) -> str:
    text = re.split(r"选项[:：]|\n?答案[:：]", text, maxsplit=1)[0]
    return clean_question(text)


def is_criminal_law_question(text: str) -> bool:
    if STRICT_CRIMINAL_RE.search(text):
        return True
    return bool(re.search(r"罪.*(?:构成|量刑|处罚|判刑|判多久|怎么判|立案|刑法)|(?:构成|涉嫌|犯|触犯)[^？?。]{0,20}罪", text))


def iter_concatenated_json(path: Path) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        idx = next_idx
        if isinstance(obj, dict):
            yield obj


def add_unique(rows: list[dict], seen: set[str], question: str, source: str, source_id: str, source_method: str) -> None:
    question = clean_question(question)
    if not acceptable_question(question):
        return
    key = re.sub(r"\s+", "", question)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "id": stable_id(source, question),
            "question": question,
            "source": source,
            "source_id": str(source_id),
            "source_method": source_method,
            "template_family": template_family(question, source_method),
        }
    )


def template_family(question: str, source_method: str) -> str:
    if source_method != "derived_question" and source_method != "curated_question":
        return "natural_or_dataset"
    for name, pattern in TEMPLATE_FAMILY_PATTERNS.items():
        if pattern.search(question):
            return name
    return "derived_other"


def collect_judicial_questions() -> tuple[list[dict], list[dict]]:
    criminal: list[dict] = []
    non_criminal_legal: list[dict] = []
    seen_criminal: set[str] = set()
    seen_other: set[str] = set()
    for obj in iter_concatenated_json(JUDICIAL_EXAM):
        source_id = str(obj.get("id", ""))
        statement = clean_question(obj.get("statement", ""))
        if not acceptable_question(statement):
            continue
        if is_criminal_law_question(statement):
            add_unique(criminal, seen_criminal, statement, "local_judicial_exam_criminal", source_id, "extracted_question")
        elif LEGAL_RE.search(statement) and not CRIMINAL_RE.search(statement):
            add_unique(non_criminal_legal, seen_other, statement, "local_judicial_exam_non_criminal_legal", source_id, "extracted_question")
    return criminal, non_criminal_legal


def collect_case_derived_questions(limit: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    if not CASE_QUERY.exists():
        return rows
    generic_templates = [
        "{crime}一般怎么判？",
        "{crime}的构成要件是什么？",
        "{crime}对应刑法哪一条？",
        "{crime}会判有期徒刑吗？",
        "{crime}能否适用缓刑？",
        "{crime}案件通常会关注哪些量刑情节？",
        "{crime}案件中自首和认罪认罚会影响量刑吗？",
        "{crime}案件的裁判依据一般是什么？",
        "{crime}和相近罪名如何区分？",
        "{crime}既遂和未遂怎么判断？",
    ]
    case_templates = [
        "{alias}应如何量刑？",
        "{alias}适用了哪些刑法规定？",
        "{alias}有哪些从轻或从重情节？",
        "检索{alias}时应重点看哪些事实？",
        "和{alias}相似的案例有哪些？",
    ]
    for obj in iter_concatenated_json(CASE_QUERY):
        crimes = obj.get("crime") or []
        if isinstance(crimes, str):
            crimes = [crimes]
        fact_text = clean_question(obj.get("q", ""))
        defendant_match = re.search(r"被告人([\u4e00-\u9fff]{2,3})(?:，|以|因|于|系|犯|到案|在|的)", fact_text)
        defendant = defendant_match.group(1) if defendant_match else ""
        for crime in crimes:
            crime = clean_question(str(crime))
            if not crime or not CRIMINAL_RE.search(crime):
                continue
            questions = [template.format(crime=crime) for template in generic_templates]
            if defendant and "XXX" not in defendant and defendant not in {"某某", "某"} and not re.search(r"驾驶|酒后|利用|拦住|随即|负责|为了|伙同|犯", defendant):
                alias = f"{defendant}{crime}案"
                questions.extend(template.format(alias=alias) for template in case_templates)
            for question in questions:
                add_unique(
                    rows,
                    seen,
                    question,
                    "local_case_query_crime_derived",
                    obj.get("path", obj.get("ridx", "")),
                    "derived_question",
                )
                if len(rows) >= limit:
                    return rows
    return rows


def collect_law_article_questions(limit: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    if not LAW_ARTICLES.exists():
        return rows
    for row in iter_jsonl(LAW_ARTICLES):
        article_no = row.get("article_no")
        title = clean_question(row.get("title", ""))
        source_id = row.get("article_id") or article_no or len(rows)
        questions = []
        if article_no:
            questions.extend(
                [
                    f"刑法第{article_no}条规定了什么？",
                    f"刑法第{article_no}条适用于哪些情形？",
                    f"刑法第{article_no}条的核心内容是什么？",
                ]
            )
        if title and title.endswith("罪"):
            questions.extend(
                [
                    f"{title}一般怎么判？",
                    f"{title}的构成要件是什么？",
                    f"{title}对应刑法哪一条？",
                    f"{title}有哪些量刑情节？",
                    f"{title}能否适用缓刑？",
                    f"{title}与相近犯罪如何区分？",
                ]
            )
        for question in questions:
            add_unique(rows, seen, question, "local_criminal_law_article_derived", source_id, "derived_question")
            if len(rows) >= limit:
                return rows
    return rows


def collect_curated_criminal_questions(limit: int) -> list[dict]:
    crimes = [
        "盗窃罪", "抢劫罪", "诈骗罪", "故意伤害罪", "故意杀人罪", "危险驾驶罪", "交通肇事罪",
        "寻衅滋事罪", "开设赌场罪", "非法拘禁罪", "敲诈勒索罪", "职务侵占罪", "贪污罪",
        "受贿罪", "非法经营罪", "帮助信息网络犯罪活动罪", "洗钱罪", "抢夺罪", "侵占罪",
        "非法吸收公众存款罪", "集资诈骗罪", "贩卖毒品罪", "走私普通货物、物品罪",
    ]
    templates = [
        "{crime}立案后通常怎么量刑？",
        "{crime}需要哪些构成要件？",
        "{crime}和相关罪名有什么区别？",
        "{crime}什么情况下可以从轻处罚？",
        "{crime}适用缓刑要看哪些条件？",
    ]
    rows: list[dict] = []
    seen: set[str] = set()
    for crime in crimes:
        for template in templates:
            add_unique(rows, seen, template.format(crime=crime), "curated_criminal_law_questions", crime, "curated_question")
            if len(rows) >= limit:
                return rows
    return rows


def hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def collect_hc3_questions(filename: str, source: str, want: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    path = hf_download("Hello-SimpleAI/HC3-Chinese", filename)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            question = json.loads(line).get("question", "")
        except json.JSONDecodeError:
            continue
        question = clean_question(question)
        is_criminal = bool(CRIMINAL_RE.search(question))
        is_legal = bool(LEGAL_RE.search(question) or is_criminal)
        if want == "general" and is_legal:
            continue
        if want == "non_criminal_legal" and (CRIMINAL_RE.search(question) or is_criminal_law_question(question) or not is_legal):
            continue
        if want == "criminal" and not is_criminal_law_question(question):
            continue
        add_unique(rows, seen, question, source, idx, "extracted_question")
        if len(rows) >= limit:
            break
    return rows


def collect_disc_law_questions(want: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    path = hf_download("ShengbinYue/DISC-Law-SFT", "DISC-Law-SFT-Pair-QA-released.jsonl")
    source = "disc_law_sft_criminal" if want == "criminal" else "disc_law_sft_non_criminal_legal"
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                question = json.loads(line).get("input", "")
            except json.JSONDecodeError:
                continue
            question = clean_question(question)
            is_criminal = bool(CRIMINAL_RE.search(question))
            if want == "criminal" and not is_criminal_law_question(question):
                continue
            if want == "non_criminal_legal" and (CRIMINAL_RE.search(question) or is_criminal_law_question(question)):
                continue
            add_unique(rows, seen, question, source, idx, "extracted_question")
            if len(rows) >= limit:
                break
    return rows


def collect_agieval_jec_questions() -> tuple[list[dict], list[dict]]:
    try:
        import pandas as pd
    except ImportError:
        return [], []
    path = hf_download("hails/agieval-jec-qa-ca", "data/test-00000-of-00001.parquet")
    frame = pd.read_parquet(path)
    criminal: list[dict] = []
    non_criminal_legal: list[dict] = []
    seen_criminal: set[str] = set()
    seen_other: set[str] = set()
    for idx, row in frame.iterrows():
        question = clean_exam_query(str(row.get("query", "")))
        if is_criminal_law_question(question):
            add_unique(criminal, seen_criminal, question, "agieval_jec_qa_criminal", str(idx), "extracted_question")
        elif not CRIMINAL_RE.search(question):
            add_unique(non_criminal_legal, seen_other, question, "agieval_jec_qa_non_criminal_legal", str(idx), "extracted_question")
    return criminal, non_criminal_legal


def take_shuffled(rows: list[dict], count: int, seed: int) -> list[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def merge_unique(groups: list[list[dict]], count: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            key = re.sub(r"\s+", "", row["question"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
            if len(merged) >= count:
                return merged
    return merged


def with_split(rows: list[dict], split_name: str) -> list[dict]:
    return [{**row, "split": split_name} for row in rows]


def remove_seen(rows: list[dict], blocked_questions: set[str]) -> list[dict]:
    kept = []
    for row in rows:
        key = re.sub(r"\s+", "", row["question"])
        if key not in blocked_questions:
            kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect unlabeled query-router training questions.")
    parser.add_argument("--criminal-count", type=int, default=2000)
    parser.add_argument("--other-count", type=int, default=3000)
    parser.add_argument("--non-criminal-legal-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    judicial_criminal, judicial_non_criminal_legal = collect_judicial_questions()
    agieval_criminal, agieval_non_criminal_legal = collect_agieval_jec_questions()
    case_derived = collect_case_derived_questions(limit=1800)
    law_derived = collect_law_article_questions(limit=1200)
    curated_criminal = collect_curated_criminal_questions(limit=200)
    hc3_criminal = collect_hc3_questions("law.jsonl", "hc3_law_criminal", "criminal", 200)
    disc_criminal = collect_disc_law_questions("criminal", 800)

    criminal_questions = merge_unique(
        [
            take_shuffled(disc_criminal, 650, args.seed + 12),
            take_shuffled(case_derived, 630, args.seed + 1),
            take_shuffled(law_derived, 560, args.seed + 2),
            take_shuffled(curated_criminal, 100, args.seed + 11),
            take_shuffled(agieval_criminal, 100, args.seed + 9),
            take_shuffled(hc3_criminal, 100, args.seed + 3),
            take_shuffled(judicial_criminal, 0, args.seed + 4),
        ],
        args.criminal_count,
    )

    hc3_non_criminal_legal = collect_hc3_questions("law.jsonl", "hc3_law_non_criminal_legal", "non_criminal_legal", 500)
    disc_non_criminal_legal = collect_disc_law_questions("non_criminal_legal", 500)
    criminal_keys = {re.sub(r"\s+", "", row["question"]) for row in criminal_questions}
    disc_non_criminal_legal = remove_seen(disc_non_criminal_legal, criminal_keys)
    hc3_non_criminal_legal = remove_seen(hc3_non_criminal_legal, criminal_keys)
    agieval_non_criminal_legal = remove_seen(agieval_non_criminal_legal, criminal_keys)
    judicial_non_criminal_legal = remove_seen(judicial_non_criminal_legal, criminal_keys)
    non_criminal_legal = merge_unique(
        [
            take_shuffled(disc_non_criminal_legal, 250, args.seed + 13),
            take_shuffled(hc3_non_criminal_legal, 200, args.seed + 5),
            take_shuffled(agieval_non_criminal_legal, 100, args.seed + 10),
            take_shuffled(judicial_non_criminal_legal, 0, args.seed + 6),
        ],
        args.non_criminal_legal_count,
    )

    hc3_open = collect_hc3_questions("open_qa.jsonl", "hc3_open_qa_general", "general", 2000)
    hc3_baike = collect_hc3_questions("baike.jsonl", "hc3_baike_general", "general", 2000)
    general_needed = args.other_count - args.non_criminal_legal_count
    general_questions = merge_unique(
        [
            take_shuffled(hc3_open, 1400, args.seed + 7),
            take_shuffled(hc3_baike, 1400, args.seed + 8),
        ],
        general_needed,
    )
    other_questions = merge_unique([non_criminal_legal, general_questions], args.other_count)

    if len(criminal_questions) != args.criminal_count:
        raise SystemExit(f"criminal question count shortfall: {len(criminal_questions)} / {args.criminal_count}")
    if len(non_criminal_legal) != args.non_criminal_legal_count:
        raise SystemExit(f"non-criminal legal question count shortfall: {len(non_criminal_legal)} / {args.non_criminal_legal_count}")
    if len(other_questions) != args.other_count:
        raise SystemExit(f"other question count shortfall: {len(other_questions)} / {args.other_count}")

    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(CRIMINAL_OUT, criminal_questions)
    write_jsonl(OTHER_OUT, other_questions)
    all_rows = with_split(criminal_questions, "criminal_law_questions") + with_split(other_questions, "other_questions")
    write_jsonl(ALL_OUT, all_rows)

    report = {
        "schema_version": "router-question-collection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "criminal_law_questions": str(CRIMINAL_OUT),
            "other_questions": str(OTHER_OUT),
            "all_questions_unlabeled": str(ALL_OUT),
        },
        "requested_counts": {
            "criminal_law_questions": args.criminal_count,
            "other_questions": args.other_count,
            "non_criminal_legal_inside_other": args.non_criminal_legal_count,
        },
        "actual_counts": {
            "criminal_law_questions": len(criminal_questions),
            "other_questions": len(other_questions),
            "non_criminal_legal_inside_other": sum(
                1
                for row in other_questions
                if "non_criminal_legal" in row.get("source", "")
            ),
            "all_questions_unlabeled": len(all_rows),
        },
        "source_counts": {
            "criminal_law_questions": Counter(row["source"] for row in criminal_questions),
            "other_questions": Counter(row["source"] for row in other_questions),
        },
        "template_family_counts": {
            "criminal_law_questions": Counter(row.get("template_family") for row in criminal_questions),
            "other_questions": Counter(row.get("template_family") for row in other_questions),
        },
        "quality_checks": quality_checks(criminal_questions, other_questions, all_rows),
        "notes": [
            "Records are not route-labeled; files are separated only to satisfy collection quotas.",
            "Non-criminal legal questions are included inside other_questions and identifiable by source metadata for audit.",
            "Derived criminal questions are generated from local case/law corpora without using an LLM.",
        ],
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["actual_counts"], ensure_ascii=False, indent=2))
    print(f"wrote {CRIMINAL_OUT}")
    print(f"wrote {OTHER_OUT}")
    print(f"wrote {ALL_OUT}")
    print(f"wrote {REPORT_OUT}")


def quality_checks(criminal_questions: list[dict], other_questions: list[dict], all_rows: list[dict]) -> dict:
    bad_format = re.compile(r"选项[:：]|答案[:：]|从A到D|<[^>]+>|&[a-z]+;|blockquote|XXX|＿|_{2,}|\*{2,}|\?{2,}|？{2,}|�|铱")
    broad_criminal = CRIMINAL_RE
    return {
        "criminal_count": len(criminal_questions),
        "other_count": len(other_questions),
        "all_count": len(all_rows),
        "duplicate_questions": len(all_rows) - len({re.sub(r"\s+", "", row["question"]) for row in all_rows}),
        "bad_format_hits": sum(bool(bad_format.search(row["question"])) for row in all_rows),
        "other_broad_criminal_hits": sum(bool(broad_criminal.search(row["question"])) for row in other_questions),
        "criminal_broad_criminal_hits": sum(bool(broad_criminal.search(row["question"])) for row in criminal_questions),
        "max_question_length": max(len(row["question"]) for row in all_rows),
        "long_question_over_180": sum(len(row["question"]) > 180 for row in all_rows),
        "max_single_source_share": max(Counter(row["source"] for row in all_rows).values()) / len(all_rows),
        "max_criminal_template_family_share": max(Counter(row.get("template_family") for row in criminal_questions).values()) / len(criminal_questions),
    }


if __name__ == "__main__":
    main()
