from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np

from common import OUT_DIR, iter_jsonl
from search_sample import ARTICLE_RE, article_no_from_citation, query_tokens


INDEX_DIR = OUT_DIR / "vector_index_sample"
_MODEL_CACHE: dict[tuple[str, str], object] = {}
FIELD_INTENT_BOOSTS = [
    (("辩称", "辩护", "辩解", "被告人说"), {"defense_text": 0.45, "reasoning_text": 0.18}),
    (("证据", "证人", "证言", "供述", "鉴定", "证明"), {"evidence_text": 0.35, "witness_testimony_texts": 0.35, "defendant_confession_texts": 0.3, "expert_opinion_texts": 0.28, "inspection_records_texts": 0.25}),
    (("为什么", "为何", "理由", "认定", "构成", "采纳", "不予采纳"), {"reasoning_text": 0.4}),
    (("事实", "经过", "查明", "案情", "发生了什么"), {"court_found_facts_text": 0.35, "accusation_text": 0.25}),
    (("判", "刑", "罚金", "处罚", "多久", "缓刑", "有期徒刑", "拘役"), {"judgment_text": 2.1, "reasoning_text": 0.05}),
]
CANONICAL_CRIME_ARTICLES = {
    "故意伤害罪": "234",
    "危险驾驶罪": "133.1",
    "诈骗罪": "266",
    "强奸罪": "236",
    "盗窃罪": "264",
}


def load_config(index_dir: Path = INDEX_DIR) -> dict:
    path = index_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing vector index config: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_metadata(index_dir: Path = INDEX_DIR) -> list[dict]:
    path = index_dir / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing vector index metadata: {path}")
    return list(iter_jsonl(path))


def format_hit(row: dict, score: float, rank: int) -> dict:
    return {
        "rank": rank,
        "score": round(float(score), 6),
        "retrieval_id": row.get("retrieval_id"),
        "chunk_id": row.get("chunk_id"),
        "source_type": row.get("source_type"),
        "case_id": row.get("case_id"),
        "title": row.get("title"),
        "field": row.get("field"),
        "article_no": row.get("article_no"),
        "law_name": row.get("law_name"),
        "law_version": row.get("law_version"),
        "source_file": row.get("source_file"),
        "text_hash": row.get("text_hash"),
        "text_preview": row.get("text_preview", ""),
        "case_metadata": row.get("case_metadata"),
        "law_metadata": row.get("law_metadata"),
        "quality_flags": row.get("quality_flags", []),
    }


def metadata_boost(query: str, row: dict) -> float:
    boost = 0.0
    tokens = query_tokens(query)
    requested_articles = {
        article_no
        for article_no in (article_no_from_citation(match.group(0)) for match in ARTICLE_RE.finditer(query))
        if article_no
    }
    title = row.get("title") or ""
    article_no = row.get("article_no")
    case_match = re.search(r"([\u4e00-\u9fff]{2,12})案", query)
    case_name = case_match.group(1) if case_match else ""
    sentence_intent = any(word in query for word in ("判", "刑", "罚金", "处罚", "多久", "缓刑", "有期徒刑", "拘役"))
    law_intent = any(word in query for word in ("法条", "刑法", "第几条", "规定", "对应"))
    if requested_articles and article_no in requested_articles:
        boost += 2.0
    if row.get("source_type") == "law_article":
        if sentence_intent and not law_intent:
            boost -= 3.4
        if case_name:
            boost -= 4.0
        for token in tokens:
            if token == title:
                boost += 1.6
            elif token.endswith("罪") and token in title:
                boost += 1.0
            elif len(token) >= 3 and token in title:
                boost += 0.35
        if any(word in query for word in ("法条", "刑法", "第几条", "规定")):
            boost += 0.15
        for crime, canonical_article in CANONICAL_CRIME_ARTICLES.items():
            if crime in query and title == crime and article_no == canonical_article:
                boost += 0.35
            elif crime in query and title == crime and article_no != canonical_article:
                boost -= 0.15
    elif any(token in title for token in tokens if len(token) >= 3):
        boost += 0.25
    if row.get("source_type") == "case_chunk" and case_name:
        if case_name in title or (len(case_name) >= 3 and case_name[:3] in title):
            boost += 1.2
        else:
            boost -= 0.8
    field = row.get("field")
    for triggers, field_boosts in FIELD_INTENT_BOOSTS:
        if any(trigger in query for trigger in triggers):
            boost += field_boosts.get(field, 0.0)
    return boost


def hash_embed_query(query: str, dimension: int) -> np.ndarray:
    vector = np.zeros((1, dimension), dtype="float32")
    chars = [ch for ch in query if not ch.isspace()]
    tokens = chars + ["".join(chars[idx:idx + 2]) for idx in range(max(0, len(chars) - 1))]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[0, value % dimension] += 1.0 if value & 1 else -1.0
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vector / norm


def search(query: str, top_k: int = 10, index_dir: Path = INDEX_DIR) -> dict:
    config = load_config(index_dir)
    metadata = load_metadata(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    if config.get("backend") == "hash":
        vector = hash_embed_query(query, int(config["dimension"]))
    else:
        from sentence_transformers import SentenceTransformer

        key = (config["model"], str(config.get("device", "cpu")))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = SentenceTransformer(config["model"], device=key[1], local_files_only=True)
        model = _MODEL_CACHE[key]
        vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    candidate_k = index.ntotal if config.get("backend") == "hash" else min(index.ntotal, max(top_k * 20, 200))
    scores, indices = index.search(np.asarray(vector, dtype="float32"), candidate_k)
    hits = []
    scored = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[int(idx)]
        scored.append((float(score) + metadata_boost(query, row), float(score), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (reranked_score, raw_score, row) in enumerate(scored[:top_k], 1):
        hit = format_hit(row, reranked_score, rank)
        hit["vector_score"] = round(raw_score, 6)
        hit["metadata_boost"] = round(reranked_score - raw_score, 6)
        hits.append(hit)
    return {
        "query": query,
        "model": config.get("model"),
        "top_k": top_k,
        "hits": hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the sample FAISS vector index.")
    parser.add_argument("query", nargs="?", default="吴必定案判了多久")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = search(args.query, args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"query: {result['query']}")
    print(f"model: {result['model']}")
    for hit in result["hits"]:
        article = f" article_no={hit['article_no']}" if hit.get("article_no") else ""
        print(f"{hit['rank']}. score={hit['score']} {hit['source_type']} {hit['field']} {hit['title']}{article}")
        print(f"   id={hit['retrieval_id']}")
        print(f"   text={hit['text_preview'][:180]}")


if __name__ == "__main__":
    main()
