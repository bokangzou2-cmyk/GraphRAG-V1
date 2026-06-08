from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from common import OUT_DIR, iter_jsonl


INPUT = OUT_DIR / "retrieval_chunks_sample.jsonl"
INDEX_DIR = OUT_DIR / "vector_index_sample"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def hf_snapshot_dir(model_name: str) -> Path | None:
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_root = cache_root / f"models--{model_name.replace('/', '--')}"
    ref = model_root / "refs" / "main"
    if not ref.exists():
        return None
    snapshot = model_root / "snapshots" / ref.read_text(encoding="utf-8").strip()
    return snapshot if snapshot.exists() else None


def model_cache_status(model_name: str) -> dict:
    snapshot = hf_snapshot_dir(model_name)
    if not snapshot:
        return {"available": False, "reason": "snapshot_missing", "snapshot": None, "missing": ["snapshot"]}
    names = {path.name for path in snapshot.iterdir()}
    has_weights = bool({"model.safetensors", "pytorch_model.bin"} & names)
    required = {"config.json", "modules.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"}
    missing = sorted(required - names)
    if not (snapshot / "1_Pooling" / "config.json").exists():
        missing.append("1_Pooling/config.json")
    if not has_weights:
        missing.append("model.safetensors|pytorch_model.bin")
    return {
        "available": not missing,
        "reason": "ok" if not missing else "incomplete_snapshot",
        "snapshot": str(snapshot),
        "missing": missing,
    }


def metadata_for(row: dict) -> dict:
    law_metadata = row.get("law_metadata") or {}
    case_metadata = row.get("case_metadata") or {}
    return {
        "retrieval_id": row.get("retrieval_id"),
        "chunk_id": row.get("chunk_id"),
        "source_type": row.get("source_type"),
        "case_id": row.get("case_id"),
        "title": row.get("title"),
        "field": row.get("field"),
        "article_no": law_metadata.get("article_no"),
        "law_name": law_metadata.get("law_name"),
        "law_version": law_metadata.get("law_version") or ("current" if row.get("source_type") == "law_article" else None),
        "source_file": row.get("source_file"),
        "text_hash": row.get("text_hash"),
        "text": row.get("text", ""),
        "text_preview": row.get("text", "")[:360],
        "case_metadata": case_metadata or None,
        "law_metadata": law_metadata or None,
        "quality_flags": row.get("quality_flags", []),
    }


def row_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("retrieval_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("text_hash", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def existing_metadata_fingerprint(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    count = 0
    digest = hashlib.sha256()
    for row in iter_jsonl(path):
        count += 1
        digest.update(str(row.get("retrieval_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("text_hash", "")).encode("utf-8"))
        digest.update(b"\n")
    return count, digest.hexdigest()


def can_reuse_existing_index(rows: list[dict], index_dir: Path, rebuild: bool, model: str, backend: str) -> bool:
    if rebuild:
        return False
    required = [index_dir / "index.faiss", index_dir / "metadata.jsonl", index_dir / "config.json"]
    if not all(path.exists() for path in required):
        return False
    try:
        config = json.loads((index_dir / "config.json").read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return False
    if config.get("model") != model or config.get("backend") != backend:
        return False
    metadata_fingerprint = existing_metadata_fingerprint(index_dir / "metadata.jsonl")
    if metadata_fingerprint is None:
        return False
    count, fingerprint = metadata_fingerprint
    return count == len(rows) and fingerprint == row_fingerprint(rows)


def hash_embed_texts(texts: list[str], dimension: int = 768) -> np.ndarray:
    vectors = np.zeros((len(texts), dimension), dtype="float32")
    for row_idx, text in enumerate(texts):
        chars = [ch for ch in text if not ch.isspace()]
        tokens = chars + ["".join(chars[idx:idx + 2]) for idx in range(max(0, len(chars) - 1))]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vectors[row_idx, value % dimension] += 1.0 if value & 1 else -1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_texts(model_name: str, texts: list[str], batch_size: int, backend: str, device: str, local_files_only: bool) -> tuple[np.ndarray, str, dict]:
    if backend == "hash":
        return hash_embed_texts(texts), "hash", {"reason": "requested_hash"}
    status = model_cache_status(model_name)
    if local_files_only and not status["available"]:
        raise RuntimeError(
            "sentence_transformers_model_not_available_locally; "
            "rerun with --allow-download or pre-download the model. "
            f"model={model_name} missing={status.get('missing')}"
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32"), "sentence-transformers", {"model_cache": status}


def write_report(index_dir: Path, config: dict, elapsed_seconds: float) -> None:
    report = {
        "created_at_utc": config["created_at_utc"],
        "updated_chunk_embedding_count": config["count"],
        "dimension": config["dimension"],
        "index_path": str(index_dir / "index.faiss"),
        "metadata_path": str(index_dir / "metadata.jsonl"),
        "config_path": str(index_dir / "config.json"),
        "model": config["model"],
        "backend": config["backend"],
        "device": config["device"],
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    (index_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_dir / "build_report.txt").write_text(
        "\n".join(
            [
                "Vector index build report",
                f"updated_chunk_embedding_count: {report['updated_chunk_embedding_count']}",
                f"dimension: {report['dimension']}",
                f"index_path: {report['index_path']}",
                f"metadata_path: {report['metadata_path']}",
                f"config_path: {report['config_path']}",
                f"model: {report['model']}",
                f"backend: {report['backend']}",
                f"device: {report['device']}",
                f"elapsed_seconds: {report['elapsed_seconds']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FAISS vector index for retrieval_chunks_sample.jsonl.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true", help="Allow sentence-transformers to download a missing model instead of using the hash fallback.")
    parser.add_argument("--local-files-only", action="store_true", help="Deprecated; local-only mode is now the default unless --allow-download is set.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    rows = list(iter_jsonl(INPUT))
    if not rows:
        raise SystemExit(f"missing or empty input: {INPUT}")
    if can_reuse_existing_index(rows, INDEX_DIR, args.rebuild, args.model, args.backend):
        print(f"vector index is current -> {INDEX_DIR}")
        print(f"count: {len(rows)}")
        return
    if INDEX_DIR.exists() and args.rebuild:
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    texts = [row.get("text", "") for row in rows]
    local_files_only = args.local_files_only or not args.allow_download
    started = time.perf_counter()
    resolved_device = resolve_device(args.device)
    vectors, effective_backend, backend_info = embed_texts(args.model, texts, args.batch_size, args.backend, resolved_device, local_files_only)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with (INDEX_DIR / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(metadata_for(row), ensure_ascii=False) + "\n")
    config = {
        "schema_version": "vector-index-sample-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT),
        "model": args.model,
        "requested_backend": args.backend,
        "backend": effective_backend,
        "backend_info": backend_info,
        "device": resolved_device,
        "local_files_only": local_files_only,
        "query_instruction": args.query_instruction,
        "input_fingerprint": row_fingerprint(rows),
        "metric": "inner_product_normalized_cosine",
        "count": len(rows),
        "dimension": int(vectors.shape[1]),
    }
    (INDEX_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    write_report(INDEX_DIR, config, elapsed)
    print(f"wrote vector index -> {INDEX_DIR}")
    print(f"count: {len(rows)} dimension: {vectors.shape[1]} model: {args.model} backend: {effective_backend}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"report: {INDEX_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
