from __future__ import annotations

import argparse
import json
import time

from embed_retrieval_chunks_sample import model_cache_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose local embedding model availability without building the full index.")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--try-load", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = {"model": args.model, "cache": model_cache_status(args.model)}
    if args.try_load and result["cache"]["available"]:
        from sentence_transformers import SentenceTransformer

        start = time.time()
        model = SentenceTransformer(args.model, device=args.device, local_files_only=True)
        vector = model.encode(["刑法测试"], normalize_embeddings=True)
        result["load"] = {
            "seconds": round(time.time() - start, 3),
            "dimension": int(vector.shape[-1]),
            "device": args.device,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
