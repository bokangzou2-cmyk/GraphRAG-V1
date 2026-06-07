from __future__ import annotations

from fastapi import APIRouter

from app.config import ROOT, settings


router = APIRouter()


@router.get("/health")
def health() -> dict:
    retrieval = ROOT / "processing_data" / "retrieval_chunks_sample.jsonl"
    graph = ROOT / "processing_data" / "graph_sample.jsonl"
    vector = ROOT / "processing_data" / "vector_index_sample" / "index.faiss"
    return {
        "status": "ok",
        "data": {
            "retrieval_chunks_sample": retrieval.exists(),
            "graph_sample": graph.exists(),
            "vector_index_sample": vector.exists(),
            "chat_history_db": settings.chat_history_db_path.exists(),
        },
        "llm": {
            "provider": settings.llm_provider,
            "configured": bool(settings.llm_api_key),
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        },
    }
