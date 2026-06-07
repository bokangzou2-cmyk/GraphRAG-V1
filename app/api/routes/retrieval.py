from __future__ import annotations

from fastapi import APIRouter, Depends

import app.processing_bridge  # noqa: F401
from app.auth import current_user
from app.schemas import RetrievalRequest
from hybrid_search_sample import search as hybrid_search


router = APIRouter()


@router.post("/retrieval/search")
def retrieval_search(request: RetrievalRequest, user: dict = Depends(current_user)) -> dict:
    return hybrid_search(request.query, top_k=request.top_k)
