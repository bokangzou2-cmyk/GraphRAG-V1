from __future__ import annotations

from fastapi import APIRouter, Depends

import app.processing_bridge  # noqa: F401
from app.audit_log import AnswerAuditStore
from app.auth import current_user
from app.config import settings
from app.llm.chat_service import latest_user_query
from app.schemas import ChatRequest, ChatResponse, ChatRunAuditListResponse, OrchestrationChatRequest, OrchestrationChatResponse
from langchain_orchestration_sample import run_orchestration_with_mode


router = APIRouter()
audit_store = AnswerAuditStore(settings.chat_history_db_path)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user: dict = Depends(current_user)) -> dict:
    messages = [message.model_dump() for message in request.messages]
    response = run_orchestration_with_mode(
        messages,
        top_k=request.top_k,
        mock_llm_mode="grounded",
        generation_mode="provider",
    )
    audit_store.record_run(path="orchestration", query=latest_user_query(messages), top_k=request.top_k, response=response, user_id=user["id"])
    return response


@router.post("/chat/orchestration", response_model=OrchestrationChatResponse)
def chat_orchestration(request: OrchestrationChatRequest, user: dict = Depends(current_user)) -> dict:
    messages = [message.model_dump() for message in request.messages]
    response = run_orchestration_with_mode(
        messages,
        top_k=request.top_k,
        mock_llm_mode=request.mock_llm_mode,
        generation_mode=request.generation_mode,
    )
    audit_store.record_run(path="orchestration", query=latest_user_query(messages), top_k=request.top_k, response=response, user_id=user["id"])
    return response


@router.get("/chat/audit-runs", response_model=ChatRunAuditListResponse)
def list_chat_audit_runs(limit: int = 50, user: dict = Depends(current_user)) -> dict:
    return {"runs": audit_store.list_runs(limit=limit, user_id=user["id"])}
