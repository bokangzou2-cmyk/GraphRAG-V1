from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=20)


class OrchestrationChatRequest(ChatRequest):
    mock_llm_mode: Literal["grounded", "invalid_json", "missing_citation", "empty_answer", "no_answer"] = "grounded"
    generation_mode: Literal["mock", "provider"] = "provider"


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthUser(BaseModel):
    id: str
    username: str
    role: Literal["admin", "user", "auditor"]
    createdAt: int
    lastLoginAt: int | None = None


class LoginResponse(BaseModel):
    token: str
    expiresAt: int
    user: AuthUser


class Citation(BaseModel):
    retrieval_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    field: str | None = None
    case_id: str | None = None
    source_file: str | None = None
    text_hash: str | None = None
    article_no: str | None = None
    law_version: str | None = None
    hybrid_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    evidence_preview: str | None = None
    amount_diagnostics: dict | None = None


class Confidence(BaseModel):
    level: Literal["none", "low", "medium", "high"]
    score: float
    reason: str | None = None
    query_type: str | None = None
    top_hybrid_score: float | None = None
    graph_supported_contexts: int | None = None
    source_diversity: int | None = None
    citation_ready_contexts: int | None = None
    vector_supported_contexts: int | None = None
    evidence_checks: list[dict] = Field(default_factory=list)
    router: dict | None = None


class ChatResponse(BaseModel):
    answer: str
    answer_type: Literal["no_answer", "low_confidence", "llm_grounded", "llm_direct", "retrieval_fallback"]
    citations: list[Citation]
    graph_paths: list[dict]
    warnings: list[str]
    confidence: Confidence
    refusal_reason: str | None = None
    retrieval_context: list[dict]
    metrics: dict = Field(default_factory=dict)


class OrchestrationChatResponse(ChatResponse):
    orchestration_logs: list[dict] = Field(default_factory=list)
    provider_error: dict | None = None


class StoredConversation(BaseModel):
    id: str
    title: str
    messages: list[dict]
    createdAt: int
    updatedAt: int | None = None


class ConversationListResponse(BaseModel):
    conversations: list[StoredConversation]


class ChatRunAudit(BaseModel):
    id: str
    createdAt: int
    userId: str | None = None
    path: Literal["stable", "orchestration"]
    query: str
    topK: int
    answerType: str
    confidenceLevel: str | None = None
    confidenceScore: float | None = None
    refusalReason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    citationIds: list[str] = Field(default_factory=list)
    graphPathCount: int
    retrievalContextCount: int
    answerPreview: str
    metrics: dict = Field(default_factory=dict)


class ChatRunAuditListResponse(BaseModel):
    runs: list[ChatRunAudit]
