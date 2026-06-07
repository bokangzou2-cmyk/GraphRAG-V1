from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth import current_user
from app.chat_history import ChatHistoryStore
from app.config import settings
from app.schemas import ConversationListResponse, StoredConversation


router = APIRouter()
store = ChatHistoryStore(settings.chat_history_db_path)


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(user: dict = Depends(current_user)) -> dict:
    return {"conversations": store.list_conversations(user_id=user["id"])}


@router.get("/conversations/{conversation_id}", response_model=StoredConversation)
def get_conversation(conversation_id: str, user: dict = Depends(current_user)) -> dict:
    conversation = store.get_conversation(conversation_id, user_id=user["id"])
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return conversation


@router.put("/conversations/{conversation_id}", response_model=StoredConversation)
def upsert_conversation(conversation_id: str, conversation: StoredConversation, user: dict = Depends(current_user)) -> dict:
    payload = conversation.model_dump()
    payload["id"] = conversation_id
    return store.upsert_conversation(payload, user_id=user["id"])


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(current_user)) -> dict:
    deleted = store.delete_conversation(conversation_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return {"deleted": True, "id": conversation_id}
