from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class ChatHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
            if "user_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")

    def list_conversations(self, user_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE user_id = ?" if user_id else ""
        params = (user_id,) if user_id else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, messages_json, created_at, updated_at
                FROM conversations
                {where}
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_conversation(self, conversation_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        user_filter = "AND user_id = ?" if user_id else ""
        params = (conversation_id, user_id) if user_id else (conversation_id,)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, title, messages_json, created_at, updated_at
                FROM conversations
                WHERE id = ?
                {user_filter}
                """,
                params,
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def upsert_conversation(self, conversation: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
        now = int(time.time() * 1000)
        conversation_id = str(conversation["id"])
        title = str(conversation.get("title") or "新的对话")[:120]
        messages = conversation.get("messages") or []
        created_at = int(conversation.get("createdAt") or conversation.get("created_at") or now)
        updated_at = int(conversation.get("updatedAt") or conversation.get("updated_at") or now)
        messages_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, title, messages_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    title = excluded.title,
                    messages_json = excluded.messages_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, user_id, title, messages_json, created_at, updated_at),
            )
        stored = self.get_conversation(conversation_id, user_id=user_id)
        if stored is None:
            raise RuntimeError("conversation_upsert_failed")
        return stored

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> bool:
        user_filter = "AND user_id = ?" if user_id else ""
        params = (conversation_id, user_id) if user_id else (conversation_id,)
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM conversations WHERE id = ? {user_filter}", params)
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "messages": json.loads(row["messages_json"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
