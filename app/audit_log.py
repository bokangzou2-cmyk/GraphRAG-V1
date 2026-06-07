from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class AnswerAuditStore:
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
                CREATE TABLE IF NOT EXISTS answer_runs (
                    id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    user_id TEXT,
                    path TEXT NOT NULL,
                    query TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    answer_type TEXT NOT NULL,
                    confidence_level TEXT,
                    confidence_score REAL,
                    refusal_reason TEXT,
                    warnings_json TEXT NOT NULL,
                    citation_ids_json TEXT NOT NULL,
                    graph_path_count INTEGER NOT NULL,
                    retrieval_context_count INTEGER NOT NULL,
                    answer_preview TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(answer_runs)").fetchall()}
            if "user_id" not in columns:
                conn.execute("ALTER TABLE answer_runs ADD COLUMN user_id TEXT")
            if "metrics_json" not in columns:
                conn.execute("ALTER TABLE answer_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_runs_created_at ON answer_runs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_runs_path ON answer_runs(path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_runs_user_id ON answer_runs(user_id)")

    def record_run(self, *, path: str, query: str, top_k: int, response: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        created_at = int(time.time() * 1000)
        confidence = response.get("confidence") or {}
        warnings = response.get("warnings") or []
        citation_ids = [
            citation.get("retrieval_id")
            for citation in response.get("citations", [])
            if citation.get("retrieval_id")
        ]
        graph_path_count = len(response.get("graph_paths") or [])
        retrieval_context_count = len(response.get("retrieval_context") or [])
        answer_preview = str(response.get("answer") or "")[:240]
        metrics = response.get("metrics") or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO answer_runs (
                    id, created_at, user_id, path, query, top_k, answer_type,
                    confidence_level, confidence_score, refusal_reason,
                    warnings_json, citation_ids_json, graph_path_count,
                    retrieval_context_count, answer_preview, metrics_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    user_id,
                    path,
                    query,
                    int(top_k),
                    str(response.get("answer_type") or ""),
                    confidence.get("level"),
                    confidence.get("score"),
                    response.get("refusal_reason"),
                    json.dumps(warnings, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(citation_ids, ensure_ascii=False, separators=(",", ":")),
                    graph_path_count,
                    retrieval_context_count,
                    answer_preview,
                    json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        stored = self.get_run(run_id)
        if stored is None:
            raise RuntimeError("answer_audit_record_failed")
        return stored

    def list_runs(self, limit: int = 50, user_id: str | None = None) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        where = "WHERE user_id = ?" if user_id else ""
        params = (user_id, safe_limit) if user_id else (safe_limit,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, created_at, user_id, path, query, top_k, answer_type,
                    confidence_level, confidence_score, refusal_reason,
                    warnings_json, citation_ids_json, graph_path_count,
                    retrieval_context_count, answer_preview, metrics_json
                FROM answer_runs
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, user_id, path, query, top_k, answer_type,
                    confidence_level, confidence_score, refusal_reason,
                    warnings_json, citation_ids_json, graph_path_count,
                    retrieval_context_count, answer_preview, metrics_json
                FROM answer_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "createdAt": row["created_at"],
            "userId": row["user_id"],
            "path": row["path"],
            "query": row["query"],
            "topK": row["top_k"],
            "answerType": row["answer_type"],
            "confidenceLevel": row["confidence_level"],
            "confidenceScore": row["confidence_score"],
            "refusalReason": row["refusal_reason"],
            "warnings": json.loads(row["warnings_json"]),
            "citationIds": json.loads(row["citation_ids_json"]),
            "graphPathCount": row["graph_path_count"],
            "retrievalContextCount": row["retrieval_context_count"],
            "answerPreview": row["answer_preview"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
        }
