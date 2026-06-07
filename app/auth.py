from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings


TOKEN_BYTES = 32
PASSWORD_ITERATIONS = 210_000
bearer_scheme = HTTPBearer(auto_error=False)


class AuthStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.ensure_bootstrap_admin()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_login_at INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON auth_sessions(expires_at)")

    def ensure_bootstrap_admin(self) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            row = conn.execute(
                """
                SELECT id, password_hash
                FROM users
                WHERE username = ?
                """,
                (settings.auth_bootstrap_username,),
            ).fetchone()
        if count == 0:
            self.create_user(settings.auth_bootstrap_username, settings.auth_bootstrap_password, role="admin")
        elif row is None:
            self.create_user(settings.auth_bootstrap_username, settings.auth_bootstrap_password, role="admin")
        elif not verify_password(settings.auth_bootstrap_password, row["password_hash"]):
            self.set_user_password(settings.auth_bootstrap_username, settings.auth_bootstrap_password)

    def create_user(self, username: str, password: str, role: str = "user") -> dict[str, Any]:
        now = int(time.time() * 1000)
        user_id = secrets.token_urlsafe(16)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, username.strip(), hash_password(password), role, now),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("user_create_failed")
        return user

    def set_user_password(self, username: str, password: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?
                WHERE username = ?
                """,
                (hash_password(password), username.strip()),
            )

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, role, created_at, last_login_at
                FROM users
                WHERE username = ?
                """,
                (username.strip(),),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        user = self._user_row_to_dict(row)
        with self._connect() as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (int(time.time() * 1000), user["id"]))
        return user

    def create_session(self, user_id: str) -> dict[str, Any]:
        now = int(time.time() * 1000)
        expires_at = now + settings.auth_session_ttl_hours * 60 * 60 * 1000
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_id, now, expires_at),
            )
        return {"token": token, "expiresAt": expires_at}

    def user_for_token(self, token: str) -> dict[str, Any] | None:
        now = int(time.time() * 1000)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.role, u.created_at, u.last_login_at
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ? AND s.revoked_at IS NULL AND s.expires_at > ?
                """,
                (token, now),
            ).fetchone()
        return self._user_row_to_dict(row) if row else None

    def revoke_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE token = ?", (int(time.time() * 1000), token))

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, role, created_at, last_login_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._user_row_to_dict(row) if row else None

    @staticmethod
    def _user_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "createdAt": row["created_at"],
            "lastLoginAt": row["last_login_at"],
        }


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


auth_store = AuthStore(settings.chat_history_db_path)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="auth_required")
    user = auth_store.user_for_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_or_expired_token")
    return user
