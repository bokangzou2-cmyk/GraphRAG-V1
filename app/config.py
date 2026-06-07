from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    chat_history_db_path: Path = Path(os.getenv("CHAT_HISTORY_DB_PATH", str(ROOT / "processing_data" / "chat_sessions.sqlite3")))
    auth_bootstrap_username: str = os.getenv("AUTH_BOOTSTRAP_USERNAME", "admin")
    auth_bootstrap_password: str = os.getenv("AUTH_BOOTSTRAP_PASSWORD", "1")
    auth_session_ttl_hours: int = int(os.getenv("AUTH_SESSION_TTL_HOURS", "24"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("MINIMAX_API_KEY", "")))
    llm_base_url: str = os.getenv("LLM_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
    llm_model: str = os.getenv("LLM_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60")))
    llm_grounded_max_tokens: int = int(os.getenv("LLM_GROUNDED_MAX_TOKENS", "2400"))
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    minimax_base_url: str = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1").rstrip("/")
    minimax_model: str = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
    minimax_timeout_seconds: float = float(os.getenv("MINIMAX_TIMEOUT_SECONDS", "60"))
    answer_use_llm: bool = os.getenv("ANSWER_USE_LLM", "true").lower() in {"1", "true", "yes", "on"}
    router_mode: str = os.getenv("ROUTER_MODE", "lora").lower()
    router_base_model: Path = Path(os.getenv("ROUTER_BASE_MODEL", str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")))
    router_adapter_dir: Path = Path(os.getenv("ROUTER_ADAPTER_DIR", str(ROOT / "processing_data" / "router_lora" / "qwen2_5_1_5b_router_lora_v2_lora_processed")))
    router_max_new_tokens: int = int(os.getenv("ROUTER_MAX_NEW_TOKENS", "128"))
    query_understanding_mode: str = os.getenv("QUERY_UNDERSTANDING_MODE", "disabled").lower()
    query_understanding_base_model: Path = Path(os.getenv("QUERY_UNDERSTANDING_BASE_MODEL", str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")))
    query_understanding_adapter_dir: Path = Path(
        os.getenv(
            "QUERY_UNDERSTANDING_ADAPTER_DIR",
            str(ROOT / "processing_data" / "query_understanding_lora" / "qwen2_5_1_5b_qu_lora_5000_v2"),
        )
    )
    query_understanding_max_new_tokens: int = int(os.getenv("QUERY_UNDERSTANDING_MAX_NEW_TOKENS", "512"))
    query_rewriter_mode: str = os.getenv("QUERY_REWRITER_MODE", "lora").lower()
    query_rewriter_base_model: Path = Path(os.getenv("QUERY_REWRITER_BASE_MODEL", str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")))
    query_rewriter_adapter_dir: Path = Path(
        os.getenv(
            "QUERY_REWRITER_ADAPTER_DIR",
            str(ROOT / "processing_data" / "query_rewriter_lora" / "qwen2_5_1_5b_rewriter_lora_gpt_v4_conservative_len768"),
        )
    )
    query_rewriter_max_new_tokens: int = int(os.getenv("QUERY_REWRITER_MAX_NEW_TOKENS", "384"))
    use_langchain_orchestration: bool = os.getenv("USE_LANGCHAIN_ORCHESTRATION", "true").lower() in {"1", "true", "yes", "on"}


settings = Settings()
