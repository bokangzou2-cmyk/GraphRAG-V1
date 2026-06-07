from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import app.auth as auth_module
import app.api.routes.auth as auth_route
import app.llm.chat_service as chat_service
import app.api.routes.chat as chat_route
import app.api.routes.conversations as conversations_route
from app.auth import AuthStore
from app.audit_log import AnswerAuditStore
from app.chat_history import ChatHistoryStore
from app.main import app
from app.llm.scope_guard import is_out_of_scope_query


@pytest.fixture(autouse=True)
def isolated_auth_store(tmp_path, monkeypatch) -> None:
    store = AuthStore(tmp_path / "auth.sqlite3")
    monkeypatch.setattr(auth_module, "auth_store", store)
    monkeypatch.setattr(auth_route, "auth_store", store)


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "1"})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_scope_guard_keeps_case_time_in_scope() -> None:
    assert not is_out_of_scope_query("吴必定案案发时间是什么")
    assert is_out_of_scope_query("现在几点了？")


def test_chat_returns_grounded_shape() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["citations"]
    assert data["confidence"]["level"] in {"low", "medium", "high"}
    assert data["confidence"]["query_type"] == "sentence"
    assert isinstance(data["graph_paths"], list)
    assert "refusal_reason" in data
    assert all(citation.get("retrieval_id") for citation in data["citations"])
    assert all(citation.get("text_hash") for citation in data["citations"] if citation.get("source_type") == "case_chunk")


def test_chat_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "auth_required"


def test_retrieval_search_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post("/api/retrieval/search", json={"query": "吴必定案判了多久", "top_k": 3})
    assert response.status_code == 401


def test_retrieval_search_allows_authenticated_user() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/retrieval/search",
        json={"query": "吴必定案判了多久", "top_k": 3},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    assert response.json()["selected_context"]


def test_chat_records_audit_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chat_route, "audit_store", AnswerAuditStore(tmp_path / "chat_sessions.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200

    audit_response = client.get("/api/chat/audit-runs?limit=5", headers=auth_headers(client))
    assert audit_response.status_code == 200
    runs = audit_response.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["path"] == "stable"
    assert runs[0]["query"] == "吴必定案判了多久"
    assert runs[0]["answerType"] == response.json()["answer_type"]
    assert runs[0]["confidenceLevel"] == response.json()["confidence"]["level"]
    assert runs[0]["citationIds"]
    assert runs[0]["retrievalContextCount"] > 0


def test_chat_missing_case_name_evidence_is_low_confidence() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "不存在案判了多久"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "low_confidence"
    assert data["confidence"]["level"] == "low"
    assert data["confidence"]["query_type"] == "sentence"
    assert data["confidence"]["reason"] == "missing_required_evidence:case_name_match,intent_field_match"
    checks = {item["name"]: item for item in data["confidence"]["evidence_checks"]}
    assert checks["case_name_match"]["passed"] is False
    assert checks["intent_field_match"]["passed"] is False
    assert "low_confidence" in data["warnings"]


def test_chat_empty_query_refuses() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "  "}], "top_k": 5}, headers=auth_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "insufficient_context"
    assert "empty_query" in data["warnings"]
    assert data["citations"] == []


def test_chat_general_query_no_longer_uses_criminal_retrieval(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "天气问题需要提供城市或允许联网查询。"

    monkeypatch.setattr(chat_service, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "天气怎么样？"}], "top_k": 5}, headers=auth_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "out_of_scope"
    assert data["citations"] == []
    assert data["retrieval_context"] == []


def test_chat_general_query_skips_retrieval_and_uses_direct_llm(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "天气问题需要提供城市或允许联网查询。"

    monkeypatch.setattr(chat_service, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "今天天气怎么样？"}], "top_k": 5}, headers=auth_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["citations"] == []
    assert data["retrieval_context"] == []
    assert data["refusal_reason"] == "out_of_scope"


def test_chat_law_only_router_filters_case_context() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "盗窃罪怎么判？"}], "top_k": 5}, headers=auth_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"]["router"]["label"] == "criminal_law_law_only"
    assert data["retrieval_context"]
    assert {row["source_type"] for row in data["retrieval_context"]} <= {"law_article"}


def test_chat_criminal_law_general_direct_skips_retrieval(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "现行刑法条文数量可能随修法和统计口径变化，应以最新官方文本为准。"

    monkeypatch.setattr(chat_service, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "刑法一共有多少条"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "llm_direct"
    assert data["confidence"]["router"]["label"] == "criminal_law_general_direct"
    assert data["retrieval_context"] == []
    assert data["citations"] == []
    assert "retrieval_skipped" in data["warnings"]


def test_chat_case_router_keeps_case_context() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5}, headers=auth_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"]["router"]["label"] in {"criminal_law_case_only", "criminal_law_case_and_law", "criminal_law_multi_case"}
    assert data["retrieval_context"]
    assert any(row["source_type"] == "case_chunk" for row in data["retrieval_context"])


def test_chat_invalid_llm_json_falls_back(monkeypatch) -> None:
    class BadClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages):
            return "不是 JSON"

    monkeypatch.setattr(chat_service, "MinimaxClient", BadClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "retrieval_fallback"
    assert "llm_invalid_json" in data["warnings"]
    assert data["citations"]


def test_chat_llm_no_answer_clears_citations(monkeypatch) -> None:
    class NoAnswerClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages):
            return (
                '{"answer":"检索上下文不足，无法可靠回答。",'
                '"answer_type":"no_answer",'
                '"citation_ids":[],'
                '"warnings":["insufficient_context"],'
                '"refusal_reason":"insufficient_context"}'
            )

    monkeypatch.setattr(chat_service, "MinimaxClient", NoAnswerClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "insufficient_context"
    assert "insufficient_context" in data["warnings"]
    assert "provider_misrefused" in data["warnings"]
    assert data["citations"] == []
    assert data["retrieval_context"] == []


def test_chat_orchestration_returns_logs() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "吴必定案判了多久"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "llm_grounded"
    assert data["citations"]
    assert data["retrieval_context"]
    assert data["orchestration_logs"]
    events = [item["event"] for item in data["orchestration_logs"]]
    assert "router_result" in events
    assert "lexical_retriever" in events
    assert "graph_retriever" in events
    assert "vector_retriever" in events
    assert "hybrid_merge_rerank" in events
    assert "answer_contract_prepared" in events
    assert "mock_llm_output" in events
    assert all(citation["retrieval_id"] in {row["retrieval_id"] for row in data["retrieval_context"]} for citation in data["citations"])
    prepared = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "answer_contract_prepared")
    assert prepared["confidence"]["query_type"] == "sentence"
    assert prepared["confidence"]["evidence_checks"]
    merge_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "hybrid_merge_rerank")
    assert {"router_result", "enabled_retrievers", "lexical_count", "graph_count", "vector_count", "final_context_count", "confidence", "warnings"} <= set(merge_log)


def test_chat_orchestration_general_query_skips_retrieval(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "天气问题需要提供城市或允许联网查询。"

    import langchain_orchestration_sample

    monkeypatch.setattr(langchain_orchestration_sample, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={"messages": [{"role": "user", "content": "天气怎么样？"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["retrieval_context"] == []
    assert data["citations"] == []
    assert data["refusal_reason"] == "out_of_scope"


def test_chat_orchestration_non_criminal_legal_skips_criminal_retrieval(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "这是非刑法法律问题，不检索刑法库。"

    import langchain_orchestration_sample

    monkeypatch.setattr(langchain_orchestration_sample, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={"messages": [{"role": "user", "content": "离婚财产怎么分？"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_context"] == []
    merge_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "hybrid_merge_rerank")
    assert merge_log["router_result"]["label"] == "non_criminal_legal"
    assert merge_log["final_context_count"] == 0


def test_chat_orchestration_law_question_returns_only_law_articles() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={"messages": [{"role": "user", "content": "盗窃罪怎么判？"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_context"]
    assert {row["source_type"] for row in data["retrieval_context"]} <= {"law_article"}
    merge_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "hybrid_merge_rerank")
    assert merge_log["router_result"]["label"] == "criminal_law_law_only"


def test_chat_orchestration_followup_uses_previous_crime_context() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [
                {"role": "user", "content": "盗窃罪怎么判"},
                {"role": "assistant", "content": "根据刑法第264条，盗窃罪分为不同量刑档次。"},
                {"role": "user", "content": "我一个朋友入室盗窃了3000元，一般会怎么判"},
            ],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] in {"llm_grounded", "retrieval_fallback"}
    assert data["confidence"]["level"] in {"medium", "high"}
    assert data["citations"]
    assert data["citations"][0]["article_no"] == "264"
    start_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "start")
    assert "盗窃罪" in start_log["retrieval_query"]
    assert "入室盗窃" in start_log["retrieval_query"]
    assert "入户盗窃" in start_log["retrieval_query"]


def test_chat_orchestration_similar_case_followup_inherits_prior_facts() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [
                {"role": "user", "content": "我一个朋友入室盗窃了3000元，一般会怎么判"},
                {"role": "assistant", "content": "根据刑法第264条，入户盗窃一般处三年以下有期徒刑、拘役或者管制，并处或者单处罚金。"},
                {"role": "user", "content": "没有类似案件参考吗"},
            ],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] in {"llm_grounded", "retrieval_fallback"}
    assert data["confidence"]["level"] in {"medium", "high"}
    assert data["citations"]
    assert data["citations"][0]["title"] == "刘方祥、刘礼芬盗窃一审刑事判决书"
    start_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "start")
    assert "盗窃罪" in start_log["route_query"]
    assert "入室盗窃" in start_log["retrieval_query"]
    assert "入户盗窃" in start_log["retrieval_query"]
    assert "3000元" in start_log["retrieval_query"]


def test_chat_orchestration_amount_semantics_keep_theft_amount_separate_from_fine() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "入户盗窃40000元怎么判，有案例参考吗"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_context"]
    top = data["retrieval_context"][0]
    assert top["title"] in {
        "刘方祥、刘礼芬盗窃一审刑事判决书",
        "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书",
    }
    assert top["amount_diagnostics"]["matched_amount_role"] in {"theft_amount_total", "theft_amount_single"}
    assert top["amount_diagnostics"]["matched_amount_role"] != "fine_amount"
    assert data["citations"][0]["amount_diagnostics"]["matched_amount_role"] in {"theft_amount_total", "theft_amount_single"}


def test_chat_orchestration_fine_amount_uses_fine_role() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "罚金40000元的盗窃案例"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_context"][0]["amount_diagnostics"]["matched_amount_role"] == "fine_amount"
    assert data["retrieval_context"][0]["amount_diagnostics"]["matched_amount_value"] == 40000


def test_chat_orchestration_case_question_returns_only_case_chunks() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={"messages": [{"role": "user", "content": "吴必定案案情是什么？"}], "top_k": 5},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_context"]
    assert {row["source_type"] for row in data["retrieval_context"]} <= {"case_chunk"}
    assert data["graph_paths"]
    merge_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "hybrid_merge_rerank")
    assert merge_log["router_result"]["label"] in {"criminal_law_case_only", "criminal_law_multi_case"}


def test_chat_orchestration_case_and_law_question_returns_both_sources() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={"messages": [{"role": "user", "content": "吴必定案判了多久，依据什么法条？"}], "top_k": 8},
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    source_types = {row["source_type"] for row in data["retrieval_context"]}
    assert "case_chunk" in source_types
    assert "law_article" in source_types
    assert data["graph_paths"]
    merge_log = next(item["payload"] for item in data["orchestration_logs"] if item["event"] == "hybrid_merge_rerank")
    assert merge_log["router_result"]["label"] == "criminal_law_case_and_law"


def test_chat_orchestration_records_audit_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chat_route, "audit_store", AnswerAuditStore(tmp_path / "chat_sessions.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "吴必定案判了多久"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200

    audit_response = client.get("/api/chat/audit-runs?limit=5", headers=auth_headers(client))
    assert audit_response.status_code == 200
    runs = audit_response.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["path"] == "orchestration"
    assert runs[0]["query"] == "吴必定案判了多久"
    assert runs[0]["answerType"] == "llm_grounded"
    assert runs[0]["citationIds"]


def test_chat_orchestration_low_confidence_skips_llm() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "不存在案判了多久"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "low_confidence"
    assert data["citations"] == []
    assert data["retrieval_context"] == []
    assert data["confidence"]["reason"] == "missing_required_evidence:case_name_match,intent_field_match"
    events = [item["event"] for item in data["orchestration_logs"]]
    assert "mock_llm_output" not in events
    assert "llm_skipped" in events


def test_chat_orchestration_invalid_json_falls_back() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "吴必定案判了多久"}],
            "top_k": 5,
            "mock_llm_mode": "invalid_json",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "retrieval_fallback"
    assert "llm_invalid_json" in data["warnings"]
    assert data["citations"]


def test_chat_orchestration_provider_without_key_falls_back(monkeypatch) -> None:
    class DisabledClient:
        def enabled(self) -> bool:
            return False

    import langchain_orchestration_sample

    monkeypatch.setattr(langchain_orchestration_sample, "MinimaxClient", DisabledClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "吴必定案判了多久"}],
            "top_k": 5,
            "generation_mode": "provider",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "retrieval_fallback"
    assert "llm_skipped_by_confidence_gate" in data["warnings"]
    events = [item["event"] for item in data["orchestration_logs"]]
    assert "llm_skipped" in events


def test_chat_orchestration_general_prompt_logs_retrieval_skip(monkeypatch) -> None:
    class DirectClient:
        def enabled(self) -> bool:
            return True

        async def chat(self, messages, temperature=0.2, max_tokens=1200):
            return "可以写诗，但不需要检索刑法库。"

    import langchain_orchestration_sample

    monkeypatch.setattr(langchain_orchestration_sample, "MinimaxClient", DirectClient)
    client = TestClient(app)
    response = client.post(
        "/api/chat/orchestration",
        json={
            "messages": [{"role": "user", "content": "帮我写一首诗"}],
            "top_k": 5,
            "mock_llm_mode": "grounded",
        },
        headers=auth_headers(client),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_type"] == "no_answer"
    assert data["refusal_reason"] == "out_of_scope"
    assert data["citations"] == []
    assert data["retrieval_context"] == []


def test_conversation_history_persists_to_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversations_route, "store", ChatHistoryStore(tmp_path / "chat_sessions.sqlite3"))
    client = TestClient(app)
    payload = {
        "id": "conv-test",
        "title": "吴必定案",
        "createdAt": 1,
        "updatedAt": 2,
        "messages": [
            {"role": "user", "content": "吴必定案判了多久"},
            {"role": "assistant", "content": "有期徒刑十年", "response": {"answer": "有期徒刑十年"}},
        ],
    }
    headers = auth_headers(client)
    response = client.put("/api/conversations/conv-test", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["title"] == "吴必定案"

    listed = client.get("/api/conversations", headers=headers)
    assert listed.status_code == 200
    conversations = listed.json()["conversations"]
    assert len(conversations) == 1
    assert conversations[0]["id"] == "conv-test"
    assert conversations[0]["messages"][0]["content"] == "吴必定案判了多久"


def test_conversation_history_delete_removes_sqlite_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversations_route, "store", ChatHistoryStore(tmp_path / "chat_sessions.sqlite3"))
    client = TestClient(app)
    headers = auth_headers(client)
    payload = {
        "id": "conv-delete",
        "title": "待删除对话",
        "createdAt": 1,
        "updatedAt": 2,
        "messages": [{"role": "user", "content": "盗窃罪怎么判"}],
    }
    created = client.put("/api/conversations/conv-delete", json=payload, headers=headers)
    assert created.status_code == 200

    deleted = client.delete("/api/conversations/conv-delete", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    listed = client.get("/api/conversations", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["conversations"] == []
