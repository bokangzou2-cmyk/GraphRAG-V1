from __future__ import annotations

import app.processing_bridge  # noqa: F401
import langchain_orchestration_sample


def test_build_chain_prefers_langgraph_when_available() -> None:
    chain = langchain_orchestration_sample.build_chain()

    assert langchain_orchestration_sample.StateGraph is not None
    assert type(chain).__name__ == "CompiledStateGraph"


def test_memory_compression_only_for_long_conversations_and_preserves_case_refs() -> None:
    short_messages = [{"role": "user", "content": "盗窃罪怎么量刑"}]
    short_memory = langchain_orchestration_sample.summarize_memory(short_messages)
    short_compressed, short_meta = langchain_orchestration_sample.compress_messages_for_context(short_messages, short_memory)

    assert short_meta["enabled"] is False
    assert short_compressed == short_messages

    long_messages = [
        {"role": "user", "content": "我朋友入户盗窃40000元，有案例参考吗"},
        {
            "role": "assistant",
            "content": "可以看第一个参考案例是《王士东、吉得才盗窃罪一审刑事判决书》，简称王士东案，金额约43852元。",
        },
    ]
    for index in range(5):
        long_messages.extend(
            [
                {"role": "user", "content": f"再说一个普通问题 {index}"},
                {"role": "assistant", "content": f"普通回答 {index}"},
            ]
        )
    long_messages.append({"role": "user", "content": "这个案子判了多久"})

    memory = langchain_orchestration_sample.summarize_memory(long_messages)
    compressed, meta = langchain_orchestration_sample.compress_messages_for_context(long_messages, memory)

    assert meta["enabled"] is True
    assert meta["dropped_message_count"] > 0
    assert len(compressed) < len(long_messages)
    assert compressed[-1]["content"] == "这个案子判了多久"
    assert "结构化上下文记忆" in compressed[-2]["content"]
    assert "王士东" in compressed[-2]["content"]
    assert langchain_orchestration_sample.extract_case_candidates(compressed)[0]["case_title"].startswith("王士东")


def test_langgraph_uses_explicit_answer_branches(monkeypatch) -> None:
    def fake_start(inputs: dict) -> dict:
        route = inputs["route"]
        return {
            "query": inputs.get("query", "吴必定案判了多久"),
            "messages": [{"role": "user", "content": inputs.get("query", "吴必定案判了多久")}],
            "top_k": 1,
            "mock_llm_mode": "grounded",
            "generation_mode": "mock",
            "route_query": inputs.get("query", "吴必定案判了多久"),
            "retrieval_query": inputs.get("query", "吴必定案判了多久"),
            "route": route,
            "enabled_retrievers": [] if not route.needs_retrieval else ["lexical", "graph", "vector"],
            "retrieval_targets": [] if not route.needs_retrieval else ["law", "case", "graph", "vector"],
            "confidence": inputs["confidence"],
            "orchestration_logs": [],
        }

    monkeypatch.setattr(langchain_orchestration_sample, "start_state", fake_start)
    monkeypatch.setattr(langchain_orchestration_sample, "router_state", lambda state: langchain_orchestration_sample.add_log(state, "router_result", state["route"].as_dict()))
    monkeypatch.setattr(langchain_orchestration_sample, "law_retriever_state", lambda state: {**state, "law_result": {"direct_hits": []}})
    monkeypatch.setattr(langchain_orchestration_sample, "case_retriever_state", lambda state: {**state, "case_result": {"direct_hits": []}})
    monkeypatch.setattr(langchain_orchestration_sample, "lexical_retriever_state", lambda state: {**state, "lexical_result": {"direct_hits": []}})
    monkeypatch.setattr(langchain_orchestration_sample, "graph_retriever_state", lambda state: {**state, "graph_result": {"candidates": [], "warnings": []}})
    monkeypatch.setattr(langchain_orchestration_sample, "vector_retriever_state", lambda state: {**state, "vector_result": [], "vector_warnings": []})
    monkeypatch.setattr(langchain_orchestration_sample, "merge_rerank_state", lambda state: {**state, "retrieval_result": {"selected_context": [], "warnings": [], "search_summary": {}}, "merge_confidence": state["confidence"]})

    def fake_evidence(state: dict) -> dict:
        state["evidence_sufficient"] = state["confidence"]["level"] == "high"
        return langchain_orchestration_sample.add_log(state, "evidence_sufficiency", {"sufficient": state["evidence_sufficient"]})

    monkeypatch.setattr(langchain_orchestration_sample, "evidence_sufficiency_state", fake_evidence)
    monkeypatch.setattr(langchain_orchestration_sample, "direct_answer_state", lambda state: langchain_orchestration_sample.add_log({**state, "response": {"answer": "direct", "answer_type": "llm_direct", "citations": [], "graph_paths": [], "warnings": [], "confidence": {}, "refusal_reason": None, "retrieval_context": [], "orchestration_logs": []}}, "direct_answer", {}))
    monkeypatch.setattr(langchain_orchestration_sample, "grounded_answer_state", lambda state: langchain_orchestration_sample.add_log({**state, "response": {"answer": "grounded", "answer_type": "llm_grounded", "citations": [], "graph_paths": [], "warnings": [], "confidence": {}, "refusal_reason": None, "retrieval_context": [], "orchestration_logs": []}}, "grounded_answer", {}))
    monkeypatch.setattr(langchain_orchestration_sample, "fallback_answer_state", lambda state: langchain_orchestration_sample.add_log({**state, "response": {"answer": "fallback", "answer_type": "no_answer", "citations": [], "graph_paths": [], "warnings": [], "confidence": {}, "refusal_reason": "low_confidence", "retrieval_context": [], "orchestration_logs": []}}, "fallback_answer", {}))

    RouteDecision = langchain_orchestration_sample.RouteDecision
    direct = RouteDecision("criminal_law_general_direct", False, ["none"], False, False, False, 0.9, "rules")
    grounded = RouteDecision("criminal_law_case_and_law", True, ["law_articles", "cases"], True, True, False, 0.9, "rules")
    fallback = RouteDecision("criminal_law_case_only", True, ["cases"], True, False, False, 0.9, "rules")

    direct_result = langchain_orchestration_sample.build_chain().invoke({"route": direct, "confidence": {"level": "high"}})
    grounded_result = langchain_orchestration_sample.build_chain().invoke({"route": grounded, "confidence": {"level": "high"}})
    fallback_result = langchain_orchestration_sample.build_chain().invoke({"route": fallback, "confidence": {"level": "low"}})

    assert [item["event"] for item in direct_result["orchestration_logs"] if item["event"] in {"direct_answer", "grounded_answer", "fallback_answer"}] == ["direct_answer"]
    assert [item["event"] for item in grounded_result["orchestration_logs"] if item["event"] in {"direct_answer", "grounded_answer", "fallback_answer"}] == ["grounded_answer"]
    assert [item["event"] for item in fallback_result["orchestration_logs"] if item["event"] in {"direct_answer", "grounded_answer", "fallback_answer"}] == ["fallback_answer"]
