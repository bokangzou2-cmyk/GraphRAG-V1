from __future__ import annotations

from app.llm.query_rewriter import complete_rewriter_payload


CASE_HISTORY = [
    {"role": "user", "content": "我朋友入户盗窃40000元，有案例参考吗"},
    {
        "role": "assistant",
        "content": (
            "可以看两个参考案例。第一个参考案例是《王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书》，"
            "简称王士东案，金额约43852元。第二个参考案例是《刘方祥、刘礼芬盗窃一审刑事判决书》，简称刘方祥案，金额约39136元。"
        ),
    },
]


def test_rewriter_completion_adds_second_case_for_two_case_compare() -> None:
    messages = CASE_HISTORY + [{"role": "user", "content": "前面两个案例量刑差异在哪里"}]
    raw_payload = {
        "standalone_query": "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书 相似案例 量刑",
        "resolved_case_refs": [
            {
                "ref": "前面两个案例量刑差异在哪里",
                "index": 1,
                "case_title": "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书",
                "source": "assistant_history",
            }
        ],
        "case_name_mentions": ["王士东案"],
        "crime_mentions": ["盗窃罪"],
        "amount_constraints": [],
        "field_intents": ["judgment_text"],
        "rewrite_required": True,
        "confidence": 0.9,
        "warnings": [],
    }

    completed = complete_rewriter_payload(raw_payload, messages)

    standalone = completed["standalone_query"]
    assert "王士东" in standalone
    assert "刘方祥" in standalone
    assert "差异" in standalone
    assert [item["index"] for item in completed["resolved_case_refs"]] == [1, 2]
    assert any("刘方祥" in item for item in completed["case_name_mentions"])
    assert {"reasoning_text", "judgment_text"} <= set(completed["field_intents"])


def test_rewriter_completion_normalizes_amount_and_crime_alias() -> None:
    messages = [{"role": "user", "content": "入室盗窃40000元一般怎么判，有案例吗"}]
    raw_payload = {
        "standalone_query": "检索入户盗窃40000元的罪名认定、构成要件、量刑标准及相关案例。",
        "resolved_case_refs": [],
        "case_name_mentions": [],
        "crime_mentions": ["入户盗窃"],
        "amount_constraints": [{"amount_text": "40000元", "normalized_amount_cny": 40000}],
        "field_intents": [],
        "rewrite_required": True,
        "confidence": 0.8,
        "warnings": [],
    }

    completed = complete_rewriter_payload(raw_payload, messages)

    assert "入户盗窃" in completed["standalone_query"]
    assert "盗窃罪" in completed["standalone_query"]
    assert "盗窃罪" in completed["crime_mentions"]
    assert completed["amount_constraints"] == [
        {"value": 40000, "role": "theft_amount_total_or_single", "raw_text": "40000元"}
    ]
