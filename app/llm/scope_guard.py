from __future__ import annotations

import re


OUT_OF_SCOPE_RE = re.compile(
    r"天气|星期|今天几号|现在几点|几点了|当前时间|现在时间|"
    r"你是谁|写一首诗|写诗|讲个笑话|翻译|"
    r"民法|民事|合同纠纷|劳动仲裁|行政处罚法|行政法|婚姻|继承|"
    r"帮我写(?:起诉状|上诉状|辩护词|律师函)|代写(?:起诉状|上诉状|辩护词|律师函)|"
    r"规避法律责任|销毁证据|假证词|骗过法院|作伪证|证据链断掉|伪造.*证据|逃避侦查|威胁证人|藏匿赃物"
)

IN_SCOPE_RE = re.compile(
    r"刑法|刑事|犯罪|罪|案|案例|判决|法条|第.+条|"
    r"量刑|判了|怎么判|判几年|有期徒刑|拘役|罚金|缓刑|死刑|"
    r"杀人|伤人|伤害|盗窃|诈骗|抢劫|入户|入室|过失致人死亡|"
    r"证据|事实|裁判理由|适用|辩称|辩护|构成"
)


def is_out_of_scope_query(query: str) -> bool:
    text = query.strip()
    if not text:
        return False
    if OUT_OF_SCOPE_RE.search(text):
        return True
    return not IN_SCOPE_RE.search(text)


def out_of_scope_response() -> dict:
    return {
        "answer": "该问题超出当前刑法案例/刑法条文样本知识库范围，无法可靠回答。",
        "answer_type": "no_answer",
        "citations": [],
        "graph_paths": [],
        "warnings": ["out_of_scope", "insufficient_context"],
        "confidence": {"level": "none", "score": 0.0, "reason": "out_of_scope"},
        "refusal_reason": "out_of_scope",
        "retrieval_context": [],
    }
