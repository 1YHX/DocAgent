from docagent.graph import build_graph


def test_graph_generates_when_grade_passes():
    def retrieve(state):
        return {"documents": ["doc"]}

    def grade(state):
        return {"relevant_documents": ["doc"]}

    def decide(state):
        return "generate"

    def generate(state):
        return {"answer": "基于资料的答案"}

    def self_check(state):
        return {"self_check": "supported"}

    graph = build_graph(
        retrieve_node=retrieve,
        grade_node=grade,
        decide_router=decide,
        generate_node=generate,
        self_check_node=self_check,
    )

    result = graph.invoke({"question": "问题", "query": "问题", "retry_count": 0, "history": []})

    assert result["answer"] == "基于资料的答案"
    assert result["self_check"] == "supported"


def test_graph_rewrites_once_then_falls_back():
    def retrieve(state):
        return {"documents": []}

    def grade(state):
        return {"relevant_documents": [], "grades": []}

    def decide(state):
        if state.get("retry_count", 0) < 1:
            return "rewrite"
        return "fallback"

    def rewrite(state):
        return {"retry_count": state.get("retry_count", 0) + 1, "query": "改写后的 query"}

    def fallback(state):
        return {"answer": "资料中未找到相关信息。"}

    graph = build_graph(
        retrieve_node=retrieve,
        grade_node=grade,
        decide_router=decide,
        rewrite_node=rewrite,
        fallback_node=fallback,
    )

    result = graph.invoke({"question": "问题", "query": "问题", "retry_count": 0, "history": []})

    assert result["query"] == "改写后的 query"
    assert result["retry_count"] == 1
    assert result["answer"] == "资料中未找到相关信息。"
