from docagent.graph import build_graph


def test_graph_generates_when_grade_passes():
    def retrieve(state):
        return {"documents": ["doc"]}

    def grade(state):
        return {"relevant_documents": ["doc"]}

    def decide(state):
        return {"route": "generate"}

    def route(state):
        return "generate"

    def generate(state):
        return {"answer": "基于资料的答案"}

    def self_check(state):
        return {"self_check": "supported"}

    graph = build_graph(
        retrieve_node=retrieve,
        grade_node=grade,
        decide_node_fn=decide,
        decide_router=route,
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
            return {"route": "rewrite"}
        return {"route": "fallback"}

    def route(state):
        return state["route"]

    def rewrite(state):
        return {"retry_count": state.get("retry_count", 0) + 1, "query": "改写后的 query"}

    def fallback(state):
        return {"answer": "资料中未找到相关信息。"}

    graph = build_graph(
        retrieve_node=retrieve,
        grade_node=grade,
        decide_node_fn=decide,
        decide_router=route,
        rewrite_node=rewrite,
        fallback_node=fallback,
    )

    result = graph.invoke({"question": "问题", "query": "问题", "retry_count": 0, "history": []})

    assert result["query"] == "改写后的 query"
    assert result["retry_count"] == 1
    assert result["answer"] == "资料中未找到相关信息。"


def test_graph_revises_unsupported_answer_once():
    checks = []

    def retrieve(state):
        return {"documents": ["doc"]}

    def grade(state):
        return {"relevant_documents": ["doc"]}

    def decide(state):
        return {"route": "generate"}

    def route(state):
        return state["route"]

    def generate(state):
        return {"answer": "错误答案"}

    def self_check(state):
        checks.append(state["answer"])
        if state["answer"] == "错误答案":
            return {"self_check": "unsupported。遗漏了项目。"}
        return {"self_check": "supported。已修正。"}

    def revise(state):
        return {"answer": "修正答案", "revised": True}

    graph = build_graph(
        retrieve_node=retrieve,
        grade_node=grade,
        decide_node_fn=decide,
        decide_router=route,
        generate_node=generate,
        self_check_node=self_check,
        revise_node=revise,
    )

    result = graph.invoke({"question": "问题", "query": "问题", "retry_count": 0, "history": []})

    assert checks == ["错误答案", "修正答案"]
    assert result["answer"] == "修正答案"
    assert result["self_check"] == "supported。已修正。"
