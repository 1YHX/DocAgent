from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from docagent.nodes import decide, fallback, generate, grade_documents, retrieve, rewrite_query, self_check
from docagent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_node("self_check", self_check)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        decide,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "self_check")
    graph.add_edge("self_check", END)
    graph.add_edge("fallback", END)

    return graph.compile()
