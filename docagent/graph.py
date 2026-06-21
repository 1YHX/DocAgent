from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from docagent.nodes import (
    decide,
    decide_node,
    fallback,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
    route_from_state,
    self_check,
)
from docagent.state import AgentState, Route


Node = Callable[[AgentState], AgentState]
Router = Callable[[AgentState], Route]


def build_graph(
    retrieve_node: Node = retrieve,
    grade_node: Node = grade_documents,
    decide_node_fn: Node = decide_node,
    decide_router: Router = route_from_state,
    rewrite_node: Node = rewrite_query,
    generate_node: Node = generate,
    fallback_node: Node = fallback,
    self_check_node: Node = self_check,
):
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("decide", decide_node_fn)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("self_check", self_check_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "decide")
    graph.add_conditional_edges(
        "decide",
        decide_router,
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
