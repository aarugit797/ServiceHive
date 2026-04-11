"""
agent/graph.py
StateGraph definition with 9 nodes, 2 conditional edge functions,
MemorySaver checkpointer for cross-turn memory.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    session_init_node,
    intent_classifier_node,
    greeter_node,
    rag_node,
    response_generator_node,
    lead_collector_node,
    tool_executor_node,
)


def route_by_intent(state: AgentState) -> str:
    """
    Route after intent classification.
    """
    profile = state.get("lead_profile")
    if profile and profile.collection_stage == "collecting":
        return "lead_collector_node"

    intent = state.get("last_intent", "greeting")

    if intent == "greeting":
        return "greeter_node"
    elif intent in ("inquiry_general", "inquiry_specific"):
        return "rag_node"
    elif intent == "hard_lead":
        return "lead_collector_node"

    # Safe default
    return "greeter_node"


def route_after_collection(state: AgentState) -> str:
    """
    Route after lead_collector_node.
    tool_executor_node is ONLY reachable through this function.
    If profile is incomplete → END and wait for the next user turn.
    """
    profile = state.get("lead_profile")
    if profile and profile.is_complete and not state.get("lead_captured"):
        return "tool_executor_node"
    return "__end__"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("session_init_node",session_init_node)
    graph.add_node("intent_classifier_node",intent_classifier_node)
    graph.add_node("greeter_node",greeter_node)
    graph.add_node("rag_node",rag_node)
    graph.add_node("response_generator_node",response_generator_node)
    graph.add_node("lead_collector_node",lead_collector_node)
    graph.add_node("tool_executor_node",tool_executor_node)

    graph.add_edge(START,"session_init_node")
    graph.add_edge("session_init_node","intent_classifier_node")
    graph.add_edge("rag_node","response_generator_node")
    graph.add_edge("response_generator_node",END)
    graph.add_edge("greeter_node",END)
    graph.add_edge("tool_executor_node",END)

    # Conditional edge 1 — after intent classification
    # ONLY hard_lead reaches lead_collector_node
    graph.add_conditional_edges(
        "intent_classifier_node",
        route_by_intent,
        {
            "greeter_node":"greeter_node",
            "rag_node":"rag_node",
            "lead_collector_node":"lead_collector_node",
        }
    )

    # Conditional edge 2 — after lead collection
    # ONLY path to tool_executor_node
    graph.add_conditional_edges(
        "lead_collector_node",
        route_after_collection,
        {
            "tool_executor_node":"tool_executor_node",
            "__end__":END,
        }
    )

    return graph.compile(checkpointer=MemorySaver())


# Singleton compiled graph
compiled_graph = build_graph()
