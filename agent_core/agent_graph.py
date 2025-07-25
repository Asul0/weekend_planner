# Файл: agent_core/agent_graph.py

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # Используем простой MemorySaver для начала

from agent_core.agent_state import AgentState  # Наше состояние
from agent_core.nodes import (  # Импортируем все наши узлы
    extract_initial_info_node,
    clarify_missing_data_node,
    search_events_node,
    present_initial_plan_node,
    clarify_address_or_build_route_node,  # Этот узел теперь строит маршрут
    present_full_plan_node,
    handle_plan_feedback_node,
    confirm_changes_node,  # Оставляем пока, если он есть в nodes.py
    error_node,
)
from agent_core.conditional_edges import (  # Импортируем наши условные ребра
    check_initial_data_sufficiency_edge,
    check_event_search_result_edge,
    # check_address_needs_clarification_or_route_exists_edge, # Решили, что это ребро, скорее всего, не нужно
    check_plan_feedback_router_edge,
    # after_confirm_changes_edge, # Решили, что это ребро не используется, т.к. confirm_changes_node ведет в END
)

logger = logging.getLogger(__name__)


def create_agent_graph() -> StateGraph:
    logger.info("Creating agent graph...")
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("extract_initial_info", extract_initial_info_node)
    graph_builder.add_node("clarify_missing_data", clarify_missing_data_node)
    graph_builder.add_node("search_events", search_events_node)
    graph_builder.add_node("present_initial_plan", present_initial_plan_node)
    graph_builder.add_node(
        "clarify_address_or_build_route", clarify_address_or_build_route_node
    )
    graph_builder.add_node("present_full_plan", present_full_plan_node)
    graph_builder.add_node("handle_plan_feedback", handle_plan_feedback_node)
    graph_builder.add_node("error_handler", error_node)

    logger.info("All nodes added to graph builder.")
    graph_builder.set_entry_point("extract_initial_info")
    logger.info("Entry point set to 'extract_initial_info'.")

    graph_builder.add_conditional_edges(
        "extract_initial_info",
        check_initial_data_sufficiency_edge,
        {
            "clarify_missing_data": "clarify_missing_data",
            "search_events": "search_events",
            "handle_plan_feedback": "handle_plan_feedback",
            "present_initial_plan": "present_initial_plan",
            "clarify_address_or_build_route": "clarify_address_or_build_route",
            "present_full_plan": "present_full_plan",
            "error_handler": "error_handler",
            END: END,
        },
    )

    graph_builder.add_conditional_edges(
        "search_events",
        check_event_search_result_edge,
        {
            "present_initial_plan": "present_initial_plan",
            "clarify_address_or_build_route": "clarify_address_or_build_route",
            "error_handler": "error_handler",
        },
    )

    graph_builder.add_conditional_edges(
        "handle_plan_feedback",
        check_plan_feedback_router_edge,
        {
            "search_events": "search_events",
            "clarify_missing_data": "clarify_missing_data",
            "extract_initial_info": "extract_initial_info",
            "present_initial_plan": "present_initial_plan",
            "error_handler": "error_handler",
            END: END,
        },
    )

    graph_builder.add_edge("clarify_missing_data", END)
    graph_builder.add_edge("present_initial_plan", END)
    graph_builder.add_edge("error_handler", END)

    graph_builder.add_edge("clarify_address_or_build_route", "present_full_plan")
    graph_builder.add_edge("present_full_plan", END)

    logger.info("All edges defined. Agent graph definition complete.")
    return graph_builder


# Компиляция графа (остается как было в вашем graph.txt)
memory_for_checkpoint = MemorySaver()
agent_graph_definition = create_agent_graph()

# Защита от рекомпиляции при многократных импортах get_compiled_agent_graph
_COMPILED_AGENT_GRAPH = None


def get_compiled_agent_graph():
    global _COMPILED_AGENT_GRAPH
    if _COMPILED_AGENT_GRAPH is None:
        logger.info("Compiling agent graph for the first time...")
        memory_for_checkpoint = MemorySaver()
        agent_graph_definition = create_agent_graph()
        _COMPILED_AGENT_GRAPH = agent_graph_definition.compile(
            checkpointer=memory_for_checkpoint
        )
        logger.info("Agent graph compiled successfully with MemorySaver checkpointer.")
    else:
        logger.debug("Returning already compiled agent graph.")
    return _COMPILED_AGENT_GRAPH
