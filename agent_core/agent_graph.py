import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # Для начала используем память в ОЗУ

from agent_core.agent_state import AgentState  # Наше состояние
from agent_core.nodes import (
    extract_initial_info_node,
    clarify_missing_data_node,
    search_events_node,
    present_initial_plan_node,
    clarify_address_or_build_route_node,
    present_full_plan_node,
    handle_plan_feedback_node,
    confirm_changes_node,
    error_node,
)
from agent_core.conditional_edges import (
    check_initial_data_sufficiency_edge,
    check_event_search_result_edge,
    check_address_needs_clarification_or_route_exists_edge,
    check_plan_feedback_router_edge,
    after_confirm_changes_edge,
)

# from llm_interface.gigachat_client import get_gigachat_client # Не нужен здесь напрямую
# from tools import list_of_all_tools # Инструменты будут привязаны в get_gigachat_client

logger = logging.getLogger(__name__)


def create_agent_graph() -> StateGraph:
    logger.info("Creating agent graph...")
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("extract_initial_info", extract_initial_info_node)
    graph_builder.add_node("clarify_missing_data", clarify_missing_data_node)
    graph_builder.add_node("search_events", search_events_node)
    graph_builder.add_node("present_initial_plan", present_initial_plan_node)
    graph_builder.add_node("build_route", clarify_address_or_build_route_node)
    graph_builder.add_node("present_full_plan", present_full_plan_node)
    graph_builder.add_node("handle_plan_feedback", handle_plan_feedback_node)
    graph_builder.add_node("confirm_changes", confirm_changes_node)
    graph_builder.add_node("error_handler", error_node)
    logger.info("All nodes added.")

    graph_builder.set_entry_point("extract_initial_info")
    logger.info("Entry point set to 'extract_initial_info'.")

    graph_builder.add_conditional_edges(
        "extract_initial_info",
        check_initial_data_sufficiency_edge,
        {
            "clarify_missing_data_node": "clarify_missing_data",
            "search_events_node": "search_events",
            "build_route_node": "build_route",
            END: END,
        },
    )

    graph_builder.add_conditional_edges(
        "search_events",
        check_event_search_result_edge,
        {
            "present_initial_plan_node": "present_initial_plan",
            "error_node": "error_handler",
        },
    )

    graph_builder.add_edge("clarify_missing_data", END)
    graph_builder.add_edge("present_initial_plan", END)
    graph_builder.add_edge("error_handler", END)

    graph_builder.add_edge("build_route", "present_full_plan")
    graph_builder.add_edge("present_full_plan", END)

    graph_builder.add_conditional_edges(
        "handle_plan_feedback",
        check_plan_feedback_router_edge,
        {
            "confirm_changes_node": "confirm_changes",
            "search_events_node": "search_events",
            "clarify_missing_data_node": "clarify_missing_data",
            END: END,
        },
    )

    graph_builder.add_edge("confirm_changes", END)

    logger.info("All edges added. Graph definition complete.")
    return graph_builder


memory_for_checkpoint = MemorySaver()

agent_graph_definition = create_agent_graph()
compiled_agent_graph = agent_graph_definition.compile(
    checkpointer=memory_for_checkpoint
)
logger.info("Agent graph compiled successfully with MemorySaver checkpointer.")


def get_compiled_agent_graph():
    # Эта функция будет использоваться в main.py для получения скомпилированного графа
    return compiled_agent_graph
