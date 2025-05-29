import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_core.agent_state import AgentState
from agent_core.nodes import (
    extract_initial_info_node,
    clarify_missing_data_node,
    gather_all_candidate_events_node,  # Новый узел
    optimal_chain_constructor_node,  # Новый узел
    present_initial_plan_node,
    clarify_address_or_build_route_node,  # Убедимся, что имя соответствует вашему (build_route)
    present_full_plan_node,
    handle_plan_feedback_node,
    confirm_changes_node,  # Этот узел пока остается, хотя его использование может быть редким
    error_node,
    search_events_node,  # Оставляем определение узла, но не используем в основном потоке
)
from agent_core.conditional_edges import (
    check_initial_data_sufficiency_edge,  # Обновленное ребро
    check_gathered_candidates_edge,  # Новое ребро
    check_optimal_chain_result_edge,  # Новое ребро (заменяет старое check_event_search_result_edge по смыслу)
    check_plan_feedback_router_edge,
    # check_address_needs_clarification_or_route_exists_edge # Это ребро может не понадобиться, если present_initial_plan -> END
)

logger = logging.getLogger(__name__)


def create_agent_graph() -> StateGraph:
    logger.info("Creating agent graph with new optimal planning logic...")
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("extract_initial_info", extract_initial_info_node)
    graph_builder.add_node("clarify_missing_data", clarify_missing_data_node)
    graph_builder.add_node("gather_all_candidates", gather_all_candidate_events_node)
    graph_builder.add_node("construct_optimal_chain", optimal_chain_constructor_node)
    graph_builder.add_node("present_initial_plan", present_initial_plan_node)
    graph_builder.add_node("build_route", clarify_address_or_build_route_node)
    graph_builder.add_node("present_full_plan", present_full_plan_node)
    graph_builder.add_node("handle_plan_feedback", handle_plan_feedback_node)
    graph_builder.add_node("confirm_changes", confirm_changes_node)
    graph_builder.add_node("error_handler", error_node)
    # search_events_node не добавляем в основной поток активно, но он может быть вызван в будущем для fallback
    # graph_builder.add_node("legacy_search_events", search_events_node)

    logger.info("All nodes (including new ones) added to graph builder.")

    graph_builder.set_entry_point("extract_initial_info")
    logger.info("Entry point set to 'extract_initial_info'.")

    graph_builder.add_conditional_edges(
        "extract_initial_info",
        check_initial_data_sufficiency_edge,
        {
            "clarify_missing_data_node": "clarify_missing_data",
            "gather_all_candidate_events_node": "gather_all_candidates",  # <--- ИЗМЕНЕНИЕ
            "clarify_address_or_build_route_node": "build_route",
            "present_initial_plan_node": "present_initial_plan",
            "handle_plan_feedback_node": "handle_plan_feedback",
            "error_node": "error_handler",
            END: END,
        },
    )

    graph_builder.add_conditional_edges(
        "gather_all_candidates",  # <--- НОВЫЙ БЛОК РЕБЕР
        check_gathered_candidates_edge,
        {
            "optimal_chain_constructor_node": "construct_optimal_chain",
            "error_handler": "error_handler",
        },
    )

    graph_builder.add_conditional_edges(
        "construct_optimal_chain",  # <--- НОВЫЙ БЛОК РЕБЕР
        check_optimal_chain_result_edge,
        {
            "present_initial_plan_node": "present_initial_plan",
            "error_handler": "error_handler",
        },
    )

    graph_builder.add_conditional_edges(
        "handle_plan_feedback",
        check_plan_feedback_router_edge,
        {
            "confirm_changes_node": "confirm_changes",
            # "search_events_node": "legacy_search_events", # Если решим использовать старый поиск для fallback
            "gather_all_candidate_events_node": "gather_all_candidates",  # Если изменения требуют нового поиска
            "clarify_missing_data_node": "clarify_missing_data",
            "extract_initial_info_node": "extract_initial_info",  # Если это новый запрос
            END: END,
        },
    )

    graph_builder.add_edge("clarify_missing_data", END)
    graph_builder.add_edge("present_initial_plan", END)
    graph_builder.add_edge("error_handler", END)
    graph_builder.add_edge("confirm_changes", END)

    graph_builder.add_edge("build_route", "present_full_plan")
    graph_builder.add_edge("present_full_plan", END)

    logger.info("All edges (including new ones) added. Graph definition complete.")
    return graph_builder


memory_for_checkpoint = MemorySaver()
agent_graph_definition = create_agent_graph()
compiled_agent_graph = agent_graph_definition.compile(
    checkpointer=memory_for_checkpoint
)
logger.info("Agent graph compiled successfully with MemorySaver checkpointer.")


def get_compiled_agent_graph():
    return compiled_agent_graph
