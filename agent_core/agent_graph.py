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

    # --- Добавление узлов ---
    # (остается как есть)
    graph_builder.add_node("extract_initial_info", extract_initial_info_node)
    graph_builder.add_node("clarify_missing_data", clarify_missing_data_node)
    graph_builder.add_node("search_events", search_events_node)
    graph_builder.add_node("present_initial_plan", present_initial_plan_node)
    graph_builder.add_node(
        "clarify_address_or_build_route", clarify_address_or_build_route_node
    )
    graph_builder.add_node("present_full_plan", present_full_plan_node)
    graph_builder.add_node("handle_plan_feedback", handle_plan_feedback_node)
    graph_builder.add_node("confirm_changes", confirm_changes_node)
    graph_builder.add_node("error_handler", error_node)
    logger.info("All nodes added.")

    graph_builder.set_entry_point("extract_initial_info")
    logger.info("Entry point set to 'extract_initial_info'.")

    # --- Рёбра ---

    # 1. От извлечения начальной информации
    graph_builder.add_conditional_edges(
        "extract_initial_info",
        check_initial_data_sufficiency_edge,  # Этот эдж должен вернуть "clarify_missing_data" или "search_events"
        {
            "clarify_missing_data_node": "clarify_missing_data",  # Переименовал для ясности
            "search_events_node": "search_events",
        },
    )

    # 3. От поиска мероприятий
    graph_builder.add_conditional_edges(
        "search_events",
        check_event_search_result_edge,  # Возвращает "present_initial_plan" или "error_handler"
        {
            "present_initial_plan_node": "present_initial_plan",  # Переименовал
            "error_node": "error_handler",  # Переименовал
        },
    )

    graph_builder.add_edge("present_initial_plan", "clarify_address_or_build_route")

    graph_builder.add_conditional_edges(
        "clarify_address_or_build_route",
        lambda state: "clarify_missing_data" if state.get("collected_data", {}).get("clarification_needed_fields") and "user_start_address_original" in state.get("collected_data", {}).get("clarification_needed_fields", []) else "present_full_plan",  # type: ignore
        {
            "clarify_missing_data": "clarify_missing_data",  # Если clarify_address_or_build_route решил, что адрес нужно УТОЧНИТЬ
            "present_full_plan": "present_full_plan",  # Если адрес есть/получен, или маршрут не нужен/построен
        },
    )

    graph_builder.add_edge(
        "present_full_plan", "handle_plan_feedback"
    )  # Пользовательский ответ будет последним в messages

    # 7. От обработки обратной связи
    graph_builder.add_conditional_edges(
        "handle_plan_feedback",
        check_plan_feedback_router_edge,
        {
            "confirm_changes_node": "confirm_changes",
            "search_events_node": "search_events",
            "clarify_missing_data_node": "clarify_missing_data",  # Если фидбек требует уточнений
            END: END,
        },
    )

    # 8. От подтверждения изменений
    # confirm_changes_node задал вопрос. Ждем ответа. Конец тика.
    # Ответ пользователя на этот вопрос должен быть обработан handle_plan_feedback_node.
    graph_builder.add_edge(
        "confirm_changes", "handle_plan_feedback"
    )  # Переименовал 'after_confirm_changes_edge'
    # Теперь это просто переход, а handle_plan_feedback разберется

    # 9. От узла ошибки
    # error_handler задал вопрос. Ждем ответа. Конец тика.
    # Ответ пользователя (новые критерии) пойдет на extract_initial_info.
    # Это ребро не нужно, так как после error_handler цикл графа для текущего пользовательского сообщения завершается,
    # и следующий ввод пользователя автоматически пойдет на точку входа "extract_initial_info".
    # graph_builder.add_edge("error_handler", "extract_initial_info") # Это вызовет рекурсию без нового ввода

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
