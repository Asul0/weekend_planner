import logging
from typing import List, Optional

from agent_core.agent_state import AgentState, CollectedUserData  # Наше состояние
from schemas.data_schemas import AnalyzedFeedback  # Для анализа намерения
from langchain_core.messages import AIMessage
from langgraph.graph import END


logger = logging.getLogger(__name__)

# --- Условные переходы ---


def check_initial_data_sufficiency_edge(state: AgentState) -> str:
    logger.debug("Edge: check_initial_data_sufficiency_edge evaluating...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    clarification_fields: Optional[List[str]] = collected_data.get(
        "clarification_needed_fields"
    )
    messages = state.get("messages", [])

    if messages and isinstance(messages[-1], AIMessage):
        logger.info(
            "Edge: Last message was AIMessage. Agent is likely waiting for user response. Ending current graph tick."
        )
        return END  # <--- ЯВНО УКАЗЫВАЕМ ЗАВЕРШИТЬ ТЕКУЩИЙ ПРОХОД ГРАФА

    # Эта логика сработает, только если последнее сообщение было HumanMessage (или если это самый первый вызов)
    if clarification_fields and len(clarification_fields) > 0:
        logger.info(
            "Edge: Data insufficient or time needs clarification (based on HumanMessage). Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"

    if (
        collected_data.get("city_id_afisha")
        and collected_data.get("parsed_dates_iso")
        and (
            collected_data.get("interests_keys_afisha")
            or collected_data.get("interests_original")
        )
    ):  # хотя бы какие-то интересы
        logger.info("Edge: Initial data sufficient. Routing to 'search_events_node'.")
        return "search_events_node"
    else:
        logger.warning(
            "Edge: Critical data still missing after HumanMessage processing. Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"


def check_event_search_result_edge(state: AgentState) -> str:
    logger.debug("Edge: check_event_search_result_edge evaluating...")
    current_events = state.get("current_events")

    if current_events and len(current_events) > 0:
        logger.info("Edge: Events found. Routing to 'present_initial_plan_node'.")
        return "present_initial_plan_node"
    else:
        logger.info("Edge: No events found. Routing to 'error_node'.")
        return "error_node"


def check_address_needs_clarification_or_route_exists_edge(state: AgentState) -> str:
    logger.debug(
        "Edge: check_address_needs_clarification_or_route_exists_edge evaluating..."
    )
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    current_events: Optional[List[Event]] = state.get("current_events")  # type: ignore
    current_route = state.get("current_route_details")

    # Если маршрут уже есть (например, после коррекции плана)
    if current_route:
        logger.info(
            "Edge: Route already exists or attempted. Routing to 'present_full_plan_node'."
        )
        return "present_full_plan_node"

    # Если адрес еще не известен и есть мероприятия (или мы только что его запросили)
    needs_address = not collected_data.get("user_start_address_validated_coords")

    if len(current_events or []) == 1 and needs_address:
        logger.info(
            "Edge: One event, user address not provided yet (or not needed for route for one event). Routing to 'clarify_address_or_build_route_node' (it will handle this)."
        )
        # clarify_address_or_build_route_node сам решит, строить маршрут или нет
        return "clarify_address_or_build_route_node"  # или present_full_plan_node?
        # clarify_address_or_build_route_node вернет route_details=None,
        # и потом present_full_plan_node его отобразит.

    if needs_address and current_events and len(current_events) > 0:
        logger.info(
            "Edge: User address needed for route or just asked. Routing to 'clarify_address_or_build_route_node'."
        )
        return "clarify_address_or_build_route_node"

    if (
        collected_data.get("user_start_address_validated_coords")
        or (current_events and len(current_events) <= 1)
    ) or (
        current_events and len(current_events) > 1
    ):  # Для случая маршрута от мероприятия к мероприятию
        logger.info(
            "Edge: Address present or route will be built between events. Routing to 'clarify_address_or_build_route_node'."
        )
        return "clarify_address_or_build_route_node"  # Этот узел теперь отвечает за всю логику маршрута

    # Если что-то пошло не так и мы здесь
    logger.warning(
        "Edge: Fallback in check_address_needs_clarification. Routing to 'present_full_plan_node' (likely without route)."
    )
    return "present_full_plan_node"  # Показать то, что есть


def check_plan_feedback_router_edge(state: AgentState) -> str:
    logger.debug("Edge: check_plan_feedback_router_edge evaluating...")
    if state.get("awaiting_final_confirmation"):
        # Это не должно происходить, если handle_plan_feedback_node сбросил флаг.
        # Но если пользователь ответил неясно, и мы снова ждем подтверждения.
        logger.info(
            "Edge: Still awaiting final confirmation after feedback. Routing back to 'handle_plan_feedback_node'."
        )
        return "handle_plan_feedback_node"  # Или, может, снова present_full_plan_node, чтобы повторить вопрос

    pending_modification = state.get("pending_plan_modification_request")

    # Если есть pending_modification, значит, пользователь запросил изменения,
    # и мы должны спросить подтверждение этих изменений (логика примера 2 из инструкции).
    if pending_modification:
        logger.info(
            "Edge: Pending plan modification. Routing to 'confirm_changes_node'."
        )
        return "confirm_changes_node"

    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    if collected_data.get("clarification_needed_fields"):
        logger.info(
            "Edge: Clarification needed after feedback. Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"

    last_ai_message_content = ""
    if state.get("messages"):
        for msg in reversed(state.get("messages", [])):  # type: ignore
            if msg.type == "ai":  # type: ignore
                last_ai_message_content = msg.content  # type: ignore
                break
    if "Отлично! Рад был помочь" in last_ai_message_content:
        logger.info("Edge: Plan confirmed and finalized. Routing to END.")
        return "__end__"  # Используем специальное имя для конечного узла LangGraph

    logger.info(
        "Edge: Changes applied or new criteria set from feedback. Routing to 'search_events_node' to rebuild plan."
    )
    return "search_events_node"  # Пересчитываем мероприятия с новыми данными


def after_confirm_changes_edge(state: AgentState) -> str:
    logger.debug("Edge: after_confirm_changes_edge evaluating...")
    logger.info(
        "Edge: After changes confirmation question. Routing to 'handle_plan_feedback_node' to process user's Yes/No."
    )
    return "handle_plan_feedback_node"
