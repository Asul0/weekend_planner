import logging
from typing import List, Optional

from agent_core.agent_state import AgentState, CollectedUserData  # Наше состояние
from schemas.data_schemas import AnalyzedFeedback  # Для анализа намерения
from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import BaseModel
from schemas.data_schemas import Event
logger = logging.getLogger(__name__)

# --- Условные переходы ---


def check_initial_data_sufficiency_edge(state: AgentState) -> str:
    logger.debug(
        f"Edge: check_initial_data_sufficiency_edge evaluating. State keys: {list(state.keys())}"
    )
    collected_data = state.get("collected_data", {})
    clarification_fields = collected_data.get("clarification_needed_fields", [])
    is_initial_plan_proposed = state.get("is_initial_plan_proposed", False)
    current_events = state.get("current_events", [])
    user_has_address_coords = bool(collected_data.get("user_start_address_validated_coords"))
    fallback_accepted_and_updated = collected_data.get(
        "fallback_accepted_and_plan_updated", False
    )
    # Флаг awaiting_address_input теперь не так важен для этого ребра,
    # т.к. extract_initial_info_node сам решит, что делать, если он ждет адрес.
    # Важнее, есть ли clarification_fields, и что в них.

    logger.debug(
        f"Edge state values: clarification_fields={clarification_fields}, is_initial_plan_proposed={is_initial_plan_proposed}, fallback_accepted={fallback_accepted_and_updated}, user_has_address_coords={user_has_address_coords}"
    )

    if clarification_fields:
        # Если единственное, что нужно уточнить - это адрес, и мы только что его запросили через clarification_context
        # (например, "Уточните номер дома" или "Не удалось распознать адрес..."),
        # то extract_initial_info_node уже вернул управление, и граф должен ждать ответа пользователя, т.е. END.
        # Однако, если clarify_missing_data_node должен задать этот вопрос, то идем туда.
        # Проверим, есть ли clarification_context, который установил extract_initial_info_node
        if state.get("clarification_context") and "user_start_address_original" in clarification_fields:
            logger.info("Edge: Address clarification needed (set by extract_node via context). Routing to 'clarify_missing_data_node' to ask user.")
            return "clarify_missing_data_node"
        
        logger.info(
            f"Edge: General clarification needed for fields: {clarification_fields}. Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"

    if fallback_accepted_and_updated:
        if user_has_address_coords:
            logger.info(
                "Edge: Fallback accepted, user has address. Routing to 'clarify_address_or_build_route_node'."
            )
            return "clarify_address_or_build_route_node"
        else:
            logger.info(
                "Edge: Fallback accepted, user has NO address. Routing to 'present_initial_plan_node' (to request address)."
            )
            # present_initial_plan_node установит awaiting_address_input = True
            return "present_initial_plan_node"

    if is_initial_plan_proposed:
        if not current_events:
            logger.info(
                "Edge: Plan was proposed, but now no events. Routing to 'error_node'."
            )
            return "error_node"
        
        # Если мы здесь, значит, extract_initial_info_node успешно обработал адрес (если его ждали)
        # или это не был этап ожидания адреса.
        # clarification_fields пуст.
        if user_has_address_coords:
             logger.info(
                "Edge: Initial plan proposed and user has address. Routing to 'clarify_address_or_build_route_node' for route building."
            )
             return "clarify_address_or_build_route_node"
        else: # Адреса нет, но он и не требовался для clarification_fields
              # Значит, present_initial_plan_node должен был запросить его, если нужно
            logger.info(
                "Edge: Initial plan proposed, address not yet validated or not needed. Routing to 'present_initial_plan_node' to ensure address is handled or to show plan."
            )
            return "present_initial_plan_node"


    if (
        collected_data.get("city_id_afisha")
        and collected_data.get("parsed_dates_iso")
        and collected_data.get("interests_keys_afisha")
    ):
        logger.info(
            "Edge: Initial data sufficient for first event search. Routing to 'search_events_node'."
        )
        return "search_events_node"

    logger.warning(
        "Edge: check_initial_data_sufficiency_edge reached fallback state (no conditions met). Defaulting to 'clarify_missing_data_node'."
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
    collected_data: CollectedUserData = state.get("collected_data", {})
    current_events: Optional[List[Event]] = state.get("current_events")
    current_route = state.get("current_route_details")

    if current_route:
        logger.info(
            "Edge: Route already exists or attempted. Routing to 'present_full_plan_node'."
        )
        return "present_full_plan_node"

    # В новой логике extract_initial_info_node сам обрабатывает адрес.
    # Если мы дошли сюда, и адреса нет, но он нужен (current_events > 0),
    # то present_initial_plan_node должен был запросить его.
    # clarify_address_or_build_route_node теперь просто строит маршрут, если все данные есть.
    if current_events and collected_data.get("user_start_address_validated_coords"):
        logger.info("Edge: Events and validated user address present. Routing to 'clarify_address_or_build_route_node' to build route.")
        return "clarify_address_or_build_route_node"
    elif current_events and len(current_events) > 1 and not collected_data.get("user_start_address_validated_coords"):
        logger.info("Edge: Multiple events, no user address. Route can be built between events. Routing to 'clarify_address_or_build_route_node'.")
        return "clarify_address_or_build_route_node"
    elif current_events: # Есть события, но адрес не валидирован (или не нужен, если 1 событие и юзер не указал)
        logger.info("Edge: Events present, address not validated or not strictly needed yet. Routing to 'present_full_plan_node' (may not show route from user).")
        return "present_full_plan_node" # Показать план без маршрута от пользователя, если адрес не удалось получить.
    
    logger.warning(
        "Edge: Fallback in check_address_needs_clarification. No events or unclear state. Routing to 'error_node'."
    )
    return "error_node"

def check_plan_feedback_router_edge(state: AgentState) -> str:
    logger.debug("Edge: check_plan_feedback_router_edge evaluating...")

    if state.get("awaiting_final_confirmation"):
        logger.info(
            "Edge: Awaiting final confirmation. Agent asked a question. Ending current graph tick to wait for user."
        )
        return END

    pending_modification = state.get("pending_plan_modification_request")
    if pending_modification:
        logger.info(
            "Edge: Pending plan modification. Routing to 'confirm_changes_node'."
        )
        return "confirm_changes_node"

    collected_data: CollectedUserData = state.get("collected_data", {})
    if collected_data.get("clarification_needed_fields"):
        logger.info(
            "Edge: Clarification needed after feedback. Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"

    last_ai_message_content = ""
    if state.get("messages"):
        for msg in reversed(state.get("messages", [])):
            if msg.type == "ai":
                last_ai_message_content = msg.content
                break
    if "Отлично! Рад был помочь" in last_ai_message_content:
        logger.info("Edge: Plan confirmed and finalized. Routing to END.")
        return END

    logger.info(
        "Edge: Changes applied or new criteria set from feedback. Routing to 'search_events_node' to rebuild plan."
    )
    return "search_events_node"

def after_confirm_changes_edge(state: AgentState) -> str:
    logger.debug("Edge: after_confirm_changes_edge evaluating...")
    logger.info(
        "Edge: After changes confirmation question. Routing to 'handle_plan_feedback_node' to process user's Yes/No."
    )
    return "handle_plan_feedback_node"
