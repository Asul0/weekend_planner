import logging
from typing import List, Optional, Dict, Any
from agent_core.agent_state import AgentState, CollectedUserData  # Наше состояние
from schemas.data_schemas import AnalyzedFeedback  # Для анализа намерения
from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import BaseModel
from schemas.data_schemas import Event

logger = logging.getLogger(__name__)

# --- Условные переходы ---


def check_initial_data_sufficiency_edge(state: AgentState) -> str:
    # DIAGNOSTIC LOG
    logger.info(
        "--- DIAGNOSTIC: STATE AT THE START of check_initial_data_sufficiency_edge ---"
    )
    collected_data_diag: Dict[str, Any] = dict(state.get("collected_data", {}))
    current_events_diag: List[Dict[str, Any]] = state.get("current_events", [])
    logger.info(f"    current_events length: {len(current_events_diag)}")
    logger.info(
        f"    is_initial_plan_proposed: {state.get('is_initial_plan_proposed')}"
    )
    logger.info(
        f"    fallback_accepted_and_plan_updated: {collected_data_diag.get('fallback_accepted_and_plan_updated')}"
    )
    logger.info(
        f"    clarification_needed_fields: {collected_data_diag.get('clarification_needed_fields')}"
    )
    logger.info(f"    just_modified_plan: {state.get('just_modified_plan')}")
    logger.info(
        "------------------------------------------------------------------------------------"
    )

    logger.debug("Edge: check_initial_data_sufficiency_edge evaluating...")
    collected_data: Dict[str, Any] = dict(state.get("collected_data", {}))

    if collected_data.get("address_clarification_status") == "SKIPPED_BY_USER":
        logger.info(
            "Edge: User skipped address input. Routing to 'present_full_plan' to show final plan without route."
        )
        collected_data["address_clarification_status"] = "SKIPPED_AND_PROCESSED"
        return "present_full_plan"

    clarification_fields: List[str] = collected_data.get(
        "clarification_needed_fields", []
    )
    is_initial_plan_proposed: bool = state.get("is_initial_plan_proposed", False)
    user_has_address_coords: bool = bool(
        collected_data.get("user_start_address_validated_coords")
    )
    fallback_accepted_and_updated: bool = collected_data.get(
        "fallback_accepted_and_plan_updated", False
    )
    plan_construction_strategy: str = collected_data.get(
        "plan_construction_strategy", "standard"
    )

    plan_modification_pending: bool = state.get("plan_modification_pending", False)
    modification_request_details: Optional[Dict[str, Any]] = state.get(
        "modification_request_details"
    )

    just_modified_plan_flag = state.get("just_modified_plan", False)

    if plan_modification_pending and modification_request_details:
        if not clarification_fields:
            logger.info(
                "Edge: Plan modification pending and details available. Routing to 'handle_plan_feedback'."
            )
            return "handle_plan_feedback"
        else:
            logger.info(
                f"Edge: Plan modification pending, but clarification needed for: {clarification_fields}. Routing to 'clarify_missing_data'."
            )
            return "clarify_missing_data"

    if clarification_fields:
        logger.info(
            f"Edge: Clarification needed for: {clarification_fields}. Routing to 'clarify_missing_data'."
        )
        return "clarify_missing_data"

    if plan_construction_strategy == "optimize_poi_time" and not clarification_fields:
        logger.info(
            "Edge: Plan construction strategy is 'optimize_poi_time' and no clarifications needed. Routing to 'search_events' for re-planning."
        )
        return "search_events"

    # FIX: Логика для fallback_accepted_and_updated должна быть более приоритетной
    if fallback_accepted_and_updated:
        current_plan_has_items_after_fallback = bool(
            state.get("current_events")
        ) or bool(collected_data.get("selected_pois_for_plan"))
        logger.info(
            f"Edge (fallback accepted branch): current_plan_has_items_after_fallback={current_plan_has_items_after_fallback}"
        )

        if not current_plan_has_items_after_fallback:
            # Этого не должно происходить с исправленным extract_initial_info_node, но на всякий случай
            logger.error(
                "Edge CRITICAL: Fallback was accepted, but plan is still empty! Routing to error_handler."
            )
            return "error_handler"

        if user_has_address_coords:
            logger.info(
                "Edge: Fallback accepted. User has address. Routing to 'clarify_address_or_build_route'."
            )
            return "clarify_address_or_build_route"
        else:
            logger.info(
                "Edge: Fallback accepted. User has NO address, but plan has items. Routing to 'present_initial_plan' to ask for address."
            )
            return "present_initial_plan"

    if is_initial_plan_proposed:
        current_plan_has_items_post_proposal = bool(
            state.get("current_events")
        ) or bool(collected_data.get("selected_pois_for_plan"))

        if (
            user_has_address_coords
            and current_plan_has_items_post_proposal
            and not just_modified_plan_flag
        ):
            logger.info(
                "Edge: Plan was proposed (not a direct modification), user has address, plan has items. Routing to 'clarify_address_or_build_route'."
            )
            return "clarify_address_or_build_route"
        elif not user_has_address_coords and current_plan_has_items_post_proposal:
            logger.info(
                "Edge: Plan was proposed, address still needed. Routing to 'present_initial_plan' (to ask for address)."
            )
            return "present_initial_plan"
        elif not current_plan_has_items_post_proposal:
            logger.info(
                "Edge: Plan was proposed, but now plan is empty. Routing to 'error_handler'."
            )
            return "error_handler"

    has_city = bool(collected_data.get("city_id_afisha")) or bool(
        collected_data.get("city_name")
    )
    has_dates = bool(collected_data.get("parsed_dates_iso"))
    has_any_activity_request = (
        bool(collected_data.get("ordered_activities"))
        or bool(collected_data.get("interests_keys_afisha"))
        or bool(collected_data.get("poi_park_query"))
        or bool(collected_data.get("poi_food_query"))
    )

    if has_city and has_dates and has_any_activity_request:
        logger.info(
            "Edge: Initial data (city, dates, activity request) sufficient for new search. Routing to 'search_events'."
        )
        return "search_events"
    else:
        logger.warning(
            f"Edge: Data insufficient for new search (city:{has_city}, dates:{has_dates}, activity:{has_any_activity_request}), but no clarification_fields set by extract_initial_info. Defaulting to 'clarify_missing_data'."
        )
        temp_clar_fields_edge = []
        if not has_city:
            temp_clar_fields_edge.append("city_name")
        if not has_dates:
            temp_clar_fields_edge.append("dates_description_original")
        if not has_any_activity_request:
            temp_clar_fields_edge.append("interests_original")
        if temp_clar_fields_edge:
            if "collected_data" not in state:
                state["collected_data"] = {}
            state["collected_data"]["clarification_needed_fields"] = list(
                set(temp_clar_fields_edge)
            )
            logger.info(
                f"Edge: Force-setting clarification_needed_fields to: {state['collected_data']['clarification_needed_fields']}"
            )
        return "clarify_missing_data"


def check_gathered_candidates_edge(state: AgentState) -> str:
    # Эта функция остается без изменений
    logger.debug("Edge: check_gathered_candidates_edge evaluating...")
    candidate_events = state.get("candidate_events_by_interest", {})
    requested_interests = state.get("collected_data", {}).get(
        "interests_keys_afisha", []
    )
    found_any_candidates = False
    if candidate_events:
        for interest_key in requested_interests:
            if candidate_events.get(interest_key):
                found_any_candidates = True
                break
    if found_any_candidates:
        logger.info(
            "Edge (gathered_candidates): Candidates found for at least one interest. Routing to 'optimal_chain_constructor_node'."
        )
        return "optimal_chain_constructor_node"
    else:
        search_errors = state.get("collected_data", {}).get(
            "search_errors_by_interest", {}
        )
        if search_errors and any(search_errors.values()):
            logger.info(
                "Edge (gathered_candidates): No candidates found, search errors present. Routing to 'error_handler'."
            )
        else:
            logger.info(
                "Edge (gathered_candidates): No candidates found for any requested interest (empty lists). Routing to 'error_handler'."
            )
        return "error_handler"


# Файл: agent_core/conditional_edges.py


def check_event_search_result_edge(state: AgentState) -> str:
    logger.debug("Edge: check_event_search_result_edge evaluating...")
    collected_data: Dict[str, Any] = state.get("collected_data", {})
    current_events: List[Dict[str, Any]] = state.get("current_events", [])
    selected_pois: List[Dict[str, Any]] = collected_data.get(
        "selected_pois_for_plan", []
    )

    has_any_results = bool(current_events) or bool(selected_pois)

    # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
    # Имя поля исправлено с 'budget_fallback_candidates' на 'budget_fallback_plan'
    has_budget_fallback = bool(collected_data.get("budget_fallback_plan"))
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    has_date_fallback = bool(collected_data.get("date_fallback_candidates"))
    has_combo_fallback = bool(collected_data.get("combo_fallback_candidates"))
    has_any_fallback = has_budget_fallback or has_date_fallback or has_combo_fallback

    plan_construction_failed_step = collected_data.get("plan_construction_failed_step")
    just_modified_plan_flag = state.get("just_modified_plan", False)
    user_has_address: bool = bool(
        collected_data.get("user_start_address_validated_coords")
    )
    fallback_was_accepted = collected_data.get(
        "fallback_accepted_and_plan_updated", False
    )

    logger.info(
        f"Edge check_event_search_result: has_any_results={has_any_results}, "
        f"has_budget_fallback={has_budget_fallback}, has_date_fallback={has_date_fallback}, "
        f"has_combo_fallback={has_combo_fallback}, just_modified_plan_flag={just_modified_plan_flag}, "
        f"user_has_address={user_has_address}"
    )

    if fallback_was_accepted:
        if user_has_address:
            logger.info(
                "Edge: Fallback accepted and user has address. Routing to 'clarify_address_or_build_route'."
            )
            return "clarify_address_or_build_route"
        else:
            logger.info(
                "Edge: Fallback accepted. User has NO address, but plan has items. Routing to 'present_initial_plan'."
            )
            return "present_initial_plan"

    if has_any_results or has_any_fallback or plan_construction_failed_step:
        logger.info(
            "Edge: Event search yielded results OR there are fallback options to present. Routing to 'present_initial_plan'."
        )
        return "present_initial_plan"
    else:
        logger.warning(
            "Edge: No results and no fallbacks/construction issues. Defaulting to 'error_handler'."
        )
        return "error_handler"


def check_optimal_chain_result_edge(state: AgentState) -> str:
    # Эта функция остается без изменений
    logger.debug("Edge: check_optimal_chain_result_edge evaluating...")
    current_events_chain = state.get("current_events", [])
    unplanned_keys = state.get("unplanned_interest_keys", [])
    candidate_events_available = state.get("candidate_events_by_interest", {})
    if current_events_chain or (
        unplanned_keys
        and any(candidate_events_available.get(key) for key in unplanned_keys)
    ):
        logger.info(
            "Edge (optimal_chain_result): Optimal chain constructed or unplanned keys with candidates exist. Routing to 'present_initial_plan_node'."
        )
        return "present_initial_plan_node"
    else:
        logger.info(
            "Edge (optimal_chain_result): No optimal chain and no actionable unplanned keys/candidates. Routing to 'error_handler'."
        )
        return "error_handler"


def check_address_needs_clarification_or_route_exists_edge(state: AgentState) -> str:
    # Эта функция остается без изменений
    logger.debug(
        "Edge: check_address_needs_clarification_or_route_exists_edge evaluating..."
    )
    collected_data_dict: dict = dict(
        state.get("collected_data", {})
    )  # Используем collected_data_dict для единообразия
    current_events_list: Optional[List[Event]] = state.get(
        "current_events"
    )  # Используем current_events_list
    current_route_details = state.get(
        "current_route_details"
    )  # Используем current_route_details

    if current_route_details:
        logger.info(
            "Edge: Route already exists or attempted. Routing to 'present_full_plan_node'."
        )
        return "present_full_plan_node"

    if current_events_list and collected_data_dict.get(
        "user_start_address_validated_coords"
    ):
        logger.info(
            "Edge: Events and validated user address present. Routing to 'clarify_address_or_build_route_node' to build route."
        )
        return "clarify_address_or_build_route_node"
    elif (
        current_events_list
        and len(current_events_list) > 1
        and not collected_data_dict.get("user_start_address_validated_coords")
    ):
        logger.info(
            "Edge: Multiple events, no user address. Route can be built between events. Routing to 'clarify_address_or_build_route_node'."
        )
        return "clarify_address_or_build_route_node"
    elif current_events_list:
        logger.info(
            "Edge: Events present, address not validated or not strictly needed yet. Routing to 'present_full_plan_node' (may not show route from user)."
        )
        return "present_full_plan_node"

    logger.warning(
        "Edge: Fallback in check_address_needs_clarification. No events or unclear state. Routing to 'error_node'."
    )
    return "error_node"


def check_plan_feedback_router_edge(state: AgentState) -> str:
    # Эта функция остается без изменений
    logger.debug("Edge: check_plan_feedback_router_edge evaluating...")
    if state.get("awaiting_final_confirmation"):
        logger.info(
            "Edge: Awaiting further clarification on feedback. Routing to END to get user input."
        )
        return END
    collected_data: dict = dict(state.get("collected_data", {}))
    if collected_data.get("clarification_needed_fields"):
        logger.info(
            f"Edge: Clarification needed after feedback processing for fields: {collected_data['clarification_needed_fields']}. Routing to 'clarify_missing_data'."
        )
        return "clarify_missing_data"
    last_ai_message_content = ""
    if state.get("messages"):
        for msg_idx in range(len(state["messages"]) - 1, -1, -1):  # type: ignore
            if state["messages"][msg_idx].type == "ai":  # type: ignore
                last_ai_message_content = state["messages"][msg_idx].content  # type: ignore
                break
    if (
        "Рад был помочь" in last_ai_message_content
        or "План окончательный. Надеюсь, вам понравится!" in last_ai_message_content
    ):
        logger.info(
            "Edge: Plan confirmed and finalized by agent in handle_plan_feedback. Routing to END."
        )
        return END
    if (
        not collected_data.get("city_name")
        and not collected_data.get("dates_description_original")
        and not collected_data.get("ordered_activities")
        and not collected_data.get("interests_original")
    ):
        logger.info(
            "Edge: Data seems reset by handle_plan_feedback for a new search. Routing to 'extract_initial_info'."
        )
        return "extract_initial_info"
    if bool(state.get("current_events")) or bool(
        collected_data.get("selected_pois_for_plan")
    ):
        logger.info(
            "Edge: Plan items exist after feedback processing, no clarifications. Routing to 'present_initial_plan'."
        )
        return "present_initial_plan"
    logger.info(
        "Edge: Changes likely applied by handle_plan_feedback or ready for search. Routing to 'search_events' to rebuild/update plan."
    )
    return "search_events"


def after_confirm_changes_edge(state: AgentState) -> str:
    # Эта функция остается без изменений
    logger.debug("Edge: after_confirm_changes_edge evaluating...")
    logger.info(
        "Edge: After changes confirmation question. Routing to 'handle_plan_feedback_node' to process user's Yes/No."
    )
    return "handle_plan_feedback_node"
