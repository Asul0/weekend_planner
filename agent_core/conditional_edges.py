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
    # current_events на этом этапе (сразу после extract_initial_info) обычно пусты, если это новый запрос
    current_events = state.get("current_events", []) 
    user_has_address_coords = bool(collected_data.get("user_start_address_validated_coords"))
    fallback_accepted_and_updated = collected_data.get(
        "fallback_accepted_and_plan_updated", False
    )

    logger.debug(
        f"Edge state values (check_initial_data_sufficiency): clarification_fields={clarification_fields}, "
        f"is_initial_plan_proposed={is_initial_plan_proposed}, "
        f"fallback_accepted={fallback_accepted_and_updated}, user_has_address_coords={user_has_address_coords}, "
        f"current_events_count={len(current_events)}"
    )

    if clarification_fields:
        # Эта логика остается, так как уточнения имеют приоритет
        if state.get("clarification_context") and "user_start_address_original" in clarification_fields:
            logger.info("Edge (initial_sufficiency): Address clarification needed (set by extract_node via context). Routing to 'clarify_missing_data_node'.")
            return "clarify_missing_data_node"
        
        logger.info(
            f"Edge (initial_sufficiency): General clarification needed for fields: {clarification_fields}. Routing to 'clarify_missing_data_node'."
        )
        return "clarify_missing_data_node"

    # Если пользователь только что ответил на предложение fallback
    if fallback_accepted_and_updated:
        # current_events уже должны быть обновлены в extract_initial_info_node
        logger.info("Edge (initial_sufficiency): Fallback was just accepted and plan updated.")
        if user_has_address_coords or not current_events: # Если адрес есть или событий нет (нечего маршрутизировать)
            logger.info("Routing to 'clarify_address_or_build_route_node' (or skip if no events).")
            return "clarify_address_or_build_route_node" # build_route проверит current_events
        else: # Адреса нет, но есть события (включая fallback)
            logger.info("Routing to 'present_initial_plan_node' (to request address for new plan).")
            return "present_initial_plan_node" 

    # Если мы уже на этапе, когда начальный план (возможно, цепочка) был предложен,
    # и пользователь дал новый ввод, который не является ответом на уточнение или fallback.
    # Например, пользователь просит изменить план, или это просто новый запрос после завершения предыдущего.
    # `extract_initial_info_node` должен был сбросить is_initial_plan_proposed, если это новый запрос.
    # Если is_initial_plan_proposed все еще True, это может быть ответ на "Как вам план?".
    # Эту логику обрабатывает `handle_plan_feedback_node`.
    # Это условие здесь может быть избыточным, если `extract_initial_info` правильно обрабатывает контекст.
    # Оставим его, но с комментарием.
    if is_initial_plan_proposed:
        logger.info("Edge (initial_sufficiency): is_initial_plan_proposed is True.")
        # Если clarification_fields пуст, это значит, что extract_initial_info_node
        # не нашел новых уточнений и не считает это новым запросом.
        # Возможно, это ответ на открытый вопрос от present_initial_plan_node ("Как вам план?").
        # В таком случае, нужно идти на handle_plan_feedback.
        # Либо, если extract_initial_info_node обработал адрес, то идем на build_route.
        
        last_message = state.get("messages", [])[-1]
        # Очень упрощенная проверка, является ли это ответом на "Как вам план?"
        # Лучше, чтобы present_initial_plan_node устанавливал флаг типа awaiting_plan_feedback
        if last_message.type == "user" and not collected_data.get("awaiting_fallback_confirmation"):
             # Проверим, не является ли это ответом на запрос адреса/бюджета от present_initial_plan
             if state.get("awaiting_clarification_for_field") and \
                state["awaiting_clarification_for_field"] in ["user_start_address_original", "budget_original"]:
                 # extract_initial_info обработает это, и мы снова попадем сюда, но уже без awaiting_clarification_for_field
                 # и, возможно, с user_has_address_coords = True
                 if user_has_address_coords and state["awaiting_clarification_for_field"] == "user_start_address_original":
                     logger.info("Edge (initial_sufficiency): Address just provided for existing plan. Routing to 'build_route'.")
                     return "clarify_address_or_build_route_node" # Имя вашего узла "build_route"
                 elif state["awaiting_clarification_for_field"] == "budget_original":
                     logger.info("Edge (initial_sufficiency): Budget just provided. Routing back to 'present_initial_plan_node'.")
                     return "present_initial_plan_node" # Чтобы перепредложить план или запросить адрес, если нужно

        # Если дошли сюда, и is_initial_plan_proposed=True, но нет явных уточнений,
        # это может быть неявный фидбек или новый запрос.
        # Логика здесь сложная, так как extract_initial_info должен был бы сбросить is_initial_plan_proposed
        # для нового запроса. Если он этого не сделал, возможно, стоит направить на handle_plan_feedback.
        # Однако, карта графа из graph.txt показывает, что extract_initial_info может вести на handle_plan_feedback.
        # Это значит, что extract_initial_info сам может решить, что это фидбек.
        # Поэтому, если clarification_fields пуст, и is_initial_plan_proposed=True,
        # и мы не на этапе обработки ответа на fallback, то, вероятно, уже можно строить маршрут или показывать план.
        if user_has_address_coords:
            logger.info("Edge (initial_sufficiency): Plan was proposed, no new clarifications, user has address. Routing to 'build_route'.")
            return "clarify_address_or_build_route_node"
        elif current_events : # План предложен, адреса нет, но есть события
            logger.info("Edge (initial_sufficiency): Plan was proposed, no new clarifications, no address. Routing to 'present_initial_plan_node' (likely to ask address).")
            return "present_initial_plan_node"


    # Основной новый путь: если все данные собраны и уточнений не нужно
    if (
        collected_data.get("city_id_afisha")
        and collected_data.get("parsed_dates_iso")
        and collected_data.get("interests_keys_afisha")
    ):
        logger.info(
            "Edge (initial_sufficiency): Initial data sufficient. Routing to 'gather_all_candidate_events_node'."
        )
        return "gather_all_candidate_events_node" # <--- ИЗМЕНЕНИЕ: Идем на сбор всех кандидатов

    # Если ничего из вышеперечисленного не сработало, значит, нужны базовые уточнения
    logger.warning(
        "Edge (initial_sufficiency): Fallback, no conditions met. Defaulting to 'clarify_missing_data_node'."
    )
    # Добавляем 'clarification_needed_fields', если их нет, чтобы clarify_missing_data_node сработал
    if not collected_data.get("clarification_needed_fields"):
        cf = []
        if not collected_data.get("city_id_afisha"): cf.append("city_name")
        if not collected_data.get("parsed_dates_iso"): cf.append("dates_description_original")
        if not collected_data.get("interests_keys_afisha"): cf.append("interests_original")
        state["collected_data"]["clarification_needed_fields"] = cf if cf else ["city_name"] # хотя бы что-то
            
    return "clarify_missing_data_node"


def check_gathered_candidates_edge(state: AgentState) -> str:
    logger.debug("Edge: check_gathered_candidates_edge evaluating...")
    candidate_events = state.get("candidate_events_by_interest", {})
    requested_interests = state.get("collected_data", {}).get("interests_keys_afisha", [])
    
    # Проверяем, есть ли кандидаты хотя бы для одного из запрошенных интересов
    found_any_candidates = False
    if candidate_events:
        for interest_key in requested_interests:
            if candidate_events.get(interest_key):
                found_any_candidates = True
                break
    
    if found_any_candidates:
        logger.info("Edge (gathered_candidates): Candidates found for at least one interest. Routing to 'optimal_chain_constructor_node'.")
        return "optimal_chain_constructor_node"
    else:
        search_errors = state.get("collected_data", {}).get("search_errors_by_interest", {})
        if search_errors and any(search_errors.values()): # Если были ошибки поиска
             logger.info("Edge (gathered_candidates): No candidates found, search errors present. Routing to 'error_handler'.")
        else: # Ошибок не было, но и кандидатов нет
             logger.info("Edge (gathered_candidates): No candidates found for any requested interest (empty lists). Routing to 'error_handler'.")
        # error_handler должен будет использовать search_errors_by_interest для более детального сообщения
        return "error_handler"

def check_event_search_result_edge(state: AgentState) -> str:
    logger.debug("Edge: check_event_search_result_edge evaluating...")
    current_events = state.get("current_events")

    if current_events and len(current_events) > 0:
        logger.info("Edge: Events found. Routing to 'present_initial_plan_node'.")
        return "present_initial_plan_node"
    else:
        logger.info("Edge: No events found. Routing to 'error_node'.")
        return "error_node"


def check_optimal_chain_result_edge(state: AgentState) -> str:
    logger.debug("Edge: check_optimal_chain_result_edge evaluating...")
    current_events_chain = state.get("current_events", []) # Это результат optimal_chain_constructor_node
    unplanned_keys = state.get("unplanned_interest_keys", [])
    candidate_events_available = state.get("candidate_events_by_interest", {})
    
    # Если есть хоть какая-то цепочка ИЛИ есть не вписавшиеся интересы, для которых есть кандидаты (т.е. есть что предложить/обсудить)
    if current_events_chain or (unplanned_keys and any(candidate_events_available.get(key) for key in unplanned_keys)):
        logger.info("Edge (optimal_chain_result): Optimal chain constructed or unplanned keys with candidates exist. Routing to 'present_initial_plan_node'.")
        return "present_initial_plan_node"
    else:
        # Цепочки нет, и для не вписавшихся интересов тоже нет кандидатов, или не вписавшихся нет.
        # Это значит, что либо изначально не было кандидатов (что должно было отсечься раньше),
        # либо конструктор не смог ничего собрать и для "отказников" тоже ничего нет.
        logger.info("Edge (optimal_chain_result): No optimal chain and no actionable unplanned keys/candidates. Routing to 'error_handler'.")
        # error_handler может использовать optimal_chain_construction_message или search_errors_by_interest
        return "error_handler"


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
