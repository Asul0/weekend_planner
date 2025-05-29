import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, date, timedelta  # Убедимся, что date импортирован
import asyncio  # Для asyncio.gather в новой логике
import re
import itertools

# Pydantic и Langchain для сообщений и схем
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool

# Наши внутренние модули
from agent_core.agent_state import AgentState, CollectedUserData
from llm_interface.gigachat_client import get_gigachat_client
from prompts.system_prompts import (
    INITIAL_INFO_EXTRACTION_PROMPT,
    GENERAL_CLARIFICATION_PROMPT_TEMPLATE,
    TIME_CLARIFICATION_PROMPT_TEMPLATE,
    PLAN_FEEDBACK_ANALYSIS_PROMPT,
    CHANGE_CONFIRMATION_PROMPT_TEMPLATE,
    EVENT_NOT_FOUND_PROMPT_TEMPLATE,
)
from schemas.data_schemas import (
    ExtractedInitialInfo,
    DateTimeParserToolArgs,
    EventSearchToolArgs,
    LocationModel,
    RouteBuilderToolArgs,
    Event,
    RouteDetails,
    ParsedDateTime,
    AnalyzedFeedback,
    RouteSegment,
)
from services.afisha_service import fetch_cities_internal  # Убедимся, что есть
from services.gis_service import get_coords_from_address, get_route
from tools.datetime_parser_tool import datetime_parser_tool
from tools.event_search_tool import event_search_tool
from tools.route_builder_tool import route_builder_tool
from services.gis_service import (
    get_geocoding_details,
    get_route,
    GeocodingResult,
)  # <--- ИЗМЕНЕНО

# Инициализация логгера
logger = logging.getLogger(__name__)


# --- Узел 1: Извлечение начальной информации ---
async def extract_initial_info_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: extract_initial_info_node executing...")
    awaiting_clarification_field: Optional[str] = state.get(
        "awaiting_clarification_for_field"
    )
    logger.info(
        f"extract_initial_info_node: Received awaiting_clarification_for_field = '{awaiting_clarification_field}'"
    )

    messages: List[BaseMessage] = state.get("messages", [])
    current_collected_data_dict: dict = dict(state.get("collected_data", {}))
    clarification_context_for_node: Optional[str] = None

    if not messages or not isinstance(messages[-1], HumanMessage):
        return {
            "collected_data": current_collected_data_dict,
            "messages": messages,
            "awaiting_clarification_for_field": awaiting_clarification_field,
            "clarification_context": clarification_context_for_node,
        }

    user_query = messages[-1].content.strip()
    user_query_lower = user_query.lower()

    reset_commands = ["новый поиск", "начни сначала", "отмена", "сброс", "стоп"]
    if any(cmd in user_query_lower for cmd in reset_commands):
        logger.info(f"User requested reset with: '{user_query}'")
        clarification_msg = (
            "Хорошо, начинаем новый поиск! Что ищем (город, даты, интересы, бюджет)?"
        )
        return {
            "collected_data": {},
            "current_events": [],
            "current_route_details": None,
            "messages": messages + [AIMessage(content=clarification_msg)],
            "status_message_to_user": clarification_msg,
            "clarification_needed_fields": [],
            "awaiting_clarification_for_field": None,
            "awaiting_fallback_confirmation": False,
            "pending_fallback_event": None,
            "last_offered_fallback_for_interest": None,
            "fallback_accepted_and_plan_updated": False,
            "not_found_interest_keys_in_primary_search": [],
            "is_initial_plan_proposed": False,
            "is_full_plan_with_route_proposed": False,
            "clarification_context": clarification_msg,
            "candidate_events_by_interest": {},
            "unplanned_interest_keys": [],
            "optimal_chain_construction_message": None,
            "actual_total_travel_time": None,
        }

    if current_collected_data_dict.get("awaiting_fallback_confirmation"):
        user_reply_fb = user_query_lower
        pending_fallback_event_data = current_collected_data_dict.get(
            "pending_fallback_event"
        )
        logger.info(f"Processing user reply '{user_reply_fb}' to fallback proposal.")

        current_collected_data_dict["awaiting_fallback_confirmation"] = False
        current_collected_data_dict["pending_fallback_event"] = None
        current_collected_data_dict["last_offered_fallback_for_interest"] = None

        if pending_fallback_event_data and isinstance(
            pending_fallback_event_data, dict
        ):
            try:
                pending_fallback_event = Event(**pending_fallback_event_data)
                if (
                    "да" in user_reply_fb
                    or "хочу" in user_reply_fb
                    or "добавить" in user_reply_fb
                ):
                    current_events_list: List[Any] = state.get(
                        "current_events", []
                    )  # Могут быть Event или dict
                    typed_current_events = []
                    for evt_data in current_events_list:
                        if isinstance(evt_data, dict):
                            typed_current_events.append(Event(**evt_data))
                        elif isinstance(evt_data, Event):
                            typed_current_events.append(evt_data)

                    updated_current_events = list(typed_current_events)

                    if not any(
                        evt.session_id == pending_fallback_event.session_id
                        for evt in updated_current_events
                    ):
                        updated_current_events.append(pending_fallback_event)
                        updated_current_events.sort(
                            key=lambda e: e.start_time_naive_event_tz
                        )

                    nf_primary = current_collected_data_dict.get(
                        "not_found_interest_keys_in_primary_search", []
                    )
                    if pending_fallback_event.event_type_key in nf_primary:
                        nf_primary.remove(pending_fallback_event.event_type_key)
                    current_collected_data_dict[
                        "not_found_interest_keys_in_primary_search"
                    ] = nf_primary
                    current_collected_data_dict[
                        "fallback_accepted_and_plan_updated"
                    ] = True

                    return {
                        "collected_data": current_collected_data_dict,
                        "messages": messages,
                        "current_events": updated_current_events,
                        "is_initial_plan_proposed": True,
                        "awaiting_clarification_for_field": None,
                    }
                else:
                    logger.info(
                        f"User rejected fallback for: {pending_fallback_event.name}"
                    )
                    current_collected_data_dict[
                        "fallback_accepted_and_plan_updated"
                    ] = False
                    return {
                        "collected_data": current_collected_data_dict,
                        "messages": messages,
                        "awaiting_clarification_for_field": None,
                    }
            except ValidationError as ve_fb:
                logger.error(f"Error validating fallback event: {ve_fb}")
                return {
                    "collected_data": current_collected_data_dict,
                    "messages": messages,
                    "clarification_context": "Произошла ошибка с предложенным вариантом.",
                    "awaiting_clarification_for_field": None,
                }
        else:
            return {
                "collected_data": current_collected_data_dict,
                "messages": messages,
                "awaiting_clarification_for_field": None,
            }

    if awaiting_clarification_field:
        logger.info(
            f"Processing '{user_query}' as clarification for '{awaiting_clarification_field}'"
        )
        new_clarification_needed_fields = list(
            current_collected_data_dict.get("clarification_needed_fields", [])
        )
        if awaiting_clarification_field in new_clarification_needed_fields:
            new_clarification_needed_fields.remove(awaiting_clarification_field)

        if awaiting_clarification_field == "city_name":
            current_collected_data_dict["city_name"] = user_query
            cities = await fetch_cities_internal()
            found_city = next(
                (c for c in cities if user_query.lower() in c["name_lower"]), None
            )
            if found_city:
                current_collected_data_dict["city_id_afisha"] = found_city["id"]
            else:
                current_collected_data_dict["city_id_afisha"] = None
                clarification_context_for_node = (
                    f"Город '{user_query}' не найден. Попробуйте другой."
                )
                if "city_name" not in new_clarification_needed_fields:
                    new_clarification_needed_fields.append("city_name")
        elif awaiting_clarification_field == "dates_description_original":
            current_collected_data_dict["dates_description_original"] = user_query
            current_collected_data_dict["raw_time_description_original"] = None
            parsed_dt_result = await datetime_parser_tool.ainvoke(
                {
                    "natural_language_date": user_query,
                    "natural_language_time_qualifier": None,
                    "base_date_iso": datetime.now().isoformat(),
                }
            )
            if parsed_dt_result.get("datetime_iso"):
                current_collected_data_dict["parsed_dates_iso"] = [
                    parsed_dt_result["datetime_iso"]
                ]
                current_collected_data_dict["parsed_end_dates_iso"] = (
                    [parsed_dt_result["end_datetime_iso"]]
                    if parsed_dt_result.get("end_datetime_iso")
                    else None
                )
                if parsed_dt_result.get("is_ambiguous"):
                    clarification_context_for_node = parsed_dt_result.get(
                        "clarification_needed"
                    )
                    if (
                        "dates_description_original"
                        not in new_clarification_needed_fields
                    ):
                        new_clarification_needed_fields.append(
                            "dates_description_original"
                        )
            else:
                clarification_context_for_node = (
                    parsed_dt_result.get("clarification_needed")
                    or "Не удалось распознать уточненную дату."
                )
                if "dates_description_original" not in new_clarification_needed_fields:
                    new_clarification_needed_fields.append("dates_description_original")
        elif awaiting_clarification_field == "interests_original":
            interests_list = [i.strip() for i in user_query.split(",") if i.strip()]
            current_collected_data_dict["interests_original"] = interests_list
            mapped_interest_keys = []
            for interest_str in interests_list:
                s = interest_str.lower()
                key = None
                if "фильм" in s or "кино" in s:
                    key = "Movie"
                elif "концерт" in s:
                    key = "Concert"
                elif "театр" in s or "спектакль" in s:
                    key = "Performance"
                elif "выставк" in s:
                    key = "Exhibition"
                elif "спорт" in s:
                    key = "SportEvent"
                elif "экскурс" in s:
                    key = "Excursion"
                elif "шоу" in s or "фестивал" in s or "ярмарк" in s:
                    key = "Event"
                elif "музей" in s:
                    key = "Музей"
                elif "прогулк" in s:
                    key = "Прогулки"
                elif "кафе" in s or "ресторан" in s or "покушать" in s or "поесть" in s:
                    key = "Кафе"
                if not key:
                    key = interest_str.capitalize()
                if key:
                    mapped_interest_keys.append(key)
            current_collected_data_dict["interests_keys_afisha"] = list(
                set(mapped_interest_keys)
            )
        elif awaiting_clarification_field == "budget_original":
            try:
                budget_val_match = re.search(r"\d+", user_query)
                if budget_val_match:
                    budget_val = int(budget_val_match.group(0))
                    current_collected_data_dict["budget_original"] = budget_val
                    current_collected_data_dict["budget_current_search"] = budget_val
                else:
                    raise ValueError("No digits in budget input")
            except ValueError:
                clarification_context_for_node = "Пожалуйста, укажите бюджет числом."
                if "budget_original" not in new_clarification_needed_fields:
                    new_clarification_needed_fields.append("budget_original")
        elif awaiting_clarification_field == "user_start_address_original":
            city_for_geocoding = current_collected_data_dict.get("city_name")
            previously_found_street = current_collected_data_dict.get(
                "partial_address_street"
            )
            address_to_geocode = user_query
            current_collected_data_dict["awaiting_address_input"] = False

            if previously_found_street and not any(
                c.isalpha()
                for c in user_query
                if c.isalpha() and c.lower() not in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            ):
                address_to_geocode = f"{previously_found_street}, {user_query}"

            if not city_for_geocoding:
                clarification_context_for_node = (
                    "Сначала нужно указать город для точного геокодинга адреса."
                )
                if "city_name" not in new_clarification_needed_fields:
                    new_clarification_needed_fields.append("city_name")
                if "user_start_address_original" not in new_clarification_needed_fields:
                    new_clarification_needed_fields.append(
                        "user_start_address_original"
                    )
                if "partial_address_street" in current_collected_data_dict:
                    del current_collected_data_dict["partial_address_street"]
            else:
                geocoding_result: GeocodingResult = await get_geocoding_details(
                    address=address_to_geocode, city=city_for_geocoding
                )
                if geocoding_result.is_precise_enough and geocoding_result.coords:
                    current_collected_data_dict["user_start_address_original"] = (
                        geocoding_result.full_address_name_gis
                    )
                    current_collected_data_dict[
                        "user_start_address_validated_coords"
                    ] = {
                        "lon": geocoding_result.coords[0],
                        "lat": geocoding_result.coords[1],
                    }
                    if "partial_address_street" in current_collected_data_dict:
                        del current_collected_data_dict["partial_address_street"]
                elif (
                    geocoding_result.match_level == "street"
                    and not previously_found_street
                ):
                    clarification_context_for_node = f"Нашел улицу '{geocoding_result.full_address_name_gis}'. Уточните номер дома."
                    current_collected_data_dict["partial_address_street"] = (
                        geocoding_result.full_address_name_gis
                    )
                    current_collected_data_dict["awaiting_address_input"] = True
                    if (
                        "user_start_address_original"
                        not in new_clarification_needed_fields
                    ):
                        new_clarification_needed_fields.append(
                            "user_start_address_original"
                        )
                else:
                    if "partial_address_street" in current_collected_data_dict:
                        del current_collected_data_dict["partial_address_street"]
                    clarification_context_for_node = f"Не удалось распознать '{user_query}' как точный адрес. Пожалуйста, укажите улицу и номер дома, или скажите 'новый поиск', чтобы начать заново."
                    current_collected_data_dict["awaiting_address_input"] = True
                    if (
                        "user_start_address_original"
                        not in new_clarification_needed_fields
                    ):
                        new_clarification_needed_fields.append(
                            "user_start_address_original"
                        )

        current_collected_data_dict["clarification_needed_fields"] = [
            f for f in new_clarification_needed_fields if f
        ]
        current_collected_data_dict["awaiting_clarification_for_field"] = None
        return {
            "collected_data": current_collected_data_dict,
            "messages": messages,
            "clarification_context": clarification_context_for_node,
            "awaiting_clarification_for_field": None,
        }

    logger.debug("extract_initial_info_node: Processing as a new/general query.")
    preserved_user_address = current_collected_data_dict.get(
        "user_start_address_original"
    )
    preserved_user_coords = current_collected_data_dict.get(
        "user_start_address_validated_coords"
    )
    current_collected_data_dict_for_new_query = {}
    if preserved_user_address:
        current_collected_data_dict_for_new_query["user_start_address_original"] = (
            preserved_user_address
        )
    if preserved_user_coords:
        current_collected_data_dict_for_new_query[
            "user_start_address_validated_coords"
        ] = preserved_user_coords

    current_collected_data_dict_for_new_query["is_initial_plan_proposed"] = False
    current_collected_data_dict_for_new_query["fallback_accepted_and_plan_updated"] = (
        False
    )
    current_collected_data_dict_for_new_query["awaiting_fallback_confirmation"] = False
    current_collected_data_dict_for_new_query["pending_fallback_event"] = None
    current_collected_data_dict_for_new_query["last_offered_fallback_for_interest"] = (
        None
    )
    current_collected_data_dict_for_new_query[
        "not_found_interest_keys_in_primary_search"
    ] = []
    current_collected_data_dict_for_new_query["fallback_candidates"] = {}

    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ExtractedInitialInfo)
    try:
        extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{user_query}"'
        extracted_info: ExtractedInitialInfo = await structured_llm.ainvoke(
            extraction_prompt_with_query
        )
        logger.info(
            f"extract_initial_info_node: LLM Extracted Info (general): {extracted_info.model_dump_json(indent=2)}"
        )
        new_clarification_needed = []

        if extracted_info.city:
            current_collected_data_dict_for_new_query["city_name"] = extracted_info.city
            cities = await fetch_cities_internal()
            found_city = next(
                (c for c in cities if extracted_info.city.lower() in c["name_lower"]),
                None,
            )
            if found_city:
                current_collected_data_dict_for_new_query["city_id_afisha"] = (
                    found_city["id"]
                )
            else:
                new_clarification_needed.append("city_name")
        else:
            new_clarification_needed.append("city_name")

        if extracted_info.interests:
            current_collected_data_dict_for_new_query["interests_original"] = (
                extracted_info.interests
            )
            mapped_interest_keys = []
            for interest_str in extracted_info.interests:
                s = interest_str.lower()
                key = None
                if "фильм" in s or "кино" in s:
                    key = "Movie"
                elif "концерт" in s:
                    key = "Concert"
                elif "театр" in s or "спектакль" in s:
                    key = "Performance"
                elif "выставк" in s:
                    key = "Exhibition"
                elif "спорт" in s:
                    key = "SportEvent"
                elif "экскурс" in s:
                    key = "Excursion"
                elif "шоу" in s or "фестивал" in s or "ярмарк" in s:
                    key = "Event"
                elif "музей" in s:
                    key = "Музей"
                elif "прогулк" in s:
                    key = "Прогулки"
                elif "кафе" in s or "ресторан" in s or "покушать" in s or "поесть" in s:
                    key = "Кафе"
                if not key:
                    key = interest_str.capitalize()
                if key:
                    mapped_interest_keys.append(key)
            current_collected_data_dict_for_new_query["interests_keys_afisha"] = list(
                set(mapped_interest_keys)
            )
        else:
            new_clarification_needed.append("interests_original")

        if extracted_info.budget is not None:
            current_collected_data_dict_for_new_query["budget_original"] = (
                extracted_info.budget
            )
            current_collected_data_dict_for_new_query["budget_current_search"] = (
                extracted_info.budget
            )

        date_desc = extracted_info.dates_description
        time_desc = extracted_info.raw_time_description
        current_collected_data_dict_for_new_query["dates_description_original"] = (
            date_desc
        )
        current_collected_data_dict_for_new_query["raw_time_description_original"] = (
            time_desc
        )

        if date_desc or time_desc:
            parsed_dt_res = await datetime_parser_tool.ainvoke(
                {
                    "natural_language_date": date_desc or "",
                    "natural_language_time_qualifier": time_desc,
                    "base_date_iso": datetime.now().isoformat(),
                }
            )
            if parsed_dt_res.get("datetime_iso"):
                current_collected_data_dict_for_new_query["parsed_dates_iso"] = [
                    parsed_dt_res["datetime_iso"]
                ]
                current_collected_data_dict_for_new_query["parsed_end_dates_iso"] = (
                    [parsed_dt_res["end_datetime_iso"]]
                    if parsed_dt_res.get("end_datetime_iso")
                    else None
                )
                if parsed_dt_res.get("is_ambiguous"):
                    new_clarification_needed.append("dates_description_original")
                    clarification_context_for_node = parsed_dt_res.get(
                        "clarification_needed"
                    )
            else:
                new_clarification_needed.append("dates_description_original")
                clarification_context_for_node = (
                    parsed_dt_res.get("clarification_needed")
                    or "Не удалось распознать дату/время."
                )
        else:
            new_clarification_needed.append("dates_description_original")

        current_collected_data_dict_for_new_query["clarification_needed_fields"] = list(
            set(new_clarification_needed)
        )
        current_collected_data_dict = current_collected_data_dict_for_new_query
    except Exception as e:
        logger.error(
            f"extract_initial_info_node: LLM extraction error: {e}", exc_info=True
        )
        current_collected_data_dict.setdefault("clarification_needed_fields", [])
        for f_key in ["city_name", "dates_description_original", "interests_original"]:
            if f_key not in current_collected_data_dict.get(
                "clarification_needed_fields", []
            ) and not current_collected_data_dict.get(f_key):
                current_collected_data_dict["clarification_needed_fields"].append(f_key)
        current_collected_data_dict["clarification_needed_fields"] = list(
            set(current_collected_data_dict["clarification_needed_fields"])
        )
        clarification_context_for_node = "Ошибка обработки запроса. Попробуйте еще раз."

    logger.info(
        f"extract_initial_info_node: Final collected_data (general): {str(current_collected_data_dict)[:500]}"
    )
    return {
        "collected_data": current_collected_data_dict,
        "messages": messages,
        "clarification_context": clarification_context_for_node,
        "awaiting_clarification_for_field": None,
        "current_events": [],
        "current_route_details": None,
    }


# --- Узел 2: Уточнение недостающих данных ---
async def clarify_missing_data_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_missing_data_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    clarification_fields: List[str] = collected_data_dict.get(
        "clarification_needed_fields", []
    )
    status_message_to_user: Optional[str] = None
    field_being_clarified: Optional[str] = None

    if not clarification_fields:
        logger.info("clarify_missing_data_node: No fields need explicit clarification.")
        return {
            "status_message_to_user": None,
            "awaiting_clarification_for_field": None,
            "clarification_context": None,
            "collected_data": collected_data_dict,
        }

    field_to_clarify_now = clarification_fields[0]
    field_being_clarified = field_to_clarify_now
    missing_critical_fields_map = {
        "city_name": "город для поиска",
        "dates_description_original": "даты или период мероприятий",
        "interests_original": "ваши интересы или тип мероприятий",
        "budget_original": "ваш примерный бюджет",
        "user_start_address_original": "ваш адрес отправления (улица и дом)",
    }

    clarification_context_from_state = state.get("clarification_context")
    if (
        isinstance(clarification_context_from_state, str)
        and clarification_context_from_state
    ):
        status_message_to_user = clarification_context_from_state
        logger.info(
            f"clarify_missing_data_node: Using pre-defined clarification context: {status_message_to_user}"
        )
    else:
        field_description_for_prompt = missing_critical_fields_map.get(
            field_to_clarify_now, f"поле '{field_to_clarify_now}'"
        )
        raw_time_desc = collected_data_dict.get("raw_time_description_original")
        prompt_for_llm: str
        if field_to_clarify_now == "dates_description_original" and raw_time_desc:
            prompt_for_llm = TIME_CLARIFICATION_PROMPT_TEMPLATE.format(
                raw_time_description=raw_time_desc,
                current_date_info=date.today().strftime("%d %B %Y года (%A)"),
            )
        else:
            last_user_message_content = "Ваш запрос"
            current_messages = state.get("messages", [])
            if current_messages and isinstance(current_messages[-1], HumanMessage):
                last_user_message_content = current_messages[-1].content
            excluded_keys_for_summary = [
                "clarification_needed_fields",
                "awaiting_clarification_for_field",
                "awaiting_address_input",
                "partial_address_street",
                "awaiting_fallback_confirmation",
                "pending_fallback_event",
                "fallback_accepted_and_plan_updated",
                "previous_confirmed_collected_data",
                "previous_confirmed_events",
                "user_time_desc_for_fallback",
                "not_found_interest_keys",
                "fallback_candidates",
                "search_errors_by_interest",
            ]
            current_data_summary_parts = []
            for k, v in collected_data_dict.items():
                if v and k not in excluded_keys_for_summary:
                    if k == "city_name":
                        current_data_summary_parts.append(f"Город: {v}")
                    elif k == "dates_description_original":
                        current_data_summary_parts.append(f"Когда: {v}")
                    elif k == "interests_original":
                        current_data_summary_parts.append(
                            f"Интересы: {', '.join(v) if isinstance(v, list) else v}"
                        )
                    elif k == "budget_original":
                        current_data_summary_parts.append(f"Бюджет: до {v} руб.")
            current_data_summary_str = (
                "; ".join(current_data_summary_parts)
                if current_data_summary_parts
                else "пока ничего не уточнено"
            )
            prompt_for_llm = GENERAL_CLARIFICATION_PROMPT_TEMPLATE.format(
                user_query=last_user_message_content,
                current_collected_data_summary=current_data_summary_str,
                missing_fields_description=field_description_for_prompt,
            )

        logger.debug(
            f"clarify_missing_data_node: Using LLM prompt for '{field_description_for_prompt}'"
        )
        llm = get_gigachat_client()
        try:
            ai_response = await llm.ainvoke(prompt_for_llm)
            status_message_to_user = ai_response.content
            logger.info(
                f"clarify_missing_data_node: LLM generated clarification question: {status_message_to_user}"
            )
        except Exception as e_clarify:
            logger.error(
                f"clarify_missing_data_node: Error during LLM call: {e_clarify}",
                exc_info=True,
            )
            status_message_to_user = f"Мне нужно уточнение по полю: {field_description_for_prompt}. Не могли бы вы помочь?"

    final_message_to_user = (
        status_message_to_user
        or f"Пожалуйста, уточните {field_description_for_prompt}."
    )
    new_messages_history = state.get("messages", []) + [
        AIMessage(content=final_message_to_user)
    ]
    return {
        "messages": new_messages_history,
        "status_message_to_user": final_message_to_user,
        "awaiting_clarification_for_field": field_being_clarified,
        "clarification_context": None,
        "collected_data": collected_data_dict,
    }


# --- Узел 3: Поиск мероприятий (ОБНОВЛЕННАЯ ВЕРСИЯ) ---
async def search_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: search_events_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    original_user_interests_keys: List[str] = list(
        collected_data_dict.get("interests_keys_afisha", [])
    )

    collected_data_dict["not_found_interest_keys_in_primary_search"] = []
    collected_data_dict["fallback_candidates"] = {}
    collected_data_dict["fallback_accepted_and_plan_updated"] = False

    city_id = collected_data_dict.get("city_id_afisha")
    parsed_dates_iso_list = collected_data_dict.get("parsed_dates_iso")
    budget = collected_data_dict.get("budget_current_search")

    if not city_id or not parsed_dates_iso_list or not original_user_interests_keys:
        logger.warning(f"search_events_node: Missing critical data for search.")
        collected_data_dict["not_found_interest_keys_in_primary_search"] = list(
            original_user_interests_keys
        )
        return {
            "current_events": [],
            "is_initial_plan_proposed": False,
            "collected_data": collected_data_dict,
        }

    try:
        user_min_start_dt_naive = datetime.fromisoformat(parsed_dates_iso_list[0])
        user_explicitly_provided_time = not (
            user_min_start_dt_naive.hour == 0 and user_min_start_dt_naive.minute == 0
        )
        api_date_from_dt = user_min_start_dt_naive.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        api_date_to_for_primary_search = api_date_from_dt + timedelta(days=1)
        parsed_end_dates_iso_list = collected_data_dict.get("parsed_end_dates_iso")
        user_max_overall_end_dt_naive: Optional[datetime] = None
        if parsed_end_dates_iso_list and parsed_end_dates_iso_list[0]:
            temp_end_dt = datetime.fromisoformat(parsed_end_dates_iso_list[0])
            if (
                temp_end_dt.hour == 0
                and temp_end_dt.minute == 0
                and temp_end_dt.second == 0
            ):
                user_max_overall_end_dt_naive = temp_end_dt.replace(
                    hour=23, minute=59, second=59
                )
            else:
                user_max_overall_end_dt_naive = temp_end_dt
            if user_max_overall_end_dt_naive.date() >= api_date_from_dt.date():
                api_date_to_for_primary_search = user_max_overall_end_dt_naive.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
    except Exception as e:
        logger.error(f"Error parsing dates in search_events_node: {e}", exc_info=True)
        return {
            "current_events": [],
            "is_initial_plan_proposed": False,
            "collected_data": collected_data_dict,
        }

    all_events_found_by_type_primary: Dict[str, List[Event]] = {}
    min_start_for_primary = (
        user_min_start_dt_naive
        if user_explicitly_provided_time
        else api_date_from_dt.replace(hour=17, minute=0)
    )
    max_start_for_primary = user_max_overall_end_dt_naive  # Это максимальное время НАЧАЛА для event_search_tool

    async def _perform_search_internal(
        interest_key: str,
        min_start: Optional[datetime],
        max_start: Optional[datetime],
        date_from_search: datetime,
        date_to_search: datetime,
    ) -> List[Event]:
        # ... (эта функция остается без изменений из предыдущего ответа)
        tool_args = EventSearchToolArgs(
            city_id=city_id,
            date_from=date_from_search,
            date_to=date_to_search,
            interests_keys=[interest_key],
            min_start_time_naive=min_start,
            max_start_time_naive=max_start,
            max_budget_per_person=budget,
        )
        logger.info(
            f"Searching for '{interest_key}'. Min start: {min_start}, Max start: {max_start}. Date range: {date_from_search.date()} - {date_to_search.date() - timedelta(days=1)}"
        )
        try:
            events_dicts: List[Dict] = await event_search_tool.ainvoke(
                tool_args.model_dump(exclude_none=True)
            )
            valid_events = []
            for evt_data in events_dicts:
                try:
                    event = Event(**evt_data)
                    if (
                        user_max_overall_end_dt_naive
                    ):  # Фильтруем по времени ОКОНЧАНИЯ здесь
                        event_end_time = event.start_time_naive_event_tz + timedelta(
                            minutes=event.duration_minutes or 120
                        )
                        if event_end_time > user_max_overall_end_dt_naive:
                            continue
                    valid_events.append(event)
                except ValidationError as ve:
                    logger.warning(
                        f"Invalid event data for '{interest_key}': {evt_data.get('name', 'N/A')}, error: {ve}"
                    )
            return valid_events
        except Exception as e_tool_search:
            logger.error(
                f"Error in event_search_tool for '{interest_key}': {e_tool_search}",
                exc_info=True,
            )
            return []

    primary_tasks = [
        _perform_search_internal(
            interest,
            min_start_for_primary,
            max_start_for_primary,
            api_date_from_dt,
            api_date_to_for_primary_search,
        )
        for interest in original_user_interests_keys
    ]
    results_primary_list: List[List[Event]] = await asyncio.gather(*primary_tasks, return_exceptions=True)  # type: ignore

    for i, interest_key in enumerate(original_user_interests_keys):
        result_item = results_primary_list[i]
        if isinstance(result_item, Exception):
            continue
        if result_item:
            all_events_found_by_type_primary[interest_key] = sorted(
                result_item, key=lambda e: e.start_time_naive_event_tz
            )
        else:
            collected_data_dict.setdefault(
                "not_found_interest_keys_in_primary_search", []
            ).append(interest_key)

    interests_for_fallback = collected_data_dict.get(
        "not_found_interest_keys_in_primary_search", []
    )
    if interests_for_fallback:
        fallback_date_from = api_date_from_dt
        fallback_date_to = api_date_from_dt + timedelta(days=7)
        fallback_tasks = [
            _perform_search_internal(
                interest, None, None, fallback_date_from, fallback_date_to
            )
            for interest in interests_for_fallback
        ]  # Ищем на весь день
        results_fallback_list: List[List[Event]] = await asyncio.gather(*fallback_tasks, return_exceptions=True)  # type: ignore
        for i, interest_key_fb in enumerate(interests_for_fallback):
            result_item_fb = results_fallback_list[i]
            if isinstance(result_item_fb, Exception):
                continue
            if result_item_fb:
                sorted_fb_events = sorted(
                    result_item_fb, key=lambda e: e.start_time_naive_event_tz
                )
                collected_data_dict.setdefault("fallback_candidates", {})[
                    interest_key_fb
                ] = sorted_fb_events[0].model_dump()

    # --- НОВАЯ ЛОГИКА ФОРМИРОВАНИЯ events_to_propose ---
    events_to_propose: List[Event] = []
    for interest_key in original_user_interests_keys:
        if (
            interest_key in all_events_found_by_type_primary
            and all_events_found_by_type_primary[interest_key]
        ):
            # Берем первое (самое раннее) подходящее событие для этого интереса
            events_to_propose.append(all_events_found_by_type_primary[interest_key][0])

    # Сортируем итоговый список предложений по времени начала
    events_to_propose.sort(key=lambda e: e.start_time_naive_event_tz)
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    logger.info(
        f"Proposing {len(events_to_propose)} events from primary. Fallback candidates for: {list(collected_data_dict.get('fallback_candidates', {}).keys())}. Not found in primary & time: {collected_data_dict.get('not_found_interest_keys_in_primary_search')}"
    )
    return {
        "current_events": events_to_propose,
        "is_initial_plan_proposed": bool(events_to_propose)
        or bool(collected_data_dict.get("fallback_candidates")),
        "collected_data": collected_data_dict,
    }


async def gather_all_candidate_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: gather_all_candidate_events_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))

    city_id = collected_data_dict.get("city_id_afisha")
    city_name_for_filter = collected_data_dict.get("city_name") # <--- ПОЛУЧАЕМ ИМЯ ГОРОДА
    parsed_dates_iso_list = collected_data_dict.get("parsed_dates_iso")
    interests_keys: List[str] = list(
        collected_data_dict.get("interests_keys_afisha", [])
    )
    budget = collected_data_dict.get("budget_current_search")

    candidate_events_by_interest: Dict[str, List[Event]] = {}
    # search_errors_by_interest теперь часть collected_data_dict, если вы так решили в AgentState
    # или можно оставить локальной переменной и потом обновить collected_data_dict
    search_errors_by_interest: Dict[str, str] = collected_data_dict.get("search_errors_by_interest", {})


    if not city_id or not parsed_dates_iso_list or not interests_keys:
        logger.warning(
            "gather_all_candidate_events_node: Missing critical data for gathering candidates."
        )
        for interest_key in interests_keys:
            search_errors_by_interest[interest_key] = (
                "Отсутствуют критические данные для поиска (город, дата или интересы)."
            )
        collected_data_dict["search_errors_by_interest"] = search_errors_by_interest
        return {
            "candidate_events_by_interest": candidate_events_by_interest,
            # "search_errors_by_interest": search_errors_by_interest, # Уже в collected_data
            "collected_data": collected_data_dict,
        }

    try:
        base_date_from_dt = datetime.fromisoformat(parsed_dates_iso_list[0]).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        api_date_to_dt = base_date_from_dt + timedelta(days=1)
        user_max_overall_end_dt_naive: Optional[datetime] = None
        if (
            collected_data_dict.get("parsed_end_dates_iso")
            and collected_data_dict["parsed_end_dates_iso"][0]
        ):
            user_max_overall_end_dt_naive = datetime.fromisoformat(
                collected_data_dict["parsed_end_dates_iso"][0]
            )
            if user_max_overall_end_dt_naive.date() >= base_date_from_dt.date():
                api_date_to_dt = user_max_overall_end_dt_naive.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)

        search_min_start_time_naive = base_date_from_dt
        search_max_start_time_naive = user_max_overall_end_dt_naive
    except Exception as e_date:
        logger.error(
            f"gather_all_candidate_events_node: Error parsing dates: {e_date}",
            exc_info=True,
        )
        for interest_key in interests_keys:
            search_errors_by_interest[interest_key] = f"Ошибка обработки дат: {e_date}"
        collected_data_dict["search_errors_by_interest"] = search_errors_by_interest
        return {
            "candidate_events_by_interest": candidate_events_by_interest,
            # "search_errors_by_interest": search_errors_by_interest,
            "collected_data": collected_data_dict,
        }

    search_tasks = []
    for interest_key in interests_keys:
        tool_args = EventSearchToolArgs(
            city_id=city_id,
            city_name=city_name_for_filter, # <--- ПЕРЕДАЕМ ИМЯ ГОРОДА
            date_from=base_date_from_dt,
            date_to=api_date_to_dt,
            interests_keys=[interest_key],
            min_start_time_naive=search_min_start_time_naive,
            max_start_time_naive=search_max_start_time_naive,
            max_budget_per_person=budget,
            exclude_session_ids=None, # Можно добавить сюда логику по exclude, если нужно
        )
        logger.debug(f"Gather candidates: Invoking event_search_tool for interest '{interest_key}' with args: {tool_args.model_dump_json(indent=2)}")
        search_tasks.append(
            event_search_tool.ainvoke(tool_args.model_dump(exclude_none=True))
        )

    results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    for i, interest_key in enumerate(interests_keys):
        result_item = results_list[i]
        if isinstance(result_item, Exception):
            logger.error(
                f"gather_all_candidate_events_node: Error searching for interest '{interest_key}': {result_item}",
                exc_info=True, # Было True, сохраняем
            )
            search_errors_by_interest[interest_key] = (
                f"Ошибка API при поиске: {result_item}"
            )
            candidate_events_by_interest[interest_key] = []
            continue

        if isinstance(result_item, list):
            valid_events_for_interest = []
            for event_data_dict in result_item:
                try:
                    event_obj = Event(**event_data_dict)
                    # Дополнительная проверка на координаты (если нужна перед передачей в конструктор цепочек)
                    if (
                        event_obj.place_coords_lon is None
                        or event_obj.place_coords_lat is None
                        or (
                            event_obj.place_coords_lon == 0.0
                            and event_obj.place_coords_lat == 0.0
                        )
                    ):
                        logger.debug(
                            f"Gather candidates: Skipping event {event_obj.name} for interest {interest_key} due to invalid/missing coordinates after tool call."
                        )
                        continue
                    
                    # Дополнительная проверка по времени окончания, если user_max_overall_end_dt_naive задан
                    if (
                        user_max_overall_end_dt_naive 
                        and event_obj.duration_minutes is not None
                    ):
                        event_end_time_naive = (
                            event_obj.start_time_naive_event_tz 
                            + timedelta(minutes=event_obj.duration_minutes)
                        )
                        if event_end_time_naive > user_max_overall_end_dt_naive:
                            logger.debug(
                                f"Gather candidates node: Event {event_obj.name} ends too late ({event_end_time_naive}), skipping."
                            )
                            continue
                    elif ( # Если длительность неизвестна, но событие НАЧИНАЕТСЯ позже максимально допустимого времени ОКОНЧАНИЯ
                        user_max_overall_end_dt_naive 
                        and event_obj.duration_minutes is None 
                        and event_obj.start_time_naive_event_tz > user_max_overall_end_dt_naive
                    ):
                         logger.debug(
                                f"Gather candidates node: Event {event_obj.name} starts too late ({event_obj.start_time_naive_event_tz}) with unknown duration, skipping."
                            )
                         continue

                    valid_events_for_interest.append(event_obj)
                except ValidationError as ve:
                    logger.warning(
                        f"gather_all_candidate_events_node: Validation error for event data for '{interest_key}': {ve}"
                    )
            
            candidate_events_by_interest[interest_key] = sorted(
                valid_events_for_interest, key=lambda e: e.start_time_naive_event_tz
            )
            if not valid_events_for_interest and interest_key not in search_errors_by_interest: # Если не было ошибки API, но список пуст
                search_errors_by_interest[interest_key] = (
                    "Мероприятия не найдены по указанным критериям (включая фильтр по городу и времени)."
                )
            logger.info(
                f"gather_all_candidate_events_node: Stored {len(valid_events_for_interest)} valid candidates for interest '{interest_key}'."
            )
        else:
            logger.error(
                f"gather_all_candidate_events_node: Unexpected result type for interest '{interest_key}': {type(result_item)}"
            )
            search_errors_by_interest[interest_key] = (
                "Неожиданный формат ответа от сервиса поиска."
            )
            candidate_events_by_interest[interest_key] = []

    collected_data_dict["search_errors_by_interest"] = search_errors_by_interest
    return {
        "candidate_events_by_interest": candidate_events_by_interest,
        "collected_data": collected_data_dict,
    }


async def _check_event_compatibility(
    first_event: Event,
    second_event_candidate: Event,
    user_max_overall_end_dt_naive: Optional[datetime],
) -> Tuple[bool, Optional[str]]:
    first_event_duration_minutes = first_event.duration_minutes or 120
    first_event_end_naive = first_event.start_time_naive_event_tz + timedelta(
        minutes=first_event_duration_minutes
    )

    if (
        user_max_overall_end_dt_naive
        and first_event_end_naive > user_max_overall_end_dt_naive
    ):
        return False, "Первое мероприятие заканчивается слишком поздно."
    if second_event_candidate.start_time_naive_event_tz < first_event_end_naive:
        return False, "Второе мероприятие начинается до окончания первого."

    route_duration_minutes = 30
    if (
        first_event.place_coords_lon
        and first_event.place_coords_lat
        and second_event_candidate.place_coords_lon
        and second_event_candidate.place_coords_lat
    ):
        try:
            route_result = await get_route(
                points=[
                    {
                        "lon": first_event.place_coords_lon,
                        "lat": first_event.place_coords_lat,
                    },
                    {
                        "lon": second_event_candidate.place_coords_lon,
                        "lat": second_event_candidate.place_coords_lat,
                    },
                ],
                transport="driving",
            )
            if route_result and route_result.get("status") == "success":
                route_duration_minutes = route_result.get("duration_seconds", 1800) / 60
            else:
                logger.warning(
                    f"Route error for compatibility check: {route_result.get('message') if route_result else 'No response'}"
                )
        except Exception as e_route:
            logger.error(
                f"get_route exception for compatibility: {e_route}", exc_info=True
            )

    arrival_at_second_event_naive = first_event_end_naive + timedelta(
        minutes=route_duration_minutes
    )
    buffer_time = timedelta(minutes=15)

    if (
        arrival_at_second_event_naive
        > second_event_candidate.start_time_naive_event_tz - buffer_time
    ):
        return False, "Не успеть на второе мероприятие."

    second_event_duration_minutes = second_event_candidate.duration_minutes or 120
    second_event_end_naive = (
        second_event_candidate.start_time_naive_event_tz
        + timedelta(minutes=second_event_duration_minutes)
    )
    if (
        user_max_overall_end_dt_naive
        and second_event_end_naive > user_max_overall_end_dt_naive
    ):
        return False, "Второе мероприятие заканчивается слишком поздно."
    return True, None


# --- Узел 4: Представление начального плана и запрос адреса/бюджета (ОБНОВЛЕННАЯ ВЕРСИЯ) ---
async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    current_optimal_chain: List[Event] = state.get("current_events", [])
    unplanned_keys: List[str] = state.get("unplanned_interest_keys", [])
    chain_construction_msg: Optional[str] = state.get(
        "optimal_chain_construction_message"
    )
    candidate_events_all: Dict[str, List[Event]] = state.get(
        "candidate_events_by_interest", {}
    )
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    response_parts: List[str] = []
    next_awaiting_clarification_field: Optional[str] = None

    interest_key_to_name_map_plural = {
        "Movie": "фильмов",
        "Performance": "спектаклей",
        "Concert": "концертов",
        "Exhibition": "выставок",
        "SportEvent": "спортивных событий",
        "Excursion": "экскурсий",
        "Event": "других событий",
        "Кафе": "кафе/ресторанов",
        "Прогулки": "прогулок",
        "Музей": "музеев",
    }
    interest_key_to_name_singular = {
        "Movie": "фильм",
        "Performance": "спектакль",
        "Concert": "концерт",
        "Exhibition": "выставку",
        "SportEvent": "спортивное событие",
        "Excursion": "экскурсию",
        "Event": "другое событие",
        "Кафе": "кафе/ресторан",
        "Прогулки": "прогулку",
        "Музей": "музей",
    }

    if current_optimal_chain:
        response_parts.append("Вот оптимальный план, который я смог составить:")
        for i, event in enumerate(current_optimal_chain):
            time_str = event.start_time_naive_event_tz.strftime("%H:%M")
            date_str = event.start_time_naive_event_tz.strftime("%d.%m.%Y")
            desc = f"{i+1}. **{event.name}** ({interest_key_to_name_singular.get(event.event_type_key, event.event_type_key)}) в '{event.place_name}' ({event.place_address or 'Адрес не указан'}). Начало в {time_str} ({date_str})."
            if event.min_price is not None:
                desc += f" Цена от {event.min_price} руб."
            if event.duration_minutes:
                desc += f" Продолжительность ~{event.duration_minutes // 60}ч {event.duration_minutes % 60}м."
            response_parts.append(desc)
        actual_travel_time_seconds = state.get("actual_total_travel_time")
        if actual_travel_time_seconds is not None and len(current_optimal_chain) > 1:
            travel_minutes = round(actual_travel_time_seconds / 60)
            response_parts.append(
                f"\nПримерное общее время на перемещения между этими мероприятиями: ~{travel_minutes} мин."
            )

    if chain_construction_msg:
        response_parts.append(f"\n{chain_construction_msg}")

    collected_data_dict["awaiting_fallback_confirmation"] = False
    collected_data_dict["pending_fallback_event"] = None
    offered_new_fallback_this_turn = False

    if unplanned_keys:
        user_min_time_str = (collected_data_dict.get("parsed_dates_iso") or [None])[0]
        user_max_time_str = (collected_data_dict.get("parsed_end_dates_iso") or [None])[
            0
        ]  # Важно, чтобы datetime_parser_tool это заполнял

        user_min_dt: Optional[datetime] = None
        if user_min_time_str:
            user_min_dt = datetime.fromisoformat(user_min_time_str)
            if user_min_dt.hour == 0 and user_min_dt.minute == 0:
                user_min_dt = user_min_dt.replace(hour=9, minute=0)

        user_max_dt: Optional[datetime] = None
        if user_max_time_str:
            user_max_dt = datetime.fromisoformat(user_max_time_str)
            if (
                user_max_dt.hour == 0
                and user_max_dt.minute == 0
                and user_max_dt.second == 0
            ):
                user_max_dt = user_max_dt.replace(hour=23, minute=59, second=59)

        for interest_key_fb in unplanned_keys:
            if interest_key_fb == collected_data_dict.get(
                "last_offered_fallback_for_interest"
            ) and not collected_data_dict.get("fallback_accepted_and_plan_updated"):
                continue

            candidate_list_for_fb = candidate_events_all.get(interest_key_fb, [])
            filtered_fb_candidates = []
            for cand_event in candidate_list_for_fb:
                valid_candidate = True
                if user_min_dt and cand_event.start_time_naive_event_tz < user_min_dt:
                    valid_candidate = False
                event_end_naive = cand_event.start_time_naive_event_tz + timedelta(
                    minutes=cand_event.duration_minutes or 120
                )
                if user_max_dt and event_end_naive > user_max_dt:
                    valid_candidate = False
                if valid_candidate:
                    filtered_fb_candidates.append(cand_event)

            if filtered_fb_candidates:
                best_fallback_event_obj = filtered_fb_candidates[0]
                type_name_plural = interest_key_to_name_map_plural.get(
                    interest_key_fb, f"'{interest_key_fb}'"
                )
                type_name_singular = interest_key_to_name_singular.get(
                    interest_key_fb, f"'{interest_key_fb}'"
                )
                fallback_msg_intro = ""
                if not current_optimal_chain and not chain_construction_msg:
                    fallback_msg_intro = (
                        f"К сожалению, не удалось составить оптимальный план. "
                    )
                elif not chain_construction_msg and unplanned_keys:
                    fallback_msg_intro = f"Мероприятия типа '{type_name_singular}' не вошли в основной план. "

                fallback_msg = f"{fallback_msg_intro}Однако, есть вариант для '{type_name_singular}': **{best_fallback_event_obj.name}** ({best_fallback_event_obj.start_time_naive_event_tz.strftime('%d.%m %H:%M')})."
                if best_fallback_event_obj.min_price is not None:
                    fallback_msg += f" Цена от {best_fallback_event_obj.min_price} руб."
                fallback_msg += " Хотите добавить его в план? (да/нет)"
                response_parts.append(f"\n{fallback_msg}")
                collected_data_dict["awaiting_fallback_confirmation"] = True
                collected_data_dict["pending_fallback_event"] = (
                    best_fallback_event_obj.model_dump()
                )
                collected_data_dict["last_offered_fallback_for_interest"] = (
                    interest_key_fb
                )
                offered_new_fallback_this_turn = True
                break

    if offered_new_fallback_this_turn:
        final_response_text = "\n".join(filter(None, response_parts)).strip()
        new_messages = state.get("messages", []) + [
            AIMessage(content=final_response_text)
        ]
        return {
            "messages": new_messages,
            "status_message_to_user": final_response_text,
            "collected_data": collected_data_dict,
            "is_initial_plan_proposed": bool(current_optimal_chain),
            "awaiting_clarification_for_field": None,
        }

    if "fallback_accepted_and_plan_updated" in collected_data_dict:
        del collected_data_dict["fallback_accepted_and_plan_updated"]

    final_uncovered_keys = []
    if unplanned_keys:
        for uk in unplanned_keys:
            is_in_current_plan = any(
                event.event_type_key == uk for event in current_optimal_chain
            )
            if not is_in_current_plan and uk != collected_data_dict.get(
                "last_offered_fallback_for_interest"
            ):
                if not candidate_events_all.get(uk) and not (
                    search_errors := collected_data_dict.get(
                        "search_errors_by_interest", {}
                    ).get(uk)
                ):  # Если нет кандидатов и не было ошибки поиска АПИ
                    final_uncovered_keys.append(
                        interest_key_to_name_map_plural.get(uk, uk)
                    )
                elif search_errors:  # Если была ошибка поиска API для этого интереса
                    final_uncovered_keys.append(
                        f"{interest_key_to_name_map_plural.get(uk, uk)} (ошибка поиска)"
                    )

    if final_uncovered_keys:
        if not current_optimal_chain and not response_parts:
            response_parts = [
                f"К сожалению, не удалось найти {', '.join(final_uncovered_keys)} по вашим критериям."
            ]
        elif not chain_construction_msg or all(
            uk not in (chain_construction_msg or "") for uk in unplanned_keys
        ):
            response_parts.append(
                f"\nТакже не удалось найти варианты для: {', '.join(final_uncovered_keys)}."
            )

    questions_to_user_suffix = []
    if not collected_data_dict.get("user_start_address_validated_coords"):
        if current_optimal_chain:
            questions_to_user_suffix.append(
                "Откуда вы планируете начать маршрут? Назовите, пожалуйста, адрес (улица и дом)."
            )
            next_awaiting_clarification_field = "user_start_address_original"

    if (
        collected_data_dict.get("budget_original") is None
        and not next_awaiting_clarification_field
    ):
        questions_to_user_suffix.append(
            "Уточните ваш бюджет на одно мероприятие (примерно)?"
        )
        next_awaiting_clarification_field = "budget_original"

    if questions_to_user_suffix:
        response_parts.append("\n" + " ".join(questions_to_user_suffix))
    elif current_optimal_chain:
        response_parts.append(
            "\nКак вам такой план? Если все устраивает или хотите что-то изменить, дайте знать."
        )
    elif not response_parts:
        response_parts.append(
            "По вашему запросу ничего не найдено. Попробуем другие критерии?"
        )

    final_response_text = "\n".join(filter(None, response_parts)).strip()
    if not final_response_text:
        final_response_text = (
            "Что-нибудь еще?"
            if current_optimal_chain
            else "Не удалось ничего найти. Попробуете изменить запрос?"
        )

    new_messages = state.get("messages", []) + [AIMessage(content=final_response_text)]
    return {
        "messages": new_messages,
        "status_message_to_user": final_response_text,
        "collected_data": collected_data_dict,
        "is_initial_plan_proposed": bool(current_optimal_chain),
        "awaiting_clarification_for_field": next_awaiting_clarification_field,
    }


# --- Узел 5: Обработка ответа на адрес ИЛИ построение маршрута, если адрес не нужен / уже есть ---
async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_address_or_build_route_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})
    current_events_from_state: Optional[List[Any]] = state.get("current_events", [])

    current_events: List[Event] = []
    if current_events_from_state:
        for evt_data in current_events_from_state:
            if isinstance(evt_data, Event):
                current_events.append(evt_data)
            elif isinstance(evt_data, dict):
                try:
                    current_events.append(Event(**evt_data))
                except ValidationError as e:
                    logger.warning(
                        f"build_route_node: Could not validate event data: {evt_data}, error: {e}"
                    )
                    continue
            else:
                logger.warning(
                    f"build_route_node: Unknown event data type: {type(evt_data)}"
                )

    if not current_events:
        logger.warning("build_route_node: No valid current events for route.")
        return {
            "current_route_details": RouteDetails(
                status="error", error_message="Нет мероприятий для построения маршрута."
            ),
            "is_full_plan_with_route_proposed": False,
        }

    user_start_address_str = collected_data.get("user_start_address_original")
    user_start_coords = collected_data.get("user_start_address_validated_coords")

    if not user_start_coords and len(current_events) <= 1:
        logger.info(
            "One event or less and no user address, no route to build from user or between events."
        )
        return {
            "current_route_details": None,
            "is_full_plan_with_route_proposed": False,
        }

    event_dates: Set[date] = {
        evt.start_time_naive_event_tz.date() for evt in current_events
    }
    multiple_days = len(event_dates) > 1
    logger.info(f"Route for events on multiple_days: {multiple_days}")

    all_route_segments: List[RouteSegment] = []
    total_duration_seconds_combined = 0
    total_distance_meters_combined = 0
    overall_route_status = "success"
    route_error_messages = []

    if multiple_days:
        if not user_start_coords:
            logger.warning("Multiple day events but no user start address for routing.")
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Для мероприятий в разные дни нужен ваш адрес отправления.",
                ),
                "is_full_plan_with_route_proposed": False,
            }
        user_start_location = LocationModel(
            lon=user_start_coords["lon"],
            lat=user_start_coords["lat"],
            address_string=user_start_address_str,
        )

        for i, event_obj in enumerate(current_events):
            event_location = None
            if (
                event_obj.place_coords_lon is not None
                and event_obj.place_coords_lat is not None
                and not (
                    event_obj.place_coords_lon == 0.0
                    and event_obj.place_coords_lat == 0.0
                )
            ):
                event_location = LocationModel(
                    lon=event_obj.place_coords_lon,
                    lat=event_obj.place_coords_lat,
                    address_string=event_obj.place_address,
                )
            elif event_obj.place_address:
                coords = await get_coords_from_address(
                    address=event_obj.place_address,
                    city=collected_data.get("city_name", ""),
                )
                if coords and not (coords[0] == 0.0 and coords[1] == 0.0):
                    event_location = LocationModel(
                        lon=coords[0],
                        lat=coords[1],
                        address_string=event_obj.place_address,
                    )

            if not event_location:
                segment_error_msg = (
                    f"Не удалось определить координаты для '{event_obj.name}'"
                )
                all_route_segments.append(
                    RouteSegment(
                        from_address=user_start_address_str or "Ваше местоположение",
                        to_address=event_obj.place_address or event_obj.name,
                        segment_status="error",
                        segment_error_message=segment_error_msg,
                    )
                )
                route_error_messages.append(segment_error_msg)
                overall_route_status = "partial_success"
                continue

            tool_args_segment = RouteBuilderToolArgs(
                start_point=user_start_location, event_points=[event_location]
            )
            route_data_segment_dict = await route_builder_tool.ainvoke(
                tool_args_segment.model_dump(exclude_none=True)
            )
            try:
                route_details_segment = RouteDetails(**route_data_segment_dict)
                if (
                    route_details_segment.status == "success"
                    and route_details_segment.segments
                ):
                    all_route_segments.extend(route_details_segment.segments)
                    total_duration_seconds_combined += (
                        route_details_segment.total_duration_seconds or 0
                    )
                    total_distance_meters_combined += (
                        route_details_segment.total_distance_meters or 0
                    )
                else:
                    overall_route_status = "partial_success"
                    error_msg_seg = (
                        route_details_segment.error_message
                        or f"Не удалось построить маршрут до '{event_obj.name}'"
                    )
                    all_route_segments.append(
                        RouteSegment(
                            from_address=user_start_address_str,
                            to_address=event_obj.place_address or event_obj.name,
                            segment_status="error",
                            segment_error_message=error_msg_seg,
                        )
                    )
                    route_error_messages.append(error_msg_seg)
            except ValidationError:
                overall_route_status = "partial_success"
                error_msg_val = "Ошибка данных маршрута"
                all_route_segments.append(
                    RouteSegment(
                        from_address=user_start_address_str,
                        to_address=event_obj.place_address or event_obj.name,
                        segment_status="error",
                        segment_error_message=error_msg_val,
                    )
                )
                route_error_messages.append(error_msg_val)
    else:
        points_for_single_route: List[LocationModel] = []
        if user_start_coords:
            points_for_single_route.append(
                LocationModel(
                    lon=user_start_coords["lon"],
                    lat=user_start_coords["lat"],
                    address_string=user_start_address_str,
                )
            )

        for event_obj in current_events:
            event_loc = None
            if (
                event_obj.place_coords_lon is not None
                and event_obj.place_coords_lat is not None
                and not (
                    event_obj.place_coords_lon == 0.0
                    and event_obj.place_coords_lat == 0.0
                )
            ):
                event_loc = LocationModel(
                    lon=event_obj.place_coords_lon,
                    lat=event_obj.place_coords_lat,
                    address_string=event_obj.place_address,
                )
            elif event_obj.place_address:
                coords = await get_coords_from_address(
                    address=event_obj.place_address,
                    city=collected_data.get("city_name", ""),
                )
                if coords and not (coords[0] == 0.0 and coords[1] == 0.0):
                    event_loc = LocationModel(
                        lon=coords[0],
                        lat=coords[1],
                        address_string=event_obj.place_address,
                    )

            if event_loc:
                points_for_single_route.append(event_loc)
            else:
                error_msg_coord = f"Не удалось определить координаты для '{event_obj.name}', маршрут может быть неполным."
                logger.warning(error_msg_coord)
                route_error_messages.append(error_msg_coord)
                overall_route_status = "partial_success"

        if len(points_for_single_route) < 2:
            logger.info(
                f"Not enough points for single-day route: {len(points_for_single_route)}"
            )
            final_error_msg = "Недостаточно точек для построения маршрута."
            if route_error_messages:
                final_error_msg = ". ".join(route_error_messages)
            return {
                "current_route_details": RouteDetails(
                    status="error", error_message=final_error_msg
                ),
                "is_full_plan_with_route_proposed": False,
            }

        tool_args_single_day = RouteBuilderToolArgs(
            start_point=points_for_single_route[0],
            event_points=points_for_single_route[1:],
        )
        logger.info(
            f"Building single-day route: {tool_args_single_day.model_dump_json(exclude_none=True, indent=2)}"
        )
        route_data_dict = await route_builder_tool.ainvoke(
            tool_args_single_day.model_dump(exclude_none=True)
        )
        try:
            single_day_route_details = RouteDetails(**route_data_dict)
            all_route_segments = single_day_route_details.segments or []
            total_duration_seconds_combined = (
                single_day_route_details.total_duration_seconds or 0
            )
            total_distance_meters_combined = (
                single_day_route_details.total_distance_meters or 0
            )
            if single_day_route_details.status != "success":
                overall_route_status = (
                    single_day_route_details.status
                    if single_day_route_details.status
                    in ["partial_success", "error", "api_error"]
                    else "error"
                )
                if single_day_route_details.error_message:
                    route_error_messages.append(single_day_route_details.error_message)
            elif overall_route_status == "success":
                overall_route_status = single_day_route_details.status
        except ValidationError as ve:
            logger.error(f"Validation error for single-day route data: {ve}")
            overall_route_status = "error"
            route_error_messages.append("Ошибка данных маршрута.")
            all_route_segments.append(
                RouteSegment(
                    segment_status="error",
                    segment_error_message="Ошибка данных маршрута",
                )
            )

    final_route_error_message = None
    if overall_route_status != "success":
        if route_error_messages:
            final_route_error_message = ". ".join(list(set(route_error_messages)))
        else:
            final_route_error_message = "Ошибка при построении маршрута."
            if overall_route_status == "partial_success":
                final_route_error_message = (
                    "Одна или несколько частей маршрута не могли быть построены."
                )

    final_route_details = RouteDetails(
        status=overall_route_status,
        segments=all_route_segments if all_route_segments else None,
        total_duration_seconds=(
            total_duration_seconds_combined
            if total_duration_seconds_combined > 0
            else None
        ),
        total_distance_meters=(
            total_distance_meters_combined
            if total_distance_meters_combined > 0
            else None
        ),
        total_duration_text=(
            f"~{round(total_duration_seconds_combined / 60)} мин"
            if total_duration_seconds_combined > 0
            else None
        ),
        total_distance_text=(
            f"~{round(total_distance_meters_combined / 1000, 1)} км"
            if total_distance_meters_combined > 0
            else None
        ),
        error_message=final_route_error_message,
    )
    return {
        "current_route_details": final_route_details,
        "is_full_plan_with_route_proposed": final_route_details.status
        in ["success", "partial_success"]
        and bool(final_route_details.segments),
    }


async def optimal_chain_constructor_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: optimal_chain_constructor_node executing...")
    candidate_events_by_interest: Dict[str, List[Event]] = state.get(
        "candidate_events_by_interest", {}
    )
    collected_data_dict: dict = dict(state.get("collected_data", {}))

    user_start_coords: Optional[Dict[str, float]] = collected_data_dict.get(
        "user_start_address_validated_coords"
    )
    parsed_dates_iso_list = collected_data_dict.get("parsed_dates_iso")
    raw_time_description_original = collected_data_dict.get(
        "raw_time_description_original"
    )

    user_min_start_dt_naive: Optional[datetime] = None
    if parsed_dates_iso_list and parsed_dates_iso_list[0]:
        user_min_start_dt_naive = datetime.fromisoformat(parsed_dates_iso_list[0])
        # Коррекция на 09:00, только если время НЕ было указано (т.е. raw_time_description пуст или не содержит цифр, указывающих на явное время)
        # и парсер вернул полночь.
        time_was_explicitly_parsed_by_llm_or_in_date_desc = False
        if raw_time_description_original and any(
            char.isdigit() for char in raw_time_description_original
        ):
            time_was_explicitly_parsed_by_llm_or_in_date_desc = True

        if (
            user_min_start_dt_naive.hour == 0
            and user_min_start_dt_naive.minute == 0
            and not time_was_explicitly_parsed_by_llm_or_in_date_desc
        ):
            user_min_start_dt_naive = user_min_start_dt_naive.replace(hour=9, minute=0)
            logger.info(
                f"OptimalChainConstructor: User min start time was 00:00 and no explicit time given, adjusted to {user_min_start_dt_naive}"
            )

    user_max_overall_end_dt_naive: Optional[datetime] = None
    if (
        collected_data_dict.get("parsed_end_dates_iso")
        and collected_data_dict["parsed_end_dates_iso"][0]
    ):
        user_max_overall_end_dt_naive = datetime.fromisoformat(
            collected_data_dict["parsed_end_dates_iso"][0]
        )
        if (
            user_max_overall_end_dt_naive.hour == 0
            and user_max_overall_end_dt_naive.minute == 0
            and user_max_overall_end_dt_naive.second == 0
        ):
            # Если это конец дня, но время 00:00, считаем это как 23:59:59 этого дня
            user_max_overall_end_dt_naive = user_max_overall_end_dt_naive.replace(
                hour=23, minute=59, second=59
            )
            logger.info(
                f"OptimalChainConstructor: User max overall end time was 00:00, adjusted to {user_max_overall_end_dt_naive}"
            )

    logger.info(
        f"OptimalChainConstructor: Effective user_min_start_dt_naive: {user_min_start_dt_naive}"
    )
    logger.info(
        f"OptimalChainConstructor: Effective user_max_overall_end_dt_naive: {user_max_overall_end_dt_naive}"
    )

    requested_interest_keys: List[str] = list(
        collected_data_dict.get("interests_keys_afisha", [])
    )
    TRAVEL_BUFFER_MINUTES = 15
    DEFAULT_EVENT_DURATION_MINUTES = 120
    best_chain: List[Event] = []
    min_total_travel_seconds = float("inf")
    construction_log: List[str] = []
    event_lists_for_product = []
    valid_interests_for_chain = []

    for interest_key in requested_interest_keys:
        candidates = candidate_events_by_interest.get(interest_key, [])
        valid_candidates_for_interest = [
            cand
            for cand in candidates
            if not (
                cand.place_coords_lon is None
                or cand.place_coords_lat is None
                or (cand.place_coords_lon == 0.0 and cand.place_coords_lat == 0.0)
            )
        ]

        # Дополнительная предварительная фильтрация кандидатов по времени
        # Событие должно начинаться не раньше user_min_start_dt_naive
        # И заканчиваться (start + duration) не позже user_max_overall_end_dt_naive (если оно есть)
        pre_filtered_candidates = []
        for cand in valid_candidates_for_interest:
            if (
                user_min_start_dt_naive
                and cand.start_time_naive_event_tz < user_min_start_dt_naive
            ):
                logger.debug(
                    f"Pre-filtering: Event {cand.name} for {interest_key} starts too early ({cand.start_time_naive_event_tz} < {user_min_start_dt_naive})"
                )
                continue
            if user_max_overall_end_dt_naive:
                cand_duration = cand.duration_minutes or DEFAULT_EVENT_DURATION_MINUTES
                cand_end_time = cand.start_time_naive_event_tz + timedelta(
                    minutes=cand_duration
                )
                if cand_end_time > user_max_overall_end_dt_naive:
                    logger.debug(
                        f"Pre-filtering: Event {cand.name} for {interest_key} ends too late ({cand_end_time} > {user_max_overall_end_dt_naive})"
                    )
                    continue
            pre_filtered_candidates.append(cand)

        if pre_filtered_candidates:
            # Берем топ-10 кандидатов, отсортированных по времени начала
            event_lists_for_product.append(
                sorted(
                    pre_filtered_candidates, key=lambda e: e.start_time_naive_event_tz
                )[:10]
            )  # Изменение: [:10]
            valid_interests_for_chain.append(interest_key)
            logger.info(
                f"OptimalChainConstructor: Added {len(event_lists_for_product[-1])} candidates for interest '{interest_key}' to product list."
            )
        else:
            logger.info(
                f"No suitable candidates (with coords & within initial time filter) for interest: {interest_key} to build a chain."
            )
            construction_log.append(
                f"Для интереса '{interest_key}' не найдено подходящих кандидатов для построения цепочки после предварительной фильтрации."
            )

    if not event_lists_for_product:
        logger.warning(
            "OptimalChainConstructor: No event lists for product after filtering, cannot build any chain."
        )
        return {
            "current_events": [],
            "unplanned_interest_keys": requested_interest_keys,
            "optimal_chain_construction_message": "Не удалось найти кандидатов для построения цепочки после первоначальной фильтрации.",
            "actual_total_travel_time": None,
            "collected_data": collected_data_dict,
        }

    if len(event_lists_for_product) < len(requested_interest_keys):
        logger.warning(
            f"OptimalChainConstructor: Not all requested interests have candidates for chain building. Requested: {len(requested_interest_keys)}, Have candidates for: {len(event_lists_for_product)}"
        )
        # Продолжаем пытаться построить цепочку из тех, что есть

    num_processed_permutations = 0
    max_permutations_to_check = 2000  # Увеличим, если кандидатов больше

    logger.info(
        f"OptimalChainConstructor: Starting chain construction with {len(event_lists_for_product)} interest lists."
    )

    for combination_tuple in itertools.product(*event_lists_for_product):
        if num_processed_permutations > max_permutations_to_check:
            logger.warning(
                f"OptimalChainConstructor: Reached max_permutations_to_check ({max_permutations_to_check}). Stopping early."
            )
            break

        for ordered_chain_candidate_list in itertools.permutations(combination_tuple):
            num_processed_permutations += 1
            if num_processed_permutations > max_permutations_to_check:
                break

            current_chain_is_valid = True
            current_chain_travel_seconds = 0

            logger.debug(
                f"Attempting chain ({num_processed_permutations}): {[e.name for e in ordered_chain_candidate_list]}"
            )

            if user_start_coords and user_min_start_dt_naive:
                current_time_available_naive = user_min_start_dt_naive
                last_event_coords = user_start_coords
            elif user_min_start_dt_naive:
                current_time_available_naive = user_min_start_dt_naive
                last_event_coords = None
            else:  # Этого не должно происходить, если extract_initial_info сработал
                current_time_available_naive = datetime.now().replace(hour=9, minute=0)
                last_event_coords = None

            temp_valid_chain_segment: List[Event] = []

            for i, current_event_in_chain in enumerate(ordered_chain_candidate_list):
                event_name_debug = current_event_in_chain.name
                logger.debug(
                    f"  Checking event {i+1}: {event_name_debug} (starts {current_event_in_chain.start_time_naive_event_tz.strftime('%H:%M')})"
                )
                logger.debug(
                    f"    Chain state before this event: Current time available: {current_time_available_naive.strftime('%H:%M')}, Last coords: {last_event_coords}"
                )

                travel_to_current_event_seconds = 0
                if last_event_coords:
                    try:
                        route_points = [
                            {
                                "lon": last_event_coords["lon"],
                                "lat": last_event_coords["lat"],
                            },
                            {
                                "lon": current_event_in_chain.place_coords_lon,
                                "lat": current_event_in_chain.place_coords_lat,
                            },
                        ]
                        route_response = await get_route(
                            points=route_points, transport="driving"
                        )
                        if route_response and route_response.get("status") == "success":
                            travel_to_current_event_seconds = route_response.get(
                                "duration_seconds", 0
                            )
                            logger.debug(
                                f"    Travel time to {event_name_debug}: {travel_to_current_event_seconds / 60:.1f} min"
                            )
                        else:
                            logger.debug(
                                f"    FAIL (Route error for {event_name_debug}): {route_response.get('message') if route_response else 'Unknown route error'}"
                            )
                            current_chain_is_valid = False
                            break
                    except Exception as e_route:
                        logger.error(
                            f"    FAIL (Route exception for {event_name_debug}): {e_route}"
                        )
                        current_chain_is_valid = False
                        break

                arrival_at_current_event_location_naive = (
                    current_time_available_naive
                    + timedelta(seconds=travel_to_current_event_seconds)
                )
                logger.debug(
                    f"    Arrival at {event_name_debug} location (after travel): {arrival_at_current_event_location_naive.strftime('%H:%M')}"
                )

                buffered_arrival_time = (
                    arrival_at_current_event_location_naive
                    + timedelta(minutes=TRAVEL_BUFFER_MINUTES)
                )
                logger.debug(
                    f"    Buffered arrival for {event_name_debug}: {buffered_arrival_time.strftime('%H:%M')}"
                )

                if (
                    buffered_arrival_time
                    > current_event_in_chain.start_time_naive_event_tz
                ):
                    logger.debug(
                        f"    FAIL (Too late for {event_name_debug}): Buffered arrival {buffered_arrival_time.strftime('%H:%M')} > Event start {current_event_in_chain.start_time_naive_event_tz.strftime('%H:%M')}"
                    )
                    current_chain_is_valid = False
                    break

                effective_event_start_time = max(
                    buffered_arrival_time,
                    current_event_in_chain.start_time_naive_event_tz,
                )
                logger.debug(
                    f"    Effective start for {event_name_debug}: {effective_event_start_time.strftime('%H:%M')}"
                )

                event_duration_actual_minutes = (
                    current_event_in_chain.duration_minutes
                    or DEFAULT_EVENT_DURATION_MINUTES
                )
                event_end_time_naive = effective_event_start_time + timedelta(
                    minutes=event_duration_actual_minutes
                )
                logger.debug(
                    f"    Calculated end for {event_name_debug}: {event_end_time_naive.strftime('%H:%M')} (duration {event_duration_actual_minutes}m)"
                )

                if (
                    user_max_overall_end_dt_naive
                    and event_end_time_naive > user_max_overall_end_dt_naive
                ):
                    logger.debug(
                        f"    FAIL (Event {event_name_debug} ends too late): Event end {event_end_time_naive.strftime('%H:%M')} > Plan limit {user_max_overall_end_dt_naive.strftime('%H:%M')}"
                    )
                    current_chain_is_valid = False
                    break

                current_chain_travel_seconds += travel_to_current_event_seconds
                current_time_available_naive = event_end_time_naive
                last_event_coords = {
                    "lon": current_event_in_chain.place_coords_lon,
                    "lat": current_event_in_chain.place_coords_lat,
                }
                temp_valid_chain_segment.append(current_event_in_chain)
                logger.debug(
                    f"    OK: Added {event_name_debug} to temp chain. Next available time: {current_time_available_naive.strftime('%H:%M')}"
                )

            if current_chain_is_valid:
                logger.info(
                    f"  Chain VALID: {[e.name for e in temp_valid_chain_segment]} with total travel time {current_chain_travel_seconds / 60:.1f} min"
                )
                if (
                    not best_chain
                    or current_chain_travel_seconds < min_total_travel_seconds
                ):
                    min_total_travel_seconds = current_chain_travel_seconds
                    best_chain = list(temp_valid_chain_segment)
                    logger.info(
                        f"    >> Found NEW BEST chain (shorter travel): {[e.name for e in best_chain]} with travel time {min_total_travel_seconds / 60:.1f} min"
                    )
                elif (
                    len(temp_valid_chain_segment) > len(best_chain)
                    and current_chain_travel_seconds <= min_total_travel_seconds * 1.15
                ):  # Допускаем 15% увеличение времени в пути для более длинной цепочки
                    min_total_travel_seconds = current_chain_travel_seconds
                    best_chain = list(temp_valid_chain_segment)
                    logger.info(
                        f"    >> Found NEW BEST chain (longer, comparable travel): {[e.name for e in best_chain]} with travel time {min_total_travel_seconds / 60:.1f} min"
                    )
            else:
                logger.debug(
                    f"  Chain INVALID: {[e.name for e in ordered_chain_candidate_list]}"
                )
        if num_processed_permutations > max_permutations_to_check:
            break

    unplanned_final_keys = list(requested_interest_keys)
    if best_chain:
        for event_in_best_chain in best_chain:
            if event_in_best_chain.event_type_key in unplanned_final_keys:
                unplanned_final_keys.remove(event_in_best_chain.event_type_key)

    if len(best_chain) < len(valid_interests_for_chain):
        best_chain_interest_keys = {e.event_type_key for e in best_chain}
        for valid_key in valid_interests_for_chain:
            if (
                valid_key not in best_chain_interest_keys
                and valid_key not in unplanned_final_keys
            ):
                unplanned_final_keys.append(valid_key)

    final_message = None
    if not best_chain and requested_interest_keys:
        final_message = "К сожалению, не удалось составить оптимальную цепочку мероприятий по вашим критериям."
        if construction_log:
            final_message += " " + " ".join(construction_log)
    elif unplanned_final_keys:
        final_message = f"Не удалось включить в оптимальный план мероприятия по следующим интересам: {', '.join(unplanned_final_keys)} из-за временных или логистических ограничений."
        if construction_log:
            filtered_log = [
                log_entry
                for log_entry in construction_log
                for key in unplanned_final_keys
                if key in log_entry
            ]
            if filtered_log:
                final_message += " Детали: " + " ".join(list(set(filtered_log)))

    logger.info(
        f"OptimalChainConstructor: Final best_chain: {[e.name for e in best_chain] if best_chain else 'None'}"
    )
    logger.info(f"OptimalChainConstructor: Unplanned keys: {unplanned_final_keys}")
    logger.info(f"OptimalChainConstructor: Construction message: {final_message}")

    return {
        "current_events": best_chain,
        "unplanned_interest_keys": unplanned_final_keys,
        "optimal_chain_construction_message": final_message,
        "actual_total_travel_time": min_total_travel_seconds if best_chain else None,
        "collected_data": collected_data_dict,
    }


# --- Узел 6: Представление полного плана (мероприятия + маршрут) ---
async def present_full_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_full_plan_node executing...")
    current_events_from_state: Optional[List[Any]] = state.get("current_events")
    current_route_details_obj: Optional[RouteDetails] = state.get(
        "current_route_details"
    )
    # Исправление 2: collected_data_dict -> collected_data
    collected_data: CollectedUserData = dict(
        state.get("collected_data", {})
    )  # Используем collected_data

    current_events: List[Event] = []
    if current_events_from_state:
        for evt_data in current_events_from_state:
            if isinstance(evt_data, Event):
                current_events.append(evt_data)
            elif isinstance(evt_data, dict):
                try:
                    current_events.append(Event(**evt_data))
                except ValidationError as e:
                    logger.warning(
                        f"present_full_plan_node: Could not validate event data: {evt_data}, error: {e}"
                    )
            else:
                logger.warning(
                    f"present_full_plan_node: Unknown event data type: {type(evt_data)}"
                )

    if not current_events:
        logger.warning("present_full_plan_node: No current events to present.")
        final_response_text = (
            "Мероприятий для отображения нет. Пожалуйста, начните новый поиск."
        )
        return {
            "messages": state.get("messages", [])
            + [AIMessage(content=final_response_text)],
            "status_message_to_user": final_response_text,
            "collected_data": {},
            "current_events": [],
            "current_route_details": None,
            "is_initial_plan_proposed": False,
            "is_full_plan_with_route_proposed": False,
            "awaiting_final_confirmation": False,
            "awaiting_fallback_confirmation": False,
            "pending_fallback_event": None,
            "candidate_events_by_interest": {},
            "unplanned_interest_keys": [],
            "optimal_chain_construction_message": None,
            "actual_total_travel_time": None,
        }

    response_parts = ["Вот ваш итоговый план:"]
    interest_key_to_name_singular = {
        "Movie": "фильм",
        "Performance": "спектакль",
        "Concert": "концерт",
        "Exhibition": "выставку",
        "SportEvent": "спортивное событие",
        "Excursion": "экскурсию",
        "Event": "другое событие",
        "Кафе": "кафе/ресторан",
        "Прогулки": "прогулку",
        "Музей": "музей",
    }

    for i, event in enumerate(current_events):
        event_time_str = event.start_time_naive_event_tz.strftime("%H:%M")
        event_date_str = event.start_time_naive_event_tz.strftime("%d.%m.%Y (%A)")
        desc = f"\n{i+1}. **{event.name}** ({interest_key_to_name_singular.get(event.event_type_key, event.event_type_key)})\n   *Место:* {event.place_name} ({event.place_address or 'Адрес не уточнен'})"
        desc += f"\n   *Время:* {event_date_str} в {event_time_str}"
        if event.duration_minutes:
            desc += f" (продолжительность ~{event.duration_minutes // 60}ч {event.duration_minutes % 60}м)"
        if event.price_text:
            desc += f"\n   *Цена:* {event.price_text}"
        elif event.min_price is not None:
            desc += f"\n   *Цена:* от {event.min_price} руб."
        response_parts.append(desc)

    event_dates_set: Set[date] = {
        evt.start_time_naive_event_tz.date() for evt in current_events
    }
    multiple_days = len(event_dates_set) > 1

    # Исправление 1.1: user_start_address_str должен быть взят из collected_data
    user_start_address_str = collected_data.get("user_start_address_original")

    if current_route_details_obj:
        if (
            current_route_details_obj.status in ["success", "partial_success"]
            and current_route_details_obj.segments
        ):
            response_parts.append("\nМаршрут:")
            if multiple_days:
                response_parts.append(
                    "  (Маршруты для каждого дня от вашего местоположения)"
                )
            for idx, segment in enumerate(current_route_details_obj.segments):
                from_name = segment.from_address
                if (
                    not from_name
                ):  # Если API не вернуло from_address, пытаемся его определить
                    if idx == 0 and user_start_address_str:
                        from_name = user_start_address_str
                    elif idx > 0 and idx - 1 < len(current_events):
                        from_name = current_events[idx - 1].name
                    else:
                        from_name = (
                            "Предыдущая точка"
                            if not multiple_days
                            else "Ваше местоположение"
                        )

                to_name = segment.to_address
                if not to_name and idx < len(
                    current_events
                ):  # Если API не вернуло to_address
                    # Для маршрутов от пользователя к каждому событию (multiple_days)
                    # или для последовательного маршрута, где idx соответствует событию
                    if multiple_days or user_start_address_str:
                        if idx < len(current_events):
                            to_name = current_events[idx].name
                    # Для маршрута между событиями (нет адреса пользователя, idx > 0)
                    elif (
                        not multiple_days
                        and not user_start_address_str
                        and idx + 1 < len(current_events)
                    ):
                        to_name = current_events[
                            idx + 1
                        ].name  # Сегмент ведет ко ВТОРОМУ, ТРЕТЬЕМУ и т.д. событию
                    elif (
                        not multiple_days
                        and not user_start_address_str
                        and idx < len(current_events)
                    ):  # Первый сегмент от первого события
                        if idx + 1 < len(current_events):
                            to_name = current_events[idx + 1].name

                if not to_name:
                    to_name = f"Мероприятие {idx+1}"

                segment_text = f"  {idx+1}. От '{from_name}' до '{to_name}': "
                if segment.segment_status == "success":
                    segment_text += f"{segment.duration_text or '? мин'}, {segment.distance_text or '? км'}."
                else:
                    segment_text += f"не удалось построить ({segment.segment_error_message or 'причина неизвестна'})."
                response_parts.append(segment_text)

            if current_route_details_obj.status == "partial_success":
                response_parts.append("\n  Не все части маршрута удалось построить.")
            elif (
                not multiple_days
                and current_route_details_obj.total_duration_text
                and len(current_route_details_obj.segments) >= 1
            ):
                response_parts.append(
                    f"\n  Общее время в пути по маршруту: {current_route_details_obj.total_duration_text}."
                )
        elif current_route_details_obj.error_message:
            response_parts.append(
                f"\nМаршрут: Не удалось построить ({current_route_details_obj.error_message})."
            )
        elif not current_route_details_obj.segments and current_events:
            response_parts.append(
                "\nМаршрут не строился, так как у вас одно мероприятие и не указан начальный адрес."
            )

    response_parts.append(
        "\n\nПлан окончательный. Если захотите что-то еще, просто напишите новый запрос!"
    )
    full_plan_text = "\n".join(response_parts)
    new_messages = state.get("messages", []) + [AIMessage(content=full_plan_text)]

    return {
        "messages": new_messages,
        "status_message_to_user": full_plan_text,
        "collected_data": {},
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,
        "awaiting_fallback_confirmation": False,
        "pending_fallback_event": None,
        "candidate_events_by_interest": {},
        "unplanned_interest_keys": [],
        "optimal_chain_construction_message": None,
        "actual_total_travel_time": None,
    }


# --- Узел 7: Обработка обратной связи по плану ---
async def handle_plan_feedback_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: handle_plan_feedback_node executing...")
    messages = state.get("messages", [])
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))
    current_events_list: Optional[List[Any]] = state.get("current_events")

    current_events: List[Event] = []
    if current_events_list:
        for evt_data in current_events_list:
            if isinstance(evt_data, Event):
                current_events.append(evt_data)
            elif isinstance(evt_data, dict):
                try:
                    current_events.append(Event(**evt_data))
                except ValidationError:
                    pass

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("handle_plan_feedback_node: No human feedback message found.")
        return {
            "status_message_to_user": "Не получил вашего ответа по плану.",
            "awaiting_final_confirmation": True,
            "collected_data": collected_data,
            "messages": messages,
        }

    user_feedback = messages[-1].content
    llm = get_gigachat_client()
    structured_llm_feedback = llm.with_structured_output(AnalyzedFeedback)
    plan_summary_parts = []
    if current_events:
        for i, event in enumerate(current_events):
            plan_summary_parts.append(
                f"Мероприятие {i+1}: {event.name} ({event.start_time_naive_event_tz.strftime('%d.%m %H:%M')})"
            )
    current_route_details: Optional[RouteDetails] = state.get("current_route_details")
    if current_route_details and current_route_details.total_duration_text:
        plan_summary_parts.append(
            f"Маршрут: ~{current_route_details.total_duration_text}"
        )
    current_plan_summary_str = (
        "\n".join(plan_summary_parts)
        if plan_summary_parts
        else "План пока не сформирован."
    )
    prompt_for_feedback_analysis = PLAN_FEEDBACK_ANALYSIS_PROMPT.format(
        current_plan_summary=current_plan_summary_str, user_feedback=user_feedback
    )

    try:
        analyzed_result: AnalyzedFeedback = await structured_llm_feedback.ainvoke(
            prompt_for_feedback_analysis
        )
        logger.info(
            f"LLM Analyzed Feedback: {analyzed_result.model_dump_json(indent=2)}"
        )
        intent = analyzed_result.intent_type
        changes = analyzed_result.change_details

        preserved_user_address = collected_data.get("user_start_address_original")
        preserved_user_coords = collected_data.get(
            "user_start_address_validated_coords"
        )

        next_state_update: Dict[str, Any] = {
            "awaiting_final_confirmation": False,
            "collected_data": {},
            "current_events": [],
        }

        if preserved_user_address:
            next_state_update["collected_data"][
                "user_start_address_original"
            ] = preserved_user_address
        if preserved_user_coords:
            next_state_update["collected_data"][
                "user_start_address_validated_coords"
            ] = preserved_user_coords

        if intent == "confirm_plan":
            logger.info("User confirmed the plan.")
            next_state_update["status_message_to_user"] = (
                "Отлично! Рад был помочь. Если понадобится что-то еще, обращайтесь!"
            )
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
            next_state_update["collected_data"] = {}
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None

        elif intent == "request_change" and changes:
            logger.info(f"User requested changes: {changes}")
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None

            new_collected_data_for_change = dict(next_state_update["collected_data"])
            new_collected_data_for_change["clarification_needed_fields"] = []

            change_target = changes.get("target")
            new_value = changes.get("value")

            if change_target == "budget":
                if isinstance(new_value, (int, float)):
                    new_collected_data_for_change["budget_current_search"] = int(
                        new_value
                    )
                    new_collected_data_for_change["budget_original"] = int(new_value)
                else:
                    new_collected_data_for_change.setdefault(
                        "clarification_needed_fields", []
                    ).append("budget_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите бюджет числом. Вы: '{new_value}'."
                    )
            elif change_target in ["date", "time", "dates_description"]:
                if isinstance(new_value, str):
                    parsed_date_res = await datetime_parser_tool.ainvoke(
                        {
                            "natural_language_date": new_value,
                            "base_date_iso": datetime.now().isoformat(),
                        }
                    )
                    if parsed_date_res.get("datetime_iso"):
                        new_collected_data_for_change["parsed_dates_iso"] = [
                            parsed_date_res["datetime_iso"]
                        ]
                        new_collected_data_for_change["dates_description_original"] = (
                            new_value
                        )
                        new_collected_data_for_change["parsed_end_dates_iso"] = (
                            [parsed_date_res["end_datetime_iso"]]
                            if parsed_date_res.get("end_datetime_iso")
                            else None
                        )
                        if parsed_date_res.get("is_ambiguous"):
                            new_collected_data_for_change.setdefault(
                                "clarification_needed_fields", []
                            ).append("dates_description_original")
                            next_state_update["clarification_context"] = (
                                parsed_date_res.get("clarification_needed")
                            )
                    else:
                        new_collected_data_for_change.setdefault(
                            "clarification_needed_fields", []
                        ).append("dates_description_original")
                        next_state_update["clarification_context"] = (
                            f"Не удалось распознать '{new_value}'. Уточните."
                        )
                else:
                    new_collected_data_for_change.setdefault(
                        "clarification_needed_fields", []
                    ).append("dates_description_original")
                    next_state_update["clarification_context"] = (
                        f"Опишите дату/время текстом. Вы: '{new_value}'."
                    )
            elif change_target in [
                "interests",
                "type",
                "event_type",
            ] or "event_" in str(change_target):
                new_interests_str_list = (
                    [new_value]
                    if isinstance(new_value, str)
                    else (
                        new_value
                        if isinstance(new_value, list)
                        and all(isinstance(s, str) for s in new_value)
                        else []
                    )
                )
                if new_interests_str_list:
                    new_collected_data_for_change["interests_original"] = (
                        new_interests_str_list
                    )
                    mapped_keys = []
                    for s_int in new_interests_str_list:
                        s_l = s_int.lower()
                        key_afisha = None
                        if "фильм" in s_l or "кино" in s_l:
                            key_afisha = "Movie"
                        elif "концерт" in s_l:
                            key_afisha = "Concert"
                        elif "театр" in s_l or "спектакль" in s_l:
                            key_afisha = "Performance"
                        if not key_afisha:
                            key_afisha = s_int.capitalize()
                        mapped_keys.append(key_afisha)
                    new_collected_data_for_change["interests_keys_afisha"] = list(
                        set(mapped_keys)
                    )
                else:
                    new_collected_data_for_change.setdefault(
                        "clarification_needed_fields", []
                    ).append("interests_original")
                    next_state_update["clarification_context"] = (
                        f"Назовите интересы. Вы: '{new_value}'."
                    )
            elif change_target == "start_location" or change_target == "address":
                if isinstance(new_value, str):
                    new_collected_data_for_change["user_start_address_original"] = (
                        new_value
                    )
                    new_collected_data_for_change[
                        "user_start_address_validated_coords"
                    ] = None
                else:
                    new_collected_data_for_change.setdefault(
                        "clarification_needed_fields", []
                    ).append("user_start_address_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите адрес. Вы: '{new_value}'."
                    )
            elif change_target == "city":
                new_collected_data_for_change["city_name"] = (
                    str(new_value) if new_value else None
                )
                new_collected_data_for_change["city_id_afisha"] = None
            else:
                logger.warning(f"Unknown change_target: {change_target}")
                next_state_update["status_message_to_user"] = (
                    "Не совсем понял, что именно вы хотите изменить. Попробуйте переформулировать, пожалуйста."
                )
                next_state_update["awaiting_final_confirmation"] = True
                next_state_update["collected_data"] = collected_data
                next_state_update["current_events"] = (
                    list(current_events) if current_events else []
                )

            if "status_message_to_user" not in next_state_update:
                next_state_update["collected_data"] = new_collected_data_for_change

            next_state_update["pending_plan_modification_request"] = None

        elif intent == "new_search":
            next_state_update["status_message_to_user"] = (
                "Хорошо, давайте начнем новый поиск. Что бы вы хотели найти?"
            )
            next_state_update["collected_data"] = {}
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
        else:
            next_state_update["status_message_to_user"] = (
                "Я вас не совсем понял. Попробуете еще раз или новые критерии?"
            )
            next_state_update["awaiting_final_confirmation"] = True
            next_state_update["collected_data"] = collected_data
            next_state_update["current_events"] = (
                list(current_events) if current_events else []
            )

        if next_state_update.get("status_message_to_user"):
            next_state_update["messages"] = messages + [
                AIMessage(content=next_state_update["status_message_to_user"])
            ]
        else:
            next_state_update["messages"] = messages
        return next_state_update
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}", exc_info=True)
        msg = "Ошибка при обработке вашего ответа. Попробуйте, пожалуйста, еще раз."
        return {
            "status_message_to_user": msg,
            "awaiting_final_confirmation": True,
            "messages": messages + [AIMessage(content=msg)],
            "collected_data": collected_data,
            "current_events": list(current_events) if current_events else [],
        }


# --- Узел 8: Подтверждение изменений (для Примера 2 из инструкции) ---
async def confirm_changes_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: confirm_changes_node executing (likely deprecated)...")
    messages = state.get("messages", [])
    pending_modification = state.get("pending_plan_modification_request", {})
    if not pending_modification:
        return {
            "status_message_to_user": "Что-то пошло не так при подтверждении изменений.",
            "messages": messages,
            "collected_data": state.get("collected_data"),
        }
    change_summary = "; ".join([f"{k}: {v}" for k, v in pending_modification.items()])
    confirmation_question = (
        f"Правильно понимаю, вы хотите изменить: {change_summary}? (да/нет)"
    )
    new_messages = messages + [AIMessage(content=confirmation_question)]
    return {
        "messages": new_messages,
        "status_message_to_user": confirmation_question,
        "collected_data": state.get("collected_data"),
    }


# --- Узел 9: Сообщение об ошибке, если мероприятия не найдены ---
async def error_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: error_node executing...")
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))
    error_msg_parts = []
    optimal_chain_msg = state.get("optimal_chain_construction_message")
    if optimal_chain_msg:
        error_msg_parts.append(optimal_chain_msg)
    search_errors = collected_data.get("search_errors_by_interest", {})
    if search_errors:
        for interest, err in search_errors.items():
            if interest not in (optimal_chain_msg or ""):
                error_msg_parts.append(f"По интересу '{interest}': {err}")

    if not error_msg_parts:
        original_interests: List[str] = collected_data.get("interests_original", [])
        search_criteria_parts = []
        if collected_data.get("city_name"):
            search_criteria_parts.append(f"город '{collected_data['city_name']}'")
        if collected_data.get("dates_description_original"):
            search_criteria_parts.append(
                f"даты '{collected_data['dates_description_original']}'"
            )
        if original_interests:
            search_criteria_parts.append(f"интересы '{', '.join(original_interests)}'")
        search_criteria_summary = (
            ", ".join(search_criteria_parts)
            if search_criteria_parts
            else "указанным вами критериям"
        )
        error_msg_parts.append(
            EVENT_NOT_FOUND_PROMPT_TEMPLATE.format(
                search_criteria_summary=search_criteria_summary
            )
        )

    final_error_msg = "\n".join(list(filter(None, error_msg_parts)))
    if not final_error_msg.strip():
        final_error_msg = "К сожалению, по вашему запросу ничего не удалось найти. Попробуете изменить критерии?"

    if "search_errors_by_interest" in collected_data:
        del collected_data["search_errors_by_interest"]
    if "fallback_candidates" in collected_data:
        del collected_data["fallback_candidates"]
    if "last_offered_fallback_for_interest" in collected_data:
        del collected_data["last_offered_fallback_for_interest"]
    if "pending_fallback_event" in collected_data:
        collected_data["pending_fallback_event"] = None
    if "awaiting_fallback_confirmation" in collected_data:
        collected_data["awaiting_fallback_confirmation"] = False

    new_messages = state.get("messages", []) + [AIMessage(content=final_error_msg)]
    return {
        "messages": new_messages,
        "status_message_to_user": final_error_msg,
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,
        "collected_data": collected_data,
        "candidate_events_by_interest": {},
        "unplanned_interest_keys": [],
        "optimal_chain_construction_message": None,
    }
