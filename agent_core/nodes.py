import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, date, timedelta, time
import asyncio
import re
from agent_core.schedule_parser import parse_schedule_and_check_open
from schemas.data_schemas import (
    AnalyzedFeedback,
    LocationModel,
    RouteDetails,
    RouteSegment,
    Event,
    ExtractedInitialInfo,
    DateTimeParserToolArgs,
    EventSearchToolArgs,
    RouteBuilderToolArgs,
    ParsedDateTime,
    ChangeRequestDetail,
)
from prompts.system_prompts import (
    PLAN_FEEDBACK_ANALYSIS_PROMPT,
    INITIAL_INFO_EXTRACTION_PROMPT,
    GENERAL_CLARIFICATION_PROMPT_TEMPLATE,
    TIME_CLARIFICATION_PROMPT_TEMPLATE,
    CHANGE_CONFIRMATION_PROMPT_TEMPLATE,
    EVENT_NOT_FOUND_PROMPT_TEMPLATE,
)
import itertools
from services.gis_service import (
    search_parks,
    search_food_places,
    get_coords_from_address,
    get_route,
    get_geocoding_details,
    GeocodingResult,
)
from tools.route_builder_tool import route_builder_tool
from tools.datetime_parser_tool import datetime_parser_tool
from tools.event_search_tool import event_search_tool
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from agent_core.agent_state import AgentState, CollectedUserData, LastPresentedPlanInfo
from llm_interface.gigachat_client import get_gigachat_client

# В файле agent_core/nodes.py

# ...
from services.afisha_service import fetch_cities, CREATION_TYPES_AFISHA  # Вот так

# ...

# Инициализация логгера
logger = logging.getLogger(__name__)


GENRE_SLUGS_TO_RUSSIAN = {
    "actionfilms": "Боевик",
    "thrillers": "Триллер",
    "comedies": "Комедия",
    "dramas": "Драма",
    "horrors": "Ужасы",
    "family": "Семейный",
    "cartoons": "Мультфильм",
    "fantasy": "Фэнтези",
    "fantastic": "Фантастика",
    "adventures": "Приключения",
    "detective": "Детектив",
    "military": "Военный",
    "historical": "Исторический",
    "documentary": "Документальный",
    "biography": "Биография",
    "musical": "Мюзикл",
    "melodramas": "Мелодрама",
    "animation": "Анимация",  # Часто дублирует cartoons, но может быть отдельно
    "kids": "Детский",
    "crime": "Криминал",
    "western": "Вестерн",
    "romance": "Романтика",  # Может быть частью мелодрам
    "sport": "Спорт",
    "sci-fi": "Научная фантастика",  # Часто синоним fantastic
    "noir": "Нуар",
    "arthouse": "Артхаус",
    "painting": "Живопись",
    "classicalart": "Классическое искусство",
    "humor": "Юмор",
    "screenadaptations": "Юмор",
    "dramafilms": "Драма",
    "war": "Военный",
    "adventure": "Приключение",
    # Дополните этот словарь по мере необходимости, проверяя, какие слаги возвращает ваше API
}


# --- Узел 1: Извлечение начальной информации ---
# Файл: agent_core/nodes.py
DAYS_RU = {
    "Monday": "понедельник",
    "Tuesday": "вторник",
    "Wednesday": "среда",
    "Thursday": "четверг",
    "Friday": "пятница",
    "Saturday": "суббота",
    "Sunday": "воскресенье",
}
# ... (предполагаемые обновленные импорты выше) ...


def escape_markdown_v2(text: str) -> str:
    if not text:
        return ""
    escape_chars = r"_*[]()~`>#+-.=|{}!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def _extract_budget_from_query_details_regex(
    query_details: Optional[str],
) -> Optional[int]:
    if not query_details:
        return None

    budget_patterns = [
        r"(?:до|не более|дешевле|не дороже|чек)\s*(\d+)\s*(?:р|руб|рублей)?",
        r"(\d+)\s*(?:р|руб|рублей)\s*(?:чек|бюджет)",
        r"бюджет\s*(\d+)",
        r"стоимостью\s*(\d+)",
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, query_details, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def format_genres_for_telegram(genres: Optional[List[str]]) -> str:
    if not genres:
        return ""
    # --- ОБНОВЛЕННАЯ ЛОГИКА ---
    russian_genres = []
    for slug in genres:
        russian_genres.append(
            GENRE_SLUGS_TO_RUSSIAN.get(slug.strip().lower(), slug.capitalize())
        )
    return ", ".join(russian_genres)


from schemas.data_schemas import (
    ExtractedInitialInfo,
    OrderedActivityItem,  # Добавил OrderedActivityItem для ясности
    Event,  # Убедитесь, что Event импортирован
    ChangeRequestDetail,  # Если используется для типизации modification_details_update_llm
)  # Добавил OrderedActivityItem для ясности

# ... (функция _check_event_compatibility, если она здесь) ...


async def extract_initial_info_node(state: AgentState) -> Dict[str, Any]:
    # REFACTORED: Финальная, архитектурно исправленная версия.
    # - FIX: Логика обработки ответов на все виды fallback-предложений (`budget_fallback_confirmation`,
    #   `fallback_confirmation_response`, `combo_fallback_confirmation`) ВЫНЕСЕНА В САМОЕ НАЧАЛО функции.
    #   Теперь это первое, что проверяется после получения сообщения от пользователя.
    # - Это гарантирует, что эти специфические ответы обрабатываются с наивысшим приоритетом
    #   и функция немедленно завершается (`return`), не доходя до общей логики парсинга и уточнений.
    # - Вся остальная логика (сброс, уточнения, основной парсинг) осталась нетронутой.
    logger.info("Node: extract_initial_info_node executing...")
    awaiting_clarification_field: Optional[str] = state.get(
        "awaiting_clarification_for_field"
    )
    messages: List[BaseMessage] = state.get("messages", [])
    original_collected_data: dict = state.get("collected_data", {})
    is_awaiting_feedback_on_final_plan: bool = state.get(
        "awaiting_feedback_on_final_plan", False
    )
    current_collected_data_dict: dict = dict(original_collected_data)

    excluded_ids_map = current_collected_data_dict.setdefault(
        "current_excluded_ids", {}
    )
    for key_ex_ids in [
        "afisha",
        "park",
        "food",
        "afisha_names_to_avoid",
        "afisha_creation_ids_to_avoid",
    ]:
        excluded_ids_map.setdefault(key_ex_ids, [])

    if "rejected_fallback_for_interests" not in current_collected_data_dict:
        current_collected_data_dict["rejected_fallback_for_interests"] = []

    node_return_values: Dict[str, Any] = {key: value for key, value in state.items()}
    node_return_values["collected_data"] = current_collected_data_dict
    node_return_values["just_modified_plan"] = state.get("just_modified_plan", False)

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("extract_initial_info_node: No new human message to process.")
        node_return_values["clarification_context"] = (
            "Пожалуйста, ответьте на предыдущий вопрос."
        )
        node_return_values["awaiting_clarification_for_field"] = (
            awaiting_clarification_field
        )
        return node_return_values

    user_query = messages[-1].content.strip()
    user_query_lower = user_query.lower()

    # --- НАЧАЛО БЛОКА ИСПРАВЛЕНИЙ ---
    # Приоритетная обработка ответов на Fallback-вопросы
    is_awaiting_fallback_confirm_local = current_collected_data_dict.get(
        "awaiting_fallback_confirmation", False
    )
    last_offered_interest_for_fb = current_collected_data_dict.get(
        "last_offered_fallback_for_interest"
    )

    if awaiting_clarification_field == "budget_fallback_confirmation":
        logger.info("Handling user response for 'budget_fallback_confirmation'")

        # FIX: Улучшена проверка ответа и добавлено диагностическое логирование
        user_query_clean = user_query_lower.strip()  # Очищаем пробелы
        logger.debug(f"Cleaned user query: '{user_query_clean}'")
        is_positive_response = (
            user_query_clean == "да"  # Явное совпадение с "да"
            or any(
                keyword in user_query_clean
                for keyword in [
                    "да",
                    "согласен",
                    "давай",
                    "хочу",
                    "добавить",
                    "конечно",
                    "увеличиваем",
                ]
            )
        )
        logger.debug(f"Is positive response: {is_positive_response}")

        pending_plan_data = current_collected_data_dict.get("pending_fallback_event")
        logger.debug(f"Pending plan data: {pending_plan_data}")

        # FIX: Добавлена проверка на наличие и корректность pending_plan_data
        if not pending_plan_data or not isinstance(pending_plan_data, dict):
            logger.error("No valid pending_fallback_event data found.")
            node_return_values["clarification_context"] = (
                "Произошла ошибка: данные о предложенном плане отсутствуют. Пожалуйста, попробуйте начать новый поиск."
            )
            current_collected_data_dict["awaiting_fallback_confirmation"] = False
            current_collected_data_dict["pending_fallback_event"] = None
            current_collected_data_dict["budget_fallback_plan"] = None
            current_collected_data_dict["fallback_accepted_and_plan_updated"] = False
            node_return_values["just_modified_plan"] = False
            node_return_values["collected_data"] = current_collected_data_dict
            node_return_values["awaiting_clarification_for_field"] = None
            node_return_values["clarification_context"] = (
                "Произошла ошибка: данные о предложенном плане отсутствуют."
            )
            logger.info(
                f"Budget fallback processed. Accepted: False, Current events: {len(node_return_values['current_events'])}, Just modified plan: False"
            )
            return node_return_values

        # Очищаем временные данные
        current_collected_data_dict["awaiting_fallback_confirmation"] = False
        current_collected_data_dict["pending_fallback_event"] = None
        current_collected_data_dict["budget_fallback_plan"] = None

        if is_positive_response:
            logger.info(
                "User accepted over-budget plan. Promoting it to current_events."
            )
            plan_items = pending_plan_data.get("plan_items", [])
            new_current_events = []
            new_selected_pois = []
            for item_data in plan_items:
                if (
                    "session_id" in item_data
                    or item_data.get("_plan_item_type") == "afisha_event"
                ):
                    try:
                        new_current_events.append(Event(**item_data))
                    except ValidationError as e:
                        logger.error(f"Error validating budget fallback event: {e}")
                        node_return_values["clarification_context"] = (
                            "Произошла ошибка при обработке плана, превышающего бюджет."
                        )
                        current_collected_data_dict[
                            "fallback_accepted_and_plan_updated"
                        ] = False
                        node_return_values["just_modified_plan"] = False
                else:
                    new_selected_pois.append(item_data)

            node_return_values["current_events"] = new_current_events
            current_collected_data_dict["selected_pois_for_plan"] = new_selected_pois
            current_collected_data_dict["budget_current_search"] = (
                pending_plan_data.get("cost")
            )
            current_collected_data_dict["fallback_accepted_and_plan_updated"] = True
            node_return_values["just_modified_plan"] = True
        else:
            logger.info(
                f"User rejected over-budget plan. Response: '{user_query_clean}'"
            )
            current_collected_data_dict["fallback_accepted_and_plan_updated"] = False
            node_return_values["just_modified_plan"] = False

        node_return_values["collected_data"] = current_collected_data_dict
        node_return_values["awaiting_clarification_for_field"] = None
        node_return_values["clarification_context"] = None

        logger.info(
            f"Budget fallback processed. Accepted: {current_collected_data_dict['fallback_accepted_and_plan_updated']}, "
            f"Current events: {len(node_return_values['current_events'])}, "
            f"Just modified plan: {node_return_values['just_modified_plan']}"
        )
        return node_return_values

    if (
        awaiting_clarification_field == "fallback_confirmation_response"
        and is_awaiting_fallback_confirm_local
        and last_offered_interest_for_fb
    ):
        is_positive_response = any(
            word in user_query_lower for word in ["да", "хочу", "добавить", "согласен"]
        )
        pending_fallback_event_data = current_collected_data_dict.get(
            "pending_fallback_event"
        )

        current_collected_data_dict["awaiting_fallback_confirmation"] = False
        node_return_values.update(
            {
                "awaiting_clarification_for_field": None,
                "clarification_context": None,
                "awaiting_feedback_on_final_plan": False,
                "modification_request_details": None,
                "plan_modification_pending": False,
                "messages": messages,
                "just_modified_plan": False,
            }
        )

        current_collected_data_dict["pending_fallback_event"] = None
        current_collected_data_dict["last_offered_fallback_for_interest"] = None

        if (
            is_positive_response
            and pending_fallback_event_data
            and isinstance(pending_fallback_event_data, dict)
        ):
            try:
                pending_fallback_event_obj = Event(**pending_fallback_event_data)
                current_events_obj_list: List[Event] = list(
                    state.get("current_events", [])
                )
                if not any(
                    evt.session_id == pending_fallback_event_obj.session_id
                    for evt in current_events_obj_list
                ):
                    current_events_obj_list.append(pending_fallback_event_obj)
                    current_events_obj_list.sort(
                        key=lambda e_obj: e_obj.start_time_naive_event_tz
                    )

                new_ordered_activities = []
                for evt in current_events_obj_list:
                    new_ordered_activities.append(
                        OrderedActivityItem(
                            activity_type=evt.user_event_type_key,
                            query_details=evt.name,
                        )
                    )
                current_collected_data_dict["ordered_activities"] = [
                    oa.model_dump(exclude_none=True) for oa in new_ordered_activities
                ]
                logger.info(
                    f"Fallback accepted. Cemented new ordered_activities: {current_collected_data_dict['ordered_activities']}"
                )

                if current_events_obj_list:
                    default_duration_for_calc = 120

                    max_end_time = max(
                        (
                            e.start_time_naive_event_tz
                            + timedelta(
                                minutes=e.duration_minutes or default_duration_for_calc
                            )
                        )
                        for e in current_events_obj_list
                    )
                    min_start_time = min(
                        e.start_time_naive_event_tz for e in current_events_obj_list
                    )

                    current_collected_data_dict["parsed_dates_iso"] = [
                        min_start_time.isoformat()
                    ]
                    current_collected_data_dict["parsed_end_dates_iso"] = [
                        max_end_time.isoformat()
                    ]
                    current_collected_data_dict["dates_description_original"] = (
                        f"План на разные даты (с {min_start_time.strftime('%d.%m')} по {max_end_time.strftime('%d.%m')})"
                    )
                    logger.info(
                        f"Time window updated to accommodate fallback: start_date is now {min_start_time.isoformat()}, end_date is now {max_end_time.isoformat()}"
                    )

                node_return_values["current_events"] = current_events_obj_list

                nf_primary: List[str] = list(
                    current_collected_data_dict.get(
                        "not_found_interest_keys_in_primary_search", []
                    )
                )
                if (
                    last_offered_interest_for_fb
                    and last_offered_interest_for_fb in nf_primary
                ):
                    nf_primary.remove(last_offered_interest_for_fb)
                current_collected_data_dict[
                    "not_found_interest_keys_in_primary_search"
                ] = nf_primary

                if last_offered_interest_for_fb in current_collected_data_dict.get(
                    "fallback_candidates", {}
                ):
                    del current_collected_data_dict["fallback_candidates"][
                        last_offered_interest_for_fb
                    ]

                current_collected_data_dict["fallback_accepted_and_plan_updated"] = True
                node_return_values["is_initial_plan_proposed"] = True
                node_return_values["just_modified_plan"] = True

            except ValidationError as ve_fb_extract_node:
                logger.error(
                    f"Error validating fallback event data: {ve_fb_extract_node}"
                )
                current_collected_data_dict["fallback_accepted_and_plan_updated"] = (
                    False
                )
                node_return_values["clarification_context"] = (
                    "Произошла ошибка с обработкой вашего ответа по запасному варианту."
                )
        else:
            logger.info(f"User rejected fallback for {last_offered_interest_for_fb}.")
            current_collected_data_dict["fallback_accepted_and_plan_updated"] = False

            if last_offered_interest_for_fb not in current_collected_data_dict.get(
                "rejected_fallback_for_interests", []
            ):
                current_collected_data_dict.setdefault(
                    "rejected_fallback_for_interests", []
                ).append(last_offered_interest_for_fb)

            if last_offered_interest_for_fb in current_collected_data_dict.get(
                "fallback_candidates", {}
            ):
                del current_collected_data_dict["fallback_candidates"][
                    last_offered_interest_for_fb
                ]

            logger.info(
                f"Updated rejected_fallback_for_interests: {current_collected_data_dict['rejected_fallback_for_interests']}"
            )
            node_return_values["is_initial_plan_proposed"] = True
            node_return_values["just_modified_plan"] = False

        node_return_values["collected_data"] = current_collected_data_dict
        return node_return_values

    if awaiting_clarification_field == "combo_fallback_confirmation":
        logger.info("Handling user response for 'combo_fallback_confirmation'")
        is_positive_response = any(
            word in user_query_lower for word in ["да", "хочу", "добавить", "согласен"]
        )
        pending_fallback_event_data = current_collected_data_dict.get(
            "pending_fallback_event"
        )

        current_collected_data_dict["awaiting_fallback_confirmation"] = False
        node_return_values.update(
            {"awaiting_clarification_for_field": None, "clarification_context": None}
        )

        if is_positive_response and isinstance(pending_fallback_event_data, dict):
            try:
                pending_fallback_event_obj = Event(**pending_fallback_event_data)
                current_events_obj_list: List[Event] = list(
                    state.get("current_events", [])
                )
                current_events_obj_list.append(pending_fallback_event_obj)
                current_events_obj_list.sort(
                    key=lambda e_obj: e_obj.start_time_naive_event_tz
                )
                node_return_values["current_events"] = current_events_obj_list
                current_collected_data_dict["fallback_accepted_and_plan_updated"] = True
            except ValidationError as e:
                logger.error(f"Error validating combo fallback event: {e}")
                current_collected_data_dict["fallback_accepted_and_plan_updated"] = (
                    False
                )
        else:
            logger.info(
                f"User rejected combo fallback for {last_offered_interest_for_fb}."
            )
            current_collected_data_dict["fallback_accepted_and_plan_updated"] = False
            if last_offered_interest_for_fb:
                current_collected_data_dict.setdefault(
                    "rejected_fallback_for_interests", []
                ).append(last_offered_interest_for_fb)

        current_collected_data_dict["pending_fallback_event"] = None
        current_collected_data_dict["last_offered_fallback_for_interest"] = None
        current_collected_data_dict["combo_fallback_candidates"] = {}
        node_return_values["collected_data"] = current_collected_data_dict
        return node_return_values

    # --- КОНЕЦ БЛОКА ИСПРАВЛЕНИЙ ---

    reset_commands = [
        "новый поиск",
        "начни сначала",
        "отмена",
        "сброс",
        "стоп",
        "/start",
    ]
    if any(cmd in user_query_lower for cmd in reset_commands):
        logger.info(f"User requested reset with: '{user_query}'")
        clarification_msg = (
            "Хорошо, начинаем новый поиск! Что ищем (город, даты, интересы, бюджет)?"
        )
        new_messages_for_reset = messages + [AIMessage(content=clarification_msg)]
        reset_collected_data = {
            "current_excluded_ids": {
                "afisha": [],
                "park": [],
                "food": [],
                "afisha_names_to_avoid": [],
                "afisha_creation_ids_to_avoid": [],
            },
            "rejected_fallback_for_interests": [],
        }
        if original_collected_data.get("user_start_address_validated_coords"):
            reset_collected_data["user_start_address_original"] = (
                original_collected_data.get("user_start_address_original")
            )
            reset_collected_data["user_start_address_validated_coords"] = (
                original_collected_data.get("user_start_address_validated_coords")
            )
        if original_collected_data.get("city_name") and original_collected_data.get(
            "city_id_afisha"
        ):
            reset_collected_data["city_name"] = original_collected_data.get("city_name")
            reset_collected_data["city_id_afisha"] = original_collected_data.get(
                "city_id_afisha"
            )
        reset_collected_data["last_poi_search_results"] = {}
        reset_collected_data["poi_warnings_in_current_plan"] = []
        reset_collected_data["plan_construction_strategy"] = "standard"
        reset_collected_data["address_clarification_status"] = None
        reset_collected_data["partial_address_street"] = None
        reset_collected_data["last_geocoding_attempt_full_address"] = None
        reset_collected_data["awaiting_address_input"] = False
        return {
            "messages": new_messages_for_reset,
            "collected_data": reset_collected_data,
            "current_events": [],
            "current_route_details": None,
            "status_message_to_user": clarification_msg,
            "clarification_needed_fields": (
                ["city_name", "dates_description_original", "interests_original"]
                if not reset_collected_data.get("city_name")
                else ["dates_description_original", "interests_original"]
            ),
            "awaiting_clarification_for_field": None,
            "clarification_context": None,
            "awaiting_fallback_confirmation": False,
            "pending_fallback_event": None,
            "last_offered_fallback_for_interest": None,
            "fallback_accepted_and_plan_updated": False,
            "not_found_interest_keys_in_primary_search": [],
            "fallback_candidates": {},
            "is_initial_plan_proposed": False,
            "is_full_plan_with_route_proposed": False,
            "awaiting_final_confirmation": False,
            "last_presented_plan": None,
            "awaiting_feedback_on_final_plan": False,
            "modification_request_details": None,
            "plan_modification_pending": False,
            "just_modified_plan": False,
        }

    if awaiting_clarification_field:
        logger.info(
            f"Processing '{user_query}' as clarification for '{awaiting_clarification_field}'"
        )
        new_clar_needed_fields = list(
            current_collected_data_dict.get("clarification_needed_fields", [])
        )
        field_clarified_ok = True
        clarification_context_for_node = None

        if awaiting_clarification_field == "poi_time_optimization_choice":
            if (
                "да" in user_query_lower
                or "лучше" in user_query_lower
                or "попроб" in user_query_lower
            ):
                current_collected_data_dict["plan_construction_strategy"] = (
                    "optimize_poi_time"
                )
            else:
                current_collected_data_dict["plan_construction_strategy"] = "standard"
            current_collected_data_dict["poi_warnings_in_current_plan"] = []
            node_return_values["just_modified_plan"] = True
        elif awaiting_clarification_field == "poi_search_optimization_choice":
            if "да" in user_query_lower or "попроб" in user_query_lower:
                current_collected_data_dict["plan_construction_strategy"] = (
                    "optimize_poi_time"
                )
            else:
                current_collected_data_dict["plan_construction_strategy"] = "standard"
            node_return_values["just_modified_plan"] = True
        elif awaiting_clarification_field == "city_name":
            current_collected_data_dict["city_name"] = user_query
            cities = await fetch_cities()
            found_c = next(
                (c for c in cities if user_query.lower() in c["name_lower"]), None
            )
            if found_c:
                current_collected_data_dict["city_id_afisha"] = found_c["id"]
            else:
                current_collected_data_dict["city_id_afisha"] = None
                clarification_context_for_node = (
                    f"Город '{escape_markdown_v2(user_query)}' не найден."
                )
                field_clarified_ok = False
        elif awaiting_clarification_field == "dates_description_original":
            current_collected_data_dict["dates_description_original"] = user_query
            current_collected_data_dict["raw_time_description_original"] = None
            parsed_dt_res = await datetime_parser_tool.ainvoke(
                {
                    "natural_language_date": user_query,
                    "natural_language_time_qualifier": None,
                    "base_date_iso": datetime.now().isoformat(),
                }
            )
            if parsed_dt_res.get("datetime_iso"):
                current_collected_data_dict["parsed_dates_iso"] = [
                    parsed_dt_res["datetime_iso"]
                ]
                current_collected_data_dict["parsed_end_dates_iso"] = (
                    [parsed_dt_res["end_datetime_iso"]]
                    if parsed_dt_res.get("end_datetime_iso")
                    else None
                )
                if parsed_dt_res.get("is_ambiguous"):
                    clarification_context_for_node = parsed_dt_res.get(
                        "clarification_needed"
                    )
                    field_clarified_ok = False
            else:
                clarification_context_for_node = (
                    parsed_dt_res.get("clarification_needed")
                    or "Не удалось распознать дату."
                )
                field_clarified_ok = False
        elif awaiting_clarification_field == "interests_original":
            interests_list_raw_clarify = [
                i.strip() for i in user_query.split(",") if i.strip()
            ]
            if interests_list_raw_clarify:
                current_collected_data_dict["interests_original"] = (
                    interests_list_raw_clarify
                )
                (
                    parsed_oa_clarify_list,
                    parsed_afisha_k_clarify_list,
                    parsed_park_q_clarify,
                    parsed_food_q_clarify,
                ) = ([], [], None, None)
                temp_ordered_activities_clarify_list = []
                for interest_str_clarify in interests_list_raw_clarify:
                    interest_lower_clarify = interest_str_clarify.lower()
                    activity_type_clarify = "UNKNOWN_INTEREST"
                    query_details_clarify = interest_str_clarify
                    if any(kw in interest_lower_clarify for kw in ["кино", "фильм"]):
                        activity_type_clarify = "MOVIE"
                    elif any(
                        kw in interest_lower_clarify
                        for kw in ["концерт", "выступление"]
                    ):
                        activity_type_clarify = "CONCERT"
                    elif any(
                        kw in interest_lower_clarify
                        for kw in ["парк", "сквер", "погулять"]
                    ):
                        activity_type_clarify = "PARK"
                        parsed_park_q_clarify = query_details_clarify
                    elif any(
                        kw in interest_lower_clarify
                        for kw in ["ресторан", "кафе", "поесть", "еда"]
                    ):
                        activity_type_clarify = "FOOD_PLACE"
                        parsed_food_q_clarify = query_details_clarify
                    elif any(
                        kw in interest_lower_clarify for kw in ["театр", "спектакль"]
                    ):
                        activity_type_clarify = "PERFORMANCE"
                    elif any(
                        kw in interest_lower_clarify
                        for kw in ["стендап", "standup", "stand-up"]
                    ):
                        activity_type_clarify = "STAND_UP"
                    elif any(
                        kw in interest_lower_clarify
                        for kw in ["музей", "выставка", "экспозиция", "галерея"]
                    ):
                        activity_type_clarify = "MUSEUM_EXHIBITION"
                    temp_ordered_activities_clarify_list.append(
                        OrderedActivityItem(
                            activity_type=activity_type_clarify,
                            query_details=query_details_clarify,
                        )
                    )
                    if (
                        activity_type_clarify
                        not in ["PARK", "FOOD_PLACE", "UNKNOWN_INTEREST"]
                        and activity_type_clarify in CREATION_TYPES_AFISHA
                    ):
                        parsed_afisha_k_clarify_list.append(activity_type_clarify)
                parsed_oa_clarify_list_dicts = [
                    oa.model_dump(exclude_none=True)
                    for oa in temp_ordered_activities_clarify_list
                ]
                if parsed_oa_clarify_list_dicts:
                    current_collected_data_dict["ordered_activities"] = (
                        parsed_oa_clarify_list_dicts
                    )
                    current_collected_data_dict["interests_keys_afisha"] = (
                        list(set(parsed_afisha_k_clarify_list))
                        if parsed_afisha_k_clarify_list
                        else None
                    )
                    current_collected_data_dict["poi_park_query"] = (
                        parsed_park_q_clarify
                    )
                    current_collected_data_dict["poi_food_query"] = (
                        parsed_food_q_clarify
                    )
                else:
                    clarification_context_for_node = "Не удалось распознать интересы."
                    field_clarified_ok = False
            else:
                clarification_context_for_node = "Укажите интересы."
                field_clarified_ok = False
        elif awaiting_clarification_field == "budget_original":
            try:
                b_match = re.search(r"\d+", user_query)
                if b_match:
                    current_collected_data_dict["budget_original"] = int(
                        b_match.group(0)
                    )
                    current_collected_data_dict["budget_current_search"] = int(
                        b_match.group(0)
                    )
                else:
                    raise ValueError()
            except ValueError:
                clarification_context_for_node = "Укажите бюджет числом."
                field_clarified_ok = False
        elif awaiting_clarification_field == "user_start_address_original":
            city_geo = current_collected_data_dict.get("city_name")

            current_collected_data_dict["address_clarification_status"] = None
            current_collected_data_dict["last_geocoding_attempt_full_address"] = None
            current_collected_data_dict["awaiting_address_input"] = False

            addr_to_geocode = user_query.strip()

            if not city_geo:
                clarification_context_for_node = "Для уточнения адреса мне сначала нужно знать город. Пожалуйста, укажите город."
                field_clarified_ok = False
                if "city_name" not in new_clar_needed_fields:
                    new_clar_needed_fields.append("city_name")
                if "partial_address_street" in current_collected_data_dict:
                    del current_collected_data_dict["partial_address_street"]

            elif not addr_to_geocode or addr_to_geocode.lower() == "пропустить":
                logger.info("User chose to skip address input.")
                current_collected_data_dict["user_start_address_original"] = None
                current_collected_data_dict["user_start_address_validated_coords"] = (
                    None
                )
                if "partial_address_street" in current_collected_data_dict:
                    del current_collected_data_dict["partial_address_street"]
                current_collected_data_dict["address_clarification_status"] = (
                    "SKIPPED_BY_USER"
                )
                current_collected_data_dict["awaiting_address_input"] = False
                field_clarified_ok = True
                if awaiting_clarification_field in new_clar_needed_fields:
                    new_clar_needed_fields.remove(awaiting_clarification_field)
            else:
                previously_known_street = current_collected_data_dict.get(
                    "partial_address_street"
                )
                is_likely_house_number_only = re.fullmatch(
                    r"[\d]+[\s]*[а-яА-Яa-zA-ZкКсС/\-\.]*[\s\d]*", addr_to_geocode
                )

                full_address_attempt = ""
                if previously_known_street and is_likely_house_number_only:
                    full_address_attempt = (
                        f"{previously_known_street}, {addr_to_geocode}"
                    )
                    logger.info(
                        f"Attempting to geocode with stored street: '{full_address_attempt}'"
                    )
                else:
                    full_address_attempt = addr_to_geocode
                    if previously_known_street and not is_likely_house_number_only:
                        logger.info(
                            f"User provided new street-like input '{addr_to_geocode}', replacing previous street '{previously_known_street}'."
                        )
                        current_collected_data_dict["partial_address_street"] = None
                        previously_known_street = None

                current_collected_data_dict["last_geocoding_attempt_full_address"] = (
                    full_address_attempt
                )
                geo_res: GeocodingResult = await get_geocoding_details(
                    address=full_address_attempt, city=city_geo
                )

                if geo_res.is_precise_enough and geo_res.coords:
                    logger.info(
                        f"Address geocoded successfully: {geo_res.full_address_name_gis}"
                    )
                    current_collected_data_dict["user_start_address_original"] = (
                        geo_res.full_address_name_gis
                    )
                    current_collected_data_dict[
                        "user_start_address_validated_coords"
                    ] = {"lon": geo_res.coords[0], "lat": geo_res.coords[1]}
                    if "partial_address_street" in current_collected_data_dict:
                        del current_collected_data_dict["partial_address_street"]
                    current_collected_data_dict["address_clarification_status"] = (
                        "VALIDATED"
                    )
                    current_collected_data_dict["awaiting_address_input"] = False
                    field_clarified_ok = True
                elif geo_res.match_level == "street":
                    logger.info(
                        f"Address recognized as street only: {geo_res.full_address_name_gis}"
                    )
                    current_collected_data_dict["partial_address_street"] = (
                        geo_res.full_address_name_gis
                    )
                    current_collected_data_dict["user_start_address_original"] = None
                    current_collected_data_dict[
                        "user_start_address_validated_coords"
                    ] = None
                    current_collected_data_dict["address_clarification_status"] = (
                        "NEED_HOUSE_NUMBER"
                    )
                    current_collected_data_dict["awaiting_address_input"] = True
                    clarification_context_for_node = (
                        f"Улица '{escape_markdown_v2(geo_res.full_address_name_gis)}' найдена. "
                        f"Пожалуйста, уточните номер дома."
                    )
                    field_clarified_ok = False
                else:
                    logger.warning(
                        f"Failed to geocode address '{full_address_attempt}'. Reason: {geo_res.error_message or geo_res.match_level}"
                    )
                    if previously_known_street or not geo_res.full_address_name_gis:
                        if "partial_address_street" in current_collected_data_dict:
                            del current_collected_data_dict["partial_address_street"]

                    current_collected_data_dict["user_start_address_original"] = None
                    current_collected_data_dict[
                        "user_start_address_validated_coords"
                    ] = None
                    current_collected_data_dict["address_clarification_status"] = (
                        "ADDRESS_NOT_FOUND"
                        if not geo_res.error_message
                        else "GEOCODING_ERROR"
                    )
                    current_collected_data_dict["awaiting_address_input"] = True

                    error_desc = geo_res.error_message or "не удалось распознать"
                    clarification_context_for_node = (
                        f"К сожалению, адрес '{escape_markdown_v2(full_address_attempt)}' {escape_markdown_v2(error_desc)}. "
                        f"Пожалуйста, попробуйте еще раз указать улицу и номер дома, или напишите 'пропустить', если не хотите указывать адрес."
                    )
                    field_clarified_ok = False
        else:
            if (
                awaiting_clarification_field == "budget_original"
                and "нет" not in user_query_lower
            ):
                field_clarified_ok = False
            elif awaiting_clarification_field not in [
                "poi_time_optimization_choice",
                "poi_search_optimization_choice",
                "fallback_confirmation_response",
                "budget_fallback_confirmation",
                "combo_fallback_confirmation",
            ]:
                field_clarified_ok = False

        if field_clarified_ok:
            if awaiting_clarification_field in new_clar_needed_fields:
                new_clar_needed_fields.remove(awaiting_clarification_field)
            current_collected_data_dict["awaiting_address_input"] = False
        elif (
            not clarification_context_for_node
            and awaiting_clarification_field
            not in [
                "poi_time_optimization_choice",
                "poi_search_optimization_choice",
                "fallback_confirmation_response",
                "budget_fallback_confirmation",
                "combo_fallback_confirmation",
            ]
        ):
            if awaiting_clarification_field not in new_clar_needed_fields:
                new_clar_needed_fields.append(awaiting_clarification_field)

        current_collected_data_dict["clarification_needed_fields"] = [
            f for f in new_clar_needed_fields if f
        ]

        next_awaiting_clarification = None
        if (
            not field_clarified_ok
            and awaiting_clarification_field
            not in [
                "poi_time_optimization_choice",
                "poi_search_optimization_choice",
                "fallback_confirmation_response",
                "budget_fallback_confirmation",
                "combo_fallback_confirmation",
            ]
            or (
                awaiting_clarification_field == "user_start_address_original"
                and current_collected_data_dict.get("awaiting_address_input")
            )
        ):
            next_awaiting_clarification = awaiting_clarification_field

        node_return_values["collected_data"] = current_collected_data_dict
        node_return_values["messages"] = messages
        node_return_values["clarification_context"] = clarification_context_for_node
        node_return_values["awaiting_clarification_for_field"] = (
            next_awaiting_clarification
        )

        node_return_values["awaiting_feedback_on_final_plan"] = False
        node_return_values["modification_request_details"] = None
        node_return_values["plan_modification_pending"] = False

        logger.info(
            f"extract_initial_info_node (clarification branch): next_awaiting_clarification='{next_awaiting_clarification}', clarification_context='{clarification_context_for_node}', awaiting_address_input={current_collected_data_dict.get('awaiting_address_input')}"
        )
        return node_return_values

    is_modification_or_feedback_context = (
        is_awaiting_feedback_on_final_plan
        or state.get("last_presented_plan") is not None
    )
    if not is_modification_or_feedback_context:
        logger.info(
            "extract_initial_info_node: Processing as a new general query. Clearing most plan-specific data but preserving address/city. Resetting excluded_ids."
        )
        preserved_data = {
            "current_excluded_ids": {
                "afisha": [],
                "park": [],
                "food": [],
                "afisha_names_to_avoid": [],
                "afisha_creation_ids_to_avoid": [],
            },
            "rejected_fallback_for_interests": [],
            "last_poi_search_results": {},
            "plan_construction_strategy": "standard",
            "poi_warnings_in_current_plan": [],
            "address_clarification_status": current_collected_data_dict.get(
                "address_clarification_status"
            ),
            "partial_address_street": current_collected_data_dict.get(
                "partial_address_street"
            ),
            "last_geocoding_attempt_full_address": current_collected_data_dict.get(
                "last_geocoding_attempt_full_address"
            ),
            "awaiting_address_input": current_collected_data_dict.get(
                "awaiting_address_input", False
            ),
        }
        if original_collected_data.get("user_start_address_validated_coords"):
            preserved_data["user_start_address_original"] = original_collected_data.get(
                "user_start_address_original"
            )
            preserved_data["user_start_address_validated_coords"] = (
                original_collected_data.get("user_start_address_validated_coords")
            )
        if original_collected_data.get("city_name") and original_collected_data.get(
            "city_id_afisha"
        ):
            preserved_data["city_name"] = original_collected_data.get("city_name")
            preserved_data["city_id_afisha"] = original_collected_data.get(
                "city_id_afisha"
            )
        current_collected_data_dict = preserved_data
        node_return_values.update(
            {
                "current_events": [],
                "current_route_details": None,
                "is_initial_plan_proposed": False,
                "is_full_plan_with_route_proposed": False,
                "last_presented_plan": None,
                "modification_request_details": None,
                "plan_modification_pending": False,
                "awaiting_feedback_on_final_plan": False,
            }
        )
        node_return_values["just_modified_plan"] = False
    else:
        logger.info(
            "extract_initial_info_node: Processing in context of existing plan/feedback. current_excluded_ids from previous state are preserved."
        )
        node_return_values["just_modified_plan"] = False

    node_return_values["awaiting_feedback_on_final_plan"] = False
    llm = get_gigachat_client()
    structured_llm_extraction = llm.with_structured_output(ExtractedInitialInfo)
    extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{escape_markdown_v2(user_query)}"'
    extracted_info_model: Optional[ExtractedInitialInfo] = None
    clarification_context_for_node = None
    try:
        extracted_info_model = await structured_llm_extraction.ainvoke(
            extraction_prompt_with_query
        )
        if extracted_info_model:
            logger.info(
                f"extract_initial_info_node: LLM Extracted: {extracted_info_model.model_dump_json(indent=2, exclude_none=True)}"
            )
    except Exception as e_llm_extract_gen:
        logger.error(
            f"extract_initial_info_node: LLM extraction error: {e_llm_extract_gen}",
            exc_info=True,
        )
        current_collected_data_dict.setdefault("clarification_needed_fields", [])
        for f_k in ["city_name", "dates_description_original", "interests_original"]:
            if (
                not current_collected_data_dict.get(f_k)
                and f_k
                not in current_collected_data_dict["clarification_needed_fields"]
            ):
                current_collected_data_dict["clarification_needed_fields"].append(f_k)
        clarification_context_for_node = (
            clarification_context_for_node
            or "Произошла ошибка при обработке вашего запроса."
        )

    newly_extracted_clar_fields_llm: List[str] = []
    modification_details_update_llm: Dict[str, Any] = {}
    plan_pending_update_llm = False

    if extracted_info_model:
        if extracted_info_model.city:
            current_collected_data_dict["city_name"] = extracted_info_model.city
            cities = await fetch_cities()
            found_city_from_llm = next(
                (
                    c
                    for c in cities
                    if extracted_info_model.city.lower() in c["name_lower"]
                ),
                None,
            )
            if found_city_from_llm:
                current_collected_data_dict["city_id_afisha"] = found_city_from_llm[
                    "id"
                ]
            else:
                newly_extracted_clar_fields_llm.append("city_name")
                clarification_context_for_node = (
                    clarification_context_for_node
                    or f"Город '{escape_markdown_v2(extracted_info_model.city)}' не найден."
                )
            if is_modification_or_feedback_context:
                modification_details_update_llm["new_city"] = extracted_info_model.city
                plan_pending_update_llm = True
        elif not current_collected_data_dict.get("city_name"):
            newly_extracted_clar_fields_llm.append("city_name")

        if extracted_info_model.ordered_activities:
            for act in extracted_info_model.ordered_activities:
                act_type_upper = act.activity_type.upper()
                if act_type_upper == "THEATER":
                    act.activity_type = "PERFORMANCE"
                elif act_type_upper == "CAFE" or act_type_upper == "RESTAURANT":
                    act.activity_type = "FOOD_PLACE"

            llm_extracted_oa_dumps = [
                oa.model_dump(exclude_none=True)
                for oa in extracted_info_model.ordered_activities
            ]
            if is_modification_or_feedback_context:
                modification_details_update_llm[
                    "new_ordered_activities_from_llm_feedback"
                ] = llm_extracted_oa_dumps
                plan_pending_update_llm = True
                node_return_values.update(
                    {
                        "current_events": [],
                        "current_route_details": None,
                        "is_initial_plan_proposed": False,
                        "is_full_plan_with_route_proposed": False,
                    }
                )
            else:
                current_collected_data_dict["ordered_activities"] = (
                    llm_extracted_oa_dumps
                )
            current_oa_for_related_fields_raw = current_collected_data_dict.get(
                "ordered_activities", []
            )
            if (
                is_modification_or_feedback_context
                and "new_ordered_activities_from_llm_feedback"
                in modification_details_update_llm
            ):
                current_oa_for_related_fields_raw = modification_details_update_llm[
                    "new_ordered_activities_from_llm_feedback"
                ]
            current_oa_for_related_fields = [
                OrderedActivityItem(**oad)
                for oad in current_oa_for_related_fields_raw
                if isinstance(oad, dict)
            ]
            af_keys_llm, park_q_llm, food_q_llm, raw_int_llm = [], None, None, []
            for act_item in current_oa_for_related_fields:
                if act_item.query_details:
                    raw_int_llm.append(act_item.query_details)
                else:
                    raw_int_llm.append(act_item.activity_type)
                if act_item.activity_type == "PARK":
                    park_q_llm = act_item.query_details or "парк"
                elif act_item.activity_type == "FOOD_PLACE":
                    food_q_llm = act_item.query_details or "еда"
                elif (
                    act_item.activity_type not in ["UNKNOWN_INTEREST", None]
                    and act_item.activity_type in CREATION_TYPES_AFISHA
                ):
                    af_keys_llm.append(act_item.activity_type)
            current_collected_data_dict["interests_keys_afisha"] = (
                list(set(af_keys_llm)) if af_keys_llm else None
            )
            current_collected_data_dict["poi_park_query"] = park_q_llm
            current_collected_data_dict["poi_food_query"] = food_q_llm
            if not is_modification_or_feedback_context or (
                is_modification_or_feedback_context
                and extracted_info_model.ordered_activities
            ):
                current_collected_data_dict["interests_original"] = (
                    raw_int_llm if raw_int_llm else None
                )
        elif not (
            current_collected_data_dict.get("ordered_activities")
            or current_collected_data_dict.get("interests_keys_afisha")
            or current_collected_data_dict.get("poi_park_query")
            or current_collected_data_dict.get("poi_food_query")
        ):
            newly_extracted_clar_fields_llm.append("interests_original")

        if extracted_info_model.budget is not None:
            current_collected_data_dict["budget_original"] = extracted_info_model.budget
            current_collected_data_dict["budget_current_search"] = (
                extracted_info_model.budget
            )
            if is_modification_or_feedback_context:
                modification_details_update_llm["new_budget"] = (
                    extracted_info_model.budget
                )
                plan_pending_update_llm = True

        new_dates_desc_llm = extracted_info_model.dates_description
        new_time_desc_llm = extracted_info_model.raw_time_description
        date_or_time_updated_by_llm = False
        if new_dates_desc_llm is not None:
            current_collected_data_dict["dates_description_original"] = (
                new_dates_desc_llm
            )
            current_collected_data_dict["parsed_dates_iso"] = None
            current_collected_data_dict["parsed_end_dates_iso"] = None
            date_or_time_updated_by_llm = True
            if is_modification_or_feedback_context:
                modification_details_update_llm["new_dates_description"] = (
                    new_dates_desc_llm
                )
                plan_pending_update_llm = True
        if new_time_desc_llm is not None:
            current_collected_data_dict["raw_time_description_original"] = (
                new_time_desc_llm
            )
            current_collected_data_dict["parsed_dates_iso"] = None
            current_collected_data_dict["parsed_end_dates_iso"] = None
            date_or_time_updated_by_llm = True
            if is_modification_or_feedback_context:
                modification_details_update_llm["new_time_description"] = (
                    new_time_desc_llm
                )
                plan_pending_update_llm = True

        should_parse_dt_after_llm = date_or_time_updated_by_llm or (
            (
                current_collected_data_dict.get("dates_description_original")
                or current_collected_data_dict.get("raw_time_description_original")
            )
            and not current_collected_data_dict.get("parsed_dates_iso")
        )
        if should_parse_dt_after_llm:
            date_to_parse = current_collected_data_dict.get(
                "dates_description_original"
            )
            time_to_parse = current_collected_data_dict.get(
                "raw_time_description_original"
            )
            if (
                not date_to_parse
                and not time_to_parse
                and not current_collected_data_dict.get("parsed_dates_iso")
            ):
                if "dates_description_original" not in newly_extracted_clar_fields_llm:
                    newly_extracted_clar_fields_llm.append("dates_description_original")
            elif date_to_parse or time_to_parse:
                parsed_dt_result_after_llm = await datetime_parser_tool.ainvoke(
                    {
                        "natural_language_date": date_to_parse or "",
                        "natural_language_time_qualifier": time_to_parse,
                        "base_date_iso": datetime.now().isoformat(),
                    }
                )
                if parsed_dt_result_after_llm.get("datetime_iso"):
                    current_collected_data_dict["parsed_dates_iso"] = [
                        parsed_dt_result_after_llm["datetime_iso"]
                    ]
                    current_collected_data_dict["parsed_end_dates_iso"] = (
                        [parsed_dt_result_after_llm["end_datetime_iso"]]
                        if parsed_dt_result_after_llm.get("end_datetime_iso")
                        else None
                    )
                    if parsed_dt_result_after_llm.get("is_ambiguous"):
                        if (
                            "dates_description_original"
                            not in newly_extracted_clar_fields_llm
                        ):
                            newly_extracted_clar_fields_llm.append(
                                "dates_description_original"
                            )
                        clarification_context_for_node = (
                            (clarification_context_for_node or "")
                            + " "
                            + (
                                parsed_dt_result_after_llm.get("clarification_needed")
                                or ""
                            )
                        ).strip()
                else:
                    if (
                        "dates_description_original"
                        not in newly_extracted_clar_fields_llm
                    ):
                        newly_extracted_clar_fields_llm.append(
                            "dates_description_original"
                        )
                    clarification_context_for_node = (
                        (clarification_context_for_node or "")
                        + " "
                        + (
                            parsed_dt_result_after_llm.get("clarification_needed")
                            or parsed_dt_result_after_llm.get("error_message")
                            or "Не удалось распознать дату/время."
                        )
                    ).strip()
        elif not current_collected_data_dict.get("parsed_dates_iso"):
            if "dates_description_original" not in newly_extracted_clar_fields_llm:
                newly_extracted_clar_fields_llm.append("dates_description_original")

        if plan_pending_update_llm:
            node_return_values["just_modified_plan"] = True

    existing_clar_fields_final = current_collected_data_dict.get(
        "clarification_needed_fields", []
    )
    current_collected_data_dict["clarification_needed_fields"] = list(
        set(existing_clar_fields_final + newly_extracted_clar_fields_llm)
    )
    current_collected_data_dict["clarification_needed_fields"] = [
        f for f in current_collected_data_dict["clarification_needed_fields"] if f
    ]
    current_collected_data_dict["fallback_accepted_and_plan_updated"] = False

    node_return_values["collected_data"] = current_collected_data_dict
    node_return_values["messages"] = messages
    node_return_values["clarification_context"] = clarification_context_for_node
    node_return_values["awaiting_clarification_for_field"] = None

    if modification_details_update_llm:
        node_return_values["modification_request_details"] = (
            modification_details_update_llm
        )
    if plan_pending_update_llm:
        node_return_values["plan_modification_pending"] = True

    logger.info(
        f"extract_initial_info_node: Final collected_data before return: {str(node_return_values.get('collected_data'))[:1000]}"
    )
    logger.info(
        f"extract_initial_info_node: just_modified_plan set to {node_return_values.get('just_modified_plan')}"
    )
    return node_return_values


# ... (остальные узлы без изменений на данный момент) ...


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
    else:
        field_description_for_prompt = missing_critical_fields_map.get(
            field_to_clarify_now, f"поле '{field_to_clarify_now}'"
        )
        raw_time_desc = collected_data_dict.get("raw_time_description_original")
        prompt_for_llm: str
        if field_to_clarify_now == "dates_description_original" and raw_time_desc:
            prompt_for_llm = TIME_CLARIFICATION_PROMPT_TEMPLATE.format(
                raw_time_description=escape_markdown_v2(raw_time_desc),  # Экранируем
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
                        current_data_summary_parts.append(
                            f"Город: {escape_markdown_v2(str(v))}"
                        )
                    elif k == "dates_description_original":
                        current_data_summary_parts.append(
                            f"Когда: {escape_markdown_v2(str(v))}"
                        )
                    elif k == "interests_original":
                        current_data_summary_parts.append(
                            f"Интересы: {escape_markdown_v2(', '.join(v) if isinstance(v, list) else str(v))}"
                        )
                    elif k == "budget_original":
                        current_data_summary_parts.append(f"Бюджет: до {v} руб.")
            current_data_summary_str = (
                "; ".join(current_data_summary_parts)
                if current_data_summary_parts
                else "пока ничего не уточнено"
            )
            prompt_for_llm = GENERAL_CLARIFICATION_PROMPT_TEMPLATE.format(
                user_query=escape_markdown_v2(last_user_message_content),  # Экранируем
                current_collected_data_summary=current_data_summary_str,  # Уже экранировано выше
                missing_fields_description=escape_markdown_v2(
                    field_description_for_prompt
                ),  # Экранируем
            )
        llm = get_gigachat_client()
        try:
            ai_response = await llm.ainvoke(prompt_for_llm)
            status_message_to_user = ai_response.content
        except Exception as e_clarify:
            logger.error(
                f"clarify_missing_data_node: Error during LLM call: {e_clarify}",
                exc_info=True,
            )
            status_message_to_user = f"Мне нужно уточнение по полю: {escape_markdown_v2(field_description_for_prompt)}. Не могли бы вы помочь?"

    final_message_to_user = (
        status_message_to_user
        or f"Пожалуйста, уточните {escape_markdown_v2(field_description_for_prompt)}."
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
# Файл: agent_core/nodes.py


default_event_duration_minutes: int = 120


# ... (импорты) ...
async def _check_event_compatibility(
    first_item_data: Dict[str, Any],
    second_item_data: Dict[str, Any],
    user_max_overall_end_dt_naive: Optional[datetime],
    city_name_for_gis: Optional[str],
    default_event_duration_minutes: int = 120,
    default_poi_duration_minutes: int = 90,
    first_item_type: Optional[str] = "event",
    min_poi_visit_duration_minutes: int = 30,
    is_today_request_for_plan: bool = False,
    user_min_start_dt_naive_for_today: Optional[datetime] = None,
) -> Tuple[bool, Optional[str], Optional[datetime], Optional[str], Optional[int]]:

    first_item_name_log = first_item_data.get("name", "Previous_Item_NoName")
    second_item_name_log = second_item_data.get("name", "Second_Item_NoName")
    second_item_plan_type = second_item_data.get("_plan_item_type")

    logger.debug(
        f"COMPAT_CHECK: Start: '{first_item_name_log}' (type: {first_item_type}) vs '{second_item_name_log}' (type: {second_item_plan_type})"
    )
    logger.debug(
        f"COMPAT_CHECK: Args: user_max_overall_end_dt_naive='{user_max_overall_end_dt_naive}', is_today_request='{is_today_request_for_plan}', user_min_start_for_today='{user_min_start_dt_naive_for_today}'"
    )

    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    calculated_second_item_end_naive: Optional[datetime] = None
    route_duration_seconds_to_second_item: Optional[int] = None

    first_item_end_naive_raw = first_item_data.get("_calculated_end_time")
    first_item_end_naive: Optional[datetime] = None
    if isinstance(first_item_end_naive_raw, str):
        try:
            first_item_end_naive = datetime.fromisoformat(first_item_end_naive_raw)
        except ValueError:
            error_message = "Ошибка формата времени окончания предыдущего элемента."
            logger.error(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return False, error_message, None, None, None
    elif isinstance(first_item_end_naive_raw, datetime):
        first_item_end_naive = first_item_end_naive_raw
    if not first_item_end_naive:
        start_dt_val_f = None
        duration_f = default_event_duration_minutes
        if (
            first_item_type == "afisha_event"
            and "start_time_naive_event_tz" in first_item_data
        ):
            start_dt_val_f = first_item_data["start_time_naive_event_tz"]
            duration_f = (
                first_item_data.get("duration_minutes")
                or default_event_duration_minutes
            )
        elif (
            first_item_type in ["park", "food"]
            and "_calculated_start_time" in first_item_data
        ):
            start_dt_val_f = first_item_data["_calculated_start_time"]
            duration_f = default_poi_duration_minutes
        elif (
            first_item_type == "poi_start"
            and "_calculated_start_time" in first_item_data
        ):
            start_dt_val_f = first_item_data["_calculated_start_time"]
            duration_f = 0
        if start_dt_val_f:
            start_dt_f_parsed = None
            if isinstance(start_dt_val_f, datetime):
                start_dt_f_parsed = start_dt_val_f
            elif isinstance(start_dt_val_f, str):
                try:
                    start_dt_f_parsed = datetime.fromisoformat(start_dt_val_f)
                except ValueError:
                    error_message = (
                        "Ошибка формата времени начала пред. элемента (строка)."
                    )
                    logger.error(
                        f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
                    )
                    return False, error_message, None, None, None
            if not isinstance(start_dt_f_parsed, datetime):
                error_message = "Ошибка времени начала пред. элемента."
                logger.error(
                    f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
                )
                return False, error_message, None, None, None
            first_item_end_naive = start_dt_f_parsed + timedelta(minutes=duration_f)
        else:
            error_message = f"Не удалось определить время окончания предыдущего шага ('{first_item_name_log}')"
            logger.error(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return False, error_message, None, None, None
    logger.debug(
        f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): first_item_end_naive = {first_item_end_naive}"
    )

    first_item_coords_lon: Optional[float] = None
    first_item_coords_lat: Optional[float] = None
    if first_item_type == "afisha_event":
        first_item_coords_lon = first_item_data.get("place_coords_lon")
        first_item_coords_lat = first_item_data.get("place_coords_lat")
    elif first_item_type in ["park", "food", "poi_start"]:
        coords_list_f = first_item_data.get("coords")
        if isinstance(coords_list_f, list) and len(coords_list_f) == 2:
            first_item_coords_lon, first_item_coords_lat = (
                coords_list_f[0],
                coords_list_f[1],
            )

    effective_limit_for_second_item = user_max_overall_end_dt_naive
    logger.debug(
        f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Initial effective_limit_for_second_item = {effective_limit_for_second_item}"
    )
    if is_today_request_for_plan and user_min_start_dt_naive_for_today:
        base_day_for_limit_calc = user_min_start_dt_naive_for_today.date()
        if (
            first_item_end_naive
            and first_item_end_naive.date() > base_day_for_limit_calc
        ):
            base_day_for_limit_calc = first_item_end_naive.date()
            logger.debug(
                f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Previous item overflowed to {base_day_for_limit_calc}, new base for limit calc."
            )

        # Default soft limit is 4 AM of the (possibly new) base_day_for_limit_calc
        potential_soft_limit = datetime.combine(base_day_for_limit_calc, time(4, 0, 0))
        # If the original plan was for "today", and we are still on "today" OR the very beginning of "tomorrow" after an event
        if (
            base_day_for_limit_calc == user_min_start_dt_naive_for_today.date()
            or base_day_for_limit_calc
            == user_min_start_dt_naive_for_today.date() + timedelta(days=1)
        ):
            # If the current activity is an Afisha event, or if the previous one was and ended late
            if second_item_plan_type == "afisha_event" or (
                first_item_type == "afisha_event"
                and first_item_end_naive
                and first_item_end_naive.hour < 6
                and first_item_end_naive.date()
                > user_min_start_dt_naive_for_today.date()
            ):  # Ended after midnight but before 6am
                # Use a later soft limit, e.g., 4 AM of the day *after* the original "today" if we've crossed midnight significantly
                if (
                    first_item_end_naive
                    and first_item_end_naive.date()
                    > user_min_start_dt_naive_for_today.date()
                ):
                    potential_soft_limit = datetime.combine(
                        first_item_end_naive.date(), time(4, 0, 0)
                    )
                else:  # Still on original "today" or film ends very late
                    potential_soft_limit = datetime.combine(
                        user_min_start_dt_naive_for_today.date() + timedelta(days=1),
                        time(4, 0, 0),
                    )

        if (
            not effective_limit_for_second_item
            or potential_soft_limit > effective_limit_for_second_item
        ):
            if (
                user_max_overall_end_dt_naive
                and user_max_overall_end_dt_naive.date()
                == user_min_start_dt_naive_for_today.date()
                and user_max_overall_end_dt_naive.time() == time(23, 59, 59)
            ):
                effective_limit_for_second_item = potential_soft_limit
                logger.debug(
                    f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Applied soft effective_limit_for_second_item: {effective_limit_for_second_item}"
                )
            elif (
                user_max_overall_end_dt_naive
                and first_item_end_naive
                and first_item_end_naive.date() > user_max_overall_end_dt_naive.date()
            ):
                # If previous item already pushed us past the original overall limit's day, use a new soft limit for the current day
                effective_limit_for_second_item = datetime.combine(
                    first_item_end_naive.date(), time(4, 0, 0)
                )
                logger.debug(
                    f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Previous item crossed overall limit day. New soft limit: {effective_limit_for_second_item}"
                )

    if (
        effective_limit_for_second_item
        and first_item_end_naive
        and first_item_end_naive > effective_limit_for_second_item
    ):
        error_message = f"'{first_item_name_log}' ({first_item_end_naive.strftime('%H:%M %d.%m')}) заканчивается позже допустимого лимита для след. активности ({effective_limit_for_second_item.strftime('%H:%M %d.%m')})."
        logger.info(
            f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
        )
        return False, error_message, None, None, None

    second_item_start_naive_raw = second_item_data.get(
        "start_time_naive_event_tz"
    )  # ... (определение second_item_start_naive, coords, duration - без изменений)
    second_item_start_naive: Optional[datetime] = None
    if isinstance(second_item_start_naive_raw, str):
        try:
            second_item_start_naive = datetime.fromisoformat(
                second_item_start_naive_raw
            )
        except ValueError:
            logger.warning(
                f"Invalid datetime string for second_item_start_naive_raw: {second_item_start_naive_raw}"
            )
    elif isinstance(second_item_start_naive_raw, datetime):
        second_item_start_naive = second_item_start_naive_raw
    second_item_coords_lon: Optional[float] = None
    second_item_coords_lat: Optional[float] = None
    second_item_duration_minutes = default_poi_duration_minutes
    if second_item_plan_type == "afisha_event":  # ... (без изменений)
        second_item_coords_lon = second_item_data.get("place_coords_lon")
        second_item_coords_lat = second_item_data.get("place_coords_lat")
        second_item_duration_minutes = (
            second_item_data.get("duration_minutes") or default_event_duration_minutes
        )
        if not second_item_start_naive:
            error_message = f"Для события Афиши '{second_item_name_log}' не определено время начала."
            logger.error(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return False, error_message, None, None, None
    elif second_item_plan_type in ["park", "food"]:  # ... (без изменений)
        if (
            "coords" in second_item_data
            and isinstance(second_item_data["coords"], list)
            and len(second_item_data["coords"]) == 2
        ):
            second_item_coords_lon = second_item_data["coords"][0]
            second_item_coords_lat = second_item_data["coords"][1]
        elif "address" in second_item_data and city_name_for_gis:
            poi_coords_s = await get_coords_from_address(
                address=second_item_data["address"], city=city_name_for_gis
            )
            if poi_coords_s:
                second_item_coords_lon, second_item_coords_lat = (
                    poi_coords_s[0],
                    poi_coords_s[1],
                )
                second_item_data["coords"] = [
                    second_item_coords_lon,
                    second_item_coords_lat,
                ]
            else:
                error_message = f"Не удалось геокодировать '{second_item_name_log}'."
                logger.error(
                    f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
                )
                return False, error_message, None, None, None
        else:
            error_message = (
                f"Нет координат или адреса/города для POI '{second_item_name_log}'."
            )
            logger.error(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return False, error_message, None, None, None
    else:
        error_message = f"Неизвестный тип элемента '{second_item_name_log}' (_plan_item_type: {second_item_plan_type})."
        logger.error(
            f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
        )
        return False, error_message, None, None, None
    if second_item_coords_lon is None or second_item_coords_lat is None:
        error_message = f"Координаты для '{second_item_name_log}' не определены."
        logger.error(
            f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
        )
        return False, error_message, None, None, None

    route_duration_calc = 0  # ... (расчет route_duration_calc - без изменений) ...
    if first_item_type == "poi_start" and not (
        first_item_coords_lon and first_item_coords_lat
    ):
        route_duration_calc = 0
    elif first_item_coords_lon and first_item_coords_lat:
        try:
            route_result = await get_route(
                points=[
                    {"lon": first_item_coords_lon, "lat": first_item_coords_lat},
                    {"lon": second_item_coords_lon, "lat": second_item_coords_lat},
                ],
                transport="driving",
            )
            if route_result and route_result.get("status") == "success":
                route_duration_calc = route_result.get("duration_seconds", 1800)
            else:
                logger.warning(
                    f"Route API error for compatibility: {route_result.get('message') if route_result else 'N/A'}. Assuming 30m."
                )
                route_duration_calc = 1800
        except Exception as e_route_compat_val_inner:
            logger.error(f"Route exc: {e_route_compat_val_inner}", exc_info=True)
            route_duration_calc = 1800
    elif first_item_type != "poi_start":
        logger.warning(
            f"Prev item '{first_item_name_log}' has no coords. Assuming 0 travel to '{second_item_name_log}'."
        )
        route_duration_calc = 0
    route_duration_seconds_to_second_item = route_duration_calc
    logger.debug(
        f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Travel time = {route_duration_seconds_to_second_item/60:.1f} min"
    )

    if not first_item_end_naive:
        error_message = "Крит. ошибка: время окончания пред. элемента не определено."
        logger.error(
            f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
        )
        return False, error_message, None, None, route_duration_seconds_to_second_item

    arrival_at_second_item_naive = first_item_end_naive + timedelta(
        seconds=route_duration_seconds_to_second_item
    )

    buffer_time = timedelta(minutes=15)
    if first_item_type == "poi_start":  # Скорректированный буфер для первого шага
        buffer_time = timedelta(minutes=0)
        logger.debug(
            f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): First step from user time, buffer_time adjusted to {buffer_time.total_seconds()/60}m."
        )

    actual_start_for_second_item: Optional[datetime] = None

    if second_item_plan_type == "afisha_event":
        actual_start_for_second_item = second_item_start_naive
        logger.debug(
            f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): Afisha Event. Arrival: {arrival_at_second_item_naive}, Buffer: {buffer_time}, Arrival+Buffer: {arrival_at_second_item_naive + buffer_time}, Event Actual Start: {actual_start_for_second_item}"
        )
        if arrival_at_second_item_naive + buffer_time > actual_start_for_second_item:
            error_message = f"Не успеть на '{second_item_name_log}' ({actual_start_for_second_item.strftime('%H:%M %d.%m')}). Расчетное время прибытия с буфером: {(arrival_at_second_item_naive + buffer_time).strftime('%H:%M %d.%m')}."
            logger.info(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return (
                False,
                error_message,
                None,
                None,
                route_duration_seconds_to_second_item,
            )
        calculated_second_item_end_naive = actual_start_for_second_item + timedelta(
            minutes=second_item_duration_minutes
        )

    elif second_item_plan_type in ["park", "food"]:
        actual_start_for_second_item = arrival_at_second_item_naive + buffer_time
        logger.debug(
            f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): POI. Actual_start_for_second_item (after travel+buffer): {actual_start_for_second_item}"
        )
        poi_schedule_str = second_item_data.get("schedule_str")

        is_open, possible_end_time_poi, schedule_err_warn_msg = (
            parse_schedule_and_check_open(
                schedule_str=poi_schedule_str,
                visit_start_dt=actual_start_for_second_item,
                desired_duration_minutes=second_item_duration_minutes,
                item_type_for_schedule=second_item_plan_type,
                min_visit_duration_minutes=min_poi_visit_duration_minutes,
                poi_name_for_log=second_item_name_log,
            )
        )

        if not is_open:
            error_message = f"'{second_item_name_log}' ({second_item_plan_type}) будет закрыто в {actual_start_for_second_item.strftime('%H:%M %d.%m')}. {schedule_err_warn_msg or ''}".strip()
            logger.info(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return (
                False,
                error_message,
                None,
                None,
                route_duration_seconds_to_second_item,
            )

        if schedule_err_warn_msg and not warning_message:
            warning_message = f"Для '{second_item_name_log}': {schedule_err_warn_msg}"

        calculated_second_item_end_naive = possible_end_time_poi
        effective_poi_duration_td = timedelta(minutes=second_item_duration_minutes)
        if calculated_second_item_end_naive:
            effective_poi_duration_td = (
                calculated_second_item_end_naive - actual_start_for_second_item
            )

        current_overall_limit_for_poi = effective_limit_for_second_item
        logger.debug(
            f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): POI. current_overall_limit_for_poi = {current_overall_limit_for_poi}, calculated_end_by_schedule = {calculated_second_item_end_naive}"
        )

        if current_overall_limit_for_poi:
            if actual_start_for_second_item >= current_overall_limit_for_poi:
                error_message = f"Начало '{second_item_name_log}' ({actual_start_for_second_item.strftime('%H:%M %d.%m')}) позже общего лимита ({current_overall_limit_for_poi.strftime('%H:%M %d.%m')})."
                logger.info(
                    f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
                )
                return (
                    False,
                    error_message,
                    None,
                    None,
                    route_duration_seconds_to_second_item,
                )

            if (
                calculated_second_item_end_naive
                and calculated_second_item_end_naive > current_overall_limit_for_poi
            ):
                logger.debug(
                    f"COMPAT_CHECK ('{first_item_name_log}' -> '{second_item_name_log}'): POI visit end {calculated_second_item_end_naive} (from schedule) is after overall limit {current_overall_limit_for_poi}. Adjusting."
                )
                calculated_second_item_end_naive = current_overall_limit_for_poi
                effective_poi_duration_td = (
                    calculated_second_item_end_naive - actual_start_for_second_item
                )

        min_duration_for_this_poi = timedelta(minutes=min_poi_visit_duration_minutes)
        if second_item_plan_type == "park":
            min_duration_for_this_poi = timedelta(minutes=1)

        if effective_poi_duration_td < min_duration_for_this_poi:
            error_message = f"После всех ограничений на '{second_item_name_log}' остается менее {min_duration_for_this_poi.total_seconds()/60:.0f} мин."
            logger.info(
                f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
            )
            return (
                False,
                error_message,
                None,
                None,
                route_duration_seconds_to_second_item,
            )

        if (
            effective_poi_duration_td < timedelta(minutes=second_item_duration_minutes)
            and effective_poi_duration_td >= min_duration_for_this_poi
        ):
            warning_detail_poi = f"На '{second_item_name_log}' останется около {int(effective_poi_duration_td.total_seconds() / 60)} мин."
            if (
                int(effective_poi_duration_td.total_seconds() / 60) < 20
                and second_item_plan_type != "park"
            ):
                warning_detail_poi += " Маловато."
            if warning_message:
                warning_message += " " + warning_detail_poi
            else:
                warning_message = warning_detail_poi

        second_item_data["_calculated_start_time"] = (
            actual_start_for_second_item.isoformat()
        )
        second_item_data["_calculated_duration_minutes"] = (
            effective_poi_duration_td.total_seconds() / 60
        )

    final_limit_to_check_against_overall = effective_limit_for_second_item
    if (
        final_limit_to_check_against_overall
        and calculated_second_item_end_naive
        and calculated_second_item_end_naive > final_limit_to_check_against_overall
    ):
        error_message = f"'{second_item_name_log}' ({calculated_second_item_end_naive.strftime('%H:%M %d.%m') if calculated_second_item_end_naive else 'N/A'}) заканчивается позже итогового лимита ({final_limit_to_check_against_overall.strftime('%H:%M %d.%m')})."
        logger.info(
            f"COMPAT_CHECK_FAIL ('{first_item_name_log}' -> '{second_item_name_log}'): {error_message}"
        )
        return False, error_message, None, None, route_duration_seconds_to_second_item

    second_item_data["_travel_time_to_seconds"] = route_duration_seconds_to_second_item
    logger.info(
        f"COMPAT_CHECK_PASS ('{first_item_name_log}' -> '{second_item_name_log}'): Start: {actual_start_for_second_item.strftime('%H:%M %d.%m') if actual_start_for_second_item else 'N/A'}, End: {calculated_second_item_end_naive.strftime('%H:%M %d.%m') if calculated_second_item_end_naive else 'N/A'}. Warn: {warning_message}. Travel: {route_duration_seconds_to_second_item/60 if route_duration_seconds_to_second_item is not None else 0:.0f}m"
    )
    return (
        True,
        None,
        calculated_second_item_end_naive,
        warning_message,
        route_duration_seconds_to_second_item,
    )


async def search_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info(
        "Node: search_events_node executing with optimal chain construction logic..."
    )
    current_collected_data_dict: dict = dict(state.get("collected_data", {}))
    excluded_ids_map = current_collected_data_dict.setdefault(
        "current_excluded_ids", {}
    )
    for key_ex in [
        "afisha",
        "park",
        "food",
        "afisha_names_to_avoid",
        "afisha_creation_ids_to_avoid",
    ]:
        excluded_ids_map.setdefault(key_ex, [])
    current_collected_data_dict["last_poi_search_results"] = {}
    current_collected_data_dict["poi_warnings_in_current_plan"] = []
    current_collected_data_dict["plan_construction_failed_step"] = None
    current_collected_data_dict["fallback_candidates"] = {}

    # NEW LOGIC: Инициализируем новые поля для "умных" fallback'ов в состоянии
    current_collected_data_dict["budget_fallback_plan"] = None
    current_collected_data_dict["combo_fallback_candidates"] = {}

    city_id_afisha = current_collected_data_dict.get("city_id_afisha")
    city_name_for_gis = current_collected_data_dict.get("city_name")
    if not city_name_for_gis and city_id_afisha:
        cities_list_temp = await fetch_cities()
        found_c_temp = next(
            (c for c in cities_list_temp if c.get("id") == city_id_afisha), None
        )
        if found_c_temp:
            city_name_for_gis = found_c_temp.get("name")
            current_collected_data_dict["city_name"] = city_name_for_gis

    parsed_dates_iso_list = current_collected_data_dict.get("parsed_dates_iso")

    # FIX: Получаем количество персон и общий бюджет для проверки в цикле
    general_budget = current_collected_data_dict.get("budget_current_search")
    person_count = current_collected_data_dict.get("person_count", 1) or 1

    user_min_start_dt_naive: Optional[datetime] = None
    user_max_overall_end_dt_naive: Optional[datetime] = None
    tool_api_date_from_dt: Optional[datetime] = None
    tool_api_date_to_dt: Optional[datetime] = None
    tool_min_start_time_str: Optional[str] = None
    tool_max_start_time_str: Optional[str] = None
    is_today_plan_request = False

    if not parsed_dates_iso_list or not parsed_dates_iso_list[0]:
        logger.error("Date info missing for search_events_node.")
        current_collected_data_dict.setdefault(
            "clarification_needed_fields", []
        ).append("dates_description_original")
        return {
            **state,
            "collected_data": current_collected_data_dict,
            "current_events": [],
            "is_initial_plan_proposed": False,
        }

    try:
        temp_parsed_start_dt = datetime.fromisoformat(parsed_dates_iso_list[0])
        dates_desc_original_lower = current_collected_data_dict.get(
            "dates_description_original", ""
        ).lower()
        raw_time_desc_original = current_collected_data_dict.get(
            "raw_time_description_original"
        )
        if (
            "сегодня" in dates_desc_original_lower
            or "на сегодня" in dates_desc_original_lower
        ):
            is_today_plan_request = True
        if (
            is_today_plan_request
            and temp_parsed_start_dt.time() == time.min
            and not (
                raw_time_desc_original
                and any(char.isdigit() for char in raw_time_desc_original)
            )
        ):
            now_datetime = datetime.now()
            if temp_parsed_start_dt.date() == now_datetime.date():
                user_min_start_dt_naive = temp_parsed_start_dt.replace(
                    hour=now_datetime.hour,
                    minute=now_datetime.minute,
                    second=now_datetime.second,
                    microsecond=now_datetime.microsecond,
                )
                current_collected_data_dict["parsed_dates_iso"] = [
                    user_min_start_dt_naive.isoformat()
                ]
                logger.info(
                    f"Adjusted user_min_start_dt_naive for 'today' to current time: {user_min_start_dt_naive}"
                )
            else:
                user_min_start_dt_naive = temp_parsed_start_dt
        else:
            user_min_start_dt_naive = temp_parsed_start_dt
        tool_api_date_from_dt = user_min_start_dt_naive.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if not (
            user_min_start_dt_naive.hour == 0 and user_min_start_dt_naive.minute == 0
        ):
            tool_min_start_time_str = user_min_start_dt_naive.strftime("%H:%M")
        parsed_end_dates_iso_list = current_collected_data_dict.get(
            "parsed_end_dates_iso"
        )
        temp_api_date_to_dt_for_search: datetime
        if parsed_end_dates_iso_list and parsed_end_dates_iso_list[0]:
            user_max_overall_end_dt_naive = datetime.fromisoformat(
                parsed_end_dates_iso_list[0]
            )
            if (
                user_max_overall_end_dt_naive.time() == time.min
                and user_max_overall_end_dt_naive.second == 0
            ):
                user_max_overall_end_dt_naive = user_max_overall_end_dt_naive.replace(
                    hour=23, minute=59, second=59
                )
            temp_api_date_to_dt_for_search = user_max_overall_end_dt_naive
            if temp_api_date_to_dt_for_search.date() == user_min_start_dt_naive.date():
                if (
                    user_max_overall_end_dt_naive.time() != time.min
                    and user_max_overall_end_dt_naive.time() != time(23, 59, 59)
                ):
                    tool_max_start_time_str = user_max_overall_end_dt_naive.strftime(
                        "%H:%M"
                    )
        else:
            user_max_overall_end_dt_naive = user_min_start_dt_naive.replace(
                hour=23, minute=59, second=59
            )
            temp_api_date_to_dt_for_search = user_max_overall_end_dt_naive
        tool_api_date_to_dt = temp_api_date_to_dt_for_search.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        logger.info(
            f"User Time Window: Start Naive: {user_min_start_dt_naive}, End Naive (Overall Plan Limit): {user_max_overall_end_dt_naive}, IsTodayRequest: {is_today_plan_request}"
        )
    except ValueError as e_date_parse:
        logger.error(f"Invalid date format in search_events_node: {e_date_parse}")
        current_collected_data_dict.setdefault(
            "clarification_needed_fields", []
        ).append("dates_description_original")
        return {
            **state,
            "collected_data": current_collected_data_dict,
            "current_events": [],
            "is_initial_plan_proposed": False,
        }

    ordered_activities_base_raw = current_collected_data_dict.get(
        "ordered_activities", []
    )
    ordered_activities_base = []
    if isinstance(ordered_activities_base_raw, list):
        for oa_raw in ordered_activities_base_raw:
            if isinstance(oa_raw, dict):
                try:
                    ordered_activities_base.append(OrderedActivityItem(**oa_raw))
                except ValidationError as e_oa_val:
                    logger.warning(
                        f"Skipping invalid OA in search_events_node: {oa_raw}. Err: {e_oa_val}"
                    )
            elif isinstance(oa_raw, OrderedActivityItem):
                ordered_activities_base.append(oa_raw)

    max_replan_attempts = 2
    replan_attempt = 0
    successful_plan_found = False
    best_final_chain_objects: List[Union[Event, Dict[str, Any]]] = []

    # FIX: Переменная для хранения "лучшего" (самого дешевого) варианта, который не уложился в бюджет
    best_overbudget_plan_info: Optional[Dict[str, Any]] = None

    while replan_attempt < max_replan_attempts and not successful_plan_found:
        replan_attempt += 1
        logger.info(f"Planning attempt #{replan_attempt}/{max_replan_attempts}...")

        ordered_activities_for_processing_obj: List[OrderedActivityItem] = list(
            ordered_activities_base
        )
        plan_modification_pending: bool = state.get("plan_modification_pending", False)
        modification_details_from_state: Optional[Dict[str, Any]] = state.get(
            "modification_request_details"
        )

        if plan_modification_pending and modification_details_from_state:
            logger.info(
                f"Applying plan modification based on: {modification_details_from_state}"
            )
            change_requests_for_activities_raw = modification_details_from_state.get(
                "change_requests", []
            )
            if change_requests_for_activities_raw:
                change_requests_parsed: List[ChangeRequestDetail] = []
                for cr_d in change_requests_for_activities_raw:
                    try:
                        change_requests_parsed.append(ChangeRequestDetail(**cr_d))
                    except ValidationError:
                        logger.warning(
                            f"Skipping invalid CR in search_events_node: {cr_d}"
                        )
                        continue

                temp_activities_after_ops = [
                    OrderedActivityItem(**oa.model_dump())
                    for oa in ordered_activities_base
                ]

                indices_to_remove = set()
                for cr_obj in change_requests_parsed:
                    if (
                        cr_obj.change_target == "specific_event_remove"
                        and cr_obj.item_to_change_details
                    ):
                        details = cr_obj.item_to_change_details
                        idx_found = -1
                        if (
                            details.item_index is not None
                            and 0 < details.item_index <= len(temp_activities_after_ops)
                        ):
                            idx_found = details.item_index - 1
                        elif details.item_type:
                            for i, act_item in enumerate(temp_activities_after_ops):
                                if (
                                    act_item.activity_type.upper()
                                    == details.item_type.upper()
                                ):
                                    idx_found = i
                                    break

                        if idx_found != -1:
                            indices_to_remove.add(idx_found)

                if indices_to_remove:
                    for idx in sorted(list(indices_to_remove), reverse=True):
                        removed_item = temp_activities_after_ops.pop(idx)
                        logger.info(
                            f"Modification: Removing activity request '{removed_item.activity_type}' at index {idx} from plan."
                        )

                for cr_obj in change_requests_parsed:
                    if (
                        cr_obj.change_target == "specific_event_replace"
                        and cr_obj.item_to_change_details
                        and cr_obj.new_value_activity
                    ):
                        details = cr_obj.item_to_change_details
                        idx_to_replace = -1
                        if (
                            details.item_index is not None
                            and 0 < details.item_index <= len(temp_activities_after_ops)
                        ):
                            idx_to_replace = details.item_index - 1
                        elif details.item_type:
                            for i, act_item in enumerate(temp_activities_after_ops):
                                if (
                                    act_item.activity_type.upper()
                                    == details.item_type.upper()
                                ):
                                    idx_to_replace = i
                                    break

                        if idx_to_replace != -1:
                            new_activity = OrderedActivityItem(
                                **cr_obj.new_value_activity.model_dump()
                            )
                            logger.info(
                                f"Modification: Replacing activity at index {idx_to_replace} with new activity '{new_activity.activity_type}'."
                            )
                            temp_activities_after_ops[idx_to_replace] = new_activity

                for cr_obj in change_requests_parsed:
                    if (
                        cr_obj.change_target == "add_activity"
                        and cr_obj.new_value_activity
                    ):
                        new_activity = OrderedActivityItem(
                            **cr_obj.new_value_activity.model_dump()
                        )
                        logger.info(
                            f"Modification: Adding new activity request '{new_activity.activity_type}' to plan."
                        )
                        temp_activities_after_ops.append(new_activity)

                ordered_activities_for_processing_obj = temp_activities_after_ops
                current_collected_data_dict["ordered_activities"] = [
                    oa.model_dump(exclude_none=True)
                    for oa in ordered_activities_for_processing_obj
                ]

        if not ordered_activities_for_processing_obj:
            if plan_modification_pending:
                logger.info("Plan is empty after modifications. Presenting empty plan.")
                return {
                    **state,
                    "current_events": [],
                    "collected_data": {
                        **current_collected_data_dict,
                        "selected_pois_for_plan": [],
                    },
                    "is_initial_plan_proposed": True,
                    "modification_request_details": None,
                    "plan_modification_pending": False,
                }
            if not current_collected_data_dict.get(
                "fallback_candidates"
            ) and not current_collected_data_dict.get("plan_construction_failed_step"):
                current_collected_data_dict.setdefault(
                    "clarification_needed_fields", []
                ).append("interests_original")
            return {
                **state,
                "current_events": [],
                "collected_data": current_collected_data_dict,
                "is_initial_plan_proposed": False,
            }

        all_candidates_by_step_index: Dict[int, List[Union[Event, Dict[str, Any]]]] = {}
        K_TOP_CANDIDATES_PER_STEP = 7
        initial_not_found_keys_this_search = []

        fixed_items_by_step_index: Dict[int, Union[Event, Dict[str, Any]]] = {}
        last_presented_plan = state.get("last_presented_plan")

        if (
            plan_modification_pending
            and modification_details_from_state
            and last_presented_plan
            and replan_attempt == 1
        ):
            logger.info(
                "Checking for unchanged items in the previous plan to preserve them."
            )

            changed_item_ids: Set[str] = set()
            for cr_dict in modification_details_from_state.get("change_requests", []):
                details = cr_dict.get("item_to_change_details")
                if details and details.get("item_id_str"):
                    changed_item_ids.add(str(details.get("item_id_str")))

            previous_plan_items: List[Dict[str, Any]] = (
                last_presented_plan.get("events", []) or []
            ) + (last_presented_plan.get("selected_pois", []) or [])

            for step_idx, activity_item in enumerate(
                ordered_activities_for_processing_obj
            ):
                activity_type_to_match = activity_item.activity_type.upper()

                for prev_item_dict in previous_plan_items:
                    prev_item_type_key = ""
                    if prev_item_dict.get("user_event_type_key"):
                        prev_item_type_key = prev_item_dict[
                            "user_event_type_key"
                        ].upper()
                    elif prev_item_dict.get("_plan_item_type"):
                        poi_type = prev_item_dict["_plan_item_type"].upper()
                        if poi_type == "PARK":
                            prev_item_type_key = "PARK"
                        elif poi_type == "FOOD":
                            prev_item_type_key = "FOOD_PLACE"

                    if activity_type_to_match == prev_item_type_key:
                        item_id_str = str(
                            prev_item_dict.get("session_id")
                            or prev_item_dict.get("id_gis")
                        )

                        if item_id_str not in changed_item_ids:
                            logger.info(
                                f"Preserving unchanged item '{prev_item_dict.get('name')}' for step {step_idx} (type: {activity_type_to_match})."
                            )

                            if prev_item_dict.get("user_event_type_key"):
                                try:
                                    fixed_items_by_step_index[step_idx] = Event(
                                        **prev_item_dict
                                    )
                                except ValidationError:
                                    fixed_items_by_step_index[step_idx] = prev_item_dict
                            else:
                                fixed_items_by_step_index[step_idx] = prev_item_dict

                            previous_plan_items.remove(prev_item_dict)
                            break
        elif replan_attempt > 1:
            logger.warning(
                f"Attempt #{replan_attempt}: Re-planning without preserving previous items to find a compatible chain."
            )

        logger.info(
            f"Phase 1: Gathering all candidates for {len(ordered_activities_for_processing_obj)} steps."
        )

        for step_idx, activity_item_obj in enumerate(
            ordered_activities_for_processing_obj
        ):
            if step_idx in fixed_items_by_step_index:
                fixed_item = fixed_items_by_step_index[step_idx]
                all_candidates_by_step_index[step_idx] = [fixed_item]
                item_name_log = (
                    fixed_item.name
                    if isinstance(fixed_item, Event)
                    else fixed_item.get("name", "Unknown POI")
                )
                logger.info(
                    f"Step {step_idx} ({activity_item_obj.activity_type}): Using preserved item '{item_name_log}' as the only candidate."
                )
                continue

            current_act_type = activity_item_obj.activity_type
            current_q_details = activity_item_obj.query_details
            activity_item_budget = activity_item_obj.activity_budget
            budget_for_this_step = (
                activity_item_budget
                if activity_item_budget is not None
                else general_budget
            )
            step_candidates_raw: List[Union[Event, Dict[str, Any]]] = []
            step_specific_exclude_afisha_ids = list(excluded_ids_map.get("afisha", []))
            step_specific_exclude_park_ids = list(excluded_ids_map.get("park", []))
            step_specific_exclude_food_ids = list(excluded_ids_map.get("food", []))

            effective_min_start_time_for_tool_afisha = tool_min_start_time_str
            effective_max_start_time_for_tool_afisha = tool_max_start_time_str

            is_being_replaced = False
            if plan_modification_pending and modification_details_from_state:
                for cr in modification_details_from_state.get("change_requests", []):
                    details = cr.get("item_to_change_details", {})
                    if details and (
                        details.get("item_index") == step_idx + 1
                        or details.get("item_type") == current_act_type
                    ):
                        is_being_replaced = True
                        break

            if (
                is_being_replaced
                and last_presented_plan
                and len(ordered_activities_for_processing_obj) > 1
                and step_idx < len(ordered_activities_for_processing_obj) - 1
            ):
                next_activity_request = ordered_activities_for_processing_obj[
                    step_idx + 1
                ]
                all_previous_items = (last_presented_plan.get("events", []) or []) + (
                    last_presented_plan.get("selected_pois", []) or []
                )

                next_fixed_item = None
                for item in all_previous_items:
                    item_type = item.get("user_event_type_key") or (
                        "PARK"
                        if item.get("_plan_item_type") == "park"
                        else (
                            "FOOD_PLACE"
                            if item.get("_plan_item_type") == "food"
                            else None
                        )
                    )
                    if (
                        item_type
                        and item_type.upper()
                        == next_activity_request.activity_type.upper()
                    ):
                        next_fixed_item = item
                        break

                if next_fixed_item:
                    try:
                        start_time_val = next_fixed_item.get(
                            "start_time_naive_event_tz"
                        ) or next_fixed_item.get("_calculated_start_time")
                        if start_time_val is None:
                            raise ValueError("Start time is None")
                        next_item_start_time = datetime.fromisoformat(
                            str(start_time_val)
                        )
                        buffer_seconds = 3 * 3600
                        max_start_time_for_current_event = (
                            next_item_start_time - timedelta(seconds=buffer_seconds)
                        )
                        effective_max_start_time_for_tool_afisha = (
                            max_start_time_for_current_event.strftime("%H:%M")
                        )
                        logger.info(
                            f"Modification constraint: Max start time for '{current_act_type}' set to {effective_max_start_time_for_tool_afisha} to fit before '{next_fixed_item.get('name')}'."
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Could not parse start time for next fixed item to create constraint: {e}"
                        )

            if current_act_type in CREATION_TYPES_AFISHA and current_act_type != "ANY":
                if (
                    not city_id_afisha
                    or not tool_api_date_from_dt
                    or not tool_api_date_to_dt
                ):
                    logger.warning(
                        f"Afisha data missing for step {step_idx} ({current_act_type})"
                    )
                    all_candidates_by_step_index[step_idx] = []
                    if current_act_type not in initial_not_found_keys_this_search:
                        initial_not_found_keys_this_search.append(current_act_type)
                        continue

                keywords_afisha = []
                if current_q_details:
                    modification_stop_words = {
                        "другой",
                        "другая",
                        "другое",
                        "замени",
                        "вместо",
                        "поменяй",
                        "измени",
                        "новый",
                        "новая",
                        "новое",
                    }
                    activity_type_names = {
                        "спектакль",
                        "фильм",
                        "кино",
                        "концерт",
                        "стендап",
                        "выставка",
                        "музей",
                    }

                    query_words = current_q_details.lower().split()
                    meaningful_words = [
                        word
                        for word in query_words
                        if word not in modification_stop_words
                        and word not in activity_type_names
                    ]

                    if meaningful_words:
                        cleaned_query = " ".join(meaningful_words)
                        keywords_afisha.append(cleaned_query)

                if (
                    step_idx == 0
                    and user_min_start_dt_naive
                    and user_min_start_dt_naive.time() != time.min
                    and user_min_start_dt_naive.date() == tool_api_date_from_dt.date()
                    and not effective_min_start_time_for_tool_afisha
                ):
                    effective_min_start_time_for_tool_afisha = (
                        user_min_start_dt_naive.strftime("%H:%M")
                    )

                tool_args_afisha_gather = EventSearchToolArgs(
                    city_id=city_id_afisha,
                    date_from=tool_api_date_from_dt,
                    date_to=tool_api_date_to_dt,
                    user_creation_type_key=current_act_type,
                    filter_keywords_in_name=(
                        keywords_afisha if keywords_afisha else None
                    ),
                    min_start_time_naive=effective_min_start_time_for_tool_afisha,
                    max_start_time_naive=effective_max_start_time_for_tool_afisha,
                    max_budget_per_person=budget_for_this_step,
                    exclude_session_ids=step_specific_exclude_afisha_ids,
                )
                try:
                    afisha_results_dicts = await event_search_tool.ainvoke(
                        tool_args_afisha_gather.model_dump(exclude_none=True)
                    )
                    for evt_data in afisha_results_dicts:
                        try:
                            cand_name_lower_afisha = evt_data.get("name", "").lower()
                            cand_afisha_id_str_val = str(evt_data.get("afisha_id", ""))
                            if cand_name_lower_afisha in [
                                n.lower()
                                for n in excluded_ids_map.get(
                                    "afisha_names_to_avoid", []
                                )
                            ]:
                                continue
                            if cand_afisha_id_str_val in excluded_ids_map.get(
                                "afisha_creation_ids_to_avoid", []
                            ):
                                continue
                            event_obj = Event(**evt_data)
                            starts_too_early = False
                            if (
                                user_min_start_dt_naive
                                and event_obj.start_time_naive_event_tz
                                < user_min_start_dt_naive
                            ):
                                starts_too_early = True
                            ends_too_late_for_non_today = False
                            if (
                                not is_today_plan_request
                                and user_max_overall_end_dt_naive
                            ):
                                event_end_check_f1 = (
                                    event_obj.start_time_naive_event_tz
                                    + timedelta(
                                        minutes=(
                                            event_obj.duration_minutes
                                            or default_event_duration_minutes
                                        )
                                    )
                                )
                                if event_end_check_f1 > user_max_overall_end_dt_naive:
                                    ends_too_late_for_non_today = True
                            if starts_too_early or ends_too_late_for_non_today:
                                logger.debug(
                                    f"Phase 1 Hard Skip: '{event_obj.name}' ({current_act_type}). Early: {starts_too_early}, LateStrict(non-today): {ends_too_late_for_non_today}"
                                )
                                continue
                            step_candidates_raw.append(event_obj)
                        except ValidationError:
                            continue
                except Exception as e_gather_afisha:
                    logger.error(
                        f"Error gathering Afisha for {current_act_type} in search_events_node: {e_gather_afisha}"
                    )
            elif current_act_type == "PARK":
                if city_name_for_gis:
                    park_poi_candidates = await search_parks(
                        original_query=current_q_details or "парк",
                        city=city_name_for_gis,
                        limit=K_TOP_CANDIDATES_PER_STEP * 3,
                        exclude_ids=step_specific_exclude_park_ids,
                    )
                    step_candidates_raw.extend(
                        [p.model_dump(exclude_none=True) for p in park_poi_candidates]
                    )
            elif current_act_type == "FOOD_PLACE":
                if city_name_for_gis:
                    food_poi_candidates = await search_food_places(
                        original_query=current_q_details or "еда",
                        city=city_name_for_gis,
                        limit=K_TOP_CANDIDATES_PER_STEP * 3,
                        exclude_ids=step_specific_exclude_food_ids,
                    )
                    step_candidates_raw.extend(
                        [f.model_dump(exclude_none=True) for f in food_poi_candidates]
                    )
            if step_candidates_raw:
                if current_act_type in CREATION_TYPES_AFISHA:
                    step_candidates_raw.sort(
                        key=lambda x: (
                            x.start_time_naive_event_tz
                            if isinstance(x, Event)
                            else datetime.max
                        )
                    )
                all_candidates_by_step_index[step_idx] = step_candidates_raw[
                    :K_TOP_CANDIDATES_PER_STEP
                ]
                logger.info(
                    f"Step {step_idx} ({current_act_type}, q:'{current_q_details}'): Gathered {len(all_candidates_by_step_index[step_idx])} candidates (from {len(step_candidates_raw)} raw)."
                )
            else:
                all_candidates_by_step_index[step_idx] = []
                logger.warning(
                    f"Step {step_idx} ({current_act_type}, q:'{current_q_details}'): No candidates found for primary search."
                )
                if current_act_type not in initial_not_found_keys_this_search:
                    initial_not_found_keys_this_search.append(current_act_type)

        current_collected_data_dict["not_found_interest_keys_in_primary_search"] = list(
            set(initial_not_found_keys_this_search)
        )

        best_final_chain_objects_attempt: List[Union[Event, Dict[str, Any]]] = []
        failed_activity_step_info_from_chaining: Optional[str] = None
        candidate_lists_for_product_phase2: List[List[Union[Event, Dict[str, Any]]]] = (
            []
        )
        ordered_activities_for_chain_construction: List[OrderedActivityItem] = []
        for i_oa, oa_item in enumerate(ordered_activities_for_processing_obj):
            candidates_for_this_step = all_candidates_by_step_index.get(i_oa)
            if candidates_for_this_step:
                candidate_lists_for_product_phase2.append(candidates_for_this_step)
                ordered_activities_for_chain_construction.append(oa_item)
            else:
                logger.warning(
                    f"No candidates for mandatory step {i_oa} ({oa_item.activity_type}) in search_events_node, this activity will be skipped in chain construction."
                )
                if not failed_activity_step_info_from_chaining:
                    failed_activity_step_info_from_chaining = f"не удалось найти кандидатов для «{oa_item.query_details or oa_item.activity_type}» на первом этапе"
        if (
            not candidate_lists_for_product_phase2
            and ordered_activities_for_processing_obj
        ):
            if not failed_activity_step_info_from_chaining:
                failed_activity_step_info_from_chaining = (
                    "не из чего строить цепочку (нет кандидатов ни для одного шага)."
                )
        user_start_coords_dict = current_collected_data_dict.get(
            "user_start_address_validated_coords"
        )
        MAX_COMBINATIONS_TO_CHECK = 3000
        processed_combinations_count = 0
        best_final_chain_score = float("inf")

        # FIX: Переменная для хранения цепочек, не прошедших по бюджету
        chains_failed_due_to_budget = []

        if candidate_lists_for_product_phase2:
            logger.info(
                f"Phase 2: Starting chain construction with {len(candidate_lists_for_product_phase2)} steps having candidates."
            )
            for current_combination_of_candidates_tuple in itertools.product(
                *candidate_lists_for_product_phase2
            ):
                processed_combinations_count += 1
                if processed_combinations_count > MAX_COMBINATIONS_TO_CHECK:
                    logger.warning(
                        f"Reached MAX_COMBINATIONS_TO_CHECK ({MAX_COMBINATIONS_TO_CHECK}). Stopping chain construction."
                    )
                    break

                # FIX: Начало блока проверки общего бюджета для комбинации
                if general_budget is not None:
                    current_combination_total_cost = 0
                    for item in current_combination_of_candidates_tuple:
                        if isinstance(item, Event) and item.min_price is not None:
                            current_combination_total_cost += item.min_price

                    total_cost_for_all_persons = (
                        current_combination_total_cost * person_count
                    )

                    if total_cost_for_all_persons > general_budget:
                        chains_failed_due_to_budget.append(
                            {
                                "plan": current_combination_of_candidates_tuple,
                                "cost": total_cost_for_all_persons,
                            }
                        )
                        logger.debug(
                            f"Chain failed BUDGET check. Cost: {total_cost_for_all_persons} > Budget: {general_budget}. Chain: {[getattr(item, 'name', 'POI') for item in current_combination_of_candidates_tuple]}"
                        )
                        continue
                # FIX: Конец блока проверки общего бюджета

                current_chain_segment_dicts: List[Dict[str, Any]] = []
                current_chain_is_valid = True
                current_total_travel_seconds = 0
                last_successful_item_data_for_check: Dict[str, Any]
                last_successful_item_type_for_check: Optional[str] = None
                current_chain_time_for_iter = user_min_start_dt_naive
                if user_start_coords_dict and current_chain_time_for_iter:
                    last_successful_item_data_for_check = {
                        "name": "UserStartLocation",
                        "_calculated_end_time": current_chain_time_for_iter.isoformat(),
                        "coords": [
                            user_start_coords_dict["lon"],
                            user_start_coords_dict["lat"],
                        ],
                        "place_coords_lon": user_start_coords_dict["lon"],
                        "place_coords_lat": user_start_coords_dict["lat"],
                        "_plan_item_type": "poi_start",
                    }
                    last_successful_item_type_for_check = "poi_start"
                elif current_chain_time_for_iter:
                    last_successful_item_data_for_check = {
                        "name": "PlanStartNoAddress",
                        "_calculated_end_time": current_chain_time_for_iter.isoformat(),
                        "_plan_item_type": "poi_start",
                    }
                    last_successful_item_type_for_check = "poi_start"
                else:
                    logger.error(
                        "Cannot start chain: user start time is unknown in search_events_node."
                    )
                    current_chain_is_valid = False
                    break
                if not current_chain_is_valid:
                    continue
                for step_idx_chain, candidate_item_for_chain in enumerate(
                    current_combination_of_candidates_tuple
                ):
                    candidate_item_dict_form: Dict[str, Any]
                    original_activity_type_for_step_chain = (
                        ordered_activities_for_chain_construction[
                            step_idx_chain
                        ].activity_type
                    )
                    if isinstance(candidate_item_for_chain, Event):
                        candidate_item_dict_form = candidate_item_for_chain.model_dump(
                            exclude_none=True
                        )
                        candidate_item_dict_form["_plan_item_type"] = "afisha_event"
                        candidate_item_dict_form.setdefault(
                            "user_event_type_key", original_activity_type_for_step_chain
                        )
                    elif isinstance(candidate_item_for_chain, dict):
                        candidate_item_dict_form = dict(candidate_item_for_chain)
                        if "_plan_item_type" not in candidate_item_dict_form:
                            if original_activity_type_for_step_chain == "PARK":
                                candidate_item_dict_form["_plan_item_type"] = "park"
                            elif original_activity_type_for_step_chain == "FOOD_PLACE":
                                candidate_item_dict_form["_plan_item_type"] = "food"
                    else:
                        current_chain_is_valid = False
                        break
                    if not current_chain_is_valid:
                        break
                    is_compatible, err_msg, calc_end_time, warn_msg, route_dur = (
                        await _check_event_compatibility(
                            first_item_data=last_successful_item_data_for_check,
                            second_item_data=candidate_item_dict_form,
                            user_max_overall_end_dt_naive=user_max_overall_end_dt_naive,
                            city_name_for_gis=city_name_for_gis,
                            first_item_type=last_successful_item_type_for_check,
                            is_today_request_for_plan=is_today_plan_request,
                            user_min_start_dt_naive_for_today=(
                                user_min_start_dt_naive
                                if is_today_plan_request
                                else None
                            ),
                        )
                    )
                    if not is_compatible:
                        current_chain_is_valid = False
                        logger.debug(
                            f"Chain broken in search_events_node: {candidate_item_dict_form.get('name')} X {last_successful_item_data_for_check.get('name')}. Reason: {err_msg}"
                        )
                        break
                    candidate_item_dict_form["_calculated_end_time"] = (
                        calc_end_time.isoformat() if calc_end_time else None
                    )
                    if "_calculated_start_time" not in candidate_item_dict_form:
                        if candidate_item_dict_form.get("start_time_naive_event_tz"):
                            start_time_val_calc = candidate_item_dict_form.get(
                                "start_time_naive_event_tz"
                            )
                            candidate_item_dict_form["_calculated_start_time"] = (
                                start_time_val_calc.isoformat()
                                if isinstance(start_time_val_calc, datetime)
                                else str(start_time_val_calc)
                            )
                        else:
                            logger.warning(
                                f"Calculated start time missing for {candidate_item_dict_form.get('name')} in search_events_node"
                            )
                            current_chain_is_valid = False
                            break
                    candidate_item_dict_form["_compatibility_warning"] = warn_msg
                    candidate_item_dict_form["_travel_time_to_seconds"] = route_dur
                    current_chain_segment_dicts.append(candidate_item_dict_form)
                    current_total_travel_seconds += route_dur or 0
                    last_successful_item_data_for_check = candidate_item_dict_form
                    last_successful_item_type_for_check = candidate_item_dict_form.get(
                        "_plan_item_type"
                    )
                if current_chain_is_valid and len(current_chain_segment_dicts) == len(
                    candidate_lists_for_product_phase2
                ):
                    if current_total_travel_seconds < best_final_chain_score:
                        best_final_chain_score = current_total_travel_seconds
                        best_final_chain_objects_attempt = []
                        for item_dict_bfco in current_chain_segment_dicts:
                            if item_dict_bfco.get("_plan_item_type") == "afisha_event":
                                try:
                                    best_final_chain_objects_attempt.append(
                                        Event(**item_dict_bfco)
                                    )
                                except ValidationError as e_val_bfco:
                                    logger.error(
                                        f"Error converting dict to Event in best chain (search_events_node): {e_val_bfco}"
                                    )
                                    best_final_chain_objects_attempt.append(
                                        item_dict_bfco
                                    )
                            else:
                                best_final_chain_objects_attempt.append(item_dict_bfco)
                        logger.info(
                            f"Found new best chain! Score (travel_s): {best_final_chain_score}. Len: {len(best_final_chain_objects_attempt)}. Names: {[item.get('name') if isinstance(item, dict) else item.name for item in best_final_chain_objects_attempt]}"
                        )

        if best_final_chain_objects_attempt:
            best_final_chain_objects = best_final_chain_objects_attempt
            successful_plan_found = True
        else:
            logger.warning(f"Attempt #{replan_attempt} failed to produce a valid plan.")
            # FIX: Если план не найден, но есть варианты, не прошедшие по бюджету, сохраняем информацию о них
            if chains_failed_due_to_budget:
                chains_failed_due_to_budget.sort(key=lambda x: x["cost"])
                cheapest_overbudget_plan_info = chains_failed_due_to_budget[0]

                # Конвертируем event-объекты в словари для сохранения в state
                plan_as_dicts = []
                for item in cheapest_overbudget_plan_info["plan"]:
                    if isinstance(item, BaseModel):
                        plan_as_dicts.append(item.model_dump(exclude_none=True))
                    else:
                        plan_as_dicts.append(item)

                best_overbudget_plan_info = {
                    "plan_items": plan_as_dicts,
                    "cost": cheapest_overbudget_plan_info["cost"],
                }
                logger.info(
                    f"Found a potential over-budget plan. Cost: {best_overbudget_plan_info['cost']}, Items: {[item.get('name') for item in plan_as_dicts]}"
                )

    # NEW LOGIC: Если идеальный план не найден, но найден "дорогой" план, сохраняем его в state для узла-презентатора
    if not best_final_chain_objects and best_overbudget_plan_info:
        current_collected_data_dict["budget_fallback_plan"] = best_overbudget_plan_info

    final_not_found_keys_after_chaining_step_val = []
    activity_types_in_best_chain_val = set()
    for item_in_chain_val in best_final_chain_objects:
        if isinstance(item_in_chain_val, Event):
            activity_types_in_best_chain_val.add(item_in_chain_val.user_event_type_key)
        elif isinstance(item_in_chain_val, dict):
            poi_type_key_map_val = {"park": "PARK", "food": "FOOD_PLACE"}
            mapped_type_val = poi_type_key_map_val.get(
                item_in_chain_val.get("_plan_item_type", "")
            )
            if mapped_type_val:
                activity_types_in_best_chain_val.add(mapped_type_val)
    for oa_item_obj_final_check_val in ordered_activities_for_processing_obj:
        if (
            oa_item_obj_final_check_val.activity_type
            not in activity_types_in_best_chain_val
        ):
            if (
                oa_item_obj_final_check_val.activity_type
                not in final_not_found_keys_after_chaining_step_val
            ):
                final_not_found_keys_after_chaining_step_val.append(
                    oa_item_obj_final_check_val.activity_type
                )
    current_collected_data_dict["not_found_interest_keys_in_primary_search"] = (
        final_not_found_keys_after_chaining_step_val
    )
    if (
        not best_final_chain_objects
        and not failed_activity_step_info_from_chaining
        and ordered_activities_for_processing_obj
        and not best_overbudget_plan_info  # FIX: Не перезаписываем сообщение об ошибке, если проблема в бюджете
    ):
        failed_activity_step_info_from_chaining = (
            "не удалось построить совместимую цепочку из найденных кандидатов."
        )

    interests_needing_fallback_val = list(
        set(
            current_collected_data_dict.get(
                "not_found_interest_keys_in_primary_search", []
            )
        )
    )


    if (
        interests_needing_fallback_val
        and city_name_for_gis
        and tool_api_date_from_dt
        and not current_collected_data_dict.get("budget_fallback_plan")
    ):
        logger.info(
            f"Phase 3: Searching fallback for: {interests_needing_fallback_val}"
        )
        fb_tool_date_from_dt_val_fb = tool_api_date_from_dt
        fb_tool_date_to_dt_val_fb = (tool_api_date_from_dt + timedelta(days=6)).replace(
            hour=23, minute=59, second=59
        )
        user_time_desc_for_fb_val_fb_val = current_collected_data_dict.get(
            "dates_description_original", "запрошенное вами время"
        )
        if current_collected_data_dict.get("raw_time_description_original"):
            user_time_desc_for_fb_val_fb_val += f" ({escape_markdown_v2(current_collected_data_dict.get('raw_time_description_original'))})"
        current_collected_data_dict["user_time_desc_for_fallback"] = (
            user_time_desc_for_fb_val_fb_val
        )

        for interest_key_fb_s_val in interests_needing_fallback_val:
            if interest_key_fb_s_val in current_collected_data_dict.get(
                "fallback_candidates", {}
            ) or interest_key_fb_s_val in current_collected_data_dict.get(
                "combo_fallback_candidates", {}
            ):
                continue

            if not (interest_key_fb_s_val in CREATION_TYPES_AFISHA):
                logger.warning(
                    f"Fallback search skipped for unknown interest key: {interest_key_fb_s_val}"
                )
                continue

            fb_exclude_ids_val_fb = list(excluded_ids_map.get("afisha", []))
            for item_in_best_chain_fb in best_final_chain_objects:
                if (
                    isinstance(item_in_best_chain_fb, Event)
                    and item_in_best_chain_fb.session_id not in fb_exclude_ids_val_fb
                ):
                    fb_exclude_ids_val_fb.append(item_in_best_chain_fb.session_id)
            tool_args_fb_s_val = EventSearchToolArgs(
                city_id=city_id_afisha,
                date_from=fb_tool_date_from_dt_val_fb,
                date_to=(
                    fb_tool_date_to_dt_val_fb.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    + timedelta(days=1)
                ),
                user_creation_type_key=interest_key_fb_s_val,
                max_budget_per_person=general_budget,  # Поиск fallback'а тоже ограничен бюджетом
                exclude_session_ids=(
                    list(set(fb_exclude_ids_val_fb)) if fb_exclude_ids_val_fb else None
                ),
            )
            try:
                fb_ev_list_raw_val: List[Dict] = await event_search_tool.ainvoke(
                    tool_args_fb_s_val.model_dump(exclude_none=True)
                )
                if fb_ev_list_raw_val:
                    fb_names_to_avoid_val = excluded_ids_map.get(
                        "afisha_names_to_avoid", []
                    )
                    fb_creation_ids_to_avoid_val = excluded_ids_map.get(
                        "afisha_creation_ids_to_avoid", []
                    )
                    final_fb_candidates_list_val = []
                    for fb_cand_dict_val in fb_ev_list_raw_val:
                        skip_val = False
                        if fb_names_to_avoid_val and fb_cand_dict_val.get(
                            "name", ""
                        ).lower() in [n.lower() for n in fb_names_to_avoid_val]:
                            skip_val = True
                        if (
                            fb_creation_ids_to_avoid_val
                            and str(fb_cand_dict_val.get("afisha_id", ""))
                            in fb_creation_ids_to_avoid_val
                        ):
                            skip_val = True
                        if not skip_val:
                            final_fb_candidates_list_val.append(fb_cand_dict_val)
                    if final_fb_candidates_list_val:
                        fb_ev_dict_chosen_val = final_fb_candidates_list_val[0]
                        fb_ev_dict_chosen_val["user_event_type_key"] = (
                            interest_key_fb_s_val
                        )
                        fb_ev_obj_val = Event(**fb_ev_dict_chosen_val)

                        # NEW LOGIC: Проверяем бюджет для fallback-кандидата по дате
                        if general_budget is not None:
                            cost_of_items_in_plan = sum(
                                item.min_price
                                for item in best_final_chain_objects
                                if isinstance(item, Event) and item.min_price
                            )
                            new_total_cost = (
                                cost_of_items_in_plan
                                + (fb_ev_obj_val.min_price or 0) * person_count
                            )

                            if new_total_cost > general_budget:
                                # Это "комбо" fallback - другая дата и дороже бюджета
                                logger.info(
                                    f"Found Combo (Date+Budget) fallback for {interest_key_fb_s_val}: {fb_ev_obj_val.name}. Cost: {new_total_cost}"
                                )
                                current_collected_data_dict.setdefault(
                                    "combo_fallback_candidates", {}
                                )[interest_key_fb_s_val] = {
                                    "event": fb_ev_obj_val.model_dump(
                                        exclude_none=True
                                    ),
                                    "total_plan_cost": new_total_cost,
                                }
                            else:
                                # Это обычный fallback по дате, он укладывается в бюджет
                                logger.info(
                                    f"Found Afisha (Date) fallback for {interest_key_fb_s_val}: {fb_ev_obj_val.name}"
                                )
                                current_collected_data_dict.setdefault(
                                    "fallback_candidates", {}
                                )[interest_key_fb_s_val] = fb_ev_obj_val.model_dump(
                                    exclude_none=True
                                )
                        else:
                            # Бюджет не указан, считаем это обычным fallback по дате
                            logger.info(
                                f"Found Afisha (Date) fallback for {interest_key_fb_s_val}: {fb_ev_obj_val.name}"
                            )
                            current_collected_data_dict.setdefault(
                                "fallback_candidates", {}
                            )[interest_key_fb_s_val] = fb_ev_obj_val.model_dump(
                                exclude_none=True
                            )

                        if interest_key_fb_s_val in current_collected_data_dict.get(
                            "not_found_interest_keys_in_primary_search", []
                        ):
                            current_collected_data_dict[
                                "not_found_interest_keys_in_primary_search"
                            ].remove(interest_key_fb_s_val)

            except Exception as e_fb_s_final_val_exc:
                logger.error(
                    f"Error in Afisha fallback for '{interest_key_fb_s_val}' in search_events_node: {e_fb_s_final_val_exc}",
                    exc_info=True,
                )

    all_requested_covered_final_val = True
    final_uncovered_activities_messages_val = []
    for oa_item_final_check_val_check in ordered_activities_for_processing_obj:
        is_in_chain_final_val = False
        for chain_item_final_val in best_final_chain_objects:
            current_item_type_key_val = ""
            if isinstance(chain_item_final_val, Event):
                current_item_type_key_val = chain_item_final_val.user_event_type_key
            elif isinstance(chain_item_final_val, dict):
                poi_type_key_map_val_check = {"park": "PARK", "food": "FOOD_PLACE"}
                current_item_type_key_val = poi_type_key_map_val_check.get(
                    chain_item_final_val.get("_plan_item_type", ""), ""
                )
            if current_item_type_key_val == oa_item_final_check_val_check.activity_type:
                is_in_chain_final_val = True
                break

        is_in_budget_fallback = False
        if current_collected_data_dict.get("budget_fallback_plan"):
            for item in current_collected_data_dict["budget_fallback_plan"][
                "plan_items"
            ]:
                if (
                    item.get("user_event_type_key")
                    == oa_item_final_check_val_check.activity_type
                ):
                    is_in_budget_fallback = True
                    break

        is_in_date_fallback = (
            oa_item_final_check_val_check.activity_type
            in current_collected_data_dict.get("fallback_candidates", {})
        )

        is_in_combo_fallback = (
            oa_item_final_check_val_check.activity_type
            in current_collected_data_dict.get("combo_fallback_candidates", {})
        )

        if (
            not is_in_chain_final_val
            and not is_in_budget_fallback
            and not is_in_date_fallback
            and not is_in_combo_fallback
        ):
            all_requested_covered_final_val = False
            final_uncovered_activities_messages_val.append(
                f"«{oa_item_final_check_val_check.query_details or oa_item_final_check_val_check.activity_type}»"
            )

    if not all_requested_covered_final_val and final_uncovered_activities_messages_val:
        failed_activity_step_info_from_chaining = f"не удалось найти или подобрать варианты для: {', '.join(final_uncovered_activities_messages_val)}"
    elif all_requested_covered_final_val and failed_activity_step_info_from_chaining:
        logger.info(
            f"Clearing plan_construction_failed_step ('{failed_activity_step_info_from_chaining}') as all items are now covered by plan or fallback."
        )
        failed_activity_step_info_from_chaining = None

    current_collected_data_dict["plan_construction_failed_step"] = (
        failed_activity_step_info_from_chaining
    )
    final_events_for_state_output_val: List[Event] = []
    final_pois_for_state_output_val: List[Dict[str, Any]] = []
    if best_final_chain_objects:
        for item_obj_or_dict_out_val in best_final_chain_objects:
            if isinstance(item_obj_or_dict_out_val, Event):
                final_events_for_state_output_val.append(item_obj_or_dict_out_val)
            elif isinstance(item_obj_or_dict_out_val, dict):
                final_pois_for_state_output_val.append(item_obj_or_dict_out_val)
    current_collected_data_dict["selected_pois_for_plan"] = (
        final_pois_for_state_output_val
    )
    is_plan_now_proposed_val_final = (
        bool(final_events_for_state_output_val)
        or bool(final_pois_for_state_output_val)
        or bool(current_collected_data_dict.get("fallback_candidates"))
        or bool(current_collected_data_dict.get("combo_fallback_candidates"))
        or bool(current_collected_data_dict.get("budget_fallback_plan"))
        or bool(current_collected_data_dict.get("plan_construction_failed_step"))
    )
    current_collected_data_dict["plan_construction_strategy"] = "standard"
    node_return_values_final_val = {
        **state,
        "current_events": final_events_for_state_output_val,
        "collected_data": current_collected_data_dict,
        "is_initial_plan_proposed": is_plan_now_proposed_val_final,
        "modification_request_details": None,
        "plan_modification_pending": False,
    }
    logger.info(
        f"search_events_node (optimal) finished. Events: {len(final_events_for_state_output_val)}, POIs: {len(final_pois_for_state_output_val)}. Date Fallbacks: {len(current_collected_data_dict.get('fallback_candidates', {}))}. Budget Fallbacks: {1 if current_collected_data_dict.get('budget_fallback_plan') else 0}. Combo Fallbacks: {len(current_collected_data_dict.get('combo_fallback_candidates', {}))}. Failed step msg: {current_collected_data_dict.get('plan_construction_failed_step')}"
    )
    return node_return_values_final_val
    # REFACTORED: Конец изменений в функции


async def gather_all_candidate_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: gather_all_candidate_events_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))

    city_id = collected_data_dict.get("city_id_afisha")
    city_name_for_filter = collected_data_dict.get(
        "city_name"
    )  # <--- ПОЛУЧАЕМ ИМЯ ГОРОДА
    parsed_dates_iso_list = collected_data_dict.get("parsed_dates_iso")
    interests_keys: List[str] = list(
        collected_data_dict.get("interests_keys_afisha", [])
    )
    budget = collected_data_dict.get("budget_current_search")

    candidate_events_by_interest: Dict[str, List[Event]] = {}
    search_errors_by_interest: Dict[str, str] = collected_data_dict.get(
        "search_errors_by_interest", {}
    )

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
            city_name=city_name_for_filter,  # <--- ПЕРЕДАЕМ ИМЯ ГОРОДА
            date_from=base_date_from_dt,
            date_to=api_date_to_dt,
            interests_keys=[interest_key],
            min_start_time_naive=search_min_start_time_naive,
            max_start_time_naive=search_max_start_time_naive,
            max_budget_per_person=budget,
            exclude_session_ids=None,  # Можно добавить сюда логику по exclude, если нужно
        )
        logger.debug(
            f"Gather candidates: Invoking event_search_tool for interest '{interest_key}' with args: {tool_args.model_dump_json(indent=2)}"
        )
        search_tasks.append(
            event_search_tool.ainvoke(tool_args.model_dump(exclude_none=True))
        )

    results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    for i, interest_key in enumerate(interests_keys):
        result_item = results_list[i]
        if isinstance(result_item, Exception):
            logger.error(
                f"gather_all_candidate_events_node: Error searching for interest '{interest_key}': {result_item}",
                exc_info=True,  # Было True, сохраняем
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
                    elif (  # Если длительность неизвестна, но событие НАЧИНАЕТСЯ позже максимально допустимого времени ОКОНЧАНИЯ
                        user_max_overall_end_dt_naive
                        and event_obj.duration_minutes is None
                        and event_obj.start_time_naive_event_tz
                        > user_max_overall_end_dt_naive
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
            if (
                not valid_events_for_interest
                and interest_key not in search_errors_by_interest
            ):  # Если не было ошибки API, но список пуст
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


# --- Утилита для экранирования MarkdownV2 ---
def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для MarkdownV2."""
    if not text:
        return ""
    escape_chars = r"_*[]()~`>#+-.=|{}!"  # Точка и дефис добавлены
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


# Словарь для более читаемых названий типов событий
AFISHA_KEY_TO_READABLE_NAME_MAP = {
    "MOVIE": "Фильм",  # Изменено на единственное число для заголовка
    "PERFORMANCE": "Спектакль",
    "CONCERT": "Концерт",
    "MUSEUM_EXHIBITION": "Музей/Выставка",
    "STAND_UP": "Стендап",
    "EVENT": "Событие",
    "EXCURSION": "Экскурсия",
    "SPORT_EVENT": "Спорт",
    "FOOD_PLACE": "Кафе/Ресторан",
    "PARK": "Парк/Сквер",
}


async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    return_state_update: Dict[str, Any] = {
        key: value for key, value in state.items() if key != "messages"
    }
    return_state_update["messages"] = list(state.get("messages", []))

    current_collected_data_dict: dict = dict(state.get("collected_data", {}))
    current_afisha_events_objects: List[Event] = state.get("current_events", [])
    selected_pois_for_plan_dicts: List[Dict[str, Any]] = (
        current_collected_data_dict.get("selected_pois_for_plan", [])
    )

    budget_fallback_plan_data: Optional[Dict[str, Any]] = (
        current_collected_data_dict.get("budget_fallback_plan")
    )
    date_fallback_candidates_map: Dict[str, Dict[str, Any]] = (
        current_collected_data_dict.get("fallback_candidates", {})
    )
    combo_fallback_candidates_map: Dict[str, Dict[str, Any]] = (
        current_collected_data_dict.get("combo_fallback_candidates", {})
    )
    person_count = current_collected_data_dict.get("person_count", 1) or 1
    original_budget = current_collected_data_dict.get("budget_original")

    response_parts = []
    ask_about_fallback = False
    pending_fallback_event_for_state_dict: Optional[Dict[str, Any]] = None
    field_to_be_clarified_next: Optional[str] = None
    plan_items_to_display: List[Dict[str, Any]] = []

    ask_for_poi_optimization = False
    poi_warnings_in_plan: List[str] = current_collected_data_dict.get(
        "poi_warnings_in_current_plan", []
    )
    if poi_warnings_in_plan:
        ask_for_poi_optimization = True
        field_to_be_clarified_next = "poi_time_optimization_choice"

    if current_afisha_events_objects:
        for evt_obj in current_afisha_events_objects:
            evt_dict = evt_obj.model_dump(exclude_none=True)
            evt_dict["_display_start_time"] = evt_obj.start_time_naive_event_tz
            evt_dict["_display_type"] = "afisha_event"
            evt_dict["_original_object"] = evt_obj
            plan_items_to_display.append(evt_dict)
    for poi_dict in selected_pois_for_plan_dicts:
        poi_dict_copy = dict(poi_dict)
        start_time_poi_raw = poi_dict.get("_calculated_start_time")
        effective_start_time_poi: Optional[datetime] = None
        if isinstance(start_time_poi_raw, str):
            try:
                effective_start_time_poi = datetime.fromisoformat(start_time_poi_raw)
            except ValueError:
                effective_start_time_poi = datetime.max
        elif isinstance(start_time_poi_raw, datetime):
            effective_start_time_poi = start_time_poi_raw
        else:
            effective_start_time_poi = datetime.max
        poi_dict_copy["_display_start_time"] = effective_start_time_poi
        poi_dict_copy["_display_type"] = poi_dict.get("_plan_item_type", "poi")
        plan_items_to_display.append(poi_dict_copy)
    plan_items_to_display.sort(key=lambda x: x.get("_display_start_time", datetime.max))

    plan_items_count = 0
    if plan_items_to_display:
        response_parts.append(
            "Вот что я смог для вас подобрать в качестве предварительного плана:"
        )
        for item_idx, item_data in enumerate(plan_items_to_display):
            plan_items_count += 1

            item_type_key_to_check = item_data.get(
                "user_event_type_key"
            ) or item_data.get("_display_type")
            if not isinstance(item_type_key_to_check, str):
                item_type_key_to_check = "EVENT"
            item_type_readable_raw = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
                item_type_key_to_check.upper(), item_type_key_to_check
            )

            item_name_raw = item_data.get("name", "Неизвестное событие/место")
            desc_item_parts = [
                f"{plan_items_count}️⃣ {item_name_raw} ({item_type_readable_raw})"
            ]
            if item_data.get("_display_type") == "afisha_event":
                event_obj_disp: Optional[Event] = item_data.get("_original_object")
                if not event_obj_disp and "session_id" in item_data:
                    try:
                        event_obj_disp = Event(**item_data)
                    except ValidationError:
                        event_obj_disp = None
                if event_obj_disp:
                    place_disp_raw = event_obj_disp.place_name or "Место не указано"
                    if (
                        event_obj_disp.place_address
                        and event_obj_disp.place_address.lower()
                        not in (event_obj_disp.place_name or "").lower()
                    ):
                        place_disp_raw += f" ({event_obj_disp.place_address})"
                    desc_item_parts.append(f"  📍 {place_disp_raw}")
                    day_name_en = event_obj_disp.start_time_naive_event_tz.strftime(
                        "%A"
                    )
                    day_name_ru = DAYS_RU.get(day_name_en, day_name_en)
                    time_str_raw = event_obj_disp.start_time_naive_event_tz.strftime(
                        "%H:%M"
                    )
                    date_str_raw = event_obj_disp.start_time_naive_event_tz.strftime(
                        f"%d.%m.%Y ({day_name_ru})"
                    )
                    duration_text = ""
                    if event_obj_disp.duration_minutes:
                        hours = event_obj_disp.duration_minutes // 60
                        minutes = event_obj_disp.duration_minutes % 60
                        duration_text = f", длительность ~{hours} ч {minutes} мин"
                    desc_item_parts.append(
                        f"  🕒 Начало: {date_str_raw} в {time_str_raw}{duration_text}"
                    )
                    price_txt_display = event_obj_disp.price_text
                    if price_txt_display == "Бесплатно":
                        price_txt_display = "Вход свободный"
                    elif event_obj_disp.min_price is not None and price_txt_display in [
                        None,
                        "Цена неизвестна",
                    ]:
                        price_txt_display = f"от {event_obj_disp.min_price} ₽"
                    if price_txt_display and price_txt_display != "Цена неизвестна":
                        desc_item_parts.append(f"  💰 Цена: {price_txt_display}")
                    if event_obj_disp.genres:
                        genres_formatted = format_genres_for_telegram(
                            event_obj_disp.genres
                        )
                        if genres_formatted:
                            desc_item_parts.append(f"  🎭 Жанры: {genres_formatted}")
            elif item_data.get("_display_type") == "park":
                duration_park = int(
                    item_data.get("_calculated_duration_minutes")
                    or item_data.get("duration_minutes")
                    or 90
                )
                address_park_raw = item_data.get("address", "Расположение на карте")
                desc_item_parts.append(f"  📍 {address_park_raw}")
                start_visit_park_dt = item_data.get("_display_start_time")
                if (
                    isinstance(start_visit_park_dt, datetime)
                    and start_visit_park_dt != datetime.max
                ):
                    end_visit_park_dt = start_visit_park_dt + timedelta(
                        minutes=duration_park
                    )
                    hours_park = duration_park // 60
                    minutes_park = duration_park % 60
                    desc_item_parts.append(
                        f"  🕒 Прогулка: {start_visit_park_dt.strftime('%d.%m с %H:%M')} до {end_visit_park_dt.strftime('%H:%M')} (~{hours_park} ч {minutes_park}м)"
                    )
                item_warning_raw = item_data.get("_compatibility_warning")
                if item_warning_raw:
                    desc_item_parts.append(
                        f"  ⚠️ {escape_markdown_v2(item_warning_raw)}"
                    )
            elif item_data.get("_display_type") == "food":
                duration_food = int(
                    item_data.get("_calculated_duration_minutes")
                    or item_data.get("duration_minutes")
                    or 60
                )
                address_food_raw = item_data.get("address", "Адрес не указан")
                desc_item_parts.append(f"  📍 {address_food_raw}")
                food_details_sub_parts = []
                rating_str_raw = item_data.get("rating_str")
                if rating_str_raw and rating_str_raw != "Рейтинг не указан":
                    rating_parts_food = rating_str_raw.split("/")
                    rating_value_food = rating_parts_food[0].strip()
                    reviews_text_food = ""
                    if len(rating_parts_food) > 1 and "(" in rating_parts_food[1]:
                        reviews_match_food = re.search(
                            r"\((\d+)\s*отзыв", rating_parts_food[1]
                        )
                        if reviews_match_food:
                            reviews_text_food = (
                                f" ({reviews_match_food.group(1)} отзывов)"
                            )
                    food_details_sub_parts.append(
                        f"⭐ Рейтинг: {rating_value_food} / 5{reviews_text_food}"
                    )
                avg_bill_raw = item_data.get("avg_bill_str")
                if avg_bill_raw:
                    food_details_sub_parts.append(f"💸 Средний чек: {avg_bill_raw}")
                if food_details_sub_parts:
                    desc_item_parts.append("  " + " ".join(food_details_sub_parts))
                start_visit_food_dt = item_data.get("_display_start_time")
                if (
                    isinstance(start_visit_food_dt, datetime)
                    and start_visit_food_dt != datetime.max
                ):
                    end_visit_food_dt = start_visit_food_dt + timedelta(
                        minutes=duration_food
                    )
                    hours_food = duration_food // 60
                    minutes_food = duration_food % 60
                    desc_item_parts.append(
                        f"  🕒 Время: {start_visit_food_dt.strftime('%d.%m с %H:%M')} до {end_visit_food_dt.strftime('%H:%M')} (~{hours_food} ч {minutes_food}м)"
                    )
                item_warning_raw = item_data.get("_compatibility_warning")
                if item_warning_raw:
                    desc_item_parts.append(
                        f"  ⚠️ {escape_markdown_v2(item_warning_raw)}"
                    )
            if desc_item_parts:
                response_parts.append("\n".join(desc_item_parts))

    rejected_fallbacks_list = current_collected_data_dict.get(
        "rejected_fallback_for_interests", []
    )

    if budget_fallback_plan_data:
        logger.info("Handling budget fallback.")
        fallback_message_parts_iter = []

        if not plan_items_to_display:
            response_parts.append("Вот вариант, который удалось подобрать:")

        items_in_fallback = budget_fallback_plan_data.get("plan_items", [])
        for item_data in items_in_fallback:
            plan_items_count += 1
            item_type_key = item_data.get("user_event_type_key", "EVENT")
            item_type_readable = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
                item_type_key.upper(), item_type_key
            )
            item_name = item_data.get("name", "Неизвестное событие")
            desc_item_parts = [f"{plan_items_count}️⃣ {item_name} ({item_type_readable})"]
            try:
                event_obj = Event(**item_data)
                place_disp = event_obj.place_name or "Место не указано"
                if (
                    event_obj.place_address
                    and event_obj.place_address.lower()
                    not in (event_obj.place_name or "").lower()
                ):
                    place_disp += f" ({event_obj.place_address})"
                desc_item_parts.append(f"  📍 {place_disp}")
                day_name = DAYS_RU.get(
                    event_obj.start_time_naive_event_tz.strftime("%A"), ""
                )
                time_str = event_obj.start_time_naive_event_tz.strftime("%H:%M")
                date_str = event_obj.start_time_naive_event_tz.strftime(
                    f"%d.%m.%Y ({day_name})"
                )
                duration_text = ""
                if event_obj.duration_minutes:
                    hours, minutes = divmod(event_obj.duration_minutes, 60)
                    duration_text = f", длительность ~{hours} ч {minutes} мин"
                desc_item_parts.append(
                    f"  🕒 Начало: {date_str} в {time_str}{duration_text}"
                )
                price_txt = event_obj.price_text or (
                    f"от {event_obj.min_price} ₽"
                    if event_obj.min_price
                    else "Цена неизвестна"
                )
                if price_txt and price_txt != "Цена неизвестна":
                    desc_item_parts.append(f"  💰 Цена: {price_txt}")
                if event_obj.genres:
                    genres_formatted = format_genres_for_telegram(event_obj.genres)
                    if genres_formatted:
                        desc_item_parts.append(f"  🎭 Жанры: {genres_formatted}")
            except Exception as e:
                logger.warning(f"Could not format budget fallback item details: {e}")

            response_parts.append("\n".join(desc_item_parts))

        total_cost = budget_fallback_plan_data.get("cost", 0)
        fallback_message_parts_iter.append(
            f"\nК сожалению, не удалось составить полный план в рамках вашего бюджета в {int(original_budget)} ₽."
        )
        fallback_message_parts_iter.append(
            f"Общая стоимость этого варианта на {person_count} чел. составит ~{int(total_cost)} ₽."
        )
        fallback_message_parts_iter.append(
            "\nХотите рассмотреть этот вариант, несмотря на превышение бюджета? (да/нет)"
        )

        response_parts.extend(fallback_message_parts_iter)
        ask_about_fallback = True
        field_to_be_clarified_next = "budget_fallback_confirmation"
        # FIX: Устанавливаем pending_fallback_event, чтобы передать данные следующему узлу.
        # Это ключевое исправление.
        pending_fallback_event_for_state_dict = budget_fallback_plan_data

    elif date_fallback_candidates_map:
        logger.info("Handling date fallback.")
        ordered_activities_for_fb_check_raw = current_collected_data_dict.get(
            "ordered_activities", []
        )
        requested_afisha_types_in_order_from_query: List[str] = []
        if ordered_activities_for_fb_check_raw:
            for oa_raw_fb in ordered_activities_for_fb_check_raw:
                try:
                    oa_obj_fb = OrderedActivityItem(**oa_raw_fb)
                    if (
                        oa_obj_fb.activity_type
                        and oa_obj_fb.activity_type != "ANY"
                        and oa_obj_fb.activity_type
                        not in requested_afisha_types_in_order_from_query
                    ):
                        requested_afisha_types_in_order_from_query.append(
                            oa_obj_fb.activity_type
                        )
                except ValidationError:
                    continue

        current_plan_displayed_activity_keys = {
            item.get("user_event_type_key")
            for item in plan_items_to_display
            if item.get("user_event_type_key")
        }

        key_for_current_fallback_proposal = next(
            (
                key
                for key in requested_afisha_types_in_order_from_query
                if key not in current_plan_displayed_activity_keys
                and key in date_fallback_candidates_map
                and key not in rejected_fallbacks_list
            ),
            None,
        )

        if key_for_current_fallback_proposal:
            fb_candidate_data_dict = date_fallback_candidates_map[
                key_for_current_fallback_proposal
            ]
            try:
                readable_type_for_fb = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
                    key_for_current_fallback_proposal.upper(),
                    f"'{key_for_current_fallback_proposal}'",
                )
                time_desc_for_fb = current_collected_data_dict.get(
                    "user_time_desc_for_fallback", "запрошенное вами время"
                )
                fallback_message_parts_iter = [
                    f"\nКстати, для типа «{escape_markdown_v2(readable_type_for_fb)}» на {escape_markdown_v2(time_desc_for_fb)} не удалось подобрать вариант в основной план."
                ]
                fb_event_obj_propose = Event(**fb_candidate_data_dict)
                fb_day_en_val = fb_event_obj_propose.start_time_naive_event_tz.strftime(
                    "%A"
                )
                fb_day_ru_val = DAYS_RU.get(fb_day_en_val, fb_day_en_val)
                fb_date_str_val = (
                    fb_event_obj_propose.start_time_naive_event_tz.strftime(
                        f"%d.%m.%Y ({fb_day_ru_val}) в %H:%M"
                    )
                )
                fallback_message_parts_iter.append(
                    f"Но есть другой вариант: {escape_markdown_v2(fb_event_obj_propose.name)} на {fb_date_str_val}."
                )
                if fb_event_obj_propose.place_name:
                    fb_place_text_val = fb_event_obj_propose.place_name
                    if (
                        fb_event_obj_propose.place_address
                        and fb_event_obj_propose.place_address.lower()
                        not in fb_event_obj_propose.place_name.lower()
                    ):
                        fb_place_text_val += f" ({fb_event_obj_propose.place_address})"
                    fallback_message_parts_iter.append(f"  📍 {fb_place_text_val}.")
                fb_price_text_render = fb_event_obj_propose.price_text
                if fb_price_text_render == "Бесплатно":
                    fb_price_text_render = "Вход свободный"
                elif (
                    fb_event_obj_propose.min_price is not None
                    and fb_price_text_render in [None, "Цена неизвестна"]
                ):
                    fb_price_text_render = f"от {fb_event_obj_propose.min_price} ₽"
                if fb_price_text_render and fb_price_text_render != "Цена неизвестна":
                    fallback_message_parts_iter.append(
                        f"  💰 Цена: {fb_price_text_render}."
                    )
                pending_fallback_event_for_state_dict = fb_event_obj_propose.model_dump(
                    exclude_none=True
                )
                fallback_message_parts_iter.append("\nХотите рассмотреть его? (да/нет)")
                response_parts.extend(fallback_message_parts_iter)
                ask_about_fallback = True
                field_to_be_clarified_next = "fallback_confirmation_response"
                current_collected_data_dict["last_offered_fallback_for_interest"] = (
                    key_for_current_fallback_proposal
                )
            except ValidationError as e:
                logger.error(
                    f"Error validating date fallback for presentation: {e}. Data: {fb_candidate_data_dict}"
                )

    elif combo_fallback_candidates_map:
        logger.info("Handling combo (date+budget) fallback.")
        ordered_activities_for_fb_check_raw = current_collected_data_dict.get(
            "ordered_activities", []
        )
        requested_afisha_types_in_order_from_query: List[str] = []
        if ordered_activities_for_fb_check_raw:
            for oa_raw_fb in ordered_activities_for_fb_check_raw:
                try:
                    oa_obj_fb = OrderedActivityItem(**oa_raw_fb)
                    if (
                        oa_obj_fb.activity_type
                        and oa_obj_fb.activity_type
                        not in requested_afisha_types_in_order_from_query
                    ):
                        requested_afisha_types_in_order_from_query.append(
                            oa_obj_fb.activity_type
                        )
                except ValidationError:
                    continue

        current_plan_displayed_activity_keys = {
            item.get("user_event_type_key")
            for item in plan_items_to_display
            if item.get("user_event_type_key")
        }
        key_for_current_fallback_proposal = next(
            (
                key
                for key in requested_afisha_types_in_order_from_query
                if key not in current_plan_displayed_activity_keys
                and key in combo_fallback_candidates_map
                and key not in rejected_fallbacks_list
            ),
            None,
        )

        if key_for_current_fallback_proposal:
            combo_fb_data = combo_fallback_candidates_map[
                key_for_current_fallback_proposal
            ]
            fb_event_dict = combo_fb_data.get("event")
            total_plan_cost = combo_fb_data.get("total_plan_cost")

            try:
                fb_event_obj_propose = Event(**fb_event_dict)
                readable_type_for_fb = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
                    key_for_current_fallback_proposal.upper(),
                    f"'{key_for_current_fallback_proposal}'",
                )

                fallback_message_parts_iter = [
                    f"\nК сожалению, на запрошенную дату не нашлось подходящих {escape_markdown_v2(readable_type_for_fb.lower())}."
                ]
                fb_day_en_val = fb_event_obj_propose.start_time_naive_event_tz.strftime(
                    "%A"
                )
                fb_day_ru_val = DAYS_RU.get(fb_day_en_val, fb_day_en_val)
                fb_date_str_val = (
                    fb_event_obj_propose.start_time_naive_event_tz.strftime(
                        f"%d.%m.%Y ({fb_day_ru_val})"
                    )
                )

                fallback_message_parts_iter.append(
                    f"Ближайший вариант — «{escape_markdown_v2(fb_event_obj_propose.name)}» на {fb_date_str_val}, но с ним общая стоимость плана будет ~{int(total_plan_cost)} ₽, что превышает ваш бюджет в {int(original_budget)} ₽."
                )
                fallback_message_parts_iter.append(
                    "\nХотите добавить этот вариант, несмотря на другую дату и превышение бюджета? (да/нет)"
                )
                response_parts.extend(fallback_message_parts_iter)

                ask_about_fallback = True
                field_to_be_clarified_next = "combo_fallback_confirmation"
                pending_fallback_event_for_state_dict = fb_event_obj_propose.model_dump(
                    exclude_none=True
                )
                current_collected_data_dict["last_offered_fallback_for_interest"] = (
                    key_for_current_fallback_proposal
                )

            except ValidationError as e:
                logger.error(
                    f"Error validating combo fallback for presentation: {e}. Data: {fb_event_dict}"
                )

    elif not ask_about_fallback:
        plan_construction_failed_step_msg_raw = current_collected_data_dict.get(
            "plan_construction_failed_step"
        )
        if plan_construction_failed_step_msg_raw:
            response_parts.append(
                f"\nК сожалению, не удалось составить запрошенный план: {escape_markdown_v2(plan_construction_failed_step_msg_raw)}."
            )

        if poi_warnings_in_plan:
            first_warning_raw = poi_warnings_in_plan[0]
            poi_name_in_warning_raw = ""
            match_poi_name = re.search(r"на '([^']+)'", first_warning_raw)
            if match_poi_name:
                poi_name_in_warning_raw = (
                    f" «{escape_markdown_v2(match_poi_name.group(1))}»"
                )
            response_parts.append(
                f"\nКак видите, время на посещение{poi_name_in_warning_raw} получается довольно коротким. Вас это устраивает, или попробовать найти вариант получше? (да, устраивает / попробовать найти лучше)"
            )

        user_has_address_coords_check = bool(
            current_collected_data_dict.get("user_start_address_validated_coords")
        )
        address_status_check = current_collected_data_dict.get(
            "address_clarification_status"
        )
        if (
            not field_to_be_clarified_next
            and not user_has_address_coords_check
            and address_status_check != "SKIPPED_BY_USER"
            and address_status_check != "SKIPPED_AND_PROCESSED"
        ):
            if plan_items_to_display:
                response_parts.append(
                    "\n📍 Откуда вы планируете начать ваш маршрут? Пожалуйста, укажите улицу и номер дома, или напишите 'пропустить'."
                )
                response_parts.append(
                    "\nℹ️ После того как вы укажете адрес и я подготовлю полный план с маршрутом, вы сможете его скорректировать, если это будет необходимо."
                )
                field_to_be_clarified_next = "user_start_address_original"
                current_collected_data_dict["awaiting_address_input"] = True

    if not field_to_be_clarified_next and not ask_about_fallback:
        if not plan_items_count and not plan_construction_failed_step_msg_raw:
            response_parts = [
                "По вашему запросу ничего не найдено. Попробуем другие критерии?"
            ]
        elif plan_items_count > 0:
            response_parts.append(
                "\nКак вам такой предварительный план? Если всё нравится или хотите что-то изменить, дайте знать!"
            )

    if ask_about_fallback:
        current_collected_data_dict["awaiting_fallback_confirmation"] = True
        current_collected_data_dict["pending_fallback_event"] = (
            pending_fallback_event_for_state_dict
        )
        current_collected_data_dict["awaiting_address_input"] = False

    is_plan_actually_proposed = (
        bool(plan_items_to_display)
        or ask_about_fallback
        or bool(plan_construction_failed_step_msg_raw)
        or ask_for_poi_optimization
        or bool(budget_fallback_plan_data)
    )
    final_response_output_text = "\n".join(
        part.strip("\n") for part in response_parts if part.strip()
    ).strip()
    if not final_response_output_text and is_plan_actually_proposed:
        final_response_output_text = "Пожалуйста, ознакомьтесь с предложенным планом."
    elif not final_response_output_text:
        final_response_output_text = "Не удалось сформировать предложение. Пожалуйста, попробуйте уточнить ваш запрос."

    return_state_update["messages"].append(
        AIMessage(content=final_response_output_text)
    )
    return_state_update["status_message_to_user"] = final_response_output_text

    final_awaiting_clarification_field_val = field_to_be_clarified_next
    if ask_about_fallback:
        if field_to_be_clarified_next == "budget_fallback_confirmation":
            final_awaiting_clarification_field_val = "budget_fallback_confirmation"
        elif field_to_be_clarified_next == "combo_fallback_confirmation":
            final_awaiting_clarification_field_val = "combo_fallback_confirmation"
        else:
            final_awaiting_clarification_field_val = "fallback_confirmation_response"

    return_state_update["awaiting_clarification_for_field"] = (
        final_awaiting_clarification_field_val
    )
    if final_awaiting_clarification_field_val == "user_start_address_original":
        current_collected_data_dict["awaiting_address_input"] = True

    return_state_update["collected_data"] = current_collected_data_dict
    return_state_update["is_initial_plan_proposed"] = is_plan_actually_proposed
    return_state_update["awaiting_final_confirmation"] = False
    return_state_update["awaiting_feedback_on_final_plan"] = False
    return_state_update["last_presented_plan"] = None
    return_state_update["just_modified_plan"] = False
    return_state_update["clarification_context"] = None

    # DIAGNOSTIC LOG
    logger.info(f"--- DIAGNOSTIC: STATE BEFORE EXITING present_initial_plan_node ---")
    logger.info(
        f"    awaiting_clarification_for_field: {final_awaiting_clarification_field_val}"
    )
    logger.info(
        f"    pending_fallback_event is None: {current_collected_data_dict.get('pending_fallback_event') is None}"
    )
    if current_collected_data_dict.get("pending_fallback_event"):
        logger.info(
            f"    pending_fallback_event content: {current_collected_data_dict.get('pending_fallback_event')}"
        )
    logger.info(
        f"----------------------------------------------------------------------"
    )

    return return_state_update


async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_address_or_build_route_node executing...")
    collected_data: dict = dict(state.get("collected_data", {}))
    current_afisha_events_objects: List[Event] = state.get("current_events", [])
    selected_pois_for_plan_dicts: List[Dict[str, Any]] = collected_data.get(
        "selected_pois_for_plan", []
    )

    all_plan_items_for_route: List[Dict[str, Any]] = []

    for event_obj in current_afisha_events_objects:
        event_start_time = event_obj.start_time_naive_event_tz
        if isinstance(event_start_time, str):  # Should be datetime from Event model
            try:
                event_start_time = datetime.fromisoformat(event_start_time)
            except ValueError:
                event_start_time = datetime.max
        elif not isinstance(event_start_time, datetime):
            event_start_time = datetime.max

        all_plan_items_for_route.append(
            {
                "address_string": event_obj.place_address,
                "lon": event_obj.place_coords_lon,
                "lat": event_obj.place_coords_lat,
                "_name_for_log": event_obj.name,
                "_effective_start_time": event_start_time,
            }
        )

    for poi_data in selected_pois_for_plan_dicts:
        effective_start_time_poi = poi_data.get("_calculated_start_time")
        if isinstance(effective_start_time_poi, str):
            try:
                effective_start_time_poi = datetime.fromisoformat(
                    effective_start_time_poi
                )
            except ValueError:
                effective_start_time_poi = datetime.max
        elif not isinstance(effective_start_time_poi, datetime):
            effective_start_time_poi = datetime.max

        poi_coords = poi_data.get("coords")
        lon_val = None
        lat_val = None
        if isinstance(poi_coords, list) and len(poi_coords) == 2:
            lon_val = poi_coords[0]
            lat_val = poi_coords[1]

        all_plan_items_for_route.append(
            {
                "address_string": poi_data.get("address"),
                "lon": lon_val,
                "lat": lat_val,
                "_name_for_log": poi_data.get("name"),
                "_effective_start_time": effective_start_time_poi,
            }
        )

    all_plan_items_for_route.sort(
        key=lambda x: x.get("_effective_start_time", datetime.max)
    )

    logger.debug(
        f"Points for routing, sorted by effective start time: {[p.get('_name_for_log') for p in all_plan_items_for_route]}"
    )

    return_state_update: Dict[str, Any] = {
        "current_route_details": RouteDetails(
            status="error", error_message="Неизвестная ошибка подготовки маршрута."
        ).model_dump(),
        "is_full_plan_with_route_proposed": False,
        "collected_data": collected_data,
        "current_events": current_afisha_events_objects,
        **{
            k: v
            for k, v in state.items()
            if k
            not in [
                "current_route_details",
                "is_full_plan_with_route_proposed",
                "collected_data",
                "current_events",
            ]
        },
    }

    if not all_plan_items_for_route:
        logger.warning(
            "clarify_address_or_build_route_node: No points in the plan to build a route for."
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="error", error_message="Нет точек для построения маршрута."
        ).model_dump()
        return return_state_update

    user_start_address_str = collected_data.get("user_start_address_original")
    user_start_coords_dict = collected_data.get("user_start_address_validated_coords")
    city_name_for_gis_context = collected_data.get("city_name")
    if not city_name_for_gis_context:
        logger.warning(
            "clarify_address_or_build_route_node: City name is missing in collected_data. Geocoding for route points might be inaccurate."
        )

    start_point_for_api_dict: Optional[Dict[str, Any]] = None
    event_points_for_api_list_dicts: List[Dict[str, Any]] = []
    routable_points_dicts: List[Dict[str, Any]] = []

    for item_data in all_plan_items_for_route:
        routable_points_dicts.append(
            {
                "address_string": item_data.get("address_string"),
                "lon": item_data.get("lon"),
                "lat": item_data.get("lat"),
            }
        )

    if user_start_coords_dict:
        start_point_for_api_dict = {
            "address_string": user_start_address_str,
            "lon": user_start_coords_dict.get("lon"),
            "lat": user_start_coords_dict.get("lat"),
        }
        event_points_for_api_list_dicts = routable_points_dicts
    elif len(routable_points_dicts) > 1:
        logger.info(
            "User address not provided, building route between plan points themselves."
        )
        start_point_for_api_dict = routable_points_dicts[0]
        event_points_for_api_list_dicts = routable_points_dicts[1:]
    elif len(routable_points_dicts) == 1:
        logger.info(
            "Only one point in plan and no user address, no route segments to build."
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="success",
            segments=[],
            total_duration_seconds=0,
            total_distance_meters=0,
        ).model_dump()
        return_state_update["is_full_plan_with_route_proposed"] = True
        return return_state_update
    else:
        logger.error(
            "clarify_address_or_build_route_node: Inconsistent state - no start point for API could be determined."
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="error", error_message="Ошибка определения точек маршрута."
        ).model_dump()
        return return_state_update

    if not start_point_for_api_dict:
        logger.error(
            "clarify_address_or_build_route_node: Start point for API is None after logic."
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="error",
            error_message="Не удалось определить начальную точку маршрута.",
        ).model_dump()
        return return_state_update

    if (
        not event_points_for_api_list_dicts
        and user_start_coords_dict
        and len(all_plan_items_for_route) == 1
    ):
        logger.info(
            "Route from user start address to a single plan point. Tool will build one segment."
        )
        pass
    elif not event_points_for_api_list_dicts:
        logger.info(
            "No subsequent event points to route to from the determined start point. No route segments will be built."
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="success",
            segments=[],
            total_duration_seconds=0,
            total_distance_meters=0,
        ).model_dump()
        return_state_update["is_full_plan_with_route_proposed"] = True
        return return_state_update

    tool_args_route = RouteBuilderToolArgs(
        start_point=start_point_for_api_dict,
        event_points=event_points_for_api_list_dicts,
        transport_type="driving",
        city_context_for_geocoding=city_name_for_gis_context,
    )

    logger.info(
        f"Calling route_builder_tool. Start: {str(start_point_for_api_dict)[:100]}, Event Points Count: {len(event_points_for_api_list_dicts)}. City Context: {city_name_for_gis_context}"
    )
    logger.debug(
        f"RouteBuilderToolArgs for invocation: {tool_args_route.model_dump_json(exclude_none=True, indent=2)}"
    )

    route_data_dict: Dict[str, Any] = await route_builder_tool.ainvoke(
        tool_args_route.model_dump(exclude_none=True)
    )

    try:
        final_route_details_obj = RouteDetails(**route_data_dict)
    except ValidationError as e_route_val:
        logger.error(
            f"clarify_address_or_build_route_node: ValidationError when creating RouteDetails from tool output: {e_route_val}. Data: {route_data_dict}"
        )
        return_state_update["current_route_details"] = RouteDetails(
            status="error",
            error_message=f"Ошибка валидации данных маршрута: {e_route_val}",
        ).model_dump()
        return_state_update["is_full_plan_with_route_proposed"] = False
        return return_state_update

    return_state_update["current_route_details"] = final_route_details_obj.model_dump(
        exclude_none=True
    )
    return_state_update["is_full_plan_with_route_proposed"] = (
        final_route_details_obj.status in ["success", "partial_success"]
        and bool(final_route_details_obj.segments)
    )

    # Ensure current_events and selected_pois_for_plan are not accidentally cleared if route fails but plan existed
    return_state_update["current_events"] = current_afisha_events_objects
    return_state_update["collected_data"][
        "selected_pois_for_plan"
    ] = selected_pois_for_plan_dicts

    return return_state_update


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
    current_afisha_events_objects: List[Event] = state.get("current_events", [])
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    selected_pois_for_plan_dicts: List[Dict[str, Any]] = collected_data_dict.get(
        "selected_pois_for_plan", []
    )

    current_route_details_obj: Optional[RouteDetails] = None
    current_route_details_dump = state.get("current_route_details")
    if isinstance(current_route_details_dump, dict):
        try:
            current_route_details_obj = RouteDetails(**current_route_details_dump)
        except ValidationError as e_val_route_present_full_node:
            logger.error(
                f"present_full_plan_node: Error validating current_route_details: {e_val_route_present_full_node}"
            )
    elif isinstance(current_route_details_dump, RouteDetails):
        current_route_details_obj = current_route_details_dump

    all_plan_items_to_display: List[Dict[str, Any]] = []
    last_presented_events_dumps: List[Dict[str, Any]] = []

    if current_afisha_events_objects:
        for evt_obj_item in current_afisha_events_objects:
            evt_dict_item = evt_obj_item.model_dump(exclude_none=True)
            last_presented_events_dumps.append(dict(evt_dict_item))
            evt_dict_item["_display_start_time"] = (
                evt_obj_item.start_time_naive_event_tz
            )
            evt_dict_item["_display_type"] = "afisha_event"
            evt_dict_item["_original_object"] = evt_obj_item
            all_plan_items_to_display.append(evt_dict_item)

    last_presented_selected_pois_dumps: List[Dict[str, Any]] = [
        dict(poi) for poi in selected_pois_for_plan_dicts
    ]

    for poi_dict_item_val in selected_pois_for_plan_dicts:
        poi_dict_copy_val = dict(poi_dict_item_val)
        start_time_poi_val_node = poi_dict_item_val.get("_calculated_start_time")
        effective_start_time_poi_node: Optional[datetime] = None
        if isinstance(start_time_poi_val_node, str):
            try:
                effective_start_time_poi_node = datetime.fromisoformat(
                    start_time_poi_val_node
                )
            except ValueError:
                effective_start_time_poi_node = datetime.max
        elif isinstance(start_time_poi_val_node, datetime):
            effective_start_time_poi_node = start_time_poi_val_node
        else:
            effective_start_time_poi_node = datetime.max
        poi_dict_copy_val["_display_start_time"] = effective_start_time_poi_node
        poi_dict_copy_val["_display_type"] = poi_dict_item_val.get(
            "_plan_item_type", "poi"
        )
        all_plan_items_to_display.append(poi_dict_copy_val)

    all_plan_items_to_display.sort(
        key=lambda x_sort_node: x_sort_node.get("_display_start_time", datetime.max)
    )

    response_parts = []
    if not all_plan_items_to_display:
        logger.warning("present_full_plan_node: No items in plan to present.")
        preserved_context_no_plan = {
            key: collected_data_dict.get(key)
            for key in [
                "city_name",
                "city_id_afisha",
                "user_start_address_original",
                "user_start_address_validated_coords",
                "current_excluded_ids",
                "ordered_activities",
            ]
            if collected_data_dict.get(key) is not None
        }
        preserved_context_no_plan.setdefault("current_excluded_ids", {})
        return {
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content="Кажется, в вашем плане нет мероприятий. Пожалуйста, начните новый поиск."
                )
            ],
            "status_message_to_user": "Кажется, в вашем плане нет мероприятий. Пожалуйста, начните новый поиск.",
            "collected_data": preserved_context_no_plan,
            "current_events": [],
            "current_route_details": None,
            "is_initial_plan_proposed": False,
            "is_full_plan_with_route_proposed": False,
            "awaiting_final_confirmation": False,
            "awaiting_feedback_on_final_plan": False,
            "last_presented_plan": None,
            "modification_request_details": None,
            "plan_modification_pending": False,
            "just_modified_plan": False,
        }

    response_parts.append("Вот ваш итоговый план:")
    plan_items_count = 0

    for item_data_display_node in all_plan_items_to_display:
        plan_items_count += 1
        item_name_display_node_raw = item_data_display_node.get(
            "name"
        ) or item_data_display_node.get("_name", "Неизвестное событие/место")
        item_name_display_node = item_name_display_node_raw

        item_display_type_val_node = item_data_display_node.get("_display_type")
        desc_item_parts = []

        item_type_key_to_check = (
            item_data_display_node.get("user_event_type_key")
            or item_display_type_val_node
        )
        if not isinstance(item_type_key_to_check, str):
            item_type_key_to_check = "EVENT"

        item_type_readable_raw = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
            item_type_key_to_check.upper(), item_type_key_to_check
        )

        desc_item_parts.append(
            f"{plan_items_count}️⃣ {item_name_display_node} ({item_type_readable_raw})"
        )

        if item_display_type_val_node == "afisha_event":
            event_obj_display_node: Optional[Event] = item_data_display_node.get(
                "_original_object"
            )
            if not event_obj_display_node and "session_id" in item_data_display_node:
                try:
                    event_obj_display_node = Event(**item_data_display_node)
                except ValidationError:
                    event_obj_display_node = None

            if event_obj_display_node:
                place_display_node_raw = (
                    event_obj_display_node.place_name or "Место не указано"
                )
                if (
                    event_obj_display_node.place_address
                    and event_obj_display_node.place_address.lower()
                    not in (event_obj_display_node.place_name or "").lower()
                ):
                    place_display_node_raw += (
                        f" ({event_obj_display_node.place_address})"
                    )
                desc_item_parts.append(f"📍 {place_display_node_raw}")

                day_name_en = event_obj_display_node.start_time_naive_event_tz.strftime(
                    "%A"
                )
                day_name_ru = DAYS_RU.get(day_name_en, day_name_en)
                time_str_node = (
                    event_obj_display_node.start_time_naive_event_tz.strftime("%H:%M")
                )
                date_str_node = (
                    event_obj_display_node.start_time_naive_event_tz.strftime(
                        f"%d.%m.%Y ({day_name_ru})"
                    )
                )

                duration_text = ""
                if event_obj_display_node.duration_minutes:
                    hours = event_obj_display_node.duration_minutes // 60
                    minutes = event_obj_display_node.duration_minutes % 60
                    duration_text = f", длительность ~{hours} ч {minutes} мин"
                desc_item_parts.append(
                    f"🕒 Начало: {date_str_node} в {time_str_node}{duration_text}"
                )

                price_text_raw = event_obj_display_node.price_text
                if price_text_raw == "Бесплатно":
                    price_text_raw = "Вход свободный"
                elif (
                    event_obj_display_node.min_price is not None
                    and price_text_raw in [None, "Цена неизвестна"]
                ):
                    price_text_raw = f"от {event_obj_display_node.min_price} ₽"

                if price_text_raw and price_text_raw != "Цена неизвестна":
                    desc_item_parts.append(f"💰 Цена: {price_text_raw}")

                if event_obj_display_node.genres:
                    genres_formatted = format_genres_for_telegram(
                        event_obj_display_node.genres
                    )
                    if genres_formatted:
                        desc_item_parts.append(f"🎭 Жанры: {genres_formatted}")

        elif item_display_type_val_node == "park":
            park_duration_minutes = int(
                item_data_display_node.get("_calculated_duration_minutes")
                or item_data_display_node.get("duration_minutes")
                or 90
            )
            park_address_node_raw = item_data_display_node.get(
                "address", item_name_display_node_raw
            )
            desc_item_parts.append(f"📍 {park_address_node_raw}")

            park_schedule_str_raw = item_data_display_node.get("schedule_str")
            if (
                park_schedule_str_raw
                and park_schedule_str_raw.strip()
                and park_schedule_str_raw != "Время работы не указано"
            ):
                desc_item_parts.append(
                    f"🕒 График работы парка: {park_schedule_str_raw}"
                )

            start_visit_dt_park_node = item_data_display_node.get("_display_start_time")
            if isinstance(start_visit_dt_park_node, datetime):
                end_visit_dt_park_node = start_visit_dt_park_node + timedelta(
                    minutes=park_duration_minutes
                )
                hours_park = park_duration_minutes // 60
                minutes_park = park_duration_minutes % 60
                desc_item_parts.append(
                    f"🕒 Запланированное время: {start_visit_dt_park_node.strftime('%d.%m с %H:%M')} до {end_visit_dt_park_node.strftime('%H:%M')} (~{hours_park} ч {minutes_park} мин)"
                )

        elif item_display_type_val_node == "food":
            food_duration_minutes = int(
                item_data_display_node.get("_calculated_duration_minutes")
                or item_data_display_node.get("duration_minutes")
                or 60
            )
            food_address_node_raw = item_data_display_node.get(
                "address", item_name_display_node_raw
            )
            desc_item_parts.append(f"📍 {food_address_node_raw}")

            food_schedule_str_raw = item_data_display_node.get("schedule_str")
            if (
                food_schedule_str_raw
                and food_schedule_str_raw.strip()
                and food_schedule_str_raw != "Время работы не указано"
            ):
                schedule_lines = [
                    f"  {line.strip()}"
                    for line in food_schedule_str_raw.split("\n")
                    if line.strip()
                ]
                if schedule_lines:
                    desc_item_parts.append("🕒 График работы:")
                    desc_item_parts.extend(schedule_lines)

            start_visit_dt_food_node = item_data_display_node.get("_display_start_time")
            if isinstance(start_visit_dt_food_node, datetime):
                end_visit_dt_food_node = start_visit_dt_food_node + timedelta(
                    minutes=food_duration_minutes
                )
                hours_food = food_duration_minutes // 60
                minutes_food = food_duration_minutes % 60
                desc_item_parts.append(
                    f"🕒 Запланированное время: {start_visit_dt_food_node.strftime('%d.%m с %H:%M')} до {end_visit_dt_food_node.strftime('%H:%M')} (~{hours_food} ч {minutes_food} мин)"
                )

            rating_str_raw = item_data_display_node.get("rating_str")
            if rating_str_raw and rating_str_raw != "Рейтинг не указан":
                rating_parts = rating_str_raw.split("/")
                rating_value = rating_parts[0].strip()
                reviews_text = ""
                if len(rating_parts) > 1 and "(" in rating_parts[1]:
                    reviews_match = re.search(r"\((\d+)\s*отзыв", rating_parts[1])
                    if reviews_match:
                        reviews_text = f" ({reviews_match.group(1)} отзывов)"
                desc_item_parts.append(f"⭐ Рейтинг: {rating_value} / 5{reviews_text}")

            avg_bill_raw = item_data_display_node.get("avg_bill_str")
            if avg_bill_raw:
                desc_item_parts.append(f"💸 Средний чек: {avg_bill_raw}")

        if desc_item_parts:
            response_parts.append("\n".join(desc_item_parts))

    if current_route_details_obj:
        if (
            current_route_details_obj.status in ["success", "partial_success"]
            and current_route_details_obj.segments
        ):
            response_parts.append("\n➡️ Маршрут:")

            actual_route_start_point_name_full_node = "Ваш начальный адрес"
            user_address_data_full_node = collected_data_dict.get(
                "user_start_address_original"
            )
            if user_address_data_full_node:
                actual_route_start_point_name_full_node = user_address_data_full_node
            elif all_plan_items_to_display:
                actual_route_start_point_name_full_node = all_plan_items_to_display[
                    0
                ].get("name") or all_plan_items_to_display[0].get(
                    "_name", "Первый пункт плана"
                )

            for idx_seg, segment_obj_full_node in enumerate(
                current_route_details_obj.segments
            ):
                from_name_route_val_full_node_raw = segment_obj_full_node.from_address
                to_name_route_val_full_node_raw = segment_obj_full_node.to_address

                if (
                    not from_name_route_val_full_node_raw
                    or "коорд." in (from_name_route_val_full_node_raw or "").lower()
                    or "неизвестн" in (from_name_route_val_full_node_raw or "").lower()
                ):
                    if idx_seg == 0:
                        from_name_route_val_full_node_raw = (
                            actual_route_start_point_name_full_node
                        )
                    elif (
                        user_address_data_full_node
                        and idx_seg > 0
                        and (idx_seg - 1) < len(all_plan_items_to_display)
                    ):
                        from_name_route_val_full_node_raw = (
                            all_plan_items_to_display[idx_seg - 1].get("name")
                            or all_plan_items_to_display[idx_seg - 1].get("_name")
                            or from_name_route_val_full_node_raw
                        )
                    elif not user_address_data_full_node and idx_seg < len(
                        all_plan_items_to_display
                    ):
                        from_name_route_val_full_node_raw = (
                            all_plan_items_to_display[idx_seg].get("name")
                            or all_plan_items_to_display[idx_seg].get("_name")
                            or from_name_route_val_full_node_raw
                        )

                if (
                    not to_name_route_val_full_node_raw
                    or "коорд." in (to_name_route_val_full_node_raw or "").lower()
                    or "неизвестн" in (to_name_route_val_full_node_raw or "").lower()
                ):
                    target_plan_item_index = (
                        idx_seg if user_address_data_full_node else idx_seg + 1
                    )
                    if target_plan_item_index < len(all_plan_items_to_display):
                        to_name_route_val_full_node_raw = (
                            all_plan_items_to_display[target_plan_item_index].get(
                                "name"
                            )
                            or all_plan_items_to_display[target_plan_item_index].get(
                                "_name"
                            )
                            or to_name_route_val_full_node_raw
                        )

                from_name_clean = (
                    from_name_route_val_full_node_raw or "предыдущего пункта"
                ).replace("\\", "")
                to_name_clean = (
                    to_name_route_val_full_node_raw or "следующего пункта"
                ).replace("\\", "")

                segment_text_full_node = (
                    f"\nОт «{from_name_clean}» до «{to_name_clean}»:"
                )
                if segment_obj_full_node.segment_status == "success":
                    duration_clean = (
                        segment_obj_full_node.duration_text or "? мин"
                    ).replace("\\", "")
                    distance_clean = (
                        segment_obj_full_node.distance_text or "? км"
                    ).replace("\\", "")
                    segment_text_full_node += f" {duration_clean}, {distance_clean}"
                else:
                    segment_text_full_node += f" не удалось построить ({segment_obj_full_node.segment_error_message or 'причина неизвестна'})"
                response_parts.append(segment_text_full_node)

            if current_route_details_obj.status == "partial_success":
                response_parts.append(
                    "\nОбратите внимание: не все части маршрута удалось построить."
                )
            elif (
                current_route_details_obj.total_duration_text
                and current_route_details_obj.segments
                and len(current_route_details_obj.segments) >= 1
            ):
                total_duration_clean = (
                    current_route_details_obj.total_duration_text
                ).replace("\\", "")
                response_parts.append(
                    f"\n🚗 Общее время в пути: {total_duration_clean}"
                )

        elif (
            current_route_details_obj.status != "success"
            and current_route_details_obj.error_message
        ):
            response_parts.append(
                f"\n⚠️ Маршрут: Не удалось построить. {current_route_details_obj.error_message}"
            )
        elif (
            not current_route_details_obj.segments
            and len(all_plan_items_to_display) > 1
        ):
            response_parts.append(
                "\n⚠️ Маршрут: Между выбранными пунктами маршрут не потребовался или не был построен."
            )

    response_parts.append(
        "\n\nПлан окончательный. Если захотите что-то изменить или начать новый поиск — просто напишите! 😊"
    )
    full_plan_text_val_node = "\n".join(response_parts)

    preserved_collected_data_dict = {
        key: collected_data_dict.get(key)
        for key in [
            "city_name",
            "city_id_afisha",
            "parsed_dates_iso",
            "parsed_end_dates_iso",
            "dates_description_original",
            "raw_time_description_original",
            "user_start_address_original",
            "user_start_address_validated_coords",
            "budget_original",
            "budget_current_search",
            "interests_original",
            "interests_keys_afisha",
            "ordered_activities",
            "current_excluded_ids",
        ]
        if collected_data_dict.get(key) is not None
    }
    preserved_collected_data_dict.setdefault("current_excluded_ids", {})
    preserved_collected_data_dict["clarification_needed_fields"] = []
    preserved_collected_data_dict["awaiting_fallback_confirmation"] = False
    preserved_collected_data_dict["pending_fallback_event"] = None
    preserved_collected_data_dict["last_offered_fallback_for_interest"] = None
    preserved_collected_data_dict["fallback_accepted_and_plan_updated"] = False
    preserved_collected_data_dict["not_found_interest_keys_in_primary_search"] = []
    preserved_collected_data_dict["fallback_candidates"] = {}
    preserved_collected_data_dict["plan_construction_failed_step"] = None
    preserved_collected_data_dict["selected_pois_for_plan"] = []
    preserved_collected_data_dict["address_clarification_status"] = None

    last_presented_plan_info: LastPresentedPlanInfo = {
        "events": last_presented_events_dumps,
        "selected_pois": last_presented_selected_pois_dumps,
        "route_details": (
            current_route_details_obj.model_dump(exclude_none=True)
            if current_route_details_obj
            else None
        ),
    }

    thread_id_val = "unknown_thread"
    messages_list = state.get("messages")
    if (
        messages_list
        and isinstance(messages_list[-1], AIMessage)
        and messages_list[-1].additional_kwargs
    ):
        thread_id_val = messages_list[-1].additional_kwargs.get(
            "thread_id", "unknown_thread"
        )
    elif (
        messages_list
        and isinstance(messages_list[-1], HumanMessage)
        and messages_list[-1].additional_kwargs
    ):
        thread_id_val = messages_list[-1].additional_kwargs.get(
            "thread_id", "unknown_thread"
        )
    logger.info(
        f"Session for chat_id ... (thread_id {thread_id_val}) reached present_full_plan. State prepared for potential feedback."
    )

    return {
        "messages": state.get("messages", [])
        + [AIMessage(content=full_plan_text_val_node)],
        "status_message_to_user": full_plan_text_val_node,
        "collected_data": preserved_collected_data_dict,
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": True,
        "awaiting_final_confirmation": False,
        "awaiting_feedback_on_final_plan": True,
        "last_presented_plan": last_presented_plan_info,
        "modification_request_details": None,
        "plan_modification_pending": False,
        "awaiting_clarification_for_field": None,
        "just_modified_plan": False,
    }


# --- Узел 7: Обработка обратной связи по плану ---
async def handle_plan_feedback_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: handle_plan_feedback_node executing...")
    messages: List[BaseMessage] = state.get("messages", [])

    original_collected_data: dict = state.get("collected_data", {})
    updated_collected_data: dict = dict(original_collected_data)

    if "current_excluded_ids" not in updated_collected_data or not isinstance(
        updated_collected_data["current_excluded_ids"], dict
    ):
        updated_collected_data["current_excluded_ids"] = {
            "afisha": [],
            "park": [],
            "food": [],
            "afisha_names_to_avoid": [],
            "afisha_creation_ids_to_avoid": [],
        }
    else:
        for key in [
            "afisha",
            "park",
            "food",
            "afisha_names_to_avoid",
            "afisha_creation_ids_to_avoid",
        ]:
            updated_collected_data["current_excluded_ids"].setdefault(key, [])

    node_return_values: Dict[str, Any] = {}
    for key, value in state.items():
        node_return_values[key] = value
    node_return_values["collected_data"] = updated_collected_data

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("handle_plan_feedback_node: No human feedback message found.")
        node_return_values["status_message_to_user"] = (
            "Не получил вашего ответа по плану. Пожалуйста, уточните."
        )
        node_return_values["awaiting_final_confirmation"] = True
        return node_return_values

    user_feedback = messages[-1].content
    llm = get_gigachat_client()
    structured_llm_feedback_analyzer = llm.with_structured_output(AnalyzedFeedback)
    plan_summary_parts = []
    last_plan_info: Optional[LastPresentedPlanInfo] = state.get("last_presented_plan")

    logger.debug(f"handle_plan_feedback_node: last_presented_plan: {last_plan_info}")

    combined_previous_items: List[Dict[str, Any]] = []

    if last_plan_info:
        if last_plan_info.get("events"):
            for i_event, event_dump in enumerate(last_plan_info.get("events", [])):
                event_name_summary = event_dump.get("name", "Неизвестное событие")
                event_time_summary_str = "время не указано"
                if event_dump.get("start_time_naive_event_tz"):
                    try:
                        event_time_summary_str = datetime.fromisoformat(
                            str(event_dump.get("start_time_naive_event_tz"))
                        ).strftime("%d.%m %H:%M")
                    except Exception:
                        pass
                plan_summary_parts.append(
                    f"Событие Афиши {len(combined_previous_items) + 1}: {escape_markdown_v2(event_name_summary)} ({escape_markdown_v2(event_time_summary_str)})"
                )
                combined_previous_items.append(
                    {
                        "index_in_plan": len(combined_previous_items),
                        **event_dump,
                        "_item_source_type": "event",
                    }
                )

        if last_plan_info.get("selected_pois"):
            for i_poi, poi_data in enumerate(last_plan_info.get("selected_pois", [])):
                poi_type_display = AFISHA_KEY_TO_READABLE_NAME_MAP.get(
                    poi_data.get("_plan_item_type", "").upper(), "Место"
                )
                plan_summary_parts.append(
                    f"{escape_markdown_v2(poi_type_display)} {len(combined_previous_items) + 1}: {escape_markdown_v2(poi_data.get('name', 'Неизвестно'))}"
                )
                combined_previous_items.append(
                    {
                        "index_in_plan": len(combined_previous_items),
                        **poi_data,
                        "_item_source_type": "poi",
                    }
                )

        route_details_summary = last_plan_info.get("route_details")
        if (
            route_details_summary
            and isinstance(route_details_summary, dict)
            and route_details_summary.get("total_duration_text")
        ):
            plan_summary_parts.append(
                f"Маршрут: общее время ~{escape_markdown_v2(route_details_summary['total_duration_text'])}"
            )

    logger.debug(
        f"handle_plan_feedback_node: combined_previous_items: {combined_previous_items}"
    )

    current_plan_summary_str = (
        "\n".join(plan_summary_parts)
        if plan_summary_parts
        else "План еще не был предложен или пуст."
    )
    prompt_for_feedback_analysis = PLAN_FEEDBACK_ANALYSIS_PROMPT.format(
        current_plan_summary=current_plan_summary_str,
        user_feedback=escape_markdown_v2(user_feedback),
    )

    try:
        analyzed_result: AnalyzedFeedback = (
            await structured_llm_feedback_analyzer.ainvoke(prompt_for_feedback_analysis)
        )
        logger.info(
            f"LLM Analyzed Feedback: {analyzed_result.model_dump_json(indent=2, exclude_none=True)}"
        )

        intent = analyzed_result.intent_type
        change_requests_list_from_llm = analyzed_result.change_requests or []

        # Normalize activity types in change requests
        for cr in change_requests_list_from_llm:
            if cr.new_value_activity:
                act_type_upper = cr.new_value_activity.activity_type.upper()
                if act_type_upper == "THEATER":
                    cr.new_value_activity.activity_type = "PERFORMANCE"
                elif act_type_upper == "CAFE" or act_type_upper == "RESTAURANT":
                    cr.new_value_activity.activity_type = "FOOD_PLACE"

        node_return_values["awaiting_final_confirmation"] = False
        node_return_values["modification_request_details"] = None
        node_return_values["plan_modification_pending"] = False
        updated_messages_list = list(messages)

        if intent == "confirm_plan":
            logger.info("User confirmed the plan.")
            confirmation_message = "Отлично! Рад был помочь с планированием. Если понадобится что-то еще или захотите начать новый поиск, просто напишите!"
            node_return_values["status_message_to_user"] = confirmation_message
            updated_messages_list.append(AIMessage(content=confirmation_message))
            confirmed_plan_collected_data = {
                "current_excluded_ids": {
                    "afisha": [],
                    "park": [],
                    "food": [],
                    "afisha_names_to_avoid": [],
                    "afisha_creation_ids_to_avoid": [],
                }
            }
            if original_collected_data.get("user_start_address_validated_coords"):
                confirmed_plan_collected_data["user_start_address_original"] = (
                    original_collected_data.get("user_start_address_original")
                )
                confirmed_plan_collected_data["user_start_address_validated_coords"] = (
                    original_collected_data.get("user_start_address_validated_coords")
                )
            if original_collected_data.get("city_name"):
                confirmed_plan_collected_data["city_name"] = (
                    original_collected_data.get("city_name")
                )
                confirmed_plan_collected_data["city_id_afisha"] = (
                    original_collected_data.get("city_id_afisha")
                )
            node_return_values["collected_data"] = confirmed_plan_collected_data
            node_return_values["current_events"] = []
            node_return_values["current_route_details"] = None
            node_return_values["is_initial_plan_proposed"] = False
            node_return_values["is_full_plan_with_route_proposed"] = False
            node_return_values["last_presented_plan"] = None
            node_return_values["awaiting_feedback_on_final_plan"] = False
            node_return_values["clarification_needed_fields"] = []
            node_return_values["awaiting_clarification_for_field"] = None

        elif intent == "request_change":
            logger.info(f"User requested changes: {change_requests_list_from_llm}")
            processed_change_requests_for_state: List[Dict] = []

            for cr_llm_obj in change_requests_list_from_llm:
                cr_dict = cr_llm_obj.model_dump(exclude_none=True)
                item_details_for_id = cr_llm_obj.item_to_change_details

                if item_details_for_id and (
                    cr_llm_obj.change_target == "specific_event_replace"
                    or cr_llm_obj.change_target == "specific_event_remove"
                ):
                    found_item_in_previous_plan_dict: Optional[Dict[str, Any]] = None

                    if (
                        item_details_for_id.item_index is not None
                        and 0
                        < item_details_for_id.item_index
                        <= len(combined_previous_items)
                    ):
                        candidate_item = combined_previous_items[
                            item_details_for_id.item_index - 1
                        ]
                        type_matches = True
                        if item_details_for_id.item_type:
                            item_source_type_cand = candidate_item.get(
                                "_item_source_type"
                            )
                            actual_type_key_cand = (
                                candidate_item.get("user_event_type_key")
                                if item_source_type_cand == "event"
                                else candidate_item.get("_plan_item_type")
                            )
                            if not (
                                actual_type_key_cand
                                and actual_type_key_cand.upper()
                                == item_details_for_id.item_type.upper()
                            ):
                                type_matches = False
                        name_matches = True
                        if (
                            item_details_for_id.item_name
                            and item_details_for_id.item_name.lower()
                            not in candidate_item.get("name", "").lower()
                        ):
                            name_matches = False
                        if type_matches and name_matches:
                            found_item_in_previous_plan_dict = candidate_item
                            logger.info(
                                f"Found item to change by index: {found_item_in_previous_plan_dict.get('name')}"
                            )

                    if (
                        not found_item_in_previous_plan_dict
                        and item_details_for_id.item_name
                    ):
                        for prev_item_dict in combined_previous_items:
                            if (
                                item_details_for_id.item_name.lower()
                                in prev_item_dict.get("name", "").lower()
                            ):
                                if item_details_for_id.item_type:
                                    item_source_type_prev = prev_item_dict.get(
                                        "_item_source_type"
                                    )
                                    actual_type_key_prev = (
                                        prev_item_dict.get("user_event_type_key")
                                        if item_source_type_prev == "event"
                                        else prev_item_dict.get("_plan_item_type")
                                    )
                                    if (
                                        actual_type_key_prev
                                        and actual_type_key_prev.upper()
                                        == item_details_for_id.item_type.upper()
                                    ):
                                        found_item_in_previous_plan_dict = (
                                            prev_item_dict
                                        )
                                        logger.info(
                                            f"Found item to change by name and type: {found_item_in_previous_plan_dict.get('name')}"
                                        )
                                        break
                                else:
                                    found_item_in_previous_plan_dict = prev_item_dict
                                    logger.info(
                                        f"Found item to change by name: {found_item_in_previous_plan_dict.get('name')}"
                                    )
                                    break

                    if (
                        not found_item_in_previous_plan_dict
                        and item_details_for_id.item_type
                    ):
                        logger.debug(
                            f"Attempting to find item to change by unique type: {item_details_for_id.item_type}"
                        )
                        candidate_items_of_type = []
                        for item in combined_previous_items:
                            item_source_type = item.get("_item_source_type")
                            actual_item_type_key = None
                            if item_source_type == "event":
                                actual_item_type_key = item.get(
                                    "user_event_type_key", ""
                                ).upper()
                            elif item_source_type == "poi":
                                actual_item_type_key = item.get(
                                    "_plan_item_type", ""
                                ).upper()
                                if actual_item_type_key == "FOOD":
                                    actual_item_type_key = "FOOD_PLACE"
                                elif actual_item_type_key == "PARK":
                                    actual_item_type_key = "PARK"

                            if (
                                actual_item_type_key
                                == item_details_for_id.item_type.upper()
                            ):
                                candidate_items_of_type.append(item)

                        logger.debug(
                            f"Candidate items of type {item_details_for_id.item_type}: {[(c.get('name'), c.get('_plan_item_type')) for c in candidate_items_of_type]}"
                        )

                        if len(candidate_items_of_type) == 1:
                            found_item_in_previous_plan_dict = candidate_items_of_type[
                                0
                            ]
                            logger.info(
                                f"Found unique item to change by type '{item_details_for_id.item_type}': {found_item_in_previous_plan_dict.get('name')}"
                            )
                        elif len(candidate_items_of_type) > 1:
                            logger.warning(
                                f"Multiple items of type '{item_details_for_id.item_type}' found in previous plan. Cannot uniquely identify. LLM needs to be more specific (name or index)."
                            )
                        else:
                            logger.warning(
                                f"No items of type '{item_details_for_id.item_type}' found in previous plan to change by type."
                            )

                    if found_item_in_previous_plan_dict:
                        item_id_to_exclude_val = None
                        item_source_type_found = found_item_in_previous_plan_dict.get(
                            "_item_source_type"
                        )
                        exclude_target_list_key = None
                        item_name_for_log = found_item_in_previous_plan_dict.get(
                            "name", "Unknown Item"
                        )

                        if item_source_type_found == "event":
                            item_id_to_exclude_val = (
                                found_item_in_previous_plan_dict.get("session_id")
                            )
                            exclude_target_list_key = "afisha"
                            item_name_to_exclude_val = (
                                found_item_in_previous_plan_dict.get("name")
                            )
                            item_afisha_id_to_exclude_val = (
                                found_item_in_previous_plan_dict.get("afisha_id")
                            )
                            if item_name_to_exclude_val:
                                names_list = updated_collected_data[
                                    "current_excluded_ids"
                                ].setdefault("afisha_names_to_avoid", [])
                                if item_name_to_exclude_val.lower() not in [
                                    n.lower() for n in names_list
                                ]:
                                    names_list.append(item_name_to_exclude_val)
                            if item_afisha_id_to_exclude_val:
                                creation_ids_list = updated_collected_data[
                                    "current_excluded_ids"
                                ].setdefault("afisha_creation_ids_to_avoid", [])
                                if (
                                    str(item_afisha_id_to_exclude_val)
                                    not in creation_ids_list
                                ):
                                    creation_ids_list.append(
                                        str(item_afisha_id_to_exclude_val)
                                    )
                        elif item_source_type_found == "poi":
                            item_id_to_exclude_val = (
                                found_item_in_previous_plan_dict.get("id_gis")
                            )
                            poi_actual_type_found = (
                                found_item_in_previous_plan_dict.get("_plan_item_type")
                            )
                            if poi_actual_type_found == "park":
                                exclude_target_list_key = "park"
                            elif poi_actual_type_found == "food":
                                exclude_target_list_key = "food"

                        if item_id_to_exclude_val and exclude_target_list_key:
                            cr_dict.setdefault("item_to_change_details", {})[
                                "item_id_str"
                            ] = str(item_id_to_exclude_val)
                            exclude_list = updated_collected_data[
                                "current_excluded_ids"
                            ].setdefault(exclude_target_list_key, [])
                            typed_id_to_exclude = (
                                int(item_id_to_exclude_val)
                                if exclude_target_list_key == "afisha"
                                else str(item_id_to_exclude_val)
                            )
                            if typed_id_to_exclude not in exclude_list:
                                exclude_list.append(typed_id_to_exclude)
                                logger.info(
                                    f"Added ID {typed_id_to_exclude} (type: {exclude_target_list_key}) to excluded_ids for item '{item_name_for_log}'."
                                )
                            else:
                                logger.info(
                                    f"ID {typed_id_to_exclude} (type: {exclude_target_list_key}) for item '{item_name_for_log}' was already in excluded_ids."
                                )
                        else:
                            logger.warning(
                                f"Could not determine ID or exclude list key for item '{item_name_for_log}' to exclude."
                            )
                    else:
                        logger.warning(
                            f"Could not find item in previous plan to replace/remove based on details: {item_details_for_id.model_dump(exclude_none=True)}"
                        )

                processed_change_requests_for_state.append(cr_dict)

            modification_details_for_next_node = {
                "change_requests": processed_change_requests_for_state
            }

            for ch_req_obj_for_global_update in change_requests_list_from_llm:
                if (
                    ch_req_obj_for_global_update.change_target == "budget"
                    and ch_req_obj_for_global_update.new_value_int is not None
                ):
                    updated_collected_data["budget_current_search"] = (
                        ch_req_obj_for_global_update.new_value_int
                    )
                    updated_collected_data["budget_original"] = (
                        ch_req_obj_for_global_update.new_value_int
                    )
                    modification_details_for_next_node["new_budget"] = (
                        ch_req_obj_for_global_update.new_value_int
                    )
                    logger.info(
                        f"Budget updated to {ch_req_obj_for_global_update.new_value_int} due to feedback."
                    )

            feedback_ack_msg_text_val = (
                "Понял ваш запрос на изменения. Сейчас попробую обновить план..."
            )

            node_return_values["modification_request_details"] = (
                modification_details_for_next_node
            )
            node_return_values["plan_modification_pending"] = True
            node_return_values["current_events"] = []
            node_return_values["current_route_details"] = None
            node_return_values["is_initial_plan_proposed"] = False
            node_return_values["is_full_plan_with_route_proposed"] = False
            node_return_values["status_message_to_user"] = feedback_ack_msg_text_val
            updated_messages_list.append(AIMessage(content=feedback_ack_msg_text_val))

        elif intent == "new_search":
            logger.info("User requested a new search.")
            new_search_msg_text_val = "Понял, давайте начнем новый поиск. Расскажите, что бы вы хотели найти теперь (город, даты, интересы)?"
            node_return_values["status_message_to_user"] = new_search_msg_text_val
            updated_messages_list.append(AIMessage(content=new_search_msg_text_val))
            new_search_collected_data = {
                "current_excluded_ids": {
                    "afisha": [],
                    "park": [],
                    "food": [],
                    "afisha_names_to_avoid": [],
                    "afisha_creation_ids_to_avoid": [],
                }
            }
            if original_collected_data.get("user_start_address_validated_coords"):
                new_search_collected_data["user_start_address_original"] = (
                    original_collected_data.get("user_start_address_original")
                )
                new_search_collected_data["user_start_address_validated_coords"] = (
                    original_collected_data.get("user_start_address_validated_coords")
                )
            if original_collected_data.get("city_name"):
                new_search_collected_data["city_name"] = original_collected_data.get(
                    "city_name"
                )
                new_search_collected_data["city_id_afisha"] = (
                    original_collected_data.get("city_id_afisha")
                )

            new_search_collected_data.setdefault("clarification_needed_fields", [])
            new_search_fields = ["dates_description_original", "interests_original"]
            if not new_search_collected_data.get("city_name"):
                new_search_fields.append("city_name")
            new_search_collected_data["clarification_needed_fields"] = list(
                set(
                    new_search_collected_data.get("clarification_needed_fields", [])
                    + new_search_fields
                )
            )
            new_search_collected_data["clarification_needed_fields"] = [
                f for f in new_search_collected_data["clarification_needed_fields"] if f
            ]

            updated_collected_data = new_search_collected_data
            node_return_values["current_events"] = []
            node_return_values["current_route_details"] = None
            node_return_values["is_initial_plan_proposed"] = False
            node_return_values["is_full_plan_with_route_proposed"] = False
            node_return_values["last_presented_plan"] = None
            node_return_values["awaiting_feedback_on_final_plan"] = False
            node_return_values["modification_request_details"] = None
            node_return_values["plan_modification_pending"] = False

        else:
            logger.info(
                f"User feedback intent was '{intent}', requires clarification or retry."
            )
            misunderstanding_msg_text_val = "Я не совсем понял ваш ответ. Не могли бы вы уточнить, что вы имели в виду, или, возможно, вы хотите попробовать другие критерии поиска?"
            node_return_values["status_message_to_user"] = misunderstanding_msg_text_val
            updated_messages_list.append(
                AIMessage(content=misunderstanding_msg_text_val)
            )
            node_return_values["awaiting_final_confirmation"] = True

        node_return_values["collected_data"] = updated_collected_data
        node_return_values["messages"] = updated_messages_list
        logger.info(
            f"handle_plan_feedback_node: Exiting. current_excluded_ids in returned collected_data: {updated_collected_data.get('current_excluded_ids')}"
        )
        return node_return_values

    except Exception as e_feedback_main_exc:
        logger.error(
            f"Error in handle_plan_feedback_node: {e_feedback_main_exc}", exc_info=True
        )
        error_message_to_user = "К сожалению, произошла ошибка при обработке вашего ответа. Пожалуйста, попробуйте еще раз или сформулируйте ваш запрос по-другому."
        node_return_values["status_message_to_user"] = error_message_to_user
        node_return_values["messages"] = messages + [
            AIMessage(content=error_message_to_user)
        ]
        node_return_values["awaiting_final_confirmation"] = True
        node_return_values["collected_data"] = updated_collected_data
        return node_return_values


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
# Файл: agent_core/nodes.py
async def error_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: error_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    messages: List[BaseMessage] = state.get("messages", [])

    # Пытаемся понять, почему мы попали в error_node
    reason_message_parts = []

    # Если была попытка построить план, но она провалилась на каком-то шаге
    plan_construction_failed_step = collected_data_dict.get(
        "plan_construction_failed_step"
    )
    if plan_construction_failed_step:
        reason_message_parts.append(
            f"Мне не удалось составить полный план, так как {plan_construction_failed_step}."
        )

    # Если не найдены основные интересы Афиши (которые были в ordered_activities)
    # not_found_primary_afisha_keys используется, если search_events_node не смог найти основной вариант
    not_found_afisha_keys = collected_data_dict.get(
        "not_found_interest_keys_in_primary_search", []
    )
    # Исключаем те, для которых есть fallback, так как он мог быть предложен
    fallback_candidates = collected_data_dict.get("fallback_candidates", {})

    actually_not_found_afisha_display_names = []
    if not_found_afisha_keys:
        for key in not_found_afisha_keys:
            if (
                key not in fallback_candidates
            ):  # Интерес действительно не найден и нет fallback
                display_name = AFISHA_KEY_TO_READABLE_NAME_MAP.get(key, key)
                actually_not_found_afisha_display_names.append(display_name)

    if actually_not_found_afisha_display_names:
        if not plan_construction_failed_step:  # Если это основная причина
            reason_message_parts.append(
                f"по вашему запросу не нашлось подходящих {', '.join(actually_not_found_afisha_display_names)}"
            )
        else:  # Если уже была ошибка построения плана, это доп. информация
            reason_message_parts.append(
                f"Также не было найдено: {', '.join(actually_not_found_afisha_display_names)}"
            )

    # Проверяем, искали ли POI и нашли ли их
    poi_park_query = collected_data_dict.get("poi_park_query")
    found_parks = collected_data_dict.get("found_parks")
    if poi_park_query and not found_parks:
        reason_message_parts.append(
            f"не удалось найти парки по запросу '{poi_park_query}'"
        )

    poi_food_query = collected_data_dict.get("poi_food_query")
    found_food = collected_data_dict.get("found_food_places")
    if poi_food_query and not found_food:
        reason_message_parts.append(
            f"не удалось найти места для еды по запросу '{poi_food_query}'"
        )

    # Формируем итоговое сообщение
    error_msg: str
    if reason_message_parts:
        error_msg = "К сожалению, " + " и ".join(reason_message_parts) + "."
    else:
        # Общее сообщение, если причина не ясна из collected_data
        error_msg = (
            "К сожалению, по вашему текущему запросу ничего подходящего не нашлось."
        )

    error_msg += "\nМожет быть, попробуем изменить критерии? Например, другие даты, интересы или бюджет?"

    # Очищаем состояние для нового поиска или уточнения.
    # Важно не сбрасывать user_start_address, если он был введен и валидирован.
    new_collected_data = {}
    if collected_data_dict.get(
        "user_start_address_original"
    ) and collected_data_dict.get("user_start_address_validated_coords"):
        new_collected_data["user_start_address_original"] = collected_data_dict.get(
            "user_start_address_original"
        )
        new_collected_data["user_start_address_validated_coords"] = (
            collected_data_dict.get("user_start_address_validated_coords")
        )
    # Также можно сохранить город, если он был успешно определен
    if collected_data_dict.get("city_name") and collected_data_dict.get(
        "city_id_afisha"
    ):
        new_collected_data["city_name"] = collected_data_dict.get("city_name")
        new_collected_data["city_id_afisha"] = collected_data_dict.get("city_id_afisha")

    # Сбрасываем поля, связанные с предыдущим неудачным поиском
    new_collected_data["clarification_needed_fields"] = (
        []
    )  # Будут определены заново, если пользователь ответит на предложение изменить критерии

    new_messages_history = messages + [AIMessage(content=error_msg)]

    return {
        "messages": new_messages_history,
        "status_message_to_user": error_msg,
        "current_events": [],  # Сбрасываем
        "current_route_details": None,  # Сбрасываем
        "is_initial_plan_proposed": False,  # План не предложен
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,
        "awaiting_fallback_confirmation": False,  # Сбрасываем
        "pending_fallback_event": None,  # Сбрасываем
        "collected_data": new_collected_data,  # Очищенные, но с возможным сохранением адреса/города
        "awaiting_clarification_for_field": None,  # После error_node обычно ждем новый ввод или уточнение общих критериев
    }
