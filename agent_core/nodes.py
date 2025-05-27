import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, date, timedelta  # Убедимся, что date импортирован
import asyncio  # Для asyncio.gather в новой логике

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
    messages: List[AIMessage | HumanMessage] = state.get("messages", [])
    current_collected_data_dict: dict = dict(
        state.get("collected_data", {})
    )  # Работаем со словарем
    current_events: List[Event] = state.get("current_events", [])

    if (
        current_collected_data_dict.get("awaiting_fallback_confirmation")
        and messages
        and isinstance(messages[-1], HumanMessage)
    ):
        user_reply = messages[-1].content.lower()
        pending_fallback_event_data = current_collected_data_dict.get(
            "pending_fallback_event"
        )
        logger.info(f"Processing user reply '{user_reply}' to fallback proposal.")

        del current_collected_data_dict["awaiting_fallback_confirmation"]
        if "pending_fallback_event" in current_collected_data_dict:
            del current_collected_data_dict["pending_fallback_event"]

        if pending_fallback_event_data and isinstance(
            pending_fallback_event_data, dict
        ):
            try:
                pending_fallback_event = Event(**pending_fallback_event_data)
                if (
                    "да" in user_reply
                    or "хочу" in user_reply
                    or "добавить" in user_reply
                ):
                    logger.info(
                        f"User confirmed fallback event: {pending_fallback_event.name}"
                    )
                    updated_current_events = list(current_events)
                    updated_current_events.append(pending_fallback_event)
                    updated_current_events.sort(
                        key=lambda e: e.start_time_naive_event_tz
                    )

                    if "not_found_interest_keys" in current_collected_data_dict:
                        if (
                            pending_fallback_event.event_type_key
                            in current_collected_data_dict["not_found_interest_keys"]
                        ):
                            current_collected_data_dict[
                                "not_found_interest_keys"
                            ].remove(pending_fallback_event.event_type_key)
                        if not current_collected_data_dict.get(
                            "not_found_interest_keys"
                        ):
                            del current_collected_data_dict["not_found_interest_keys"]

                    current_collected_data_dict[
                        "fallback_accepted_and_plan_updated"
                    ] = True
                    return {
                        "collected_data": current_collected_data_dict,
                        "messages": messages,
                        "current_events": updated_current_events,
                        "is_initial_plan_proposed": True,
                        "status_message_to_user": None,
                    }
                else:
                    logger.info(
                        f"User rejected or gave unclear answer for fallback event: {pending_fallback_event.name}"
                    )
                    return {
                        "collected_data": current_collected_data_dict,
                        "messages": messages,
                        "current_events": current_events,
                        "status_message_to_user": None,
                        "is_initial_plan_proposed": bool(current_events),
                    }
            except ValidationError as ve_fallback_extract:
                logger.error(
                    f"Error validating pending_fallback_event data: {ve_fallback_extract}"
                )
                return {
                    "collected_data": current_collected_data_dict,
                    "messages": messages,
                    "current_events": current_events,
                    "status_message_to_user": "Произошла ошибка с предложенным вариантом.",
                    "is_initial_plan_proposed": bool(current_events),
                }
        else:
            logger.warning(
                "awaiting_fallback_confirmation but no valid pending_fallback_event data."
            )
            return {
                "collected_data": current_collected_data_dict,
                "messages": messages,
                "current_events": current_events,
                "status_message_to_user": None,
                "is_initial_plan_proposed": bool(current_events),
            }

    if not messages or not isinstance(messages[-1], HumanMessage):
        return {
            "collected_data": current_collected_data_dict,
            "messages": messages,
            "clarification_context": state.get("clarification_context"),
        }

    user_query = messages[-1].content.strip()
    is_awaiting_address = current_collected_data_dict.get(
        "awaiting_address_input", False
    )
    clarification_context_for_node: Optional[str] = None

    current_collected_data_dict["clarification_needed_fields"] = [
        f
        for f in current_collected_data_dict.get("clarification_needed_fields", [])
        if f != "user_start_address_original"
    ]

    # Флаг, который укажет, нужно ли проваливаться в основную логику извлечения после обработки адреса/команд
    proceed_to_general_extraction = not is_awaiting_address

    if is_awaiting_address:
        logger.info(f"Input '{user_query}' received while awaiting address.")
        user_query_lower = user_query.lower()
        reset_commands = ["новый поиск", "начни сначала", "отмена", "сброс", "стоп"]

        if any(cmd in user_query_lower for cmd in reset_commands):
            logger.info(
                f"User requested to reset/start new search with: '{user_query}'"
            )
            current_collected_data_dict = {
                "clarification_needed_fields": [
                    "city_name",
                    "dates_description_original",
                    "interests_original",
                ],
                "awaiting_address_input": False,
            }
            clarification_context_for_node = (
                "Хорошо, давайте начнем сначала. Расскажите, что бы вы хотели найти?"
            )
            return {
                "collected_data": current_collected_data_dict,
                "messages": messages
                + [AIMessage(content=clarification_context_for_node)],
                "clarification_context": clarification_context_for_node,
            }

        city_for_geocoding = current_collected_data_dict.get("city_name")
        previously_found_street = current_collected_data_dict.get(
            "partial_address_street"
        )
        address_to_geocode = user_query

        if previously_found_street and not any(
            c.isalpha()
            for c in user_query
            if c.isalpha()
            and c.lower()
            not in [
                "а",
                "б",
                "в",
                "г",
                "д",
                "е",
                "ж",
                "з",
                "к",
                "л",
                "м",
                "н",
                "о",
                "п",
                "р",
                "с",
                "т",
                "у",
                "ф",
                "х",
                "ц",
                "ч",
                "ш",
                "щ",
                "ъ",
                "ы",
                "ь",
                "э",
                "ю",
                "я",
            ]
        ):
            address_to_geocode = f"{previously_found_street}, {user_query}"
            logger.info(
                f"Attempting to geocode combined address: '{address_to_geocode}'"
            )

        if not city_for_geocoding:
            logger.error("City name missing while expecting address. Cannot geocode.")
            current_collected_data_dict.setdefault(
                "clarification_needed_fields", []
            ).append("city_name")
            current_collected_data_dict["awaiting_address_input"] = False
            if "partial_address_street" in current_collected_data_dict:
                del current_collected_data_dict["partial_address_street"]
        else:
            geocoding_result: GeocodingResult = await get_geocoding_details(
                address=address_to_geocode, city=city_for_geocoding
            )

            if geocoding_result.is_precise_enough and geocoding_result.coords:
                logger.info(
                    f"Address '{address_to_geocode}' geocoded as 'building': {geocoding_result.full_address_name_gis}"
                )
                current_collected_data_dict["user_start_address_original"] = (
                    geocoding_result.full_address_name_gis
                )
                current_collected_data_dict["user_start_address_validated_coords"] = {
                    "lon": geocoding_result.coords[0],
                    "lat": geocoding_result.coords[1],
                }
                current_collected_data_dict["awaiting_address_input"] = False
                if "partial_address_street" in current_collected_data_dict:
                    del current_collected_data_dict["partial_address_street"]
            elif (
                geocoding_result.match_level == "street" and not previously_found_street
            ):
                logger.info(
                    f"Address '{user_query}' geocoded as 'street': {geocoding_result.full_address_name_gis}."
                )
                clarification_context_for_node = f"Я нашел улицу '{geocoding_result.full_address_name_gis}'. Пожалуйста, уточните номер дома."
                current_collected_data_dict["partial_address_street"] = (
                    geocoding_result.full_address_name_gis
                )
                current_collected_data_dict["awaiting_address_input"] = True
                current_collected_data_dict.setdefault(
                    "clarification_needed_fields", []
                ).append("user_start_address_original")
            else:
                if "partial_address_street" in current_collected_data_dict:
                    del current_collected_data_dict["partial_address_street"]
                logger.warning(
                    f"Could not geocode '{address_to_geocode}' precisely. Match: {geocoding_result.match_level}. Error: {geocoding_result.error_message}. Trying to interpret as new query."
                )

                llm = get_gigachat_client()
                structured_llm = llm.with_structured_output(ExtractedInitialInfo)
                extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{user_query}"'
                try:
                    potential_new_request_info: ExtractedInitialInfo = (
                        await structured_llm.ainvoke(extraction_prompt_with_query)
                    )
                    if (
                        potential_new_request_info.city
                        or potential_new_request_info.dates_description
                        or potential_new_request_info.interests
                    ):
                        logger.info(
                            f"Interpreting '{user_query}' as a NEW REQUEST while awaiting address."
                        )

                        preserved_city = current_collected_data_dict.get("city_name")
                        preserved_city_id = current_collected_data_dict.get(
                            "city_id_afisha"
                        )
                        preserved_budget_orig = current_collected_data_dict.get(
                            "budget_original"
                        )
                        preserved_budget_curr = current_collected_data_dict.get(
                            "budget_current_search"
                        )
                        # Даты и интересы не сохраняем при таком сценарии, т.к. новый запрос их перезапишет

                        current_collected_data_dict = {}
                        current_collected_data_dict["clarification_needed_fields"] = []
                        current_collected_data_dict["awaiting_address_input"] = False

                        # Переносим извлеченное из potential_new_request_info в current_collected_data_dict
                        # или сохраняем старые, если LLM не вернул новые
                        current_collected_data_dict["city_name"] = (
                            potential_new_request_info.city or preserved_city
                        )
                        if (
                            current_collected_data_dict.get("city_name")
                            == preserved_city
                            and preserved_city_id
                        ):  # Если город не изменился, сохраняем ID
                            current_collected_data_dict["city_id_afisha"] = (
                                preserved_city_id
                            )
                        # Бюджет
                        if potential_new_request_info.budget is not None:
                            current_collected_data_dict["budget_original"] = (
                                potential_new_request_info.budget
                            )
                            current_collected_data_dict["budget_current_search"] = (
                                potential_new_request_info.budget
                            )
                        elif preserved_budget_orig is not None:
                            current_collected_data_dict["budget_original"] = (
                                preserved_budget_orig
                            )
                            current_collected_data_dict["budget_current_search"] = (
                                preserved_budget_curr
                            )

                        # Интересы, даты, время будут взяты из potential_new_request_info ниже
                        proceed_to_general_extraction = (
                            True  # Флаг, чтобы войти в основную логику извлечения
                        )
                        user_query_for_general_extraction = (
                            user_query  # Будем использовать оригинальный user_query
                        )
                    else:
                        clarification_context_for_node = f"Не удалось распознать '{user_query}' как точный адрес. Пожалуйста, укажите улицу и номер дома. Если хотите начать новый поиск, скажите 'новый поиск'."
                        current_collected_data_dict["awaiting_address_input"] = True
                        current_collected_data_dict.setdefault(
                            "clarification_needed_fields", []
                        ).append("user_start_address_original")
                except Exception as e_llm_on_addr_fail:
                    logger.error(
                        f"LLM failed to process '{user_query}' after geocoding attempt failed: {e_llm_on_addr_fail}"
                    )
                    clarification_context_for_node = f"Не удалось распознать '{user_query}' как точный адрес. Пожалуйста, укажите улицу и номер дома. Если хотите начать новый поиск, скажите 'новый поиск'."
                    current_collected_data_dict["awaiting_address_input"] = True
                    current_collected_data_dict.setdefault(
                        "clarification_needed_fields", []
                    ).append("user_start_address_original")

        if current_collected_data_dict.get(
            "awaiting_address_input"
        ) or current_collected_data_dict.get("clarification_needed_fields"):
            if (
                not proceed_to_general_extraction
            ):  # Если мы не перешли к новому запросу, возвращаем состояние для уточнения адреса
                logger.debug(
                    f"extract_initial_info_node: End of address block, clarification needed for address. Data: {str(current_collected_data_dict)[:500]}"
                )
                return {
                    "collected_data": current_collected_data_dict,
                    "messages": messages,
                    "clarification_context": clarification_context_for_node,
                }

        if not proceed_to_general_extraction and not current_collected_data_dict.get(
            "awaiting_address_input"
        ):  # Если адрес успешно обработан
            logger.debug(
                f"extract_initial_info_node: End of address block, address processed. Data: {str(current_collected_data_dict)[:500]}"
            )
            return {
                "collected_data": current_collected_data_dict,
                "messages": messages,
                "clarification_context": None,
            }

    # --- Основная логика извлечения (если не is_awaiting_address изначально, или если мы сюда "провалились") ---
    if proceed_to_general_extraction:
        logger.debug(
            "extract_initial_info_node: Proceeding with general information extraction."
        )

        # Если current_collected_data пуст (т.е. это самый первый запрос сессии, или был полный сброс)
        # ИЛИ если мы пришли сюда из блока is_awaiting_address с решением, что это новый запрос
        # и current_collected_data уже частично заполнен из potential_new_request_info,
        # то нужно корректно использовать prev_ значения.
        # Если current_collected_data НЕ пуст и это не первый запрос, но и не ожидание адреса,
        # то prev_ значения будут из текущего состояния.
        if not current_collected_data_dict or (
            is_awaiting_address and proceed_to_general_extraction
        ):
            # Если это новый запрос, то prev_ значения должны быть None, чтобы не подмешивать старые данные
            # кроме тех, что мы уже перенесли в current_collected_data_dict из potential_new_request_info
            prev_dates_desc = current_collected_data_dict.get(
                "dates_description_original"
            )
            prev_parsed_dates = current_collected_data_dict.get("parsed_dates_iso")
            prev_parsed_end_dates = current_collected_data_dict.get(
                "parsed_end_dates_iso"
            )
            prev_interests_orig = current_collected_data_dict.get("interests_original")
            prev_interests_keys = current_collected_data_dict.get(
                "interests_keys_afisha"
            )
            prev_raw_time_desc = current_collected_data_dict.get(
                "raw_time_description_original"
            )
            # Город и бюджет уже должны быть в current_collected_data_dict, если они сохранялись
        else:  # Сохраняем текущие значения перед вызовом LLM, если это не полный сброс
            prev_city_name = current_collected_data_dict.get("city_name")
            prev_city_id_afisha = current_collected_data_dict.get("city_id_afisha")
            prev_dates_desc = current_collected_data_dict.get(
                "dates_description_original"
            )
            prev_parsed_dates = current_collected_data_dict.get("parsed_dates_iso")
            prev_parsed_end_dates = current_collected_data_dict.get(
                "parsed_end_dates_iso"
            )
            prev_interests_orig = current_collected_data_dict.get("interests_original")
            prev_interests_keys = current_collected_data_dict.get(
                "interests_keys_afisha"
            )
            prev_raw_time_desc = current_collected_data_dict.get(
                "raw_time_description_original"
            )
            prev_budget = current_collected_data_dict.get("budget_original")

        llm = get_gigachat_client()
        structured_llm = llm.with_structured_output(ExtractedInitialInfo)
        try:
            logger.debug(
                f"extract_initial_info_node: Querying LLM for general info from: '{user_query}'"
            )
            extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{user_query}"'
            extracted_info: ExtractedInitialInfo = await structured_llm.ainvoke(
                extraction_prompt_with_query
            )
            logger.info(
                f"extract_initial_info_node: LLM Extracted Info: {extracted_info.model_dump_json(indent=2)}"
            )

            # Обновляем город, только если LLM его извлек
            if extracted_info.city:
                current_collected_data_dict["city_name"] = extracted_info.city
                cities_from_afisha = await fetch_cities_internal()
                found_city_afisha = next(
                    (
                        c
                        for c in cities_from_afisha
                        if extracted_info.city
                        and extracted_info.city.lower() in c["name_lower"]
                    ),
                    None,
                )
                if found_city_afisha:
                    current_collected_data_dict["city_id_afisha"] = found_city_afisha[
                        "id"
                    ]
                else:
                    current_collected_data_dict["city_id_afisha"] = None
                    current_collected_data_dict.setdefault(
                        "clarification_needed_fields", []
                    ).append("city_name")
            elif not current_collected_data_dict.get(
                "city_name"
            ):  # Если и LLM не дал, и раньше не было
                current_collected_data_dict.setdefault(
                    "clarification_needed_fields", []
                ).append("city_name")

            if extracted_info.interests:
                current_collected_data_dict["interests_original"] = (
                    extracted_info.interests
                )
                mapped_interest_keys = []
                for interest_str in extracted_info.interests:
                    key = None
                    s = interest_str.lower()
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
                    elif (
                        "кафе" in s
                        or "ресторан" in s
                        or "покушать" in s
                        or "поесть" in s
                    ):
                        key = "Кафе"
                    if not key:
                        key = interest_str.capitalize()
                    mapped_interest_keys.append(key)
                current_collected_data_dict["interests_keys_afisha"] = list(
                    set(mapped_interest_keys)
                )
            elif not current_collected_data_dict.get("interests_keys_afisha"):
                current_collected_data_dict.setdefault(
                    "clarification_needed_fields", []
                ).append("interests_original")

            if extracted_info.budget is not None:
                current_collected_data_dict["budget_original"] = extracted_info.budget
                current_collected_data_dict["budget_current_search"] = (
                    extracted_info.budget
                )
            elif (
                current_collected_data_dict.get("budget_original") is None
                and prev_budget is not None
            ):  # Если LLM не дал, а раньше был
                current_collected_data_dict["budget_original"] = prev_budget
                current_collected_data_dict["budget_current_search"] = prev_budget

            dates_desc_from_llm = extracted_info.dates_description
            time_qualifier_from_llm = extracted_info.raw_time_description

            if dates_desc_from_llm or time_qualifier_from_llm:
                if dates_desc_from_llm:
                    current_collected_data_dict["dates_description_original"] = (
                        dates_desc_from_llm
                    )
                if time_qualifier_from_llm:
                    current_collected_data_dict["raw_time_description_original"] = (
                        time_qualifier_from_llm
                    )

                natural_date_for_parser = dates_desc_from_llm
                if not natural_date_for_parser and time_qualifier_from_llm:
                    if prev_dates_desc:
                        natural_date_for_parser = prev_dates_desc
                    else:
                        natural_date_for_parser = "сегодня"
                        current_collected_data_dict["dates_description_original"] = (
                            "сегодня"
                        )
                elif not natural_date_for_parser and not time_qualifier_from_llm:
                    if prev_parsed_dates:
                        current_collected_data_dict["dates_description_original"] = (
                            prev_dates_desc
                        )
                        current_collected_data_dict["parsed_dates_iso"] = (
                            prev_parsed_dates
                        )
                        if prev_parsed_end_dates:
                            current_collected_data_dict["parsed_end_dates_iso"] = (
                                prev_parsed_end_dates
                            )
                        if prev_raw_time_desc:
                            current_collected_data_dict[
                                "raw_time_description_original"
                            ] = prev_raw_time_desc
                    else:
                        current_collected_data_dict.setdefault(
                            "clarification_needed_fields", []
                        ).append("dates_description_original")

                if natural_date_for_parser:
                    current_iso_for_parser = datetime.now().isoformat()
                    parsed_date_time_result = await datetime_parser_tool.ainvoke(
                        {
                            "natural_language_date": natural_date_for_parser,
                            "natural_language_time_qualifier": time_qualifier_from_llm,
                            "base_date_iso": current_iso_for_parser,
                        }
                    )
                    if parsed_date_time_result.get("datetime_iso"):
                        current_collected_data_dict["parsed_dates_iso"] = [
                            parsed_date_time_result["datetime_iso"]
                        ]
                        if parsed_date_time_result.get("end_datetime_iso"):
                            current_collected_data_dict["parsed_end_dates_iso"] = [
                                parsed_date_time_result["end_datetime_iso"]
                            ]
                        elif "parsed_end_dates_iso" in current_collected_data_dict:
                            del current_collected_data_dict["parsed_end_dates_iso"]
                        if parsed_date_time_result.get("is_ambiguous"):
                            current_collected_data_dict.setdefault(
                                "clarification_needed_fields", []
                            ).append("dates_description_original")
                            clarification_context_for_node = (
                                parsed_date_time_result.get("clarification_needed")
                            )
                    else:
                        current_collected_data_dict.setdefault(
                            "clarification_needed_fields", []
                        ).append("dates_description_original")
                        clarification_context_for_node = (
                            parsed_date_time_result.get("clarification_needed")
                            or parsed_date_time_result.get("error_message")
                            or "Не удалось распознать дату или время из вашего запроса."
                        )
            elif prev_parsed_dates:
                current_collected_data_dict["dates_description_original"] = (
                    prev_dates_desc
                )
                current_collected_data_dict["parsed_dates_iso"] = prev_parsed_dates
                if prev_parsed_end_dates:
                    current_collected_data_dict["parsed_end_dates_iso"] = (
                        prev_parsed_end_dates
                    )
                if prev_raw_time_desc:
                    current_collected_data_dict["raw_time_description_original"] = (
                        prev_raw_time_desc
                    )
            elif not current_collected_data_dict.get("parsed_dates_iso"):
                current_collected_data_dict.setdefault(
                    "clarification_needed_fields", []
                ).append("dates_description_original")
        except Exception as e:
            logger.error(
                f"extract_initial_info_node: Critical error during LLM call or info processing: {e}",
                exc_info=True,
            )
            for f_key in [
                "city_name",
                "dates_description_original",
                "interests_original",
            ]:
                if not current_collected_data_dict.get(
                    f_key
                ) and f_key not in current_collected_data_dict.get(
                    "clarification_needed_fields", []
                ):
                    current_collected_data_dict.setdefault(
                        "clarification_needed_fields", []
                    ).append(f_key)

        if "clarification_needed_fields" in current_collected_data_dict:
            fields_list_unique = list(
                dict.fromkeys(
                    current_collected_data_dict["clarification_needed_fields"]
                )
            )
            current_collected_data_dict["clarification_needed_fields"] = [
                f for f in fields_list_unique if f
            ]
            if not current_collected_data_dict["clarification_needed_fields"]:
                del current_collected_data_dict["clarification_needed_fields"]

    logger.info(
        f"extract_initial_info_node: Final collected_data state: {str(current_collected_data_dict)[:500]}"
    )
    if clarification_context_for_node:
        logger.info(
            f"extract_initial_info_node: Clarification context set to: {clarification_context_for_node}"
        )

    return {
        "collected_data": current_collected_data_dict,
        "messages": messages,
        "clarification_context": clarification_context_for_node,
    }


# --- Узел 2: Уточнение недостающих данных ---
async def clarify_missing_data_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_missing_data_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    clarification_fields: List[str] = collected_data.get("clarification_needed_fields", [])  # type: ignore
    status_message_to_user: Optional[str] = None
    prompt_for_llm: Optional[str] = None  # Инициализируем prompt_for_llm

    if not clarification_fields:
        logger.info("clarify_missing_data_node: No fields need explicit clarification.")
        return {
            "status_message_to_user": None,
            "awaiting_final_confirmation": False,
            "clarification_context": None,
        }

    missing_critical_fields_map = {
        "city_name": "город для поиска",
        "dates_description_original": "даты или период мероприятий",
        "interests_original": "ваши интересы или тип мероприятий",
    }
    raw_time_desc_original: Optional[str] = collected_data.get(
        "raw_time_description_original"
    )
    clarification_context_from_state: Optional[Any] = state.get("clarification_context")

    # 1. Используем готовый вопрос из контекста, если он есть и является строкой
    if (
        isinstance(clarification_context_from_state, str)
        and clarification_context_from_state
    ):
        status_message_to_user = clarification_context_from_state
        logger.info(
            f"clarify_missing_data_node: Using pre-defined clarification context: {status_message_to_user}"
        )

    # 2. Если готового вопроса нет, формируем промпт для LLM
    elif (
        raw_time_desc_original and "dates_description_original" in clarification_fields
    ):
        current_date_info = date.today().strftime("%d %B %Y года (%A)")
        prompt_for_llm = TIME_CLARIFICATION_PROMPT_TEMPLATE.format(
            raw_time_description=raw_time_desc_original,
            current_date_info=current_date_info,
        )
        logger.debug(
            f"clarify_missing_data_node: Will use TIME_CLARIFICATION_PROMPT for '{raw_time_desc_original}'"
        )
    else:
        fields_to_ask_user_text_parts = [
            missing_critical_fields_map[f_key]
            for f_key in [
                "city_name",
                "dates_description_original",
                "interests_original",
            ]
            if f_key in clarification_fields
        ]
        if not fields_to_ask_user_text_parts:
            # Этого не должно происходить, если clarification_fields не пуст и не сработали предыдущие условия
            status_message_to_user = "Кажется, мне нужно немного больше информации. Не могли бы вы уточнить ваш запрос?"
            logger.warning(
                f"clarify_missing_data_node: No specific critical fields for general prompt, but clarification_fields is: {clarification_fields}"
            )
        else:
            missing_fields_text_for_prompt = " и ".join(fields_to_ask_user_text_parts)
            user_query_for_prompt = "Ваш запрос."
            messages_history: List[HumanMessage | AIMessage] = state.get("messages", [])  # type: ignore
            if messages_history and isinstance(messages_history[-1], HumanMessage):
                user_query_for_prompt = messages_history[-1].content

            collected_summary_parts = []
            if collected_data.get("city_name"):
                collected_summary_parts.append(f"город: {collected_data['city_name']}")
            if collected_data.get("parsed_dates_iso"):
                collected_summary_parts.append(f"даты: {', '.join(collected_data['parsed_dates_iso'])}")  # type: ignore
            if collected_data.get("interests_original"):
                collected_summary_parts.append(f"интересы: {', '.join(collected_data['interests_original'])}")  # type: ignore
            current_collected_data_summary_str = (
                "; ".join(collected_summary_parts)
                if collected_summary_parts
                else "пока ничего не уточнено."
            )

            prompt_for_llm = GENERAL_CLARIFICATION_PROMPT_TEMPLATE.format(
                user_query=user_query_for_prompt,
                current_collected_data_summary=current_collected_data_summary_str,
                missing_fields_description=missing_fields_text_for_prompt,
            )
            logger.debug(
                f"clarify_missing_data_node: Will use GENERAL_CLARIFICATION_PROMPT for: {missing_fields_text_for_prompt}"
            )

    # 3. Вызываем LLM, только если status_message_to_user еще не установлен и prompt_for_llm был сформирован
    if not status_message_to_user and prompt_for_llm:
        llm = get_gigachat_client()
        try:
            ai_response = await llm.ainvoke(prompt_for_llm)
            status_message_to_user = ai_response.content
            logger.info(
                f"clarify_missing_data_node: LLM generated clarification question: {status_message_to_user}"
            )
        except Exception as e_clarify:
            logger.error(
                f"clarify_missing_data_node: Error during LLM call for clarification: {e_clarify}",
                exc_info=True,
            )
            status_message_to_user = "Произошла ошибка при попытке уточнить ваш запрос. Пожалуйста, попробуйте сформулировать его иначе."
    elif (
        not status_message_to_user
    ):  # Если clarification_fields есть, но ни один из кейсов выше не установил сообщение
        logger.error(
            f"clarify_missing_data_node: Unhandled clarification scenario. Clarification_fields: {clarification_fields}, Context: {clarification_context_from_state}"
        )
        status_message_to_user = "Мне нужно несколько уточнений. Не могли бы вы переформулировать ваш запрос?"

    new_messages_history = state.get("messages", []) + [AIMessage(content=status_message_to_user or "Не могу продолжить без уточнений.")]  # type: ignore

    return {
        "messages": new_messages_history,
        "status_message_to_user": status_message_to_user,
        "awaiting_final_confirmation": False,
        "clarification_context": None,
    }


# --- Узел 3: Поиск мероприятий (ОБНОВЛЕННАЯ ВЕРСИЯ) ---
async def search_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: search_events_node executing...")
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))
    original_user_interests_keys: List[str] = list(
        collected_data.get("interests_keys_afisha", [])
    )

    collected_data["not_found_interest_keys"] = []
    collected_data["fallback_candidates"] = {}

    city_id = collected_data.get("city_id_afisha")
    parsed_dates_iso_list = collected_data.get("parsed_dates_iso")
    parsed_end_dates_iso_list = collected_data.get("parsed_end_dates_iso")
    budget = collected_data.get("budget_current_search")

    if not city_id or not parsed_dates_iso_list or not original_user_interests_keys:
        logger.error(
            f"search_events_node: Missing critical data. CityID: {city_id}, Dates: {parsed_dates_iso_list}, Interests: {original_user_interests_keys}"
        )
        if not original_user_interests_keys and collected_data.get(
            "interests_original"
        ):
            collected_data["not_found_interest_keys"] = list(collected_data.get("interests_keys_afisha", []))  # type: ignore
        elif original_user_interests_keys:
            collected_data["not_found_interest_keys"] = list(original_user_interests_keys)  # type: ignore
        return {
            "current_events": [],
            "status_message_to_user": "Мне не хватает информации (город, даты или интересы).",
            "is_initial_plan_proposed": False,
            "collected_data": collected_data,
        }

    try:
        user_min_start_dt_naive = datetime.fromisoformat(parsed_dates_iso_list[0])
        user_explicitly_provided_time = not (
            user_min_start_dt_naive.hour == 0 and user_min_start_dt_naive.minute == 0
        )
        api_date_from_dt = user_min_start_dt_naive.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        api_date_to_dt = api_date_from_dt + timedelta(days=1)
        user_max_overall_end_dt_naive: Optional[datetime] = None
        if parsed_end_dates_iso_list and parsed_end_dates_iso_list[0]:
            user_max_overall_end_dt_naive = datetime.fromisoformat(
                parsed_end_dates_iso_list[0]
            )
            logger.info(
                f"User max overall end time for events: {user_max_overall_end_dt_naive.isoformat()}"
            )
    except Exception as e:
        logger.error(f"Error parsing dates in search_events_node: {e}", exc_info=True)
        return {
            "current_events": [],
            "status_message_to_user": "Некорректный формат дат.",
            "is_initial_plan_proposed": False,
            "collected_data": collected_data,
        }

    all_events_found_by_type_primary: Dict[str, List[Event]] = {}

    primary_search_stages_config = []
    if user_explicitly_provided_time:
        primary_search_stages_config.append(
            {
                "name": "User Specified Time",
                "min_start": user_min_start_dt_naive,
                "max_start": user_max_overall_end_dt_naive,
            }
        )
    else:
        default_evening_start_time = api_date_from_dt.replace(hour=17, minute=0)
        primary_search_stages_config.append(
            {
                "name": "Default Evening Time",
                "min_start": default_evening_start_time,
                "max_start": user_max_overall_end_dt_naive,
            }
        )

    async def _perform_search_for_interest_internal(
        interest_key: str,
        min_start_dt: Optional[datetime],
        max_start_dt: Optional[datetime],
    ) -> List[Event]:
        tool_args_internal = EventSearchToolArgs(city_id=city_id, date_from=api_date_from_dt, date_to=api_date_to_dt, interests_keys=[interest_key], min_start_time_naive=min_start_dt, max_start_time_naive=max_start_dt, max_budget_per_person=budget)  # type: ignore
        logger.info(
            f"Calling event_search_tool for '{interest_key}' with args: {tool_args_internal.model_dump_json(indent=2, exclude_none=True)}"
        )
        try:
            events_dict_list_internal: List[Dict] = await event_search_tool.ainvoke(
                tool_args_internal.model_dump(exclude_none=True)
            )
            return (
                [
                    Event(**evt_data_internal)
                    for evt_data_internal in events_dict_list_internal
                ]
                if events_dict_list_internal
                else []
            )
        except Exception as e_search_tool_internal:
            logger.error(
                f"Error in event_search_tool for '{interest_key}': {e_search_tool_internal}",
                exc_info=True,
            )
            return []

    for stage_config in primary_search_stages_config:
        logger.info(
            f"--- PRIMARY STAGE: {stage_config['name']} --- min_start: {stage_config['min_start']}, max_start: {stage_config['max_start']}"
        )
        primary_tasks, interests_for_primary_stage_tasks = [], []
        for interest_p in original_user_interests_keys:
            if not all_events_found_by_type_primary.get(interest_p):
                primary_tasks.append(
                    _perform_search_for_interest_internal(
                        interest_p, stage_config["min_start"], stage_config["max_start"]
                    )
                )
                interests_for_primary_stage_tasks.append(interest_p)

        if not primary_tasks:
            continue

        results_primary: List[List[Event]] = await asyncio.gather(*primary_tasks, return_exceptions=True)  # type: ignore
        for i, interest_key_res_p in enumerate(interests_for_primary_stage_tasks):
            result_p_item = results_primary[i]
            if isinstance(result_p_item, Exception):
                logger.error(
                    f"Exception for '{interest_key_res_p}' in primary stage '{stage_config['name']}': {result_p_item}"
                )
            elif result_p_item:
                logger.info(
                    f"Primary Stage '{stage_config['name']}': Found {len(result_p_item)} events for '{interest_key_res_p}'."
                )
                all_events_found_by_type_primary.setdefault(
                    interest_key_res_p, []
                ).extend(result_p_item)

    interests_needing_fallback_search = [
        key
        for key in original_user_interests_keys
        if not all_events_found_by_type_primary.get(key)
    ]

    if interests_needing_fallback_search:
        logger.info(
            f"Performing FALLBACK search (whole day) for: {interests_needing_fallback_search}"
        )
        fallback_search_tasks, fallback_interests_for_fb_tasks = [], []
        min_fb_start = api_date_from_dt

        for interest_key_fb_search in interests_needing_fallback_search:
            fallback_search_tasks.append(
                _perform_search_for_interest_internal(
                    interest_key_fb_search, min_fb_start, None
                )
            )
            fallback_interests_for_fb_tasks.append(interest_key_fb_search)

        if fallback_search_tasks:
            fallback_search_results: List[List[Event]] = await asyncio.gather(*fallback_search_tasks, return_exceptions=True)  # type: ignore
            for i, interest_key_fb_res_item_fb in enumerate(
                fallback_interests_for_fb_tasks
            ):
                result_fb_item_search = fallback_search_results[i]
                if isinstance(result_fb_item_search, Exception):
                    logger.error(
                        f"Exception in fallback for '{interest_key_fb_res_item_fb}': {result_fb_item_search}"
                    )
                elif result_fb_item_search:
                    result_fb_item_search.sort(
                        key=lambda e_fb: e_fb.start_time_naive_event_tz
                    )
                    collected_data.setdefault("fallback_candidates", {})[interest_key_fb_res_item_fb] = result_fb_item_search[0].model_dump()  # type: ignore
                    logger.info(
                        f"Fallback: Stored '{result_fb_item_search[0].name}' for '{interest_key_fb_res_item_fb}'."
                    )
                else:
                    logger.info(
                        f"Fallback search: No events found for '{interest_key_fb_res_item_fb}'."
                    )
                    collected_data.setdefault("not_found_interest_keys", []).append(interest_key_fb_res_item_fb)  # type: ignore

    # Сортировка результатов основного поиска
    for ik_sort_val in all_events_found_by_type_primary:
        all_events_found_by_type_primary[ik_sort_val].sort(
            key=lambda e_sort: e_sort.start_time_naive_event_tz
        )

    events_to_propose_list: List[Event] = []
    if len(original_user_interests_keys) == 1:
        interest_key_single_prop = original_user_interests_keys[0]
        candidates_single_prop = all_events_found_by_type_primary.get(
            interest_key_single_prop, []
        )
        if candidates_single_prop:
            first_event_cand_single_prop = candidates_single_prop[0]
            first_event_end_single_prop = (
                first_event_cand_single_prop.start_time_naive_event_tz
                + timedelta(
                    minutes=first_event_cand_single_prop.duration_minutes or 120
                )
            )
            if (
                not user_max_overall_end_dt_naive
                or first_event_end_single_prop <= user_max_overall_end_dt_naive
            ):
                events_to_propose_list.append(first_event_cand_single_prop)
                if len(candidates_single_prop) > 1:
                    second_event_cand_single_prop = candidates_single_prop[1]
                    is_compatible_single_prop, _ = await _check_event_compatibility(
                        events_to_propose_list[0],
                        second_event_cand_single_prop,
                        user_max_overall_end_dt_naive,
                    )
                    if is_compatible_single_prop:
                        events_to_propose_list.append(second_event_cand_single_prop)
            else:
                logger.debug(
                    f"Primary candidate {first_event_cand_single_prop.name} for single interest ends too late ({first_event_end_single_prop} vs {user_max_overall_end_dt_naive})."
                )
    else:
        best_cand_per_req_type_p_prop: Dict[str, Event] = {}
        for req_ik_p_prop in original_user_interests_keys:
            if all_events_found_by_type_primary.get(
                req_ik_p_prop
            ):  # Проверяем, что список не пуст
                candidate_event_multi = all_events_found_by_type_primary[req_ik_p_prop][
                    0
                ]
                event_end_time_multi = (
                    candidate_event_multi.start_time_naive_event_tz
                    + timedelta(minutes=candidate_event_multi.duration_minutes or 120)
                )
                if (
                    not user_max_overall_end_dt_naive
                    or event_end_time_multi <= user_max_overall_end_dt_naive
                ):
                    best_cand_per_req_type_p_prop[req_ik_p_prop] = candidate_event_multi
                else:
                    logger.debug(
                        f"Primary candidate {candidate_event_multi.name} for interest {req_ik_p_prop} ends too late ({event_end_time_multi} vs {user_max_overall_end_dt_naive})."
                    )

        sorted_best_cand_p_prop = sorted(
            best_cand_per_req_type_p_prop.values(),
            key=lambda e_p: e_p.start_time_naive_event_tz,
        )
        if sorted_best_cand_p_prop:
            events_to_propose_list.append(sorted_best_cand_p_prop[0])
            if len(sorted_best_cand_p_prop) > 1:
                for pot_second_ev_p_prop in sorted_best_cand_p_prop[1:]:
                    if (
                        pot_second_ev_p_prop.event_type_key
                        != events_to_propose_list[0].event_type_key
                    ):
                        is_comp_p_prop, _ = await _check_event_compatibility(
                            events_to_propose_list[0],
                            pot_second_ev_p_prop,
                            user_max_overall_end_dt_naive,
                        )
                        if is_comp_p_prop:
                            events_to_propose_list.append(pot_second_ev_p_prop)
                            break

        if (
            len(events_to_propose_list) == 1 and best_cand_per_req_type_p_prop
        ):  # Проверяем, что словарь не пуст
            first_ev_type_p_prop = events_to_propose_list[0].event_type_key
            other_cand_same_type_p_prop = [
                ev_p_prop
                for ev_p_prop in all_events_found_by_type_primary.get(
                    first_ev_type_p_prop, []
                )
                if ev_p_prop.session_id != events_to_propose_list[0].session_id
            ]
            if other_cand_same_type_p_prop:
                is_comp_same_p_prop, _ = await _check_event_compatibility(
                    events_to_propose_list[0],
                    other_cand_same_type_p_prop[0],
                    user_max_overall_end_dt_naive,
                )
                if is_comp_same_p_prop:
                    events_to_propose_list.append(other_cand_same_type_p_prop[0])

    if not events_to_propose_list and any(all_events_found_by_type_primary.values()):
        flat_list_primary_all_prop = [
            ev_f_prop
            for lst_f_prop in all_events_found_by_type_primary.values()
            for ev_f_prop in lst_f_prop
        ]
        if flat_list_primary_all_prop:
            flat_list_primary_all_prop.sort(
                key=lambda e_f_prop: e_f_prop.start_time_naive_event_tz
            )
            first_cand_f_prop = flat_list_primary_all_prop[0]
            first_cand_f_end_prop = (
                first_cand_f_prop.start_time_naive_event_tz
                + timedelta(minutes=first_cand_f_prop.duration_minutes or 120)
            )
            if (
                not user_max_overall_end_dt_naive
                or first_cand_f_end_prop <= user_max_overall_end_dt_naive
            ):
                events_to_propose_list.append(first_cand_f_prop)

    current_not_found_keys_final = []
    for key_original_nf in original_user_interests_keys:
        is_proposed_from_primary_search = any(
            event_prop_nf.event_type_key == key_original_nf
            for event_prop_nf in events_to_propose_list
        )
        has_fallback_candidate_nf = bool(collected_data.get("fallback_candidates", {}).get(key_original_nf))  # type: ignore
        if not is_proposed_from_primary_search and not has_fallback_candidate_nf:
            current_not_found_keys_final.append(key_original_nf)

    if current_not_found_keys_final:
        collected_data["not_found_interest_keys"] = current_not_found_keys_final  # type: ignore
    elif "not_found_interest_keys" in collected_data:
        del collected_data["not_found_interest_keys"]  # type: ignore

    if not events_to_propose_list and not collected_data.get("fallback_candidates"):
        logger.info(
            "No events for proposal from primary or fallback. All original interests considered not found."
        )
        if not any(all_events_found_by_type_primary.values()) and not collected_data.get("fallback_candidates"):  # type: ignore
            collected_data["not_found_interest_keys"] = list(original_user_interests_keys)  # type: ignore
        return {
            "current_events": [],
            "status_message_to_user": None,
            "is_initial_plan_proposed": False,
            "collected_data": collected_data,
        }

    logger.info(f"Proposing {len(events_to_propose_list)} events from primary. Fallback candidates for: {list(collected_data.get('fallback_candidates', {}).keys())}")  # type: ignore
    return {
        "current_events": events_to_propose_list,
        "status_message_to_user": None,
        "is_initial_plan_proposed": bool(events_to_propose_list),
        "collected_data": collected_data,
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
        logger.debug(
            f"First event {first_event.session_id} ends at {first_event_end_naive.strftime('%H:%M')} (after user's max {user_max_overall_end_dt_naive.strftime('%H:%M')})"
        )
        return False, "Первое мероприятие заканчивается слишком поздно."

    if second_event_candidate.start_time_naive_event_tz < first_event_end_naive:
        logger.debug(
            f"Second event candidate {second_event_candidate.session_id} starts before first event ends."
        )
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
                error_msg = (
                    route_result.get("message", "unknown error")
                    if route_result
                    else "no response"
                )
                logger.warning(
                    f"Route error between {first_event.session_id} and {second_event_candidate.session_id} (Error: {error_msg}). Assuming 30 min."
                )
        except Exception as e_route:
            logger.error(
                f"get_route exception for compatibility check: {e_route}", exc_info=True
            )

    arrival_at_second_event_naive = first_event_end_naive + timedelta(
        minutes=route_duration_minutes
    )
    buffer_time = timedelta(minutes=15)

    if (
        arrival_at_second_event_naive
        > second_event_candidate.start_time_naive_event_tz - buffer_time
    ):
        logger.debug(
            f"Candidate {second_event_candidate.session_id} ({second_event_candidate.name} at {second_event_candidate.start_time_naive_event_tz.strftime('%H:%M')}) "
            f"not suitable. Arrival: {arrival_at_second_event_naive.strftime('%H:%M')}, Need by: {(second_event_candidate.start_time_naive_event_tz - buffer_time).strftime('%H:%M')}."
        )
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
        logger.debug(
            f"Second event candidate {second_event_candidate.session_id} ends at {second_event_end_naive.strftime('%H:%M')} "
            f"(after user's max {user_max_overall_end_dt_naive.strftime('%H:%M')})."
        )
        return False, "Второе мероприятие заканчивается слишком поздно."

    return True, None


# --- Узел 4: Представление начального плана и запрос адреса/бюджета (ОБНОВЛЕННАЯ ВЕРСИЯ) ---
async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    current_events: List[Event] = state.get("current_events", [])
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))

    not_found_at_all_keys_list: List[str] = collected_data.get(
        "not_found_interest_keys", []
    )
    fallback_candidates_dict: Dict[str, Dict] = collected_data.get(
        "fallback_candidates", {}
    )
    original_user_interests_keys_list: List[str] = collected_data.get(
        "interests_keys_afisha", []
    )

    response_parts_list = []
    awaiting_fallback_conf_flag_node = False
    pending_fallback_event_for_state_dict: Optional[Dict] = None
    fallback_accepted_previously = collected_data.get(
        "fallback_accepted_and_plan_updated", False
    )

    if current_events:
        response_parts_list.append("Вот что я смог найти для вас в указанное время:")
        for i, event_item_node in enumerate(current_events):
            time_str_node = event_item_node.start_time_naive_event_tz.strftime("%H:%M")
            date_str_node = event_item_node.start_time_naive_event_tz.strftime(
                "%d.%m.%Y"
            )
            desc_item_node = f"{i+1}. {event_item_node.name} ({event_item_node.event_type_key}) в '{event_item_node.place_name}' ({event_item_node.place_address or 'Адрес не указан'}). Начало в {time_str_node} ({date_str_node})."
            if event_item_node.min_price is not None:
                desc_item_node += f" Цена от {event_item_node.min_price} руб."
            if event_item_node.duration_minutes:
                desc_item_node += f" Продолжительность ~{event_item_node.duration_minutes // 60}ч {event_item_node.duration_minutes % 60}м."
            response_parts_list.append(desc_item_node)

    interest_key_to_name_mapping = {
        "Movie": "фильмы",
        "Performance": "спектакли",
        "Concert": "концерты",
        "Exhibition": "выставки",
        "SportEvent": "спортивные события",
        "Excursion": "экскурсии",
        "Event": "другие события",
    }

    keys_without_primary_offer = [
        key_orig
        for key_orig in original_user_interests_keys_list
        if key_orig not in {ev_curr.event_type_key for ev_curr in current_events}
    ]

    if (
        keys_without_primary_offer
        and fallback_candidates_dict
        and not fallback_accepted_previously
    ):
        for key_unfulfilled_item in keys_without_primary_offer:
            if awaiting_fallback_conf_flag_node:
                break

            fallback_event_data_item_val = fallback_candidates_dict.get(
                key_unfulfilled_item
            )
            if fallback_event_data_item_val:
                try:
                    fb_event_item_val = Event(**fallback_event_data_item_val)
                    type_name_item_val = interest_key_to_name_mapping.get(
                        key_unfulfilled_item, key_unfulfilled_item
                    )
                    fb_time_str_item_val = (
                        fb_event_item_val.start_time_naive_event_tz.strftime("%H:%M")
                    )
                    fb_date_str_item_val = (
                        fb_event_item_val.start_time_naive_event_tz.strftime("%d.%m.%Y")
                    )

                    user_time_desc_fb = collected_data.get(
                        "raw_time_description_original", "указанное вами время"
                    )
                    parsed_dates_list_fb_val = collected_data.get("parsed_dates_iso", [])  # type: ignore

                    if (
                        not user_time_desc_fb
                        or user_time_desc_fb
                        == collected_data.get("dates_description_original")
                    ):
                        user_time_desc_fb = "вечернее время"
                    elif (
                        parsed_dates_list_fb_val
                        and parsed_dates_list_fb_val[0]
                        and user_time_desc_fb
                        and "по" not in user_time_desc_fb
                    ):
                        try:
                            user_time_val_from_iso = datetime.fromisoformat(
                                parsed_dates_list_fb_val[0]
                            ).strftime("%H:%M")
                            user_time_desc_fb = f"время около {user_time_val_from_iso}"
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not parse time from '{parsed_dates_list_fb_val[0]}' for fallback message, using raw: '{user_time_desc_fb}'"
                            )

                    fallback_msg_item_val = f"\nК сожалению, {type_name_item_val} на {user_time_desc_fb} не нашлось. "
                    fallback_msg_item_val += f"Однако, есть вариант на {fb_date_str_item_val} в {fb_time_str_item_val}: «{fb_event_item_val.name}»"
                    if fb_event_item_val.place_name:
                        fallback_msg_item_val += f" в «{fb_event_item_val.place_name}»"
                    if fb_event_item_val.min_price is not None:
                        fallback_msg_item_val += (
                            f" (цена от {fb_event_item_val.min_price} руб.)"
                        )
                    fallback_msg_item_val += ". Хотите добавить его в план? (да/нет)"
                    response_parts_list.append(fallback_msg_item_val)

                    awaiting_fallback_conf_flag_node = True
                    pending_fallback_event_for_state_dict = (
                        fb_event_item_val.model_dump()
                    )
                    break
                except ValidationError as ve_fb_val_pres_node:
                    logger.error(
                        f"Error validating fallback candidate for {key_unfulfilled_item} in present_node: {ve_fb_val_pres_node}"
                    )
                    if key_unfulfilled_item not in not_found_at_all_keys_list:
                        not_found_at_all_keys_list.append(key_unfulfilled_item)
            else:
                if key_unfulfilled_item not in not_found_at_all_keys_list:
                    not_found_at_all_keys_list.append(key_unfulfilled_item)

    if not_found_at_all_keys_list and not awaiting_fallback_conf_flag_node:
        not_found_names_final_list = [
            interest_key_to_name_mapping.get(key, key)
            for key in not_found_at_all_keys_list
        ]
        if not_found_names_final_list:
            if (
                not current_events and not fallback_candidates_dict
            ):  # Если вообще ничего не было найдено и нет других fallback
                response_parts_list = [
                    f"К сожалению, не удалось найти подходящие {', '.join(not_found_names_final_list)} по вашим критериям."
                ]
            elif (
                current_events or fallback_candidates_dict
            ):  # Если что-то нашли или есть другие fallback, но эти типы совсем не нашлись
                response_parts_list.append(
                    f"\nТакже не удалось найти подходящие {', '.join(not_found_names_final_list)} по вашим критериям."
                )

    if "not_found_interest_keys" in collected_data:
        del collected_data["not_found_interest_keys"]
    if "fallback_candidates" in collected_data:
        del collected_data["fallback_candidates"]

    if awaiting_fallback_conf_flag_node and pending_fallback_event_for_state_dict:
        collected_data["awaiting_fallback_confirmation"] = True
        collected_data["pending_fallback_event"] = pending_fallback_event_for_state_dict
    else:
        if "awaiting_fallback_confirmation" in collected_data:
            del collected_data["awaiting_fallback_confirmation"]
        if "pending_fallback_event" in collected_data:
            del collected_data["pending_fallback_event"]

    if collected_data.get(
        "fallback_accepted_and_plan_updated"
    ):  # Сброс после использования в этом узле
        del collected_data["fallback_accepted_and_plan_updated"]

    plan_text_final_node = "\n".join(filter(None, response_parts_list))
    questions_to_user_list_node = []

    if not awaiting_fallback_conf_flag_node:
        if "awaiting_address_input" in collected_data:
            del collected_data["awaiting_address_input"]
        if not collected_data.get(
            "user_start_address_original"
        ) and not collected_data.get("user_start_address_validated_coords"):
            if (
                current_events
            ):  # Запрашиваем адрес только если есть какие-то события в основном плане
                questions_to_user_list_node.append(
                    "Откуда вы планируете начать маршрут? Назовите, пожалуйста, адрес (улица и дом)."
                )
                collected_data["awaiting_address_input"] = True

        if (
            collected_data.get("budget_current_search") is None
            and collected_data.get("budget_original") is None
        ):
            questions_to_user_list_node.append(
                "Кстати, чтобы лучше подобрать варианты, уточните ваш примерный бюджет на одно мероприятие?"
            )

        if questions_to_user_list_node:
            if plan_text_final_node.strip():
                plan_text_final_node += "\n\n" + " ".join(questions_to_user_list_node)
            else:
                plan_text_final_node = " ".join(questions_to_user_list_node)
        elif current_events and plan_text_final_node.strip():
            plan_text_final_node += "\n\nКак вам такой предварительный план? Если что-то не подходит, скажите, попробуем изменить."
        elif (
            not current_events
            and not awaiting_fallback_conf_flag_node
            and not plan_text_final_node.strip()
        ):
            plan_text_final_node = "К сожалению, по вашему запросу ничего не найдено. Попробуем другие критерии?"

    final_msg = (
        plan_text_final_node.strip()
        if plan_text_final_node.strip()
        else "По вашему запросу ничего не найдено. Попробуем другие критерии?"
    )
    new_messages_list = state.get("messages", []) + [AIMessage(content=final_msg)]

    logger.debug(
        f"present_initial_plan_node: final collected_data: {str(collected_data)[:500]}"
    )
    return {
        "messages": new_messages_list,
        "status_message_to_user": final_msg,
        "collected_data": collected_data,
        "is_initial_plan_proposed": bool(current_events)
        and not awaiting_fallback_conf_flag_node,
        "awaiting_final_confirmation": False,
    }


# --- Узел 5: Обработка ответа на адрес ИЛИ построение маршрута, если адрес не нужен / уже есть ---
async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_address_or_build_route_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    current_events: List[Event] = state.get("current_events", [])  # type: ignore

    if not current_events:
        logger.warning("build_route_node: No current events for route.")
        return {
            "current_route_details": RouteDetails(
                status="error", error_message="Нет мероприятий."
            ),
            "is_full_plan_with_route_proposed": False,
        }

    user_start_address_str = collected_data.get("user_start_address_original")
    user_start_coords = collected_data.get("user_start_address_validated_coords")

    if (
        not user_start_coords
    ):  # Если адреса пользователя нет, маршрут не строим (или строим только между событиями, если их > 1)
        if len(current_events) > 1:
            logger.info(
                "User address not provided, will attempt route between events if possible."
            )
            # Для маршрута между событиями, start_point будет первым событием
            # Эта логика уже должна быть в RouteBuilderToolArgs или здесь
        else:  # Одно событие и нет адреса
            logger.info("One event and no user address, no route to build from user.")
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

    # Проверяем, все ли события в один день
    event_dates: Set[date] = {
        evt.start_time_naive_event_tz.date() for evt in current_events
    }
    multiple_days = len(event_dates) > 1

    logger.info(f"Route for events on multiple_days: {multiple_days}")

    all_route_segments: List[RouteSegment] = []
    total_duration_seconds_combined = 0
    total_distance_meters_combined = 0
    overall_route_status = "success"

    if multiple_days:
        if not user_start_coords:  # Нужен адрес пользователя для маршрутов в разные дни
            logger.warning("Multiple day events but no user start address for routing.")
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Для мероприятий в разные дни нужен ваш адрес отправления.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        user_start_location = LocationModel(lon=user_start_coords["lon"], lat=user_start_coords["lat"], address_string=user_start_address_str)  # type: ignore

        for i, event_obj in enumerate(current_events):
            event_location = None
            if (
                event_obj.place_coords_lon is not None
                and event_obj.place_coords_lat is not None
            ):
                event_location = LocationModel(
                    lon=event_obj.place_coords_lon,
                    lat=event_obj.place_coords_lat,
                    address_string=event_obj.place_address,
                )
            elif event_obj.place_address:
                coords = await get_coords_from_address(address=event_obj.place_address, city=collected_data.get("city_name", ""))  # type: ignore
                if coords:
                    event_location = LocationModel(
                        lon=coords[0],
                        lat=coords[1],
                        address_string=event_obj.place_address,
                    )

            if not event_location:
                logger.warning(
                    f"Could not get location for event {event_obj.name} for multi-day route."
                )
                segment_error = RouteSegment(
                    from_address=user_start_address_str or "Ваше местоположение",
                    to_address=event_obj.name,  # Используем имя события как to_address
                    segment_status="error",
                    segment_error_message=f"Не удалось определить координаты для '{event_obj.name}'",
                )
                all_route_segments.append(segment_error)
                overall_route_status = "partial_success"
                continue

            tool_args_segment = RouteBuilderToolArgs(
                start_point=user_start_location, event_points=[event_location]
            )
            logger.info(
                f"Building route from User to Event {i+1} ({event_obj.name}): {tool_args_segment.model_dump_json(exclude_none=True)}"
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
            except ValidationError:
                overall_route_status = "partial_success"
                all_route_segments.append(
                    RouteSegment(
                        from_address=user_start_address_str,
                        to_address=event_obj.place_address or event_obj.name,
                        segment_status="error",
                        segment_error_message="Ошибка данных маршрута",
                    )
                )

    else:  # Все события в один день - строим последовательный маршрут
        start_location_for_api: Optional[LocationModel] = None
        event_points_for_api: List[LocationModel] = []

        if user_start_coords:
            start_location_for_api = LocationModel(lon=user_start_coords["lon"], lat=user_start_coords["lat"], address_string=user_start_address_str)  # type: ignore
            target_events_for_route = current_events
        elif len(current_events) > 1:  # Нет адреса пользователя, но есть >1 события
            first_event = current_events[0]
            if (
                first_event.place_coords_lon is not None
                and first_event.place_coords_lat is not None
            ):
                start_location_for_api = LocationModel(
                    lon=first_event.place_coords_lon,
                    lat=first_event.place_coords_lat,
                    address_string=first_event.place_address,
                )
            # ... (геокодирование первого события, если нужно, как в вашем старом коде) ...
            target_events_for_route = current_events[1:]
        else:  # Одно событие, нет адреса пользователя - уже обработано выше
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

        if not start_location_for_api:  # Если стартовая точка не определилась
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Не удалось определить начальную точку маршрута.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        for event_obj in target_events_for_route:
            # ... (логика добавления event_obj в event_points_for_api с геокодингом, как в вашем старом коде) ...
            if (
                event_obj.place_coords_lon is not None
                and event_obj.place_coords_lat is not None
            ):
                event_points_for_api.append(
                    LocationModel(
                        lon=event_obj.place_coords_lon,
                        lat=event_obj.place_coords_lat,
                        address_string=event_obj.place_address,
                    )
                )
            # ... (else с геокодингом)

        if (
            not event_points_for_api and target_events_for_route
        ):  # Если есть к чему строить, но не смогли получить точки
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Не удалось определить координаты событий для маршрута.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        if event_points_for_api:  # Только если есть куда строить маршрут
            tool_args_single_day = RouteBuilderToolArgs(
                start_point=start_location_for_api, event_points=event_points_for_api
            )
            logger.info(
                f"Building single-day route: {tool_args_single_day.model_dump_json(exclude_none=True)}"
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
                overall_route_status = single_day_route_details.status
                if (
                    single_day_route_details.status != "success"
                    and single_day_route_details.error_message
                ):
                    logger.error(
                        f"Single-day route error: {single_day_route_details.error_message}"
                    )  # Доп. лог

            except ValidationError as ve:
                logger.error(f"Validation error for single-day route data: {ve}")
                overall_route_status = "error"
                all_route_segments.append(
                    RouteSegment(
                        segment_status="error",
                        segment_error_message="Ошибка данных маршрута",
                    )
                )
        elif (
            not event_points_for_api and user_start_coords and len(current_events) == 1
        ):  # Маршрут от пользователя до одного события
            # Эта логика покрывается multiple_days = False и target_events_for_route = current_events
            # если start_location_for_api это юзер, а event_points_for_api это одно событие.
            pass  # Уже должно быть обработано выше
        elif (
            not event_points_for_api
        ):  # Если вообще не к чему строить маршрут (например, одно событие без адреса пользователя)
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

    # Формируем итоговый RouteDetails
    final_route_details = RouteDetails(
        status=overall_route_status,
        segments=all_route_segments,
        total_duration_seconds=total_duration_seconds_combined,
        total_distance_meters=total_distance_meters_combined,
        total_duration_text=(
            f"~{round(total_duration_seconds_combined / 60)} мин"
            if total_duration_seconds_combined
            else None
        ),
        total_distance_text=(
            f"~{round(total_distance_meters_combined / 1000, 1)} км"
            if total_distance_meters_combined
            else None
        ),
        error_message=(
            "Одна или несколько частей маршрута не могли быть построены."
            if overall_route_status == "partial_success"
            else (
                "Ошибка построения маршрута."
                if overall_route_status == "error"
                else None
            )
        ),
    )

    return {
        "current_route_details": final_route_details,
        "is_full_plan_with_route_proposed": final_route_details.status
        in ["success", "partial_success"]
        and bool(final_route_details.segments),
    }


# --- Узел 6: Представление полного плана (мероприятия + маршрут) ---
async def present_full_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_full_plan_node executing...")
    current_events: List[Event] = state.get("current_events", [])  # type: ignore
    current_route_details_obj: Optional[RouteDetails] = state.get("current_route_details")  # type: ignore
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore

    # not_found_interest_keys и fallback_candidates уже должны быть обработаны present_initial_plan_node
    # или при обработке ответа на fallback. Здесь мы просто отображаем то, что есть в current_events и current_route_details.

    if not current_events:
        # Эта ситуация маловероятна, если граф работает правильно, т.к. до этого узла должны дойти только с событиями.
        logger.warning("present_full_plan_node: No current events to present.")
        return {
            "status_message_to_user": "Мероприятий для отображения нет. Пожалуйста, начните новый поиск."
        }

    response_parts = ["Вот ваш итоговый план:"]
    # ... (отображение мероприятий, как в последней вашей версии)
    for i, event in enumerate(current_events):
        event_time_str = event.start_time_naive_event_tz.strftime("%H:%M")
        event_date_str = event.start_time_naive_event_tz.strftime(
            "%d.%m.%Y (%A)"
        )  # Добавил день недели
        desc = f"\n{i+1}. **{event.name}** ({event.event_type_key})\n   *Место:* {event.place_name} ({event.place_address or 'Адрес не уточнен'})"
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
                from_name = segment.from_address or (
                    f"Ваше местоположение" if multiple_days else f"Точка {idx+1}"
                )
                to_name = (
                    segment.to_address
                    or f"Мероприятие {idx+1 if not multiple_days else ''}"
                )  # Уточнить имя события для to_name

                # Попытка получить имя события для точки назначения
                # Это упрощенно, т.к. сегменты могут быть не строго по одному на событие в общем случае
                # но для нашей логики (от юзера к каждому событию ИЛИ последовательно) должно работать
                target_event_for_segment: Optional[Event] = None
                if multiple_days and idx < len(current_events):
                    target_event_for_segment = current_events[idx]
                    to_name = f"«{target_event_for_segment.name}»"
                elif not multiple_days and idx < len(
                    current_events
                ):  # Если user_start_address, то idx = 0 это первое событие
                    if collected_data.get("user_start_address_original") and idx < len(
                        current_events
                    ):
                        target_event_for_segment = current_events[idx]
                        to_name = f"«{target_event_for_segment.name}»"
                    elif not collected_data.get(
                        "user_start_address_original"
                    ) and idx + 1 < len(
                        current_events
                    ):  # Маршрут между событиями
                        target_event_for_segment = current_events[idx + 1]
                        to_name = f"«{target_event_for_segment.name}»"

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
                and len(current_route_details_obj.segments) > 1
            ):  # Общее для однодневного последовательного
                response_parts.append(
                    f"\n  Общее время в пути по маршруту: {current_route_details_obj.total_duration_text}."
                )

        elif current_route_details_obj.status != "success":
            response_parts.append(
                f"\nМаршрут: Не удалось построить ({current_route_details_obj.error_message or 'причина неизвестна'})."
            )

    response_parts.append(
        "\n\nПлан окончательный. Если захотите что-то еще, просто напишите новый запрос!"
    )
    full_plan_text = "\n".join(response_parts)
    new_messages = state.get("messages", []) + [AIMessage(content=full_plan_text)]  # type: ignore

    # Очищаем состояние для возможного нового запроса от пользователя
    # (кроме messages, которые LangGraph обрабатывает)
    # Это лучше делать в условном ребре, ведущем к __END__, или в обработчике /start
    final_collected_data = {}  # Сбрасываем собранные данные для нового цикла

    return {
        "messages": new_messages,
        "status_message_to_user": full_plan_text,
        "collected_data": final_collected_data,  # Очищенные данные
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,  # План уже представлен как финальный
        "awaiting_final_confirmation": False,
        "awaiting_fallback_confirmation": False,
        "pending_fallback_event": None,
    }


# --- Узел 7: Обработка обратной связи по плану ---
async def handle_plan_feedback_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: handle_plan_feedback_node executing...")
    messages = state.get("messages", [])
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore
    current_events: Optional[List[Event]] = state.get("current_events")

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
    current_route_details: Optional[RouteDetails] = state.get("current_route_details")  # type: ignore
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
        next_state_update: Dict[str, Any] = {
            "awaiting_final_confirmation": False,
            "collected_data": collected_data,
            "current_events": list(current_events) if current_events else [],
        }

        if intent == "confirm_plan":
            logger.info("User confirmed the plan.")
            next_state_update["status_message_to_user"] = (
                "Отлично! Рад был помочь. Если понадобится что-то еще, обращайтесь!"
            )
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
        elif intent == "request_change" and changes:
            logger.info(f"User requested changes: {changes}")
            next_state_update["previous_confirmed_collected_data"] = dict(
                collected_data
            )
            next_state_update["previous_confirmed_events"] = (
                list(current_events) if current_events else []
            )
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None

            new_collected_data = dict(next_state_update["collected_data"])  # type: ignore
            new_collected_data["clarification_needed_fields"] = []
            change_target = changes.get("change_target")
            new_value = changes.get("new_value")

            if change_target == "budget":
                if isinstance(new_value, (int, float)):
                    new_collected_data["budget_current_search"] = int(new_value)
                    new_collected_data["budget_original"] = int(new_value)
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("budget_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите бюджет числом. Вы: '{new_value}'."
                    )
            elif change_target in ["date", "time"]:
                if isinstance(new_value, str):
                    parsed_date_res = await datetime_parser_tool.ainvoke(
                        {
                            "natural_language_date": new_value,
                            "base_date_iso": datetime.now().isoformat(),
                        }
                    )
                    if parsed_date_res.get("datetime_iso"):
                        new_collected_data["parsed_dates_iso"] = [
                            parsed_date_res["datetime_iso"]
                        ]
                        new_collected_data["dates_description_original"] = new_value
                        if parsed_date_res.get("end_datetime_iso"):
                            new_collected_data["parsed_end_dates_iso"] = [
                                parsed_date_res["end_datetime_iso"]
                            ]
                        elif "parsed_end_dates_iso" in new_collected_data:
                            del new_collected_data["parsed_end_dates_iso"]
                        if parsed_date_res.get("is_ambiguous"):
                            new_collected_data.setdefault(
                                "clarification_needed_fields", []
                            ).append("dates_description_original")
                            next_state_update["clarification_context"] = (
                                parsed_date_res.get("clarification_needed")
                            )
                    else:
                        new_collected_data.setdefault(
                            "clarification_needed_fields", []
                        ).append("dates_description_original")
                        next_state_update["clarification_context"] = (
                            f"Не удалось распознать '{new_value}'. Уточните."
                        )
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("dates_description_original")
                    next_state_update["clarification_context"] = (
                        f"Опишите дату/время текстом. Вы: '{new_value}'."
                    )
            elif change_target in ["interests", "type"] or "event_" in str(
                change_target
            ):
                # ... (логика изменения интересов как в прошлом ответе)
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
                    new_collected_data["interests_original"] = new_interests_str_list
                    mapped_keys = []
                    for s_int in new_interests_str_list:
                        s_l = s_int.lower()
                        key_afisha = None
                        if "фильм" in s_l or "кино" in s_l:
                            key_afisha = "Movie"
                        # ... другие маппинги ...
                        elif "театр" in s_l:
                            key_afisha = "Performance"
                        if not key_afisha:
                            key_afisha = s_int.capitalize()
                        mapped_keys.append(key_afisha)
                    new_collected_data["interests_keys_afisha"] = list(set(mapped_keys))
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("interests_original")
                    next_state_update["clarification_context"] = (
                        f"Назовите интересы. Вы: '{new_value}'."
                    )
            elif change_target == "start_location":
                if isinstance(new_value, str):
                    new_collected_data["user_start_address_original"] = new_value
                    new_collected_data["user_start_address_validated_coords"] = None
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("user_start_address_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите адрес. Вы: '{new_value}'."
                    )
            else:
                logger.warning(f"Unknown change_target: {change_target}")
                next_state_update["clarification_context"] = "Не понял, что изменить."
            next_state_update["collected_data"] = new_collected_data
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
        else:  # clarify_misunderstanding или неизвестное
            next_state_update["status_message_to_user"] = (
                "Я вас не совсем понял. Попробуете еще раз или новые критерии?"
            )
            next_state_update["awaiting_final_confirmation"] = (
                True  # Остаемся на подтверждении текущего плана
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
        msg = "Ошибка при обработке ответа. Попробуйте еще раз."
        return {
            "status_message_to_user": msg,
            "awaiting_final_confirmation": True,
            "messages": messages + [AIMessage(content=msg)],
            "collected_data": collected_data,
        }


# --- Узел 8: Подтверждение изменений (для Примера 2 из инструкции) ---
async def confirm_changes_node(
    state: AgentState,
) -> Dict[str, Any]:  # Вероятно, не будет использоваться активно
    logger.info("Node: confirm_changes_node executing (likely deprecated)...")
    messages = state.get("messages", [])
    pending_modification = state.get("pending_plan_modification_request", {})
    if not pending_modification:
        return {
            "status_message_to_user": "Что-то пошло не так.",
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
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore
    not_found_keys: List[str] = collected_data.get("not_found_interest_keys", [])  # type: ignore
    original_interests: List[str] = collected_data.get("interests_original", [])  # type: ignore
    search_criteria_parts = []
    if collected_data.get("city_name"):
        search_criteria_parts.append(f"город '{collected_data['city_name']}'")
    if collected_data.get("dates_description_original"):
        search_criteria_parts.append(
            f"даты '{collected_data['dates_description_original']}'"
        )

    key_to_name = {
        "Movie": "фильмы",
        "Performance": "спектакли",
        "Concert": "концерты",
        "Exhibition": "выставки",
        "SportEvent": "спорт",
        "Excursion": "экскурсии",
        "Event": "события",
    }
    if not_found_keys:
        names = [key_to_name.get(k, k) for k in not_found_keys]
        if len(not_found_keys) == len(collected_data.get("interests_keys_afisha", [])):
            search_criteria_parts.append(f"интересы '{', '.join(original_interests)}'")
        else:
            search_criteria_parts.append(
                f"интересы '{', '.join(original_interests)}' (не найдено: {', '.join(names)})"
            )
    elif original_interests:
        search_criteria_parts.append(f"интересы '{', '.join(original_interests)}'")

    search_criteria_summary = (
        ", ".join(search_criteria_parts)
        if search_criteria_parts
        else "указанным вами критериям"
    )
    error_msg = EVENT_NOT_FOUND_PROMPT_TEMPLATE.format(
        search_criteria_summary=search_criteria_summary
    )

    if "not_found_interest_keys" in collected_data:
        del collected_data["not_found_interest_keys"]  # type: ignore
    if "fallback_candidates" in collected_data:
        del collected_data["fallback_candidates"]  # type: ignore

    new_messages = state.get("messages", []) + [AIMessage(content=error_msg)]  # type: ignore
    return {
        "messages": new_messages,
        "status_message_to_user": error_msg,
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,
        "collected_data": collected_data,
    }
