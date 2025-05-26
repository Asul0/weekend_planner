import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from datetime import datetime, timedelta
import aiohttp
from services.gis_service import (
    get_coords_from_address,
    get_route,
)  # Добавлен get_route

# Pydantic и Langchain для сообщений и схем
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)  # <--- ДОБАВЛЕНЫ BaseModel и Field
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
    ParsedDateTime,  # Если выносили в schemas
    AnalyzedFeedback,  # Импортируем из schemas
)
from services.afisha_service import fetch_cities_internal
from services.gis_service import get_coords_from_address, get_route
from tools.datetime_parser_tool import datetime_parser_tool
from tools.event_search_tool import event_search_tool
from tools.route_builder_tool import route_builder_tool

# Инициализация логгера
logger = logging.getLogger(__name__)

# Pydantic модель для структурированного ответа от LLM при анализе обратной связи
# (Она была определена внутри узла, но лучше вынести на уровень модуля или в schemas, если будет переиспользоваться)
# Пока оставим здесь для контекста, но если она используется только в одном узле, это нормально.
# Если выносили в schemas.data_schemas.py, то здесь этот класс не нужен, а нужен импорт.
# Судя по предыдущему коду, я ее определил локально в узле handle_plan_feedback_node.
# Давайте для чистоты вынесем ее в schemas.data_schemas.py и импортируем оттуда.
# Я добавлю ее в schemas/data_schemas.py в следующем шаге, а здесь пока закомментирую.
# class AnalyzedFeedback(BaseModel):
#     intent_type: str = Field(description="Тип намерения пользователя: 'confirm_plan', 'request_change', 'clarify_misunderstanding', 'new_search'.")
#     change_details: Optional[Dict[str, Any]] = Field(None, description="Словарь с деталями запрошенного изменения...")
# Вместо этого импортируем:
from schemas.data_schemas import (
    AnalyzedFeedback,
)  # Предполагаем, что мы ее туда перенесли


# ... (далее идет код ваших узлов, начиная с extract_initial_info_node) ...


logger = logging.getLogger(__name__)


# --- Узел 1: Извлечение начальной информации ---
async def extract_initial_info_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: extract_initial_info_node executing...")
    messages = state.get("messages", [])
    current_collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.debug(
            "extract_initial_info_node: No new HumanMessage. Returning current state."
        )
        return {
            "collected_data": current_collected_data,
            "messages": messages,
            "clarification_context": state.get("clarification_context"),
        }

    user_query = messages[-1].content
    is_reply_to_address_request = current_collected_data.get(
        "awaiting_address_input", False
    )

    logger.info(
        f"extract_initial_info_node: Processing HumanMessage: '{user_query}'. Is reply to address request: {is_reply_to_address_request}"
    )
    logger.debug(
        f"extract_initial_info_node: Initial collected_data state: {current_collected_data}"
    )

    current_clarification_context = None
    current_collected_data["clarification_needed_fields"] = []  # type: ignore

    if is_reply_to_address_request:
        # ... (блок обработки адреса остается без изменений, как в твоей последней версии nodes.txt)
        logger.info(f"Input '{user_query}' is being treated as an address.")
        city_for_geocoding = current_collected_data.get("city_name")
        if not city_for_geocoding:
            logger.error(
                "extract_initial_info_node: City name missing while expecting address."
            )
            current_collected_data.setdefault("clarification_needed_fields", []).append("city_name")  # type: ignore
            if "awaiting_address_input" in current_collected_data:
                del current_collected_data["awaiting_address_input"]  # type: ignore
        else:
            coords = await get_coords_from_address(
                address=user_query, city=city_for_geocoding
            )
            if coords:
                current_collected_data["user_start_address_original"] = user_query
                current_collected_data["user_start_address_validated_coords"] = {
                    "lon": coords[0],
                    "lat": coords[1],
                }
                logger.info(
                    f"Address '{user_query}' geocoded to {coords} in city '{city_for_geocoding}'."
                )
                if "awaiting_address_input" in current_collected_data:
                    del current_collected_data["awaiting_address_input"]  # type: ignore
            else:
                logger.warning(
                    f"Could not geocode address '{user_query}' in city '{city_for_geocoding}'."
                )
                current_collected_data.setdefault("clarification_needed_fields", []).append("user_start_address_original")  # type: ignore
                current_clarification_context = "Не удалось распознать этот адрес. Пожалуйста, попробуйте ввести его еще раз (улица и номер дома) или укажите другой."
                current_collected_data["awaiting_address_input"] = True  # type: ignore
        logger.debug(
            f"extract_initial_info_node: collected_data after address processing: {current_collected_data}"
        )
        return {
            "collected_data": current_collected_data,
            "messages": messages,
            "clarification_context": current_clarification_context,
        }

    # Если НЕ ответ на запрос адреса, то полный разбор LLM
    prev_city_name = current_collected_data.get("city_name")
    prev_city_id_afisha = current_collected_data.get("city_id_afisha")
    prev_dates_desc = current_collected_data.get("dates_description_original")
    prev_parsed_dates = current_collected_data.get("parsed_dates_iso")
    prev_parsed_end_dates = current_collected_data.get(
        "parsed_end_dates_iso"
    )  # Сохраняем предыдущее конечное время
    prev_interests_orig = current_collected_data.get("interests_original")
    prev_interests_keys = current_collected_data.get("interests_keys_afisha")
    prev_raw_time_desc = current_collected_data.get("raw_time_description_original")

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

        # Город (логика без изменений)
        if extracted_info.city:
            current_collected_data["city_name"] = extracted_info.city
            cities_from_afisha = await fetch_cities_internal()
            found_city_afisha = next((c for c in cities_from_afisha if extracted_info.city.lower() in c["name_lower"]), None)  # type: ignore
            if found_city_afisha:
                current_collected_data["city_id_afisha"] = found_city_afisha["id"]  # type: ignore
            else:
                current_collected_data["city_id_afisha"] = None  # type: ignore
                if "city_name" not in current_collected_data.get(
                    "clarification_needed_fields", []
                ):
                    current_collected_data.setdefault("clarification_needed_fields", []).append("city_name")  # type: ignore
        elif prev_city_name:
            current_collected_data["city_name"] = prev_city_name
            current_collected_data["city_id_afisha"] = prev_city_id_afisha  # type: ignore
        elif not current_collected_data.get("city_name"):
            if "city_name" not in current_collected_data.get(
                "clarification_needed_fields", []
            ):
                current_collected_data.setdefault("clarification_needed_fields", []).append("city_name")  # type: ignore

        # Интересы (логика без изменений)
        if extracted_info.interests:
            current_collected_data["interests_original"] = extracted_info.interests  # type: ignore
            mapped_interest_keys = []
            for interest_str in extracted_info.interests:  # type: ignore
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
                if not key:
                    key = interest_str.capitalize()
                mapped_interest_keys.append(key)
            current_collected_data["interests_keys_afisha"] = list(set(mapped_interest_keys))  # type: ignore
        elif prev_interests_orig:
            current_collected_data["interests_original"] = prev_interests_orig  # type: ignore
            current_collected_data["interests_keys_afisha"] = prev_interests_keys  # type: ignore
        elif not current_collected_data.get("interests_keys_afisha"):
            if "interests_original" not in current_collected_data.get(
                "clarification_needed_fields", []
            ):
                current_collected_data.setdefault("clarification_needed_fields", []).append("interests_original")  # type: ignore

        # Бюджет (логика без изменений)
        if extracted_info.budget is not None:
            current_collected_data["budget_original"] = extracted_info.budget  # type: ignore
            current_collected_data["budget_current_search"] = extracted_info.budget  # type: ignore

        # Даты и Время
        dates_desc_from_llm = extracted_info.dates_description
        time_qualifier_from_llm = extracted_info.raw_time_description

        if dates_desc_from_llm or time_qualifier_from_llm:
            if dates_desc_from_llm:
                current_collected_data["dates_description_original"] = dates_desc_from_llm  # type: ignore
            if time_qualifier_from_llm:
                current_collected_data["raw_time_description_original"] = time_qualifier_from_llm  # type: ignore

            natural_date_for_parser = (
                dates_desc_from_llm if dates_desc_from_llm else "сегодня"
            )
            if (
                not dates_desc_from_llm and time_qualifier_from_llm and prev_dates_desc
            ):  # Если есть только уточнение времени и была предыдущая дата
                natural_date_for_parser = prev_dates_desc  # type: ignore
            elif (
                not dates_desc_from_llm and time_qualifier_from_llm
            ):  # Если есть только уточнение времени и НЕ было предыдущей даты
                current_collected_data["dates_description_original"] = "сегодня"  # type: ignore

            current_iso_for_parser = datetime.now().isoformat()
            parsed_date_time_result = await datetime_parser_tool.ainvoke(
                {
                    "natural_language_date": natural_date_for_parser,  # type: ignore
                    "natural_language_time_qualifier": time_qualifier_from_llm,
                    "base_date_iso": current_iso_for_parser,
                }
            )

            if parsed_date_time_result.get("datetime_iso"):
                current_collected_data["parsed_dates_iso"] = [parsed_date_time_result["datetime_iso"]]  # type: ignore

                if parsed_date_time_result.get(
                    "end_datetime_iso"
                ):  # Сохраняем конечное время
                    current_collected_data["parsed_end_dates_iso"] = [parsed_date_time_result["end_datetime_iso"]]  # type: ignore
                elif (
                    "parsed_end_dates_iso" in current_collected_data
                ):  # Очищаем, если его нет в новом результате
                    del current_collected_data["parsed_end_dates_iso"]  # type: ignore

                if parsed_date_time_result.get("is_ambiguous"):
                    if "dates_description_original" not in current_collected_data.get("clarification_needed_fields", []):  # type: ignore
                        current_collected_data.setdefault("clarification_needed_fields", []).append("dates_description_original")  # type: ignore
                    current_clarification_context = parsed_date_time_result.get(
                        "clarification_needed"
                    )
            else:
                if "dates_description_original" not in current_collected_data.get("clarification_needed_fields", []):  # type: ignore
                    current_collected_data.setdefault("clarification_needed_fields", []).append("dates_description_original")  # type: ignore
                current_clarification_context = (
                    parsed_date_time_result.get("clarification_needed")
                    or parsed_date_time_result.get("error_message")
                    or "Не удалось распознать дату или время из вашего запроса."
                )

        elif prev_parsed_dates:
            current_collected_data["dates_description_original"] = prev_dates_desc  # type: ignore
            current_collected_data["parsed_dates_iso"] = prev_parsed_dates  # type: ignore
            if prev_parsed_end_dates:
                current_collected_data["parsed_end_dates_iso"] = prev_parsed_end_dates  # type: ignore
            if prev_raw_time_desc:
                current_collected_data["raw_time_description_original"] = prev_raw_time_desc  # type: ignore
        elif not current_collected_data.get("parsed_dates_iso"):
            if "dates_description_original" not in current_collected_data.get("clarification_needed_fields", []):  # type: ignore
                current_collected_data.setdefault("clarification_needed_fields", []).append("dates_description_original")  # type: ignore

    except Exception as e:
        logger.error(
            f"extract_initial_info_node: Critical error during LLM call or info processing: {e}",
            exc_info=True,
        )
        for f_key in ["city_name", "dates_description_original", "interests_original"]:
            if not current_collected_data.get(f_key):  # type: ignore
                if f_key not in current_collected_data.get("clarification_needed_fields", []):  # type: ignore
                    current_collected_data.setdefault("clarification_needed_fields", []).append(f_key)  # type: ignore

    if "clarification_needed_fields" in current_collected_data and current_collected_data.get("clarification_needed_fields"):  # type: ignore
        fields_list = current_collected_data["clarification_needed_fields"]  # type: ignore
        unique_fields = []
        for f_item in fields_list:  # type: ignore
            if f_item and f_item not in unique_fields:
                unique_fields.append(f_item)
        current_collected_data["clarification_needed_fields"] = unique_fields  # type: ignore
        if not current_collected_data.get("clarification_needed_fields"):  # type: ignore
            del current_collected_data["clarification_needed_fields"]  # type: ignore

    logger.info(
        f"extract_initial_info_node: Final collected_data after extraction: {str(current_collected_data)[:500]}"
    )
    logger.info(
        f"extract_initial_info_node: Clarification needed for after extraction: {current_collected_data.get('clarification_needed_fields')}"
    )
    if current_clarification_context:
        logger.info(
            f"extract_initial_info_node: Clarification context set to: {current_clarification_context}"
        )

    return {
        "collected_data": current_collected_data,
        "messages": messages,
        "clarification_context": current_clarification_context,
    }


# --- Узел 2: Уточнение недостающих данных ---
async def clarify_missing_data_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_missing_data_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    clarification_fields = collected_data.get("clarification_needed_fields", [])

    status_message_to_user: Optional[str] = None

    if not clarification_fields:  # Если список пуст или None
        logger.info(
            "clarify_missing_data_node: No fields need explicit clarification based on clarification_needed_fields."
        )
        return {"status_message_to_user": None, "awaiting_final_confirmation": False}

    missing_critical_fields_map = {
        "city_name": "город для поиска",
        "dates_description_original": "даты или период мероприятий",
        "interests_original": "ваши интересы или тип мероприятий",
    }

    raw_time_desc_original = collected_data.get("raw_time_description_original")
    clarification_context_from_state = state.get("clarification_context")

    # Приоритет на уточнение времени, если оно было явно нечетким
    # и dates_description_original помечен для уточнения
    if raw_time_desc_original and "dates_description_original" in clarification_fields:
        current_date_info = date.today().strftime("%d %B %Y года (%A)")
        prompt_for_llm = TIME_CLARIFICATION_PROMPT_TEMPLATE.format(
            raw_time_description=raw_time_desc_original,
            current_date_info=current_date_info,
        )
        logger.debug(
            f"clarify_missing_data_node: Using TIME_CLARIFICATION_PROMPT for '{raw_time_desc_original}'"
        )
    # Если есть контекст от парсера дат (например, он вернул is_ambiguous=True и clarification_needed)
    elif (
        "dates_description_original" in clarification_fields
        and isinstance(clarification_context_from_state, str)
        and clarification_context_from_state
    ):
        prompt_for_llm = (
            clarification_context_from_state  # Используем готовый вопрос от парсера
        )
        logger.debug(
            f"clarify_missing_data_node: Using date parser's clarification_context: '{prompt_for_llm}'"
        )
    else:  # Общий случай для критических данных
        fields_to_ask_user_text_parts = []
        for field_key in [
            "city_name",
            "dates_description_original",
            "interests_original",
        ]:
            if field_key in clarification_fields:
                fields_to_ask_user_text_parts.append(
                    missing_critical_fields_map[field_key]
                )

        if not fields_to_ask_user_text_parts:
            logger.warning(
                "clarify_missing_data_node: Clarification needed but no specific critical fields identified for general prompt."
            )
            # Это маловероятно, если clarification_fields не пуст и не было спец. обработки времени
            status_message_to_user = "Кажется, мне нужно немного больше информации. Не могли бы вы уточнить ваш запрос?"
        else:
            missing_fields_text_for_prompt = " и ".join(fields_to_ask_user_text_parts)
            user_query_for_prompt = "Ваш запрос."  # Значение по умолчанию
            messages_history = state.get("messages", [])
            if messages_history:
                last_message = messages_history[-1]
                # Проверяем тип сообщения, чтобы убедиться, что это HumanMessage
                if isinstance(
                    last_message, HumanMessage
                ):  # или last_message.type == "human"
                    user_query_for_prompt = last_message.content

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
                f"clarify_missing_data_node: Using GENERAL_CLARIFICATION_PROMPT for: {missing_fields_text_for_prompt}"
            )

    if (
        not status_message_to_user and "prompt_for_llm" in locals()
    ):  # Если сообщение еще не установлено и промпт был сформирован
        llm = get_gigachat_client()
        try:
            ai_response = await llm.ainvoke(prompt_for_llm)
            status_message_to_user = ai_response.content
            logger.info(
                f"clarify_missing_data_node: LLM clarification question: {status_message_to_user}"
            )
        except Exception as e:
            logger.error(
                f"clarify_missing_data_node: Error during LLM call for clarification: {e}",
                exc_info=True,
            )
            status_message_to_user = "Произошла ошибка при попытке уточнить ваш запрос. Пожалуйста, попробуйте сформулировать его иначе."
    elif (
        not status_message_to_user
    ):  # Если clarification_fields есть, но ни один из кейсов не сработал
        logger.error(
            "clarify_missing_data_node: Unhandled clarification scenario. Clarification_fields: {clarification_fields}"
        )
        status_message_to_user = "Мне нужно несколько уточнений. Не могли бы вы переформулировать ваш запрос?"

    # Очищаем флаги уточнений, так как мы только что задали вопрос
    # Новый ответ пользователя будет заново проанализирован extract_initial_info_node
    # collected_data["clarification_needed_fields"] = [] # Это лучше делать в extract_initial_info_node

    # Возвращаем сообщение для пользователя и AIMessage в историю
    new_messages = state.get("messages", []) + [AIMessage(content=status_message_to_user if status_message_to_user else "")]  # type: ignore

    return {
        "messages": new_messages,
        "status_message_to_user": status_message_to_user,
        "awaiting_final_confirmation": False,
    }


# ... (предыдущий код extract_initial_info_node и clarify_missing_data_node остается выше) ...
# --- Импорты, которые могут понадобиться дополнительно для этой части ---
# from schemas.data_schemas import Event # Уже импортирован
from tools.event_search_tool import event_search_tool  # Наш инструмент поиска


async def get_route_duration(
    from_lon: float,
    from_lat: float,
    to_lon: float,
    to_lat: float,
    transport_type: str = "public_transport",
    start_time: datetime = None,
) -> dict:
    """
    Получает длительность маршрута между двумя точками через 2ГИС (или твой API).
    Возвращает словарь: {'duration_minutes': int, ...}
    """
    logger = logging.getLogger(__name__)

    # Пример реального запроса к API 2ГИС (замени на свои реальные данные)
    GIS_API_KEY = "ТВОЙ_API_КЛЮЧ_2ГИС"  # Лучше вынести в env/config
    base_url = "https://catalog.api.2gis.com/transport/route"
    params = {
        "key": GIS_API_KEY,
        "points": f"{from_lon},{from_lat},{to_lon},{to_lat}",
        "type": transport_type,
    }
    # Если start_time нужен — добавь обработку в params (зависит от API)
    if start_time:
        params["departure_time"] = start_time.isoformat()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=15) as response:
                if response.status != 200:
                    logger.warning(f"2GIS API returned status {response.status}")
                    raise Exception("Ошибка 2ГИС API")
                result = await response.json()
                # Разбор ответа (см. структуру API, ниже пример):
                # Допусти, ответ такой: {'result': {'routes': [{'total_duration': 34}, ...]}}
                route = result.get("result", {}).get("routes", [{}])[0]
                duration_minutes = route.get(
                    "total_duration", 15
                )  # по дефолту 15 мин если нет данных
                return {"duration_minutes": duration_minutes}
    except Exception as ex:
        logger.error(f"Ошибка запроса к 2ГИС: {ex}")
        # На случай ошибки возвращаем дефолтное время — можно менять
        return {"duration_minutes": 15}


# --- Узел 3: Поиск мероприятий ---
async def search_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: search_events_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})

    city_id = collected_data.get("city_id_afisha")
    parsed_dates_iso_list = collected_data.get("parsed_dates_iso")
    parsed_end_dates_iso_list = collected_data.get("parsed_end_dates_iso")
    interests_keys = collected_data.get(
        "interests_keys_afisha", collected_data.get("interests_original")
    )
    budget = collected_data.get("budget_current_search")

    if not city_id or not parsed_dates_iso_list or not interests_keys:
        logger.error(
            f"search_events_node: Missing critical data. CityID: {city_id}, Dates: {parsed_dates_iso_list}, Interests: {interests_keys}"
        )
        return {
            "current_events": [],
            "status_message_to_user": "Мне не хватает информации для поиска (город, даты или интересы). Пожалуйста, уточните.",
            "is_initial_plan_proposed": False,
        }

    try:
        date_from_dt_extracted = datetime.fromisoformat(parsed_dates_iso_list[0])
        user_explicitly_provided_time = not (
            date_from_dt_extracted.hour == 0
            and date_from_dt_extracted.minute == 0
            and date_from_dt_extracted.second == 0
            and date_from_dt_extracted.microsecond == 0
        )
        api_date_from = date_from_dt_extracted.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        api_date_to_date_obj = api_date_from.date() + timedelta(days=1)
        api_date_to = datetime(
            api_date_to_date_obj.year,
            api_date_to_date_obj.month,
            api_date_to_date_obj.day,
            0,
            0,
            0,
        )

        max_start_dt_extracted: Optional[datetime] = None
        if parsed_end_dates_iso_list and parsed_end_dates_iso_list[0]:
            try:
                max_start_dt_extracted = datetime.fromisoformat(
                    parsed_end_dates_iso_list[0]
                )
                logger.info(
                    f"search_events_node: Max start time extracted: {max_start_dt_extracted.isoformat()}"
                )
            except (ValueError, TypeError) as e_max_date:
                logger.warning(
                    f"search_events_node: Could not parse parsed_end_dates_iso: {parsed_end_dates_iso_list[0]}. Error: {e_max_date}"
                )

        logger.info(
            f"search_events_node: Date from (extracted with time): {date_from_dt_extracted.isoformat()}"
        )
        if max_start_dt_extracted:
            logger.info(
                f"search_events_node: Max start time for search (extracted): {max_start_dt_extracted.isoformat()}"
            )
        logger.info(
            f"search_events_node: User explicitly provided time (hour/minute != 00:00): {user_explicitly_provided_time}"
        )
        logger.info(
            f"search_events_node: API Date From (for Afisha query): {api_date_from.isoformat()}"
        )
        logger.info(
            f"search_events_node: API Date To (for Afisha query): {api_date_to.isoformat()}"
        )

    except (ValueError, IndexError, TypeError) as e:
        logger.error(
            f"search_events_node: Error parsing dates from ISO strings {parsed_dates_iso_list}: {e}",
            exc_info=True,
        )
        return {
            "current_events": [],
            "status_message_to_user": "Не удалось распознать даты для поиска. Пожалуйста, уточните формат дат.",
            "is_initial_plan_proposed": False,
        }

    found_events_obj_list: List[Event] = []

    async def _perform_search_internal(
        search_min_start_time_naive: Optional[datetime],
        search_max_start_time_naive: Optional[datetime],
    ) -> List[Event]:
        tool_args = EventSearchToolArgs(
            city_id=city_id,
            date_from=api_date_from,
            date_to=api_date_to,
            interests_keys=interests_keys,
            min_start_time_naive=search_min_start_time_naive,
            max_start_time_naive=search_max_start_time_naive,
            max_budget_per_person=budget,
            exclude_session_ids=None,
        )
        logger.info(
            f"search_events_node (_perform_search_internal): Calling event_search_tool with args: {tool_args.model_dump_json(indent=2, exclude_none=True)}"
        )
        try:
            events_dict_list: List[Dict] = await event_search_tool.ainvoke(
                tool_args.model_dump(exclude_none=True)
            )
            internal_list: List[Event] = []
            if events_dict_list:
                logger.info(
                    f"search_events_node (_perform_search_internal): Received {len(events_dict_list)} raw events from tool."
                )
                for evt_data in events_dict_list:
                    try:
                        internal_list.append(Event(**evt_data))
                    except Exception as val_err:
                        logger.warning(
                            f"search_events_node (_perform_search_internal): Validation error for event data: {evt_data.get('session_id', 'Unknown_ID')}. Error: {val_err}"
                        )
            else:
                logger.info(
                    "search_events_node (_perform_search_internal): Received 0 raw events from tool."
                )
            return internal_list
        except Exception as e_search:
            logger.error(
                f"search_events_node (_perform_search_internal): Error calling event_search_tool: {e_search}",
                exc_info=True,
            )
            return []

    if user_explicitly_provided_time:
        logger.info(
            f"search_events_node: Stage 1 - User specified time. Searching with min_start_time_naive={date_from_dt_extracted.isoformat()}"
            + (
                f" and max_start_time_naive={max_start_dt_extracted.isoformat()}"
                if max_start_dt_extracted
                else ""
            )
        )
        found_events_obj_list = await _perform_search_internal(
            search_min_start_time_naive=date_from_dt_extracted,
            search_max_start_time_naive=max_start_dt_extracted,
        )
        logger.info(
            f"search_events_node: Stage 1 - Found {len(found_events_obj_list)} events."
        )

    if not user_explicitly_provided_time and not found_events_obj_list:
        logger.info(
            "search_events_node: Stage 2 - User did not specify time. Attempting default evening (17:00)."
        )
        default_evening_start_time = api_date_from.replace(hour=17, minute=0)
        found_events_obj_list = await _perform_search_internal(
            search_min_start_time_naive=default_evening_start_time,
            search_max_start_time_naive=max_start_dt_extracted,  # Если пользователь сказал "вечером до 20", то max_start_dt_extracted будет, иначе None
        )
        logger.info(
            f"search_events_node: Stage 2 - Found {len(found_events_obj_list)} events."
        )

    if not found_events_obj_list:
        logger.info(
            "search_events_node: Stage 3 - Still no events (or skipped previous stages). Attempting search for the whole day."
        )
        found_events_obj_list = await _perform_search_internal(
            search_min_start_time_naive=None, search_max_start_time_naive=None
        )
        logger.info(
            f"search_events_node: Stage 3 - Found {len(found_events_obj_list)} events."
        )

    if not found_events_obj_list:
        logger.info("search_events_node: No events found by tool after all attempts.")
        return {
            "current_events": [],
            "status_message_to_user": None,
            "is_initial_plan_proposed": False,
        }

    first_event = found_events_obj_list[0]
    events_to_propose = [first_event]

    if len(found_events_obj_list) > 1:
        first_event_end_naive = first_event.start_time_naive_event_tz
        if first_event.duration_minutes:
            first_event_end_naive += timedelta(minutes=first_event.duration_minutes)
        else:
            first_event_end_naive += timedelta(hours=2)

        for candidate_event in found_events_obj_list[1:]:
            if not (
                first_event.place_coords_lon
                and first_event.place_coords_lat
                and candidate_event.place_coords_lon
                and candidate_event.place_coords_lat
            ):
                logger.warning(
                    f"search_events_node: Skipping candidate {candidate_event.session_id} due to missing coordinates for route calculation."
                )
                continue
            try:
                route_result = await get_route(
                    points=[
                        {
                            "lon": first_event.place_coords_lon,
                            "lat": first_event.place_coords_lat,
                        },
                        {
                            "lon": candidate_event.place_coords_lon,
                            "lat": candidate_event.place_coords_lat,
                        },
                    ],
                    transport="driving",
                )
                route_duration_seconds = 0
                if route_result and route_result.get("status") == "success":
                    route_duration_seconds = route_result.get("duration_seconds", 0)
                else:
                    error_msg = (
                        route_result.get("message", "unknown error")
                        if route_result
                        else "no response"
                    )
                    logger.warning(
                        f"search_events_node: Could not get route duration between {first_event.session_id} and {candidate_event.session_id} (Error: {error_msg}). Assuming 30 min for check."
                    )
                    route_duration_seconds = 30 * 60
                route_duration_minutes = route_duration_seconds / 60
            except Exception as e_route:
                logger.error(
                    f"search_events_node: Error calling get_route: {e_route}",
                    exc_info=True,
                )
                route_duration_minutes = 30

            arrival_at_candidate_naive = first_event_end_naive + timedelta(
                minutes=route_duration_minutes
            )
            buffer_time = timedelta(minutes=15)
            if (
                arrival_at_candidate_naive
                <= candidate_event.start_time_naive_event_tz - buffer_time
            ):
                events_to_propose.append(candidate_event)
                if len(events_to_propose) >= 2:
                    break
            else:
                logger.debug(
                    f"search_events_node: Candidate {candidate_event.session_id} "
                    f"({candidate_event.name} at {candidate_event.start_time_naive_event_tz.strftime('%H:%M')}) "
                    f"is not suitable. Arrival from previous event at {arrival_at_candidate_naive.strftime('%H:%M')}. "
                    f"Need to arrive by {(candidate_event.start_time_naive_event_tz - buffer_time).strftime('%H:%M')} (event start minus buffer)."
                )

    logger.info(
        f"search_events_node: Proposing {len(events_to_propose)} events to the user."
    )
    return {
        "current_events": events_to_propose,
        "status_message_to_user": None,
        "is_initial_plan_proposed": True,
    }


async def _perform_search(
    current_date_from: datetime,
    current_date_to: datetime,
    current_min_start_time: Optional[datetime],
) -> List[Event]:
    tool_args = EventSearchToolArgs(
        city_id=city_id,
        date_from=current_date_from,  # Используем переданные, а не из state напрямую
        date_to=current_date_to,  # Используем переданные
        interests_keys=interests_keys,
        min_start_time_naive=current_min_start_time,
        max_budget_per_person=budget,
        exclude_session_ids=None,  # Добавить логику для исключения, если нужно при повторном поиске
    )
    logger.info(
        f"search_events_node: Calling event_search_tool with args: {tool_args.model_dump_json(indent=2, exclude_none=True)}"
    )
    try:
        events_dict_list: List[Dict] = await event_search_tool.ainvoke(
            tool_args.model_dump(exclude_none=True)
        )
        current_events_obj_list_internal: List[Event] = []
        if events_dict_list:
            for evt_data in events_dict_list:
                try:
                    current_events_obj_list_internal.append(Event(**evt_data))
                except Exception as val_err:
                    logger.warning(
                        f"search_events_node: Validation error for event data: {evt_data.get('session_id', 'Unknown_ID')}. Error: {val_err}"
                    )
        return current_events_obj_list_internal
    except Exception as e_search:
        logger.error(
            f"search_events_node: Error calling event_search_tool: {e_search}",
            exc_info=True,
        )
        # Не возвращаем здесь сообщение пользователю, обработаем выше
        return []


# --- Логика двухэтапного поиска ---


# --- Узел 4: Представление начального плана и запрос адреса/бюджета ---
async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    current_events = state.get("current_events")
    # Получаем копию, чтобы изменения точно отразились в возвращаемом словаре
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore

    if not current_events:
        logger.warning("present_initial_plan_node: No current events to present.")
        return {
            "status_message_to_user": "Кажется, мероприятий для предложения нет. Попробуем снова?"
        }

    event_descriptions = []
    for i, event in enumerate(current_events):
        event_time_str = event.start_time_naive_event_tz.strftime("%H:%M")
        event_date_str = event.start_time_naive_event_tz.strftime("%d.%m.%Y")
        desc = f"{i+1}. {event.name} ({event.event_type_key}) в '{event.place_name}' ({event.place_address or 'Адрес не указан'}). Начало в {event_time_str} ({event_date_str})."
        if event.min_price is not None:
            desc += f" Цена от {event.min_price} руб."
        event_descriptions.append(desc)

    plan_text = "Вот что я смог найти для вас:\n" + "\n".join(event_descriptions)
    questions_to_user = []

    # Сбрасываем флаг перед проверкой, чтобы избежать ложного срабатывания, если он остался от предыдущего шага
    if "awaiting_address_input" in collected_data:
        del collected_data["awaiting_address_input"]  # type: ignore

    if not collected_data.get("user_start_address_original") and not collected_data.get(
        "user_start_address_validated_coords"
    ):
        questions_to_user.append(
            "Откуда вы планируете начать маршрут? Назовите, пожалуйста, адрес (улица и дом)."
        )
        collected_data["awaiting_address_input"] = True  # type: ignore
        # Убираем user_start_address_original из clarification_needed_fields, так как мы сейчас его запросим через awaiting_address_input
        if "clarification_needed_fields" in collected_data and collected_data.get("clarification_needed_fields"):  # type: ignore
            collected_data["clarification_needed_fields"] = [  # type: ignore
                f for f in collected_data["clarification_needed_fields"] if f != "user_start_address_original"  # type: ignore
            ]
            if not collected_data.get("clarification_needed_fields"):  # type: ignore
                del collected_data["clarification_needed_fields"]  # type: ignore

    if (
        collected_data.get("budget_current_search") is None
        and collected_data.get("budget_original") is None
    ):
        questions_to_user.append(
            "Кстати, чтобы лучше подобрать варианты в будущем или если эти не совсем подойдут, уточните, какой у вас примерный бюджет на одно мероприятие?"
        )

    if questions_to_user:
        plan_text += "\n\n" + " ".join(questions_to_user)
    else:
        plan_text += "\n\nКак вам такой предварительный план?"

    logger.debug(
        f"present_initial_plan_node: collected_data before return: {collected_data}"
    )
    new_messages = state.get("messages", []) + [AIMessage(content=plan_text)]  # type: ignore

    return {
        "messages": new_messages,
        "status_message_to_user": plan_text,
        "collected_data": collected_data,  # Убеждаемся, что возвращаем измененный collected_data
        "is_initial_plan_proposed": True,
    }


# ... (предыдущий код extract_initial_info_node, clarify_missing_data_node, search_events_node, present_initial_plan_node остается выше) ...
# --- Импорты, которые могут понадобиться дополнительно для этой части ---
from schemas.data_schemas import LocationModel, RouteDetails  # Уже импортированы
from tools.route_builder_tool import route_builder_tool  # Наш инструмент


# --- Узел 5: Обработка ответа на адрес ИЛИ построение маршрута, если адрес не нужен / уже есть ---
# agent_core/nodes.py


async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info(
        "Node: build_route_node (formerly clarify_address_or_build_route_node) executing..."
    )
    collected_data: CollectedUserData = state.get("collected_data", {})
    current_events: Optional[List[Event]] = state.get("current_events")
    user_start_coords = collected_data.get("user_start_address_validated_coords")

    if not current_events:
        logger.warning("build_route_node: No current events, cannot build route.")
        return {
            "current_route_details": RouteDetails(
                status="error", error_message="Нет мероприятий для построения маршрута."
            ),
            "is_full_plan_with_route_proposed": False,
        }

    if len(current_events) == 1 and not user_start_coords:
        logger.info(
            "build_route_node: One event and no user address, route not built as per logic."
        )
        return {
            "current_route_details": None,
            "is_full_plan_with_route_proposed": False,
        }

    start_location_for_api: Optional[LocationModel] = None
    if user_start_coords:
        start_location_for_api = LocationModel(
            lon=user_start_coords["lon"],  # type: ignore
            lat=user_start_coords["lat"],  # type: ignore
            address_string=collected_data.get("user_start_address_original"),
        )
    elif len(current_events) > 1:
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
        elif first_event.place_address:
            city_for_geocoding = collected_data.get("city_name", "")
            coords = await get_coords_from_address(
                address=first_event.place_address, city=city_for_geocoding
            )
            if coords:
                start_location_for_api = LocationModel(
                    lon=coords[0],
                    lat=coords[1],
                    address_string=first_event.place_address,
                )
            else:
                logger.error(
                    f"build_route_node: Could not geocode first event address '{first_event.place_address}' for routing."
                )
                return {
                    "current_route_details": RouteDetails(
                        status="error",
                        error_message=f"Не удалось определить координаты первого мероприятия '{first_event.name}'.",
                    ),
                    "is_full_plan_with_route_proposed": False,
                }
        else:
            logger.error(
                f"build_route_node: First event '{first_event.name}' has no coordinates or address for routing."
            )
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message=f"У мероприятия '{first_event.name}' не указан адрес для маршрута.",
                ),
                "is_full_plan_with_route_proposed": False,
            }
    else:
        logger.info(
            "build_route_node: Not enough points for routing and no user_start_address."
        )
        return {
            "current_route_details": None,
            "is_full_plan_with_route_proposed": False,
        }

    event_points_for_api: List[LocationModel] = []
    events_to_route = current_events if user_start_coords else current_events[1:]

    for event_obj in events_to_route:
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
        elif event_obj.place_address:
            city_for_geocoding = collected_data.get("city_name", "")
            coords = await get_coords_from_address(
                address=event_obj.place_address, city=city_for_geocoding
            )
            if coords:
                event_points_for_api.append(
                    LocationModel(
                        lon=coords[0],
                        lat=coords[1],
                        address_string=event_obj.place_address,
                    )
                )
            else:
                logger.warning(
                    f"build_route_node: Could not geocode address for event '{event_obj.name}' ({event_obj.place_address}). Skipping for route."
                )
        else:
            logger.warning(
                f"build_route_node: Event '{event_obj.name}' has no coordinates or address. Skipping for route."
            )

    if not start_location_for_api:
        logger.error("build_route_node: Start location for API is not defined.")
        return {
            "current_route_details": RouteDetails(
                status="error", error_message="Не определена начальная точка маршрута."
            ),
            "is_full_plan_with_route_proposed": False,
        }

    if not event_points_for_api and len(events_to_route) > 0:
        logger.error(
            "build_route_node: No valid event points with coordinates to build a route to."
        )
        return {
            "current_route_details": RouteDetails(
                status="error",
                error_message="Не удалось определить координаты для мероприятий назначения.",
            ),
            "is_full_plan_with_route_proposed": False,
        }

    if not event_points_for_api and start_location_for_api:
        logger.info(
            "build_route_node: Only a single start point, no event points to route to. Route is trivial."
        )
        return {
            "current_route_details": RouteDetails(
                status="success",
                segments=[],
                total_duration_seconds=0,
                total_duration_text="0 мин",
                total_distance_meters=0,
                total_distance_text="0 км",
            ),
            "is_full_plan_with_route_proposed": True,
        }

    tool_args = RouteBuilderToolArgs(
        start_point=start_location_for_api, event_points=event_points_for_api
    )
    logger.info(
        f"build_route_node: Calling route_builder_tool with args: {tool_args.model_dump_json(indent=2, exclude_none=True)}"
    )

    route_data_dict = await route_builder_tool.ainvoke(
        tool_args.model_dump(exclude_none=True)
    )

    try:
        route_details_obj = RouteDetails(**route_data_dict)

        if (
            route_details_obj.status == "success"
            or route_details_obj.status == "partial_success"
        ):
            logger.info(
                f"build_route_node: Route building process finished. Status: {route_details_obj.status}. Total Duration: {route_details_obj.total_duration_text}"
            )
        else:
            logger.error(
                f"build_route_node: Route building failed. Status: {route_details_obj.status}, Msg: {route_details_obj.error_message}"
            )

        return {
            "current_route_details": route_details_obj,  # Возвращаем ОБЪЕКТ RouteDetails
            "is_full_plan_with_route_proposed": route_details_obj.status
            in ["success", "partial_success"],
        }
    except ValidationError as ve:
        logger.error(
            f"build_route_node: Validation error for route_data returned by tool: {route_data_dict}. Error: {ve}"
        )
        return {
            "current_route_details": RouteDetails(
                status="error",
                error_message="Ошибка при обработке данных маршрута от инструмента.",
            ),
            "is_full_plan_with_route_proposed": False,
        }


# --- Узел 6: Представление полного плана (мероприятия + маршрут) ---
async def present_full_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_full_plan_node executing...")
    current_events: Optional[List[Event]] = state.get("current_events")
    # current_route_details теперь будет объектом RouteDetails
    current_route_details_obj: Optional[RouteDetails] = state.get("current_route_details")  # type: ignore

    if not current_events:
        logger.warning("present_full_plan_node: No current events to present.")
        return {
            "status_message_to_user": "Мероприятий для отображения нет. Пожалуйста, начните новый поиск."
        }

    response_parts = ["Вот ваш план:"]
    for i, event in enumerate(current_events):
        event_time_str = event.start_time_naive_event_tz.strftime("%H:%M")
        event_date_str = event.start_time_naive_event_tz.strftime("%d.%m.%Y")
        desc = f"\n{i+1}. {event.name} ({event.event_type_key})\n   Место: {event.place_name} ({event.place_address or 'Адрес не уточнен'})"
        desc += f"\n   Время: {event_date_str} в {event_time_str}"
        if event.duration_minutes:
            desc += f" (продолжительность ~{event.duration_minutes // 60}ч {event.duration_minutes % 60}м)"
        if event.price_text:
            desc += f"\n   Цена: {event.price_text}"
        response_parts.append(desc)

    if current_route_details_obj:
        if (
            current_route_details_obj.status == "success"
            or current_route_details_obj.status == "partial_success"
        ):
            if (
                current_route_details_obj.segments
                and len(current_route_details_obj.segments) > 0
            ):
                response_parts.append("\nМаршрут:")
                for idx, segment in enumerate(current_route_details_obj.segments):
                    from_name = segment.from_address or f"Точка {idx+1}"
                    to_name = segment.to_address or f"Точка {idx+2}"
                    segment_text = f"  {idx+1}. От '{from_name}' до '{to_name}': "
                    if segment.segment_status == "success":
                        segment_text += f"{segment.duration_text or '? мин'}, {segment.distance_text or '? км'}."
                    else:
                        segment_text += f"не удалось построить ({segment.segment_error_message or 'причина неизвестна'})."
                    response_parts.append(segment_text)

                if (
                    current_route_details_obj.total_duration_text
                    and current_route_details_obj.total_distance_text
                    and len(current_route_details_obj.segments) > 1
                ):
                    response_parts.append(
                        f"\n  Общее время в пути по построенным сегментам: {current_route_details_obj.total_duration_text}, общее расстояние: {current_route_details_obj.total_distance_text}."
                    )
                elif current_route_details_obj.status == "partial_success":
                    response_parts.append(
                        "\n  Не все части маршрута удалось построить."
                    )

            elif (
                current_route_details_obj.total_duration_seconds == 0
                and not current_route_details_obj.segments
            ):  # Маршрут не нужен (1 точка)
                response_parts.append("\nМаршрут: одна точка, маршрут не требуется.")
            else:  # Успех, но нет сегментов - странно, но обрабатываем
                response_parts.append(
                    "\nМаршрут: построен, но детали сегментов отсутствуют."
                )

        elif (
            current_route_details_obj.status != "success"
        ):  # Например, status == "error"
            response_parts.append(
                f"\nМаршрут: Не удалось построить ({current_route_details_obj.error_message or 'причина неизвестна'})."
            )

    elif not current_route_details_obj:  # Если current_route_details вообще None
        if (
            current_events
            and len(current_events) > 1
            and state.get("collected_data", {}).get(
                "user_start_address_validated_coords"
            )
        ):
            response_parts.append(
                "\nМаршрут: Построение маршрута не выполнялось или произошла ошибка до его формирования."
            )
        elif (
            current_events
            and len(current_events) == 1
            and not state.get("collected_data", {}).get(
                "user_start_address_validated_coords"
            )
        ):
            response_parts.append(
                "\nМаршрут: Не указан адрес отправления, поэтому маршрут до мероприятия не построен."
            )

    response_parts.append(
        "\n\nКак вам такой план? Можем что-то изменить, добавить, убрать или подобрать другие варианты с учетом бюджета/количества людей, если вы их уточните?"
    )
    full_plan_text = "\n".join(response_parts)
    new_messages = state.get("messages", []) + [AIMessage(content=full_plan_text)]  # type: ignore

    return {
        "messages": new_messages,
        "status_message_to_user": full_plan_text,
        "awaiting_final_confirmation": True,
        "is_full_plan_with_route_proposed": True,
    }


# ... (предыдущий код extract_initial_info_node, clarify_missing_data_node, search_events_node,
# present_initial_plan_node, clarify_address_or_build_route_node остается выше) ...
# --- Импорты, которые могут понадобиться дополнительно для этой части ---
from prompts.system_prompts import (
    PLAN_FEEDBACK_ANALYSIS_PROMPT,
    CHANGE_CONFIRMATION_PROMPT_TEMPLATE,
    EVENT_NOT_FOUND_PROMPT_TEMPLATE,
)
from schemas.data_schemas import LocationModel  # Уже импортирован


# --- Узел 7: Обработка обратной связи по плану ---
async def handle_plan_feedback_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: handle_plan_feedback_node executing...")
    messages = state.get("messages", [])
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    current_events: Optional[List[Event]] = state.get("current_events")

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("handle_plan_feedback_node: No human feedback message found.")
        # Если нет фидбека, возможно, ошибка в графе, или нужно просто перейти к END, если план был подтвержден неявно
        return {
            "status_message_to_user": "Не получил вашего ответа по плану.",
            "awaiting_final_confirmation": True,
        }

    user_feedback = messages[-1].content
    llm = get_gigachat_client()  # Получаем клиент GigaChat
    structured_llm_feedback = llm.with_structured_output(AnalyzedFeedback)

    # Формируем краткое описание текущего плана для контекста LLM
    plan_summary_parts = []
    if current_events:
        for i, event in enumerate(current_events):
            plan_summary_parts.append(
                f"Мероприятие {i+1}: {event.name} в {event.place_name} ({event.start_time_naive_event_tz.strftime('%d.%m %H:%M')})"
            )
    if state.get("current_route_details"):
        plan_summary_parts.append(f"Маршрут: {state['current_route_details'].duration_text}")  # type: ignore
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
            f"handle_plan_feedback_node: LLM Analyzed Feedback: {analyzed_result.model_dump_json(indent=2)}"
        )

        intent = analyzed_result.intent_type
        changes = analyzed_result.change_details

        # Сбрасываем флаг ожидания подтверждения, так как мы получили ответ
        next_state_update: Dict[str, Any] = {"awaiting_final_confirmation": False}
        next_state_update["collected_data"] = dict(
            collected_data
        )  # Копируем, чтобы изменять
        next_state_update["current_events"] = (
            list(current_events) if current_events else []
        )  # Копируем

        if intent == "confirm_plan":
            logger.info("handle_plan_feedback_node: User confirmed the plan.")
            next_state_update["status_message_to_user"] = (
                "Отлично! Рад был помочь. Если понадобится что-то еще, обращайтесь!"
            )
            # Этот узел сам по себе не переходит в END, это решает условное ребро
            # Но мы можем подготовить состояние к завершению, если это финальное подтверждение

        elif intent == "request_change" and changes:
            logger.info(f"handle_plan_feedback_node: User requested changes: {changes}")
            # Сохраняем текущие подтвержденные данные перед применением изменений
            # Это важно для логики "переспросить, правильно ли понял"
            next_state_update["previous_confirmed_collected_data"] = dict(
                collected_data
            )
            next_state_update["previous_confirmed_events"] = (
                list(current_events) if current_events else []
            )

            # Применяем изменения к collected_data или готовим pending_plan_modification_request
            # Эта логика может быть сложной и потребует итераций
            change_target = changes.get("change_target")
            new_value = changes.get("new_value")

            if change_target == "budget":
                if isinstance(new_value, (int, float)):
                    next_state_update["collected_data"]["budget_current_search"] = int(new_value)  # type: ignore
                    logger.info(f"Updated budget to: {new_value}")
            elif change_target == "date":
                if isinstance(new_value, str):  # Ожидаем, что LLM вернет описание даты
                    # Повторно используем datetime_parser_tool
                    parsed_date_res = await datetime_parser_tool.ainvoke(
                        {"natural_language_date_time": new_value}
                    )
                    if parsed_date_res.get("datetime_iso"):
                        next_state_update["collected_data"]["parsed_dates_iso"] = [parsed_date_res["datetime_iso"]]  # type: ignore
                        logger.info(
                            f"Updated dates to: {parsed_date_res['datetime_iso']}"
                        )
                        if parsed_date_res.get("is_ambiguous"):
                            next_state_update["collected_data"].setdefault("clarification_needed_fields", []).append("dates_description_original")  # type: ignore
                            next_state_update["clarification_context"] = (
                                parsed_date_res.get("clarification_needed")
                            )
                    else:
                        logger.warning(f"Could not parse new date: {new_value}")
                        next_state_update["collected_data"].setdefault("clarification_needed_fields", []).append("dates_description_original")  # type: ignore
                        next_state_update["clarification_context"] = (
                            f"Не удалось распознать новую дату '{new_value}'. Пожалуйста, уточните."
                        )
            elif change_target == "start_location":
                if isinstance(new_value, str):
                    next_state_update["collected_data"]["user_start_address_original"] = new_value  # type: ignore
                    next_state_update["collected_data"][
                        "user_start_address_validated_coords"
                    ] = None  # Сбросить координаты, нужна перепроверка
                    logger.info(f"Updated start location to: {new_value}")
            elif "event_" in str(
                change_target
            ):  # Например, "event_1_interest" или "remove_event_2"
                # Здесь должна быть более сложная логика для замены/удаления мероприятия
                # Например, если "remove_event_2", то event_index_to_remove = 1 (0-based)
                # Если "replace_event_1_with_interest": "спектакль"
                # Пока реализуем простую заглушку - помечаем все интересы для пересмотра
                logger.info(
                    f"Change target is event-related: {change_target}, new_value: {new_value}"
                )
                next_state_update["collected_data"].setdefault("clarification_needed_fields", []).append("interests_original")  # type: ignore
                next_state_update["clarification_context"] = (
                    f"Пользователь хочет изменить мероприятие. Запрос: '{user_feedback}'. Уточни детали изменения."
                )
            else:  # Общее изменение интересов или типа мероприятия
                if isinstance(
                    new_value, str
                ):  # Предполагаем, что new_value - это новый интерес
                    next_state_update["collected_data"]["interests_original"] = [new_value]  # type: ignore
                    # TODO: Маппинг на Afisha ключи
                    next_state_update["collected_data"]["interests_keys_afisha"] = [new_value]  # type: ignore
                    logger.info(f"Updated interests to: {[new_value]}")

            # Устанавливаем, что изменения ожидают подтверждения (для Примера 2 из инструкции)
            next_state_update["pending_plan_modification_request"] = (
                changes  # Сохраняем запрос на изменение
            )
            next_state_update["status_message_to_user"] = (
                None  # Сообщение будет от confirm_changes_node
            )

        elif intent == "clarify_misunderstanding" or intent == "new_search":
            logger.info(
                f"handle_plan_feedback_node: User response is unclear or requests new search ('{intent}')."
            )
            # Просим пользователя переформулировать или начать заново
            next_state_update["status_message_to_user"] = (
                "Я вас не совсем понял. Не могли бы вы переформулировать ваш запрос или указать новые критерии для поиска?"
            )
            # Сбрасываем текущий план, чтобы избежать путаницы, если пользователь хочет начать заново
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
            # Можно также очистить часть collected_data или пометить всё для уточнения,
            # если это new_search, но по вашей инструкции новый поиск - это новый цикл.
            # Пока что просто просим переформулировать.
        else:  # Неизвестное намерение
            logger.warning(
                f"handle_plan_feedback_node: Unknown intent from LLM: {intent}"
            )
            next_state_update["status_message_to_user"] = (
                "Я не уверен, что понял вас. Попробуете еще раз?"
            )

        # Добавляем AI сообщение в историю, если оно сформировано
        if next_state_update.get("status_message_to_user"):
            next_state_update["messages"] = messages + [
                AIMessage(content=next_state_update["status_message_to_user"])
            ]
        else:  # Если сообщение будет от следующего узла (confirm_changes_node)
            next_state_update["messages"] = (
                messages  # Просто сохраняем текущие сообщения
            )

        return next_state_update

    except Exception as e:
        logger.error(
            f"handle_plan_feedback_node: Error analyzing feedback: {e}", exc_info=True
        )
        msg = "Произошла ошибка при обработке вашего ответа. Пожалуйста, попробуйте еще раз."
        return {
            "status_message_to_user": msg,
            "awaiting_final_confirmation": True,
            "messages": messages + [AIMessage(content=msg)],
        }


# --- Узел 8: Подтверждение изменений (для Примера 2 из инструкции) ---
async def confirm_changes_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: confirm_changes_node executing...")
    messages = state.get("messages", [])
    # previous_collected_data = state.get("previous_confirmed_collected_data", {})
    # current_collected_data = state.get("collected_data", {})
    # pending_modification = state.get("pending_plan_modification_request", {})

    # TODO: Сформировать user_friendly описание original_criteria, requested_change, new_criteria
    # Это может потребовать отдельного вызова LLM или более сложной логики форматирования.
    # Пока сделаем очень упрощенно, предполагая, что LLM в handle_plan_feedback_node
    # уже извлек суть изменений и мы просто просим подтвердить новые критерии.

    collected_data_summary_parts = []
    current_collected: CollectedUserData = state.get("collected_data", {})  # type: ignore
    if current_collected.get("city_name"):
        collected_data_summary_parts.append(f"город: {current_collected['city_name']}")
    if current_collected.get("parsed_dates_iso"):
        collected_data_summary_parts.append(f"даты: {', '.join(current_collected['parsed_dates_iso'])}")  # type: ignore
    if current_collected.get("interests_original"):
        collected_data_summary_parts.append(f"интересы: {', '.join(current_collected['interests_original'])}")  # type: ignore
    if current_collected.get("budget_current_search") is not None:
        collected_data_summary_parts.append(
            f"бюджет до: {current_collected['budget_current_search']} руб."
        )
    new_criteria_summary_str = (
        "; ".join(collected_data_summary_parts)
        if collected_data_summary_parts
        else "новые критерии не ясны"
    )

    # Используем промпт из system_prompts
    prompt_text = CHANGE_CONFIRMATION_PROMPT_TEMPLATE.format(
        original_criteria_summary="ранее согласованные",  # Заглушка, т.к. previous_confirmed_collected_data может быть сложным
        requested_change_description=str(
            state.get("pending_plan_modification_request", "Общее изменение")
        ),
        new_criteria_summary=new_criteria_summary_str,
    )

    llm = get_gigachat_client()
    try:
        ai_response = await llm.ainvoke(prompt_text)
        confirmation_question = ai_response.content
        logger.info(
            f"confirm_changes_node: LLM generated confirmation question: {confirmation_question}"
        )
    except Exception as e:
        logger.error(
            f"confirm_changes_node: Error generating confirmation question: {e}",
            exc_info=True,
        )
        confirmation_question = (
            f"Итак, мы ищем по критериям: {new_criteria_summary_str}. Верно?"
        )

    new_messages = messages + [AIMessage(content=confirmation_question)]
    return {
        "messages": new_messages,
        "status_message_to_user": confirmation_question,
        "pending_plan_modification_request": None,
    }  # Сбрасываем


# --- Узел 9: Сообщение об ошибке, если мероприятия не найдены ---
async def error_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: error_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore

    # Формируем сообщение об ошибке
    search_criteria_list = []
    if collected_data.get("city_name"):
        search_criteria_list.append(f"город '{collected_data['city_name']}'")
    if collected_data.get("dates_description_original"):
        search_criteria_list.append(
            f"даты '{collected_data['dates_description_original']}'"
        )
    if collected_data.get("interests_original"):
        search_criteria_list.append(f"интересы '{', '.join(collected_data['interests_original'])}'")  # type: ignore
    search_criteria_summary_str = (
        ", ".join(search_criteria_list)
        if search_criteria_list
        else "указанным вами критериям"
    )

    error_message = EVENT_NOT_FOUND_PROMPT_TEMPLATE.format(
        search_criteria_summary=search_criteria_summary_str
    )
    logger.warning(f"error_node: Presenting error to user: {error_message}")

    new_messages = state.get("messages", []) + [AIMessage(content=error_message)]  # type: ignore

    # Сбрасываем текущие найденные события, так как их нет
    # Флаги готовности плана также сбрасываем
    return {
        "messages": new_messages,
        "status_message_to_user": error_message,
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,  # После ошибки не ждем подтверждения, а ждем новых критериев
    }
