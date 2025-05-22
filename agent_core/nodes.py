import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from datetime import datetime, timedelta
import aiohttp

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
    # 1. Инициализация collected_data
    if "collected_data" not in state or state["collected_data"] is None:
        collected_data_state: CollectedUserData = CollectedUserData()
    else:
        collected_data_state = state["collected_data"]

    # 2. Проверяем, что последнее сообщение — HumanMessage (новый пользовательский ввод)
    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.debug(
            "extract_initial_info_node: Last message is not HumanMessage or no messages. No new extraction will be performed."
        )
        return {
            "collected_data": collected_data_state,
            "messages": messages,
            "clarification_context": state.get("clarification_context"),
            "status_message_to_user": state.get("status_message_to_user"),
        }

    # 3. Начинаем разбор нового сообщения пользователя
    logger.info(
        f"extract_initial_info_node: Processing new HumanMessage: '{messages[-1].content}'"
    )
    collected_data_state["clarification_needed_fields"] = []
    clarification_context_for_current_step = None
    user_query = messages[-1].content
    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ExtractedInitialInfo)
    try:
        logger.debug(
            f"extract_initial_info_node: Querying LLM for initial info from: '{user_query}'"
        )
        extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{user_query}"'
        extracted_info: ExtractedInitialInfo = await structured_llm.ainvoke(
            extraction_prompt_with_query
        )
        logger.info(
            f"extract_initial_info_node: LLM Extracted Info: {extracted_info.model_dump_json(indent=2)}"
        )

        # --- Город ---
        if extracted_info.city:
            city_changed = (
                not collected_data_state.get("city_name")
                or collected_data_state.get("city_name", "").lower()
                != extracted_info.city.lower()
            )
            if city_changed:
                collected_data_state["city_name"] = extracted_info.city
                cities_from_afisha = await fetch_cities_internal()
                found_city_afisha = next(
                    (
                        c
                        for c in cities_from_afisha
                        if extracted_info.city.lower() in c["name_lower"]
                    ),
                    None,
                )
                if found_city_afisha:
                    collected_data_state["city_id_afisha"] = found_city_afisha["id"]
                else:
                    collected_data_state["city_id_afisha"] = None
                    collected_data_state.setdefault(
                        "clarification_needed_fields", []
                    ).append("city_name")
        elif not collected_data_state.get("city_name"):
            collected_data_state.setdefault("clarification_needed_fields", []).append(
                "city_name"
            )

        # --- Даты ---
        if extracted_info.dates_description:
            collected_data_state["dates_description_original"] = (
                extracted_info.dates_description
            )
            current_date_iso = datetime.now().isoformat()
            parsed_date_time_result = await datetime_parser_tool.ainvoke(
                {
                    "natural_language_date_time": extracted_info.dates_description,
                    "base_date_iso": current_date_iso,
                }
            )
            logger.info(
                f"extract_initial_info_node: DateTimeParserTool result: {parsed_date_time_result}"
            )
            if parsed_date_time_result.get("datetime_iso"):
                collected_data_state["parsed_dates_iso"] = [
                    parsed_date_time_result["datetime_iso"]
                ]
                if parsed_date_time_result.get("is_ambiguous"):
                    collected_data_state.setdefault(
                        "clarification_needed_fields", []
                    ).append("dates_description_original")
                    clarification_context_for_current_step = (
                        parsed_date_time_result.get("clarification_needed")
                    )
            elif not collected_data_state.get("parsed_dates_iso"):
                collected_data_state.setdefault(
                    "clarification_needed_fields", []
                ).append("dates_description_original")
        elif not collected_data_state.get("parsed_dates_iso"):
            collected_data_state.setdefault("clarification_needed_fields", []).append(
                "dates_description_original"
            )

        # --- Интересы ---
        if extracted_info.interests:
            collected_data_state["interests_original"] = extracted_info.interests
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
                if not key:
                    key = interest_str.capitalize()
                mapped_interest_keys.append(key)
            collected_data_state["interests_keys_afisha"] = list(
                set(mapped_interest_keys)
            )
        elif not collected_data_state.get("interests_keys_afisha"):
            collected_data_state.setdefault("clarification_needed_fields", []).append(
                "interests_original"
            )

        # --- Бюджет ---
        if extracted_info.budget is not None:
            collected_data_state["budget_original"] = extracted_info.budget
            collected_data_state["budget_current_search"] = extracted_info.budget

        # --- Сырые временные описания (если есть) ---
        if extracted_info.raw_time_description:
            if not collected_data_state.get("dates_description_original"):
                collected_data_state["dates_description_original"] = (
                    extracted_info.raw_time_description
                )
            collected_data_state["raw_time_description_original"] = (
                extracted_info.raw_time_description
            )
            if not clarification_context_for_current_step:
                clarification_context_for_current_step = f"Пользователь указал время как '{extracted_info.raw_time_description}'. Уточните, пожалуйста, время."
            collected_data_state.setdefault("clarification_needed_fields", []).append(
                "dates_description_original"
            )

    except Exception as e:
        logger.error(
            f"extract_initial_info_node: Critical error during LLM call or info processing: {e}",
            exc_info=True,
        )
        # Если ошибка, запросить ВСЕ
        for f in ["city_name", "dates_description_original", "interests_original"]:
            if f not in collected_data_state.get("clarification_needed_fields", []):
                collected_data_state.setdefault(
                    "clarification_needed_fields", []
                ).append(f)

    # Очищаем дубли в clarification_needed_fields
    if "clarification_needed_fields" in collected_data_state:
        fields = collected_data_state["clarification_needed_fields"]
        collected_data_state["clarification_needed_fields"] = list(
            dict.fromkeys([f for f in fields if f])
        )

    logger.info(
        f"extract_initial_info_node: Final collected_data after extraction: {str(collected_data_state)[:500]}"
    )
    logger.info(
        f"extract_initial_info_node: Clarification needed for after extraction: {collected_data_state.get('clarification_needed_fields')}"
    )

    return {
        "collected_data": collected_data_state,
        "messages": messages,
        "clarification_context": clarification_context_for_current_step,
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
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore

    city_id = collected_data.get("city_id_afisha")
    parsed_dates_iso = collected_data.get("parsed_dates_iso")
    interests_keys = collected_data.get(
        "interests_keys_afisha", collected_data.get("interests_original")
    )
    budget = collected_data.get("budget_current_search")

    if not city_id or not parsed_dates_iso or not interests_keys:
        logger.error(
            f"search_events_node: Missing critical data for event search. CityID: {city_id}, Dates: {parsed_dates_iso}, Interests: {interests_keys}"
        )
        return {
            "current_events": [],
            "status_message_to_user": "Мне не хватает информации для поиска (город, даты или интересы). Пожалуйста, уточните.",
            "is_initial_plan_proposed": False,
        }

    try:
        date_from_dt = datetime.fromisoformat(parsed_dates_iso[0])
        date_to_dt = (
            datetime.fromisoformat(parsed_dates_iso[1])
            if len(parsed_dates_iso) > 1
            else (
                datetime.fromisoformat(parsed_dates_iso[0]) + timedelta(days=1)
            ).replace(hour=0, minute=0, second=0, microsecond=0)
        )
    except (ValueError, IndexError, TypeError) as e:
        logger.error(
            f"search_events_node: Error parsing dates from ISO strings {parsed_dates_iso}: {e}",
            exc_info=True,
        )
        return {
            "current_events": [],
            "status_message_to_user": "Не удалось распознать даты для поиска. Пожалуйста, уточните формат дат.",
            "is_initial_plan_proposed": False,
        }

    tool_args = EventSearchToolArgs(
        city_id=city_id,
        date_from=date_from_dt,
        date_to=date_to_dt,
        interests_keys=interests_keys,
        min_start_time_naive=None,
        max_budget_per_person=budget,
        exclude_session_ids=None,
    )

    logger.info(
        f"search_events_node: Calling event_search_tool with args: {tool_args.model_dump_json(indent=2, exclude_none=True)}"
    )

    try:
        events_dict_list: List[Dict] = await event_search_tool.ainvoke(
            tool_args.model_dump(exclude_none=True)
        )
        current_events_obj_list: List[Event] = []
        if events_dict_list:
            for evt_data in events_dict_list:
                try:
                    current_events_obj_list.append(Event(**evt_data))
                except Exception as val_err:
                    logger.warning(
                        f"search_events_node: Validation error for event data: {evt_data.get('session_id', 'Unknown_ID')}. Error: {val_err}"
                    )
    except Exception as e:
        logger.error(
            f"search_events_node: Error calling event_search_tool: {e}", exc_info=True
        )
        status_msg = "Произошла ошибка при поиске мероприятий. Попробуйте позже или измените запрос."
        return {
            "current_events": [],
            "status_message_to_user": status_msg,
            "is_initial_plan_proposed": False,
        }

    if not current_events_obj_list:
        logger.info("search_events_node: No events found by tool.")
        search_criteria_summary = f"город: {collected_data.get('city_name')}, даты: {parsed_dates_iso}, интересы: {interests_keys}"
        status_msg = f"К сожалению, по вашим критериям ({search_criteria_summary}) ничего не нашлось. Может, попробуем другие даты или интересы?"
        return {
            "current_events": [],
            "status_message_to_user": status_msg,
            "is_initial_plan_proposed": False,
        }

    # --- Новый алгоритм: Учитываем время окончания первого + дорогу до второго через 2ГИС ---
    # 1. Берём первое событие как отправную точку
    first_event = current_events_obj_list[0]
    events_to_propose = [first_event]

    first_event_end = first_event.start_time_naive_event_tz
    if first_event.duration_minutes:
        first_event_end += timedelta(minutes=first_event.duration_minutes)
    else:
        # Если нет duration — берём плюс 2 часа как дефолт
        first_event_end += timedelta(hours=2)

    # 2. Пытаемся подобрать второе мероприятие, чтобы успеть после первого (учитывая дорогу)
    for candidate in current_events_obj_list[1:]:
        # Время между первым и кандидатом
        route_duration = None

        # --- Здесь вызов 2ГИС для оценки времени в пути (асинхронный) ---
        try:
            # Импортируй/напиши свою функцию, например get_route_duration()
            route_result = await get_route_duration(
                from_lon=first_event.place_coords_lon,
                from_lat=first_event.place_coords_lat,
                to_lon=candidate.place_coords_lon,
                to_lat=candidate.place_coords_lat,
                transport_type="public_transport",  # Можно менять тип
                start_time=first_event_end,
            )
            route_duration = route_result["duration_minutes"]
        except Exception as e:
            logger.warning(
                f"search_events_node: Не удалось получить маршрут через 2ГИС: {e}"
            )
            # Если не получилось — пробуем без маршрута, но лучше пропустить такой кандидат
            continue

        # Время, когда можно попасть на второе мероприятие
        arrival_time = first_event_end + timedelta(minutes=route_duration or 0)
        # Можем ли попасть? Надо чтобы arrival_time <= candidate.start_time_naive_event_tz + запас 5-10 минут
        if arrival_time <= candidate.start_time_naive_event_tz + timedelta(minutes=10):
            events_to_propose.append(candidate)
            break  # Только два события

    return {
        "current_events": events_to_propose,
        "status_message_to_user": None,
        "is_initial_plan_proposed": True,
    }


# --- Узел 4: Представление начального плана и запрос адреса/бюджета ---
async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    current_events = state.get("current_events")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore

    if not current_events:
        logger.warning("present_initial_plan_node: No current events to present.")
        # Это состояние не должно достигаться, если search_events_node вернул пустой список
        # и условный переход направил на error_node.
        # Но на всякий случай, если мы сюда попали.
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
    # Запрос начальной точки, если она еще неизвестна
    if not collected_data.get("user_start_address_original") and not collected_data.get(
        "user_start_address_validated_coords"
    ):
        questions_to_user.append(
            "Откуда вы планируете начать маршрут? Назовите, пожалуйста, адрес (улица и дом)."
        )
        collected_data.setdefault("clarification_needed_fields", []).append("user_start_address_original")  # type: ignore

    # Опциональный запрос бюджета, если он не был указан и еще не спрашивали
    # (или если хотим уточнить после первого поиска)
    if (
        collected_data.get("budget_current_search") is None
        and collected_data.get("budget_original") is None
    ):
        # Можно добавить флаг is_budget_clarified в AgentState, чтобы не спрашивать повторно, если уже отказались.
        questions_to_user.append(
            "Кстати, чтобы лучше подобрать варианты в будущем или если эти не совсем подойдут, уточните, какой у вас примерный бюджет на одно мероприятие?"
        )
        # Не помечаем для уточнения, так как это опционально

    if questions_to_user:
        plan_text += "\n\n" + " ".join(questions_to_user)
    else:
        # Если адрес и бюджет уже есть (или бюджет не нужен), можно сразу перейти к построению маршрута
        # или к запросу подтверждения, если маршрут не строится для одного мероприятия без адреса.
        # Пока просто предлагаем план. Логика маршрута будет в следующем узле.
        plan_text += "\n\nКак вам такой предварительный план?"

    # Очищаем clarification_needed_fields, если они были только для адреса/бюджета и мы их задали
    if "clarification_needed_fields" in collected_data:
        collected_data["clarification_needed_fields"] = [  # type: ignore
            f for f in collected_data.get("clarification_needed_fields", []) if f not in ["user_start_address_original"]  # type: ignore
        ]
        if not collected_data["clarification_needed_fields"]:  # type: ignore
            del collected_data["clarification_needed_fields"]  # type: ignore

    new_messages = state.get("messages", []) + [AIMessage(content=plan_text)]  # type: ignore

    return {
        "messages": new_messages,
        "status_message_to_user": plan_text,
        "collected_data": collected_data,
        "is_initial_plan_proposed": True,  # Подтверждаем, что начальный план предложен
    }


# ... (предыдущий код extract_initial_info_node, clarify_missing_data_node, search_events_node, present_initial_plan_node остается выше) ...
# --- Импорты, которые могут понадобиться дополнительно для этой части ---
from schemas.data_schemas import LocationModel, RouteDetails  # Уже импортированы
from tools.route_builder_tool import route_builder_tool  # Наш инструмент


# --- Узел 5: Обработка ответа на адрес ИЛИ построение маршрута, если адрес не нужен / уже есть ---
async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_address_or_build_route_node executing...")
    messages = state.get("messages", [])
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    current_events: Optional[List[Event]] = state.get("current_events")

    user_start_address_original = collected_data.get("user_start_address_original")
    user_start_coords = collected_data.get("user_start_address_validated_coords")

    # Если последний ответ был от пользователя и мы ждали адрес
    if (
        messages
        and isinstance(messages[-1], HumanMessage)
        and (
            "user_start_address_original"
            in collected_data.get("clarification_needed_fields", [])
            or (not user_start_address_original and not user_start_coords)
        )
    ):  # Если адрес не был известен и не был уточнен ранее

        potential_address = messages[-1].content
        logger.info(
            f"clarify_address_or_build_route_node: User provided potential address: '{potential_address}'"
        )
        # Пытаемся получить координаты для введенного адреса
        # Город должен быть уже в collected_data["city_name"]
        city_for_geocoding = collected_data.get("city_name")
        if not city_for_geocoding:  # На всякий случай, если города нет
            logger.error(
                "clarify_address_or_build_route_node: City name missing in collected_data, cannot geocode address."
            )
            # Возвращаем запрос на уточнение города, если его нет
            collected_data.setdefault("clarification_needed_fields", []).append("city_name")  # type: ignore
            return {
                "collected_data": collected_data,
                "status_message_to_user": "Пожалуйста, сначала уточните город.",
                "awaiting_final_confirmation": False,
                "current_route_details": None,  # Маршрут не построен
            }

        coords = await get_coords_from_address(
            address=potential_address, city=city_for_geocoding
        )
        if coords:
            collected_data["user_start_address_original"] = potential_address
            collected_data["user_start_address_validated_coords"] = {
                "lon": coords[0],
                "lat": coords[1],
            }
            logger.info(
                f"clarify_address_or_build_route_node: Address '{potential_address}' geocoded to {coords}"
            )
            # Убираем адрес из списка для уточнения, если он там был
            if (
                "clarification_needed_fields" in collected_data
                and collected_data["clarification_needed_fields"]
            ):
                collected_data["clarification_needed_fields"] = [  # type: ignore
                    f for f in collected_data["clarification_needed_fields"] if f != "user_start_address_original"  # type: ignore
                ]
                if not collected_data["clarification_needed_fields"]:  # type: ignore
                    del collected_data["clarification_needed_fields"]  # type: ignore
            user_start_coords = collected_data[
                "user_start_address_validated_coords"
            ]  # Обновляем для текущего вызова
        else:
            logger.warning(
                f"clarify_address_or_build_route_node: Could not geocode address '{potential_address}' in city '{city_for_geocoding}'."
            )
            # Адрес не распознан, снова просим уточнить ИМЕННО АДРЕС
            collected_data.setdefault("clarification_needed_fields", []).append("user_start_address_original")  # type: ignore
            msg = "Не удалось распознать этот адрес. Пожалуйста, попробуйте ввести его еще раз (улица и номер дома) или укажите другой."
            new_messages = messages + [AIMessage(content=msg)]
            return {
                "collected_data": collected_data,
                "messages": new_messages,
                "status_message_to_user": msg,
                "awaiting_final_confirmation": False,
                "current_route_details": None,
            }

    # Логика построения маршрута
    if not current_events:
        logger.warning(
            "clarify_address_or_build_route_node: No current events, cannot build route."
        )
        return {
            "current_route_details": None,
            "status_message_to_user": "Нет мероприятий для построения маршрута.",
        }

    # Если только одно мероприятие и не указан адрес пользователя, то маршрут не строится (согласно инструкции п1)
    if len(current_events) == 1 and not user_start_coords:
        logger.info(
            "clarify_address_or_build_route_node: One event and no user address, route not built."
        )
        return {
            "current_route_details": None,
            "is_full_plan_with_route_proposed": False,
        }  # План без маршрута

    # Формируем точки для маршрута
    points_for_api: List[Dict[str, Any]] = []
    start_location_for_api: Optional[LocationModel] = None

    if user_start_coords:
        start_location_for_api = LocationModel(
            lon=user_start_coords["lon"],
            lat=user_start_coords["lat"],
            address_string=collected_data.get("user_start_address_original"),
        )
    elif (
        len(current_events) > 1
    ):  # Если адреса пользователя нет, но мероприятий больше одного, строим от первого
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
        elif (
            first_event.place_address
        ):  # Если у первого мероприятия нет координат, но есть адрес
            logger.debug(
                f"clarify_address_or_build_route_node: Geocoding first event address: '{first_event.place_address}'"
            )
            city_for_geocoding = collected_data.get(
                "city_name", ""
            )  # Город должен быть
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
                    f"clarify_address_or_build_route_node: Could not geocode first event address for routing: {first_event.place_address}"
                )
                # Ошибка, не можем построить маршрут
                return {
                    "current_route_details": None,
                    "status_message_to_user": f"Не удалось определить координаты первого мероприятия '{first_event.name}' для построения маршрута.",
                }
        else:  # Нет ни координат ни адреса у первого мероприятия
            logger.error(
                f"clarify_address_or_build_route_node: First event '{first_event.name}' has no coordinates or address for routing."
            )
            return {
                "current_route_details": None,
                "status_message_to_user": f"У мероприятия '{first_event.name}' не указан адрес или координаты, маршрут не построить.",
            }
    else:  # Меньше 2х мероприятий и нет адреса пользователя - маршрут не нужен
        logger.info(
            "clarify_address_or_build_route_node: Not enough points for routing and no user_start_address."
        )
        return {
            "current_route_details": None,
            "is_full_plan_with_route_proposed": False,
        }

    event_points_for_api: List[LocationModel] = []
    # Если старт от пользователя, то все current_events идут в event_points
    # Если старт от первого мероприятия, то оно уже в start_location_for_api, а остальные в event_points
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
        elif (
            event_obj.place_address
        ):  # Если нет координат, но есть адрес, пытаемся геокодировать
            logger.debug(
                f"clarify_address_or_build_route_node: Geocoding event address for routing: '{event_obj.place_address}'"
            )
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
                    f"clarify_address_or_build_route_node: Could not geocode address for event '{event_obj.name}' ({event_obj.place_address}). Skipping for route."
                )
                # Можно решить, прерывать ли построение маршрута или строить без этой точки
                # Пока пропускаем точку, если не удалось геокодировать
        else:  # Нет ни координат, ни адреса
            logger.warning(
                f"clarify_address_or_build_route_node: Event '{event_obj.name}' has no coordinates or address. Skipping for route."
            )

    if not start_location_for_api or (
        len(current_events) > 0
        and not event_points_for_api
        and len(current_events) > (1 if not user_start_coords else 0)
    ):
        # Если начальная точка не определена, или есть мероприятия, но ни для одного не удалось получить координаты/адрес
        logger.error(
            "clarify_address_or_build_route_node: Not enough valid points with coordinates to build a route."
        )
        return {
            "current_route_details": None,
            "status_message_to_user": "Не удалось определить координаты для мероприятий, чтобы построить маршрут.",
        }

    # Если только start_location_for_api и нет event_points_for_api (например, одно мероприятие и это оно), то маршрут не нужен.
    if start_location_for_api and not event_points_for_api and len(current_events) <= 1:
        logger.info(
            "clarify_address_or_build_route_node: Only a single destination point, route not strictly needed from tool."
        )
        # Возвращаем "успех", но с нулевой длительностью, агент это обработает
        return {
            "current_route_details": RouteDetails(
                status="success",
                duration_seconds=0,
                duration_text="0 мин",
                distance_meters=0,
                distance_text="0 км",
            ),
            "is_full_plan_with_route_proposed": False,  # Маршрут не "строился"
        }

    tool_args = RouteBuilderToolArgs(
        start_point=start_location_for_api,
        event_points=event_points_for_api,
        # transport_type по умолчанию 'driving'
    )

    logger.info(
        f"clarify_address_or_build_route_node: Calling route_builder_tool with args: {tool_args.model_dump_json(indent=2, exclude_none=True)}"
    )
    route_data_dict = await route_builder_tool.ainvoke(
        tool_args.model_dump(exclude_none=True)
    )

    try:
        route_details_obj = RouteDetails(**route_data_dict)
        if route_details_obj.status == "success":
            logger.info(
                f"clarify_address_or_build_route_node: Route successfully built. Duration: {route_details_obj.duration_text}"
            )
            return {
                "current_route_details": route_details_obj,
                "is_full_plan_with_route_proposed": True,
            }
        else:
            logger.error(
                f"clarify_address_or_build_route_node: Route building failed by tool. Status: {route_details_obj.status}, Msg: {route_details_obj.error_message}"
            )
            return {
                "current_route_details": route_details_obj,
                "status_message_to_user": f"Не удалось построить маршрут: {route_details_obj.error_message}",
                "is_full_plan_with_route_proposed": False,
            }
    except ValidationError as ve:
        logger.error(
            f"clarify_address_or_build_route_node: Validation error for route_data: {route_data_dict}. Error: {ve}"
        )
        return {
            "current_route_details": None,
            "status_message_to_user": "Ошибка при обработке данных маршрута.",
            "is_full_plan_with_route_proposed": False,
        }


# --- Узел 6: Представление полного плана (мероприятия + маршрут) ---
async def present_full_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_full_plan_node executing...")
    current_events: Optional[List[Event]] = state.get("current_events")
    current_route: Optional[RouteDetails] = state.get("current_route_details")

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

    if current_route and current_route.status == "success":
        response_parts.append("\nМаршрут:")
        if current_route.duration_text and current_route.distance_text:
            response_parts.append(
                f"  Общее время в пути: {current_route.duration_text}, расстояние: {current_route.distance_text}."
            )
        else:  # Если только одно мероприятие и маршрут "нулевой"
            response_parts.append("  Вы уже на месте или маршрут не требуется.")

    elif current_route and current_route.status != "success":
        response_parts.append(
            f"\nМаршрут: Не удалось построить ({current_route.error_message or 'причина неизвестна'})."
        )
    elif (
        not current_route and len(current_events) > 1
    ):  # Маршрут должен был быть, но его нет
        response_parts.append(
            "\nМаршрут: Построение маршрута не удалось или не запрашивалось для одной точки."
        )
    elif (
        not current_route
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
        "awaiting_final_confirmation": True,  # Ожидаем подтверждения
        "is_full_plan_with_route_proposed": True,  # Флаг, что полный план предложен
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
