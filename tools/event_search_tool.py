import logging
from typing import List, Optional, Dict
from datetime import datetime

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import EventSearchToolArgs, Event  # Импортируем наши схемы
from services.afisha_service import search_sessions_internal  # Наш сервис для Афиши

logger = logging.getLogger(__name__)


@tool("event_search_tool", args_schema=EventSearchToolArgs)
async def event_search_tool(
    city_id: int,
    date_from: datetime,
    date_to: datetime,
    interests_keys: Optional[List[str]] = None,
    min_start_time_naive: Optional[datetime] = None,
    max_budget_per_person: Optional[int] = None,
    time_constraints_for_next_event: Optional[
        Dict[str, datetime]
    ] = None,  # Пока не используется в search_sessions_internal, но передается
    exclude_session_ids: Optional[List[int]] = None,
) -> List[Dict]:  # Возвращаем список словарей, соответствующих схеме Event
    """
    Ищет мероприятия по заданным критериям через API Афиши.
    Аргумент time_constraints_for_next_event (словарь с ключами 'start_after_naive' и/или 'end_before_naive')
    помогает найти мероприятие, которое можно посетить после/до другого, учитывая время.
    Все datetime аргументы должны быть уже в корректном формате.
    """
    logger.info(
        f"EventSearchTool: Called with city_id={city_id}, date_from={date_from.isoformat()}, date_to={date_to.isoformat()}, "
        f"interests_keys={interests_keys}, min_start_time_naive={min_start_time_naive.isoformat() if min_start_time_naive else None}, "
        f"max_budget_per_person={max_budget_per_person}, time_constraints={time_constraints_for_next_event}, "
        f"exclude_session_ids={exclude_session_ids}"
    )

    found_events_data = []

    types_to_search = interests_keys if interests_keys else ["ANY"]

    all_raw_sessions = []

    for interest_key in types_to_search:
        logger.debug(f"EventSearchTool: Searching for interest_key='{interest_key}'")
        try:
            # date_from и date_to для search_sessions_internal должны быть объектами date
            raw_sessions = await search_sessions_internal(
                city_id=city_id,
                date_from=date_from.date(),
                date_to=date_to.date(),
                creation_type_key=interest_key,  # interest_key должен быть ключом из CREATION_TYPES_AFISHA_INTERNAL
                min_start_time_naive=min_start_time_naive,
                max_budget_per_person=max_budget_per_person,
                exclude_session_ids=exclude_session_ids,
            )
            if raw_sessions:
                all_raw_sessions.extend(raw_sessions)
                logger.info(
                    f"EventSearchTool: Found {len(raw_sessions)} raw sessions for interest '{interest_key}'."
                )
            else:
                logger.info(
                    f"EventSearchTool: No raw sessions found for interest '{interest_key}'."
                )
        except Exception as e:
            logger.error(
                f"EventSearchTool: Error calling afisha_service for interest '{interest_key}': {e}",
                exc_info=True,
            )
            # Можно вернуть ошибку или продолжить с другими интересами
            # Пока просто логируем и продолжаем

    if not all_raw_sessions:
        logger.warning(
            "EventSearchTool: No events found after searching all interests."
        )
        return []

    # Обработка time_constraints_for_next_event (если они есть)
    # Эта логика должна применяться ПОСЛЕ получения всех сеансов от Афиши,
    # так как API Афиши не принимает такие точные временные рамки напрямую.
    # Предполагаем, что datetime объекты в time_constraints уже наивные.
    filtered_by_time_constraints = []
    if time_constraints_for_next_event:
        start_after = time_constraints_for_next_event.get("start_after_naive")
        # end_before = time_constraints_for_next_event.get("end_before_naive") # На данный момент не используется

        for session_data in all_raw_sessions:
            session_start_naive = session_data.get("start_time_naive_event_tz")
            if not isinstance(session_start_naive, datetime):  # Проверка типа
                logger.warning(
                    f"EventSearchTool: Skipping session due to invalid start_time_naive_event_tz: {session_data.get('session_id')}"
                )
                continue

            if start_after and session_start_naive < start_after:
                logger.debug(
                    f"EventSearchTool: Filtering out session {session_data.get('session_id')} starting at {session_start_naive} (before {start_after})"
                )
                continue
            # if end_before and session_start_naive > end_before: # Если понадобится ограничение "до"
            #     continue
            filtered_by_time_constraints.append(session_data)
        logger.info(
            f"EventSearchTool: After time_constraints, {len(filtered_by_time_constraints)} sessions remaining."
        )
        processed_sessions = filtered_by_time_constraints
    else:
        processed_sessions = all_raw_sessions

    # Преобразование ответа API Афиши в нашу схему Event и валидация
    for session_data in processed_sessions:
        try:
            # Убедимся, что все необходимые поля присутствуют или имеют значения по умолчанию
            # в соответствии со схемой Event
            event_dict = {
                "session_id": session_data.get("session_id"),
                "afisha_id": session_data.get("afisha_id"),
                "name": session_data.get("name", "Название не указано"),
                "event_type_key": session_data.get(
                    "event_type_key", "Unknown"
                ),  # Важно, чтобы это был ключ из CREATION_TYPES
                "place_name": session_data.get("place_name", "Место не указано"),
                "place_address": session_data.get("place_address"),
                "place_coords_lon": session_data.get("place_coords_lon"),
                "place_coords_lat": session_data.get("place_coords_lat"),
                "start_time_iso": session_data.get("start_time_iso"),
                "start_time_naive_event_tz": session_data.get(
                    "start_time_naive_event_tz"
                ),
                "duration_minutes": session_data.get("duration_minutes"),
                "min_price": session_data.get("min_price"),
                "price_text": session_data.get("price_text"),
                "duration_description": session_data.get("duration_description"),
                "rating": session_data.get("rating"),
                "age_restriction": session_data.get("age_restriction"),
            }
            # Валидация через Pydantic модель Event
            event_obj = Event(**event_dict)
            found_events_data.append(
                event_obj.model_dump()
            )  # Возвращаем как dict для совместимости с LLM
        except ValidationError as ve:
            logger.warning(
                f"EventSearchTool: Validation error for session data {session_data.get('session_id')}: {ve}"
            )
        except Exception as ex:
            logger.error(
                f"EventSearchTool: Unexpected error processing session data {session_data.get('session_id')}: {ex}",
                exc_info=True,
            )

    logger.info(
        f"EventSearchTool: Returning {len(found_events_data)} validated events."
    )
    return found_events_data
