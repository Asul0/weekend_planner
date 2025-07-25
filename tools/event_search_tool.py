# Файл: tools/event_search_tool.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, time

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import EventSearchToolArgs, Event

try:
    from services.afisha_service import search_sessions
except ImportError:
    try:
        import afisha_merged as afisha_service

        search_sessions = afisha_service.search_sessions
    except ImportError:
        logging.getLogger(__name__).critical(
            "Критическая ошибка: Не удалось импортировать сервис Афиши (search_sessions)."
        )

        async def search_sessions(*args, **kwargs) -> List[Dict[str, Any]]:  # Заглушка
            logging.getLogger(__name__).error(
                "ЗАГЛУШКА search_sessions вызвана из-за ошибки импорта сервиса Афиши."
            )
            return []


logger = logging.getLogger(__name__)


@tool("event_search_tool", args_schema=EventSearchToolArgs)
async def event_search_tool(
    city_id: int,
    date_from: datetime,
    date_to: datetime,
    user_creation_type_key: str = "ANY",
    filter_keywords_in_name: Optional[List[str]] = None,
    filter_genres: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    min_start_time_naive: Optional[str] = None,
    max_start_time_naive: Optional[str] = None,
    max_budget_per_person: Optional[int] = None,
    time_constraints_for_next_event: Optional[Dict[str, datetime]] = None,
    exclude_session_ids: Optional[List[int]] = None,
    city_name: Optional[str] = None,  # Добавлен аргумент
) -> List[Dict]:
    logger.info(
        f"EventSearchTool: Called with city_id={city_id}, city_name='{city_name}', "  # Добавлено в лог
        f"date_from={date_from.isoformat()}, date_to={date_to.isoformat()}, "
        f"user_creation_type_key='{user_creation_type_key}', "
        f"keywords={filter_keywords_in_name}, budget={max_budget_per_person}"
    )

    min_start_time_obj: Optional[time] = None
    if min_start_time_naive:
        try:
            min_start_time_obj = time.fromisoformat(min_start_time_naive)
        except ValueError:
            logger.warning(
                f"Invalid format for min_start_time_naive: '{min_start_time_naive}'. Ignoring."
            )

    max_start_time_obj: Optional[time] = None
    if max_start_time_naive:
        try:
            max_start_time_obj = time.fromisoformat(max_start_time_naive)
        except ValueError:
            logger.warning(
                f"Invalid format for max_start_time_naive: '{max_start_time_naive}'. Ignoring."
            )

    try:
        raw_sessions = await search_sessions(
            city_id=city_id,
            date_from=date_from,
            date_to=date_to,
            user_creation_type_key=user_creation_type_key,
            filter_keywords_in_name=filter_keywords_in_name,
            filter_genres=filter_genres,
            filter_tags=filter_tags,
            min_start_time_naive=min_start_time_obj,
            max_start_time_naive=max_start_time_obj,
            max_budget_per_person=max_budget_per_person,
            exclude_session_ids=exclude_session_ids,
            city_name_for_filter=city_name,  # Передаем имя города для фильтрации
        )
    except Exception as e:
        logger.error(
            f"EventSearchTool: Error calling afisha_service for type '{user_creation_type_key}': {e}",
            exc_info=True,
        )
        return []

    if not raw_sessions:
        logger.warning("EventSearchTool: No events found after afisha_service call.")
        return []

    found_events_data: List[Dict[str, Any]] = []
    for session_data in raw_sessions:
        try:
            # Валидация происходит внутри search_sessions, здесь можно доверять данным
            # или провести дополнительную валидацию через Pydantic-модель, как у вас и сделано
            event_obj = Event(**session_data)
            found_events_data.append(event_obj.model_dump(exclude_none=True))
        except ValidationError as ve:
            logger.warning(f"EventSearchTool: Validation error for session data: {ve}")
        except Exception as ex:
            logger.error(
                f"EventSearchTool: Unexpected error processing session data: {ex}",
                exc_info=True,
            )

    logger.info(
        f"EventSearchTool: Returning {len(found_events_data)} validated events."
    )
    return found_events_data
