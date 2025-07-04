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
    max_start_time_naive: Optional[datetime] = None,  # Новый аргумент
    max_budget_per_person: Optional[int] = None,
    time_constraints_for_next_event: Optional[Dict[str, datetime]] = None,
    exclude_session_ids: Optional[List[int]] = None,
) -> List[Dict]:
    logger.info(
        f"EventSearchTool: Called with city_id={city_id}, date_from={date_from.isoformat()}, date_to={date_to.isoformat()}, "
        f"interests_keys={interests_keys}, "
        f"min_start_time_naive={min_start_time_naive.isoformat() if min_start_time_naive else None}, "
        f"max_start_time_naive={max_start_time_naive.isoformat() if max_start_time_naive else None}, "
        f"max_budget_per_person={max_budget_per_person}, time_constraints={time_constraints_for_next_event}, "
        f"exclude_session_ids={exclude_session_ids}"
    )

    found_events_data = []
    types_to_search = interests_keys if interests_keys else ["ANY"]
    all_raw_sessions = []

    for interest_key in types_to_search:
        logger.debug(f"EventSearchTool: Searching for interest_key='{interest_key}'")
        try:
            raw_sessions = await search_sessions_internal(
                city_id=city_id,
                date_from=date_from.date(),
                date_to=date_to.date(),
                creation_type_key=interest_key,
                min_start_time_naive=min_start_time_naive,
                max_start_time_naive=max_start_time_naive,  # Передаем дальше
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

    if not all_raw_sessions:
        logger.warning(
            "EventSearchTool: No events found after searching all interests."
        )
        return []

    processed_sessions = all_raw_sessions
    if time_constraints_for_next_event:
        start_after = time_constraints_for_next_event.get("start_after_naive")
        filtered_by_time_constraints = []
        for session_data in all_raw_sessions:
            session_start_naive = session_data.get("start_time_naive_event_tz")
            if not isinstance(session_start_naive, datetime):
                logger.warning(
                    f"EventSearchTool: Skipping session due to invalid start_time_naive_event_tz: {session_data.get('session_id')}"
                )
                continue
            if start_after and session_start_naive < start_after:
                continue
            filtered_by_time_constraints.append(session_data)
        logger.info(
            f"EventSearchTool: After time_constraints, {len(filtered_by_time_constraints)} sessions remaining."
        )
        processed_sessions = filtered_by_time_constraints

    for session_data in processed_sessions:
        try:
            event_obj = Event(
                **session_data
            )  # Предполагаем, что session_data уже содержит все поля для Event
            found_events_data.append(event_obj.model_dump())
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
