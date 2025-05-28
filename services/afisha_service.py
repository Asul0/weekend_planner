import aiohttp
import logging
import asyncio
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any

from config.settings import settings  # Используем наш settings

# from services.gis_service import get_coords_from_address # Закомментировано, т.к. геокодирование из Афиши убрано

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.LOG_LEVEL.upper()) # Глобальная настройка

CREATION_TYPES_AFISHA_INTERNAL = {  # Переименовал, чтобы избежать конфликта имен, если CREATION_TYPES есть где-то еще
    "Concert": "Concert",
    "Performance": "Performance",
    "UserEvent": "UserEvent",
    "Excursion": "Excursion",
    "Movie": "Movie",
    "Event": "Event",
    "Admission": "Admission",
    "SportEvent": "SportEvent",
    "ANY": None,
}


async def _make_afisha_request_internal(  # Переименовал
    session: aiohttp.ClientSession,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any] | List[Any]]:
    full_url = f"{settings.AFISHA_PROXY_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {}

    logger.debug(f"Afisha Proxy Request: GET {full_url} with params {params}")
    try:
        async with session.get(
            full_url,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            logger.debug(
                f"Afisha Proxy Response: Status {response.status} for {response.url}"
            )
            if response.status >= 400:
                error_body = await response.text()
                logger.error(
                    f"Afisha Proxy Client Response Error for {full_url}: {response.status}. Body: {error_body[:500]}"
                )
                return None

            data = await response.json()
            logger.debug(
                f"Afisha Proxy Response data for {response.url} (first 500 chars): {str(data)[:500]}"
            )
            return data
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Afisha Proxy ClientConnectorError for {endpoint}: {e}")
        return None
    except asyncio.TimeoutError:
        logger.error(
            f"Afisha Proxy Request timeout for {endpoint} with params {params}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error in Afisha Proxy request to {endpoint}: {e}",
            exc_info=True,
        )
        return None


async def fetch_cities_internal() -> List[Dict[str, Any]]:  # Переименовал
    async with aiohttp.ClientSession() as session:
        params: Dict[str, Any] = {}
        data = await _make_afisha_request_internal(session, "/v3/cities", params=params)
        if data is None or not isinstance(data, list):
            logger.error(
                f"Afisha: Failed to fetch cities or invalid format. Response: {str(data)[:200]}"
            )
            return []

        cities_processed = []
        for city_data in data:
            if (
                isinstance(city_data, dict)
                and city_data.get("Id")
                and city_data.get("Name")
            ):
                cities_processed.append(
                    {
                        "id": city_data["Id"],
                        "name": city_data["Name"],
                        "name_lower": city_data[
                            "Name"
                        ].lower(),  # Для регистронезависимого поиска
                    }
                )
        logger.info(f"Afisha: Fetched {len(cities_processed)} cities.")
        return cities_processed


async def search_sessions_internal(
    city_id: int,
    date_from: date,
    date_to: date,
    creation_type_key: Optional[str] = "ANY",
    min_start_time_naive: Optional[datetime] = None,
    max_start_time_naive: Optional[datetime] = None, # Новый аргумент
    max_budget_per_person: Optional[int] = None,
    exclude_session_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    all_found_sessions: List[Dict[str, Any]] = []
    date_from_str = date_from.isoformat()
    date_to_str = date_to.isoformat()
    cursor = None
    actual_creation_type = CREATION_TYPES_AFISHA_INTERNAL.get(creation_type_key)

    logger.info(
        f"Afisha: Searching sessions: city_id={city_id}, date={date_from_str}-{date_to_str}, "
        f"type_key='{creation_type_key}' (API type: {actual_creation_type}), budget={max_budget_per_person}, "
        f"min_start={min_start_time_naive.isoformat() if min_start_time_naive else None}, " # Обновлен лог
        f"max_start={max_start_time_naive.isoformat() if max_start_time_naive else None}"   # Обновлен лог
    )

    async with aiohttp.ClientSession() as session:
        page_count = 0
        max_pages_to_fetch = 20 

        while page_count < max_pages_to_fetch:
            page_count += 1
            creations_params: Dict[str, Any] = {
                "CityId": city_id, "DateFrom": date_from_str, "DateTo": date_to_str, "Limit": 25,
            }
            if actual_creation_type:
                creations_params["CreationType"] = actual_creation_type
            if cursor:
                creations_params["Cursor"] = cursor

            logger.debug(f"Afisha: Fetching creations page {page_count} with params: {creations_params}")
            creations_data_page = await _make_afisha_request_internal(session, "/v3/creations/page", params=creations_params)

            if creations_data_page is None or not isinstance(creations_data_page, dict):
                logger.error(f"Afisha: Failed to fetch creations page {page_count} for city {city_id}.")
                break

            creations_on_page = creations_data_page.get("Creations", [])
            cursor = creations_data_page.get("cursor")

            if not creations_on_page:
                logger.info(f"Afisha: No more creations on page {page_count}. Cursor: {cursor}")
                if not cursor: break
                continue 

            logger.info(f"Afisha: Found {len(creations_on_page)} creations on page {page_count}.")
            schedule_tasks = []
            for creation in creations_on_page:
                creation_id = creation.get("Id")
                if not creation_id: continue
                schedule_params: Dict[str, Any] = {"CityId": city_id, "DateFrom": date_from_str, "DateTo": date_to_str}
                task = _make_afisha_request_internal(session, f"/v3/creations/{creation_id}/schedule", params=schedule_params)
                schedule_tasks.append((task, creation))

            schedule_results_with_creations = await asyncio.gather(*(t for t, _ in schedule_tasks), return_exceptions=True)
            current_time_utc = datetime.now(timezone.utc)

            for i, schedule_result in enumerate(schedule_results_with_creations):
                original_creation = schedule_tasks[i][1]
                if isinstance(schedule_result, Exception):
                    logger.error(f"Afisha: Error fetching schedule for creation {original_creation.get('Id')}: {schedule_result}")
                    continue
                
                schedule_data_list = schedule_result
                if not isinstance(schedule_data_list, list): continue

                for schedule_block in schedule_data_list:
                    place_info = schedule_block.get("Place")
                    sessions_in_block = schedule_block.get("Sessions")
                    if not place_info or not isinstance(sessions_in_block, list): continue
                    
                    place_coords_lon: Optional[float] = None
                    place_coords_lat: Optional[float] = None
                    place_address = place_info.get("Address")
                    if place_info.get("Coordinates"):
                        try:
                            lon, lat = place_info["Coordinates"].get("Longitude"), place_info["Coordinates"].get("Latitude")
                            if lon is not None and lat is not None: place_coords_lon, place_coords_lat = float(lon), float(lat)
                        except (ValueError, TypeError): pass

                    for session_info in sessions_in_block:
                        session_id, session_datetime_str = session_info.get("Id"), session_info.get("SessionDateTime")
                        if not session_datetime_str or not session_id: continue
                        if exclude_session_ids and session_id in exclude_session_ids: continue

                        try:
                            session_dt_aware = datetime.fromisoformat(session_datetime_str)
                            session_dt_naive_event_tz = session_dt_aware.replace(tzinfo=None)
                            current_time_in_event_tz = current_time_utc.astimezone(session_dt_aware.tzinfo)
                            if session_dt_aware <= current_time_in_event_tz: continue
                        except ValueError: continue

                        if min_start_time_naive and session_dt_naive_event_tz < min_start_time_naive:
                            continue

                        # Новая фильтрация по max_start_time_naive
                        if max_start_time_naive and session_dt_naive_event_tz > max_start_time_naive:
                            logger.debug(f"Afisha: Filtering out session {session_id} by max_start_time_naive. Session: {session_dt_naive_event_tz}, Max: {max_start_time_naive}")
                            continue

                        min_price = session_info.get("MinPrice")
                        if max_budget_per_person is not None and min_price is not None and min_price > max_budget_per_person:
                            continue
                        
                        # --- ДОБАВЬТЕ city_id ---
                        all_found_sessions.append({
                            "session_id": session_id,
                            "afisha_id": original_creation.get("Id"),
                            "event_type_key": creation_type_key,
                            "name": original_creation.get("Name", "Название не указано"),
                            "place_name": place_info.get("Name", "Место не указано"),
                            "place_address": place_address,
                            "place_coords_lon": place_coords_lon,
                            "place_coords_lat": place_coords_lat,
                            "start_time_iso": session_datetime_str,
                            "start_time_naive_event_tz": session_dt_naive_event_tz,
                            "duration_minutes": original_creation.get("Duration"),
                            "duration_description": original_creation.get("DurationDescription"),
                            "min_price": min_price,
                            "price_text": (f"{int(min_price)} ₽" if min_price is not None else "Цена неизвестна"),
                            "rating": original_creation.get("Rating"),
                            "age_restriction": original_creation.get("AgeRestriction"),
                            "city_id": place_info.get("CityId"),  # <-- Вот это поле!
                        })
                        logger.info(f"Adding session: {original_creation.get('Name')} | {place_info.get('Name')} | {place_address} | {session_datetime_str}")
            if not cursor:
                logger.info("Afisha: No more cursor from /page, finished fetching creations.")
                break
        
        all_found_sessions.sort(key=lambda s: s["start_time_naive_event_tz"])
        logger.info(f"Afisha: Total sessions found and filtered: {len(all_found_sessions)}")
        return all_found_sessions
def filter_events_by_city(events: list[dict], city_name: str, city_id: int = None) -> list[dict]:
    """
    Оставляет только те события, которые проходят в указанном городе по CityId и адресу.
    Исключает сёла, посёлки и т.п., даже если в адресе есть название города.
    """
    filtered = []
    forbidden_prefixes = [
        "с.", "п.", "дер.", "пос.", "рп.", "ст.", "д.", "село", "поселок", "деревня", "станица"
    ]
    city_name_lower = city_name.lower()
    for event in events:
        event_city_id = event.get("city_id") or event.get("CityId")
        if not event_city_id and event.get("place_name"):
            place_info = event.get("place_info") or {}
            event_city_id = place_info.get("CityId")
        address = (event.get("place_address") or "").lower().strip()
        address_starts = address.split(",")[0]

        # 1. Исключаем по префиксам (село, посёлок и т.д.)
        if any(address_starts.startswith(prefix) for prefix in forbidden_prefixes):
            continue

        # 3. Основная фильтрация по CityId
        if city_id and event_city_id and int(event_city_id) == int(city_id):
            filtered.append(event)
            continue

        # 4. Фолбэк: если нет CityId, фильтруем по названию города в адресе
        if city_name_lower in address:
            filtered.append(event)
    logger.info(f"filter_events_by_city: {len(filtered)} из {len(events)} событий после фильтрации по CityId '{city_id}' и адресу")
    return filtered
