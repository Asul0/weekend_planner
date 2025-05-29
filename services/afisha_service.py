import aiohttp
import logging
import asyncio
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any

from config.settings import settings  # Используем наш settings

# from services.gis_service import get_coords_from_address # Закомментировано, т.к. геокодирование из Афиши убрано

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.LOG_LEVEL.upper()) # Глобальная настройка

CREATION_TYPES_AFISHA_INTERNAL = {
    "Movie": "Movie",
    "Performance": "Performance", # Театр, Опера и балет
    "Concert": "Concert",
    "SportEvent": "SportEvent",
    "Excursion": "Excursion",
    "Event": "Event",             # Фестивали, Вечеринки, Квизы, Лекции, Выставки (как вариант)
    "Admission": "Admission",     # Музеи, Выставки (как вариант)

    # Наши внутренние ключи, которые мапятся на существующие API типы
    "Exhibition": "Admission",    # Решили, что выставки будем пробовать через Admission
    "Festival": "Event",
    "StandUp": "Concert",
    "OperaBallet": "Performance", # Уже покрывается Performance
    "Party": "Event",
    "Quiz": "Event",
    "MasterClass": "Event",
    "Lecture": "Event",
    
    "Museum": "Admission",

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
    creation_type_key: Optional[str] = "ANY", # Это наш внутренний ключ ("Movie", "Museum", etc.)
    min_start_time_naive: Optional[datetime] = None,
    max_start_time_naive: Optional[datetime] = None,
    max_budget_per_person: Optional[int] = None,
    exclude_session_ids: Optional[List[int]] = None,
    # widget_key: Optional[str] = settings.AFISHA_WIDGET_KEY # Если WidgetKey нужен и хранится в settings
) -> List[Dict[str, Any]]:
    all_found_sessions: List[Dict[str, Any]] = []
    date_from_str = date_from.isoformat()
    date_to_str = date_to.isoformat()
    
    # Получаем API-совместимый тип из нашего словаря CREATION_TYPES_AFISHA_INTERNAL
    # creation_type_key - это наш внутренний ключ (например, "Museum")
    # api_type_to_request - это то, что пойдет в API (например, "Admission")
    api_type_to_request = CREATION_TYPES_AFISHA_INTERNAL.get(creation_type_key if creation_type_key else "ANY")

    logger.info(
        f"AfishaService: Searching sessions for city_id={city_id}, date_range=[{date_from_str} to {date_to_str}], "
        f"requested_internal_type_key='{creation_type_key}', API_type_to_request='{api_type_to_request}'. "
        f"Time constraints: min_start={min_start_time_naive}, max_start={max_start_time_naive}. "
        f"Budget: {max_budget_per_person}. Exclude IDs: {exclude_session_ids}."
    )

    current_cursor: Optional[str] = None # Курсор для первого запроса отсутствует
    page_count = 0
    max_pages_to_fetch = 10 # Ограничение, чтобы не уйти в бесконечный цикл при ошибках API

    async with aiohttp.ClientSession() as session:
        while page_count < max_pages_to_fetch:
            page_count += 1
            creations_params: Dict[str, Any] = {
                "CityId": city_id,
                "DateFrom": date_from_str,
                "DateTo": date_to_str,
                "Limit": 20, # Количество "произведений" (creations) на страницу. Можно увеличить.
                # "WidgetKey": widget_key, # Раскомментируйте, если WidgetKey обязателен и передается так
            }
            if api_type_to_request: # Передаем тип, только если он не None (т.е. не "ANY")
                creations_params["CreationType"] = api_type_to_request
            if current_cursor: # Для последующих страниц используем полученный курсор
                creations_params["Cursor"] = current_cursor

            logger.debug(f"AfishaService: Fetching creations page {page_count} with params: {creations_params}")
            
            creations_data_page = await _make_afisha_request_internal(
                session, "/v3/creations/page", params=creations_params
            )

            if creations_data_page is None or not isinstance(creations_data_page, dict):
                logger.error(f"AfishaService: Failed to fetch creations page {page_count} or invalid format for city {city_id}. Params: {creations_params}")
                break # Прерываем пагинацию при ошибке получения страницы

            creations_on_page = creations_data_page.get("Creations", [])
            api_returned_cursor = creations_data_page.get("cursor") # Курсор из ТЕКУЩЕГО ответа

            logger.debug(f"AfishaService: Page {page_count} response. Creations on page: {len(creations_on_page)}, Cursor from API: '{api_returned_cursor}'")

            if not creations_on_page:
                logger.info(f"AfishaService: No creations found on page {page_count} for params {creations_params}.")
                if not api_returned_cursor: # Если нет ни событий на странице, ни курсора на следующую
                    logger.info(f"AfishaService: And no further cursor. Stopping pagination.")
                    break 
                current_cursor = api_returned_cursor # Обновляем курсор и идем на следующую страницу
                continue # Переходим к следующей итерации цикла while

            # Обработка найденных "Произведений" (Creations)
            schedule_tasks = []
            for creation_item_data in creations_on_page: # creation_item_data - это словарь одного Creation
                if not isinstance(creation_item_data, dict):
                    logger.warning(f"AfishaService: Invalid creation item data: {creation_item_data}")
                    continue
                creation_id = creation_item_data.get("Id")
                if not creation_id:
                    logger.warning(f"AfishaService: Creation item missing ID: {creation_item_data}")
                    continue
                
                schedule_params_for_creation: Dict[str, Any] = {
                    "CityId": city_id,
                    "DateFrom": date_from_str,
                    "DateTo": date_to_str,
                    # "WidgetKey": widget_key, # Если нужен для /schedule
                }
                task = _make_afisha_request_internal(
                    session,
                    f"/v3/creations/{creation_id}/schedule",
                    params=schedule_params_for_creation,
                )
                schedule_tasks.append((task, creation_item_data)) # Передаем сам словарь creation_item_data

            schedule_results_with_creation_data = await asyncio.gather(
                *(t for t, _ in schedule_tasks), return_exceptions=True
            )
            
            current_time_utc = datetime.now(timezone.utc)

            for i, schedule_result_item in enumerate(schedule_results_with_creation_data):
                original_creation_dict = schedule_tasks[i][1] # Это наш словарь Creation
                
                if isinstance(schedule_result_item, Exception):
                    logger.error(f"AfishaService: Error fetching schedule for creation ID {original_creation_dict.get('Id')}: {schedule_result_item}")
                    continue

                schedule_data_list_for_creation = schedule_result_item # Это список блоков [ {place, sessions}, ... ]
                if not isinstance(schedule_data_list_for_creation, list):
                    logger.warning(f"AfishaService: Schedule data for creation ID {original_creation_dict.get('Id')} is not a list: {type(schedule_data_list_for_creation)}. Data: {str(schedule_data_list_for_creation)[:200]}")
                    continue

                # Извлекаем данные из original_creation_dict (один раз для всех его сеансов)
                actual_event_type_from_api = original_creation_dict.get("Type")
                creation_genres = original_creation_dict.get("Genres")
                creation_description = original_creation_dict.get("Description")
                creation_short_description = original_creation_dict.get("ShortDescription")
                creation_duration_minutes = original_creation_dict.get("Duration")
                creation_duration_description = original_creation_dict.get("DurationDescription")
                creation_rating = original_creation_dict.get("Rating")
                creation_age_restriction = original_creation_dict.get("AgeRestriction")
                creation_id_from_object = original_creation_dict.get("Id")
                creation_name = original_creation_dict.get("Name", "Название не указано")

                for schedule_block in schedule_data_list_for_creation:
                    if not isinstance(schedule_block, dict): continue
                    place_info = schedule_block.get("Place")
                    sessions_in_block = schedule_block.get("Sessions")
                    if not isinstance(place_info, dict) or not isinstance(sessions_in_block, list):
                        continue
                    
                    place_name = place_info.get("Name", "Место не указано")
                    place_address = place_info.get("Address")
                    place_coords_lon: Optional[float] = None
                    place_coords_lat: Optional[float] = None
                    coordinates_data = place_info.get("Coordinates")
                    if isinstance(coordinates_data, dict):
                        try:
                            lon = coordinates_data.get("Longitude")
                            lat = coordinates_data.get("Latitude")
                            if lon is not None and lat is not None:
                                place_coords_lon, place_coords_lat = float(lon), float(lat)
                        except (ValueError, TypeError):
                            logger.warning(f"AfishaService: Could not parse coordinates for place '{place_name}': {coordinates_data}")

                    for session_info in sessions_in_block:
                        if not isinstance(session_info, dict): continue
                        session_id = session_info.get("Id")
                        session_datetime_str = session_info.get("SessionDateTime")
                        min_price = session_info.get("MinPrice")

                        if not session_datetime_str or not session_id: continue
                        if exclude_session_ids and session_id in exclude_session_ids: continue
                        
                        try:
                            session_dt_aware = datetime.fromisoformat(session_datetime_str)
                            session_dt_naive_event_tz = session_dt_aware.replace(tzinfo=None)
                            # Сравниваем с текущим временем в том же часовом поясе, что и событие
                            current_time_in_event_tz = current_time_utc.astimezone(session_dt_aware.tzinfo).replace(tzinfo=None)
                            if session_dt_naive_event_tz <= current_time_in_event_tz:
                                continue
                        except ValueError as ve_date:
                            logger.warning(f"AfishaService: Error parsing session datetime '{session_datetime_str}': {ve_date}")
                            continue
                        
                        if min_start_time_naive and session_dt_naive_event_tz < min_start_time_naive: continue
                        if max_start_time_naive and session_dt_naive_event_tz > max_start_time_naive: continue
                        if max_budget_per_person is not None and min_price is not None:
                            try:
                                if float(min_price) > max_budget_per_person: continue
                            except ValueError:
                                logger.warning(f"AfishaService: Could not parse min_price '{min_price}' to float.")
                        
                        all_found_sessions.append({
                            "session_id": session_id,
                            "afisha_id": creation_id_from_object,
                            "event_type_key": creation_type_key, # Наш внутренний ключ ("Museum", "StandUp")
                            "actual_api_type": actual_event_type_from_api, # Фактический тип от API
                            "name": creation_name,
                            "place_name": place_name,
                            "place_address": place_address,
                            "place_coords_lon": place_coords_lon,
                            "place_coords_lat": place_coords_lat,
                            "start_time_iso": session_datetime_str,
                            "start_time_naive_event_tz": session_dt_naive_event_tz,
                            "duration_minutes": creation_duration_minutes,
                            "duration_description": creation_duration_description,
                            "min_price": int(min_price) if min_price is not None else None,
                            "price_text": (f"от {int(min_price)} ₽" if min_price is not None else None), # Уточнил формат
                            "rating": creation_rating,
                            "age_restriction": creation_age_restriction,
                            "genres": creation_genres,
                            "description": creation_description,
                            "short_description": creation_short_description,
                        })
            
            # После обработки всех "произведений" на текущей странице,
            # обновляем курсор для следующего запроса в цикле while.
            if not api_returned_cursor: 
                logger.info("AfishaService: No cursor returned by API on current page, assuming end of pagination.")
                break 
            current_cursor = api_returned_cursor
        
    # Сортировка всех найденных сеансов по времени начала
    all_found_sessions.sort(key=lambda s: s["start_time_naive_event_tz"])
    logger.info(f"AfishaService: Total sessions found and pre-filtered for internal key '{creation_type_key}' after all pages: {len(all_found_sessions)}")
    return all_found_sessions
