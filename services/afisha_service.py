import aiohttp
import logging
import asyncio
from datetime import date, datetime, timezone, time  # Added time
from typing import Optional, List, Dict, Any, Tuple
from datetime import timedelta

try:
    from config.settings import settings
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(
        "Не удалось импортировать settings из config.settings. Используется заглушка для AFISHA_PROXY_BASE_URL."
    )

    class MockAfishaSettings:
        AFISHA_PROXY_BASE_URL: str = "http://localhost:8000/afisha-proxy"
        LOG_LEVEL: str = "INFO"

    settings = MockAfishaSettings()

logger = logging.getLogger(__name__)

CREATION_TYPES_AFISHA = {
    "ANY": None,
    "CONCERT": "Concert",
    "PERFORMANCE": "Performance",
    "USER_EVENT": "UserEvent",
    "EXCURSION": "Excursion",
    "MOVIE": "Movie",
    "EVENT": "Event",
    "ADMISSION": "Admission",
    "SPORT_EVENT": "SportEvent",
    "STAND_UP": "Concert_OR_Event_STANDUP",
    "MUSEUM_EXHIBITION": "Admission_OR_Event_Excursion_MUSEUM",
}


async def _make_afisha_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any] | List[Any]]:
    if (
        not hasattr(settings, "AFISHA_PROXY_BASE_URL")
        or not settings.AFISHA_PROXY_BASE_URL
    ):
        logger.error("AFISHA_PROXY_BASE_URL не настроен в settings.")
        return None

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
                    f"Afisha Proxy Error for {full_url}: {response.status}. Body: {error_body[:2]}"
                )
                return None
            if response.status == 204:
                logger.debug(
                    f"Afisha Proxy Response: Status 204 No Content for {response.url}"
                )
                return []
            data = await response.json()
            if logger.isEnabledFor(logging.DEBUG):
                log_data = str(data)
                logger.debug(
                    f"Afisha Proxy Data for {response.url} (up to 500 chars): {log_data[:2]}"
                )
            return data
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Afisha Proxy ClientConnectorError for {endpoint}: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Afisha Proxy Timeout for {endpoint} with params {params}")
    except Exception as e:
        logger.error(
            f"Afisha Proxy Unexpected error for {endpoint}: {e}", exc_info=True
        )
    return None


async def fetch_cities() -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        data = await _make_afisha_request(session, "/v3/cities")
        if data is None or not isinstance(data, list):
            logger.error(
                f"Afisha: Failed to fetch cities or invalid format. Response: {str(data)[:200]}"
            )
            return []
        cities_processed = [
            {"id": c["Id"], "name": c["Name"], "name_lower": c["Name"].lower()}
            for c in data
            if isinstance(c, dict) and c.get("Id") and c.get("Name")
        ]
        logger.info(f"Afisha: Fetched {len(cities_processed)} cities.")
        return cities_processed


async def fetch_creation_details(
    session: aiohttp.ClientSession, creation_id: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(creation_id, (str, int)):
        logger.error(
            f"Afisha: Invalid creation_id type for fetching details: {creation_id} ({type(creation_id)})"
        )
        return None
    details_data = await _make_afisha_request(
        session, f"/v3/creations/{str(creation_id)}"
    )
    if details_data and isinstance(details_data, dict):
        return details_data
    logger.warning(f"Afisha: Could not fetch details for creation ID {creation_id}")
    return None


def _check_creation_filters(
    creation_details: Dict[str, Any],
    creation_name_from_page: str,
    filter_genres: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
) -> bool:

    if filter_genres:
        api_event_genres_original = creation_details.get("Genres", [])
        api_event_genres_lower = [
            str(g).lower().strip()
            for g in api_event_genres_original
            if isinstance(g, str)
        ]

        logger.debug(f"  Genre check for '{creation_name_from_page}':")
        logger.debug(f"    User filter_genres (lower): {filter_genres}")
        logger.debug(f"    API event_genres (original): {api_event_genres_original}")
        logger.debug(
            f"    API event_genres (processed lower): {api_event_genres_lower}"
        )

        genre_match_found = False
        if api_event_genres_lower:
            for user_genre_filter_token in filter_genres:
                if user_genre_filter_token in api_event_genres_lower:
                    logger.debug(
                        f"    >>> Genre match SUCCESS (API Genre): '{user_genre_filter_token}' found in API genres {api_event_genres_lower} for '{creation_name_from_page}'."
                    )
                    genre_match_found = True
                    break

        if not genre_match_found:
            rubric_type_lower = (
                creation_details.get("Rubric", {}).get("Type", "").lower()
            )
            if rubric_type_lower:
                rubric_to_genre_map = {
                    "museum": "музеи",
                    "exhibition": "выставки",
                    "gallery": "галереи",
                    "concerthall": "концерты",
                    "theatre": "спектакли",
                    "standupclub": "стендап",
                }
                implied_genre_from_rubric = rubric_to_genre_map.get(rubric_type_lower)

                if (
                    implied_genre_from_rubric
                    and implied_genre_from_rubric in filter_genres
                ):
                    logger.debug(
                        f"    >>> Genre match SUCCESS (Rubric): Rubric.Type='{rubric_type_lower}' implies genre '{implied_genre_from_rubric}' which is in filter_genres for '{creation_name_from_page}'."
                    )
                    genre_match_found = True

        if not genre_match_found:
            logger.debug(
                f"  Filter fail (genres/rubric): No match found for '{creation_name_from_page}' with filter_genres {filter_genres}."
            )
            logger.debug(f"    Checked API genres: {api_event_genres_lower}")
            logger.debug(
                f"    Checked Rubric.Type: {creation_details.get('Rubric', {}).get('Type', '').lower()}"
            )
            return False

    if filter_tags:
        api_event_tags_original = creation_details.get("Tags", [])
        api_event_tags_lower = [
            str(t).lower().strip()
            for t in api_event_tags_original
            if isinstance(t, str)
        ]

        logger.debug(f"  Tag check for '{creation_name_from_page}':")
        logger.debug(f"    User filter_tags (lower): {filter_tags}")
        logger.debug(f"    API event_tags (original): {api_event_tags_original}")
        logger.debug(f"    API event_tags (processed lower): {api_event_tags_lower}")

        tag_match_found = False
        if not filter_tags:
            tag_match_found = True
        elif api_event_tags_lower:
            for user_tag_filter in filter_tags:
                if user_tag_filter in api_event_tags_lower:
                    tag_match_found = True
                    logger.debug(
                        f"    >>> Tag match SUCCESS: '{user_tag_filter}' found in API tags."
                    )
                    break

        if not tag_match_found:
            logger.debug(
                f"  Filter fail (tags): No match found for '{creation_name_from_page}'."
            )
            return False

    logger.debug(
        f"  Filter pass in _check: '{creation_name_from_page}' (all specified filters passed or no filters were specified to fail)."
    )
    return True


async def search_sessions(
    city_id: int,
    date_from: datetime,
    date_to: datetime,
    user_creation_type_key: str = "ANY",
    filter_keywords_in_name: Optional[List[str]] = None,
    filter_genres: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    min_start_time_naive: Optional[time] = None,
    max_start_time_naive: Optional[time] = None,
    max_budget_per_person: Optional[int] = None,
    exclude_session_ids: Optional[List[int]] = None,
    limit_results: int = 100,
    city_name_for_filter: Optional[str] = None,  # Добавлен параметр для фильтрации
) -> List[Dict[str, Any]]:

    all_found_sessions: List[Dict[str, Any]] = []

    api_date_from_str = date_from.date().isoformat()
    api_date_to_str = (date_to.date() + timedelta(days=1)).isoformat()

    api_creation_types_to_query: List[Optional[str]] = []
    current_effective_keywords = (
        [kw.lower().strip() for kw in filter_keywords_in_name if kw.strip()]
        if filter_keywords_in_name
        else []
    )
    current_effective_genres = (
        [g.lower().strip() for g in filter_genres if g.strip()] if filter_genres else []
    )
    current_effective_tags = (
        [t.lower().strip() for t in filter_tags if t.strip()] if filter_tags else []
    )

    if user_creation_type_key == "ANY":
        api_creation_types_to_query.append(None)
    elif user_creation_type_key == "STAND_UP":
        api_creation_types_to_query.extend(
            [CREATION_TYPES_AFISHA["CONCERT"], CREATION_TYPES_AFISHA["EVENT"]]
        )
        current_effective_keywords.extend(
            ["стендап", "stand-up", "standup", "stand up", "открытый микрофон"]
        )
        current_effective_genres.extend(
            [
                "юмор",
                "humor",
                "комедия",
                "comedy",
                "стендап",
                "stand-up comedy",
                "разговорный жанр",
            ]
        )
    elif user_creation_type_key == "MUSEUM_EXHIBITION":
        api_creation_types_to_query.extend(
            [
                CREATION_TYPES_AFISHA["ADMISSION"],
                CREATION_TYPES_AFISHA["EVENT"],
                CREATION_TYPES_AFISHA["EXCURSION"],
            ]
        )
        current_effective_keywords.extend(
            [
                "музей",
                "выставка",
                "экспозиция",
                "галерея",
                "вернисаж",
                "арт-пространство",
                "биеннале",
            ]
        )
    elif user_creation_type_key == "SPORT_EVENT":
        api_creation_types_to_query.append(CREATION_TYPES_AFISHA["SPORT_EVENT"])
        current_effective_keywords.extend(
            [
                "матч",
                "футбол",
                "хоккей",
                "баскетбол",
                "волейбол",
                "турнир",
                "соревнования",
            ]
        )
    else:
        api_type_value = CREATION_TYPES_AFISHA.get(user_creation_type_key)
        if api_type_value is not None:
            api_creation_types_to_query.append(api_type_value)
        else:
            api_creation_types_to_query.append(None)

    api_creation_types_to_query = sorted(list(set(api_creation_types_to_query)))
    if len(api_creation_types_to_query) > 1 and None in api_creation_types_to_query:
        api_creation_types_to_query = [None]

    final_effective_keywords = (
        sorted(list(set(current_effective_keywords)))
        if current_effective_keywords
        else None
    )

    # Для сложных типов, где ключевые слова важнее жанров, не фильтруем по жанрам на этапе details
    genres_for_filter_check = (
        current_effective_genres
        if user_creation_type_key not in ["MUSEUM_EXHIBITION", "SPORT_EVENT"]
        else None
    )
    tags_for_filter_check = current_effective_tags

    logger.info(
        f"Afisha: Searching sessions: city_id={city_id}, city_name_filter='{city_name_for_filter}', date_range=[{api_date_from_str} to {api_date_to_str}], "
        f"user_type='{user_creation_type_key}', api_types={api_creation_types_to_query}, "
        f"keywords={final_effective_keywords}, genres_for_check={genres_for_filter_check}, "
        f"budget={max_budget_per_person}"
    )

    async with aiohttp.ClientSession() as session:
        processed_creation_ids_globally = set()

        for api_creation_type in api_creation_types_to_query:
            cursor, page_count, max_pages_per_type = None, 0, 5
            while (
                page_count < max_pages_per_type
                and len(all_found_sessions) < limit_results
            ):
                page_count += 1
                creations_params: Dict[str, Any] = {
                    "CityId": city_id,
                    "DateFrom": api_date_from_str,
                    "DateTo": api_date_to_str,
                    "Limit": 20,
                }
                if api_creation_type:
                    creations_params["CreationType"] = api_creation_type
                if cursor:
                    creations_params["Cursor"] = cursor

                creations_data_page = await _make_afisha_request(
                    session, "/v3/creations/page", params=creations_params
                )
                if not isinstance(creations_data_page, dict):
                    break

                creations_on_page = creations_data_page.get("Creations", [])
                cursor = creations_data_page.get("cursor")
                if not creations_on_page:
                    if not cursor:
                        break
                    continue

                creations_to_fetch_details_info = []
                for creation in creations_on_page:
                    creation_id = str(creation.get("Id", ""))
                    if (
                        not creation_id
                        or creation_id in processed_creation_ids_globally
                    ):
                        continue

                    if final_effective_keywords:
                        if not any(
                            kw in creation.get("Name", "").lower()
                            for kw in final_effective_keywords
                        ):
                            continue
                    creations_to_fetch_details_info.append(creation)

                if not creations_to_fetch_details_info:
                    if not cursor:
                        break
                    continue

                detail_tasks = [
                    fetch_creation_details(session, cr.get("Id"))
                    for cr in creations_to_fetch_details_info
                ]
                creation_details_list = await asyncio.gather(
                    *detail_tasks, return_exceptions=True
                )

                creations_for_schedule_request = []
                for i, details_result in enumerate(creation_details_list):
                    original_creation = creations_to_fetch_details_info[i]
                    processed_creation_ids_globally.add(
                        str(original_creation.get("Id"))
                    )
                    if isinstance(details_result, Exception) or not details_result:
                        if not genres_for_filter_check and not tags_for_filter_check:
                            creations_for_schedule_request.append(
                                (original_creation, {})
                            )
                        continue

                    if _check_creation_filters(
                        details_result,
                        original_creation.get("Name", ""),
                        genres_for_filter_check,
                        tags_for_filter_check,
                    ):
                        creations_for_schedule_request.append(
                            (original_creation, details_result)
                        )

                if not creations_for_schedule_request:
                    if not cursor:
                        break
                    continue

                schedule_tasks = [
                    (
                        _make_afisha_request(
                            session,
                            f"/v3/creations/{cr_data[0].get('Id')}/schedule",
                            params={
                                "CityId": city_id,
                                "DateFrom": api_date_from_str,
                                "DateTo": api_date_to_str,
                            },
                        ),
                        cr_data[0],
                        cr_data[1],
                    )
                    for cr_data in creations_for_schedule_request
                ]

                schedule_results = await asyncio.gather(
                    *(t[0] for t in schedule_tasks), return_exceptions=True
                )

                current_time_utc = datetime.now(timezone.utc)
                for i, schedule_data in enumerate(schedule_results):
                    original_creation_data = schedule_tasks[i][1]
                    creation_details_data = schedule_tasks[i][2]

                    if isinstance(schedule_data, Exception) or not isinstance(
                        schedule_data, list
                    ):
                        continue

                    for schedule_block in schedule_data:
                        place_info = schedule_block.get("Place")
                        sessions_in_block = schedule_block.get("Sessions")
                        if not place_info or not isinstance(sessions_in_block, list):
                            continue

                        # Новая логика фильтрации по городу
                        place_name = place_info.get("Name", "").lower()
                        place_address = place_info.get("Address", "").lower()
                        if (
                            city_name_for_filter
                            and city_name_for_filter.lower() not in place_address
                            and city_name_for_filter.lower() not in place_name
                        ):
                            logger.debug(
                                f"Skipping place '{place_name}' due to city mismatch (filter: '{city_name_for_filter}'). Address: '{place_address}'"
                            )
                            continue

                        place_coords_lon, place_coords_lat = None, None
                        if coords := place_info.get("Coordinates"):
                            try:
                                place_coords_lon, place_coords_lat = float(
                                    coords.get("Longitude")
                                ), float(coords.get("Latitude"))
                            except (ValueError, TypeError, AttributeError):
                                pass

                        for session_info in sessions_in_block:
                            session_id, session_datetime_str = session_info.get(
                                "Id"
                            ), session_info.get("SessionDateTime")
                            if not session_id or not session_datetime_str:
                                continue
                            if (
                                exclude_session_ids
                                and session_id in exclude_session_ids
                            ):
                                continue
                            if not session_info.get("IsSaleAvailable", True):
                                continue

                            try:
                                session_dt_aware = datetime.fromisoformat(
                                    session_datetime_str
                                )
                                if session_dt_aware <= current_time_utc:
                                    continue
                                session_dt_naive_event_tz = session_dt_aware.replace(
                                    tzinfo=None
                                )
                            except ValueError:
                                continue

                            if (
                                min_start_time_naive
                                and session_dt_naive_event_tz.time()
                                < min_start_time_naive
                            ):
                                continue
                            if (
                                max_start_time_naive
                                and session_dt_naive_event_tz.time()
                                > max_start_time_naive
                            ):
                                continue

                            min_price = session_info.get("MinPrice")
                            if (
                                max_budget_per_person is not None
                                and min_price is not None
                                and min_price > max_budget_per_person
                            ):
                                continue

                            max_price = session_info.get("MaxPrice")
                            price_display = "Цена неизвестна"
                            if min_price is not None:
                                price_display = f"от {int(min_price)} ₽"
                                if max_price is not None and max_price != min_price:
                                    price_display += f" до {int(max_price)} ₽"
                                elif max_price == min_price:
                                    price_display = f"{int(min_price)} ₽"

                            all_found_sessions.append(
                                {
                                    "session_id": session_id,
                                    "afisha_id": original_creation_data.get("Id"),
                                    "user_event_type_key": user_creation_type_key,
                                    "api_creation_type": original_creation_data.get(
                                        "Type"
                                    ),
                                    "name": original_creation_data.get(
                                        "Name", "Название не указано"
                                    ),
                                    "place_name": place_info.get(
                                        "Name", "Место не указано"
                                    ),
                                    "place_address": place_info.get("Address"),
                                    "place_coords_lon": place_coords_lon,
                                    "place_coords_lat": place_coords_lat,
                                    "start_time_iso": session_datetime_str,
                                    "start_time_naive_event_tz": session_dt_naive_event_tz,
                                    "duration_minutes": original_creation_data.get(
                                        "Duration"
                                    ),
                                    "duration_description": original_creation_data.get(
                                        "DurationDescription"
                                    ),
                                    "min_price": min_price,
                                    "max_price": max_price,
                                    "price_text": price_display,
                                    "rating": original_creation_data.get("Rating"),
                                    "age_restriction": original_creation_data.get(
                                        "AgeRestriction"
                                    ),
                                    "genres": creation_details_data.get("Genres", []),
                                    "tags": creation_details_data.get("Tags", []),
                                    "rubric": creation_details_data.get("Rubric", {}),
                                }
                            )
                            if len(all_found_sessions) >= limit_results:
                                break
                        if len(all_found_sessions) >= limit_results:
                            break
                    if len(all_found_sessions) >= limit_results:
                        break

                if not cursor:
                    break
            if page_count >= max_pages_per_type and not cursor:
                logger.info(f"Afisha: Reached max pages for type {api_creation_type}.")

    all_found_sessions.sort(key=lambda s: s["start_time_naive_event_tz"])
    logger.info(
        f"Afisha: Total sessions found and filtered: {len(all_found_sessions)}. Returning up to {limit_results}."
    )
    return all_found_sessions[:limit_results]


async def get_creation_genres_and_tags(creation_id: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        details = await fetch_creation_details(session, creation_id)
        if details:
            return {
                "id": details.get("Id"),
                "name": details.get("Name", "Неизвестное название"),
                "creation_type_api": details.get("Type"),
                "genres": details.get("Genres", []),
                "tags": details.get("Tags", []),
                "rubric": details.get("Rubric", {}),
                "full_details_dump": details,
            }
    return {
        "id": creation_id,
        "name": f"Не удалось получить детали для ID {creation_id}",
        "creation_type_api": None,
        "genres": [],
        "tags": [],
        "rubric": {},
        "full_details_dump": None,
    }
