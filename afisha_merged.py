# afisha_merged.py

import aiohttp
import logging
import asyncio
from datetime import date, datetime, timezone, time  # Добавил time
from typing import Optional, List, Dict, Any, Tuple  # Добавил Tuple

# Вместо прямого импорта settings, будем использовать заглушку,
# если основной settings не будет доступен при запуске этого скрипта автономно.
# Это позволит скрипту быть более самодостаточным для тестирования.
try:
    from config.settings import settings
except ImportError:
    logger_init = logging.getLogger("afisha_merged_init")  # Дал уникальное имя
    logger_init.warning(
        "Не удалось импортировать settings из config.settings. Используется заглушка для AFISHA_PROXY_BASE_URL и LOG_LEVEL."
    )

    class MockAfishaSettings:
        AFISHA_PROXY_BASE_URL: str = (
            "http://localhost:8000/afisha-proxy"  # Стандартный URL для прокси
        )
        LOG_LEVEL: str = "DEBUG"  # Для подробного логирования при тестах

    settings = MockAfishaSettings()

logger = logging.getLogger("afisha_merged")  # Дал уникальное имя логгеру модуля

# Более полный словарь типов, включая пользовательские (как STAND_UP, MUSEUM_EXHIBITION)
CREATION_TYPES_AFISHA = {
    "ANY": None,  # Для поиска по всем типам API
    "CONCERT": "Concert",
    "PERFORMANCE": "Performance",
    "USER_EVENT": "UserEvent",
    "EXCURSION": "Excursion",
    "MOVIE": "Movie",
    "EVENT": "Event",
    "ADMISSION": "Admission",
    "SPORT_EVENT": "SportEvent",
    "STAND_UP": "Concert_OR_Event_STANDUP",  # Ключ для пользовательского типа "Стендап"
    "MUSEUM_EXHIBITION": "Admission_OR_Event_Excursion_MUSEUM",  # Ключ для "Музеи и выставки"
}


async def _make_afisha_request(  # Убрал суффикс _internal
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
            timeout=aiohttp.ClientTimeout(total=60),  # Увеличил общий таймаут
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
            if response.status == 204:  # No Content
                logger.debug(
                    f"Afisha Proxy Response: Status 204 No Content for {response.url}"
                )
                return (
                    []
                )  # Возвращаем пустой список, а не None, т.к. это валидный ответ "нет данных"

            data = await response.json()
            if logger.isEnabledFor(
                logging.DEBUG
            ):  # Более эффективная проверка уровня лога
                log_data_str = str(data)
                logger.debug(
                    f"Afisha Proxy Response data for {response.url} (first 500 chars): {log_data_str[:500]}"
                )
            return data
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Afisha Proxy ClientConnectorError for {endpoint}: {e}")
    except asyncio.TimeoutError:
        logger.error(
            f"Afisha Proxy Request timeout for {endpoint} with params {params}"
        )
    except Exception as e:
        logger.error(
            f"An unexpected error in Afisha Proxy request to {endpoint}: {e}",
            exc_info=True,
        )
    return None


async def fetch_cities() -> List[Dict[str, Any]]:  # Убрал суффикс _internal
    async with aiohttp.ClientSession() as session:
        # params: Dict[str, Any] = {} # Не нужны параметры для /v3/cities
        data = await _make_afisha_request(session, "/v3/cities")  # Убрал params
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
                        "name_lower": city_data["Name"].lower(),
                    }
                )
        logger.info(f"Afisha: Fetched {len(cities_processed)} cities.")
        return cities_processed


async def fetch_creation_details(
    session: aiohttp.ClientSession, creation_id: str
) -> Optional[Dict[str, Any]]:
    """Вспомогательная функция для получения деталей 'творения' по его ID."""
    if not isinstance(creation_id, (str, int)):  # Проверка типа ID
        logger.error(
            f"Afisha: Invalid creation_id type for fetching details: {creation_id} ({type(creation_id)})"
        )
        return None
    details_data = await _make_afisha_request(
        session, f"/v3/creations/{str(creation_id)}"  # Явное приведение ID к строке
    )
    if details_data and isinstance(details_data, dict):
        return details_data
    logger.warning(
        f"Afisha: Could not fetch details for creation ID {creation_id}. Response: {str(details_data)[:200]}"
    )
    return None


def _check_creation_filters(
    creation_details: Dict[str, Any],
    creation_name_from_page: str,
    # filter_keywords_in_name: Optional[List[str]] = None, # Уже не используется здесь
    filter_genres: Optional[List[str]] = None,  # Ожидаются уже в lowercase
    filter_tags: Optional[List[str]] = None,  # Ожидаются уже в lowercase
) -> bool:
    """Проверяет 'творение' по фильтрам жанров и тегов на основе полученных деталей."""
    # creation_name_to_check_lower = creation_name_from_page.lower() # Не нужно для ключевых слов здесь

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
        # 1. Проверка по списку жанров из API
        if api_event_genres_lower:
            for user_genre_filter_token in filter_genres:
                if user_genre_filter_token in api_event_genres_lower:
                    logger.debug(
                        f"    >>> Genre match SUCCESS (API Genre): '{user_genre_filter_token}' found in API genres {api_event_genres_lower} for '{creation_name_from_page}'."
                    )
                    genre_match_found = True
                    break

        # 2. Если не найдено в API Genres, проверяем Rubric.Type
        if not genre_match_found:
            rubric_data = creation_details.get("Rubric", {})
            rubric_type_lower = (
                rubric_data.get("Type", "").lower()
                if isinstance(rubric_data, dict)
                else ""
            )

            if rubric_type_lower:
                rubric_to_genre_map = {
                    "museum": "музеи",  # Канонический термин для сопоставления
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
        # if not filter_tags: # Это условие избыточно, т.к. filter_tags уже проверен выше
        #     tag_match_found = True
        if api_event_tags_lower:  # Проверяем только если есть теги от API
            for user_tag_filter in filter_tags:
                if user_tag_filter in api_event_tags_lower:
                    tag_match_found = True
                    logger.debug(
                        f"    >>> Tag match SUCCESS: '{user_tag_filter}' found in API tags."
                    )
                    break

        if not tag_match_found:  # Если фильтр по тегам был, но совпадений не найдено
            logger.debug(
                f"  Filter fail (tags): No match found for '{creation_name_from_page}' with tags {filter_tags} in API tags {api_event_tags_lower}."
            )
            return False

    logger.debug(
        f"  Filter pass in _check_creation_filters: '{creation_name_from_page}' (all specified filters passed or no filters were specified to fail)."
    )
    return True


async def search_sessions(  # Убрал суффикс _internal, добавил новые аргументы
    city_id: int,
    date_from: date,
    date_to: date,
    user_creation_type_key: str = "ANY",  # Был creation_type_key, переименовал для ясности
    filter_keywords_in_name: Optional[List[str]] = None,  # Новый аргумент
    filter_genres: Optional[List[str]] = None,  # Новый аргумент
    filter_tags: Optional[List[str]] = None,  # Новый аргумент
    min_start_time_naive: Optional[time] = None,  # Изменил тип на time для удобства
    max_start_time_naive: Optional[time] = None,  # Изменил тип на time
    max_budget_per_person: Optional[int] = None,
    exclude_session_ids: Optional[List[int]] = None,
    limit_results: int = 100,  # Увеличил дефолтный лимит для тестов
) -> List[Dict[str, Any]]:
    all_found_sessions: List[Dict[str, Any]] = []
    date_from_str = date_from.isoformat()
    date_to_str = date_to.isoformat()

    api_creation_types_to_query: List[Optional[str]] = []

    # Приведение пользовательских фильтров к нижнему регистру и очистка
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

    # Логика определения API типов и добавления дефолтных фильтров
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
        current_effective_genres.extend(
            [
                "музеи",
                "выставки",
                "галереи",
                "искусство",
                "фотография",
                "скульптура",
                "живопись",
                "современное искусство",
                "классическое искусство",
                "исторические выставки",
                "культурное наследие",
                "экспозиции",
                "экскурсии",
                "painting",
                "classicalart",
                "modernart",
                "art",
                "exhibition",
                "excursion",
                "photo",
                "sculpture",
                "history",
                "heritage",
                "museums",
                "galleries",
            ]
        )
    else:
        # Для простых типов, напрямую из словаря
        api_type_value = CREATION_TYPES_AFISHA.get(user_creation_type_key)
        if api_type_value is not None:
            api_creation_types_to_query.append(api_type_value)
        else:  # Если ключ не найден, считаем это как ANY
            logger.warning(
                f"Unknown user_creation_type_key: '{user_creation_type_key}'. Defaulting to 'ANY' (all API types)."
            )
            api_creation_types_to_query.append(None)

    # Очистка и финализация списков API типов для запроса
    api_creation_types_to_query = sorted(list(set(api_creation_types_to_query)))
    if len(api_creation_types_to_query) > 1 and None in api_creation_types_to_query:
        api_creation_types_to_query = [
            None
        ]  # Если есть 'ANY', он перекрывает остальные

    # Финальные эффективные фильтры (уникальные, отсортированные)
    final_effective_keywords = (
        sorted(list(set(current_effective_keywords)))
        if current_effective_keywords
        else None
    )
    final_effective_genres = (
        sorted(list(set(current_effective_genres)))
        if current_effective_genres
        else None
    )
    final_effective_tags = (
        sorted(list(set(current_effective_tags))) if current_effective_tags else None
    )

    logger.info(
        f"Afisha: Searching sessions: city_id={city_id}, date_range=[{date_from_str} to {date_to_str}], "
        f"user_type='{user_creation_type_key}', resolved_api_types={api_creation_types_to_query}, "
        f"keywords={final_effective_keywords}, genres={final_effective_genres}, tags={final_effective_tags}, "
        f"budget={max_budget_per_person}, "
        f"min_start_time={min_start_time_naive.isoformat() if min_start_time_naive else 'Any'}, "
        f"max_start_time={max_start_time_naive.isoformat() if max_start_time_naive else 'Any'}, "
        f"limit_results={limit_results}"
    )

    async with aiohttp.ClientSession() as session:
        processed_creation_ids_globally = (
            set()
        )  # Для дедупликации творений между запросами к разным типам API

        for api_creation_type_for_request in api_creation_types_to_query:
            if len(all_found_sessions) >= limit_results:
                logger.info(
                    f"Afisha: Reached limit_results ({limit_results}) before processing all API types. Stopping."
                )
                break

            cursor = None  # Сбрасываем курсор для каждого нового типа API
            page_count = 0
            # max_pages_per_type = 20 # Из вашего "основного" кода, можно настроить
            max_pages_per_type = 5  # Уменьшил для ускорения тестов, можно вернуть 20

            logger.info(
                f"Afisha: Querying for API CreationType: {api_creation_type_for_request or 'ANY (all)'}"
            )

            while (
                page_count < max_pages_per_type
                and len(all_found_sessions)
                < limit_results  # Проверяем лимит перед каждым запросом страницы
            ):
                page_count += 1
                creations_params: Dict[str, Any] = {
                    "CityId": city_id,
                    "DateFrom": date_from_str,
                    "DateTo": date_to_str,
                    "Limit": 20,  # Количество творений на странице
                }
                if api_creation_type_for_request:  # Если не None (не ANY)
                    creations_params["CreationType"] = api_creation_type_for_request
                if cursor:
                    creations_params["Cursor"] = cursor

                logger.debug(
                    f"Afisha: Fetching creations page {page_count} for API type '{api_creation_type_for_request or 'ANY'}' with params: {creations_params}"
                )
                creations_data_page = await _make_afisha_request(
                    session, "/v3/creations/page", params=creations_params
                )

                if creations_data_page is None or not isinstance(
                    creations_data_page, dict
                ):
                    logger.error(
                        f"Afisha: Failed to fetch creations page {page_count} for city {city_id}, API type '{api_creation_type_for_request or 'ANY'}'. Stopping for this API type."
                    )
                    break  # Прерываем цикл по страницам для этого типа API

                creations_on_page = creations_data_page.get("Creations", [])
                cursor = creations_data_page.get("cursor")

                if not creations_on_page:
                    logger.info(
                        f"Afisha: No more creations on page {page_count} for API type '{api_creation_type_for_request or 'ANY'}'. Cursor: {cursor}"
                    )
                    if (
                        not cursor
                    ):  # Если нет творений и нет курсора, то для этого типа API все
                        break
                    continue  # Если есть курсор, но нет творений (редко, но возможно), идем на след. страницу

                logger.info(
                    f"Afisha: Found {len(creations_on_page)} creations on page {page_count} for API type '{api_creation_type_for_request or 'ANY'}'."
                )

                # Шаг 1: Предварительная фильтрация по ключевым словам и сбор ID для запроса деталей
                creations_to_fetch_details_info: List[Dict[str, Any]] = []
                for creation_fp in creations_on_page:  # fp = from page
                    creation_id_str = str(creation_fp.get("Id", ""))
                    creation_name_fp = creation_fp.get("Name", "")

                    if (
                        not creation_id_str
                        or creation_id_str in processed_creation_ids_globally
                    ):
                        continue  # Пропускаем уже обработанные или без ID

                    # Предварительная фильтрация по ключевым словам в названии
                    if final_effective_keywords:
                        name_lower = creation_name_fp.lower()
                        if not any(kw in name_lower for kw in final_effective_keywords):
                            logger.debug(
                                f"  Pre-Filter FAIL (keywords): '{creation_name_fp}' (ID: {creation_id_str}) for keywords {final_effective_keywords}"
                            )
                            continue
                        logger.debug(
                            f"  Pre-Filter PASS (keywords): '{creation_name_fp}' (ID: {creation_id_str}) for keywords {final_effective_keywords}"
                        )

                    creations_to_fetch_details_info.append(creation_fp)

                if not creations_to_fetch_details_info:
                    if not cursor:
                        break
                    continue

                logger.debug(
                    f"  Creations to fetch details for (after keyword pre-filter): {[c.get('Name') for c in creations_to_fetch_details_info]}"
                )

                # Шаг 2: Запрос деталей для отобранных творений
                detail_tasks_map = (
                    {  # Карта для сопоставления результатов с исходными данными
                        str(c.get("Id")): (
                            fetch_creation_details(session, str(c.get("Id"))),
                            c,  # Сохраняем исходные данные творения со страницы
                        )
                        for c in creations_to_fetch_details_info
                    }
                )

                creation_details_results = await asyncio.gather(
                    *[task_tuple[0] for task_tuple in detail_tasks_map.values()],
                    return_exceptions=True,
                )

                # Шаг 3: Фильтрация по жанрам/тегам на основе деталей и сбор для запроса расписаний
                creations_for_schedule_request: List[
                    Tuple[Dict[str, Any], Dict[str, Any]]
                ] = []

                detail_results_list = list(
                    detail_tasks_map.values()
                )  # Для итерации в том же порядке
                for i, details_result_or_exc in enumerate(creation_details_results):
                    _task_tuple_ignored, original_creation_from_page = (
                        detail_results_list[i]
                    )
                    creation_id_str = str(original_creation_from_page.get("Id"))
                    creation_name_from_page = original_creation_from_page.get(
                        "Name", ""
                    )

                    processed_creation_ids_globally.add(
                        creation_id_str
                    )  # Отмечаем как обработанный (даже если неудачно)

                    if (
                        isinstance(details_result_or_exc, Exception)
                        or not details_result_or_exc
                    ):
                        logger.warning(
                            f"    Could not get details for '{creation_name_from_page}' (ID: {creation_id_str}). Exc: {details_result_or_exc if isinstance(details_result_or_exc, Exception) else 'No details'}"
                        )
                        # Если нет фильтров по жанрам/тегам, можем пропустить это творение без деталей
                        if not final_effective_genres and not final_effective_tags:
                            logger.debug(
                                f"      No genre/tag filters, adding '{creation_name_from_page}' for schedule check without details."
                            )
                            creations_for_schedule_request.append(
                                (original_creation_from_page, {})
                            )  # Пустые детали
                        else:
                            logger.debug(
                                f"      Skipping '{creation_name_from_page}' due to failed detail fetch and active genre/tag filters."
                            )
                        continue

                    creation_details = details_result_or_exc  # Это уже dict
                    if _check_creation_filters(
                        creation_details,
                        creation_name_from_page,
                        # final_effective_keywords, # Ключевые слова уже применены
                        final_effective_genres,
                        final_effective_tags,
                    ):
                        creations_for_schedule_request.append(
                            (original_creation_from_page, creation_details)
                        )
                    else:
                        logger.debug(
                            f"  Filter FAIL (_check_creation_filters): '{creation_name_from_page}' (ID: {creation_id_str})"
                        )

                if not creations_for_schedule_request:
                    if not cursor:
                        break
                    continue

                logger.info(
                    f"Afisha: {len(creations_for_schedule_request)} creations passed all filters. Fetching schedules..."
                )

                # Шаг 4: Запрос расписаний для окончательно отфильтрованных творений
                schedule_tasks_final_map = {  # Карта для сопоставления результатов
                    str(orig_c.get("Id")): (
                        _make_afisha_request(
                            session,
                            f"/v3/creations/{orig_c.get('Id')}/schedule",
                            params={
                                "CityId": city_id,
                                "DateFrom": date_from_str,
                                "DateTo": date_to_str,
                            },
                        ),
                        orig_c,  # original_creation_from_page
                        details_c,  # creation_details (может быть пустым, если детали не получены, но фильтры пройдены)
                    )
                    for orig_c, details_c in creations_for_schedule_request
                }

                schedule_results = await asyncio.gather(
                    *[
                        task_tuple[0]
                        for task_tuple in schedule_tasks_final_map.values()
                    ],
                    return_exceptions=True,
                )

                current_time_utc = datetime.now(timezone.utc)
                schedule_results_list = list(schedule_tasks_final_map.values())

                for i, schedule_result_or_exc in enumerate(schedule_results):
                    _task_ignored, original_creation, creation_details_for_item = (
                        schedule_results_list[i]
                    )

                    if len(all_found_sessions) >= limit_results:
                        break  # Проверка лимита перед обработкой каждого расписания

                    if isinstance(schedule_result_or_exc, Exception):
                        logger.error(
                            f"Afisha: Error fetching schedule for creation {original_creation.get('Name')} (ID: {original_creation.get('Id')}): {schedule_result_or_exc}"
                        )
                        continue

                    schedule_data_list = schedule_result_or_exc  # Это list
                    if not isinstance(
                        schedule_data_list, list
                    ):  # Доп. проверка, хотя _make_afisha_request должен вернуть list или None
                        logger.warning(
                            f"Afisha: Schedule data for {original_creation.get('Name')} is not a list: {type(schedule_data_list)}"
                        )
                        continue

                    for schedule_block in schedule_data_list:
                        if len(all_found_sessions) >= limit_results:
                            break

                        place_info = schedule_block.get("Place")
                        sessions_in_block = schedule_block.get("Sessions")
                        if not place_info or not isinstance(sessions_in_block, list):
                            continue

                        place_coords_lon: Optional[float] = None
                        place_coords_lat: Optional[float] = None
                        place_address = place_info.get("Address")

                        coords_data = place_info.get("Coordinates")
                        if isinstance(coords_data, dict):
                            try:
                                lon, lat = coords_data.get(
                                    "Longitude"
                                ), coords_data.get("Latitude")
                                if lon is not None and lat is not None:
                                    place_coords_lon, place_coords_lat = float(
                                        lon
                                    ), float(lat)
                            except (ValueError, TypeError):
                                pass  # Координаты могут отсутствовать или быть некорректными

                        for session_info in sessions_in_block:
                            if len(all_found_sessions) >= limit_results:
                                break

                            session_id_val = session_info.get(
                                "Id"
                            )  # Переименовал, чтобы не конфликтовать с модулем
                            session_datetime_str = session_info.get("SessionDateTime")
                            is_sale_available = session_info.get(
                                "IsSaleAvailable", True
                            )  # По умолчанию считаем, что продажа доступна

                            if not session_datetime_str or not session_id_val:
                                continue
                            if (
                                exclude_session_ids
                                and session_id_val in exclude_session_ids
                            ):
                                continue
                            if not is_sale_available:
                                logger.debug(
                                    f"  Session {session_id_val} for '{original_creation.get('Name')}' has no tickets (IsSaleAvailable=false). Skipping."
                                )
                                continue

                            try:
                                session_dt_aware = datetime.fromisoformat(
                                    session_datetime_str
                                )
                                session_dt_naive_event_tz = session_dt_aware.replace(
                                    tzinfo=None
                                )

                                # Сравнение с текущим временем: сеанс должен быть в будущем
                                # current_time_in_event_tz = current_time_utc.astimezone(session_dt_aware.tzinfo) # Это было в старом коде
                                # if session_dt_aware <= current_time_in_event_tz: # Сравнение aware объектов
                                if (
                                    session_dt_aware <= current_time_utc
                                ):  # Простое сравнение с UTC временем сейчас
                                    logger.debug(
                                        f"  Session {session_id_val} for '{original_creation.get('Name')}' is in the past. Session UTC: {session_dt_aware}, Current UTC: {current_time_utc}. Skipping."
                                    )
                                    continue
                            except ValueError:
                                logger.warning(
                                    f"  Invalid date format for session {session_id_val}: {session_datetime_str}"
                                )
                                continue

                            # Фильтрация по времени начала/окончания (используем только time часть)
                            session_time_naive = session_dt_naive_event_tz.time()
                            if (
                                min_start_time_naive
                                and session_time_naive < min_start_time_naive
                            ):
                                logger.debug(
                                    f"  Filtering out session {session_id_val} by min_start_time_naive. Session time: {session_time_naive}, Min: {min_start_time_naive}"
                                )
                                continue
                            if (
                                max_start_time_naive
                                and session_time_naive > max_start_time_naive
                            ):
                                logger.debug(
                                    f"  Filtering out session {session_id_val} by max_start_time_naive. Session time: {session_time_naive}, Max: {max_start_time_naive}"
                                )
                                continue

                            min_price = session_info.get("MinPrice")
                            max_price = session_info.get(
                                "MaxPrice"
                            )  # Получаем MaxPrice
                            if (
                                max_budget_per_person is not None
                                and min_price
                                is not None  # Проверяем, что min_price есть
                                and min_price > max_budget_per_person
                            ):
                                continue

                            # Формирование текста цены
                            price_text_parts = []
                            if min_price is not None:
                                price_text_parts.append(f"от {int(min_price)} ₽")
                            if max_price is not None and max_price != min_price:
                                price_text_parts.append(f"до {int(max_price)} ₽")

                            price_display = (
                                " ".join(price_text_parts)
                                if price_text_parts
                                else "Цена неизвестна"
                            )
                            if (
                                min_price is not None
                                and max_price is not None
                                and min_price == max_price
                            ):
                                price_display = f"{int(min_price)} ₽"

                            all_found_sessions.append(
                                {
                                    "session_id": session_id_val,
                                    "afisha_id": original_creation.get("Id"),
                                    "user_event_type_key": user_creation_type_key,  # Важно для понимания, по какому запросу пришло
                                    "api_creation_type": original_creation.get(
                                        "Type"
                                    ),  # Фактический тип из API
                                    "name": original_creation.get(
                                        "Name", "Название не указано"
                                    ),
                                    "place_name": place_info.get(
                                        "Name", "Место не указано"
                                    ),
                                    "place_address": place_address,
                                    "place_coords_lon": place_coords_lon,
                                    "place_coords_lat": place_coords_lat,
                                    "start_time_iso": session_datetime_str,
                                    "start_time_naive_event_tz": session_dt_naive_event_tz,
                                    "duration_minutes": original_creation.get(
                                        "Duration"
                                    ),
                                    "duration_description": original_creation.get(
                                        "DurationDescription"
                                    ),
                                    "min_price": min_price,
                                    "max_price": max_price,  # Добавлено
                                    "price_text": price_display,  # Обновлено
                                    "rating": original_creation.get("Rating"),
                                    "age_restriction": original_creation.get(
                                        "AgeRestriction"
                                    ),
                                    "genres": creation_details_for_item.get(
                                        "Genres", []
                                    ),  # Из деталей
                                    "tags": creation_details_for_item.get(
                                        "Tags", []
                                    ),  # Из деталей
                                    "rubric": creation_details_for_item.get(
                                        "Rubric", {}
                                    ),  # Из деталей
                                }
                            )
                        if len(all_found_sessions) >= limit_results:
                            break
                    if len(all_found_sessions) >= limit_results:
                        break

                if not cursor:
                    logger.info(
                        f"Afisha: No more cursor from /page for API type '{api_creation_type_for_request or 'ANY'}', finished fetching creations for this type."
                    )
                    break  # Выход из while по страницам для текущего типа API

            # Лог о завершении обработки страниц для текущего типа API
            if page_count >= max_pages_per_type:
                logger.info(
                    f"Afisha: Reached max_pages_per_type ({max_pages_per_type}) for API type '{api_creation_type_for_request or 'ANY'}'."
                )

        all_found_sessions.sort(key=lambda s: s["start_time_naive_event_tz"])
        logger.info(
            f"Afisha: Total sessions found and filtered: {len(all_found_sessions)}. Returning up to {limit_results} results."
        )
        return all_found_sessions[:limit_results]


async def get_creation_genres_and_tags(creation_id: str) -> Dict[str, Any]:
    """Публичная функция для получения жанров и тегов 'творения'."""
    async with aiohttp.ClientSession() as session:
        details = await fetch_creation_details(
            session, creation_id
        )  # Используем внутреннюю функцию
        if details:
            return {
                "id": details.get("Id"),
                "name": details.get("Name", "Неизвестное название"),
                "creation_type_api": details.get("Type"),  # Фактический тип из API
                "genres": details.get("Genres", []),
                "tags": details.get("Tags", []),
                "rubric": details.get("Rubric", {}),  # Добавлено поле Rubric
                "full_details_dump": details,  # Для отладки можно вернуть все детали
            }
    # Если детали не получены
    return {
        "id": creation_id,
        "name": f"Не удалось получить детали для ID {creation_id}",
        "creation_type_api": None,
        "genres": [],
        "tags": [],
        "rubric": {},
        "full_details_dump": None,
    }
