import logging
from typing import List, Optional, Dict
from datetime import datetime
import re # Добавим импорт re для более гибкого поиска

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import EventSearchToolArgs, Event
from services.afisha_service import search_sessions_internal

logger = logging.getLogger(__name__)


@tool("event_search_tool", args_schema=EventSearchToolArgs)
async def event_search_tool(
    city_id: int,
    date_from: datetime,
    date_to: datetime,
    city_name: Optional[str] = None,
    interests_keys: Optional[List[str]] = None,
    min_start_time_naive: Optional[datetime] = None,
    max_start_time_naive: Optional[datetime] = None,
    max_budget_per_person: Optional[int] = None,
    time_constraints_for_next_event: Optional[Dict[str, datetime]] = None,
    exclude_session_ids: Optional[List[int]] = None,
) -> List[Dict]:
    logger.info(
        f"EventSearchTool: Called with city_id={city_id}, city_name='{city_name}', "
        f"date_from={date_from.isoformat()}, date_to={date_to.isoformat()}, "
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
            raw_sessions_for_interest = await search_sessions_internal(
                city_id=city_id,
                date_from=date_from.date(),
                date_to=date_to.date(),
                creation_type_key=interest_key,
                min_start_time_naive=min_start_time_naive,
                max_start_time_naive=max_start_time_naive,
                max_budget_per_person=max_budget_per_person,
                exclude_session_ids=exclude_session_ids,
            )
            if raw_sessions_for_interest:
                all_raw_sessions.extend(raw_sessions_for_interest)
                logger.info(
                    f"EventSearchTool: Found {len(raw_sessions_for_interest)} raw sessions for interest '{interest_key}' from API."
                )
            else:
                logger.info(
                    f"EventSearchTool: No raw sessions found for interest '{interest_key}' from API."
                )
        except Exception as e:
            logger.error(
                f"EventSearchTool: Error calling afisha_service for interest '{interest_key}': {e}",
                exc_info=True,
            )

    if not all_raw_sessions:
        logger.warning(
            "EventSearchTool: No events found after searching all interests from API."
        )
        return []

    # --- ОБНОВЛЕННЫЙ БЛОК: Фильтрация по city_name ---
    if city_name:
        logger.info(f"EventSearchTool: Initial raw sessions count before city_name filter: {len(all_raw_sessions)} for target city '{city_name}'")
        filtered_sessions = []
        target_city_name_lower = city_name.lower()
        
        # Список известных префиксов для городов/населенных пунктов, которые мы хотим ИСКЛЮЧИТЬ, если они НЕ target_city_name
        # Эти префиксы указывают на то, что это, скорее всего, другой населенный пункт.
        # Мы будем искать эти слова, чтобы определить, не относится ли адрес к другому городу.
        # Важно: этот список НЕ должен содержать сам target_city_name.
        # Пример: если target_city_name="воронеж", то "г. воронеж" в адресе - это нормально.
        # А вот "г. павловск", "с. новая усмань" - это другие города/села.
        
        # Простая эвристика: ищем слова, обозначающие тип населенного пункта, или известные соседние города.
        # Это нужно будет тщательно подбирать и тестировать.
        # Для примера, если мы ищем "Воронеж", то "Павловск", "Новая Усмань", "Лиски", "Россошь" 
        # могут быть такими "чужими" городами.
        # Также можно добавить общие слова типа "село", "поселок", "деревня", "станица" и т.п.
        # если они встречаются в адресах из других регионов, которые API может ошибочно вернуть.
        
        # Нам нужен способ определить, является ли найденное в адресе название города/села *другим* городом,
        # а не частью адреса в *целевом* городе (например, улица "Павловская" в Воронеже).
        
        # Более надежный подход: если в адресе есть явное указание другого города (например, "г. Павловск"), отбрасываем.
        # Если в адресе нет явного указания города, но он очевидно не из целевого города (например, "с. Новая Усмань, ул. Ленина"), отбрасываем.
        # Если адрес не содержит явного указания *никакого* города, и мы ищем для Воронежа (city_id Воронежа), то считаем его относящимся к Воронежу.

        other_city_indicators_regex = r"\b(г\.|город|село|пос\.|поселок|деревня|станица|х\.|хутор)\s+([А-Яа-яЁё-]+)\b"
        # Для примера добавим конкретные известные "чужие" города для Воронежа, если они часто попадаются
        known_other_localities_lower = ["павловск", "новая усмань", "лиски", "семилуки", "рамОнь"] # Примерный список

        for session_data in all_raw_sessions:
            address = session_data.get("place_address")
            keep_session = True # По умолчанию сохраняем сессию

            if address:
                address_lower = address.lower()
                
                # 1. Проверяем на явное указание *другого* города/села с префиксами
                matches = re.finditer(other_city_indicators_regex, address, re.IGNORECASE)
                for match in matches:
                    found_locality_prefix = match.group(1).lower().replace(".", "") # "г", "село"
                    found_locality_name = match.group(2).lower() # "павловск", "новая усмань"
                    
                    # Если найденное название местности НЕ является целевым городом, то это другой город.
                    if found_locality_name != target_city_name_lower:
                        # Дополнительно проверяем, не является ли это улицей с таким же названием в целевом городе
                        # (например, улица "Павловская" в "Воронеже")
                        # Это сложная эвристика, пока упростим: если нашли "г. Павловск", и ищем "Воронеж" - отбрасываем.
                        if f"{found_locality_prefix} {found_locality_name}" in address_lower or \
                           found_locality_name in address_lower : # менее строго, если просто "павловск"
                            # Убедимся, что это не часть названия самого целевого города, если он составной (маловероятно для Воронежа)
                            if target_city_name_lower not in found_locality_name and found_locality_name not in target_city_name_lower :
                                logger.debug(f"EventSearchTool: Filtering out session {session_data.get('session_id')} ({session_data.get('name')}) "
                                             f"due to address containing other locality: '{found_locality_prefix} {found_locality_name}'. Address: '{address}'")
                                keep_session = False
                                break
                if not keep_session:
                    continue

                # 2. Проверяем на известные "чужие" населенные пункты без явных префиксов "г.", "с."
                # Это может быть более рискованно, так как может совпасть с названием улицы.
                # Используем с осторожностью и только для очень явных случаев.
                for other_loc in known_other_localities_lower:
                    if other_loc in address_lower and target_city_name_lower not in other_loc: 
                        # Пример: адрес "Павловск, ул. Ленина". Ищем для "Воронеж".
                        # Если "павловск" != "воронеж", то отфильтровываем.
                        # Это предотвратит фильтрацию "ул. Воронежская" в "Павловске", если бы мы искали "Павловск".
                        # И не отфильтрует "ул. Павловская" в "Воронеже".
                        
                        # Проверим, что это не просто улица с таким названием в целевом городе
                        # Если "павловск" есть в адресе, и мы ищем "воронеж",
                        # и сам "воронеж" в адресе отсутствует (чтобы не отбросить "г. Воронеж, ул. Павловская")
                        if target_city_name_lower not in address_lower:
                             # Более строгая проверка: ищем "павловск," или "павловск " чтобы убедиться что это город, а не часть слова
                            if re.search(r'\b' + re.escape(other_loc) + r'\b', address_lower):
                                logger.debug(f"EventSearchTool: Filtering out session {session_data.get('session_id')} ({session_data.get('name')}) "
                                            f"due to address containing known other locality: '{other_loc}'. Address: '{address}'")
                                keep_session = False
                                break
                if not keep_session:
                    continue
                
                # 3. Если сессия все еще не отфильтрована, и адрес НЕ содержит целевой город И НЕ содержит других городов,
                #    но при этом сам адрес короткий и похож на просто улицу без города (например, "Кольцовская, 35"),
                #    то мы предполагаем, что это адрес в целевом городе, так как API уже вернуло его для city_id.
                #    Если же адрес содержит что-то, что явно указывает на другой город (например, "г. Павловск"),
                #    то предыдущие фильтры должны были это поймать.
                #    Этот шаг больше для логгирования или очень специфических случаев.
                #    В данном сценарии, если предыдущие фильтры не сработали, мы считаем событие валидным для city_id.
                
                # Пример: адрес "Кольцовская, 35". Целевой город "Воронеж".
                # "воронеж" не в адресе. Других городов тоже нет. Значит, это, скорее всего, Воронеж.
                if target_city_name_lower not in address_lower and keep_session:
                     logger.debug(f"EventSearchTool: Keeping session {session_data.get('session_id')} ({session_data.get('name')}). "
                                 f"Address '{address}' does not explicitly mention '{target_city_name_lower}' but also no other identifiable city found. Assuming it belongs to target city_id.")

            else: # Если адреса нет, мы не можем быть уверены.
                  # Раньше мы отбрасывали. Теперь, если city_name задан, но адреса нет, это сомнительно.
                  # Однако, если API вернуло это для city_id, возможно, стоит оставить, если нет других признаков.
                  # Для большей строгости, если city_name указан, а адреса нет, можно отфильтровать.
                logger.debug(f"EventSearchTool: Session {session_data.get('session_id')} ({session_data.get('name')}) has no address. Keeping for now as it matches city_id, but location is uncertain.")
                # Если хотим быть строже:
                # logger.debug(f"EventSearchTool: Filtering out session {session_data.get('session_id')} ({session_data.get('name')}) due to missing address when city_name filter is active.")
                # keep_session = False

            if keep_session:
                filtered_sessions.append(session_data)

        all_raw_sessions = filtered_sessions
        logger.info(f"EventSearchTool: Sessions count after revised city_name filter: {len(all_raw_sessions)}")
        if not all_raw_sessions:
            logger.warning(
                f"EventSearchTool: No events remaining after revised filtering for city_name '{city_name}'."
            )
            return []
    # --- КОНЕЦ ОБНОВЛЕННОГО БЛОКА ---

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
            )
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