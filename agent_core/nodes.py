import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, date, timedelta  # Убедимся, что date импортирован
import asyncio  # Для asyncio.gather в новой логике
import re

# Pydantic и Langchain для сообщений и схем
from pydantic import BaseModel, Field, ValidationError
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
    ParsedDateTime,
    AnalyzedFeedback,
    RouteSegment,
)
from services.afisha_service import fetch_cities_internal  # Убедимся, что есть
from services.gis_service import get_coords_from_address, get_route
from tools.datetime_parser_tool import datetime_parser_tool
from tools.event_search_tool import event_search_tool
from tools.route_builder_tool import route_builder_tool
from services.gis_service import (
    get_geocoding_details,
    get_route,
    GeocodingResult,
)  # <--- ИЗМЕНЕНО

# Инициализация логгера
logger = logging.getLogger(__name__)


# --- Узел 1: Извлечение начальной информации ---
async def extract_initial_info_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: extract_initial_info_node executing...")
    awaiting_clarification_field: Optional[str] = state.get("awaiting_clarification_for_field")
    logger.info(f"extract_initial_info_node: Received awaiting_clarification_for_field = '{awaiting_clarification_field}'")

    messages: List[BaseMessage] = state.get("messages", [])
    # Используем dict() для создания изменяемой копии, если collected_data существует, иначе пустой словарь
    current_collected_data_dict: dict = dict(state.get("collected_data", {})) 

    clarification_context_for_node: Optional[str] = None
    # new_clarification_needed будет использоваться для сбора полей, требующих уточнения в этом вызове узла
    new_clarification_needed_in_this_step: List[str] = []


    if not messages or not isinstance(messages[-1], HumanMessage):
        # Если нет сообщений или последнее не от человека (например, первый вызов графа без ввода)
        # Возвращаем текущее состояние без изменений, но с пустым clarification_context
        # Это важно, чтобы не зациклиться, если граф вызван без начальных данных
        logger.warning("extract_initial_info_node: No messages or last message is not HumanMessage. Returning current state.")
        # Если это первый вход, и мы ожидаем, что пользователь что-то скажет,
        # то, возможно, граф не должен был сюда попадать без UserMessage.
        # Однако, если это возможно, нужно гарантировать, что мы не зациклимся на запросе уточнений.
        # Если clarification_needed_fields уже есть в стейте, они останутся.
        # Если нет, то и запрашивать нечего.
        return {
            "collected_data": current_collected_data_dict,
            "messages": messages,
            "clarification_context": None, # Сбрасываем контекст, если он был
            "awaiting_clarification_for_field": awaiting_clarification_field # Сохраняем, если ждем что-то конкретное
        }

    user_query = messages[-1].content.strip()
    user_query_lower = user_query.lower()

    reset_commands = ["новый поиск", "начни сначала", "отмена", "сброс", "стоп", "reset"]
    if any(cmd in user_query_lower for cmd in reset_commands):
        logger.info(f"User requested reset with: '{user_query}'")
        reset_message = "Хорошо, начинаем новый поиск! Расскажите, что бы вы хотели найти: город, даты и ваши интересы. 😊"
        return {
            "collected_data": {}, # Полный сброс собранных данных
            "current_events": [],
            "current_route_details": None,
            "messages": messages + [AIMessage(content=reset_message)],
            "status_message_to_user": reset_message,
            "clarification_needed_fields": [], # Очищаем список полей для уточнения
            "clarification_context": None, # Очищаем контекст уточнения
            "awaiting_clarification_for_field": None, # Сбрасываем ожидаемое поле
            "is_initial_plan_proposed": False,
            "is_full_plan_with_route_proposed": False,
            "awaiting_final_confirmation": False,
            "pending_plan_modification_request": None,
            "previous_confirmed_collected_data": None,
            "previous_confirmed_events": None,
        }

    # --- Обработка ответа на предыдущий уточняющий вопрос ---
    if awaiting_clarification_field:
        logger.info(f"Processing '{user_query}' as clarification for '{awaiting_clarification_field}'")
        
        # Получаем текущий список полей, требующих уточнения, из collected_data
        # Это важно, так как new_clarification_needed_in_this_step - это для ТЕКУЩЕГО шага.
        # А current_collected_data_dict["clarification_needed_fields"] - это то, что было ДО этого шага.
        existing_clarification_fields = list(current_collected_data_dict.get("clarification_needed_fields", []))

        # Удаляем поле, по которому пришло уточнение, из списка необходимых уточнений
        if awaiting_clarification_field in existing_clarification_fields:
            existing_clarification_fields.remove(awaiting_clarification_field)
        
        # Применяем уточнение к current_collected_data_dict
        if awaiting_clarification_field == "city_name":
            current_collected_data_dict["city_name"] = user_query
            cities = await fetch_cities_internal() # Эта функция должна быть доступна
            found_city = next((c for c in cities if user_query.lower() in c["name_lower"]), None)
            if found_city:
                current_collected_data_dict["city_id_afisha"] = found_city["id"]
                logger.info(f"Clarified city: '{user_query}' mapped to ID {found_city['id']}")
            else:
                current_collected_data_dict["city_id_afisha"] = None
                clarification_context_for_node = f"Город '{user_query}' не найден. Пожалуйста, попробуйте указать другой город."
                if "city_name" not in existing_clarification_fields: # Если его там не было, добавляем снова
                    existing_clarification_fields.append("city_name")

        elif awaiting_clarification_field == "dates_description_original":
            current_collected_data_dict["dates_description_original"] = user_query
            current_collected_data_dict["raw_time_description_original"] = None # Сбрасываем, если уточняем всю дату
            # Вызываем datetime_parser_tool
            parsed_dt_result = await datetime_parser_tool.ainvoke({
                "natural_language_date": user_query,
                "natural_language_time_qualifier": None, # При уточнении даты, время не передаем отдельно
                "base_date_iso": datetime.now().isoformat()
            })
            if parsed_dt_result.get("datetime_iso"):
                current_collected_data_dict["parsed_dates_iso"] = [parsed_dt_result["datetime_iso"]]
                current_collected_data_dict["parsed_end_dates_iso"] = [parsed_dt_result["end_datetime_iso"]] if parsed_dt_result.get("end_datetime_iso") else None
                logger.info(f"Clarified dates: '{user_query}' parsed to ISO {current_collected_data_dict['parsed_dates_iso']}")
                if parsed_dt_result.get("is_ambiguous"):
                    clarification_context_for_node = parsed_dt_result.get("clarification_needed")
                    if "dates_description_original" not in existing_clarification_fields:
                         existing_clarification_fields.append("dates_description_original")
            else:
                clarification_context_for_node = parsed_dt_result.get("clarification_needed") or "Не удалось распознать уточненную дату. Попробуйте еще раз."
                if "dates_description_original" not in existing_clarification_fields:
                    existing_clarification_fields.append("dates_description_original")
        
        elif awaiting_clarification_field == "interests_original":
            raw_interests_list_clarified = [i.strip() for i in user_query.split(",") if i.strip()]
            current_collected_data_dict["interests_original"] = raw_interests_list_clarified
            
            mapped_interest_keys_clarified = []
            user_requested_restaurant_explicitly_clarified = False
            for interest_str in raw_interests_list_clarified:
                s = interest_str.lower().strip()
                key = None
                # <<< КОПИРУЕМ БЛОК МАППИНГА ИНТЕРЕСОВ ОТСЮДА >>>
                if "кино" == s or "фильм" == s or "фильмы" == s or "кинотеатр" == s: key = "Movie"
                elif "театр" == s or "спектакль" == s or "спектакли" == s or "пьес" in s: key = "Performance"
                elif "опер" in s or "балет" == s: key = "OperaBallet"
                elif "концерт" == s or "концерты" == s: key = "Concert"
                elif "выставк" in s or "экспозици" in s: key = "Exhibition"
                elif "фестивал" in s or "фест" == s: key = "Festival"
                elif "стендап" in s or "stand-up" in s or "stand up" in s: key = "StandUp"
                elif "спорт" == s or "матч" == s or "соревновани" in s: key = "SportEvent"
                elif "вечерин" in s or "пати" == s or "party" == s or "тусовк" in s or "дискотек" in s: key = "Party"
                elif "квиз" == s or "quiz" == s or "викторин" in s : key = "Quiz"
                elif "мастер-класс" in s or "мастер класс" in s or "воркшоп" == s or "workshop" == s: key = "MasterClass"
                elif "лекци" in s or "семинар" == s or "доклад" in s or ("конференци" in s and "пресс" not in s): key = "Lecture"
                elif "экскурс" in s: key = "Excursion"
                elif "музей" in s or "музеи" == s: key = "Museum"
                elif "ресторан" in s or "кафе" == s or "бар" == s or "поесть" in s or "покушать" in s:
                    user_requested_restaurant_explicitly_clarified = True
                
                if key: mapped_interest_keys_clarified.append(key)

            current_collected_data_dict["interests_keys_afisha"] = list(set(mapped_interest_keys_clarified)) if mapped_interest_keys_clarified else None
            logger.info(f"Clarified interests: '{user_query}' mapped to Afisha keys: {current_collected_data_dict['interests_keys_afisha']}")

            if user_requested_restaurant_explicitly_clarified and not current_collected_data_dict["interests_keys_afisha"]:
                clarification_context_for_node = "Похоже, вы снова указали только рестораны. Я не могу их искать как мероприятия. Пожалуйста, назовите другие типы активностей."
                if "interests_original" not in existing_clarification_fields:
                    existing_clarification_fields.append("interests_original")
            elif not current_collected_data_dict["interests_keys_afisha"] and raw_interests_list_clarified: # Ввели что-то, но не смапилось
                clarification_context_for_node = "Не смог распознать ваши уточненные интересы. Попробуйте еще раз (например, кино, театр)."
                if "interests_original" not in existing_clarification_fields:
                     existing_clarification_fields.append("interests_original")

        elif awaiting_clarification_field == "budget_original":
            try:
                budget_val_match = re.search(r"\d+", user_query)
                if budget_val_match:
                    budget_val = int(budget_val_match.group(0))
                    current_collected_data_dict["budget_original"] = budget_val
                    current_collected_data_dict["budget_current_search"] = budget_val
                    logger.info(f"Clarified budget: {budget_val}")
                else:
                    raise ValueError("No digits in budget input")
            except ValueError:
                clarification_context_for_node = "Пожалуйста, укажите бюджет числом."
                if "budget_original" not in existing_clarification_fields:
                     existing_clarification_fields.append("budget_original")
        
        elif awaiting_clarification_field == "user_start_address_original":
            logger.info(f"Processing address clarification: '{user_query}'")
            city_for_geocoding = current_collected_data_dict.get("city_name")
            previously_found_street = current_collected_data_dict.get("partial_address_street")
            address_to_geocode = user_query
            
            # Сбрасываем флаг ожидания ввода адреса (awaiting_address_input был в старой схеме, сейчас не используется явно в AgentState)
            # Но если он был в current_collected_data_dict, его нужно убрать или поставить False
            current_collected_data_dict.pop("awaiting_address_input", None) 

            if previously_found_street and not any(c.isalpha() for c in user_query if c.isalpha() and c.lower() not in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
                address_to_geocode = f"{previously_found_street}, {user_query}"

            if not city_for_geocoding:
                clarification_context_for_node = "Для уточнения адреса мне сначала нужно знать город. Пожалуйста, укажите город."
                if "city_name" not in existing_clarification_fields: existing_clarification_fields.append("city_name")
                current_collected_data_dict.pop("partial_address_street", None) # Сброс частично найденной улицы
            else:
                geocoding_result: GeocodingResult = await get_geocoding_details(address=address_to_geocode, city=city_for_geocoding)
                if geocoding_result.is_precise_enough and geocoding_result.coords:
                    current_collected_data_dict["user_start_address_original"] = geocoding_result.full_address_name_gis
                    current_collected_data_dict["user_start_address_validated_coords"] = {"lon": geocoding_result.coords[0], "lat": geocoding_result.coords[1]}
                    current_collected_data_dict.pop("partial_address_street", None)
                    logger.info(f"Address '{address_to_geocode}' geocoded successfully to {geocoding_result.full_address_name_gis}")
                elif geocoding_result.match_level == "street" and not previously_found_street : # Только если это первый раз улица найдена
                    clarification_context_for_node = f"Нашел улицу '{geocoding_result.full_address_name_gis}'. Уточните, пожалуйста, номер дома."
                    current_collected_data_dict["partial_address_street"] = geocoding_result.full_address_name_gis
                    if "user_start_address_original" not in existing_clarification_fields:
                         existing_clarification_fields.append("user_start_address_original")
                else: # Неточный результат или повторный запрос после улицы
                    clarification_context_for_node = f"Не удалось точно распознать адрес '{address_to_geocode}'. Попробуйте указать его еще раз, например: 'улица Ленина, 10'."
                    current_collected_data_dict.pop("partial_address_street", None) # Сбрасываем, если не помогло
                    if "user_start_address_original" not in existing_clarification_fields:
                         existing_clarification_fields.append("user_start_address_original")
        
        # Обновляем список полей для уточнения в стейте
        current_collected_data_dict["clarification_needed_fields"] = [f for f in existing_clarification_fields if f] # Удаляем None/пустые строки
        current_collected_data_dict["awaiting_clarification_for_field"] = None # Сбрасываем поле, которое только что уточнили

        logger.debug(f"After clarification. New collected_data: {str(current_collected_data_dict)[:300]}. Clarification context for next node: {clarification_context_for_node}")
        return {
            "collected_data": current_collected_data_dict,
            "messages": messages, # История сообщений не меняется этим узлом, только добавляется AIMessage следующим узлом
            "clarification_context": clarification_context_for_node, # Передаем контекст (например, ошибку) дальше
            "awaiting_clarification_for_field": None, # Мы обработали ожидаемое поле
        }

    # --- Обработка НОВОГО (не уточняющего) запроса ---
    logger.debug("extract_initial_info_node: Processing as a new/general query (awaiting_clarification_field is None).")
    
    # Сохраняем только валидированный адрес пользователя между новыми запросами (если он есть)
    # Город, даты, интересы, бюджет - должны извлекаться из нового запроса или запрашиваться заново.
    preserved_user_address_original = current_collected_data_dict.get("user_start_address_original")
    preserved_user_coords = current_collected_data_dict.get("user_start_address_validated_coords")
    
    current_collected_data_dict_for_new_query = {} # Начинаем с чистого листа для нового запроса
    if preserved_user_address_original and preserved_user_coords: # Сохраняем только если адрес полностью валиден
        current_collected_data_dict_for_new_query["user_start_address_original"] = preserved_user_address_original
        current_collected_data_dict_for_new_query["user_start_address_validated_coords"] = preserved_user_coords
        logger.debug(f"Preserving user address: {preserved_user_address_original}")

    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ExtractedInitialInfo) # ExtractedInitialInfo - Pydantic модель

    try:
        extraction_prompt_with_query = f'{INITIAL_INFO_EXTRACTION_PROMPT}\n\nИзвлеки информацию из следующего запроса пользователя:\n"{user_query}"'
        logger.debug(f"Sending to LLM for extraction: {user_query}")
        extracted_info: ExtractedInitialInfo = await structured_llm.ainvoke(extraction_prompt_with_query)
        logger.info(f"extract_initial_info_node: LLM Extracted Info (Pydantic): {extracted_info.model_dump_json(indent=2)}")

        # new_clarification_needed_in_this_step уже инициализирован как []
        
        # 1. Город
        if extracted_info.city:
            current_collected_data_dict_for_new_query["city_name"] = extracted_info.city
            cities = await fetch_cities_internal()
            found_city = next((c for c in cities if extracted_info.city.lower() in c["name_lower"]), None)
            if found_city:
                current_collected_data_dict_for_new_query["city_id_afisha"] = found_city["id"]
            else:
                new_clarification_needed_in_this_step.append("city_name")
                clarification_context_for_node = (clarification_context_for_node or "") + f" Город '{extracted_info.city}' не найден. "
        else:
            new_clarification_needed_in_this_step.append("city_name")

        # 2. Интересы
        user_requested_restaurant_explicitly_new = False
        if extracted_info.interests:
            current_collected_data_dict_for_new_query["interests_original"] = extracted_info.interests
            mapped_interest_keys_new = []
            for interest_str in extracted_info.interests:
                s = interest_str.lower().strip()
                key = None
                # <<< КОПИРУЕМ БЛОК МАППИНГА ИНТЕРЕСОВ ОТСЮДА (такой же, как в блоке Clarification) >>>
                if "кино" == s or "фильм" == s or "фильмы" == s or "кинотеатр" == s: key = "Movie"
                elif "театр" == s or "спектакль" == s or "спектакли" == s or "пьес" in s: key = "Performance"
                elif "опер" in s or "балет" == s: key = "OperaBallet"
                elif "концерт" == s or "концерты" == s: key = "Concert"
                elif "выставк" in s or "экспозици" in s: key = "Exhibition"
                elif "фестивал" in s or "фест" == s: key = "Festival"
                elif "стендап" in s or "stand-up" in s or "stand up" in s: key = "StandUp"
                elif "спорт" == s or "матч" == s or "соревновани" in s: key = "SportEvent"
                elif "вечерин" in s or "пати" == s or "party" == s or "тусовк" in s or "дискотек" in s: key = "Party"
                elif "квиз" == s or "quiz" == s or "викторин" in s : key = "Quiz"
                elif "мастер-класс" in s or "мастер класс" in s or "воркшоп" == s or "workshop" == s: key = "MasterClass"
                elif "лекци" in s or "семинар" == s or "доклад" in s or ("конференци" in s and "пресс" not in s): key = "Lecture"
                elif "экскурс" in s: key = "Excursion"
                elif "музей" in s or "музеи" == s: key = "Museum"
                elif "ресторан" in s or "кафе" == s or "бар" == s or "поесть" in s or "покушать" in s:
                    user_requested_restaurant_explicitly_new = True
                # <<< ДОСЮДА >>>
                if key: mapped_interest_keys_new.append(key)
            
            current_collected_data_dict_for_new_query["interests_keys_afisha"] = list(set(mapped_interest_keys_new)) if mapped_interest_keys_new else None
            
            if user_requested_restaurant_explicitly_new and not current_collected_data_dict_for_new_query["interests_keys_afisha"]:
                clarification_context_for_node = (clarification_context_for_node or "") + " Я не ищу рестораны как мероприятия. Могу поискать другие типы активностей. "
                new_clarification_needed_in_this_step.append("interests_original")
                # Очистим исходные интересы, если там были только рестораны
                if current_collected_data_dict_for_new_query.get("interests_original"):
                    current_collected_data_dict_for_new_query["interests_original"] = [
                        i for i in current_collected_data_dict_for_new_query["interests_original"] 
                        if not any(restr_kw in i.lower() for restr_kw in ["ресторан", "кафе", "бар", "поесть", "покушать"])
                    ]
                    if not current_collected_data_dict_for_new_query["interests_original"]:
                        current_collected_data_dict_for_new_query["interests_original"] = None
            elif not current_collected_data_dict_for_new_query["interests_keys_afisha"] and extracted_info.interests: # Извлекли, но не смапили
                new_clarification_needed_in_this_step.append("interests_original")
                clarification_context_for_node = (clarification_context_for_node or "") + " Не удалось точно определить ваши интересы. "
        else: # LLM не извлек интересы
            if not user_requested_restaurant_explicitly_new: # И это не был неявный запрос ресторана, который мы отфильтровали
                new_clarification_needed_in_this_step.append("interests_original")

        # 3. Бюджет
        if extracted_info.budget is not None:
            current_collected_data_dict_for_new_query["budget_original"] = extracted_info.budget
            current_collected_data_dict_for_new_query["budget_current_search"] = extracted_info.budget
        # Запрос бюджета опционален, поэтому не добавляем в new_clarification_needed, если не указан.
        # present_initial_plan_node может запросить его позже, если нужно.

        # 4. Даты и Время
        date_desc_llm = extracted_info.dates_description
        time_desc_llm = extracted_info.raw_time_description
        current_collected_data_dict_for_new_query["dates_description_original"] = date_desc_llm
        current_collected_data_dict_for_new_query["raw_time_description_original"] = time_desc_llm

        if date_desc_llm or time_desc_llm: # Если есть хотя бы описание даты или времени
            # Вызываем datetime_parser_tool
            parsed_dt_res = await datetime_parser_tool.ainvoke({
                "natural_language_date": date_desc_llm or "", # Передаем пустую строку, если только время
                "natural_language_time_qualifier": time_desc_llm,
                "base_date_iso": datetime.now().isoformat()
            })
            if parsed_dt_res.get("datetime_iso"):
                current_collected_data_dict_for_new_query["parsed_dates_iso"] = [parsed_dt_res["datetime_iso"]]
                current_collected_data_dict_for_new_query["parsed_end_dates_iso"] = [parsed_dt_res["end_datetime_iso"]] if parsed_dt_res.get("end_datetime_iso") else None
                if parsed_dt_res.get("is_ambiguous"):
                    new_clarification_needed_in_this_step.append("dates_description_original")
                    clarification_context_for_node = (clarification_context_for_node or "") + (parsed_dt_res.get("clarification_needed") or "") + " "
            else: # Если парсер не вернул datetime_iso
                new_clarification_needed_in_this_step.append("dates_description_original")
                clarification_context_for_node = (clarification_context_for_node or "") + (parsed_dt_res.get("clarification_needed") or "Не удалось распознать дату/время. ") + " "
        else: # Если LLM не извлек ни дату, ни время
            new_clarification_needed_in_this_step.append("dates_description_original")
        
        # Финальное формирование списка полей для уточнения и общего контекста
        current_collected_data_dict_for_new_query["clarification_needed_fields"] = list(set(new_clarification_needed_in_this_step))
        
        # Заменяем текущие собранные данные на данные нового запроса
        current_collected_data_dict = current_collected_data_dict_for_new_query 

    except ValidationError as ve_llm: # Ошибка валидации ответа LLM по Pydantic схеме
        logger.error(f"extract_initial_info_node: LLM Pydantic validation error: {ve_llm}", exc_info=True)
        # Запрашиваем все основные поля, так как не смогли извлечь
        current_collected_data_dict["clarification_needed_fields"] = list(set(["city_name", "dates_description_original", "interests_original"]))
        clarification_context_for_node = "Произошла ошибка при обработке вашего запроса. Давайте попробуем собрать информацию по частям."
    except Exception as e_llm: # Любая другая ошибка при вызове LLM
        logger.error(f"extract_initial_info_node: LLM extraction error: {e_llm}", exc_info=True)
        current_collected_data_dict.setdefault("clarification_needed_fields", []) # Гарантируем, что список есть
        for f_key in ["city_name", "dates_description_original", "interests_original"]:
            # Добавляем поле для уточнения, если оно еще не заполнено и не в списке
            if not current_collected_data_dict.get(f_key) and f_key not in current_collected_data_dict["clarification_needed_fields"]:
                current_collected_data_dict["clarification_needed_fields"].append(f_key)
        current_collected_data_dict["clarification_needed_fields"] = list(set(current_collected_data_dict["clarification_needed_fields"]))
        clarification_context_for_node = "Ошибка обработки вашего запроса. Попробуйте, пожалуйста, еще раз или переформулируйте."

    logger.info(f"extract_initial_info_node: Final collected_data for this step: {str(current_collected_data_dict)[:500]}. Clarification context: {clarification_context_for_node}")
    
    # Убедимся, что clarification_context_for_node не пустая строка, а None если нет контекста
    if clarification_context_for_node is not None and not clarification_context_for_node.strip():
        clarification_context_for_node = None

    return {
        "collected_data": current_collected_data_dict,
        "messages": messages,
        "clarification_context": clarification_context_for_node,
        "awaiting_clarification_for_field": None, # Для нового запроса мы не ждем уточнения конкретного поля
    }


# --- Узел 2: Уточнение недостающих данных ---
async def clarify_missing_data_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_missing_data_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    clarification_fields: List[str] = collected_data_dict.get(
        "clarification_needed_fields", []
    )
    status_message_to_user: Optional[str] = None
    field_being_clarified: Optional[str] = None

    if not clarification_fields:
        logger.info("clarify_missing_data_node: No fields need explicit clarification.")
        return {
            "status_message_to_user": None,
            "awaiting_clarification_for_field": None,
            "clarification_context": None,
            "collected_data": collected_data_dict,
        }

    field_to_clarify_now = clarification_fields[0]
    field_being_clarified = field_to_clarify_now

    missing_critical_fields_map = {
        "city_name": "город для поиска",
        "dates_description_original": "даты или период мероприятий",
        "interests_original": "ваши интересы или тип мероприятий",
        "budget_original": "ваш примерный бюджет",
        "user_start_address_original": "ваш адрес отправления (улица и дом)",
    }

    clarification_context_from_state = state.get("clarification_context")
    if (
        isinstance(clarification_context_from_state, str)
        and clarification_context_from_state
    ):
        status_message_to_user = clarification_context_from_state
        logger.info(
            f"clarify_missing_data_node: Using pre-defined clarification context: {status_message_to_user}"
        )
    else:
        field_description_for_prompt = missing_critical_fields_map.get(
            field_to_clarify_now, f"поле '{field_to_clarify_now}'"
        )

        raw_time_desc = collected_data_dict.get("raw_time_description_original")
        prompt_for_llm: str
        if field_to_clarify_now == "dates_description_original" and raw_time_desc:
            prompt_for_llm = TIME_CLARIFICATION_PROMPT_TEMPLATE.format(
                raw_time_description=raw_time_desc,
                current_date_info=date.today().strftime("%d %B %Y года (%A)"),
            )
        else:
            last_user_message_content = "Ваш запрос"
            current_messages = state.get("messages", [])
            if current_messages and isinstance(current_messages[-1], HumanMessage):
                last_user_message_content = current_messages[-1].content

            # Формируем summary без полей, которые не нужно показывать пользователю или которые служебные
            excluded_keys_for_summary = [
                "clarification_needed_fields",
                "awaiting_clarification_for_field",
                "awaiting_address_input",
                "partial_address_street",
                "awaiting_fallback_confirmation",
                "pending_fallback_event",
                "fallback_accepted_and_plan_updated",
                "previous_confirmed_collected_data",
                "previous_confirmed_events",
                "user_time_desc_for_fallback",
                "not_found_interest_keys",
                "fallback_candidates",
            ]
            current_data_summary_parts = []
            for k, v in collected_data_dict.items():
                if v and k not in excluded_keys_for_summary:
                    # Можно добавить более человекочитаемые названия для ключей
                    if k == "city_name":
                        current_data_summary_parts.append(f"Город: {v}")
                    elif k == "dates_description_original":
                        current_data_summary_parts.append(f"Когда: {v}")
                    elif k == "interests_original":
                        current_data_summary_parts.append(
                            f"Интересы: {', '.join(v) if isinstance(v, list) else v}"
                        )
                    elif k == "budget_original":
                        current_data_summary_parts.append(f"Бюджет: до {v} руб.")
                    # Добавь другие поля по необходимости

            current_data_summary_str = (
                "; ".join(current_data_summary_parts)
                if current_data_summary_parts
                else "пока ничего не уточнено"
            )

            prompt_for_llm = GENERAL_CLARIFICATION_PROMPT_TEMPLATE.format(
                user_query=last_user_message_content,
                current_collected_data_summary=current_data_summary_str,
                missing_fields_description=field_description_for_prompt,
            )

        logger.debug(
            f"clarify_missing_data_node: Using LLM prompt for '{field_description_for_prompt}'"
        )
        llm = get_gigachat_client()
        try:
            ai_response = await llm.ainvoke(prompt_for_llm)
            status_message_to_user = ai_response.content
            logger.info(
                f"clarify_missing_data_node: LLM generated clarification question: {status_message_to_user}"
            )
        except Exception as e_clarify:
            logger.error(
                f"clarify_missing_data_node: Error during LLM call: {e_clarify}",
                exc_info=True,
            )
            status_message_to_user = f"Мне нужно уточнение по полю: {field_description_for_prompt}. Не могли бы вы помочь?"

    final_message_to_user = (
        status_message_to_user
        or f"Пожалуйста, уточните {field_description_for_prompt}."
    )
    new_messages_history = state.get("messages", []) + [
        AIMessage(content=final_message_to_user)
    ]

    # Обновляем clarification_needed_fields в collected_data, чтобы в следующем шаге (в ребре)
    # не было этого поля, если мы по нему уже спросили.
    # Но это нужно делать осторожно, возможно, лучше это делать в extract_initial_info после успешного ответа.
    # Пока оставляем как есть, чтобы awaiting_clarification_for_field был главным.

    return {
        "messages": new_messages_history,
        "status_message_to_user": final_message_to_user,
        "awaiting_clarification_for_field": field_being_clarified,
        "clarification_context": None,
        "collected_data": collected_data_dict,  # Передаем collected_data дальше без изменений в этом узле
    }



# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ SEARCH_EVENTS_NODE ---

async def _fetch_events_via_tool_for_interest(
    internal_key: str,
    city_id: int,
    api_date_from_dt: datetime, 
    api_date_to_dt: datetime,   
    min_start_event_time_filter: Optional[datetime], 
    max_start_event_time_filter: Optional[datetime], 
    budget: Optional[int],
    user_max_overall_end_dt_naive_plan: Optional[datetime] 
) -> List[Event]:
    logger.debug(
        f"_fetch_events_via_tool: key='{internal_key}', city={city_id}, "
        f"API_dates=[{api_date_from_dt.date()} to {api_date_to_dt.date()-timedelta(days=1)}], "
        f"min_event_start={min_start_event_time_filter}, max_event_start={max_start_event_time_filter}, "
        f"user_max_plan_end={user_max_overall_end_dt_naive_plan}"
    )
    try:
        event_dicts_from_tool: List[Dict] = await event_search_tool.ainvoke({
            "city_id": city_id,
            "date_from": api_date_from_dt, # Этот параметр ожидает event_search_tool
            "date_to": api_date_to_dt,     # Этот параметр ожидает event_search_tool
            "interests_keys": [internal_key], 
            "min_start_time_naive": min_start_event_time_filter,
            "max_start_time_naive": max_start_event_time_filter,
            "max_budget_per_person": budget,
            "exclude_session_ids": None,
        })
        
        valid_events: List[Event] = []
        if not isinstance(event_dicts_from_tool, list):
            logger.error(f"event_search_tool for key '{internal_key}' returned non-list: {type(event_dicts_from_tool)}")
            return []

        for evt_dict in event_dicts_from_tool:
            if not isinstance(evt_dict, dict):
                logger.warning(f"Skipping non-dict item from event_search_tool for key '{internal_key}': {type(evt_dict)}")
                continue
            try:
                event_obj = Event(**evt_dict) 
                
                if user_max_overall_end_dt_naive_plan:
                    # Приблизительное время окончания события
                    event_duration_minutes = event_obj.duration_minutes or 120 # Дефолтная длительность, если неизвестна
                    event_end_time_approx = event_obj.start_time_naive_event_tz + timedelta(minutes=event_duration_minutes)
                    if event_end_time_approx > user_max_overall_end_dt_naive_plan:
                        logger.debug(f"Filtering out event '{event_obj.name}' (key: {internal_key}) as it ends ({event_end_time_approx}) after user_max_overall_plan_end_time ({user_max_overall_end_dt_naive_plan})")
                        continue
                
                valid_events.append(event_obj)
            except ValidationError as ve:
                logger.warning(f"Validation error for event data from tool for key '{internal_key}': {str(evt_dict)[:300]}. Error: {ve}")
        
        valid_events.sort(key=lambda e: e.start_time_naive_event_tz) # Сортируем сразу
        logger.info(f"Fetched and validated {len(valid_events)} events for internal_key='{internal_key}'")
        return valid_events
    except Exception as e_tool_invoke:
        logger.error(f"Error invoking event_search_tool for internal_key='{internal_key}': {e_tool_invoke}", exc_info=True)
        return []

# --- Вспомогательные функции для фильтрации событий ---

def _is_event_standup(event: Event) -> bool:
    """Проверяет, является ли событие стендапом на основе его данных."""
    if not event:
        return False

    # 1. Проверка типа API, с которым мы его искали (ожидаем Concert)
    # actual_api_type берется из поля 'Type' самого Creation из API
    if event.actual_api_type != "Concert":
        logger.debug(f"Event '{event.name}' (actual_api_type: {event.actual_api_type}) is not 'Concert', skipping StandUp check.")
        return False

    name_lower = event.name.lower()
    description_lower = ((event.description or "") + " " + (event.short_description or "")).lower()
    genres_lower = [g.lower() for g in (event.genres or [])]

    standup_keywords_name = ["стендап", "stand-up", "stand up", "открытый микрофон", "проверка материала", "комик", "сольный концерт"] # Добавил "сольный концерт"
    standup_keywords_description = ["стендап", "stand-up", "комик", "юмористическое шоу", "вечер комедии", "шутки", "монологи"]
    standup_genre_keywords = ["humor", "юмор"] # "comedy" может быть у комедийных спектаклей

    # Ключевые слова, указывающие, что это НЕ стендап (даже если тип "Concert")
    negative_keywords = [
        "поэзии", "литературный", "маяковский", "есенин", "стихи", "музыкальный",
        "группы", "песни", "спектакль", "пьеса", "театр", "драма", "опера", "балет"
    ]

    if any(neg_kw in name_lower or neg_kw in description_lower for neg_kw in negative_keywords):
        logger.debug(f"Event '{event.name}' filtered out as non-standup (negative keywords).")
        return False

    name_match = any(kw in name_lower for kw in standup_keywords_name)
    desc_match = any(kw in description_lower for kw in standup_keywords_description)
    genre_match = any(gkw in genres_lower for gkw in standup_genre_keywords)
    
    place_name_match = False
    if event.place_name and ("стендап-клуб" in event.place_name.lower() or "comedy club" in event.place_name.lower()):
        place_name_match = True

    # Считаем стендапом, если есть прямое указание в названии,
    # или подходящее описание/жанр/место и нет явных негативных признаков.
    if name_match: # Если в названии есть "стендап" и т.п. - это самый сильный признак
        return True
    if place_name_match and (desc_match or genre_match): # Если в стендап-клубе и есть признаки юмора/комедии
        return True
    if genre_match and desc_match: # Если жанр юмор и описание подходящее
        return True
    
    # Если только описание или только жанр (без явного названия или места) - можем быть более осторожны
    # Например, если просто genre_match, это может быть комедийный концерт, но не стендап.
    # Если только desc_match, тоже стоит проверить внимательнее.
    # Пока оставим так: если есть хоть один из desc_match, genre_match, place_name_match (и прошло негативные фильтры)
    if desc_match or genre_match:
         # Можно добавить лог, чтобы посмотреть на такие случаи
        logger.debug(f"Event '{event.name}' considered StandUp based on description/genre (desc: {desc_match}, genre: {genre_match}, place: {place_name_match})")
        return True

    return False


def _is_event_museum(event: Event) -> bool:
    """Проверяет, является ли событие музеем на основе его данных."""
    if not event:
        return False

    # actual_api_type - тип самого "Creation" из API Афиши
    # event.place.Type - тип места из данных /schedule (если есть)
    # Для музеев мы можем ожидать actual_api_type 'Admission' или 'Event' (если ищем по 'Event')
    
    if event.actual_api_type not in ["Admission", "Event"]:
        logger.debug(f"Event '{event.name}' (actual_api_type: {event.actual_api_type}) is not 'Admission' or 'Event', skipping Museum check.")
        return False

    name_lower = event.name.lower()
    place_name_lower = (event.place_name or "").lower()
    description_lower = ((event.description or "") + " " + (event.short_description or "")).lower()
    
    # Основные ключевые слова
    museum_keywords = [
        "музей", "экспона", "коллекци", "галере", "усадьб", "дом-музей", 
        "историческ", "художествен", "краеведческ", "выставк", "экспозици", # Выставки часто в музеях
        "панорама", "диорама", "археологическ", "мемориальн"
    ]
    # Типы мест, которые могут быть музеями
    place_type_keywords = ["музей", "галерея", "океанариум", "планетарий", "выставочный зал", "павильон"]


    # 1. Прямое указание в названии события или места
    if any(kw in name_lower for kw in museum_keywords) or \
       any(kw in place_name_lower for kw in museum_keywords) or \
       any(ptkw in place_name_lower for ptkw in place_type_keywords):
        logger.debug(f"Event '{event.name}' (Place: '{event.place_name}') considered a Museum (keyword in name/place_name).")
        return True

    # 2. Проверка описания
    if any(kw in description_lower for kw in museum_keywords):
        logger.debug(f"Event '{event.name}' considered a Museum (keyword in description).")
        return True
        
    # 3. Если тип места из API явно "Museum" (поле Place.Type из /schedule)
    #    Это поле не стандартизировано в вашей схеме Event, но если бы было, можно было бы использовать.
    #    Пока что ваш `event.actual_api_type` - это тип *события*, а не *места*.
    #    В логах мы видели, что "Воронежский океанариум" приходил с `Place.Type: 'Museum'`
    #    Если вы добавите `place_api_type: Optional[str]` в схему `Event` и будете его заполнять
    #    из `schedule_block.get("Place", {}).get("Type")` в `afisha_service.py`, то можно будет проверить:
    #    if event.place_api_type and event.place_api_type.lower() == "museum":
    #        logger.debug(f"Event '{event.name}' at place with API type 'Museum' considered a Museum.")
    #        return True

    logger.debug(f"Event '{event.name}' (Place: '{event.place_name}') did not pass Museum filters.")
    return False

# --- Узел 3: Поиск мероприятий (ОБНОВЛЕННАЯ ВЕРСИЯ с улучшенной фильтрацией) ---
async def search_events_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: search_events_node executing...")
    collected_data_dict: dict = dict(state.get("collected_data", {}))
    original_user_interests_keys: List[str] = list(collected_data_dict.get("interests_keys_afisha", []))

    # Сброс полей перед каждым новым поиском
    collected_data_dict["not_found_interest_keys_in_primary_search"] = []
    collected_data_dict["fallback_candidates"] = {} # Словарь для {interest_key: Event_Pydantic_object}
    collected_data_dict["fallback_accepted_and_plan_updated"] = False

    city_id = collected_data_dict.get("city_id_afisha")
    parsed_dates_iso_list = collected_data_dict.get("parsed_dates_iso")
    budget = collected_data_dict.get("budget_current_search")

    if not city_id or not parsed_dates_iso_list or not original_user_interests_keys:
        logger.warning(f"search_events_node: Missing critical data. City: {city_id}, Dates: {parsed_dates_iso_list}, Interests: {original_user_interests_keys}")
        # Заполняем not_found_interest_keys_in_primary_search исходными интересами, если они были
        collected_data_dict["not_found_interest_keys_in_primary_search"] = list(original_user_interests_keys) if original_user_interests_keys else list(collected_data_dict.get("interests_original",[]))
        return {"current_events": [], "is_initial_plan_proposed": False, "collected_data": collected_data_dict}

    try:
        user_min_start_dt_naive = datetime.fromisoformat(parsed_dates_iso_list[0])
        api_date_from_dt = user_min_start_dt_naive.replace(hour=0, minute=0, second=0, microsecond=0)
        
        user_max_overall_end_dt_naive: Optional[datetime] = None
        parsed_end_dates_iso_list = collected_data_dict.get("parsed_end_dates_iso")
        if parsed_end_dates_iso_list and parsed_end_dates_iso_list[0]:
            temp_end_dt = datetime.fromisoformat(parsed_end_dates_iso_list[0])
            # Если время не указано (00:00), то считаем до конца дня
            user_max_overall_end_dt_naive = temp_end_dt.replace(hour=23, minute=59, second=59) if temp_end_dt.hour == 0 and temp_end_dt.minute == 0 else temp_end_dt
        
        # Дата окончания для API Афиши (всегда +1 день от последней интересующей даты, время 00:00)
        api_date_to_for_primary_search_dt = (user_max_overall_end_dt_naive or api_date_from_dt).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    except Exception as e_date_parse:
        logger.error(f"Error parsing dates in search_events_node: {e_date_parse}", exc_info=True)
        collected_data_dict["not_found_interest_keys_in_primary_search"] = list(original_user_interests_keys)
        return {"current_events": [], "is_initial_plan_proposed": False, "collected_data": collected_data_dict}

    min_start_for_primary_search: Optional[datetime] = None
    # Если в user_min_start_dt_naive время не 00:00, значит пользователь указал конкретное время начала
    if not (user_min_start_dt_naive.hour == 0 and user_min_start_dt_naive.minute == 0):
        min_start_for_primary_search = user_min_start_dt_naive
    
    # max_start_for_primary_search - максимальное ВРЕМЯ НАЧАЛА события в рамках интересующего дня/диапазона
    # Если user_max_overall_end_dt_naive не None, то это и есть ограничение.
    # Иначе, если ищем на один день без указания времени окончания, то ограничения по max_start_time нет.
    max_start_for_primary_search = user_max_overall_end_dt_naive 

    # --- Шаг 1: Первичный поиск по каждому интересу пользователя ---
    all_events_found_by_type_primary: Dict[str, List[Event]] = {} # {interest_key: [Event, ...]}
    
    primary_search_tasks = []
    for interest_key in original_user_interests_keys:
        task = _fetch_events_via_tool_for_interest( # Эта функция уже возвращает List[Event]
            internal_key=interest_key,
            city_id=city_id,
            api_date_from_dt=api_date_from_dt,
            api_date_to_dt=api_date_to_for_primary_search_dt,
            min_start_event_time_filter=min_start_for_primary_search,
            max_start_event_time_filter=max_start_for_primary_search,
            budget=budget,
            user_max_overall_end_dt_naive_plan=user_max_overall_end_dt_naive # Для фильтрации по времени ОКОНЧАНИЯ события
        )
        primary_search_tasks.append(task)
    
    results_of_primary_searches: List[List[Event]] = await asyncio.gather(*primary_search_tasks, return_exceptions=True)

    for i, internal_key in enumerate(original_user_interests_keys):
        result_list_or_exc = results_of_primary_searches[i]
        if isinstance(result_list_or_exc, Exception):
            logger.error(f"Exception during primary search for interest_key='{internal_key}': {result_list_or_exc}")
            all_events_found_by_type_primary[internal_key] = []
        elif result_list_or_exc: # result_list_or_exc это List[Event]
            all_events_found_by_type_primary[internal_key] = result_list_or_exc # Уже отсортированы в _fetch_...
        else:
            all_events_found_by_type_primary[internal_key] = []

    # --- Шаг 2: Дополнительная фильтрация и выбор лучших кандидатов ---
    events_to_propose: List[Event] = []
    proposed_session_ids: Set[int] = set() # Чтобы не предлагать одно и то же событие (сеанс) дважды

    for internal_interest_key in original_user_interests_keys:
        candidate_events_for_key: List[Event] = all_events_found_by_type_primary.get(internal_interest_key, [])
        
        if not candidate_events_for_key:
            logger.info(f"No API results for internal_interest_key='{internal_interest_key}' to perform post-filtering.")
            if internal_interest_key not in collected_data_dict["not_found_interest_keys_in_primary_search"]:
                 collected_data_dict["not_found_interest_keys_in_primary_search"].append(internal_interest_key)
            continue

        logger.debug(f"Post-filtering for internal_interest_key='{internal_interest_key}' with {len(candidate_events_for_key)} candidates.")
        best_event_for_this_key: Optional[Event] = None

        if internal_interest_key == "Museum":
            for event_candidate in candidate_events_for_key:
                if event_candidate.session_id in proposed_session_ids: continue
                if _is_event_museum(event_candidate):
                    best_event_for_this_key = event_candidate
                    break
        
        elif internal_interest_key == "StandUp":
            for event_candidate in candidate_events_for_key:
                if event_candidate.session_id in proposed_session_ids: continue
                if _is_event_standup(event_candidate):
                    best_event_for_this_key = event_candidate
                    break
        
        # Добавьте elif для других "сложных" категорий (Exhibition, Festival, etc.)
        # используя аналогичные функции-фильтры _is_event_exhibition, _is_event_festival

        else: # Для "простых" категорий, где тип API обычно достаточно надежен (Movie, Concert (не стендап!), Performance (театр))
            # Можно добавить более строгую проверку event.actual_api_type, если необходимо
            # Например, для "Concert" (если это не стендап) можно проверить, что это не "Humor" жанр
            # или не содержит ключевых слов стендапа, если "StandUp" тоже мапится на "Concert".
            # Сейчас для простоты берем первого подходящего.
            
            expected_api_type: Optional[str] = None
            # Пример: если "Concert" используется и для обычных концертов, и для стендапов,
            # то здесь мы бы хотели отобрать "Concert", который НЕ стендап.
            if internal_interest_key == "Concert": # Это должен быть "обычный" концерт
                expected_api_type = "Concert"
                for event_candidate in candidate_events_for_key:
                    if event_candidate.session_id in proposed_session_ids: continue
                    if event_candidate.actual_api_type == expected_api_type and not _is_event_standup(event_candidate): # Убедимся, что это не стендап
                        best_event_for_this_key = event_candidate
                        break
            elif internal_interest_key == "Performance": # Ожидаем театр/оперу/балет
                 expected_api_type = "Performance"
                 for event_candidate in candidate_events_for_key:
                    if event_candidate.session_id in proposed_session_ids: continue
                    # Здесь тоже можно добавить исключающую логику, если "Performance" используется для стендапов
                    if event_candidate.actual_api_type == expected_api_type and not _is_event_standup(event_candidate):
                        best_event_for_this_key = event_candidate
                        break
            # Добавьте другие "простые" ключи
            # ...

            if not best_event_for_this_key and candidate_events_for_key: # Если не нашли по строгим правилам, берем первого доступного
                for event_candidate in candidate_events_for_key:
                    if event_candidate.session_id not in proposed_session_ids:
                        # Для "простых" типов можно также проверить, что actual_api_type соответствует ожидаемому
                        # Например, для "Movie" -> event_candidate.actual_api_type == "Movie"
                        # Но это уже должно было быть отфильтровано в afisha_service по CreationType
                        best_event_for_this_key = event_candidate
                        logger.info(f"Taking first available for simple key '{internal_interest_key}': {best_event_for_this_key.name}")
                        break
        
        if best_event_for_this_key:
            logger.info(f"Selected event '{best_event_for_this_key.name}' (ID: {best_event_for_this_key.session_id}, Actual API type: {best_event_for_this_key.actual_api_type}) for internal_interest_key='{internal_interest_key}'.")
            events_to_propose.append(best_event_for_this_key)
            proposed_session_ids.add(best_event_for_this_key.session_id)
        else:
            logger.info(f"No suitable event found for internal_interest_key='{internal_interest_key}' after primary search post-filtering.")
            if internal_interest_key not in collected_data_dict["not_found_interest_keys_in_primary_search"]:
                collected_data_dict["not_found_interest_keys_in_primary_search"].append(internal_interest_key)

    events_to_propose.sort(key=lambda e: e.start_time_naive_event_tz)

    # --- Шаг 3: Логика Fallback для ненайденных интересов ---
    interests_for_fallback_search = collected_data_dict.get("not_found_interest_keys_in_primary_search", [])
    
    if interests_for_fallback_search:
        logger.info(f"Attempting fallback search for interests: {interests_for_fallback_search}")
        # Для fallback ищем на более широкий диапазон дат и без строгих временных ограничений пользователя
        fallback_api_date_from_dt = api_date_from_dt # Начинаем с той же даты, что и основной поиск
        fallback_api_date_to_dt = fallback_api_date_from_dt + timedelta(days=7) # На неделю вперед
        
        fallback_search_tasks = []
        for interest_fb_key in interests_for_fallback_search:
            task_fb = _fetch_events_via_tool_for_interest(
                internal_key=interest_fb_key,
                city_id=city_id,
                api_date_from_dt=fallback_api_date_from_dt,
                api_date_to_dt=fallback_api_date_to_dt, 
                min_start_event_time_filter=None, # Без ограничений по времени начала
                max_start_event_time_filter=None, # Без ограничений по времени начала
                budget=budget, # Бюджет сохраняем
                user_max_overall_end_dt_naive_plan=None # Без ограничений по времени окончания плана
            )
            fallback_search_tasks.append(task_fb)
            
        results_of_fallback_searches: List[List[Event]] = await asyncio.gather(*fallback_search_tasks, return_exceptions=True)

        current_fallback_candidates: Dict[str, Event] = {} # Собираем здесь перед записью в state

        for i, internal_key_fb in enumerate(interests_for_fallback_search):
            fb_event_list_or_exc = results_of_fallback_searches[i]
            
            if isinstance(fb_event_list_or_exc, Exception) or not fb_event_list_or_exc:
                logger.warning(f"Fallback search for '{internal_key_fb}' yielded no results or an error: {fb_event_list_or_exc if isinstance(fb_event_list_or_exc, Exception) else 'No results'}")
                continue
            
            # Применяем ту же пост-фильтрацию к результатам fallback
            best_fallback_candidate_for_key_obj: Optional[Event] = None
            
            if internal_key_fb == "Museum":
                for event_fb_candidate in fb_event_list_or_exc: # fb_event_list_or_exc это List[Event]
                    if event_fb_candidate.session_id in proposed_session_ids: continue # Не предлагать то, что уже в основном плане
                    if _is_event_museum(event_fb_candidate):
                        best_fallback_candidate_for_key_obj = event_fb_candidate
                        break 
            elif internal_key_fb == "StandUp":
                for event_fb_candidate in fb_event_list_or_exc:
                    if event_fb_candidate.session_id in proposed_session_ids: continue
                    if _is_event_standup(event_fb_candidate):
                        best_fallback_candidate_for_key_obj = event_fb_candidate
                        break
            # Добавьте elif для других "сложных" категорий для fallback
            else: # Для "простых" категорий в fallback можно быть менее строгим или просто взять первое
                # Но лучше все равно какую-то базовую проверку типа сделать, если это важно
                if fb_event_list_or_exc: # Если есть хоть какие-то результаты
                    for event_fb_candidate in fb_event_list_or_exc:
                        if event_fb_candidate.session_id in proposed_session_ids: continue
                        # Здесь можно добавить проверку, например, что CreationType от API совпадает с ожидаемым для этого internal_key_fb
                        # Например, если internal_key_fb == "Movie", то event_fb_candidate.actual_api_type == "Movie"
                        best_fallback_candidate_for_key_obj = event_fb_candidate
                        logger.info(f"Taking first available for simple fallback key '{internal_key_fb}': {best_fallback_candidate_for_key_obj.name}")
                        break

            if best_fallback_candidate_for_key_obj:
                # Убедимся, что этот fallback еще не предлагается и не был отклонен ранее (если есть такая логика)
                # И что он не конфликтует по времени с уже выбранными events_to_propose (если их >0)
                # Пока что просто добавляем, если он релевантен по типу.
                if best_fallback_candidate_for_key_obj.session_id not in proposed_session_ids:
                    current_fallback_candidates[internal_key_fb] = best_fallback_candidate_for_key_obj
                    logger.info(f"Found RELEVANT fallback candidate for '{internal_key_fb}': {best_fallback_candidate_for_key_obj.name} on {best_fallback_candidate_for_key_obj.start_time_naive_event_tz.date()}")
        
        # Обновляем collected_data_dict только валидными fallback кандидатами
        if current_fallback_candidates:
            collected_data_dict["fallback_candidates"] = {
                key: event.model_dump() for key, event in current_fallback_candidates.items()
            }


    # Определяем, был ли предложен начальный план
    # План считается предложенным, если есть что показать: либо прямые события, либо релевантные fallback-кандидаты
    is_initial_plan_now_proposed = bool(events_to_propose) or bool(collected_data_dict.get("fallback_candidates"))
    
    # Обновляем список ненайденных интересов: только те, для которых нет ни прямого события, ни fallback
    final_not_found_keys = []
    if collected_data_dict.get("not_found_interest_keys_in_primary_search"):
        for key_not_found_in_primary in collected_data_dict["not_found_interest_keys_in_primary_search"]:
            is_covered_by_fallback = key_not_found_in_primary in collected_data_dict.get("fallback_candidates", {})
            is_covered_by_direct = any(event.event_type_key == key_not_found_in_primary for event in events_to_propose)
            
            if not is_covered_by_fallback and not is_covered_by_direct:
                final_not_found_keys.append(key_not_found_in_primary)
    
    collected_data_dict["not_found_interest_keys_final"] = final_not_found_keys # Новое поле для error_node

    logger.info(
        f"search_events_node final proposal: {len(events_to_propose)} direct events. "
        f"Fallback candidates for: {list(collected_data_dict.get('fallback_candidates', {}).keys())}. "
        f"Interests truly not found (after primary and fallback filters): {final_not_found_keys}"
    )
    
    return {
        "current_events": events_to_propose, # Только те, что прошли все фильтры для немедленного предложения
        "is_initial_plan_proposed": is_initial_plan_now_proposed,
        "collected_data": collected_data_dict, # С обновленными fallback_candidates и not_found_interest_keys_final
    }



async def _check_event_compatibility(
    first_event: Event,
    second_event_candidate: Event,
    user_max_overall_end_dt_naive: Optional[datetime],
) -> Tuple[bool, Optional[str]]:
    first_event_duration_minutes = first_event.duration_minutes or 120
    first_event_end_naive = first_event.start_time_naive_event_tz + timedelta(
        minutes=first_event_duration_minutes
    )

    if (
        user_max_overall_end_dt_naive
        and first_event_end_naive > user_max_overall_end_dt_naive
    ):
        return False, "Первое мероприятие заканчивается слишком поздно."
    if second_event_candidate.start_time_naive_event_tz < first_event_end_naive:
        return False, "Второе мероприятие начинается до окончания первого."

    route_duration_minutes = 30
    if (
        first_event.place_coords_lon
        and first_event.place_coords_lat
        and second_event_candidate.place_coords_lon
        and second_event_candidate.place_coords_lat
    ):
        try:
            route_result = await get_route(
                points=[
                    {
                        "lon": first_event.place_coords_lon,
                        "lat": first_event.place_coords_lat,
                    },
                    {
                        "lon": second_event_candidate.place_coords_lon,
                        "lat": second_event_candidate.place_coords_lat,
                    },
                ],
                transport="driving",
            )
            if route_result and route_result.get("status") == "success":
                route_duration_minutes = route_result.get("duration_seconds", 1800) / 60
            else:
                logger.warning(
                    f"Route error for compatibility check: {route_result.get('message') if route_result else 'No response'}"
                )
        except Exception as e_route:
            logger.error(
                f"get_route exception for compatibility: {e_route}", exc_info=True
            )

    arrival_at_second_event_naive = first_event_end_naive + timedelta(
        minutes=route_duration_minutes
    )
    buffer_time = timedelta(minutes=15)

    if (
        arrival_at_second_event_naive
        > second_event_candidate.start_time_naive_event_tz - buffer_time
    ):
        return False, "Не успеть на второе мероприятие."

    second_event_duration_minutes = second_event_candidate.duration_minutes or 120
    second_event_end_naive = (
        second_event_candidate.start_time_naive_event_tz
        + timedelta(minutes=second_event_duration_minutes)
    )
    if (
        user_max_overall_end_dt_naive
        and second_event_end_naive > user_max_overall_end_dt_naive
    ):
        return False, "Второе мероприятие заканчивается слишком поздно."
    return True, None


# --- Узел 4: Представление начального плана и запрос адреса/бюджета (ОБНОВЛЕННАЯ ВЕРСИЯ) ---
async def present_initial_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_initial_plan_node executing...")
    current_events: List[Event] = state.get(
        "current_events", []
    )  # Это то, что отображаем СРАЗУ
    collected_data_dict: dict = dict(state.get("collected_data", {}))

    response_parts = []
    awaiting_fallback_conf = False
    pending_fallback_event_for_state: Optional[Dict] = None
    field_to_be_clarified_next: Optional[str] = None

    if current_events:
        response_parts.append("Вот что я смог найти для вас по вашему запросу:")
        for i, event in enumerate(current_events):
            time_str = event.start_time_naive_event_tz.strftime("%H:%M")
            date_str = event.start_time_naive_event_tz.strftime("%d.%m.%Y")
            desc = f"{i+1}. **{event.name}** ({event.event_type_key}) в '{event.place_name}' ({event.place_address or 'Адрес не указан'}). Начало в {time_str} ({date_str})."
            if event.min_price is not None:
                desc += f" Цена от {event.min_price} руб."
            if event.duration_minutes:
                desc += f" Продолжительность ~{event.duration_minutes // 60}ч {event.duration_minutes % 60}м."
            response_parts.append(desc)

    # Предлагаем fallback для тех интересов, по которым не было событий в current_events
    # но которые были изначально запрошены и для которых есть fallback-кандидат.
    original_user_interests_keys = collected_data_dict.get("interests_keys_afisha", [])
    fallback_candidates = collected_data_dict.get("fallback_candidates", {})
    # Исключаем интересы, которые уже есть в current_events
    interests_covered_by_current_events = {e.event_type_key for e in current_events}

    interest_key_to_name_map = {
        "Movie": "фильмов",
        "Performance": "спектаклей",
        "Concert": "концертов",
    }

    for interest_key_fb in original_user_interests_keys:
        if (
            interest_key_fb in interests_covered_by_current_events
        ):  # Уже есть в основном предложении
            continue

        fb_event_data = fallback_candidates.get(interest_key_fb)
        if fb_event_data and not collected_data_dict.get(
            "fallback_accepted_and_plan_updated"
        ):  # И еще не предлагали/принимали fallback по этому интересу
            # Проверяем, не предлагали ли мы уже этот fallback
            if collected_data_dict.get(
                "last_offered_fallback_for_interest"
            ) == interest_key_fb and not state.get("messages", [])[
                -1
            ].content.lower().startswith(
                "да"
            ):  # Если уже предлагали и ответ был не "да"
                continue

            try:
                fb_event = Event(**fb_event_data)
                type_name = interest_key_to_name_map.get(
                    interest_key_fb, interest_key_fb
                )
                fb_time_str = fb_event.start_time_naive_event_tz.strftime("%H:%M")
                fb_date_str = fb_event.start_time_naive_event_tz.strftime("%d.%m.%Y")
                time_desc_orig = collected_data_dict.get(
                    "dates_description_original", "запрошенное время"
                )
                if collected_data_dict.get("raw_time_description_original"):
                    time_desc_orig += (
                        f" ({collected_data_dict['raw_time_description_original']})"
                    )

                fallback_msg = f"\nК сожалению, подходящих {type_name} на {time_desc_orig} не нашлось. "
                fallback_msg += f"Однако, есть другой вариант: «{fb_event.name}» ({type_name[:-1].lower() if type_name.endswith('ов') else type_name.lower()}) на {fb_date_str} в {fb_time_str}"
                if fb_event.place_name:
                    fallback_msg += f" в «{fb_event.place_name}»"
                if fb_event.min_price is not None:
                    fallback_msg += f" (цена от {fb_event.min_price} руб.)"
                fallback_msg += ". Хотите добавить его в план? (да/нет)"
                response_parts.append(fallback_msg)

                awaiting_fallback_conf = True
                pending_fallback_event_for_state = fb_event.model_dump()
                collected_data_dict["last_offered_fallback_for_interest"] = (
                    interest_key_fb
                )
                break
            except ValidationError as e:
                logger.error(
                    f"Error validating fallback event {fb_event_data.get('name')}: {e}"
                )

    # Сообщение о ненайденных интересах (если по ним нет ни current_event, ни предложенного fallback)
    if not awaiting_fallback_conf:  # Только если не ждем ответа по fallback
        truly_not_found_keys = []
        for key in original_user_interests_keys:
            if (
                key not in interests_covered_by_current_events
                and key not in fallback_candidates
            ):
                truly_not_found_keys.append(interest_key_to_name_map.get(key, key))

        if truly_not_found_keys:
            if not current_events:  # Если и основных событий нет
                response_parts = [
                    f"К сожалению, не удалось найти {', '.join(truly_not_found_keys)} по вашим критериям."
                ]
            else:
                response_parts.append(
                    f"\nТакже не удалось найти подходящие {', '.join(truly_not_found_keys)}."
                )

    if awaiting_fallback_conf:
        collected_data_dict["awaiting_fallback_confirmation"] = True
        collected_data_dict["pending_fallback_event"] = pending_fallback_event_for_state
    else:
        if "awaiting_fallback_confirmation" in collected_data_dict:
            del collected_data_dict["awaiting_fallback_confirmation"]
        if "pending_fallback_event" in collected_data_dict:
            del collected_data_dict["pending_fallback_event"]
        if "last_offered_fallback_for_interest" in collected_data_dict:
            del collected_data_dict["last_offered_fallback_for_interest"]

        questions_to_user = []
        if not collected_data_dict.get(
            "user_start_address_original"
        ) and not collected_data_dict.get("user_start_address_validated_coords"):
            if current_events or collected_data_dict.get(
                "fallback_accepted_and_plan_updated"
            ):  # Если есть что-то в плане
                questions_to_user.append(
                    "Откуда вы планируете начать маршрут? Назовите, пожалуйста, адрес (улица и дом)."
                )
                field_to_be_clarified_next = "user_start_address_original"

        # Бюджет запрашиваем, только если его нет и не спрашиваем адрес
        if (
            collected_data_dict.get("budget_original") is None
        ):  # Используем budget_original для проверки, был ли он указан
            if not field_to_be_clarified_next:
                questions_to_user.append("Уточните ваш бюджет на одно мероприятие?")
                field_to_be_clarified_next = "budget_original"

        if questions_to_user:
            current_plan_text = "\n".join(filter(None, response_parts))
            if current_plan_text.strip() and not current_plan_text.startswith(
                "К сожалению, не удалось найти"
            ):
                final_msg_text = (
                    current_plan_text + "\n\n" + " ".join(questions_to_user)
                )
            else:  # Если событий нет или только сообщение об ошибке, а потом вопросы
                if (
                    not current_plan_text.strip() and not current_events
                ):  # Если вообще ничего не было
                    final_msg_text = " ".join(questions_to_user)
                else:  # Если было сообщение об ошибке
                    final_msg_text = (
                        current_plan_text + "\n" + " ".join(questions_to_user)
                    )

            response_parts = [final_msg_text]
        elif current_events:
            response_parts.append(
                "\n\nКак вам такой предварительный план? Если что-то не подходит, скажите, попробуем изменить."
            )
        elif (
            not current_events
            and not awaiting_fallback_conf
            and not ("".join(response_parts)).strip()
        ):
            response_parts = [
                "К сожалению, по вашему запросу ничего не найдено. Попробуем другие критерии?"
            ]
            field_to_be_clarified_next = None

    if "fallback_accepted_and_plan_updated" in collected_data_dict:
        del collected_data_dict[
            "fallback_accepted_and_plan_updated"
        ]  # Сбрасываем после использования

    final_response_text = ("\n".join(filter(None, response_parts))).strip()
    if not final_response_text:
        final_response_text = (
            "План готов. Что-нибудь еще?"
            if current_events
            else "По вашему запросу ничего не найдено. Попробуем другие критерии?"
        )

    new_messages = state.get("messages", []) + [AIMessage(content=final_response_text)]

    return {
        "messages": new_messages,
        "status_message_to_user": final_response_text,
        "collected_data": collected_data_dict,
        "is_initial_plan_proposed": bool(current_events)
        and not awaiting_fallback_conf,  # План предложен, если есть события и не ждем ответа на fallback
        "awaiting_final_confirmation": False,
        "awaiting_clarification_for_field": field_to_be_clarified_next,
    }


# --- Узел 5: Обработка ответа на адрес ИЛИ построение маршрута, если адрес не нужен / уже есть ---
async def clarify_address_or_build_route_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: clarify_address_or_build_route_node executing...")
    collected_data: CollectedUserData = state.get("collected_data", {})  # type: ignore
    current_events: List[Event] = state.get("current_events", [])  # type: ignore

    if not current_events:
        logger.warning("build_route_node: No current events for route.")
        return {
            "current_route_details": RouteDetails(
                status="error", error_message="Нет мероприятий."
            ),
            "is_full_plan_with_route_proposed": False,
        }

    user_start_address_str = collected_data.get("user_start_address_original")
    user_start_coords = collected_data.get("user_start_address_validated_coords")

    if (
        not user_start_coords
    ):  # Если адреса пользователя нет, маршрут не строим (или строим только между событиями, если их > 1)
        if len(current_events) > 1:
            logger.info(
                "User address not provided, will attempt route between events if possible."
            )
            # Для маршрута между событиями, start_point будет первым событием
            # Эта логика уже должна быть в RouteBuilderToolArgs или здесь
        else:  # Одно событие и нет адреса
            logger.info("One event and no user address, no route to build from user.")
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

    # Проверяем, все ли события в один день
    event_dates: Set[date] = {
        evt.start_time_naive_event_tz.date() for evt in current_events
    }
    multiple_days = len(event_dates) > 1

    logger.info(f"Route for events on multiple_days: {multiple_days}")

    all_route_segments: List[RouteSegment] = []
    total_duration_seconds_combined = 0
    total_distance_meters_combined = 0
    overall_route_status = "success"

    if multiple_days:
        if not user_start_coords:  # Нужен адрес пользователя для маршрутов в разные дни
            logger.warning("Multiple day events but no user start address for routing.")
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Для мероприятий в разные дни нужен ваш адрес отправления.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        user_start_location = LocationModel(lon=user_start_coords["lon"], lat=user_start_coords["lat"], address_string=user_start_address_str)  # type: ignore

        for i, event_obj in enumerate(current_events):
            event_location = None
            if (
                event_obj.place_coords_lon is not None
                and event_obj.place_coords_lat is not None
            ):
                event_location = LocationModel(
                    lon=event_obj.place_coords_lon,
                    lat=event_obj.place_coords_lat,
                    address_string=event_obj.place_address,
                )
            elif event_obj.place_address:
                coords = await get_coords_from_address(address=event_obj.place_address, city=collected_data.get("city_name", ""))  # type: ignore
                if coords:
                    event_location = LocationModel(
                        lon=coords[0],
                        lat=coords[1],
                        address_string=event_obj.place_address,
                    )

            if not event_location:
                logger.warning(
                    f"Could not get location for event {event_obj.name} for multi-day route."
                )
                segment_error = RouteSegment(
                    from_address=user_start_address_str or "Ваше местоположение",
                    to_address=event_obj.name,  # Используем имя события как to_address
                    segment_status="error",
                    segment_error_message=f"Не удалось определить координаты для '{event_obj.name}'",
                )
                all_route_segments.append(segment_error)
                overall_route_status = "partial_success"
                continue

            tool_args_segment = RouteBuilderToolArgs(
                start_point=user_start_location, event_points=[event_location]
            )
            logger.info(
                f"Building route from User to Event {i+1} ({event_obj.name}): {tool_args_segment.model_dump_json(exclude_none=True)}"
            )
            route_data_segment_dict = await route_builder_tool.ainvoke(
                tool_args_segment.model_dump(exclude_none=True)
            )
            try:
                route_details_segment = RouteDetails(**route_data_segment_dict)
                if (
                    route_details_segment.status == "success"
                    and route_details_segment.segments
                ):
                    all_route_segments.extend(route_details_segment.segments)
                    total_duration_seconds_combined += (
                        route_details_segment.total_duration_seconds or 0
                    )
                    total_distance_meters_combined += (
                        route_details_segment.total_distance_meters or 0
                    )
                else:
                    overall_route_status = "partial_success"
                    error_msg_seg = (
                        route_details_segment.error_message
                        or f"Не удалось построить маршрут до '{event_obj.name}'"
                    )
                    all_route_segments.append(
                        RouteSegment(
                            from_address=user_start_address_str,
                            to_address=event_obj.place_address or event_obj.name,
                            segment_status="error",
                            segment_error_message=error_msg_seg,
                        )
                    )
            except ValidationError:
                overall_route_status = "partial_success"
                all_route_segments.append(
                    RouteSegment(
                        from_address=user_start_address_str,
                        to_address=event_obj.place_address or event_obj.name,
                        segment_status="error",
                        segment_error_message="Ошибка данных маршрута",
                    )
                )

    else:  # Все события в один день - строим последовательный маршрут
        start_location_for_api: Optional[LocationModel] = None
        event_points_for_api: List[LocationModel] = []

        if user_start_coords:
            start_location_for_api = LocationModel(lon=user_start_coords["lon"], lat=user_start_coords["lat"], address_string=user_start_address_str)  # type: ignore
            target_events_for_route = current_events
        elif len(current_events) > 1:  # Нет адреса пользователя, но есть >1 события
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
            # ... (геокодирование первого события, если нужно, как в вашем старом коде) ...
            target_events_for_route = current_events[1:]
        else:  # Одно событие, нет адреса пользователя - уже обработано выше
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

        if not start_location_for_api:  # Если стартовая точка не определилась
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Не удалось определить начальную точку маршрута.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        for event_obj in target_events_for_route:
            # ... (логика добавления event_obj в event_points_for_api с геокодингом, как в вашем старом коде) ...
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
            # ... (else с геокодингом)

        if (
            not event_points_for_api and target_events_for_route
        ):  # Если есть к чему строить, но не смогли получить точки
            return {
                "current_route_details": RouteDetails(
                    status="error",
                    error_message="Не удалось определить координаты событий для маршрута.",
                ),
                "is_full_plan_with_route_proposed": False,
            }

        if event_points_for_api:  # Только если есть куда строить маршрут
            tool_args_single_day = RouteBuilderToolArgs(
                start_point=start_location_for_api, event_points=event_points_for_api
            )
            logger.info(
                f"Building single-day route: {tool_args_single_day.model_dump_json(exclude_none=True)}"
            )
            route_data_dict = await route_builder_tool.ainvoke(
                tool_args_single_day.model_dump(exclude_none=True)
            )
            try:
                single_day_route_details = RouteDetails(**route_data_dict)
                all_route_segments = single_day_route_details.segments or []
                total_duration_seconds_combined = (
                    single_day_route_details.total_duration_seconds or 0
                )
                total_distance_meters_combined = (
                    single_day_route_details.total_distance_meters or 0
                )
                overall_route_status = single_day_route_details.status
                if (
                    single_day_route_details.status != "success"
                    and single_day_route_details.error_message
                ):
                    logger.error(
                        f"Single-day route error: {single_day_route_details.error_message}"
                    )  # Доп. лог

            except ValidationError as ve:
                logger.error(f"Validation error for single-day route data: {ve}")
                overall_route_status = "error"
                all_route_segments.append(
                    RouteSegment(
                        segment_status="error",
                        segment_error_message="Ошибка данных маршрута",
                    )
                )
        elif (
            not event_points_for_api and user_start_coords and len(current_events) == 1
        ):  # Маршрут от пользователя до одного события
            # Эта логика покрывается multiple_days = False и target_events_for_route = current_events
            # если start_location_for_api это юзер, а event_points_for_api это одно событие.
            pass  # Уже должно быть обработано выше
        elif (
            not event_points_for_api
        ):  # Если вообще не к чему строить маршрут (например, одно событие без адреса пользователя)
            return {"current_route_details": None, "is_full_plan_with_route_proposed": False}  # type: ignore

    # Формируем итоговый RouteDetails
    final_route_details = RouteDetails(
        status=overall_route_status,
        segments=all_route_segments,
        total_duration_seconds=total_duration_seconds_combined,
        total_distance_meters=total_distance_meters_combined,
        total_duration_text=(
            f"~{round(total_duration_seconds_combined / 60)} мин"
            if total_duration_seconds_combined
            else None
        ),
        total_distance_text=(
            f"~{round(total_distance_meters_combined / 1000, 1)} км"
            if total_distance_meters_combined
            else None
        ),
        error_message=(
            "Одна или несколько частей маршрута не могли быть построены."
            if overall_route_status == "partial_success"
            else (
                "Ошибка построения маршрута."
                if overall_route_status == "error"
                else None
            )
        ),
    )

    return {
        "current_route_details": final_route_details,
        "is_full_plan_with_route_proposed": final_route_details.status
        in ["success", "partial_success"]
        and bool(final_route_details.segments),
    }


# --- Узел 6: Представление полного плана (мероприятия + маршрут) ---
async def present_full_plan_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: present_full_plan_node executing...")
    current_events: List[Event] = state.get("current_events", [])  # type: ignore
    current_route_details_obj: Optional[RouteDetails] = state.get("current_route_details")  # type: ignore
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore

    # not_found_interest_keys и fallback_candidates уже должны быть обработаны present_initial_plan_node
    # или при обработке ответа на fallback. Здесь мы просто отображаем то, что есть в current_events и current_route_details.

    if not current_events:
        # Эта ситуация маловероятна, если граф работает правильно, т.к. до этого узла должны дойти только с событиями.
        logger.warning("present_full_plan_node: No current events to present.")
        return {
            "status_message_to_user": "Мероприятий для отображения нет. Пожалуйста, начните новый поиск."
        }

    response_parts = ["Вот ваш итоговый план:"]
    # ... (отображение мероприятий, как в последней вашей версии)
    for i, event in enumerate(current_events):
        event_time_str = event.start_time_naive_event_tz.strftime("%H:%M")
        event_date_str = event.start_time_naive_event_tz.strftime(
            "%d.%m.%Y (%A)"
        )  # Добавил день недели
        desc = f"\n{i+1}. **{event.name}** ({event.event_type_key})\n   *Место:* {event.place_name} ({event.place_address or 'Адрес не уточнен'})"
        desc += f"\n   *Время:* {event_date_str} в {event_time_str}"
        if event.duration_minutes:
            desc += f" (продолжительность ~{event.duration_minutes // 60}ч {event.duration_minutes % 60}м)"
        if event.price_text:
            desc += f"\n   *Цена:* {event.price_text}"
        elif event.min_price is not None:
            desc += f"\n   *Цена:* от {event.min_price} руб."
        response_parts.append(desc)

    event_dates_set: Set[date] = {
        evt.start_time_naive_event_tz.date() for evt in current_events
    }
    multiple_days = len(event_dates_set) > 1

    if current_route_details_obj:
        if (
            current_route_details_obj.status in ["success", "partial_success"]
            and current_route_details_obj.segments
        ):
            response_parts.append("\nМаршрут:")
            if multiple_days:
                response_parts.append(
                    "  (Маршруты для каждого дня от вашего местоположения)"
                )

            for idx, segment in enumerate(current_route_details_obj.segments):
                from_name = segment.from_address or (
                    f"Ваше местоположение" if multiple_days else f"Точка {idx+1}"
                )
                to_name = (
                    segment.to_address
                    or f"Мероприятие {idx+1 if not multiple_days else ''}"
                )  # Уточнить имя события для to_name

                # Попытка получить имя события для точки назначения
                # Это упрощенно, т.к. сегменты могут быть не строго по одному на событие в общем случае
                # но для нашей логики (от юзера к каждому событию ИЛИ последовательно) должно работать
                target_event_for_segment: Optional[Event] = None
                if multiple_days and idx < len(current_events):
                    target_event_for_segment = current_events[idx]
                    to_name = f"«{target_event_for_segment.name}»"
                elif not multiple_days and idx < len(
                    current_events
                ):  # Если user_start_address, то idx = 0 это первое событие
                    if collected_data.get("user_start_address_original") and idx < len(
                        current_events
                    ):
                        target_event_for_segment = current_events[idx]
                        to_name = f"«{target_event_for_segment.name}»"
                    elif not collected_data.get(
                        "user_start_address_original"
                    ) and idx + 1 < len(
                        current_events
                    ):  # Маршрут между событиями
                        target_event_for_segment = current_events[idx + 1]
                        to_name = f"«{target_event_for_segment.name}»"

                segment_text = f"  {idx+1}. От '{from_name}' до '{to_name}': "
                if segment.segment_status == "success":
                    segment_text += f"{segment.duration_text or '? мин'}, {segment.distance_text or '? км'}."
                else:
                    segment_text += f"не удалось построить ({segment.segment_error_message or 'причина неизвестна'})."
                response_parts.append(segment_text)

            if current_route_details_obj.status == "partial_success":
                response_parts.append("\n  Не все части маршрута удалось построить.")
            elif (
                not multiple_days
                and current_route_details_obj.total_duration_text
                and len(current_route_details_obj.segments) > 1
            ):  # Общее для однодневного последовательного
                response_parts.append(
                    f"\n  Общее время в пути по маршруту: {current_route_details_obj.total_duration_text}."
                )

        elif current_route_details_obj.status != "success":
            response_parts.append(
                f"\nМаршрут: Не удалось построить ({current_route_details_obj.error_message or 'причина неизвестна'})."
            )

    response_parts.append(
        "\n\nПлан окончательный. Если захотите что-то еще, просто напишите новый запрос!"
    )
    full_plan_text = "\n".join(response_parts)
    new_messages = state.get("messages", []) + [AIMessage(content=full_plan_text)]  # type: ignore

    # Очищаем состояние для возможного нового запроса от пользователя
    # (кроме messages, которые LangGraph обрабатывает)
    # Это лучше делать в условном ребре, ведущем к __END__, или в обработчике /start
    final_collected_data = {}  # Сбрасываем собранные данные для нового цикла

    return {
        "messages": new_messages,
        "status_message_to_user": full_plan_text,
        "collected_data": final_collected_data,  # Очищенные данные
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,  # План уже представлен как финальный
        "awaiting_final_confirmation": False,
        "awaiting_fallback_confirmation": False,
        "pending_fallback_event": None,
    }


# --- Узел 7: Обработка обратной связи по плану ---
async def handle_plan_feedback_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: handle_plan_feedback_node executing...")
    messages = state.get("messages", [])
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore
    current_events: Optional[List[Event]] = state.get("current_events")

    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("handle_plan_feedback_node: No human feedback message found.")
        return {
            "status_message_to_user": "Не получил вашего ответа по плану.",
            "awaiting_final_confirmation": True,
            "collected_data": collected_data,
            "messages": messages,
        }

    user_feedback = messages[-1].content
    llm = get_gigachat_client()
    structured_llm_feedback = llm.with_structured_output(AnalyzedFeedback)
    plan_summary_parts = []
    if current_events:
        for i, event in enumerate(current_events):
            plan_summary_parts.append(
                f"Мероприятие {i+1}: {event.name} ({event.start_time_naive_event_tz.strftime('%d.%m %H:%M')})"
            )
    current_route_details: Optional[RouteDetails] = state.get("current_route_details")  # type: ignore
    if current_route_details and current_route_details.total_duration_text:
        plan_summary_parts.append(
            f"Маршрут: ~{current_route_details.total_duration_text}"
        )
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
            f"LLM Analyzed Feedback: {analyzed_result.model_dump_json(indent=2)}"
        )
        intent = analyzed_result.intent_type
        changes = analyzed_result.change_details
        next_state_update: Dict[str, Any] = {
            "awaiting_final_confirmation": False,
            "collected_data": collected_data,
            "current_events": list(current_events) if current_events else [],
        }

        if intent == "confirm_plan":
            logger.info("User confirmed the plan.")
            next_state_update["status_message_to_user"] = (
                "Отлично! Рад был помочь. Если понадобится что-то еще, обращайтесь!"
            )
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
        elif intent == "request_change" and changes:
            logger.info(f"User requested changes: {changes}")
            next_state_update["previous_confirmed_collected_data"] = dict(
                collected_data
            )
            next_state_update["previous_confirmed_events"] = (
                list(current_events) if current_events else []
            )
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None

            new_collected_data = dict(next_state_update["collected_data"])  # type: ignore
            new_collected_data["clarification_needed_fields"] = []
            change_target = changes.get("change_target")
            new_value = changes.get("new_value")

            if change_target == "budget":
                if isinstance(new_value, (int, float)):
                    new_collected_data["budget_current_search"] = int(new_value)
                    new_collected_data["budget_original"] = int(new_value)
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("budget_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите бюджет числом. Вы: '{new_value}'."
                    )
            elif change_target in ["date", "time"]:
                if isinstance(new_value, str):
                    parsed_date_res = await datetime_parser_tool.ainvoke(
                        {
                            "natural_language_date": new_value,
                            "base_date_iso": datetime.now().isoformat(),
                        }
                    )
                    if parsed_date_res.get("datetime_iso"):
                        new_collected_data["parsed_dates_iso"] = [
                            parsed_date_res["datetime_iso"]
                        ]
                        new_collected_data["dates_description_original"] = new_value
                        if parsed_date_res.get("end_datetime_iso"):
                            new_collected_data["parsed_end_dates_iso"] = [
                                parsed_date_res["end_datetime_iso"]
                            ]
                        elif "parsed_end_dates_iso" in new_collected_data:
                            del new_collected_data["parsed_end_dates_iso"]
                        if parsed_date_res.get("is_ambiguous"):
                            new_collected_data.setdefault(
                                "clarification_needed_fields", []
                            ).append("dates_description_original")
                            next_state_update["clarification_context"] = (
                                parsed_date_res.get("clarification_needed")
                            )
                    else:
                        new_collected_data.setdefault(
                            "clarification_needed_fields", []
                        ).append("dates_description_original")
                        next_state_update["clarification_context"] = (
                            f"Не удалось распознать '{new_value}'. Уточните."
                        )
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("dates_description_original")
                    next_state_update["clarification_context"] = (
                        f"Опишите дату/время текстом. Вы: '{new_value}'."
                    )
            elif change_target in ["interests", "type"] or "event_" in str(
                change_target
            ):
                # ... (логика изменения интересов как в прошлом ответе)
                new_interests_str_list = (
                    [new_value]
                    if isinstance(new_value, str)
                    else (
                        new_value
                        if isinstance(new_value, list)
                        and all(isinstance(s, str) for s in new_value)
                        else []
                    )
                )
                if new_interests_str_list:
                    new_collected_data["interests_original"] = new_interests_str_list
                    mapped_keys = []
                    for s_int in new_interests_str_list:
                        s_l = s_int.lower()
                        key_afisha = None
                        if "фильм" in s_l or "кино" in s_l:
                            key_afisha = "Movie"
                        # ... другие маппинги ...
                        elif "театр" in s_l:
                            key_afisha = "Performance"
                        if not key_afisha:
                            key_afisha = s_int.capitalize()
                        mapped_keys.append(key_afisha)
                    new_collected_data["interests_keys_afisha"] = list(set(mapped_keys))
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("interests_original")
                    next_state_update["clarification_context"] = (
                        f"Назовите интересы. Вы: '{new_value}'."
                    )
            elif change_target == "start_location":
                if isinstance(new_value, str):
                    new_collected_data["user_start_address_original"] = new_value
                    new_collected_data["user_start_address_validated_coords"] = None
                else:
                    new_collected_data.setdefault(
                        "clarification_needed_fields", []
                    ).append("user_start_address_original")
                    next_state_update["clarification_context"] = (
                        f"Укажите адрес. Вы: '{new_value}'."
                    )
            else:
                logger.warning(f"Unknown change_target: {change_target}")
                next_state_update["clarification_context"] = "Не понял, что изменить."
            next_state_update["collected_data"] = new_collected_data
            next_state_update["pending_plan_modification_request"] = None
        elif intent == "new_search":
            next_state_update["status_message_to_user"] = (
                "Хорошо, давайте начнем новый поиск. Что бы вы хотели найти?"
            )
            next_state_update["collected_data"] = {}
            next_state_update["current_events"] = []
            next_state_update["current_route_details"] = None
            next_state_update["is_initial_plan_proposed"] = False
            next_state_update["is_full_plan_with_route_proposed"] = False
        else:  # clarify_misunderstanding или неизвестное
            next_state_update["status_message_to_user"] = (
                "Я вас не совсем понял. Попробуете еще раз или новые критерии?"
            )
            next_state_update["awaiting_final_confirmation"] = (
                True  # Остаемся на подтверждении текущего плана
            )

        if next_state_update.get("status_message_to_user"):
            next_state_update["messages"] = messages + [
                AIMessage(content=next_state_update["status_message_to_user"])
            ]
        else:
            next_state_update["messages"] = messages
        return next_state_update
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}", exc_info=True)
        msg = "Ошибка при обработке ответа. Попробуйте еще раз."
        return {
            "status_message_to_user": msg,
            "awaiting_final_confirmation": True,
            "messages": messages + [AIMessage(content=msg)],
            "collected_data": collected_data,
        }


# --- Узел 8: Подтверждение изменений (для Примера 2 из инструкции) ---
async def confirm_changes_node(
    state: AgentState,
) -> Dict[str, Any]:  # Вероятно, не будет использоваться активно
    logger.info("Node: confirm_changes_node executing (likely deprecated)...")
    messages = state.get("messages", [])
    pending_modification = state.get("pending_plan_modification_request", {})
    if not pending_modification:
        return {
            "status_message_to_user": "Что-то пошло не так.",
            "messages": messages,
            "collected_data": state.get("collected_data"),
        }
    change_summary = "; ".join([f"{k}: {v}" for k, v in pending_modification.items()])
    confirmation_question = (
        f"Правильно понимаю, вы хотите изменить: {change_summary}? (да/нет)"
    )
    new_messages = messages + [AIMessage(content=confirmation_question)]
    return {
        "messages": new_messages,
        "status_message_to_user": confirmation_question,
        "collected_data": state.get("collected_data"),
    }


# --- Узел 9: Сообщение об ошибке, если мероприятия не найдены ---
async def error_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: error_node executing...")
    collected_data: CollectedUserData = dict(state.get("collected_data", {}))  # type: ignore
    not_found_keys: List[str] = collected_data.get("not_found_interest_keys", [])  # type: ignore
    original_interests: List[str] = collected_data.get("interests_original", [])  # type: ignore
    search_criteria_parts = []
    if collected_data.get("city_name"):
        search_criteria_parts.append(f"город '{collected_data['city_name']}'")
    if collected_data.get("dates_description_original"):
        search_criteria_parts.append(
            f"даты '{collected_data['dates_description_original']}'"
        )

    key_to_name = {
        "Movie": "фильмы",
        "Performance": "спектакли",
        "Concert": "концерты",
        "Exhibition": "выставки",
        "SportEvent": "спорт",
        "Excursion": "экскурсии",
        "Event": "события",
    }
    if not_found_keys:
        names = [key_to_name.get(k, k) for k in not_found_keys]
        if len(not_found_keys) == len(collected_data.get("interests_keys_afisha", [])):
            search_criteria_parts.append(f"интересы '{', '.join(original_interests)}'")
        else:
            search_criteria_parts.append(
                f"интересы '{', '.join(original_interests)}' (не найдено: {', '.join(names)})"
            )
    elif original_interests:
        search_criteria_parts.append(f"интересы '{', '.join(original_interests)}'")

    search_criteria_summary = (
        ", ".join(search_criteria_parts)
        if search_criteria_parts
        else "указанным вами критериям"
    )
    error_msg = EVENT_NOT_FOUND_PROMPT_TEMPLATE.format(
        search_criteria_summary=search_criteria_summary
    )

    if "not_found_interest_keys" in collected_data:
        del collected_data["not_found_interest_keys"]  # type: ignore
    if "fallback_candidates" in collected_data:
        del collected_data["fallback_candidates"]  # type: ignore

    new_messages = state.get("messages", []) + [AIMessage(content=error_msg)]  # type: ignore
    return {
        "messages": new_messages,
        "status_message_to_user": error_msg,
        "current_events": [],
        "current_route_details": None,
        "is_initial_plan_proposed": False,
        "is_full_plan_with_route_proposed": False,
        "awaiting_final_confirmation": False,
        "collected_data": collected_data,
    }
