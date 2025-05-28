from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages  # Правильный импорт для LangGraph

from schemas.data_schemas import Event, RouteDetails  # Наши Pydantic модели

# --- Вспомогательные TypedDict для collected_data ---


class CollectedUserData(TypedDict, total=False):
    """
    Данные, собранные от пользователя или извлеченные в ходе диалога.
    total=False означает, что не все ключи обязательны.
    """

    city_name: Optional[str]
    city_id_afisha: Optional[int]  # ID города для API Афиши

    # Даты могут быть представлены как диапазон или одна дата
    # datetime_parser_tool вернет ISO строку, которую мы можем хранить
    # или преобразовывать в datetime объекты при необходимости
    parsed_dates_iso: Optional[List[str]]  # Список из 1 или 2 ISO строк дат

    # Оригинальное описание дат от пользователя для возможных уточнений
    dates_description_original: Optional[str]

    interests_original: Optional[List[str]]  # Исходные интересы пользователя словами
    interests_keys_afisha: Optional[List[str]]  # Ключи интересов для API Афиши

    budget_original: Optional[int]  # Исходный бюджет
    budget_current_search: Optional[int]  # Бюджет, используемый для текущего поиска

    user_start_address_original: Optional[str]  # Исходный адрес пользователя
    user_start_address_validated_coords: Optional[
        Dict[str, float]
    ]  # {'lon': X, 'lat': Y}

    # Если пользователь указал время в нечетком формате (например, "вечером")
    raw_time_description_original: Optional[str]
    # Уточненное время после работы datetime_parser_tool или дальнейшей логики агента
    # Может быть диапазоном { "start_time_naive": datetime, "end_time_naive": datetime }
    # или одним временем { "exact_time_naive": datetime }
    clarified_time_naive: Optional[Dict[str, datetime]]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    collected_data: CollectedUserData
    current_events: Optional[List[Event]]
    current_route_details: Optional[RouteDetails]
    status_message_to_user: Optional[str]
    clarification_needed_fields: Optional[List[str]]
    clarification_context: Optional[Any]
    awaiting_clarification_for_field: Optional[str] 
    is_initial_plan_proposed: bool
    is_full_plan_with_route_proposed: bool
    awaiting_final_confirmation: bool
    pending_plan_modification_request: Optional[Dict[str, Any]]
    previous_confirmed_collected_data: Optional[CollectedUserData]
    previous_confirmed_events: Optional[List[Event]]
