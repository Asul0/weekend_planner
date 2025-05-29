from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages  # Правильный импорт для LangGraph

from schemas.data_schemas import Event, RouteDetails  # Наши Pydantic модели

# --- Вспомогательные TypedDict для collected_data ---


class CollectedUserData(TypedDict, total=False):
    city_name: Optional[str]
    city_id_afisha: Optional[int]
    parsed_dates_iso: Optional[List[str]]
    dates_description_original: Optional[str]
    interests_original: Optional[List[str]]
    interests_keys_afisha: Optional[List[str]]
    budget_original: Optional[int]
    budget_current_search: Optional[int]
    user_start_address_original: Optional[str]
    user_start_address_validated_coords: Optional[Dict[str, float]]
    raw_time_description_original: Optional[str]
    clarified_time_naive: Optional[Dict[str, datetime]] # Это поле было в state.txt

    # Добавленные/измененные поля в CollectedUserData, если они там логичны
    # (например, ошибки поиска на уровне интересов, которые могут быть важны для последующих уточнений)
    search_errors_by_interest: Optional[Dict[str, str]] # Ошибки от gather_all_candidate_events_node
    partial_address_street: Optional[str] # Уже было в schema.txt для CollectedUserData Pydantic
    awaiting_address_input: bool # Уже было в schema.txt, но управляется узлами

    # Флаги, которые управляют логикой диалога и были в вашей Pydantic схеме для CollectedUserData
    clarification_needed_fields: Optional[List[str]]
    awaiting_fallback_confirmation: bool
    pending_fallback_event: Optional[Dict[str, Any]]
    last_offered_fallback_for_interest: Optional[str]
    fallback_accepted_and_plan_updated: bool
    not_found_interest_keys_in_primary_search: Optional[List[str]]
    fallback_candidates: Optional[Dict[str, Dict[str, Any]]]
    
    previous_confirmed_collected_data: Optional[Dict[str, Any]] # Если оставляем Dict
    previous_confirmed_events: Optional[List[Dict[str, Any]]] # Если оставляем List[Dict]

class AgentState(TypedDict):
    candidate_events_by_interest: Optional[Dict[str, List[Event]]] 
    messages: Annotated[List[BaseMessage], add_messages]
    collected_data: CollectedUserData
    unplanned_interest_keys: Optional[List[str]] 
    current_events: Optional[List[Event]]
    optimal_chain_construction_message: Optional[str]
    current_route_details: Optional[RouteDetails]
    actual_total_travel_time: Optional[int]
    status_message_to_user: Optional[str]
    clarification_needed_fields: Optional[List[str]]
    clarification_context: Optional[Any]
    awaiting_clarification_for_field: Optional[str] 
    awaiting_fallback_confirmation: bool # У тебя уже есть
    pending_fallback_event: Optional[Dict[str, Any]] # У тебя уже есть
    last_offered_fallback_for_interest: Optional[str] 
    fallback_accepted_and_plan_updated: bool # У тебя уже есть
    not_found_interest_keys_in_primary_search: Optional[List[str]]
    is_initial_plan_proposed: bool
    is_full_plan_with_route_proposed: bool
    awaiting_final_confirmation: bool
    pending_plan_modification_request: Optional[Dict[str, Any]]
    previous_confirmed_collected_data: Optional[CollectedUserData] # Используем тип CollectedUserData
    previous_confirmed_events: Optional[List[Event]] # Используем тип Event
