# Файл: agent_core/agent_state.py
from typing import List, Optional, Dict, Any, Union
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from schemas.data_schemas import Event, RouteDetails, OrderedActivityItem

# [ДОБАВЛЕНО] Новая структура для детальной информации о проблеме с бюджетом
class BudgetIssueDetails(TypedDict, total=False):
    """Детали для представления плана, который превышает бюджет."""
    issue_type: str  # 'combination' (сумма > бюджет) или 'single_item_too_expensive' (один элемент > бюджета)
    overage_amount: int  # На сколько превышен бюджет
    total_cost: int  # Итоговая стоимость "проблемного" плана
    person_count: int # Количество человек, для которого считалось
    ok_plan_part: List[Dict[str, Any]]  # Часть плана, которая укладывается в бюджет (список Event/POI словарей)
    problematic_item: Dict[str, Any] # "Проблемный" элемент, который не влез (Event/POI словарь)


class LastPresentedPlanInfo(TypedDict, total=False):
    events: Optional[List[Dict[str, Any]]]
    selected_pois: Optional[List[Dict[str, Any]]]
    route_details: Optional[Dict[str, Any]]

class CollectedUserData(TypedDict, total=False):
    # ОБЩАЯ ИНФОРМАЦИЯ О ЗАПРОСЕ
    city_name: Optional[str]
    city_id_afisha: Optional[int]
    dates_description_original: Optional[str]
    raw_time_description_original: Optional[str]
    parsed_dates_iso: Optional[List[str]]
    parsed_end_dates_iso: Optional[List[str]]
    
    # ИНТЕРЕСЫ И АКТИВНОСТИ
    interests_original: Optional[List[str]]
    interests_keys_afisha: Optional[List[str]]
    ordered_activities: Optional[List[OrderedActivityItem]]
    poi_park_query: Optional[str]
    poi_food_query: Optional[str]

    # БЮДЖЕТ И КОЛИЧЕСТВО ЧЕЛОВЕК
    budget_original: Optional[int]
    person_count: int

    # АДРЕС ПОЛЬЗОВАТЕЛЯ
    user_start_address_original: Optional[str]
    user_start_address_validated_coords: Optional[Dict[str, float]]
    partial_address_street: Optional[str]
    address_clarification_status: Optional[str]
    last_geocoding_attempt_full_address: Optional[str]
    awaiting_address_input: bool

    # РЕЗУЛЬТАТЫ ПОИСКА И ПОСТРОЕНИЯ ПЛАНА
    # [ИЗМЕНЕНО] Убрал plan_construction_result в пользу более конкретных полей
    selected_pois_for_plan: Optional[List[Dict[str, Any]]]
    plan_construction_strategy: Optional[str]
    plan_construction_failed_step: Optional[str]
    poi_warnings_in_current_plan: Optional[List[str]]
    last_poi_search_results: Optional[Dict[str, List[Dict[str, Any]]]]
    
    # [ДОБАВЛЕНО] Новые поля для обработки проблем с бюджетом и альтернатив
    budget_issue_details: Optional[BudgetIssueDetails] # Для Сценариев 1 и 2
    date_fallback_candidates: Optional[List[Dict[str, Any]]] # Для Сценария 3 (альтернативы на другие даты)
    budget_fallback_candidates: Optional[List[Dict[str, Any]]] # Для Сценария 3 (альтернативы дешевле/дороже)


    # УПРАВЛЕНИЕ ДИАЛОГОМ (УТОЧНЕНИЯ, FALLBACK'И, МЕНЮ)
    clarification_needed_fields: List[str]
    awaiting_fallback_confirmation: bool
    pending_fallback_event: Optional[Dict[str, Any]]
    last_offered_fallback_for_interest: Optional[str]
    fallback_accepted_and_plan_updated: bool
    awaiting_menu_choice: bool
    last_offered_menu: Optional[Dict[str, Any]]

    # ИСТОРИЯ И ИСКЛЮЧЕНИЯ
    previous_confirmed_collected_data: Optional[Dict[str, Any]]
    previous_confirmed_events: Optional[List[Dict[str, Any]]]
    previous_plan_items_for_modification: Optional[List[Dict[str, Any]]]
    current_excluded_ids: Optional[Dict[str, List[Union[int, str]]]]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    collected_data: CollectedUserData
    current_events: Optional[List[Event]]
    current_route_details: Optional[RouteDetails]
    status_message_to_user: Optional[str]
    awaiting_clarification_for_field: Optional[str]
    clarification_context: Optional[Any]
    is_initial_plan_proposed: bool
    is_full_plan_with_route_proposed: bool
    awaiting_final_confirmation: bool
    awaiting_feedback_on_final_plan: bool
    just_modified_plan: bool
    last_presented_plan: Optional[LastPresentedPlanInfo]
    modification_request_details: Optional[Dict[str, Any]]
    plan_modification_pending: bool
