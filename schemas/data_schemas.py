# Файл: schemas/data_schemas.py
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NewActivityValue(BaseModel):
    """Описание новой активности для добавления или замены."""

    activity_type: str = Field(
        description="Тип новой активности (например, 'MOVIE', 'PARK')."
    )
    query_details: Optional[str] = Field(
        default=None, description="Детали запроса для новой активности."
    )
    activity_budget: Optional[int] = Field(
        default=None,
        description="Бюджет для этой конкретной новой активности, если указан.",
    )


class ItemToChangeDetails(BaseModel):
    """Детали элемента плана, к которому относится изменение."""

    item_type: Optional[str] = Field(
        default=None,
        description="Тип изменяемого элемента (например, 'MOVIE', 'CONCERT', 'PARK', 'FOOD_PLACE').",
    )
    item_name: Optional[str] = Field(
        default=None,
        description="Оригинальное название элемента из плана, если пользователь его упоминает.",
    )
    item_index: Optional[int] = Field(
        default=None,
        description="Порядковый номер элемента в представленном плане (1-based), если пользователь ссылается на него.",
    )
    original_query_details: Optional[str] = Field(
        default=None,
        description="Исходные детали запроса для этого элемента, если релевантно.",
    )
    add_after_all: Optional[bool] = Field(
        default=None,
        description="Флаг, указывающий, нужно ли добавить новую активность после всех существующих элементов.",
    )
    item_id_str: Optional[str] = Field(
        default=None,
        description="ID (session_id или id_gis) заменяемого/удаляемого элемента, если известен.",
    )


class ChangeRequestDetail(BaseModel):
    """Детали одного конкретного запроса на изменение в плане."""

    change_target: str = Field(
        description="Цель изменения: 'budget', 'date', 'time', 'interests', 'activity_type', 'specific_event_remove', 'specific_event_replace', 'add_activity', 'start_location', 'activity_budget_change', etc."
    )
    new_value_str: Optional[str] = Field(
        default=None,
        description="Новое строковое значение (для дат, времени, текстовых запросов).",
    )
    new_value_int: Optional[int] = Field(
        default=None,
        description="Новое целочисленное значение (для бюджета, индекса и т.п.).",
    )
    new_value_list_str: Optional[List[str]] = Field(
        default=None, description="Новый список строк (например, для интересов)."
    )
    new_value_activity: Optional[NewActivityValue] = Field(
        default=None, description="Описание новой активности для замены или добавления."
    )
    item_to_change_details: Optional[ItemToChangeDetails] = Field(
        default=None,
        description="Детали идентифицируемого элемента плана, к которому относится изменение.",
    )


class RouteSegment(BaseModel):
    """Детали одного сегмента маршрута между двумя точками."""

    from_address: Optional[str] = Field(
        default=None, description="Адрес начальной точки сегмента (если известен)"
    )
    to_address: Optional[str] = Field(
        default=None, description="Адрес конечной точки сегмента (если известен)"
    )
    duration_seconds: Optional[int] = Field(
        default=None, description="Длительность сегмента в секундах."
    )
    duration_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление длительности сегмента (например, '~15 мин').",
    )
    distance_meters: Optional[int] = Field(
        default=None, description="Расстояние сегмента в метрах."
    )
    distance_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление расстояния сегмента (например, '~2 км').",
    )
    transport_type: Optional[str] = Field(
        default=None, description="Тип транспорта для этого сегмента."
    )
    segment_status: str = Field(
        default="unknown",
        description="Статус построения этого сегмента ('success', 'error')",
    )
    segment_error_message: Optional[str] = Field(
        default=None,
        description="Сообщение об ошибке для этого сегмента, если segment_status='error'",
    )


class OrderedActivityItem(BaseModel):
    """Представление одной активности в запрошенной пользователем последовательности."""

    activity_type: str = Field(
        description="Тип активности, например, 'MOVIE', 'CONCERT', 'PARK', 'FOOD_PLACE', 'STAND_UP', 'MUSEUM_EXHIBITION', 'UNKNOWN_INTEREST'."
    )
    query_details: Optional[str] = Field(
        default=None,
        description="Оригинальная формулировка пользователя для этой активности, если применимо.",
    )
    activity_budget: Optional[int] = Field(
        default=None,
        description="Бюджет, специфичный для данной активности, если он был указан пользователем для нее.",
    )


class OverBudgetOption(BaseModel):
    """
    Описание опции для добавления в план, которая приводит к превышению бюджета.
    Используется, когда есть частичный план в рамках бюджета, но можно добавить еще что-то сверх него.
    """

    items_to_add: List[Dict[str, Any]] = Field(
        description="Список мероприятий (сериализованных в dict), которые можно добавить в план."
    )
    total_plan_cost: int = Field(
        description="Итоговая стоимость всего плана, если эти мероприятия будут добавлены."
    )
    overage_amount: int = Field(
        description="Сумма, на которую будет превышен исходный бюджет пользователя."
    )
    person_count: int = Field(
        default=1, description="Количество человек, для которого рассчитана стоимость."
    )


class ConstructionProblemInfo(BaseModel):
    """
    Структурированное описание проблемы, возникшей при построении плана для конкретного интереса.
    """

    problem_type: str = Field(
        description="Тип проблемы: 'date' (нет вариантов на дату), 'budget' (варианты есть, но они не вписываются в бюджет)."
    )
    interest_key: str = Field(
        description="Ключ интереса (activity_type), с которым возникла проблема. Например, 'CONCERT'."
    )
    details: Optional[str] = Field(
        default=None,
        description="Дополнительные детали проблемы. Например, 'Самый дешевый билет стоит 2000, что превышает бюджет'.",
    )


class PlanConstructionResult(BaseModel):
    """
    Итоговая структура, содержащая результат работы узла `search_events_node`.
    Агрегирует в себе все возможные варианты для представления пользователю:
    - успешный полный план
    - частичный план с опцией превышения бюджета
    - альтернативы по дате/бюджету
    - информацию о возникших проблемах.
    """

    in_budget_plan_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Список мероприятий (Afisha Event или POI, сериализованных в dict), которые формируют план, полностью укладывающийся в бюджет.",
    )
    over_budget_option: Optional[OverBudgetOption] = Field(
        default=None,
        description="Опция для добавления мероприятий сверх бюджета, если был составлен частичный план.",
    )
    date_fallback_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Список мероприятий-кандидатов на другие даты (если на искомую дату ничего не найдено).",
    )
    budget_fallback_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Список мероприятий-кандидатов, которые не вписались в бюджет, но могут быть предложены как альтернатива.",
    )
    problems: List[ConstructionProblemInfo] = Field(
        default_factory=list,
        description="Структурированный список проблем, возникших при построении плана (нет на дату, слишком дорого и т.д.).",
    )
    person_count: int = Field(
        default=1, description="Количество человек, на которое рассчитан план."
    )
    user_budget: Optional[int] = Field(
        default=None, description="Изначальный общий бюджет пользователя."
    )


class ExtractedInitialInfo(BaseModel):
    """Извлечённая из пользовательского запроса информация для подбора мероприятий."""

    city: Optional[str] = Field(default=None, description="Название города.")
    dates_description: Optional[str] = Field(
        default=None, description="Описание дат или периода."
    )
    ordered_activities: Optional[List[OrderedActivityItem]] = Field(
        default=None, description="Упорядоченный список активностей."
    )
    budget: Optional[int] = Field(
        default=None, description="Общий бюджет пользователя, если указан."
    )
    person_count: Optional[int] = Field(
        default=1, description="Количество человек, если указано."
    )
    raw_time_description: Optional[str] = Field(
        default=None, description="Необработанное описание времени."
    )


class DateTimeParserToolArgs(BaseModel):
    """Аргументы для инструмента парсинга описания даты и времени на естественном языке."""

    natural_language_date: str = Field(
        description="Описание даты или периода на естественном языке."
    )
    natural_language_time_qualifier: Optional[str] = Field(
        default=None, description="Дополнительное описание времени."
    )
    base_date_iso: Optional[str] = Field(
        default=None, description="ISO строка базовой даты."
    )


class EventSearchToolArgs(BaseModel):
    """Аргументы для инструмента поиска мероприятий, учитывающие город, даты, тип события и другие фильтры."""

    city_id: int = Field(
        description="Числовой ID города (согласно API Афиши) для поиска мероприятий."
    )
    date_from: datetime = Field(
        description="Дата и время начала периода поиска мероприятий (объект datetime)."
    )
    date_to: datetime = Field(
        description="Дата и время окончания периода поиска мероприятий (объект datetime)."
    )
    user_creation_type_key: str = Field(
        default="ANY",
        description="Строковый ключ типа события, который запросил пользователь.",
    )
    filter_keywords_in_name: Optional[List[str]] = Field(
        default=None,
        description="Список ключевых слов для фильтрации мероприятий по названию.",
    )
    filter_genres: Optional[List[str]] = Field(
        default=None, description="Список жанров для фильтрации мероприятий."
    )
    filter_tags: Optional[List[str]] = Field(
        default=None, description="Список тегов для фильтрации мероприятий."
    )
    min_start_time_naive: Optional[str] = Field(
        default=None,
        description="Минимальное ВРЕМЯ НАЧАЛА мероприятия (строка 'ЧЧ:ММ', наивное) для фильтрации.",
    )
    max_start_time_naive: Optional[str] = Field(
        default=None,
        description="Максимальное ВРЕМЯ НАЧАЛА мероприятия (строка 'ЧЧ:ММ', наивное) для фильтрации.",
    )
    max_budget_per_person: Optional[int] = Field(
        default=None,
        description="Максимальный бюджет на одного человека в рублях для фильтрации мероприятий по цене.",
    )
    time_constraints_for_next_event: Optional[Dict[str, datetime]] = Field(
        default=None,
        description="Словарь с временными ограничениями для следующего мероприятия.",
    )
    exclude_session_ids: Optional[List[int]] = Field(
        default=None,
        description="Список числовых ID сессий мероприятий, которые нужно исключить из результатов поиска.",
    )
    city_name: Optional[str] = Field(
        default=None,
        description="Название города для дополнительной фильтрации результатов, чтобы исключить пригороды.",
    )


class LocationModel(BaseModel):
    """Модель для представления географической локации с адресом и/или координатами широты и долготы."""

    address_string: Optional[str] = Field(
        default=None, description="Полный текстовый адрес объекта или точки."
    )
    lon: Optional[float] = Field(
        default=None, description="Географическая долгота точки (longitude)."
    )
    lat: Optional[float] = Field(
        default=None, description="Географическая широта точки (latitude)."
    )

    @field_validator("lon", "lat", mode="before", check_fields=False)
    @classmethod
    def check_coords_v2(cls, v: Any, info: Any) -> Any:
        current_field_name = info.field_name
        data = info.data
        lon_val = data.get("lon") if current_field_name != "lon" else v
        lat_val = data.get("lat") if current_field_name != "lat" else v
        address_str_val = data.get("address_string")
        if lon_val is not None and lat_val is not None:
            return v
        if address_str_val:
            return v
        return v


class RouteBuilderToolArgs(BaseModel):
    """Аргументы для инструмента построения маршрута между несколькими географическими точками."""

    start_point: LocationModel = Field(description="Начальная точка маршрута.")
    event_points: List[LocationModel] = Field(description="Список точек мероприятий.")
    transport_type: str = Field(
        default="driving", description="Предпочитаемый тип транспорта."
    )
    city_context_for_geocoding: Optional[str] = Field(
        default=None, description="Городской контекст для геокодирования."
    )


class Event(BaseModel):
    """Детальная информация о конкретном мероприятии, включая место, время, цену и другие атрибуты."""

    session_id: int = Field(
        description="Уникальный числовой ID сеанса мероприятия от API Афиши."
    )
    afisha_id: Optional[str] = Field(
        default=None,
        description="Уникальный строковый ID самого события (Creation ID) от API Афиши.",
    )
    name: str = Field(description="Полное название мероприятия.")
    user_event_type_key: str = Field(
        description="Строковый ключ типа события, который был запрошен пользователем или использовался для поиска."
    )
    api_creation_type: Optional[str] = Field(
        default=None,
        description="Фактический тип 'творения' (CreationType) из API Афиши.",
    )
    place_name: str = Field(description="Название места проведения мероприятия.")
    place_address: Optional[str] = Field(
        default=None, description="Полный текстовый адрес места проведения."
    )
    place_coords_lon: Optional[float] = Field(
        default=None, description="Географическая долгота места проведения."
    )
    place_coords_lat: Optional[float] = Field(
        default=None, description="Географическая широта места проведения."
    )
    start_time_iso: str = Field(
        description="Время начала мероприятия в формате ISO строки с указанием часового пояса."
    )
    start_time_naive_event_tz: datetime = Field(
        description="Наивное время начала мероприятия (объект datetime) в локальном часовом поясе события."
    )
    duration_minutes: Optional[int] = Field(
        default=None,
        description="Продолжительность мероприятия в минутах, если известна.",
    )
    min_price: Optional[int] = Field(
        default=None,
        description="Минимальная цена билета на мероприятие в рублях, если известна.",
    )
    max_price: Optional[int] = Field(
        default=None,
        description="Максимальная цена билета на мероприятие в рублях, если известна.",
    )
    price_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление диапазона цен на мероприятие.",
    )
    duration_description: Optional[str] = Field(
        default=None, description="Текстовое описание продолжительности мероприятия."
    )
    rating: Optional[Any] = Field(
        default=None, description="Рейтинг мероприятия, если доступен."
    )
    age_restriction: Optional[str] = Field(
        default=None, description="Возрастное ограничение для мероприятия."
    )
    genres: Optional[List[str]] = Field(
        default=None, description="Список жанров мероприятия, полученных из API."
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Список тегов мероприятия, полученных из API."
    )
    rubric: Optional[Dict[str, Any]] = Field(
        default=None, description="Информация о рубрике мероприятия из API."
    )


class RouteDetails(BaseModel):
    """Полные детали построенного маршрута, включая все сегменты."""

    status: str = Field(description="Общий статус построения всего маршрута.")
    segments: Optional[List[RouteSegment]] = Field(
        default=None, description="Список сегментов маршрута."
    )
    total_duration_seconds: Optional[int] = Field(
        default=None,
        description="Суммарная длительность всех успешно построенных сегментов в секундах.",
    )
    total_duration_text: Optional[str] = Field(
        default=None, description="Текстовое представление суммарной длительности."
    )
    total_distance_meters: Optional[int] = Field(
        default=None,
        description="Суммарное расстояние всех успешно построенных сегментов в метрах.",
    )
    total_distance_text: Optional[str] = Field(
        default=None, description="Текстовое представление суммарного расстояния."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Общее сообщение об ошибке, если status не 'success' или 'partial_success'.",
    )


class ParsedDateTime(BaseModel):
    """Структурированный результат парсинга описания даты и времени из текста."""

    year: Optional[int] = Field(default=None, description="Извлеченный год.")
    month: Optional[int] = Field(default=None, description="Извлеченный месяц.")
    day: Optional[int] = Field(default=None, description="Извлеченный день месяца.")
    hour: Optional[int] = Field(default=None, description="Извлеченный час НАЧАЛА.")
    minute: Optional[int] = Field(
        default=None, description="Извлеченная минута НАЧАЛА."
    )
    end_hour: Optional[int] = Field(
        default=None, description="Извлеченный час КОНЦА временного диапазона."
    )
    end_minute: Optional[int] = Field(
        default=None, description="Извлеченная минута КОНЦА временного диапазона."
    )
    is_ambiguous: bool = Field(
        default=False,
        description="Флаг, указывающий, является ли распознанная дата/время неполной или неоднозначной.",
    )
    clarification_needed: Optional[str] = Field(
        default=None,
        description="Текст уточняющего вопроса для пользователя, если is_ambiguous=True.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Сообщение об ошибке, если не удалось распознать дату или время из текста.",
    )


class AnalyzedFeedback(BaseModel):
    """Результат анализа обратной связи пользователя по предложенному плану мероприятий и маршруту."""

    intent_type: str = Field(
        description="Классифицированный тип намерения пользователя: 'confirm_plan', 'request_change', 'clarify_misunderstanding', 'new_search'."
    )
    change_requests: Optional[List[ChangeRequestDetail]] = Field(
        default=None,
        description="Список деталей запрошенных изменений, если intent_type='request_change'.",
    )


class FoodPlaceInfo(BaseModel):
    """Детальная информация о заведении питания из GIS-сервиса."""

    id_gis: str = Field(description="Уникальный ID заведения в GIS-сервисе.")
    name: str = Field(description="Название заведения.")
    full_name: Optional[str] = Field(
        default=None,
        description="Полное название заведения, если отличается от основного.",
    )
    address: Optional[str] = Field(default=None, description="Адрес заведения.")
    coords: Optional[List[float]] = Field(
        default=None, description="Координаты заведения [долгота, широта]."
    )
    schedule_str: str = Field(
        default="Время работы не указано",
        description="Строковое представление расписания работы.",
    )
    rating_str: str = Field(
        default="Рейтинг не указан",
        description="Строковое представление рейтинга (например, '4.5/5 (100 отзывов)').",
    )
    avg_bill_str: str = Field(
        default="Средний чек не указан",
        description="Строковое представление среднего чека (например, '700-1000 ₽').",
    )
    avg_bill_numeric: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = Field(
        default=None,
        description="Числовое представление среднего чека: одно число или кортеж (min_bill, max_bill).",
    )
    contacts_str: str = Field(
        default="Контакты не указаны",
        description="Строковое представление контактной информации.",
    )
    type_gis: Optional[str] = Field(
        default=None, description="Тип объекта в GIS (например, 'branch')."
    )
    rubrics_gis: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Список рубрик из GIS-сервиса."
    )
    raw_attribute_groups: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Сырые данные групп атрибутов из GIS API."
    )
    raw_adverts: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Сырые данные рекламных объявлений из GIS API."
    )
    raw_contact_groups: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Сырые данные контактных групп из GIS API."
    )
    raw_reviews: Optional[Dict[str, Any]] = Field(
        default=None, description="Сырые данные отзывов из GIS API."
    )


class ParkInfo(BaseModel):
    """Детальная информация о парке или месте для прогулок из GIS-сервиса."""

    id_gis: str = Field(description="Уникальный ID парка в GIS-сервисе.")
    name: str = Field(description="Название парка.")
    full_name: Optional[str] = Field(
        default=None, description="Полное название парка, если отличается от основного."
    )
    address: Optional[str] = Field(
        default=None, description="Адрес парка или его описание местоположения."
    )
    coords: Optional[List[float]] = Field(
        default=None, description="Координаты парка [долгота, широта]."
    )
    schedule_str: str = Field(
        default="Время работы не указано",
        description="Строковое представление расписания работы (если применимо).",
    )
    type_gis: Optional[str] = Field(default=None, description="Тип объекта в GIS.")
    rubrics_gis: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Список рубрик из GIS-сервиса."
    )


class CollectedUserData(BaseModel):
    """Агрегированные данные, собранные в ходе диалога с пользователем."""

    city_name: Optional[str] = Field(default=None, description="Название города.")
    city_id_afisha: Optional[int] = Field(
        default=None, description="ID города для API Афиши."
    )
    interests_original: Optional[List[str]] = Field(
        default=None, description="Исходные интересы пользователя (список строк)."
    )
    interests_keys_afisha: Optional[List[str]] = Field(
        default=None,
        description="Ключи интересов, преобразованные для использования с API Афиши или внутренними типами.",
    )
    ordered_activities: Optional[List[OrderedActivityItem]] = Field(
        default=None, description="Упорядоченный список активностей с деталями."
    )
    budget_original: Optional[int] = Field(
        default=None, description="Общий бюджет пользователя на весь план."
    )
    person_count: Optional[int] = Field(
        default=1, description="Количество человек, для которого составляется план."
    )
    dates_description_original: Optional[str] = Field(
        default=None, description="Оригинальное описание дат от пользователя."
    )
    raw_time_description_original: Optional[str] = Field(
        default=None, description="Оригинальное описание времени от пользователя."
    )
    parsed_dates_iso: Optional[List[str]] = Field(
        default=None, description="Список ISO строк дат начала."
    )
    parsed_end_dates_iso: Optional[List[str]] = Field(
        default=None, description="Список ISO строк дат окончания."
    )
    user_start_address_original: Optional[str] = Field(
        default=None, description="Исходный адрес отправления пользователя."
    )
    user_start_address_validated_coords: Optional[Dict[str, float]] = Field(
        default=None,
        description="Валидированные координаты адреса пользователя {'lon': X, 'lat': Y}.",
    )
    partial_address_street: Optional[str] = Field(
        default=None, description="Частично распознанная улица, ожидается номер дома."
    )
    address_clarification_status: Optional[str] = Field(
        default=None,
        description="Статус уточнения адреса (например, 'NEED_HOUSE_NUMBER', 'ADDRESS_NOT_FOUND').",
    )
    last_geocoding_attempt_full_address: Optional[str] = Field(
        default=None,
        description="Последний полный адрес, по которому производилась попытка геокодирования.",
    )
    awaiting_address_input: bool = Field(
        default=False, description="Флаг ожидания ввода адреса от пользователя."
    )
    poi_park_query: Optional[str] = Field(
        default=None, description="Запрос пользователя на поиск парка."
    )
    poi_food_query: Optional[str] = Field(
        default=None, description="Запрос пользователя на поиск заведения питания."
    )
    selected_pois_for_plan: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Список POI (парки, еда), включенных в план (сырые данные).",
    )
    awaiting_fallback_confirmation: bool = Field(
        default=False,
        description="Флаг ожидания подтверждения fallback-варианта для событий Афиши.",
    )
    pending_fallback_event: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Данные fallback-мероприятия (Afisha или POI), ожидающего подтверждения.",
    )
    last_offered_fallback_for_interest: Optional[str] = Field(
        default=None,
        description="Ключ интереса, для которого последним предлагался fallback.",
    )
    fallback_accepted_and_plan_updated: bool = Field(
        default=False, description="Флаг, что fallback был принят и план обновлен."
    )
    plan_construction_result: Optional[PlanConstructionResult] = Field(
        default=None,
        description="Агрегированный результат последней попытки построения плана, содержащий все варианты (в бюджете, сверх бюджета, альтернативы) и проблемы.",
    )
    awaiting_menu_choice: bool = Field(
        default=False,
        description="Флаг, указывающий, что агент ожидает от пользователя выбор из предложенного меню (например, '1', '2', 'оба').",
    )
    last_offered_menu: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Сохраняем структуру последнего предложенного меню, чтобы обработать ответ пользователя (например, {'1': event_data, '2': event_data}).",
    )
    clarification_needed_fields: List[str] = Field(
        default_factory=list,
        description="Список полей, требующих уточнения у пользователя.",
    )
    previous_confirmed_collected_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Сохраненные данные предыдущего подтвержденного набора критериев.",
    )
    previous_confirmed_events: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Список мероприятий предыдущего подтвержденного плана.",
    )
    plan_construction_failed_step: Optional[str] = Field(
        default=None,
        description="Сообщение о шаге, на котором не удалось построить план.",
    )
    previous_plan_items_for_modification: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Список элементов предыдущего плана для модификации."
    )
    current_excluded_ids: Dict[str, List[Union[int, str]]] = Field(
        default_factory=lambda: {
            "afisha": [],
            "park": [],
            "food": [],
            "afisha_names_to_avoid": [],
            "afisha_creation_ids_to_avoid": [],
        },
        description="ID и имена исключенных элементов.",
    )
    plan_construction_strategy: str = Field(
        default="standard",
        description="Стратегия построения плана (например, 'standard', 'optimize_poi_time').",
    )
    last_poi_search_results: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Последние результаты поиска POI {'park': [...], 'food': [...]}.",
    )
    poi_warnings_in_current_plan: List[str] = Field(
        default_factory=list,
        description="Предупреждения, связанные с POI в текущем плане.",
    )

    class Config:
        extra = "allow"
