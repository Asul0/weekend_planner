from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)  # <--- Изменяем импорт для field_validator
from typing import Optional, Any
import logging


# --- Схемы для извлечения информации из запроса пользователя ---
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
    # Примечание: можно добавить поле 'error_message' для сегмента, если его построение не удалось
    segment_status: str = Field(
        default="unknown",
        description="Статус построения этого сегмента ('success', 'error')",
    )
    segment_error_message: Optional[str] = Field(
        default=None,
        description="Сообщение об ошибке для этого сегмента, если segment_status='error'",
    )


class ExtractedInitialInfo(BaseModel):
    """Извлечённая из пользовательского запроса информация для подбора мероприятий: город, даты, интересы, бюджет, исходное время."""

    city: Optional[str] = Field(
        default=None,
        description="Название города, в котором пользователь хочет найти мероприятие.",
    )
    dates_description: Optional[str] = Field(
        default=None,
        description="Описание дат или периода времени словами, когда пользователь хочет пойти на мероприятие (например: 'на завтра', 'на выходных', '5 июня', 'вечером').",
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="Список интересов или типов мероприятий, которые указал пользователь (например: ['фильм'], ['концерт', 'джаз'], ['театр']). Поле ВСЕГДА должно быть списком строк, даже если интерес один.",
    )
    budget: Optional[int] = Field(  # Изменил на int для бюджета
        default=None,
        description="Примерный бюджет пользователя на мероприятие в рублях (например: 1000, 2500), если указан.",
    )
    raw_time_description: Optional[str] = Field(
        default=None,
        description="Необработанное описание времени, если пользователь указал что-то вроде 'вечером', 'днем', которое требует дополнительного уточнения.",
    )

    @field_validator("interests", mode="before")
    @classmethod
    def interests_always_list(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            if v.strip() == "":  # Если строка пустая, считаем как None
                return None
            return [v.strip()]
        if isinstance(v, list):
            # Очищаем список от пустых строк и оставляем только непустые
            return [
                item.strip() for item in v if isinstance(item, str) and item.strip()
            ] or None
        # Если тип не строка и не список, или список пустых строк, Pydantic вызовет ошибку при валидации типа List[str]
        # или вернется None, что тоже приемлемо для Optional[List[str]]
        logger = logging.getLogger(__name__)  # Временный логгер для отладки валидатора
        logger.warning(
            f"Validator interests_always_list received unexpected type {type(v)} or content, returning as is for Pydantic to handle."
        )
        return v  # Позволим Pydantic обработать другие типы


# --- Схемы для инструментов ---


class DateTimeParserToolArgs(BaseModel):
    """Аргументы для инструмента парсинга описания даты и времени на естественном языке."""

    natural_language_date: str = Field(
        description="Описание даты или периода на естественном языке (например, 'завтра', 'на выходных', '15 июля', 'сегодня'). Может включать и время, если оно там есть."
    )
    natural_language_time_qualifier: Optional[str] = Field(
        default=None,
        description="Дополнительное описание времени, если оно было извлечено отдельно (например, 'после 18:00', 'утром', 'с 14 до 17'). Может быть None.",
    )
    base_date_iso: Optional[str] = Field(
        default=None,
        description="ISO строка базовой даты (YYYY-MM-DDTHH:MM:SS) для разрешения относительных дат. Если не указана, используется текущая дата и время.",
    )


class EventSearchToolArgs(BaseModel):
    """Аргументы для инструмента поиска мероприятий через API Афиши, учитывающие город, даты, интересы и другие фильтры."""

    city_id: int = Field(
        description="Числовой ID города (согласно API Афиши) для поиска мероприятий."
    )
    date_from: datetime = Field(
        description="Дата и время начала периода поиска мероприятий (объект datetime). Фактически используется как дата начала дня."
    )
    date_to: datetime = Field(
        description="Дата и время окончания периода поиска мероприятий (объект datetime). Фактически используется как дата начала следующего дня."
    )
    interests_keys: Optional[List[str]] = Field(
        default=None,
        description="Список строковых ключей типов мероприятий (например, ['Movie', 'Concert']) для фильтрации в API Афиши.",
    )
    min_start_time_naive: Optional[datetime] = Field(
        default=None,
        description="Минимальное ВРЕМЯ НАЧАЛА мероприятия (наивное, без таймзоны, объект datetime) для дополнительной фильтрации. Мероприятие должно НАЧАТЬСЯ не ранее этого времени.",
    )
    max_start_time_naive: Optional[datetime] = Field(
        default=None,
        description="Максимальное ВРЕМЯ НАЧАЛА мероприятия (наивное, без таймзоны, объект datetime) для фильтрации. Мероприятие должно НАЧАТЬСЯ не позже этого времени.",
    )  # НОВОЕ
    max_budget_per_person: Optional[int] = Field(
        default=None,
        description="Максимальный бюджет на одного человека в рублях для фильтрации мероприятий по цене.",
    )
    time_constraints_for_next_event: Optional[Dict[str, datetime]] = Field(
        default=None,
        description="Словарь с временными ограничениями для следующего мероприятия...",
    )  # остается
    exclude_session_ids: Optional[List[int]] = Field(
        default=None,
        description="Список числовых ID сессий мероприятий, которые нужно исключить из результатов поиска.",
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
    def check_coords_v2(
        cls, v: Any, info: Any
    ) -> Any:  # Оставил Any для info для совместимости, реально это FieldValidationInfo
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

    start_point: LocationModel = Field(
        description="Начальная точка маршрута, представленная моделью LocationModel (адрес пользователя или предыдущее мероприятие)."
    )
    event_points: List[LocationModel] = Field(
        description="Список точек мероприятий (LocationModel), которые необходимо посетить последовательно."
    )
    transport_type: str = Field(
        default="driving",
        description="Предпочитаемый тип транспорта для построения маршрута: 'driving' (автомобиль), 'walking' (пешком), 'public_transport' (общественный транспорт).",
    )


# --- Схемы для объектов данных, используемых в AgentState ---


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
    event_type_key: str = Field(
        description="Внутренний ключ запрошенного типа мероприятия (например, 'Movie', 'Museum', 'StandUp')." 
    )
    actual_api_type: Optional[str] = Field(
        default=None, 
        description="Фактический тип события, полученный из поля 'Type' ответа API Афиши (например, 'Movie', 'Event', 'Admission')."
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
    price_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление диапазона цен на мероприятие (например, 'от 500 ₽').",
    )
    duration_description: Optional[str] = Field(
        default=None,
        description="Текстовое описание продолжительности мероприятия (например, '2 часа 30 минут').",
    )
    rating: Optional[Any] = Field(
        default=None,
        description="Рейтинг мероприятия, если доступен (может быть числом или строкой).",
    )
    age_restriction: Optional[str] = Field(
        default=None,
        description="Возрастное ограничение для мероприятия (например, '18+').",
    )
    genres: Optional[List[str]] = Field(
        default=None, 
        description="Список жанров мероприятия (из поля 'Genres' объекта Creation API Афиши)."
    )
    description: Optional[str] = Field(
        default=None, 
        description="Полное описание мероприятия (из поля 'Description' объекта Creation API Афиши)."
    )
    short_description: Optional[str] = Field(
        default=None, 
        description="Краткое описание мероприятия (из поля 'ShortDescription' объекта Creation API Афиши)."
    )


class RouteDetails(BaseModel):
    """
    Полные детали построенного маршрута, включая все сегменты.
    """

    status: str = Field(
        description="Общий статус построения всего маршрута ('success', 'partial_success', 'error', 'api_error', etc.). 'partial_success' если не все сегменты построены."
    )
    segments: Optional[List[RouteSegment]] = Field(
        default=None, description="Список сегментов маршрута."
    )
    total_duration_seconds: Optional[int] = Field(
        default=None,
        description="Суммарная длительность всех успешно построенных сегментов в секундах.",
    )
    total_duration_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление суммарной длительности (например, '~45 мин').",
    )
    total_distance_meters: Optional[int] = Field(
        default=None,
        description="Суммарное расстояние всех успешно построенных сегментов в метрах.",
    )
    total_distance_text: Optional[str] = Field(
        default=None,
        description="Текстовое представление суммарного расстояния (например, '~10 км').",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Общее сообщение об ошибке, если status не 'success' или 'partial_success'.",
    )


class ParsedDateTime(BaseModel):
    """Структурированный результат парсинга описания даты и времени из текста. Содержит компоненты даты и времени, флаг неоднозначности и возможные уточнения или ошибки."""

    year: Optional[int] = Field(default=None, description="Извлеченный год (число).")
    month: Optional[int] = Field(
        default=None, description="Извлеченный месяц (число от 1 до 12)."
    )
    day: Optional[int] = Field(
        default=None, description="Извлеченный день месяца (число)."
    )
    hour: Optional[int] = Field(
        default=None, description="Извлеченный час НАЧАЛА (число от 0 до 23)."
    )
    minute: Optional[int] = Field(
        default=None, description="Извлеченная минута НАЧАЛА (число от 0 до 59)."
    )

    end_hour: Optional[int] = Field(
        default=None,
        description="Извлеченный час КОНЦА временного диапазона (число от 0 до 23), если указан.",
    )  # НОВОЕ
    end_minute: Optional[int] = Field(
        default=None,
        description="Извлеченная минута КОНЦА временного диапазона (число от 0 до 59), если указана.",
    )  # НОВОЕ

    is_ambiguous: bool = Field(
        default=False,
        description="Флаг (True/False), указывающий, является ли распознанная дата/время неполной или неоднозначной и требует ли дополнительного уточнения от пользователя.",
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
        description="Классифицированный тип намерения пользователя: 'confirm_plan' (план подтвержден), 'request_change' (запрос на изменение плана), 'clarify_misunderstanding' (пользователь не понял или просит уточнения у агента), 'new_search' (пользователь хочет начать совершенно новый поиск)."
    )
    change_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Словарь с деталями запрошенного изменения, если intent_type='request_change'. Ключи и значения зависят от типа изменения. Например: {'target': 'budget', 'value': 2000} или {'target': 'event', 'event_index': 1, 'action': 'replace', 'new_interest': 'театр'} или {'target': 'date', 'new_value': '15 июля'}.",
    )


class CollectedUserData(BaseModel):
    city_name: Optional[str] = None
    city_id_afisha: Optional[int] = None
    interests_original: Optional[List[str]] = None
    interests_keys_afisha: Optional[List[str]] = None
    budget_original: Optional[int] = None
    budget_current_search: Optional[int] = None
    dates_description_original: Optional[str] = None
    raw_time_description_original: Optional[str] = None
    parsed_dates_iso: Optional[List[str]] = None
    parsed_end_dates_iso: Optional[List[str]] = None
    user_start_address_original: Optional[str] = None
    user_start_address_validated_coords: Optional[Dict[str, float]] = None
    partial_address_street: Optional[str] = Field(default=None, description="Частично распознанная улица, ожидается номер дома.")
    awaiting_address_input: bool = False
    awaiting_fallback_confirmation: bool = False
    pending_fallback_event: Optional[Dict[str, Any]] = None
    not_found_interest_keys: Optional[List[str]] = None
    fallback_candidates: Optional[Dict[str, Dict[str, Any]]] = None
    fallback_accepted_and_plan_updated: bool = False
    clarification_needed_fields: Optional[List[str]] = Field(default_factory=list)
    previous_confirmed_collected_data: Optional[Dict[str, Any]] = None
    previous_confirmed_events: Optional[List[Dict[str, Any]]] = None
    user_time_desc_for_fallback: Optional[str] = None
    
    class Config:
        extra = "allow"
