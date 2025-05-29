# Файл: tools/datetime_parser_tool.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, Tuple, Any

import dateparser  # Убедитесь, что dateparser установлен
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from llm_interface.gigachat_client import get_gigachat_client
from schemas.data_schemas import DateTimeParserToolArgs

logger = logging.getLogger(__name__)


class TimeRangeResult(BaseModel):
    """
    Извлекает время начала и, возможно, время окончания из текстового описания.
    Используется для определения конкретных часов и минут.
    """

    start_hour: Optional[int] = Field(
        None, description="Извлеченный час НАЧАЛА (0-23)."
    )
    start_minute: Optional[int] = Field(
        None,
        description="Извлеченные минуты НАЧАЛА (0-59). Если не указаны и start_hour есть, то 0.",
    )
    end_hour: Optional[int] = Field(
        None, description="Извлеченный час ОКОНЧАНИЯ (0-23), если указан диапазон."
    )
    end_minute: Optional[int] = Field(
        None,
        description="Извлеченные минуты ОКОНЧАНИЯ (0-59), если указан диапазон и end_hour есть, то 0.",
    )
    is_range_indicator: bool = Field(
        False,
        description="True, если был распознан явный диапазон 'с X до Y', 'X-Y' или указание 'до Y'.",
    )


async def _parse_time_with_llm_flexible(
    time_qualifier: str, base_datetime_for_time_prompt: datetime, llm_instance: Any
) -> TimeRangeResult:
    if not time_qualifier:
        return TimeRangeResult()
    logger.debug(f"_parse_time_with_llm_flexible: Парсинг времени='{time_qualifier}'")
    time_prompt_content = f"""
Извлеки час и минуту НАЧАЛА и, если указан диапазон (например, "с X до Y", "от X до Y", "X-Y часов"), то и час и минуту ОКОНЧАНИЯ из следующего описания времени: "{time_qualifier}".
Контекстная дата: {base_datetime_for_time_prompt.strftime('%Y-%m-%d')}.

Примеры интерпретации:
- "утром": start_hour=9, start_minute=0.
- "днем", "в обед": start_hour=13, start_minute=0.
- "вечером": start_hour=18, start_minute=0.
- "в X часов", "X часов": start_hour=X, start_minute=0.
- "X:YY": start_hour=X, start_minute=YY.
- "с X до Y", "от X до Y", "X-Y": start_hour=X (часть), start_minute=X (минутная часть, если есть, иначе 0), end_hour=Y (часть), end_minute=Y (минутная часть, если есть, иначе 0), is_range_indicator=true.
- "до Y часов", "к Y часам": end_hour=Y, end_minute=0 (или YY, если "до Y:YY"), is_range_indicator=true. start_hour/minute могут быть null.
- "после X часов", "от X часов": start_hour=X, start_minute=0 (или YY, если "после X:YY"). end_hour/minute могут быть null.

Верни результат, используя предоставленную структуру TimeRangeResult. Имя этой структуры TimeRangeResult.
Если для какого-то времени (начала или конца) указан час, а минуты нет, верни для него minute: 0.
Если не можешь извлечь валидный час для начала или конца, верни для соответствующего поля hour: null и minute: null.
Поле is_range_indicator должно быть true, только если в запросе явно указан диапазон "с X до Y", "X-Y" или указание "до Y".
"""
    try:
        structured_llm_time = llm_instance.with_structured_output(TimeRangeResult)
        parsed_response = await structured_llm_time.ainvoke(time_prompt_content)
        if not isinstance(parsed_response, TimeRangeResult):
            return TimeRangeResult()
        if (
            parsed_response.start_hour is not None
            and parsed_response.start_minute is None
        ):
            parsed_response.start_minute = 0
        if parsed_response.end_hour is not None and parsed_response.end_minute is None:
            parsed_response.end_minute = 0
        logger.debug(
            f"_parse_time_with_llm_flexible: LLM результат: {parsed_response.model_dump_json(indent=2)}"
        )
        return parsed_response
    except Exception as e:
        logger.error(
            f"_parse_time_with_llm_flexible: Ошибка парсинга времени '{time_qualifier}': {e}",
            exc_info=True,
        )
        return TimeRangeResult()


def _calculate_weekend_dates_simple(
    base_dt: datetime, next_weekend: bool = False
) -> Tuple[datetime, datetime]:
    days_to_monday = base_dt.weekday()
    if next_weekend:
        start_of_next_week_monday = (
            base_dt - timedelta(days=days_to_monday) + timedelta(days=7)
        )
        saturday = start_of_next_week_monday + timedelta(days=5)
    else:
        if base_dt.weekday() >= 5:
            saturday = base_dt - timedelta(days=(base_dt.weekday() - 5))
        else:
            saturday = base_dt + timedelta(days=(5 - base_dt.weekday()))
    sunday = saturday + timedelta(days=1)
    return (
        saturday.replace(hour=0, minute=0, second=0, microsecond=0),
        sunday.replace(hour=23, minute=59, second=59, microsecond=0),
    )


def _calculate_specific_weekday(
    base_dt: datetime, target_weekday_iso: int, is_next_week_explicit: bool = False
) -> datetime:
    current_weekday_iso = base_dt.weekday()
    days_difference = (target_weekday_iso - current_weekday_iso + 7) % 7
    result_dt = base_dt + timedelta(days=days_difference)
    if is_next_week_explicit:
        if result_dt.isocalendar()[1] == base_dt.isocalendar()[1]:
            if target_weekday_iso <= current_weekday_iso or (
                target_weekday_iso > current_weekday_iso and days_difference > 0
            ):
                result_dt += timedelta(days=7)
            elif (
                result_dt.date() == base_dt.date()
                and target_weekday_iso == current_weekday_iso
            ):
                result_dt += timedelta(days=7)
    elif result_dt.date() < base_dt.date() and days_difference == 0:
        result_dt += timedelta(days=7)
    return result_dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _apply_time_to_datetime(
    dt: datetime, hour: Optional[int], minute: Optional[int]
) -> Optional[datetime]:
    if hour is not None and minute is not None:
        try:
            return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            logger.warning(
                f"Не удалось применить время {hour}:{minute} к дате {dt.date()}"
            )
            return None
    return dt


def python_date_parser(
    natural_language_date: str, base_datetime: datetime
) -> Optional[Dict[str, Union[datetime, bool, str, None]]]:
    nl_date_lower = natural_language_date.lower().strip()
    parsed_info: Dict[str, Union[datetime, bool, str, None]] = {
        "start_datetime": None,
        "end_datetime": None,
        "is_range": False,
        "is_ambiguous": False,
        "source": "python_direct",
        "clarification_needed": None,
    }
    start_dt: Optional[datetime] = None
    if not nl_date_lower:
        return None

    if nl_date_lower == "сегодня" or nl_date_lower == "на сегодня":
        start_dt = base_datetime
    elif nl_date_lower == "завтра" or nl_date_lower == "на завтра":
        start_dt = base_datetime + timedelta(days=1)
    elif nl_date_lower == "послезавтра" or nl_date_lower == "на послезавтра":
        start_dt = base_datetime + timedelta(days=2)

    if start_dt:
        parsed_info["start_datetime"] = start_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return parsed_info

    if any(
        kw in nl_date_lower
        for kw in [
            "эти выходные",
            "на выходных",
            "текущие выходные",
            "ближайшие выходные",
        ]
    ):
        saturday, sunday = _calculate_weekend_dates_simple(
            base_datetime, next_weekend=False
        )
        (
            parsed_info["start_datetime"],
            parsed_info["end_datetime"],
            parsed_info["is_range"],
        ) = (saturday, sunday, True)
        return parsed_info
    if any(kw in nl_date_lower for kw in ["следующие выходные"]):
        saturday, sunday = _calculate_weekend_dates_simple(
            base_datetime, next_weekend=True
        )
        (
            parsed_info["start_datetime"],
            parsed_info["end_datetime"],
            parsed_info["is_range"],
        ) = (saturday, sunday, True)
        return parsed_info

    weekday_search_forms = {
        "понедельник": 0,
        "понедельника": 0,
        "вторник": 1,
        "вторника": 1,
        "среду": 2,
        "среды": 2,
        "четверг": 3,
        "четверга": 3,
        "пятницу": 4,
        "пятницы": 4,
        "субботу": 5,
        "субботы": 5,
        "воскресенье": 6,
        "воскресенья": 6,
    }
    is_next_week_explicit = "следующ" in nl_date_lower
    for form, day_iso in weekday_search_forms.items():
        if form in nl_date_lower:
            if ("выходные" in nl_date_lower or "уикенд" in nl_date_lower) and (
                form == "субботу" or form == "воскресенье"
            ):
                continue
            start_dt = _calculate_specific_weekday(
                base_datetime, day_iso, is_next_week_explicit
            )
            parsed_info["start_datetime"], parsed_info["source"] = (
                start_dt,
                "python_weekday_logic",
            )
            return parsed_info

    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base_datetime,
        "STRICT_PARSING": False,
    }
    try:
        dp_result = dateparser.parse(
            natural_language_date, languages=["ru"], settings=settings
        )
        if dp_result and all(
            getattr(dp_result, attr) is not None for attr in ["year", "month", "day"]
        ):
            parsed_info["start_datetime"], parsed_info["source"] = (
                dp_result.replace(hour=0, minute=0, second=0, microsecond=0),
                "dateparser",
            )
            return parsed_info
    except Exception:
        pass
    return None


@tool("datetime_parser_tool", args_schema=DateTimeParserToolArgs)
async def datetime_parser_tool(
    natural_language_date: str,
    natural_language_time_qualifier: Optional[str] = None,
    base_date_iso: Optional[str] = None,
) -> Dict[str, Optional[Union[str, bool, None]]]:
    logger.info(
        f"DateTimeParserTool: Вход: date='{natural_language_date}', time_qualifier='{natural_language_time_qualifier}', base_date_iso='{base_date_iso}'"
    )
    current_datetime_base = (
        datetime.fromisoformat(base_date_iso) if base_date_iso else datetime.now()
    )

    final_start_dt: Optional[datetime] = None
    final_end_dt: Optional[datetime] = None
    is_ambiguous = False
    clarification_needed: Optional[str] = None
    error_message: Optional[str] = None

    python_parsed_date_info = (
        python_date_parser(natural_language_date, current_datetime_base)
        if natural_language_date
        else None
    )

    if python_parsed_date_info and python_parsed_date_info.get("start_datetime"):
        final_start_dt = python_parsed_date_info["start_datetime"]
        final_end_dt = python_parsed_date_info.get("end_datetime")
        is_ambiguous = python_parsed_date_info.get(
            "is_ambiguous", False
        )  # Сохраняем неоднозначность от python_date_parser
        clarification_needed = python_parsed_date_info.get("clarification_needed")
        logger.info(
            f"DateTimeParserTool: Python-парсер ({python_parsed_date_info['source']}) нашел дату(ы). Start: {final_start_dt}, End: {final_end_dt}"
        )
    elif natural_language_date:
        is_ambiguous = True
        clarification_needed = f'Не удалось точно определить дату для "{natural_language_date}". Пожалуйста, укажите дату более подробно.'
        error_message = "Python-парсер не определил дату."

    if natural_language_time_qualifier and not error_message:
        base_for_llm_time_prompt = (
            final_start_dt
            if final_start_dt
            else current_datetime_base.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        if not final_start_dt:
            final_start_dt = base_for_llm_time_prompt

        if "весь день" in natural_language_time_qualifier.lower():
            logger.info("DateTimeParserTool: Обнаружен 'весь день'.")
            final_start_dt = (
                final_start_dt.replace(hour=0, minute=0) if final_start_dt else None
            )
            if (
                final_start_dt
            ):  # Если есть начальная дата, устанавливаем и конечную дату на конец этого дня
                final_end_dt = final_start_dt.replace(hour=23, minute=59, second=59)
        else:
            llm_instance = get_gigachat_client()
            time_range_res = await _parse_time_with_llm_flexible(
                natural_language_time_qualifier, base_for_llm_time_prompt, llm_instance
            )

            if time_range_res.start_hour is not None and final_start_dt is not None:
                applied_start_time = _apply_time_to_datetime(
                    final_start_dt,
                    time_range_res.start_hour,
                    time_range_res.start_minute,
                )
                if applied_start_time:
                    final_start_dt = applied_start_time
                else:
                    is_ambiguous = True
                    clarification_needed = (
                        (clarification_needed + " " if clarification_needed else "")
                        + f"Некорректное время начала '{natural_language_time_qualifier}'."
                    )

            if time_range_res.end_hour is not None and final_start_dt is not None:
                # Базовая дата для применения времени окончания
                # Если final_end_dt уже установлен python-парсером (например, для выходных - это будет дата воскресенья)
                # и LLM указал, что это диапазон, то применяем время к этой дате.
                # Иначе, применяем к дате final_start_dt.
                date_to_apply_end_time = final_start_dt.date()
                if (
                    final_end_dt
                    and time_range_res.is_range_indicator
                    and final_end_dt.date() >= final_start_dt.date()
                ):
                    date_to_apply_end_time = final_end_dt.date()

                base_dt_for_end_apply = datetime.combine(
                    date_to_apply_end_time, datetime.min.time()
                )
                applied_end_time = _apply_time_to_datetime(
                    base_dt_for_end_apply,
                    time_range_res.end_hour,
                    time_range_res.end_minute,
                )

                if applied_end_time:
                    final_end_dt = applied_end_time
                else:
                    is_ambiguous = True
                    clarification_needed = (
                        (clarification_needed + " " if clarification_needed else "")
                        + f"Некорректное время окончания '{natural_language_time_qualifier}'."
                    )

            if (
                not time_range_res.start_hour
                and not time_range_res.end_hour
                and natural_language_time_qualifier
            ):
                is_ambiguous = True
                clarification_needed = (
                    (clarification_needed + " " if clarification_needed else "")
                    + f"Не удалось точно определить время для '{natural_language_time_qualifier}'. Пожалуйста, уточните."
                )
                if not error_message:
                    error_message = "LLM не определил время."

    if (
        not final_start_dt
        and not natural_language_date
        and not natural_language_time_qualifier
    ):
        is_ambiguous = True
        clarification_needed = "Пожалуйста, укажите дату или время."
        error_message = "Пустой ввод."

    if clarification_needed and not is_ambiguous:
        is_ambiguous = True

    result_dict = {
        "datetime_iso": final_start_dt.isoformat() if final_start_dt else None,
        "end_datetime_iso": final_end_dt.isoformat() if final_end_dt else None,
        "is_ambiguous": is_ambiguous,
        "clarification_needed": (
            clarification_needed.strip() if clarification_needed else None
        ),
        "error_message": error_message,
    }
    logger.info(f"DateTimeParserTool: Возвращаю: {result_dict}")
    return result_dict
