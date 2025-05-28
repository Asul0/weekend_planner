# Файл: tools/datetime_parser_tool.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, Tuple

import dateparser
from pydantic import BaseModel, Field  # <--- ДОБАВИЛИ Field
from langchain_core.tools import tool

from llm_interface.gigachat_client import get_gigachat_client
from schemas.data_schemas import ParsedDateTime, DateTimeParserToolArgs

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментируй для детальных логов этого модуля


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
        sunday.replace(hour=0, minute=0, second=0, microsecond=0),
    )


def _calculate_specific_weekday(
    base_dt: datetime, target_weekday_iso: int, is_next_week_explicit: bool = False
) -> datetime:
    current_weekday_iso = base_dt.weekday()
    days_difference = (target_weekday_iso - current_weekday_iso + 7) % 7
    result_dt = base_dt + timedelta(days=days_difference)

    if is_next_week_explicit:
        if result_dt.isocalendar()[1] == base_dt.isocalendar()[1]:
            result_dt += timedelta(days=7)
        elif (
            result_dt.date() == base_dt.date()
            and target_weekday_iso == current_weekday_iso
        ):
            result_dt += timedelta(days=7)
    elif result_dt.date() < base_dt.date():
        result_dt += timedelta(days=7)

    return result_dt.replace(hour=0, minute=0, second=0, microsecond=0)


async def _parse_time_with_llm(
    time_qualifier: str, base_datetime_for_time_prompt: datetime, llm_instance
) -> Tuple[Optional[int], Optional[int]]:
    if not time_qualifier:
        return None, None

    logger.debug(f"_parse_time_with_llm: Парсинг времени='{time_qualifier}'")

    class TimeResult(BaseModel):
        """Извлекает час и минуту из текстового описания времени."""

        hour: Optional[int] = Field(
            None, description="Извлеченный час в 24-часовом формате (0-23)."
        )
        minute: Optional[int] = Field(
            None,
            description="Извлеченные минуты (0-59). Если не указаны, должно быть 0, если час указан.",
        )

    time_prompt_content = f"""
Извлеки час и минуту из следующего описания времени: "{time_qualifier}".
Контекстная дата: {base_datetime_for_time_prompt.strftime('%Y-%m-%d')}.
"утром" = 9:00. "днем" или "в обед" = 13:00. "вечером" = 18:00.
"в X часов", "X часов" - извлеки X как hour, minute должно быть 0.
"X:YY" - извлеки X как hour и YY как minute.
"с X до Y" - извлеки X (час и минуту) как начальное время.
Верни результат, используя предоставленную структуру TimeResult.
Если час указан, а минуты нет, верни minute: 0.
Если не можешь извлечь валидный час, верни hour: null и minute: null.
"""
    try:
        structured_llm_time = llm_instance.with_structured_output(TimeResult)
        time_parsed_response = await structured_llm_time.ainvoke(time_prompt_content)

        if isinstance(time_parsed_response, TimeResult):
            time_parsed = time_parsed_response
            if time_parsed.hour is not None and time_parsed.minute is None:
                time_parsed.minute = 0
        else:
            logger.error(
                f"_parse_time_with_llm: Неожиданный тип ответа: {type(time_parsed_response)}"
            )
            return None, None

        logger.debug(
            f"_parse_time_with_llm: LLM результат для времени: {time_parsed.model_dump_json(indent=2)}"
        )
        if time_parsed.hour is not None and time_parsed.minute is not None:
            return time_parsed.hour, time_parsed.minute
        else:
            return None, None
    except Exception as e:
        logger.error(
            f"_parse_time_with_llm: Ошибка парсинга времени '{time_qualifier}': {e}",
            exc_info=True,
        )
        return None, None


def _apply_time_to_datetime(
    dt: datetime, hour: Optional[int], minute: Optional[int]
) -> datetime:
    if hour is not None and minute is not None:
        try:
            return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError as e:
            logger.warning(f"Не удалось применить время {hour}:{minute} к {dt}: {e}")
            return dt
    return dt


def python_date_parser(
    natural_language_date: str, base_datetime: datetime
) -> Optional[Dict[str, Union[datetime, bool, str, None]]]:
    nl_date_lower = natural_language_date.lower().strip()
    logger.debug(
        f"python_date_parser: Вход: nl_date_lower='{nl_date_lower}', base_datetime='{base_datetime.isoformat()}'"
    )

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
        logger.debug(
            f"python_date_parser (direct): Распознано '{nl_date_lower}' -> {parsed_info['start_datetime']}"
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
        parsed_info["start_datetime"] = saturday
        parsed_info["end_datetime"] = sunday.replace(hour=23, minute=59, second=59)
        parsed_info["is_range"] = True
        logger.debug(
            f"python_date_parser (direct): Распознаны 'эти выходные' -> СБ={saturday}, ВС={sunday}"
        )
        return parsed_info

    if any(kw in nl_date_lower for kw in ["следующие выходные"]):
        saturday, sunday = _calculate_weekend_dates_simple(
            base_datetime, next_weekend=True
        )
        parsed_info["start_datetime"] = saturday
        parsed_info["end_datetime"] = sunday.replace(hour=23, minute=59, second=59)
        parsed_info["is_range"] = True
        logger.debug(
            f"python_date_parser (direct): Распознаны 'следующие выходные' -> СБ={saturday}, ВС={sunday}"
        )
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
    matched_day_iso = None
    for form, day_iso in weekday_search_forms.items():
        if form in nl_date_lower:
            if ("выходные" in nl_date_lower or "уикенд" in nl_date_lower) and (
                form == "субботу" or form == "воскресенье"
            ):
                continue
            matched_day_iso = day_iso
            break
    if matched_day_iso is not None:
        start_dt = _calculate_specific_weekday(
            base_datetime, matched_day_iso, is_next_week_explicit
        )
        parsed_info["start_datetime"] = start_dt
        parsed_info["source"] = "python_weekday_logic"
        logger.debug(
            f"python_date_parser (weekday_logic): Распознано '{natural_language_date}' -> {start_dt}"
        )
        return parsed_info

    logger.debug(
        f"python_date_parser: Попытка парсинга с dateparser для: '{natural_language_date}'"
    )
    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base_datetime}
    try:
        dp_result = dateparser.parse(
            natural_language_date, languages=["ru"], settings=settings
        )
        if dp_result:
            parsed_info["start_datetime"] = dp_result
            parsed_info["source"] = "dateparser"
            logger.debug(
                f"python_date_parser (dateparser): Распознано '{natural_language_date}' -> {dp_result}"
            )
            if (
                dp_result.year is None
                or dp_result.month is None
                or dp_result.day is None
            ):
                logger.warning(
                    "python_date_parser: dateparser вернул None для Y/M/D. Не используем."
                )
                return None
            return parsed_info
        else:
            logger.debug(
                f"python_date_parser: dateparser не смог распознать '{natural_language_date}'."
            )
    except Exception as e:
        logger.error(
            f"python_date_parser: Ошибка dateparser для '{natural_language_date}': {e}",
            exc_info=True,
        )

    logger.debug(
        f"python_date_parser: Не удалось распознать '{natural_language_date}' Python-методами."
    )
    return None


@tool("datetime_parser_tool", args_schema=DateTimeParserToolArgs)
async def datetime_parser_tool(
    natural_language_date: str,
    natural_language_time_qualifier: Optional[str] = None,
    base_date_iso: Optional[str] = None,
) -> Dict[str, Optional[Union[str, bool, None]]]:
    logger.info(
        f"DateTimeParserTool: Вход: date='{natural_language_date}', "
        f"time_qualifier='{natural_language_time_qualifier}', base_date_iso='{base_date_iso}'"
    )
    current_datetime_base = datetime.now()
    if base_date_iso:
        try:
            current_datetime_base = datetime.fromisoformat(base_date_iso)
        except ValueError:
            logger.warning(
                f"Неверный base_date_iso '{base_date_iso}'. Используется now()."
            )

    python_parsed_date_info = None
    if natural_language_date:
        python_parsed_date_info = python_date_parser(
            natural_language_date, current_datetime_base
        )

    if python_parsed_date_info and python_parsed_date_info.get("start_datetime"):
        start_dt_py: datetime = python_parsed_date_info["start_datetime"]
        end_dt_py: Optional[datetime] = python_parsed_date_info.get("end_datetime")

        logger.info(
            f"DateTimeParserTool: Python-парсер ({python_parsed_date_info['source']}) нашел дату: {start_dt_py.date()}"
            + (f" - {end_dt_py.date()}" if end_dt_py else "")
        )

        parsed_hour, parsed_minute = None, None
        final_time_ambiguous = False
        final_time_clarification = None

        if natural_language_time_qualifier:
            if "весь день" in natural_language_time_qualifier.lower():
                logger.info(
                    "DateTimeParserTool: Обнаружен 'весь день', время остается 00:00 для начала."
                )
                start_dt_py = start_dt_py.replace(
                    hour=0, minute=0
                )  # Убедимся, что время 00:00
                if end_dt_py:  # Если это диапазон (выходные), конец остается 23:59
                    end_dt_py = end_dt_py.replace(hour=23, minute=59, second=59)
            else:
                llm_instance_for_time = get_gigachat_client()
                parsed_hour, parsed_minute = await _parse_time_with_llm(
                    natural_language_time_qualifier, start_dt_py, llm_instance_for_time
                )
                if parsed_hour is not None and parsed_minute is not None:
                    logger.info(
                        f"DateTimeParserTool: LLM распарсил время: {parsed_hour:02d}:{parsed_minute:02d} для даты от Python-парсера."
                    )
                    start_dt_py = _apply_time_to_datetime(
                        start_dt_py, parsed_hour, parsed_minute
                    )
                else:
                    logger.info(
                        f"DateTimeParserTool: LLM не смог распарсить время '{natural_language_time_qualifier}'."
                    )
                    final_time_ambiguous = True
                    final_time_clarification = f"Не удалось точно определить время для '{natural_language_time_qualifier}'. Пожалуйста, уточните время."
                    start_dt_py = start_dt_py.replace(hour=0, minute=0)
        else:
            logger.debug(
                "DateTimeParserTool: natural_language_time_qualifier не указан, время остается от Python-парсера (обычно 00:00)."
            )

        final_is_ambiguous = (
            python_parsed_date_info.get("is_ambiguous", False) or final_time_ambiguous
        )
        final_clarification = (
            python_parsed_date_info.get("clarification_needed")
            or final_time_clarification
        )

        return {
            "datetime_iso": (
                start_dt_py.isoformat() if not final_time_ambiguous else None
            ),
            "end_datetime_iso": (
                end_dt_py.isoformat()
                if end_dt_py and not final_time_ambiguous
                else None
            ),
            "is_ambiguous": final_is_ambiguous,
            "clarification_needed": final_clarification,
            "error_message": None,
        }

    if natural_language_date:
        logger.info(
            f"DateTimeParserTool: Python-парсер не смог определить дату для '{natural_language_date}'. Запрос на уточнение."
        )
        return {
            "datetime_iso": None,
            "end_datetime_iso": None,
            "is_ambiguous": True,
            "clarification_needed": f"Не удалось точно определить дату для \"{natural_language_date}\". Пожалуйста, укажите дату более подробно (например, '15 июня', 'следующий вторник').",
            "error_message": "Python-парсер не определил дату.",
        }
    elif natural_language_time_qualifier:
        logger.info(
            f"DateTimeParserTool: Дата не указана, парсим только время '{natural_language_time_qualifier}' с LLM и привязываем к сегодня."
        )
        llm_instance = get_gigachat_client()
        parsed_hour, parsed_minute = await _parse_time_with_llm(
            natural_language_time_qualifier, current_datetime_base, llm_instance
        )
        if parsed_hour is not None and parsed_minute is not None:
            today_with_time = _apply_time_to_datetime(
                current_datetime_base, parsed_hour, parsed_minute
            )
            logger.info(
                f"DateTimeParserTool: Распознано только время для сегодня: {today_with_time.isoformat()}"
            )
            return {
                "datetime_iso": today_with_time.isoformat(),
                "end_datetime_iso": None,
                "is_ambiguous": False,
                "clarification_needed": None,
                "error_message": None,
            }
        else:
            logger.info(
                f"DateTimeParserTool: Не удалось распознать только время '{natural_language_time_qualifier}'."
            )
            return {
                "datetime_iso": None,
                "end_datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": f'Не удалось точно определить время для "{natural_language_time_qualifier}". Пожалуйста, уточните.',
                "error_message": "LLM не определил время.",
            }
    else:
        logger.warning("DateTimeParserTool: Пустой ввод для даты и времени.")
        return {
            "datetime_iso": None,
            "end_datetime_iso": None,
            "is_ambiguous": True,
            "clarification_needed": "Пожалуйста, укажите дату или время.",
            "error_message": "Пустой ввод.",
        }
