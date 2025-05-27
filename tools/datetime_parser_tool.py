import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Union
from schemas.data_schemas import ParsedDateTime, DateTimeParserToolArgs
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm_interface.gigachat_client import get_gigachat_client
from schemas.data_schemas import (
    DateTimeParserToolArgs,
)  # Импортируем нашу схему аргументов

logger = logging.getLogger(__name__)


# Pydantic модель для структурированного ответа от LLM


@tool("datetime_parser_tool", args_schema=DateTimeParserToolArgs)
async def datetime_parser_tool(
    natural_language_date: str,
    natural_language_time_qualifier: Optional[str] = None,
    base_date_iso: Optional[str] = None,
) -> Dict[str, Optional[Union[str, bool, None]]]:
    logger.info(
        f"DateTimeParserTool: Parsing date='{natural_language_date}', time_qualifier='{natural_language_time_qualifier}' with base_date_iso='{base_date_iso}'"
    )

    llm = get_gigachat_client()

    current_datetime_base = datetime.now()
    if base_date_iso:
        try:
            current_datetime_base = datetime.fromisoformat(base_date_iso)
        except ValueError:
            logger.warning(
                f"DateTimeParserTool: Invalid base_date_iso format '{base_date_iso}'. Using current datetime."
            )

    base_year = current_datetime_base.year
    base_month = current_datetime_base.month
    base_day = current_datetime_base.day
    base_weekday_iso = current_datetime_base.weekday()  # Понедельник 0, Воскресенье 6

    base_date_for_prompt_str = current_datetime_base.strftime("%Y-%m-%d (%A, %d %B %Y)")

    full_description_for_llm = natural_language_date
    if natural_language_time_qualifier:
        full_description_for_llm += f" ({natural_language_time_qualifier})"

    logger.debug(
        f"DateTimeParserTool: Combined description for LLM: '{full_description_for_llm}'"
    )

    system_prompt = f"""
Ты - эксперт по точному распознаванию и вычислению дат и времени из текста. Твоя задача - извлечь или ВЫЧИСЛИТЬ дату и время из предоставленного текста и вернуть их в структурированном виде, используя JSON, соответствующий схеме ParsedDateTime.
Обязательно вычисляй относительные даты на основе предоставленной базовой даты.

Базовая дата для вычислений: {base_year}-{base_month:02d}-{base_day:02d} (год-месяц-день).
День недели базовой даты (ISO, где понедельник=0, воскресенье=6): {base_weekday_iso}.

Входной текст: "{full_description_for_llm}"

Проанализируй входной текст. Если есть относительные указания даты, ВЫЧИСЛИ их:
- "сегодня": Используй {base_year}, {base_month}, {base_day}.
- "завтра": Это ({base_year}-{base_month:02d}-{base_day:02d} + 1 день). Вычисли и верни точные year, month, day.
- "послезавтра": Это ({base_year}-{base_month:02d}-{base_day:02d} + 2 дня). Вычисли и верни точные year, month, day.
- "на этих выходных" или "эти выходные":
    - Определи ближайшую субботу.
    - Если базовая дата - суббота ({base_weekday_iso} == 5), то суббота это {base_year}-{base_month:02d}-{base_day:02d}.
    - Если базовая дата - воскресенье ({base_weekday_iso} == 6), то "эти выходные" это {base_year}-{base_month:02d}-{base_day:02d} (т.е. сама базовая дата, если запрос о воскресенье). Если запрос о субботе, то это прошедшая суббота (base_date - 1 день).
    - Если базовая дата - будний день (Пн-Пт, {base_weekday_iso} < 5): ближайшая суббота это ({base_year}-{base_month:02d}-{base_day:02d} + {5 - base_weekday_iso} дней).
    - Верни year, month, day для этой вычисленной субботы.
    - Если текст явно указывает на весь диапазон выходных (например, "на все выходные", "сб и вс"), то дополнительно вычисли end_year, end_month, end_day для воскресенья (вычисленная суббота + 1 день) и установи end_hour=23, end_minute=59. В остальных случаях, если не указан явный диапазон, касающийся воскресенья, поля end_* оставь null.
- "в следующие выходные":
    - Ближайший понедельник после базовой даты это ({base_year}-{base_month:02d}-{base_day:02d} + {(7 - base_weekday_iso) % 7} дней).
    - Суббота следующей недели это (этот понедельник + 5 дней). Вычисли и верни year, month, day для этой субботы.
    - Если нужен весь диапазон следующих выходных, то end_year, end_month, end_day будут для воскресенья (суббота следующей недели + 1 день), end_hour=23, end_minute=59.

Если ты успешно ВЫЧИСЛИЛ или извлек конкретную дату (year, month, day), установи is_ambiguous = false.
Если ты НЕ СМОГ ВЫЧИСЛИТЬ или извлечь однозначные year, month, day (например, текст "в июле" без года, или "в субботу" без контекста, или если ты сомневаешься в вычислении "выходных" из-за неясности запроса), ТОЛЬКО ТОГДА установи is_ambiguous = true, clarification_needed с твоим вопросом, и ОБЯЗАТЕЛЬНО верни year, month, day как null.

Для времени:
- Если указано конкретное время (например, "в 10:00", "7 вечера" (это 19:00)), извлеки час и минуту.
- "после X часов" (например, "после 18:00") означает час=X, минута=0.
- "утром" означает час=9, минута=0. "днем" означает час=13, минута=0. "вечером" означает час=17, минута=0.
- Если есть явный диапазон (например, "с 17:00 по 21:00"), извлеки hour/minute начала и end_hour/end_minute конца.
- Если время не указано, час=0, минута=0.

Правила для явных дат (типа "15 июля"):
- Если год не указан, используй год из базовой даты ({base_year}). Если дата (день.месяц) в текущем году ({base_year}) уже прошла, используй следующий год ({base_year + 1}).
- Если распознано только время без явной даты, используй год, месяц, день из базовой даты.

Всегда возвращай результат в формате JSON, соответствующий схеме ParsedDateTime. Поля, которые не удалось извлечь, должны быть null.
"""

    try:
        structured_llm = llm.with_structured_output(ParsedDateTime)
        parsed_result: ParsedDateTime = await structured_llm.ainvoke(system_prompt)
        logger.debug(
            f"DateTimeParserTool: LLM raw parsed result: {parsed_result.model_dump_json(indent=2)}"
        )

        if parsed_result.error_message:
            logger.warning(
                f"DateTimeParserTool: LLM error for '{full_description_for_llm}': {parsed_result.error_message}"
            )
            return {
                "datetime_iso": None,
                "end_datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": parsed_result.clarification_needed,
                "error_message": parsed_result.error_message,
            }

        if not (parsed_result.year and parsed_result.month and parsed_result.day):
            is_only_time_case = (
                parsed_result.hour is not None and not parsed_result.is_ambiguous
            )
            if not is_only_time_case:
                logger.info(
                    f"DateTimeParserTool: Essential date components (year/month/day) are missing from LLM. Input: '{full_description_for_llm}'. LLM: {parsed_result.model_dump_json(indent=1)}"
                )
                clarification = (
                    parsed_result.clarification_needed
                    or "Не удалось точно определить дату. Пожалуйста, уточните."
                )
                if (
                    parsed_result.is_ambiguous
                    and not parsed_result.clarification_needed
                ):  # Если LLM сам не задал вопрос при неоднозначности
                    clarification = f"Не удалось однозначно определить дату для '{full_description_for_llm}'. Можете уточнить?"
                return {
                    "datetime_iso": None,
                    "end_datetime_iso": None,
                    "is_ambiguous": True,
                    "clarification_needed": clarification,
                    "error_message": "LLM не предоставил компоненты даты.",
                }

        year_to_use = (
            parsed_result.year
            if parsed_result.year is not None
            else current_datetime_base.year
        )
        month_to_use = (
            parsed_result.month
            if parsed_result.month is not None
            else current_datetime_base.month
        )
        day_to_use = (
            parsed_result.day
            if parsed_result.day is not None
            else current_datetime_base.day
        )

        if not (
            year_to_use and month_to_use and day_to_use
        ):  # Дополнительная проверка после подстановки из base_date
            logger.error(
                f"DateTimeParserTool: Date components still None after defaults. Y={year_to_use} M={month_to_use} D={day_to_use}. LLM: {parsed_result.model_dump()}"
            )
            return {
                "datetime_iso": None,
                "end_datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": "Ошибка при сборке даты.",
                "error_message": "Внутренняя ошибка сборки даты.",
            }

        hour_to_use = parsed_result.hour if parsed_result.hour is not None else 0
        minute_to_use = parsed_result.minute if parsed_result.minute is not None else 0

        if parsed_result.year is None and not (
            natural_language_date.lower()
            in ["сегодня", "завтра", "послезавтра", "вчера", "позавчера"]
            or "выходн" in natural_language_date.lower()
        ):
            try:
                temp_date_current_year = datetime(
                    current_datetime_base.year, month_to_use, day_to_use
                )
                comparison_base_date_start_of_day = current_datetime_base.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                if temp_date_current_year < comparison_base_date_start_of_day:
                    year_to_use = current_datetime_base.year + 1
            except (ValueError, TypeError) as e_year_check:
                logger.warning(
                    f"DateTimeParserTool: Could not perform 'date passed' check. Y={year_to_use} M={month_to_use} D={day_to_use}. Error: {e_year_check}"
                )
                return {
                    "datetime_iso": None,
                    "end_datetime_iso": None,
                    "is_ambiguous": True,
                    "clarification_needed": "Не удалось точно определить год для указанной даты.",
                    "error_message": "Ошибка при проверке года.",
                }

        datetime_iso_str = None
        end_datetime_iso_str = None

        try:
            final_start_datetime = datetime(
                year_to_use, month_to_use, day_to_use, hour_to_use, minute_to_use
            )
            datetime_iso_str = final_start_datetime.isoformat()

            if (
                parsed_result.end_hour is not None
                and parsed_result.end_minute is not None
            ):
                end_year_to_use = (
                    parsed_result.end_year
                    if parsed_result.end_year is not None
                    else year_to_use
                )
                end_month_to_use = (
                    parsed_result.end_month
                    if parsed_result.end_month is not None
                    else month_to_use
                )
                end_day_to_use = (
                    parsed_result.end_day
                    if parsed_result.end_day is not None
                    else day_to_use
                )

                final_end_datetime = datetime(
                    end_year_to_use,
                    end_month_to_use,
                    end_day_to_use,
                    parsed_result.end_hour,
                    parsed_result.end_minute,
                )
                if final_end_datetime < final_start_datetime and not (
                    parsed_result.end_year
                    and parsed_result.end_month
                    and parsed_result.end_day
                ):
                    final_end_datetime += timedelta(days=1)
                end_datetime_iso_str = final_end_datetime.isoformat()

            final_is_ambiguous = parsed_result.is_ambiguous
            final_clarification_needed = parsed_result.clarification_needed
            if (
                final_is_ambiguous and not final_clarification_needed
            ):  # Если LLM сказал ambiguos, но не дал вопрос
                final_clarification_needed = f"Не удалось однозначно определить дату для '{full_description_for_llm}'. Можете уточнить?"

            logger.info(
                f"DateTimeParserTool: Successfully parsed '{full_description_for_llm}' to start='{datetime_iso_str}'"
                + (f", end='{end_datetime_iso_str}'" if end_datetime_iso_str else "")
                + f", ambiguous={final_is_ambiguous}"
            )
            return {
                "datetime_iso": datetime_iso_str,
                "end_datetime_iso": end_datetime_iso_str,
                "is_ambiguous": final_is_ambiguous,
                "clarification_needed": final_clarification_needed,
                "error_message": None,
            }
        except (ValueError, TypeError) as e:
            logger.error(
                f"DateTimeParserTool: Error constructing datetime from components: Y={year_to_use} M={month_to_use} D={day_to_use} h={hour_to_use} m={minute_to_use}. LLM: {parsed_result.model_dump()}. Error: {e}"
            )
            return {
                "datetime_iso": None,
                "end_datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": parsed_result.clarification_needed
                or "Не удалось собрать корректную дату из распознанных компонентов.",
                "error_message": f"Ошибка сборки даты: {e}",
            }
    except Exception as e:
        logger.error(
            f"DateTimeParserTool: Unexpected error for '{full_description_for_llm}': {e}",
            exc_info=True,
        )
        return {
            "datetime_iso": None,
            "end_datetime_iso": None,
            "is_ambiguous": True,
            "clarification_needed": "Произошла внутренняя ошибка при распознавании даты.",
            "error_message": str(e),
        }
