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
) -> Dict[
    str, Optional[Union[str, bool, None]]
]:  # Добавил None в Union для end_datetime_iso
    """
    Преобразует описание даты и/или времени на естественном языке в структурированный формат.
    Учитывает как основное описание даты/периода, так и возможное отдельное уточнение по времени.
    Возвращает словарь с полями:
    'datetime_iso' (ISO строка распознанной даты/времени начала, если успешно),
    'end_datetime_iso' (ISO строка распознанной даты/времени конца диапазона, если применимо),
    'is_ambiguous' (bool, нужна ли дальнейшая кларификация),
    'clarification_needed' (str, вопрос для кларификации),
    'error_message' (str, сообщение об ошибке).
    """
    logger.info(
        f"DateTimeParserTool: Parsing date='{natural_language_date}', time_qualifier='{natural_language_time_qualifier}' with base_date_iso='{base_date_iso}'"
    )

    llm = get_gigachat_client()
    try:  # Отладочный блок для схемы
        from langchain_core.utils.function_calling import convert_to_openai_function

        schema_for_llm = convert_to_openai_function(
            ParsedDateTime
        )  # Используем обновленную ParsedDateTime
        logger.debug(
            f"DateTimeParserTool: Schema for ParsedDateTime to be used by LLM: {schema_for_llm}"
        )
        if not ParsedDateTime.__doc__ and not schema_for_llm.get(
            "description"
        ):  # Проверка на описание класса
            logger.error(
                "DateTimeParserTool: CRITICAL - Pydantic schema for ParsedDateTime HAS NO DESCRIPTION (docstring) for LLM!"
            )
    except Exception as e_debug:
        logger.error(f"DateTimeParserTool: Debug schema print error: {e_debug}")

    if base_date_iso:
        try:
            base_date_for_prompt = datetime.fromisoformat(base_date_iso).strftime(
                "%d %B %Y года (%A)"
            )
        except ValueError:
            logger.warning(
                f"DateTimeParserTool: Invalid base_date_iso format '{base_date_iso}'. Using current date."
            )
            base_date_for_prompt = date.today().strftime("%d %B %Y года (%A)")
    else:
        base_date_for_prompt = date.today().strftime("%d %B %Y года (%A)")

    full_description_for_llm = natural_language_date
    if natural_language_time_qualifier:
        full_description_for_llm += f" ({natural_language_time_qualifier})"

    logger.debug(
        f"DateTimeParserTool: Combined description for LLM: '{full_description_for_llm}'"
    )

    system_prompt = f"""
Ты - эксперт по распознаванию дат и времени из текста. Твоя задача - извлечь дату и время из предоставленного текста и вернуть их в структурированном виде, используя JSON, соответствующий схеме ParsedDateTime.
Текущая или базовая дата для относительных вычислений: {base_date_for_prompt}.

Входной текст: "{full_description_for_llm}"

Проанализируй входной текст. Обрати внимание на все части, описывающие дату и время.
Приоритет отдавай явным указаниям времени во входном тексте над общими правилами (например, если сказано "утром с 10 до 11", то час начала 10, а не 9).

1.  **Если текст содержит явное указание ДАТЫ (например, "15 июля", "завтра") И ЯВНОЕ УКАЗАНИЕ ВРЕМЕНИ НАЧАЛА (например, "в 7 вечера", "в 10:00", "после 18:00", "утром", "днем", "вечером"):**
    - Извлеки год, месяц, день.
    - Для времени НАЧАЛА:
        - Если указано конкретное время (например, "в 10:00", "7 вечера" (это 19:00)), извлеки час и минуту.
        - "после X часов" (например, "после 18:00") означает час=X, минута=0.
        - "утром" означает час=9, минута=0 (если нет другого уточнения типа "ранним утром").
        - "днем" означает час=13, минута=0.
        - "вечером" означает час=17, минута=0.
    - Поля end_hour, end_minute оставь null, если не указан явный диапазон.
    - Установи is_ambiguous = false.

2.  **Если текст содержит явное указание ДАТЫ И ЯВНЫЙ ВРЕМЕННОЙ ДИАПАЗОН (например, "завтра с 17:00 по 21:00", "5 июля от 10 утра до 2 часов дня"):**
    - Извлеки год, месяц, день из части с датой.
    - Извлеки час и минуту НАЧАЛА диапазона в поля 'hour' и 'minute'.
    - Извлеки час и минуту КОНЦА диапазона в поля 'end_hour' и 'end_minute'.
    - Установи is_ambiguous = false.

3.  **Если текст содержит только указание ДАТЫ БЕЗ ЯВНОГО ВРЕМЕНИ И БЕЗ ВРЕМЕННОГО КВАЛИФИКАТОРА** (например, "завтра", "15 июля", "на следующих выходных"):
    - Извлеки год, месяц, день. Установи час=0, минута=0. Поля end_hour, end_minute оставь null.
    - Установи is_ambiguous = false (это означает "весь день", начиная с 00:00).

4.  **Если текст содержит неполное или неоднозначное указание даты/времени, которое ты не можешь однозначно разрешить с помощью правил выше** (например, "в июле" без года, "в субботу" без уточнения недели, или просто "пораньше" без контекста):
    - Попытайся предположить наиболее вероятную дату/время.
    - Установи is_ambiguous = true.
    - Сформируй четкий уточняющий вопрос в поле clarification_needed.

5.  **Если текст вообще не содержит информации о дате или времени, или ее невозможно интерпретировать:**
    - Установи is_ambiguous = true, error_message="Не удалось распознать дату или время из текста." и clarification_needed="Пожалуйста, уточните дату и время."

Общие правила:
- Если год не указан, используй год из базовой даты ({base_date_for_prompt}) или следующий год, если указанная дата в текущем году уже прошла (применяется к дате начала).
- Если распознано только время (начала и, возможно, конца) без явной даты, используй дату из базовой даты ({base_date_for_prompt}).
- Всегда возвращай результат в формате JSON, соответствующий схеме ParsedDateTime. Поля, которые не удалось извлечь, должны быть null.
"""

    try:
        structured_llm = llm.with_structured_output(
            ParsedDateTime
        )  # Используем обновленную ParsedDateTime
        parsed_result: ParsedDateTime = await structured_llm.ainvoke(system_prompt)
        logger.debug(
            f"DateTimeParserTool: LLM raw parsed result: {parsed_result.model_dump_json(indent=2)}"
        )

        if parsed_result.error_message:
            logger.warning(
                f"DateTimeParserTool: Failed to parse '{full_description_for_llm}'. LLM error: {parsed_result.error_message}"
            )
            return {
                "datetime_iso": None,
                "end_datetime_iso": None,
                "is_ambiguous": (
                    parsed_result.is_ambiguous
                    if parsed_result.is_ambiguous is not None
                    else True
                ),
                "clarification_needed": parsed_result.clarification_needed,
                "error_message": parsed_result.error_message,
            }

        if parsed_result.is_ambiguous:
            logger.info(
                f"DateTimeParserTool: Parsed '{full_description_for_llm}' as ambiguous. Clarification: {parsed_result.clarification_needed}"
            )

        current_datetime_base = datetime.now()
        if base_date_iso:
            try:
                current_datetime_base = datetime.fromisoformat(base_date_iso)
            except ValueError:
                pass

        year = (
            parsed_result.year
            if parsed_result.year is not None
            else current_datetime_base.year
        )
        month = (
            parsed_result.month
            if parsed_result.month is not None
            else current_datetime_base.month
        )
        day = (
            parsed_result.day
            if parsed_result.day is not None
            else current_datetime_base.day
        )
        hour = parsed_result.hour if parsed_result.hour is not None else 0
        minute = parsed_result.minute if parsed_result.minute is not None else 0

        is_only_time_extracted = (
            parsed_result.year is None
            and parsed_result.month is None
            and parsed_result.day is None
            and (parsed_result.hour is not None or parsed_result.minute is not None)
        )

        if not is_only_time_extracted and parsed_result.year is None:
            temp_date_current_year = datetime(
                year, month, day, hour, minute
            )  # Используем уже определенные hour, minute
            comparison_base_date_start_of_day = current_datetime_base.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if temp_date_current_year < comparison_base_date_start_of_day and (
                parsed_result.month is not None or parsed_result.day is not None
            ):
                logger.debug(
                    f"DateTimeParserTool: Start date {year}-{month}-{day} has passed, assuming next year."
                )
                year += 1

        datetime_iso_str = None
        end_datetime_iso_str = None

        try:
            final_start_datetime = datetime(year, month, day, hour, minute)
            datetime_iso_str = final_start_datetime.isoformat()

            if (
                parsed_result.end_hour is not None
                and parsed_result.end_minute is not None
            ):
                # Для конечной даты используем ту же дату (year, month, day), что и для начальной,
                # если LLM не предоставил другую (наш промпт и не просит этого пока).
                final_end_datetime = datetime(
                    year, month, day, parsed_result.end_hour, parsed_result.end_minute
                )
                if (
                    final_end_datetime < final_start_datetime
                ):  # Если время конца раньше времени начала (н-р, с 22:00 до 02:00)
                    final_end_datetime += timedelta(days=1)
                    logger.debug(
                        f"DateTimeParserTool: End time '{final_end_datetime.isoformat()}' is on the next day based on start time."
                    )
                end_datetime_iso_str = final_end_datetime.isoformat()

            logger.info(
                f"DateTimeParserTool: Successfully parsed '{full_description_for_llm}' to start='{datetime_iso_str}'"
                + (f", end='{end_datetime_iso_str}'" if end_datetime_iso_str else "")
                + f", ambiguous={parsed_result.is_ambiguous}"
            )
            return {
                "datetime_iso": datetime_iso_str,
                "end_datetime_iso": end_datetime_iso_str,
                "is_ambiguous": parsed_result.is_ambiguous,
                "clarification_needed": parsed_result.clarification_needed,
                "error_message": None,
            }
        except ValueError as e:
            logger.error(
                f"DateTimeParserTool: Error constructing datetime from LLM_result {parsed_result.model_dump()}: {e}"
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
