import logging
from datetime import datetime, date
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
    natural_language_date_time: str, base_date_iso: Optional[str] = None
) -> Dict[str, Optional[Union[str, bool]]]:
    """
    Преобразует описание даты и/или времени на естественном языке в структурированный формат.
    Возвращает словарь с полями: 'datetime_iso' (ISO строка распознанной даты/времени, если успешно),
    'is_ambiguous' (bool, нужна ли дальнейшая кларификация), 'clarification_needed' (str, вопрос для кларификации),
    'error_message' (str, сообщение об ошибке).
    Предпочтительно возвращать 'datetime_iso' в формате YYYY-MM-DDTHH:MM:SS.
    Если распознано только дата, время можно установить на 00:00:00.
    Если распознано только время, дата должна быть текущей датой (или base_date).
    """
    logger.info(
        f"DateTimeParserTool: Parsing '{natural_language_date_time}' with base_date_iso '{base_date_iso}'"
    )

    llm = (
        get_gigachat_client()
    )  # Получаем клиент GigaChat (без специфичных инструментов)
    try:
        from langchain_core.utils.function_calling import convert_to_openai_function

        schema_for_llm = convert_to_openai_function(
            ParsedDateTime
        )  # Эта функция используется для генерации схемы
        logger.debug(
            f"DateTimeParserTool: Schema for ParsedDateTime to be used by LLM: {schema_for_llm}"
        )
        if not schema_for_llm.get("description"):
            logger.error(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            logger.error(
                "DateTimeParserTool: CRITICAL - Pydantic schema for ParsedDateTime HAS NO DESCRIPTION FIELD for LLM!"
            )
            logger.error(f"ParsedDateTime docstring: {ParsedDateTime.__doc__}")
            logger.error(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

    except Exception as e_debug:
        logger.error(f"DateTimeParserTool: Debug schema print error: {e_debug}")
    # --- КОНЕЦ ОТЛАДКИ ---

    # Определяем базовую дату для LLM
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

    system_prompt = f"""
Ты - эксперт по распознаванию дат и времени из текста. Твоя задача - извлечь дату и время из предоставленного текста и вернуть их в структурированном виде.
Текущая или базовая дата для относительных вычислений: {base_date_for_prompt}.

Входной текст: "{natural_language_date_time}"

Проанализируй входной текст.
Если текст содержит явное указание даты и времени (например, "15 июля в 7 вечера", "завтра в 10:00", "через 2 дня в полдень"):
- Верни год, месяц, день, час, минуту.
- Установи is_ambiguous = false.
Если текст содержит неполное или неоднозначное указание (например, "вечером", "на выходных", "в июле", "в субботу"):
- Попытайся предположить наиболее вероятную дату/время, исходя из базовой даты.
- Установи is_ambiguous = true.
- Сформируй уточняющий вопрос в поле clarification_needed, например: "Уточните, пожалуйста, дату для 'в июле'?" или "Вечером - это после 18:00?".
Если текст вообще не содержит информации о дате или времени, или ее невозможно однозначно интерпретировать:
- Установи is_ambiguous = true.
- Верни сообщение об ошибке в error_message.

Если удается распознать только дату, установи время на 00:00.
Если удается распознать только время, используй базовую дату (сегодняшнюю или переданную).
Если год не указан, используй текущий год или следующий, если указанная дата в текущем году уже прошла.
Всегда возвращай результат в формате JSON соответствующий схеме ParsedDateTime.
"""

    try:
        # Используем with_structured_output для получения Pydantic модели
        structured_llm = llm.with_structured_output(ParsedDateTime)
        parsed_result: ParsedDateTime = await structured_llm.ainvoke(system_prompt)

        logger.debug(
            f"DateTimeParserTool: LLM raw parsed result: {parsed_result.model_dump_json(indent=2)}"
        )

        if parsed_result.error_message:
            logger.warning(
                f"DateTimeParserTool: Failed to parse '{natural_language_date_time}'. LLM error: {parsed_result.error_message}"
            )
            return {
                "datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": parsed_result.clarification_needed,
                "error_message": parsed_result.error_message,
            }

        if parsed_result.is_ambiguous:
            logger.info(
                f"DateTimeParserTool: Parsed '{natural_language_date_time}' as ambiguous. Clarification: {parsed_result.clarification_needed}"
            )
            # Даже если неоднозначно, попытаемся собрать datetime, если есть компоненты
            # Это полезно, если агент сможет задать уточняющий вопрос и потом дополнить эту дату

        # Собираем datetime объект, если есть компоненты
        current_datetime = datetime.now()
        if base_date_iso:
            try:
                current_datetime = datetime.fromisoformat(base_date_iso)
            except ValueError:
                pass  # Используем datetime.now() если base_date_iso некорректный

        year = (
            parsed_result.year
            if parsed_result.year is not None
            else current_datetime.year
        )
        month = (
            parsed_result.month
            if parsed_result.month is not None
            else current_datetime.month
        )
        day = (
            parsed_result.day if parsed_result.day is not None else current_datetime.day
        )
        hour = (
            parsed_result.hour if parsed_result.hour is not None else 0
        )  # По умолчанию 00 часов, если не указано
        minute = (
            parsed_result.minute if parsed_result.minute is not None else 0
        )  # По умолчанию 00 минут

        # Простая логика для "будущего" если год не был извлечен, а дата уже прошла
        if (
            parsed_result.year is None
            and datetime(year, month, day, hour, minute)
            < current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            and (parsed_result.month is not None or parsed_result.day is not None)
        ):
            # Если месяц или день были указаны, но год нет, и дата в этом году уже прошла - берем следующий год
            if not (
                parsed_result.month is None and parsed_result.day is None
            ):  # Проверяем, что не просто время текущего дня
                logger.debug(
                    f"DateTimeParserTool: Date {year}-{month}-{day} has passed, assuming next year."
                )
                year += 1

        try:
            final_datetime = datetime(year, month, day, hour, minute)
            datetime_iso_str = final_datetime.isoformat()
            logger.info(
                f"DateTimeParserTool: Successfully parsed '{natural_language_date_time}' to '{datetime_iso_str}', ambiguous={parsed_result.is_ambiguous}"
            )
            return {
                "datetime_iso": datetime_iso_str,
                "is_ambiguous": parsed_result.is_ambiguous,
                "clarification_needed": parsed_result.clarification_needed,
                "error_message": None,
            }
        except ValueError as e:
            # Эта ошибка может возникнуть, если LLM вернула некорректные day/month/year (например, 30 февраля)
            logger.error(
                f"DateTimeParserTool: Error constructing datetime from LLM_result {parsed_result.model_dump()}: {e}"
            )
            return {
                "datetime_iso": None,
                "is_ambiguous": True,
                "clarification_needed": "Не удалось собрать корректную дату из распознанных компонентов.",
                "error_message": f"Ошибка сборки даты: {e}",
            }

    except Exception as e:
        logger.error(
            f"DateTimeParserTool: Unexpected error during LLM call or processing for '{natural_language_date_time}': {e}",
            exc_info=True,
        )
        return {
            "datetime_iso": None,
            "is_ambiguous": True,
            "clarification_needed": "Произошла внутренняя ошибка при распознавании даты.",
            "error_message": str(e),
        }
