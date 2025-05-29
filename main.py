import asyncio
import logging
import uuid
import os
import sys  # Добавлен импорт sys
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any, Optional

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler,
)

from config.settings import settings

# Предполагаем, что get_compiled_agent_graph находится в agent_graph.py
# Если он в main.py, то импорт не нужен или будет циклическим.
# Если create_agent_graph и compiled_agent_graph определяются прямо в main.py,
# то get_compiled_agent_graph() будет вызываться локально.
from agent_core.agent_graph import get_compiled_agent_graph
from agent_core.agent_state import (
    AgentState,
)  # Не используется напрямую в main, но нужно для get_compiled_agent_graph

# --- Начало конфигурации логирования ---
# 1. Получаем корневой логгер
# 2. Устанавливаем базовый уровень DEBUG, чтобы перехватывать все сообщения от наших модулей
# 3. Создаем обработчик для stdout
# 4. Устанавливаем форматтер
# 5. Добавляем обработчик к корневому логгеру (или к логгерам наших модулей)
# 6. Повышаем уровни для "шумных" сторонних библиотек

# Базовая конфигурация для вывода ВСЕХ логов DEBUG и выше в консоль
# Это может быть очень многословно из-за библиотек, поэтому ниже мы их приглушим.
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     stream=sys.stdout # Явное указание потока
# )

# Более контролируемая настройка:
logger_main = logging.getLogger("__main__")
logger_agent_core = logging.getLogger("agent_core")
logger_services = logging.getLogger("services")
logger_tools = logging.getLogger("tools")
logger_llm_interface = logging.getLogger("llm_interface")  # Если у вас есть такой

# Устанавливаем уровень DEBUG для наших модулей
logger_main.setLevel(logging.DEBUG)
logger_agent_core.setLevel(logging.DEBUG)
logger_services.setLevel(logging.DEBUG)
logger_tools.setLevel(logging.DEBUG)
logger_llm_interface.setLevel(logging.DEBUG)


# Настраиваем корневой логгер или создаем обработчик для наших логгеров
# Если мы хотим, чтобы ТОЛЬКО наши модули логировали на уровне DEBUG, а остальные были тише:
root_logger = logging.getLogger()
# Установим общий уровень для корневого логгера, например, INFO
# Это предотвратит вывод DEBUG-сообщений от библиотек, для которых мы не задали уровень явно.
# root_logger.setLevel(logging.INFO) # Раскомментируйте, если хотите меньше логов от неизвестных источников

# Удалим существующие обработчики с корневого логгера, чтобы избежать дублей,
# если какая-то библиотека уже настроила свой глобальный обработчик.
# Делайте это с осторожностью.
# for handler in root_logger.handlers[:]:
# root_logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Добавляем обработчик к нашим логгерам, а не к корневому,
# чтобы контролировать только их вывод.
# Или добавьте к корневому, если хотите единый обработчик для всего.
# Если добавлять к каждому:
# logger_main.addHandler(console_handler)
# logger_agent_core.addHandler(console_handler)
# logger_services.addHandler(console_handler)
# logger_tools.addHandler(console_handler)
# logger_llm_interface.addHandler(console_handler)
# logger_main.propagate = False # Чтобы сообщения не шли дальше к корневому, если у него есть свои хендлеры
# ... и так далее для других ваших логгеров

# Проще - настроить корневой и приглушить библиотеки
if not root_logger.hasHandlers():  # Добавляем наш формат и обработчик, если их еще нет
    root_logger.addHandler(console_handler)
    root_logger.setLevel(
        logging.DEBUG
    )  # Корневой ловит все, фильтруем на уровне библиотек

# Приглушение внешних библиотек
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(
    logging.INFO
)  # telegram.ext, telegram.bot и т.д.
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("gigachat").setLevel(
    logging.INFO
)  # Оставляем INFO для токенов и ключевых вызовов
logging.getLogger("dateparser").setLevel(
    logging.WARNING
)  # dateparser может быть шумным на INFO/DEBUG

# --- Конец конфигурации логирования ---

logger = logging.getLogger(
    __name__
)  # Переопределяем logger для __main__ после настройки

compiled_agent = get_compiled_agent_graph()
user_sessions: Dict[int, str] = {}


async def handle_telegram_message(
    chat_id: int, user_message_text: str
) -> Optional[str]:
    logger.info(f"Handling message from chat_id {chat_id}: '{user_message_text}'")
    if chat_id not in user_sessions:
        user_sessions[chat_id] = str(uuid.uuid4())
        logger.info(
            f"New LangGraph session for chat_id {chat_id}, thread_id: {user_sessions[chat_id]}"
        )
    thread_id = user_sessions[chat_id]
    logger.info(
        f"Using LangGraph session (thread_id) for chat_id {chat_id}: {thread_id}"
    )

    inputs = {"messages": [HumanMessage(content=user_message_text)]}
    config = {"configurable": {"thread_id": thread_id}}
    agent_response_text: Optional[str] = None

    try:
        logger.debug(
            f"Invoking agent for thread_id {thread_id} with input: '{user_message_text}'"
        )
        final_node_output_dict: Optional[Dict[str, Any]] = (
            None  # Будем хранить последний словарь состояния
        )

        async for output_chunk in compiled_agent.astream(
            inputs, config, stream_mode="values"
        ):
            # output_chunk здесь - это словарь состояния после выполнения каждого узла
            # Сохраняем самый последний (который будет перед END или при завершении потока)
            final_node_output_dict = output_chunk
            node_name_in_chunk = "UnknownNode"
            # Пытаемся получить имя узла из ключей словаря output_chunk, если это возможно
            # Это зависит от того, как LangGraph формирует `stream_mode="values"`
            # Обычно, если узел один, то ключ это имя узла. Если их несколько, то сложнее.
            # Для `stream_mode="values"` output_chunk это и есть `AgentState` после узла.
            # LangGraph не добавляет имя узла в сам AgentState при stream_mode="values".
            # Имя узла можно получить, если стримить в `stream_mode="updates"` или `stream_mode="debug"`.
            # Но для простоты оставим так, главное - это содержимое состояния.
            logger.debug(
                f"Intermediate stream output for thread_id {thread_id}: state_keys={list(final_node_output_dict.keys()) if final_node_output_dict else 'None'}"
            )

        if final_node_output_dict:
            agent_response_text = final_node_output_dict.get("status_message_to_user")
            if not agent_response_text:
                messages_history: List[BaseMessage] = final_node_output_dict.get(
                    "messages", []
                )
                if messages_history and isinstance(messages_history[-1], AIMessage):
                    agent_response_text = messages_history[-1].content
            if not agent_response_text:
                logger.warning(
                    f"No status_message_to_user and no last AIMessage for thread_id {thread_id}. State: {str(final_node_output_dict)[:300]}"
                )
                agent_response_text = (
                    "Кажется, я сейчас не знаю, что ответить. Попробуйте еще раз."
                )

            current_graph_state_after_stream = await compiled_agent.aget_state(config)
            if (
                current_graph_state_after_stream
                and not current_graph_state_after_stream.next
            ):
                logger.info(
                    f"LangGraph conversation for thread_id {thread_id} (chat_id {chat_id}) reached END state."
                )
                if agent_response_text and (
                    "Рад был помочь" in agent_response_text
                    or "План окончательный" in agent_response_text
                ):
                    if chat_id in user_sessions:
                        del user_sessions[chat_id]
                        logger.info(
                            f"Session for chat_id {chat_id} (thread_id {thread_id}) cleared after finalization."
                        )
        else:
            logger.error(
                f"No final output from agent for thread_id {thread_id} after astream."
            )
            agent_response_text = "Извините, что-то пошло не так при обработке вашего запроса (нет ответа от агента)."
    except GraphRecursionError as r_err:
        logger.error(
            f"GraphRecursionError for chat_id {chat_id} (thread_id {thread_id}): {r_err}",
            exc_info=True,
        )
        agent_response_text = "Кажется, я запутался в своих мыслях. Попробуйте, пожалуйста, начать заново или переформулировать запрос."
        if chat_id in user_sessions:
            del user_sessions[chat_id]
    except Exception as e:
        logger.error(
            f"Error processing message for chat_id {chat_id} (thread_id {thread_id}): {e}",
            exc_info=True,
        )
        agent_response_text = (
            "Извините, произошла внутренняя ошибка. Пожалуйста, попробуйте позже."
        )

    logger.info(
        f"Agent response for chat_id {chat_id}: '{str(agent_response_text)[:200]}'"
    )
    return agent_response_text


async def start_command_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    chat_id = update.message.chat_id
    logger.info(f"/start command received from chat_id {chat_id}")
    if chat_id in user_sessions:
        thread_id_to_clear = user_sessions[chat_id]
        del user_sessions[chat_id]
        logger.info(
            f"Session for chat_id {chat_id} (thread_id {thread_id_to_clear}) cleared due to /start command."
        )
    await update.message.reply_text(
        "Привет! Я ваш ассистент по планированию мероприятий. 😊\n\nЧтобы подобрать лучшие варианты, уточните детали:\n\n📍 Город (например: Москва, Санкт-Петербург)\n📅 Дата (сегодня, завтра, 15 июня или диапазон дат)\n⏰ Время (например: вечер, с 19:00, или интервал 14:00-17:00)\n🎭 Интересы (кино, концерт, театр, выставка, фестиваль, стендап, ресторан и т.д.)\n💰 Бюджет на человека/компанию (пример: до 2000 ₽, 5000-10000 ₽)\n\nПример запроса:\nИщу концерт в Москве 20 июля вечером, бюджет до 5000 ₽Готов предложить крутые варианты! 🎉"
    )


async def telegram_message_handler_wrapper(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.message and update.message.text:
        chat_id = update.message.chat_id
        user_text = update.message.text
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        bot_reply = await handle_telegram_message(chat_id, user_text)
        if bot_reply:
            await update.message.reply_text(bot_reply)
        else:
            logger.warning(
                f"Bot reply was empty for chat_id {chat_id} (message: '{user_text}'). Sending default error message."
            )
            await update.message.reply_text(
                "Извините, не удалось обработать ваш запрос. Пожалуйста, попробуйте еще раз."
            )


if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "TELEGRAM_BOT_TOKEN не найден в .env файле или переменных окружения!"
        )
        exit()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command_handler))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND, telegram_message_handler_wrapper
        )
    )
    logger.info("Telegram bot starting polling...")
    try:
        application.run_polling()
    except Exception as e:
        logger.critical(
            f"Telegram bot failed to start or polling error: {e}", exc_info=True
        )
    logger.info("Telegram bot stopped.")
