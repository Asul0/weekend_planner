import asyncio
import logging
import uuid  # Для генерации thread_id
import os  # Для получения токена
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any, Optional

# Импорты для Telegram бота
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler,
)

# Наши модули
from config.settings import settings  # settings уже загружает .env
from agent_core.agent_graph import get_compiled_agent_graph
from agent_core.agent_state import AgentState

# Настройка логирования
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
        # logging.FileHandler("agent.log") # Можно раскомментировать для записи в файл
    ],
)
logger = logging.getLogger(__name__)

# Получаем скомпилированный граф агента
compiled_agent = get_compiled_agent_graph()

# Словарь для хранения thread_id для каждого пользователя Telegram (chat_id)
# ВАЖНО: Для продакшена это должно быть персистентное хранилище (БД, Redis и т.д.)
user_sessions: Dict[int, str] = {}


async def handle_telegram_message(chat_id: int, user_message_text: str) -> Optional[str]:
    logger.info(f"Handling message from chat_id {chat_id}: '{user_message_text}'")

    if chat_id not in user_sessions:
        user_sessions[chat_id] = str(uuid.uuid4())
        logger.info(f"New session for chat_id {chat_id}, thread_id: {user_sessions[chat_id]}")
    thread_id = user_sessions[chat_id]

    inputs = {"messages": [HumanMessage(content=user_message_text)]}
    config = {"configurable": {"thread_id": thread_id}}
    agent_response_text: Optional[str] = None
    
    try:
        logger.debug(f"Invoking agent for thread_id {thread_id} with input: '{user_message_text}'")
        
        # Выполняем граф до тех пор, пока не получим сообщение для пользователя
        # или пока граф не завершится (END).
        # Мы будем использовать ainvoke, чтобы получить финальное состояние после одного "полного" ответа агента.
        # Или, если агент должен задавать серию вопросов, нам нужно стримить и проверять status_message_to_user на каждом шаге.

        # Используем stream, но берем только последнее состояние, где есть ответ пользователю.
        last_state_with_user_message: Optional[AgentState] = None

        async for output_chunk in compiled_agent.astream(inputs, config, stream_mode="values"):
            current_state_snapshot: AgentState = output_chunk # type: ignore
            logger.debug(f"Intermediate state for thread_id {thread_id}: node={current_state_snapshot.get('__node__', 'N/A')}, status_msg={current_state_snapshot.get('status_message_to_user')}") # __node__ добавляется Langgraph
            
            # Если узел сгенерировал сообщение для пользователя, это то, что мы должны отправить
            if current_state_snapshot.get("status_message_to_user"):
                last_state_with_user_message = current_state_snapshot
                # НЕ выходим из цикла сразу, даем графу дойти до естественной точки остановки
                # (например, до узла, который ждет следующий HumanMessage, или до END)
                # Однако, если граф зациклится без отправки status_message_to_user, это проблема.

            # Проверка, не завершился ли граф
            # state.next будет None или пустой кортеж, если это END узел Pregeла
            # Мы можем получить актуальное состояние и проверить .next
            # current_pregel_state = await compiled_agent.aget_state(config)
            # if not current_pregel_state.next: # Достигли END
            #     break
        
        # После того как astream завершился (т.е. граф дошел до точки, где он ждет или завершился)
        # Берем последнее состояние, в котором было сообщение для пользователя
        final_state_to_process: Optional[AgentState] = None
        if last_state_with_user_message:
            final_state_to_process = last_state_with_user_message
        else: # Если ни один узел не установил status_message_to_user, берем последнее известное состояние
            final_state_snapshot_after_stream = await compiled_agent.aget_state(config)
            if final_state_snapshot_after_stream:
                final_state_to_process = final_state_snapshot_after_stream # type: ignore

        if final_state_to_process:
            logger.debug(f"Final state to process for thread_id {thread_id}: {str(final_state_to_process)[:500]}")
            agent_response_text = final_state_to_process.get("status_message_to_user")

            if not agent_response_text:
                messages_history: List[BaseMessage] = final_state_to_process.get("messages", []) # type: ignore
                if messages_history and isinstance(messages_history[-1], AIMessage):
                    agent_response_text = messages_history[-1].content
                    logger.info(f"Using last AI message as response: {str(agent_response_text)[:100]}")
            
            if not agent_response_text:
                logger.warning(f"No status_message_to_user and no last AIMessage for thread_id {thread_id}")
                agent_response_text = "Кажется, у меня нет ответа на это. Попробуйте спросить что-нибудь еще."

            # Проверка на завершение цикла (END state)
            current_graph_state_snapshot = await compiled_agent.aget_state(config) # Получаем самое актуальное состояние
            if current_graph_state_snapshot and not current_graph_state_snapshot.next:
                logger.info(f"Conversation cycle for chat_id {chat_id} (thread_id {thread_id}) has ended.")
                if chat_id in user_sessions:
                    del user_sessions[chat_id]
        else:
            logger.error(f"No final state could be processed from agent for thread_id {thread_id}")
            agent_response_text = "Извините, что-то пошло не так при обработке вашего запроса."

    except GraphRecursionError as r_err: # Ловим конкретную ошибку
        logger.error(f"GraphRecursionError for chat_id {chat_id} (thread_id {thread_id}): {r_err}", exc_info=True)
        agent_response_text = "Кажется, я запутался в своих мыслях. Попробуйте, пожалуйста, начать заново или переформулировать запрос."
        if chat_id in user_sessions: # Сбрасываем сессию при рекурсии
            del user_sessions[chat_id]
    except Exception as e:
        logger.error(f"Error processing message for chat_id {chat_id} (thread_id {thread_id}): {e}", exc_info=True)
        agent_response_text = "Извините, произошла внутренняя ошибка. Пожалуйста, попробуйте позже."

    logger.info(f"Agent response for chat_id {chat_id}: '{str(agent_response_text)[:200]}'")
    return agent_response_text

async def start_command_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    chat_id = update.message.chat_id
    logger.info(f"/start command received from chat_id {chat_id}")
    # При старте можно сбросить сессию, если она была
    if chat_id in user_sessions:
        del user_sessions[chat_id]
        logger.info(f"Session reset for chat_id {chat_id} due to /start command.")

    await update.message.reply_text(
        "Привет! Я ваш ассистент по планированию мероприятий. "
        "Расскажите, что бы вы хотели найти? Укажите город, даты и ваши интересы."
    )


async def telegram_message_handler_wrapper(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Обработчик текстовых сообщений от пользователя."""
    if update.message and update.message.text:
        chat_id = update.message.chat_id
        user_text = update.message.text

        logger.info(f"Received Telegram message from chat_id {chat_id}: '{user_text}'")

        # Показываем индикатор "печатает..."
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        bot_reply = await handle_telegram_message(chat_id, user_text)

        if bot_reply:
            await update.message.reply_text(bot_reply)
        else:
            # Если bot_reply пустой или None, отправляем стандартное сообщение
            logger.warning(
                f"Bot reply was empty for chat_id {chat_id}. Sending default message."
            )
            await update.message.reply_text(
                "Не смог обработать ваш запрос. Попробуйте еще раз."
            )


if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "TELEGRAM_BOT_TOKEN не найден в .env файле или переменных окружения!"
        )
        exit()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start_command_handler))

    # Добавляем обработчик текстовых сообщений
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND, telegram_message_handler_wrapper
        )
    )

    logger.info("Telegram bot starting...")
    try:
        application.run_polling()
    except Exception as e:
        logger.critical(
            f"Telegram bot failed to start or polling error: {e}", exc_info=True
        )
    logger.info("Telegram bot stopped.")
