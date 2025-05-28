import asyncio
import logging
import uuid
import os
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
from agent_core.agent_graph import get_compiled_agent_graph
from agent_core.agent_state import AgentState

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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

        final_node_output: Optional[AgentState] = None
        async for output_chunk in compiled_agent.astream(
            inputs, config, stream_mode="values"
        ):
            # output_chunk будет последним состоянием узла, который вернул значение перед тем,
            # как граф приостановился (дошел до END) или завершил обработку этого тика.
            final_node_output = output_chunk  # type: ignore
            node_name = final_node_output.get("__node__", "UnknownNode") if final_node_output else "NoOutput"  # type: ignore
            logger.debug(
                f"Intermediate/Final node output for thread_id {thread_id}: node={node_name}, state_keys={list(final_node_output.keys()) if final_node_output else 'None'}"
            )

        # После завершения astream, final_node_output содержит состояние последнего выполненного узла перед END
        # или последнего узла в цепочке, если граф не дошел до END, а просто ожидает (что достигается добавлением ребра в END)

        if final_node_output:
            agent_response_text = final_node_output.get("status_message_to_user")

            # Если status_message_to_user пуст, пытаемся взять последнее AI сообщение
            if not agent_response_text:
                messages_history: List[BaseMessage] = final_node_output.get("messages", [])  # type: ignore
                if messages_history and isinstance(messages_history[-1], AIMessage):
                    agent_response_text = messages_history[-1].content
                    logger.info(
                        f"Using last AI message as response for thread_id {thread_id}: {str(agent_response_text)[:100]}"
                    )

            if not agent_response_text:
                logger.warning(
                    f"No status_message_to_user and no last AIMessage for thread_id {thread_id}. State: {str(final_node_output)[:300]}"
                )
                agent_response_text = (
                    "Кажется, я сейчас не знаю, что ответить. Попробуйте еще раз."
                )

            # Проверяем, действительно ли граф завершил свою работу для этой сессии
            # Это полезно, если есть узлы, которые могут приводить к состоянию, где пользователь должен ответить,
            # но граф сам по себе еще не "закончился" для данной сессии (например, если не все пути ведут в END явно)
            # Однако, если все узлы, отправляющие сообщение пользователю, ведут в END, то astream сам завершится.
            current_graph_state_after_stream = await compiled_agent.aget_state(config)
            if (
                current_graph_state_after_stream
                and not current_graph_state_after_stream.next
            ):
                logger.info(
                    f"LangGraph conversation for thread_id {thread_id} (chat_id {chat_id}) reached END state."
                )
                # Решение о сбросе сессии user_sessions[chat_id] здесь может быть преждевременным,
                # так как пользователь может захотеть продолжить с того же места, если агент не сказал "до свидания".
                # Сброс лучше делать при команде /start или если агент явно завершил диалог (например, сказал "Рад был помочь").
                if agent_response_text and "Рад был помочь" in agent_response_text:
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
            logger.info(
                f"Session for chat_id {chat_id} (thread_id {thread_id}) cleared due to recursion error."
            )
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
        # Опционально: можно очистить состояние графа для этого thread_id, если LangGraph это поддерживает
        # config = {"configurable": {"thread_id": thread_id_to_clear}}
        # await compiled_agent.update_state(config, None) # Это сбросит состояние, если нужно
        # Либо просто удаляем его из нашего словаря, чтобы при следующем сообщении создался новый
        del user_sessions[chat_id]
        logger.info(
            f"Session for chat_id {chat_id} (thread_id {thread_id_to_clear}) cleared due to /start command."
        )

    await update.message.reply_text(
    "Привет! Я ваш ассистент по планированию мероприятий. 😊\n\n"
    "Чтобы подобрать лучшие варианты, уточните детали:\n\n"
    "📍 <b>Город</b> (например: Москва, Санкт-Петербург)\n"
    "📅 <b>Дата</b> (сегодня, завтра, 15 июня или диапазон дат)\n"
    "⏰ <b>Время</b> (например: вечер, с 19:00, или интервал 14:00-17:00)\n"
    "🎭 <b>Интересы</b> (кино, концерт, театр, выставка, фестиваль, стендап, ресторан и т.д.)\n"
    "💰 <b>Бюджет</b> на человека/компанию (пример: до 2000 ₽, 5000-10000 ₽)\n\n"
    "Пример запроса:\n"
    "<i>\"Ищу концерт в Москве 20 июля вечером, бюджет до 5000 ₽\"</i>\n\n"
    "Готов предложить крутые варианты! 🎉"
)


async def telegram_message_handler_wrapper(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.message and update.message.text:
        chat_id = update.message.chat_id
        user_text = update.message.text

        # Не логгируем здесь, т.к. handle_telegram_message уже логгирует
        # logger.info(f"Received Telegram message from chat_id {chat_id}: '{user_text}'")

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
