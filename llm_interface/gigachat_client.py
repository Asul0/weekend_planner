from langchain_gigachat import GigaChat # Используем актуальный импорт
from langchain_core.tools import BaseTool
from typing import List, Optional

from config.settings import settings

_gigachat_instance: Optional[GigaChat] = None

def get_gigachat_client(tools: Optional[List[BaseTool]] = None) -> GigaChat:
    global _gigachat_instance
    
    if _gigachat_instance is None or tools is not None:
        client_params = {
            "credentials": settings.GIGACHAT_CREDENTIALS,
            "scope": settings.GIGACHAT_SCOPE,
            "verify_ssl_certs": False, # Как правило, для локальной разработки
            "model": "GigaChat-Pro", # Модель можно указать здесь или при вызове
            "timeout": 120, # Увеличим общий таймаут для потенциально долгих операций
            "profanity_check": False # Отключаем, если не требуется спецификой задачи
        }
        
        current_client = GigaChat(**client_params)

        if tools:
            try:
                _gigachat_instance = current_client.bind_tools(tools)
            except Exception as e:
                print(f"ERROR: Failed to bind tools to GigaChat: {e}")
                _gigachat_instance = current_client 
        else:
            _gigachat_instance = current_client
            
    elif _gigachat_instance and tools is None:
        pass


    if not _gigachat_instance:
        # Эта ситуация не должна возникнуть при корректной логике выше,
        # но как fallback создаем экземпляр без инструментов.
        _gigachat_instance = GigaChat(
            credentials=settings.GIGACHAT_CREDENTIALS,
            scope=settings.GIGACHAT_SCOPE,
            verify_ssl_certs=False,
            timeout=120,
            profanity_check=False
        )
        print("WARNING: GigaChat client was unexpectedly re-initialized without tools as a fallback.")

    return _gigachat_instance