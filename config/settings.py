import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # GigaChat API
    GIGACHAT_CREDENTIALS: str = os.getenv("GIGACHAT_CREDENTIALS", "")
    GIGACHAT_SCOPE: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS") # Значение по умолчанию

    # GIS (2GIS) API
    GIS_API_KEY: str = os.getenv("GIS_API_KEY", "")

    # Afisha Proxy
    AFISHA_PROXY_BASE_URL: str = os.getenv("AFISHA_PROXY_BASE_URL", "http://localhost:8000") # Пример

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __init__(self):
        if not self.GIGACHAT_CREDENTIALS:
            print("WARNING: GIGACHAT_CREDENTIALS не найден в .env файле или переменных окружения.")
        if not self.GIS_API_KEY:
            print("WARNING: GIS_API_KEY не найден в .env файле или переменных окружения.")

settings = Settings()