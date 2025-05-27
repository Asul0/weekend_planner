# agent_YT/proxy/server.py

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response
import httpx
import json
import logging  # Добавляем logging

REAL_API_BASE_URL = "https://api.afisha.ru"
PROXY_INTERNAL_WIDGET_KEY = "01092eca-f696-44bc-b573-428d53011100"
PROXY_INTERNAL_PARTNER_KEY = "bcb6a92d-a520-41f0-82e4-1e9e276e05bb"

app = FastAPI(title="Afisha Proxy Server")
_proxy_client = None

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG) # Это будет настроено в uvicorn.run


@app.on_event("startup")
async def startup_event():
    global _proxy_client
    # Решение: создаем httpx.AsyncClient с отключенной автоматической обработкой Content-Encoding
    # Это значит, что httpx сам не будет пытаться распаковывать ответы от api.afisha.ru
    # и не будет автоматически добавлять Accept-Encoding: gzip к запросам
    _proxy_client = httpx.AsyncClient(
        http2=True, follow_redirects=True
    )  # http2 и follow_redirects по умолчанию, но можно указать явно
    logger.info(f"Afisha Proxy Server started. Forwarding to: {REAL_API_BASE_URL}")
    logger.info(
        f"Using internal WidgetKey ending with: ...{PROXY_INTERNAL_WIDGET_KEY[-4:]}"
    )


@app.on_event("shutdown")
async def shutdown_event():
    global _proxy_client
    if _proxy_client:
        await _proxy_client.aclose()
    logger.info("Afisha Proxy Server stopped.")


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_afisha_requests(full_path: str, request: Request):
    global _proxy_client
    if not _proxy_client:
        raise HTTPException(status_code=503, detail="Proxy client not initialized")

    target_url = f"{REAL_API_BASE_URL}/{full_path}"
    client_params = dict(request.query_params)
    client_headers_dict = dict(request.headers)  # Преобразуем в обычный dict

    headers_to_afisha = {}
    for key, value in client_headers_dict.items():
        # Пропускаем заголовки, которые не должны идти к целевому серверу
        if key.lower() not in [
            "host",
            "content-length",
            "transfer-encoding",
            "connection",
            "accept-encoding",
        ]:
            headers_to_afisha[key] = value

    # Устанавливаем ключи для API Афиши
    headers_to_afisha["X-ApiAuth-PartnerKey"] = PROXY_INTERNAL_PARTNER_KEY
    # Accept-Encoding удален, чтобы запросить несжатый ответ от Афиши

    params_to_afisha = client_params.copy()
    params_to_afisha["WidgetKey"] = PROXY_INTERNAL_WIDGET_KEY
    params_to_afisha.pop("PartnerKey", None)
    params_to_afisha.pop("X-ApiAuth-PartnerKey", None)

    body_bytes = await request.body()

    logger.debug(
        f"Proxying to Afisha: URL={target_url}, PARAMS={params_to_afisha}, HEADERS_TO_AFISHA={headers_to_afisha}"
    )

    try:
        response_from_afisha = await _proxy_client.request(
            method=request.method,
            url=target_url,
            params=params_to_afisha,
            headers=headers_to_afisha,  # Заголовки без Accept-Encoding
            content=body_bytes,
            timeout=60.0,
        )

        # Читаем сырые байты ответа
        response_content_bytes = await response_from_afisha.aread()

        # Формируем заголовки для ответа клиенту (afisha_service.py)
        # Удаляем Content-Encoding, так как мы передаем несжатые данные
        # Также удаляем Content-Length, так как он может быть неверным после удаления Content-Encoding
        response_headers_to_client = {
            key: value
            for key, value in response_from_afisha.headers.items()
            if key.lower()
            not in ["content-encoding", "content-length", "transfer-encoding"]
        }
        # Если Content-Type не установлен, FastAPI/Uvicorn могут его не передать правильно
        if (
            "content-type" not in response_headers_to_client
            and response_from_afisha.status_code < 400
        ):  # Добавляем только для успешных ответов
            # Пытаемся угадать, если это JSON (это может быть небезопасно, если Афиша вернет не JSON)
            # Но для /v3/cities это должен быть JSON
            try:
                json.loads(
                    response_content_bytes.decode("utf-8")
                )  # Проверка, что это валидный JSON
                response_headers_to_client["content-type"] = (
                    "application/json; charset=utf-8"
                )
                logger.debug(
                    "Proxy: Added 'application/json; charset=utf-8' as Content-Type for client."
                )
            except:
                logger.warning(
                    "Proxy: Could not determine Content-Type, not adding it."
                )

        logger.debug(
            f"Proxy: Response from Afisha Status: {response_from_afisha.status_code}"
        )
        logger.debug(f"Proxy: Headers TO CLIENT: {response_headers_to_client}")
        logger.debug(f"Proxy: Content length TO CLIENT: {len(response_content_bytes)}")

        if response_from_afisha.status_code >= 400:
            error_detail_text = response_content_bytes.decode("utf-8", errors="replace")
            try:
                error_detail_json = json.loads(error_detail_text)
                logger.error(
                    f"Error from Afisha API ({response_from_afisha.status_code}): {json.dumps(error_detail_json, indent=2, ensure_ascii=False)}"
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Error from Afisha API ({response_from_afisha.status_code}), non-JSON body: {error_detail_text}"
                )
            # Важно: передаем error_detail_text, а не error_detail_json в HTTPException
            raise HTTPException(
                status_code=response_from_afisha.status_code, detail=error_detail_text
            )

        return Response(
            content=response_content_bytes,
            status_code=response_from_afisha.status_code,
            headers=response_headers_to_client,
        )

    except httpx.HTTPStatusError as e:
        # ... (остальная часть обработки ошибок остается)
        logger.error(
            f"HTTPStatusError while proxying to {e.request.url}: Status {e.response.status_code}, Response: {e.response.text[:1000]}"
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.error(f"RequestError while proxying to {e.request.url}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in proxy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s",
    )
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
