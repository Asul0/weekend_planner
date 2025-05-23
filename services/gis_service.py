import aiohttp
import logging
import asyncio
from typing import Optional, List, Dict, Any
import json
from config.settings import settings  # Используем наш settings

# GIS_API_KEY берется из settings
GIS_API_BASE_URL = "https://catalog.api.2gis.com/3.0"
ROUTING_API_BASE_URL = "https://routing.api.2gis.com/routing/7.0.0/global"

logger = logging.getLogger(__name__)
# Настройка логирования будет производиться глобально в main.py или при инициализации
# logging.basicConfig(level=settings.LOG_LEVEL.upper())


async def get_coords_from_address(
    address: str, city: Optional[str] = None
) -> Optional[List[float]]:
    url = f"{GIS_API_BASE_URL}/items"
    params = {
        "q": address,
        "fields": "items.geometry.centroid",
        "page_size": 1,
        "key": settings.GIS_API_KEY,
    }
    if city:
        if city.lower() not in address.lower():
            query_with_city = f"{city}, {address}"
            params["q"] = query_with_city
            logger.debug(f"2GIS Geocoding: Using query with city: '{query_with_city}'")
        else:
            logger.debug(
                f"2GIS Geocoding: Using original query: '{address}' (city already present)"
            )

    logger.info(f"2GIS Geocoding: Requesting coords for '{params['q']}'")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                logger.debug(
                    f"2GIS Geocoding Request: GET {response.url} Status: {response.status}"
                )
                if response.status == 200:
                    data = await response.json()
                    logger.debug(
                        f"2GIS Geocoding: Raw response for '{params['q']}': {str(data)[:500]}"
                    )
                    items = data.get("result", {}).get("items")
                    if items and len(items) > 0:
                        point = items[0].get("geometry", {}).get("centroid")
                        if point:
                            try:
                                lon_str, lat_str = (
                                    point.replace("POINT(", "").replace(")", "").split()
                                )
                                lon = float(lon_str)
                                lat = float(lat_str)
                                logger.info(
                                    f"2GIS Geocoding: Coords found for '{params['q']}': {[lon, lat]}"
                                )
                                return [lon, lat]
                            except (
                                ValueError,
                                AttributeError,
                                IndexError,
                            ) as parse_err:
                                logger.error(
                                    f"2GIS Geocoding: Failed to parse centroid '{point}' for '{params['q']}': {parse_err}"
                                )
                                return None
                        else:
                            logger.warning(
                                f"2GIS Geocoding: No 'geometry.centroid' in item for '{params['q']}'. Item: {items[0]}"
                            )
                            return None
                    else:
                        logger.warning(
                            f"2GIS Geocoding: No 'items' in response for '{params['q']}'"
                        )
                        return None
                else:
                    error_text = await response.text()
                    logger.error(
                        f"2GIS Geocoding API Error: Status {response.status} for {response.url}. Response: {error_text[:500]}"
                    )
                    return None
    except aiohttp.ClientConnectorError as e:
        logger.error(f"2GIS Geocoding: Connection error - {e}")
        return None
    except asyncio.TimeoutError:
        logger.error(f"2GIS Geocoding: Request timeout for '{params['q']}'")
        return None
    except Exception as e:
        logger.error(f"2GIS Geocoding: Unexpected error - {e}", exc_info=True)
        return None


async def get_route(
    points: List[Dict[str, Any]],  # Ожидаем список словарей с 'lon' и 'lat'
    transport: str = "driving",
) -> Dict[str, Any]:
    url = f"{ROUTING_API_BASE_URL}?key={settings.GIS_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Формируем точки для API 2GIS. API ожидает 'x' для долготы и 'y' для широты.
    # Но если ошибка "'lat' or 'lon' is missed" относится к этому payload, попробуем 'lon' и 'lat'.
    # Проверьте документацию API 2GIS Routing на актуальный формат!
    # Пока оставим как было, предполагая, что 'x' и 'y' это правильно, а ошибка где-то еще.
    api_points = []
    for p_idx, p_val in enumerate(points):
        if not isinstance(p_val, dict) or "lon" not in p_val or "lat" not in p_val:
            msg = f"Неверный формат точки {p_idx+1}. Ожидался {{'lon': float, 'lat': float}}, получено: {p_val}"
            logger.error(
                f"2GIS Routing: Invalid point format at index {p_idx}: {p_val}."
            )
            return {
                "status": "error",
                "message": msg,
                "error_details": "Invalid input point format",
            }

        # Стандартный формат для 2GIS Routing API (x=lon, y=lat)
        api_points.append(
            {
                "lon": p_val["lon"],  # Долгота
                "lat": p_val["lat"],  # Широта
                "type": ("pedo" if transport == "walking" else "auto"),
            }
        )

    payload = {"transport": transport, "points": api_points}
    logger.info(
        f"2GIS Routing: Requesting route. Transport: {transport}, Points: {len(api_points)}"
    )
    logger.debug(f"2GIS Routing: Request URL: {url}, Payload: {str(payload)[:500]}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                response_text = await response.text()  # Читаем текст ответа один раз
                logger.debug(
                    f"2GIS Routing: Response Status: {response.status}, Body (first 500 chars): {response_text[:500]}"
                )

                if response.status == 200:
                    try:
                        # Пытаемся парсить JSON из уже прочитанного текста
                        data = (
                            await response.json(content_type=None)
                            if response.content_type == "application/json"
                            else json.loads(response_text)
                        )
                    except (aiohttp.ContentTypeError, json.JSONDecodeError) as json_err:
                        logger.error(
                            f"2GIS Routing API returned non-JSON or malformed JSON (status 200): {response_text[:500]}. Error: {json_err}"
                        )
                        return {
                            "status": "error",
                            "message": "Неожиданный или некорректный формат ответа API маршрутов (не JSON).",
                            "error_details": str(json_err),
                        }

                    # Убедимся, что data['result'] это список (как в вашем логе)
                    result_list = data.get("result")
                    if (
                        result_list
                        and isinstance(result_list, list)
                        and len(result_list) > 0
                    ):
                        route_item = result_list[0]  # Берем первый маршрут
                        total_duration_seconds = route_item.get("total_duration")
                        total_distance_meters = route_item.get("total_distance")

                        if (
                            total_duration_seconds is not None
                            and total_distance_meters is not None
                        ):
                            duration_minutes = round(total_duration_seconds / 60)
                            distance_km = round(total_distance_meters / 1000, 1)
                            duration_text = f"~{duration_minutes} мин"
                            distance_text = f"~{distance_km} км"
                            logger.info(
                                f"2GIS Routing: Route found - Duration: {duration_text}, Distance: {distance_text}"
                            )
                            return {
                                "status": "success",
                                "duration_seconds": total_duration_seconds,
                                "duration_text": duration_text,
                                "distance_meters": total_distance_meters,
                                "distance_text": distance_text,
                            }
                        else:
                            logger.warning(
                                f"2GIS Routing: Could not extract total_duration/total_distance from route: {route_item}"
                            )
                            return {
                                "status": "error",
                                "message": "Не удалось извлечь детали маршрута (длительность/расстояние).",
                                "error_details": "Missing duration/distance in API response.",
                            }
                    # Обработка ошибок и предупреждений от API 2GIS, если они есть в ответе 200 OK
                    elif data.get("error"):
                        error_info = data.get("error")
                        error_message = error_info.get(
                            "message",
                            "Неизвестная ошибка API маршрутов 2GIS (в ответе 200 OK)",
                        )
                        logger.error(
                            f"2GIS Routing API returned error in 200 OK response: {error_info}"
                        )
                        return {
                            "status": "api_error",
                            "message": error_message,
                            "error_details": error_info,
                        }
                    elif data.get("warning"):
                        warning_info = data.get("warning")
                        warning_message = warning_info.get(
                            "message",
                            "API маршрутов 2GIS вернуло предупреждение (в ответе 200 OK)",
                        )
                        logger.warning(
                            f"2GIS Routing API returned warning in 200 OK response: {warning_info}"
                        )
                        # Можно решить, считать ли это успехом или ошибкой
                        return {
                            "status": "api_warning",
                            "message": warning_message,
                            "warning_details": warning_info,
                        }
                    else:
                        logger.warning(
                            f"2GIS Routing API: Unexpected response structure in 200 OK (no 'result' list or 'error'): {str(data)[:500]}"
                        )
                        return {
                            "status": "error",
                            "message": "Неожиданный формат успешного ответа от API 2GIS.",
                            "error_details": "Missing 'result' or 'error' in 200 OK response.",
                        }

                elif response.status == 422:  # Явная обработка 422
                    error_details_422 = ""
                    try:
                        error_data_422 = (
                            await response.json(content_type=None)
                            if response.content_type == "application/json"
                            else json.loads(response_text)
                        )
                        error_details_422 = error_data_422.get(
                            "message", response_text[:200]
                        )
                    except:
                        error_details_422 = response_text[:200]
                    logger.error(
                        f"2GIS Routing API HTTP error: Status {response.status}. Message: {error_details_422}"
                    )
                    return {
                        "status": "http_error",
                        "code": response.status,
                        "message": f"Ошибка валидации данных для API маршрутов 2GIS: {error_details_422}",
                        "error_details": error_details_422,
                    }
                else:  # Другие HTTP ошибки
                    error_message_text = response_text[:200]
                    logger.error(
                        f"2GIS Routing API HTTP error: Status {response.status}. Message: {error_message_text}"
                    )
                    return {
                        "status": "http_error",
                        "code": response.status,
                        "message": f"Ошибка API маршрутов 2GIS (код: {response.status}): {error_message_text}",
                        "error_details": error_message_text,
                    }

    except aiohttp.ClientConnectorError as e:
        logger.error(f"2GIS Routing: Connection error - {e}")
        return {
            "status": "connection_error",
            "message": f"Ошибка соединения с сервисом маршрутов 2GIS: {e}",
            "error_details": str(e),
        }
    except asyncio.TimeoutError:
        logger.error("2GIS Routing: Request timeout.")
        return {
            "status": "timeout",
            "message": "Запрос к API маршрутов 2GIS истек по таймауту.",
            "error_details": "Timeout",
        }
    except Exception as e:
        logger.error(f"2GIS Routing: Unexpected error - {e}", exc_info=True)
        return {
            "status": "unknown_error",
            "message": f"Неизвестная ошибка при построении маршрута через 2GIS: {e}",
            "error_details": str(e),
        }
