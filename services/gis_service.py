import aiohttp
import logging
import asyncio
from typing import Optional, List, Dict, Any

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
    points: List[Dict[str, Any]],
    transport: str = "driving",
) -> Dict[str, Any]:  # Уточнил тип возвращаемого значения для консистентности
    url = f"{ROUTING_API_BASE_URL}?key={settings.GIS_API_KEY}"
    headers = {"Content-Type": "application/json"}

    formatted_points = []
    for p_idx, p_val in enumerate(points):
        if not isinstance(p_val, dict) or "lon" not in p_val or "lat" not in p_val:
            msg = f"Неверный формат точки {p_idx+1}. Ожидался {{'lon': float, 'lat': float}}, получено: {p_val}"
            logger.error(
                f"2GIS Routing: Invalid point format at index {p_idx}: {p_val}."
            )
            return {"status": "error", "message": msg}
        formatted_points.append(
            {
                "x": p_val["lon"],
                "y": p_val["lat"],
                "type": ("pedo" if transport == "walking" else "auto"),
            }
        )

    payload = {"transport": transport, "points": formatted_points}
    logger.info(
        f"2GIS Routing: Requesting route. Transport: {transport}, Points: {len(formatted_points)}"
    )
    logger.debug(f"2GIS Routing: Request URL: {url}, Payload: {str(payload)[:500]}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                response_text = await response.text()
                logger.debug(
                    f"2GIS Routing: Response Status: {response.status}, Body (first 500 chars): {response_text[:500]}"
                )

                if response.status == 200:
                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError:
                        logger.error(
                            f"2GIS Routing API returned non-JSON response (status 200): {response_text[:500]}"
                        )
                        return {
                            "status": "error",
                            "message": "Неожиданный формат ответа API маршрутов (не JSON).",
                        }

                    routes = data.get("routes")
                    if routes and isinstance(routes, list) and len(routes) > 0:
                        route_item = routes[0]
                        total_duration_seconds = route_item.get("duration")
                        total_distance_meters = route_item.get("distance")

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
                                f"2GIS Routing: Could not extract duration/distance from route: {route_item}"
                            )
                            return {
                                "status": "error",
                                "message": "Не удалось извлечь детали маршрута из ответа API 2GIS.",
                            }
                    elif data.get("error"):
                        error_info = data.get("error")
                        error_message = error_info.get(
                            "message", "Неизвестная ошибка API маршрутов 2GIS"
                        )
                        logger.error(f"2GIS Routing API returned error: {error_info}")
                        return {"status": "api_error", "message": error_message}
                    elif data.get("warning"):
                        warning_info = data.get("warning")
                        warning_message = warning_info.get(
                            "message", "API маршрутов 2GIS вернуло предупреждение"
                        )
                        logger.warning(
                            f"2GIS Routing API returned warning: {warning_info}"
                        )
                        return {"status": "api_warning", "message": warning_message}
                    else:
                        logger.warning(
                            f"2GIS Routing API: Unexpected response structure (status 200): {str(data)[:500]}"
                        )
                        return {
                            "status": "error",
                            "message": "Неожиданный формат ответа API 2GIS.",
                        }
                else:
                    error_message_text = response_text[:200]
                    try:
                        error_data = await response.json()
                        api_err_msg = error_data.get("error", {}).get("message")
                        if api_err_msg:
                            error_message_text = api_err_msg
                    except:
                        pass
                    logger.error(
                        f"2GIS Routing API HTTP error: Status {response.status}. Message: {error_message_text}"
                    )
                    return {
                        "status": "http_error",
                        "code": response.status,
                        "message": f"Ошибка API маршрутов 2GIS: {error_message_text}",
                    }
    except aiohttp.ClientConnectorError as e:
        logger.error(f"2GIS Routing: Connection error - {e}")
        return {
            "status": "connection_error",
            "message": f"Ошибка соединения с сервисом маршрутов 2GIS: {e}",
        }
    except asyncio.TimeoutError:
        logger.error("2GIS Routing: Request timeout.")
        return {
            "status": "timeout",
            "message": "Запрос к API маршрутов 2GIS истек по таймауту.",
        }
    except Exception as e:
        logger.error(f"2GIS Routing: Unexpected error - {e}", exc_info=True)
        return {
            "status": "unknown_error",
            "message": f"Неизвестная ошибка при построении маршрута через 2GIS: {e}",
        }
async def get_route_duration(
    from_lon: float,
    from_lat: float,
    to_lon: float,
    to_lat: float,
    transport_type: str = "driving",
) -> Optional[int]:
    """
    Получает длительность маршрута между двумя точками в секундах.
    Возвращает None, если маршрут не найден или произошла ошибка.
    """
    points = [{"lon": from_lon, "lat": from_lat}, {"lon": to_lon, "lat": to_lat}]
    route_result = await get_route(points, transport=transport_type)
    
    if route_result and route_result.get("status") == "success":
        duration = route_result.get("total_duration_seconds")
        if duration is not None:
            return int(duration) # Убедимся, что это целое число
    logger.warning(f"get_route_duration: Failed to get duration for {from_lon},{from_lat} to {to_lon},{to_lat} via {transport_type}. Result: {route_result}")
    return None
