import aiohttp
import logging
import asyncio
from typing import Optional, List, Dict, Any
import json
from config.settings import settings  # Используем наш settings
from pydantic import BaseModel, Field, ValidationError
# GIS_API_KEY берется из settings
GIS_API_BASE_URL = "https://catalog.api.2gis.com/3.0"
ROUTING_API_BASE_URL = "https://routing.api.2gis.com/routing/7.0.0/global"

logger = logging.getLogger(__name__)
# Настройка логирования будет производиться глобально в main.py или при инициализации
# logging.basicConfig(level=settings.LOG_LEVEL.upper())


class GeocodingResult(BaseModel):
    coords: Optional[List[float]] = None
    match_level: str = (
        "not_found"  # 'building', 'street', 'city_district', 'city', 'ambiguous_multiple', 'not_found', 'error'
    )
    full_address_name_gis: Optional[str] = None
    is_precise_enough: bool = False  # True if match_level == 'building'
    error_message: Optional[str] = None


async def get_geocoding_details(
    address: str, city: Optional[str] = None
) -> GeocodingResult:
    url = f"{GIS_API_BASE_URL}/items"
    query_to_log = address

    params = {
        "q": address,
        "fields": "items.geometry.centroid,items.address_name,items.type,items.subtype,items.name",
        "page_size": 5,  # Берем несколько, чтобы оценить неоднозначность
        "key": settings.GIS_API_KEY,
    }

    if city:
        if city.lower() not in address.lower():
            query_with_city = f"{city}, {address}"
            params["q"] = query_with_city
            query_to_log = query_with_city
            logger.debug(f"2GIS Geocoding: Using query with city: '{query_with_city}'")
        else:
            logger.debug(
                f"2GIS Geocoding: Using original query: '{address}' (city already present)"
            )

    logger.info(f"2GIS Geocoding: Requesting geocoding for '{params['q']}'")

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

                    result_data = data.get("result", {})
                    items = result_data.get("items")
                    total_items = result_data.get("total", 0)

                    if items and len(items) > 0:
                        # Анализируем первый (наиболее релевантный) результат
                        item = items[0]
                        item_type = item.get("type")
                        item_subtype = item.get("subtype")
                        item_name = item.get("name")
                        item_address_name = item.get("address_name")
                        full_name_gis = item.get(
                            "full_name", item_address_name or item_name
                        )

                        point_str = item.get("geometry", {}).get("centroid")
                        coords_val = None
                        if point_str:
                            try:
                                lon_str, lat_str = (
                                    point_str.replace("POINT(", "")
                                    .replace(")", "")
                                    .split()
                                )
                                coords_val = [float(lon_str), float(lat_str)]
                            except Exception:
                                logger.warning(f"Failed to parse centroid: {point_str}")

                        if item_type == "building":
                            logger.info(
                                f"2GIS Geocoding: Found 'building' for '{query_to_log}'. Coords: {coords_val}. Address: {full_name_gis}"
                            )
                            return GeocodingResult(
                                coords=coords_val,
                                match_level="building",
                                full_address_name_gis=full_name_gis,
                                is_precise_enough=bool(
                                    coords_val
                                ),  # Точно, если есть координаты здания
                            )
                        elif item_type == "street":
                            logger.info(
                                f"2GIS Geocoding: Found 'street' for '{query_to_log}'. Address: {full_name_gis}"
                            )
                            return GeocodingResult(
                                coords=coords_val,  # Координаты улицы могут быть, могут не быть
                                match_level="street",
                                full_address_name_gis=full_name_gis,
                                is_precise_enough=False,
                            )
                        # Если результатов больше одного, и первый не здание, считаем неоднозначным
                        elif total_items > 1:
                            logger.info(
                                f"2GIS Geocoding: Found multiple ({total_items}) results for '{query_to_log}', first is '{item_type}'. Considered ambiguous."
                            )
                            return GeocodingResult(
                                match_level="ambiguous_multiple",
                                full_address_name_gis=f"Несколько результатов, например: {full_name_gis}",
                                is_precise_enough=False,
                            )
                        # Если найден только город или район
                        elif item_type == "adm_div" and (
                            item_subtype == "city"
                            or item_subtype == "city_district"
                            or item_subtype == "settlement"
                        ):
                            logger.info(
                                f"2GIS Geocoding: Found '{item_subtype}' for '{query_to_log}'. Address: {full_name_gis}"
                            )
                            return GeocodingResult(
                                coords=coords_val,
                                match_level=item_subtype,
                                full_address_name_gis=full_name_gis,
                                is_precise_enough=False,
                            )
                        else:  # Другие типы или неясный результат
                            logger.warning(
                                f"2GIS Geocoding: Found item of type '{item_type}/{item_subtype}' for '{query_to_log}', considered not precise. Item: {item}"
                            )
                            return GeocodingResult(
                                coords=coords_val,
                                match_level="other_type",
                                full_address_name_gis=full_name_gis,
                                is_precise_enough=False,
                            )
                    else:
                        logger.warning(
                            f"2GIS Geocoding: No 'items' in response for '{params['q']}'"
                        )
                        return GeocodingResult(match_level="not_found")
                else:
                    error_text = await response.text()
                    logger.error(
                        f"2GIS Geocoding API Error: Status {response.status} for {response.url}. Response: {error_text[:500]}"
                    )
                    return GeocodingResult(
                        match_level="error",
                        error_message=f"API Error {response.status}",
                    )
    except aiohttp.ClientConnectorError as e:
        logger.error(f"2GIS Geocoding: Connection error - {e}")
        return GeocodingResult(
            match_level="error", error_message=f"Connection error: {e}"
        )
    except asyncio.TimeoutError:
        logger.error(f"2GIS Geocoding: Request timeout for '{query_to_log}'")
        return GeocodingResult(match_level="error", error_message="Request timeout")
    except Exception as e:
        logger.error(f"2GIS Geocoding: Unexpected error - {e}", exc_info=True)
        return GeocodingResult(
            match_level="error", error_message=f"Unexpected error: {e}"
        )


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
) -> Dict[str, Any]:
    url = f"{ROUTING_API_BASE_URL}?key={settings.GIS_API_KEY}"
    headers = {"Content-Type": "application/json"}
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
        api_points.append(
            {
                "lon": p_val["lon"],
                "lat": p_val["lat"],
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
                response_text = await response.text()
                logger.debug(
                    f"2GIS Routing: Response Status: {response.status}, Body (first 500 chars): {response_text[:500]}"
                )

                if response.status == 200:
                    try:
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

                    result_list = data.get("result")
                    if (
                        result_list
                        and isinstance(result_list, list)
                        and len(result_list) > 0
                    ):
                        route_item = result_list[0]
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

                elif response.status == 422:
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
                else:
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
