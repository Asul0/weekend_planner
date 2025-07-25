import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import (
    RouteBuilderToolArgs,
    LocationModel,
    RouteDetails,
    RouteSegment,
)

try:
    from services.gis_service import get_route, get_coords_from_address
except ImportError as e:
    # Эта заглушка оставлена, так как она важна для работоспособности файла при проблемах с импортом
    logger_init = logging.getLogger(__name__ + "_init")
    logger_init.critical(
        f"Критическая ошибка: Не удалось импортировать функции из services.gis_service: {e}. Проверьте, что файл services/gis_service.py существует и содержит необходимые функции (бывший gis_new.py)."
    )

    async def get_route(*args, **kwargs) -> Dict[str, Any]:
        logging.getLogger(__name__).error(
            "ЗАГЛУШКА get_route вызвана из-за ошибки импорта services.gis_service."
        )
        return {
            "status": "error",
            "message": "GIS сервис (маршруты) недоступен из-за ошибки импорта.",
        }

    async def get_coords_from_address(*args, **kwargs) -> Optional[List[float]]:
        logging.getLogger(__name__).error(
            "ЗАГЛУШКА get_coords_from_address вызвана из-за ошибки импорта services.gis_service."
        )
        return None


logger = logging.getLogger(__name__)


async def _resolve_location_coords(
    location: LocationModel,
    city_for_geocoding: Optional[str],  # Этот параметр должен приходить
) -> Optional[LocationModel]:
    if location.lon is not None and location.lat is not None:
        # Если координаты уже есть, ничего не делаем
        if (
            location.lon != 0.0 or location.lat != 0.0
        ):  # Добавим проверку на нулевые координаты
            logger.debug(
                f"RouteBuilderTool: Location '{location.address_string or f'{location.lat},{location.lon}'}' already has valid coordinates."
            )
            return location
        else:
            logger.debug(
                f"RouteBuilderTool: Location '{location.address_string}' has zero coordinates, will attempt geocoding if address exists."
            )

    if location.address_string:
        # Важно: используем city_for_geocoding, который передан в route_builder_tool
        logger.debug(
            f"RouteBuilderTool: No valid coords for '{location.address_string}', attempting geocoding (city context: {city_for_geocoding})."
        )
        # get_coords_from_address должен использовать get_geocoding_details, который принимает city
        coords = await get_coords_from_address(
            address=location.address_string,
            city=city_for_geocoding,  # Передаем контекст города
        )
        if coords:
            location.lon = coords[0]
            location.lat = coords[1]
            logger.debug(
                f"RouteBuilderTool: Geocoded '{location.address_string}' to lon: {location.lon}, lat: {location.lat} using city context '{city_for_geocoding}'"
            )
            return location
        else:
            logger.warning(
                f"RouteBuilderTool: Failed to geocode address '{location.address_string}' for routing (city context: {city_for_geocoding})."
            )
            return None  # Не удалось геокодировать

    logger.warning(
        "RouteBuilderTool: Location has neither valid coords nor address string for geocoding."
    )
    return None


@tool("route_builder_tool", args_schema=RouteBuilderToolArgs)
async def route_builder_tool(
    start_point: Dict[str, Any],
    event_points: List[Dict[str, Any]],
    transport_type: str = "driving",
    city_context_for_geocoding: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        start_point_model = LocationModel(**start_point)
        event_points_models = [LocationModel(**ep_dict) for ep_dict in event_points]
    except ValidationError as ve_init:
        logger.error(
            f"RouteBuilderTool: Invalid input for LocationModel during initial validation: {ve_init}"
        )
        return RouteDetails(
            status="error",
            error_message=f"Ошибка входных данных для точек маршрута: {ve_init}",
        ).model_dump()
    # Исправление для случая, если на вход уже приходят Pydantic модели
    except TypeError:
        if isinstance(start_point, LocationModel) and all(
            isinstance(ep, LocationModel) for ep in event_points
        ):
            start_point_model = start_point
            event_points_models = event_points
            logger.info(
                "RouteBuilderTool: Inputs were already LocationModel instances."
            )
        else:
            raise  # Перевыбрасываем ошибку, если это не тот случай TypeError

    start_address_log = start_point_model.address_string or (
        f"координаты ({start_point_model.lat}, {start_point_model.lon})"
        if start_point_model.lat is not None
        else "не указан"
    )
    logger.info(
        f"RouteBuilderTool: Building route with {len(event_points_models)} event_points. "
        f"Start: {start_address_log}. Transport: {transport_type}."
    )

    resolved_start_point = await _resolve_location_coords(
        start_point_model, city_context_for_geocoding
    )
    if not resolved_start_point:
        logger.error("RouteBuilderTool: Could not resolve start point coordinates.")
        return RouteDetails(
            status="error",
            error_message="Не удалось определить координаты начальной точки маршрута.",
        ).model_dump()

    resolved_event_points: List[LocationModel] = []
    for i, ep_model_orig in enumerate(event_points_models):
        resolved_ep = await _resolve_location_coords(
            ep_model_orig, city_context_for_geocoding
        )
        if resolved_ep:
            resolved_event_points.append(resolved_ep)
        else:
            logger.warning(
                f"RouteBuilderTool: Could not resolve coordinates for event point {i+1} ('{ep_model_orig.address_string or 'N/A'}'). This point will be skipped in routing."
            )

    if (
        not resolved_event_points and event_points_models
    ):  # Если были точки назначения, но ни одна не разрешилась
        logger.error(
            "RouteBuilderTool: Could not resolve coordinates for any of the provided event points."
        )
        return RouteDetails(
            status="error",
            error_message="Не удалось определить координаты для точек назначения мероприятий.",
        ).model_dump()

    if (
        not event_points_models
    ):  # Если изначально не было точек назначения (только старт)
        logger.info(
            "RouteBuilderTool: No event points provided, no route to build beyond start point."
        )
        return RouteDetails(
            status="success",
            segments=[],
            total_duration_seconds=0,
            total_distance_meters=0,
            total_duration_text="0 мин",
            total_distance_text="0 км",
        ).model_dump()

    all_sequential_points_for_api = [resolved_start_point] + resolved_event_points

    built_segments: List[RouteSegment] = []
    aggregated_duration_seconds = 0
    aggregated_distance_meters = 0
    overall_status = "success"
    any_segment_failed = False

    # Если после разрешения осталось меньше 2 точек (например, старт + 1 событие, но событие не разрешилось)
    if len(all_sequential_points_for_api) < 2:
        logger.info(
            "RouteBuilderTool: Less than 2 resolvable points available (start + events), no route segments to build."
        )
        # Проверяем, были ли изначально точки, которые должны были войти в маршрут
        if event_points_models and not resolved_event_points:  # Были, но не разрешились
            return RouteDetails(
                status="error",
                error_message="Не удалось разрешить координаты ни для одной из точек назначения для построения сегментов.",
            ).model_dump()
        return RouteDetails(  # Не было точек для сегментов или старт + 1 разрешенная точка (маршрут не нужен)
            status="success",
            segments=[],
            total_duration_seconds=0,
            total_distance_meters=0,
            total_duration_text="0 мин",
            total_distance_text="0 км",
        ).model_dump()

    for i in range(len(all_sequential_points_for_api) - 1):
        point_a = all_sequential_points_for_api[i]
        point_b = all_sequential_points_for_api[i + 1]

        if (
            point_a.lon is None
            or point_a.lat is None
            or point_b.lon is None
            or point_b.lat is None
        ):
            logger.error(
                f"RouteBuilderTool: Internal error - missing coordinates for resolved points in segment {i+1}. Point A: {point_a.model_dump()}, Point B: {point_b.model_dump()}. Skipping segment."
            )
            built_segments.append(
                RouteSegment(
                    from_address=point_a.address_string or "Неизвестная точка A",
                    to_address=point_b.address_string or "Неизвестная точка B",
                    segment_status="error",
                    segment_error_message="Внутренняя ошибка: отсутствуют координаты для построения сегмента.",
                )
            )
            any_segment_failed = True
            continue

        segment_from_address_log = (
            point_a.address_string or f"коорд. ({point_a.lat:.4f}, {point_a.lon:.4f})"
        )
        segment_to_address_log = (
            point_b.address_string or f"коорд. ({point_b.lat:.4f}, {point_b.lon:.4f})"
        )

        logger.info(
            f"RouteBuilderTool: Requesting API for segment from '{segment_from_address_log}' to '{segment_to_address_log}'"
        )

        segment_api_result = await get_route(
            points=[
                {"lon": point_a.lon, "lat": point_a.lat},
                {"lon": point_b.lon, "lat": point_b.lat},
            ],
            transport=transport_type,
        )

        if segment_api_result and segment_api_result.get("status") == "success":
            duration_s = segment_api_result.get("duration_seconds", 0)
            distance_m = segment_api_result.get("distance_meters", 0)
            built_segments.append(
                RouteSegment(
                    from_address=point_a.address_string,
                    to_address=point_b.address_string,
                    duration_seconds=duration_s,
                    duration_text=segment_api_result.get("duration_text"),
                    distance_meters=distance_m,
                    distance_text=segment_api_result.get("distance_text"),
                    transport_type=transport_type,
                    segment_status="success",
                )
            )
            aggregated_duration_seconds += duration_s
            aggregated_distance_meters += distance_m
        else:
            error_msg_segment = (
                segment_api_result.get("message")
                if isinstance(segment_api_result, dict)
                else "Не удалось построить сегмент маршрута."
            )
            logger.error(
                f"RouteBuilderTool: Failed to build segment {i+1} ('{segment_from_address_log}' to '{segment_to_address_log}'). API Error: {error_msg_segment}"
            )
            built_segments.append(
                RouteSegment(
                    from_address=segment_from_address_log,
                    to_address=segment_to_address_log,
                    segment_status="error",
                    segment_error_message=error_msg_segment,
                )
            )
            any_segment_failed = True

    if any_segment_failed and any(
        s.segment_status == "success" for s in built_segments
    ):
        overall_status = "partial_success"
    elif any_segment_failed:  # Все сегменты провалились
        overall_status = "error"
    elif not built_segments and len(all_sequential_points_for_api) >= 2:
        overall_status = "error"
        logger.warning(
            "RouteBuilderTool: No segments were built despite having sufficient points."
        )
    # Если overall_status остался "success", но нет сегментов (т.к. len(all_sequential_points_for_api) < 2),
    # это уже обработано выше и возвращен пустой успешный маршрут.

    overall_error_message = None
    if overall_status == "error":
        if all(s.segment_status == "error" for s in built_segments) and built_segments:
            overall_error_message = "Не удалось построить ни одну часть маршрута."
        elif not built_segments and (event_points_models and not resolved_event_points):
            overall_error_message = "Не удалось определить координаты для точек назначения мероприятий, поэтому маршрут не построен."
        elif not built_segments and len(all_sequential_points_for_api) >= 2:
            overall_error_message = (
                "Маршрут не был построен по неизвестной причине, хотя точки были."
            )
        else:
            first_error_segment = next(
                (s for s in built_segments if s.segment_status == "error"), None
            )
            overall_error_message = (
                first_error_segment.segment_error_message
                if first_error_segment
                else "Не удалось построить полный маршрут."
            )
        if (
            not built_segments
            and overall_error_message is None
            and overall_status == "error"
        ):  # Дополнительный fallback
            overall_error_message = "Не удалось построить маршрут."

    elif overall_status == "partial_success":
        overall_error_message = "Не все части маршрута удалось успешно построить."

    final_route_details = RouteDetails(
        status=overall_status,
        segments=(built_segments if built_segments else None),
        total_duration_seconds=(
            aggregated_duration_seconds if aggregated_duration_seconds > 0 else None
        ),
        total_duration_text=(
            f"~{round(aggregated_duration_seconds / 60)} мин"
            if aggregated_duration_seconds > 0
            else None
        ),
        total_distance_meters=(
            aggregated_distance_meters if aggregated_distance_meters > 0 else None
        ),
        total_distance_text=(
            f"~{round(aggregated_distance_meters / 1000, 1)} км"
            if aggregated_distance_meters > 0
            else None
        ),
        error_message=overall_error_message,
    )

    logger.info(
        f"RouteBuilderTool: Finished. Overall status: {final_route_details.status}. Segments: {len(final_route_details.segments or [])}"
    )
    return final_route_details.model_dump(exclude_none=True)
