import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import (
    RouteBuilderToolArgs,
    LocationModel,
    RouteDetails,
    RouteSegment,
)  # Импортируем наши схемы
from services.gis_service import (
    get_route,
    get_coords_from_address,
)  # Наш сервис для 2GIS

logger = logging.getLogger(__name__)


@tool("route_builder_tool", args_schema=RouteBuilderToolArgs)
async def route_builder_tool(
    start_point: LocationModel,
    event_points: List[LocationModel],
    transport_type: str = "driving",
) -> Dict[str, Any]:
    logger.info(
        f"RouteBuilderTool: Building route with {len(event_points)} event_points from start: {start_point.address_string or 'coords'}"
    )

    all_sequential_points = [start_point] + event_points

    built_segments: List[RouteSegment] = []
    aggregated_duration_seconds = 0
    aggregated_distance_meters = 0
    overall_status = "success"  # Изначально предполагаем успех
    any_segment_failed = False

    if len(all_sequential_points) < 2:
        logger.info(
            "RouteBuilderTool: Less than 2 points provided, no route segments to build."
        )
        # Возвращаем успешный RouteDetails с пустыми сегментами
        return RouteDetails(
            status="success",
            segments=[],
            total_duration_seconds=0,
            total_duration_text="0 мин",
            total_distance_meters=0,
            total_distance_text="0 км",
        ).model_dump()

    for i in range(len(all_sequential_points) - 1):
        point_a = all_sequential_points[i]
        point_b = all_sequential_points[i + 1]

        segment_from_address = (
            point_a.address_string or f"Точка ({point_a.lat:.4f}, {point_a.lon:.4f})"
        )
        segment_to_address = (
            point_b.address_string or f"Точка ({point_b.lat:.4f}, {point_b.lon:.4f})"
        )

        logger.info(
            f"RouteBuilderTool: Requesting segment from '{segment_from_address}' to '{segment_to_address}'"
        )

        if (
            point_a.lon is None
            or point_a.lat is None
            or point_b.lon is None
            or point_b.lat is None
        ):
            logger.warning(
                f"Segment {i+1} skipped: missing coordinates for '{segment_from_address}' or '{segment_to_address}'."
            )
            built_segments.append(
                RouteSegment(
                    from_address=segment_from_address,
                    to_address=segment_to_address,
                    segment_status="error",
                    segment_error_message="Отсутствуют координаты для одной из точек сегмента.",
                )
            )
            any_segment_failed = True
            continue

        # Передаем только две точки для одного сегмента
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
                    from_address=point_a.address_string,  # Используем оригинальный адрес для описания
                    to_address=point_b.address_string,  # Используем оригинальный адрес для описания
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
            error_msg_segment = segment_api_result.get(
                "message", "Не удалось построить сегмент."
            )
            logger.error(
                f"RouteBuilderTool: Failed to build segment {i+1} from '{segment_from_address}' to '{segment_to_address}'. Error: {error_msg_segment}"
            )
            built_segments.append(
                RouteSegment(
                    from_address=segment_from_address,
                    to_address=segment_to_address,
                    segment_status="error",
                    segment_error_message=error_msg_segment,
                )
            )
            any_segment_failed = True
            # Продолжаем пытаться строить остальные сегменты

    if (
        any_segment_failed and built_segments
    ):  # Если были ошибки, но есть и успешные сегменты
        overall_status = "partial_success"
    elif any_segment_failed and not any(
        s.segment_status == "success" for s in built_segments
    ):  # Если все сегменты провалились
        overall_status = "error"
        # Можно взять сообщение об ошибке первого неудачного сегмента как общее
        first_error_segment = next(
            (s for s in built_segments if s.segment_status == "error"), None
        )
        overall_error_message = (
            first_error_segment.segment_error_message
            if first_error_segment
            else "Не удалось построить маршрут."
        )
    elif (
        not built_segments
    ):  # Если по какой-то причине нет сегментов (например, было меньше 2 точек изначально)
        overall_status = "error"
        overall_error_message = "Нет данных для построения маршрута."
    else:  # Все сегменты успешно построены
        overall_status = "success"
        overall_error_message = None

    final_route_details = RouteDetails(
        status=overall_status,
        segments=built_segments if built_segments else None,
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
        error_message=overall_error_message if overall_status == "error" else None,  # type: ignore
    )

    logger.info(
        f"RouteBuilderTool: Finished. Overall status: {final_route_details.status}"
    )
    return final_route_details.model_dump()
