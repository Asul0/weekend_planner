import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas.data_schemas import (
    RouteBuilderToolArgs,
    LocationModel,
    RouteDetails,
)  # Импортируем наши схемы
from services.gis_service import (
    get_route,
    get_coords_from_address,
)  # Наш сервис для 2GIS

logger = logging.getLogger(__name__)


@tool("route_builder_tool", args_schema=RouteBuilderToolArgs)
async def route_builder_tool(
    start_point: LocationModel,  # Уже LocationModel
    event_points: List[LocationModel],  # Уже List[LocationModel]
    transport_type: str = "driving",
) -> Dict[str, Any]:
    logger.info(
        f"RouteBuilderTool: Called with start_point={start_point.model_dump_json(exclude_none=True)[:100]}, "
        f"num_event_points={len(event_points)}, transport_type='{transport_type}'"
    )

    all_points_for_routing: List[Dict[str, float]] = []

    # 1. Обработка начальной точки
    # start_point УЖЕ является объектом LocationModel
    if start_point.lon is None or start_point.lat is None:
        if start_point.address_string:
            logger.debug(
                f"RouteBuilderTool: Geocoding start_point address: '{start_point.address_string}'"
            )
            # Предполагаем, что агент передаст city_name в get_coords_from_address, если это нужно сервису
            # Сейчас get_coords_from_address принимает city опционально.
            coords = await get_coords_from_address(address=start_point.address_string)
            if coords:
                start_point.lon, start_point.lat = (
                    coords[0],
                    coords[1],
                )  # Обновляем существующий объект
                logger.info(
                    f"RouteBuilderTool: Geocoded start_point to lon={coords[0]}, lat={coords[1]}"
                )
            else:
                msg = f"Не удалось получить координаты для начальной точки: {start_point.address_string}"
                logger.error(f"RouteBuilderTool: {msg}")
                return RouteDetails(status="error", error_message=msg).model_dump()
        else:
            msg = "Начальная точка не содержит ни координат, ни адреса для геокодирования."
            logger.error(f"RouteBuilderTool: {msg}")
            return RouteDetails(status="error", error_message=msg).model_dump()

    if start_point.lon is not None and start_point.lat is not None:
        all_points_for_routing.append({"lon": start_point.lon, "lat": start_point.lat})
    else:
        msg = f"Не удалось определить координаты для начальной точки: {start_point.address_string or 'Адрес не указан'}"
        logger.error(f"RouteBuilderTool: {msg}")
        return RouteDetails(status="error", error_message=msg).model_dump()

    # 2. Обработка точек мероприятий
    for i, event_location_obj in enumerate(
        event_points
    ):  # event_location_obj УЖЕ LocationModel
        if event_location_obj.lon is None or event_location_obj.lat is None:
            if event_location_obj.address_string:
                logger.debug(
                    f"RouteBuilderTool: Geocoding event_point {i+1} address: '{event_location_obj.address_string}'"
                )
                coords = await get_coords_from_address(
                    address=event_location_obj.address_string
                )
                if coords:
                    event_location_obj.lon, event_location_obj.lat = (
                        coords[0],
                        coords[1],
                    )  # Обновляем существующий объект
                    logger.info(
                        f"RouteBuilderTool: Geocoded event_point {i+1} to lon={coords[0]}, lat={coords[1]}"
                    )
                else:
                    msg = f"Не удалось получить координаты для точки мероприятия {i+1}: {event_location_obj.address_string}"
                    logger.error(f"RouteBuilderTool: {msg}")
                    return RouteDetails(status="error", error_message=msg).model_dump()
            else:
                msg = f"Точка мероприятия {i+1} не содержит ни координат, ни адреса для геокодирования."
                logger.error(f"RouteBuilderTool: {msg}")
                return RouteDetails(status="error", error_message=msg).model_dump()

        if event_location_obj.lon is not None and event_location_obj.lat is not None:
            all_points_for_routing.append(
                {"lon": event_location_obj.lon, "lat": event_location_obj.lat}
            )
        else:
            msg = f"Не удалось определить координаты для точки мероприятия {i+1}: {event_location_obj.address_string or 'Адрес не указан'}"
            logger.error(f"RouteBuilderTool: {msg}")
            return RouteDetails(status="error", error_message=msg).model_dump()

    # 3. Построение маршрута
    if len(all_points_for_routing) < 2:
        if len(event_points) == 0:
            logger.info(
                "RouteBuilderTool: Only one point (start_point is the event), no route needed."
            )
            return RouteDetails(
                status="success",
                duration_seconds=0,
                duration_text="0 мин",
                distance_meters=0,
                distance_text="0 км",
                error_message="Маршрут не требуется, так как указана только одна точка (место назначения).",
            ).model_dump()
        else:
            msg = "Для построения маршрута необходимо как минимум две валидные точки с координатами."
            logger.warning(
                f"RouteBuilderTool: {msg} (got {len(all_points_for_routing)})"
            )
            return RouteDetails(status="error", error_message=msg).model_dump()

    logger.info(
        f"RouteBuilderTool: Building route for {len(all_points_for_routing)} points with transport '{transport_type}'."
    )

    try:
        route_data = await get_route(
            points=all_points_for_routing, transport=transport_type
        )
        if not route_data:
            logger.error("RouteBuilderTool: get_route returned None.")
            return RouteDetails(
                status="error", error_message="Сервис маршрутов не вернул данные."
            ).model_dump()

        # route_data это уже словарь от get_route, который должен соответствовать RouteDetails
        route_details_obj = RouteDetails(**route_data)
        logger.info(f"RouteBuilderTool: Route build status: {route_details_obj.status}")
        return route_details_obj.model_dump()

    except Exception as e:
        logger.error(
            f"RouteBuilderTool: Unexpected error during route building: {e}",
            exc_info=True,
        )
        return RouteDetails(
            status="error",
            error_message=f"Неожиданная ошибка при построении маршрута: {e}",
        ).model_dump()
