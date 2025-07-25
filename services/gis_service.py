# gis.py

import aiohttp
import logging
import asyncio
from typing import Optional, List, Dict, Any
import json
import re

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    print(
        "ОШИБКА: Библиотека Pydantic не установлена. Пожалуйста, установите ее: pip install pydantic"
    )
    raise

# --- Настройки ---
# Пытаемся импортировать реальные настройки
try:
    from dotenv import load_dotenv

    load_dotenv()  # Загружаем .env сразу же
    from config.settings import settings
except (ImportError, ModuleNotFoundError) as e:
    logger_init_gis = logging.getLogger(__name__ + "_init_gis")
    logger_init_gis.warning(
        f"Не удалось импортировать settings из config.settings или dotenv: {str(e)}. "
        "Будет использована заглушка для GIS."
    )

    class PlaceholderGisSettingsFallback:
        GIS_API_KEY: Optional[str] = None
        LOG_LEVEL: str = "INFO"

        def __init__(self):
            try:
                from dotenv import load_dotenv

                load_dotenv()  # Попытаемся загрузить .env ещё раз
            except ImportError:
                logger_init_gis.warning(
                    "python-dotenv не установлен. Будет использовано os.getenv."
                )

            import os

            self.GIS_API_KEY = os.getenv("GIS_API_KEY")
            if not self.GIS_API_KEY:
                logger_init_gis.error(
                    "КРИТИЧЕСКАЯ ОШИБКА: GIS_API_KEY не установлен ни в .env, ни через config.settings."
                )

    settings = PlaceholderGisSettingsFallback()

GIS_API_KEY = "e3e42d53-d91e-44f5-9454-15b48a11e595"
GIS_API_BASE_URL = "https://catalog.api.2gis.com/3.0"
ROUTING_API_BASE_URL = "https://routing.api.2gis.com/routing/7.0.0/global"

logger = logging.getLogger(__name__)

# --- Модели данных ---


class GeocodingResult(BaseModel):
    coords: Optional[List[float]] = None
    match_level: str = "not_found"
    full_address_name_gis: Optional[str] = None
    is_precise_enough: bool = False
    error_message: Optional[str] = None


class ParkInfo(BaseModel):
    id_gis: str
    name: str
    full_name: Optional[str] = None
    address: Optional[str] = None
    coords: Optional[List[float]] = None
    schedule_str: str = "Время работы не указано"
    type_gis: Optional[str] = None
    rubrics_gis: Optional[List[Dict[str, Any]]] = None


class FoodPlaceInfo(BaseModel):
    id_gis: str
    name: str
    full_name: Optional[str] = None
    address: Optional[str] = None
    coords: Optional[List[float]] = None
    schedule_str: str = "Время работы не указано"
    rating_str: str = "Рейтинг не указан"
    avg_bill_str: str = "Средний чек не указан"
    contacts_str: str = "Контакты не указаны"
    type_gis: Optional[str] = None
    rubrics_gis: Optional[List[Dict[str, Any]]] = None
    raw_attribute_groups: Optional[List[Dict[str, Any]]] = None
    raw_adverts: Optional[List[Dict[str, Any]]] = None
    raw_contact_groups: Optional[List[Dict[str, Any]]] = None
    raw_reviews: Optional[Dict[str, Any]] = None


# --- Константы для парсинга времени (общие или специфичные для еды) ---
API_DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYS_RUS_SHORT_MAP = {
    "Mon": "Пн",
    "Tue": "Вт",
    "Wed": "Ср",
    "Thu": "Чт",
    "Fri": "Пт",
    "Sat": "Сб",
    "Sun": "Вс",
}

# --- Вспомогательные функции ---


def _parse_schedule_to_str_general(
    schedule_data: Optional[Dict[str, Any]], detailed_format: bool = False
) -> str:
    if not schedule_data:
        return "Время работы не указано"

    if detailed_format:
        intervals = []
        for day_code in API_DAYS_ORDER:
            day_info = schedule_data.get(day_code)
            if (
                day_info
                and isinstance(day_info, dict)
                and "working_hours" in day_info
                and isinstance(day_info["working_hours"], list)
                and len(day_info["working_hours"]) > 0
            ):
                wh = day_info["working_hours"][0]
                from_time, to_time = wh.get("from"), wh.get("to")
                if from_time and to_time:
                    intervals.append((day_code, f"{from_time}–{to_time}"))

        if not intervals:
            comment = schedule_data.get("comment")
            return (
                comment.strip()
                if comment and isinstance(comment, str) and comment.strip()
                else "Время работы не указано"
            )

        grouped, prev_time, current_group = [], None, []
        for day_code, time_str in intervals:
            if time_str != prev_time:
                if current_group:
                    grouped.append((current_group, prev_time))
                current_group, prev_time = [day_code], time_str
            else:
                current_group.append(day_code)
        if current_group:
            grouped.append((current_group, prev_time))

        def format_days(days_api_codes: List[str]) -> str:
            if not days_api_codes:
                return ""
            if len(days_api_codes) == 1:
                return DAYS_RUS_SHORT_MAP[days_api_codes[0]]
            indices = [API_DAYS_ORDER.index(d) for d in days_api_codes]
            is_sequential = all(
                indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)
            )
            return (
                f"{DAYS_RUS_SHORT_MAP[days_api_codes[0]]}–{DAYS_RUS_SHORT_MAP[days_api_codes[-1]]}"
                if is_sequential
                else ", ".join(DAYS_RUS_SHORT_MAP[d] for d in days_api_codes)
            )

        schedule_strs = [
            f"{format_days(days)}  {time_str}" for days, time_str in grouped
        ]
        return "\n".join(schedule_strs) if schedule_strs else "Время работы не указано"
    else:
        comment = schedule_data.get("comment")
        if comment and isinstance(comment, str) and comment.strip():
            if "круглосуточно" in comment.lower() or "24 часа" in comment.lower():
                return "Круглосуточно"
            return comment.strip()
        return "Время работы требует уточнения"


async def _geocode_city_to_coords(
    session: aiohttp.ClientSession, city_name: str, purpose: str = "поиска"
) -> Optional[List[float]]:
    if not settings.GIS_API_KEY:  # Проверяем актуальный settings
        logger.error(
            f"GIS_API_KEY не установлен для геокодирования города для {purpose}."
        )
        return None
    url = f"{GIS_API_BASE_URL}/items/geocode"
    params = {
        "q": city_name,
        "fields": "items.point",
        "key": settings.GIS_API_KEY,  # Используем актуальный settings
        "type": "adm_div.city",
    }
    logger.info(f"Геокодирование города '{city_name}' для {purpose}...")
    try:
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=7)
        ) as response:
            if response.status == 200:
                data = await response.json()
                logger.debug(f"Ответ геокодера для '{city_name}': {str(data)[:300]}")
                items = data.get("result", {}).get("items")
                if items and len(items) > 0 and items[0].get("point"):
                    point = items[0]["point"]
                    logger.info(
                        f"Координаты для '{city_name}': {point['lon']}, {point['lat']}"
                    )
                    return [point["lon"], point["lat"]]
                logger.warning(
                    f"Не удалось геокодировать город '{city_name}'. Ответ API: {str(data)[:300]}"
                )
            else:
                logger.error(
                    f"Ошибка геокодирования '{city_name}': {response.status}. Ответ: {await response.text()[:200]}"
                )
    except Exception as e:
        logger.error(f"Исключение при геокодировании '{city_name}': {e}", exc_info=True)
    return None


# --- Вспомогательные функции для Food Places ---


def _extract_rating(reviews_data: Optional[Dict[str, Any]]) -> str:
    if reviews_data and reviews_data.get("rating") is not None:
        try:
            rating_val = float(reviews_data["rating"])
            count_str = reviews_data.get(
                "review_count", reviews_data.get("general_review_count", "0")
            )
            count = 0
            try:
                count = int(count_str)
            except:
                pass
            return (
                f"{rating_val:.1f}/5 ({count} отзывов)"
                if count > 0
                else f"{rating_val:.1f}/5"
            )
        except (ValueError, TypeError):
            return "Рейтинг не указан (ошибка формата)"
    return "Рейтинг не указан"


def _extract_avg_bill(
    attribute_groups_data: Optional[List[Dict[str, Any]]],
    adverts_data: Optional[List[Dict[str, Any]]],
) -> str:
    if attribute_groups_data:
        for group in attribute_groups_data:
            if not isinstance(group, dict) or "attributes" not in group:
                continue
            for attr in group.get("attributes", []):
                if not isinstance(attr, dict):
                    continue
                attr_name_original, attr_name_lower = (
                    attr.get("name", ""),
                    attr.get("name", "").lower(),
                )
                attr_tag_lower, attr_value = attr.get("tag", "").lower(), attr.get(
                    "value"
                )
                if attr_tag_lower == "food_service_avg_price":
                    if "чек " in attr_name_lower:
                        match = re.search(
                            r"чек\s*([\d\s–~-]+)\s*₽?",
                            attr_name_original,
                            re.IGNORECASE,
                        )  # Добавил IGNORECASE
                        if match:
                            return match.group(1).strip().replace(" ", "")
                    elif attr_value:
                        return str(attr_value).strip()
                if "средний чек" in attr_name_lower or (
                    "чек " in attr_name_lower and "кассовый" not in attr_name_lower
                ):
                    if attr_value:
                        return str(attr_value).strip()
                    bill_part_from_name = (
                        attr_name_lower.split("средний чек:", 1)[-1]
                        if "средний чек:" in attr_name_lower
                        else (
                            attr_name_lower.split("чек ", 1)[-1]
                            if "чек " in attr_name_lower
                            else attr_name_lower
                        )
                    )
                    bill_part_from_name = (
                        bill_part_from_name.replace("руб.", "").replace("₽", "").strip()
                    )
                    match = re.search(r"([\d\s–~-]+)", bill_part_from_name)
                    if match:
                        found_val = match.group(1).strip().replace(" ", "")
                        if found_val and any(char.isdigit() for char in found_val):
                            return found_val
                if attr_tag_lower == "food_average_bill" and attr_value:
                    return str(attr_value).strip()
    if adverts_data:
        for advert in adverts_data:
            if isinstance(advert, dict) and advert.get("text"):
                text_lower = advert.get("text").lower()
                if "средний чек" in text_lower:
                    parts = text_lower.split("средний чек")
                    if len(parts) > 1:
                        return (
                            parts[1]
                            .strip()
                            .split(maxsplit=1)[0]
                            .replace(":", "")
                            .strip()
                        )
    return "Средний чек не указан"


def _extract_contacts(contact_groups_data: Optional[List[Dict[str, Any]]]) -> str:
    if not contact_groups_data:
        return "Контакты не указаны"
    phones, websites, emails, social_nets_map = [], [], [], {}
    for group in contact_groups_data:
        if not isinstance(group, dict) or "contacts" not in group:
            continue
        for item in group.get("contacts", []):
            if not isinstance(item, dict):
                continue
            ctype, val, url, txt = (
                item.get("type"),
                item.get("value"),
                item.get("url"),
                item.get("text"),
            )
            if ctype == "phone" and val:
                phones.append(str(val))
            elif ctype == "email" and val:
                emails.append(str(val))
            elif ctype == "website" and url:
                disp_url = (
                    f"{txt} ({url})"
                    if txt and txt.lower().strip() != url.lower().strip()
                    else url
                )
                is_soc = False
                if any(s in url for s in ["vk.com", "vkontakte."]):
                    social_nets_map.setdefault("VK", []).append(url)
                    is_soc = True
                elif "instagram.com" in url:
                    social_nets_map.setdefault("Instagram", []).append(url)
                    is_soc = True
                elif "facebook.com" in url:
                    social_nets_map.setdefault("Facebook", []).append(url)
                    is_soc = True
                elif any(s in url for s in ["t.me", "telegram.me"]):
                    social_nets_map.setdefault("Telegram", []).append(url)
                    is_soc = True
                if not is_soc:
                    websites.append(disp_url)
            elif val and not url and ("http" in val or "www." in val):
                websites.append(str(val))
    parts = []
    if phones:
        parts.append(f"Тел: {', '.join(sorted(list(set(phones))))}")
    if websites:
        parts.append(f"Сайты: {', '.join(sorted(list(set(websites))))}")
    if emails:
        parts.append(f"Email: {', '.join(sorted(list(set(emails))))}")
    social_out = [
        f"{n}: {', '.join(sorted(list(set(us))))}" for n, us in social_nets_map.items()
    ]
    if social_out:
        parts.append(f"Соцсети: {'; '.join(social_out)}")
    return "\n".join(parts) if parts else "Контакты не указаны"


# --- Основные функции API ---


async def get_geocoding_details(
    address: str, city: Optional[str] = None
) -> GeocodingResult:
    if not settings.GIS_API_KEY:  # Проверяем актуальный settings
        return GeocodingResult(
            match_level="error", error_message="API ключ не настроен"
        )
    url = f"{GIS_API_BASE_URL}/items"
    query_to_log = address
    params = {
        "q": address,
        "fields": "items.geometry.centroid,items.address_name,items.type,items.subtype,items.name,items.full_name",
        "page_size": 5,
        "key": settings.GIS_API_KEY,  # Используем актуальный settings
    }
    if city and city.lower() not in address.lower():
        params["q"] = f"{city}, {address}"
        query_to_log = params["q"]

    logger.info(f"2GIS Geocoding: Запрос для '{query_to_log}'")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                logger.debug(
                    f"2GIS Geocoding Request: {response.url} Status: {response.status}"
                )
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"2GIS Geocoding Raw response: {str(data)[:500]}")
                    res_data = data.get("result", {})
                    items = res_data.get("items")

                    if items:
                        item = items[0]
                        p_str = item.get("geometry", {}).get("centroid")
                        coords = None
                        if p_str:
                            try:
                                lon, lat = map(
                                    float,
                                    p_str.replace("POINT(", "")
                                    .replace(")", "")
                                    .split(),
                                )
                                coords = [lon, lat]
                            except:
                                logger.warning(f"Ошибка парсинга centroid: {p_str}")

                        match_lvl = item.get("type", "unknown")
                        if item.get("subtype"):
                            match_lvl = item.get("subtype")
                        is_precise = match_lvl == "building" and bool(coords)
                        determined_match_level = "not_found"
                        if item.get("type") == "building":
                            determined_match_level = "building"
                        elif item.get("type") == "street":
                            determined_match_level = "street"
                        elif item.get("type") == "adm_div":
                            if item.get("subtype") in [
                                "city",
                                "settlement",
                                "district",
                                "living_area",
                                "place",
                                "station_platform",
                            ]:
                                determined_match_level = item.get("subtype")
                            else:
                                determined_match_level = "adm_div_other"
                        elif item.get("type"):
                            determined_match_level = item.get("type")

                        return GeocodingResult(
                            coords=coords,
                            match_level=determined_match_level,
                            full_address_name_gis=item.get(
                                "full_name", item.get("address_name", item.get("name"))
                            ),
                            is_precise_enough=is_precise,
                        )
                    else:
                        return GeocodingResult(
                            match_level="not_found",
                            error_message="No items in response",
                        )
                else:
                    err_txt = await response.text()
                    logger.error(
                        f"2GIS Geocoding API Error: {response.status} for {response.url}. Resp: {err_txt[:200]}"
                    )
                    return GeocodingResult(
                        match_level="error",
                        error_message=f"API Error {response.status}: {err_txt[:100]}",
                    )
    except Exception as e:
        logger.error(f"2GIS Geocoding Unexpected error: {e}", exc_info=True)
        return GeocodingResult(match_level="error", error_message=str(e))


async def get_coords_from_address(
    address: str, city: Optional[str] = None
) -> Optional[List[float]]:
    logger.info(f"Запрос координат для: '{address}' (город: {city})")
    result = await get_geocoding_details(address, city)
    if result.coords and result.is_precise_enough:
        logger.info(f"Координаты найдены и точны: {result.coords}")
        return result.coords
    elif result.coords:
        logger.warning(
            f"Координаты найдены, но могут быть неточными (уровень: {result.match_level}). Координаты: {result.coords}"
        )
        return result.coords
    logger.warning(
        f"Координаты не найдены для '{address}'. Уровень: {result.match_level}, Ошибка: {result.error_message}"
    )
    return None


async def search_parks(
    original_query: str,
    city: Optional[str] = None,
    limit: int = 5,
    exclude_ids: Optional[List[str]] = None,  # Новый параметр
) -> List[ParkInfo]:
    if (
        not GIS_API_KEY
    ):  # Используем GIS_API_KEY напрямую, если он определен глобально в этом файле
        logger.error("GIS_API_KEY не установлен. Поиск парков невозможен.")
        return []

    url = f"{GIS_API_BASE_URL}/items"
    relevant_types_str = "attraction,adm_div.place,adm_div.living_area,adm_div.district_area,station_platform"
    api_q = original_query
    api_page_size = min(
        max(limit * 2, 5), 10
    )  # Немного увеличил page_size для большего выбора

    params = {
        "q": api_q,
        "type": relevant_types_str,
        "fields": "items.id,items.name,items.full_name,items.address_name,items.geometry.centroid,items.schedule,items.type,items.subtype,items.rubrics,items.adm_div",
        "page_size": api_page_size,
        "key": GIS_API_KEY,
        "locale": "ru_RU",
    }
    query_to_log_parts = [
        f"api_q='{params['q']}'",
        f"type='{relevant_types_str}' (orig_q='{original_query}')",
    ]

    async with aiohttp.ClientSession() as session:
        if city:
            city_coords_list = await _geocode_city_to_coords(
                session, city, purpose="поиска парков"
            )
            if city_coords_list:
                params["point"] = f"{city_coords_list[0]},{city_coords_list[1]}"
                params["radius"] = 20000
                query_to_log_parts.append(
                    f"(в районе {city} [{params['point']}], R={params['radius']}м)"
                )
            elif city.lower() not in params["q"].lower():
                params["q"] = f"{city}, {params['q']}"
                query_to_log_parts[0] = f"api_q='{params['q']}'"

        logger.info(
            f"Park Search: {', '.join(query_to_log_parts)}. URL: {url}?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
        )
        found_parks_list: List[ParkInfo] = []
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                resp_text = await response.text()
                logger.debug(
                    f"Park Search: Status {response.status} for URL: {response.url}"
                )
                if response.status == 200:
                    try:
                        data = json.loads(resp_text)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Park Search JSON Decode Error: {resp_text[:1000]}"
                        )
                        return []

                    items = data.get("result", {}).get("items")
                    if items:
                        logger.info(
                            f"Кандидатов от API для парков: {len(items)}. Фильтруем..."
                        )
                        for item_data in items:
                            item_id_gis = item_data.get("id")
                            if (
                                exclude_ids and item_id_gis in exclude_ids
                            ):  # Проверка на исключение
                                logger.debug(
                                    f"Skipping park '{item_data.get('name')}' (ID: {item_id_gis}) as it is in exclude_ids."
                                )
                                continue

                            item_type = item_data.get("type")
                            item_subtype = item_data.get("subtype")
                            item_name_orig = item_data.get("name", "")
                            item_name_lower = item_name_orig.lower()
                            item_rubrics = item_data.get("rubrics", [])
                            is_park_like = False
                            park_keywords = [
                                "парк",
                                "сквер",
                                "сад",
                                "аллея",
                                "бульвар",
                                "лесопарк",
                                "дендрарий",
                                "территория отдыха",
                            ]
                            if any(kw in item_name_lower for kw in park_keywords):
                                is_park_like = True

                            park_rubric_ids = ["36", "37", "168"]  # Добавил 168 из лога
                            if item_rubrics and any(
                                str(r.get("id")) in park_rubric_ids
                                for r in item_rubrics
                            ):
                                is_park_like = True

                            if (
                                original_query.lower() in item_name_lower
                                and len(original_query) > 3
                            ):
                                is_park_like = True

                            # Уточняющие фильтры
                            if (
                                "парк" == original_query.lower()
                                and "парк" not in item_name_lower
                                and "сквер" in item_name_lower
                            ):
                                is_park_like = False
                            if (
                                "сквер" == original_query.lower()
                                and "сквер" not in item_name_lower
                                and "парк" in item_name_lower
                            ):
                                is_park_like = False
                            if (
                                "кладбище" in item_name_lower
                                or "промышленный парк" in item_name_lower
                            ):
                                is_park_like = False

                            if not is_park_like:
                                logger.debug(
                                    f"'{item_name_orig}' (тип {item_type}/{item_subtype}) не прошел фильтр парков."
                                )
                                continue

                            logger.info(
                                f"'{item_name_orig}' (тип {item_type}/{item_subtype}) прошел как парк/сквер."
                            )
                            coords = None
                            if p_str_park := item_data.get("geometry", {}).get(
                                "centroid"
                            ):
                                try:
                                    lon_p, lat_p = map(
                                        float,
                                        p_str_park.replace("POINT(", "")
                                        .replace(")", "")
                                        .split(),
                                    )
                                    coords = [lon_p, lat_p]
                                except ValueError:
                                    logger.warning(
                                        f"Ошибка парсинга координат парка: {p_str_park}"
                                    )

                            found_parks_list.append(
                                ParkInfo(
                                    id_gis=item_id_gis,
                                    name=item_name_orig,
                                    full_name=item_data.get("full_name"),
                                    address=item_data.get("address_name"),
                                    coords=coords,
                                    schedule_str=_parse_schedule_to_str_general(
                                        item_data.get("schedule"), detailed_format=False
                                    ),
                                    type_gis=(
                                        item_subtype if item_subtype else item_type
                                    ),
                                    rubrics_gis=item_rubrics,
                                )
                            )
                            if len(found_parks_list) >= limit:
                                break
                    else:
                        logger.warning(
                            f"API не вернул 'items' для парков. Ответ: {resp_text[:1000]}"
                        )
                elif response.status == 400:
                    logger.error(
                        f"Park Search API Error 400: {await response.text(encoding='utf-8')[:1000]}"
                    )
                elif response.status == 404:
                    logger.info("Park Search: API не нашел результатов (404).")
                else:
                    logger.error(
                        f"Park Search API Error {response.status}: {await response.text(encoding='utf-8')[:500]}"
                    )
            return found_parks_list[:limit]
        except Exception as e:
            logger.error(f"Park Search Exception: {e}", exc_info=True)
            return []


async def search_food_places(
    original_query: str,
    city: Optional[str] = None,
    limit: int = 5,
    exclude_ids: Optional[List[str]] = None,
) -> List[FoodPlaceInfo]:
    if not GIS_API_KEY:
        logger.error("GIS_API_KEY не установлен.")
        return []

    url, query_lower = f"{GIS_API_BASE_URL}/items", original_query.lower()
    is_very_general_food_query = query_lower in [
        "покушать",
        "еда",
        "где поесть",
        "кафе ресторан",
        "где покушать",
        "где покушать можно",
        "поесть где нибудь",
        "заведения питания",
    ]

    target_rubric_ids, is_specific_type_query = [], False
    if "ресторан" in query_lower:
        target_rubric_ids.append("164")
        is_specific_type_query = True
    if query_lower == "кафе" or (
        "кафе" in query_lower and "автокафе" not in query_lower
    ):
        target_rubric_ids.append("161")
        is_specific_type_query = True
    if "бар" in query_lower:
        target_rubric_ids.append("159")
        is_specific_type_query = True
    if (
        "автокафе" in query_lower
        or "еда на колесах" in query_lower
        or query_lower == "быстрое питание"
    ):
        target_rubric_ids.append("165")
        is_specific_type_query = True
    if "столовая" in query_lower:
        target_rubric_ids.append("160")
        is_specific_type_query = True
    if "кофейня" in query_lower or "кофе" == query_lower:
        target_rubric_ids.append("163")
        is_specific_type_query = True

    if not is_specific_type_query and is_very_general_food_query:
        target_rubric_ids.extend(["164", "161", "159", "165", "160", "162", "163"])
        logger.info(
            f"Food Search: Общий запрос '{original_query}', ищем по широким рубрикам еды."
        )

    unique_target_rubrics = sorted(list(set(target_rubric_ids)))

    fields_to_request = "items.id,items.name,items.full_name,items.address_name,items.rubrics,items.type,items.geometry.centroid,items.schedule,items.reviews,items.attribute_groups,items.adverts,items.contact_groups"
    page_size = min(max(limit * 2, 5), 10)

    params: Dict[str, Any] = {
        "q": (
            original_query
            if not is_very_general_food_query
            else city if city else "еда"
        ),
        "type": "branch",
        "fields": fields_to_request,
        "page_size": page_size,
        "key": GIS_API_KEY,
        "locale": "ru_RU",
    }
    if unique_target_rubrics:
        params["rubric_id"] = ",".join(unique_target_rubrics)

    log_parts = [f"q='{params['q']}'", f"type='{params['type']}'"]
    if params.get("rubric_id"):
        log_parts.append(f"api_rubric_id='{params['rubric_id']}'")

    async with aiohttp.ClientSession() as session:
        if city:
            coords_city = await _geocode_city_to_coords(
                session, city, purpose="поиска заведений питания"
            )
            if coords_city:
                params["point"], params["radius"] = (
                    f"{coords_city[0]},{coords_city[1]}",
                    20000,
                )
                log_parts.append(f"point='{params['point']}' R={params['radius']}")
            elif (
                city.lower() not in params["q"].lower()
                and not is_very_general_food_query
            ):
                params["q"] = f"{city}, {params['q']}"
                log_parts[0] = f"q='{params['q']}'"

        logger.info(
            f"Food Search: {', '.join(log_parts)} (orig_q='{original_query}'). URL: {url}?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
        )
        result_list_food: List[FoodPlaceInfo] = []
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                resp_text_food = await response.text()
                if response.status == 200:
                    try:
                        data_food = json.loads(resp_text_food)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Food Search JSON Decode Error: {resp_text_food[:1000]}"
                        )
                        return []

                    items_food = data_food.get("result", {}).get("items")
                    if items_food:
                        logger.info(
                            f"Кандидатов от API для еды: {len(items_food)}. Фильтруем для '{original_query}'..."
                        )
                        for item_data_food in items_food:
                            item_id_gis_food = item_data_food.get("id")
                            if exclude_ids and item_id_gis_food in exclude_ids:
                                logger.debug(
                                    f"Skipping food place '{item_data_food.get('name')}' (ID: {item_id_gis_food}) as it is in exclude_ids."
                                )
                                continue

                            if item_data_food.get("type") != "branch":
                                continue
                            name_orig_food, name_lower_food = (
                                item_data_food.get("name", ""),
                                item_data_food.get("name", "").lower(),
                            )
                            rubrics_api_food = item_data_food.get("rubrics", [])
                            passes_filter_food = False

                            if is_very_general_food_query:
                                passes_filter_food = True
                            elif is_specific_type_query:
                                if query_lower in name_lower_food:
                                    passes_filter_food = True
                                elif (
                                    unique_target_rubrics
                                    and rubrics_api_food
                                    and any(
                                        str(r.get("id")) in unique_target_rubrics
                                        for r in rubrics_api_food
                                    )
                                ):
                                    passes_filter_food = True
                            else:
                                if query_lower in name_lower_food:
                                    passes_filter_food = True

                            if not passes_filter_food:
                                logger.debug(
                                    f"'{name_orig_food}' не прошел фильтрацию для еды по запросу '{original_query}'."
                                )
                                continue

                            logger.info(
                                f"'{name_orig_food}' (тип {item_data_food.get('type')}) прошел фильтры для еды."
                            )
                            item_coords_food = None
                            if p_str_food := item_data_food.get("geometry", {}).get(
                                "centroid"
                            ):
                                try:
                                    lon_f, lat_f = map(
                                        float,
                                        p_str_food.replace("POINT(", "")
                                        .replace(")", "")
                                        .split(),
                                    )
                                    item_coords_food = [lon_f, lat_f]
                                except ValueError:
                                    logger.warning(
                                        f"Парсинг координат еды: {p_str_food}"
                                    )

                            place_food = FoodPlaceInfo(
                                id_gis=item_id_gis_food,
                                name=name_orig_food,
                                full_name=item_data_food.get("full_name"),
                                address=item_data_food.get("address_name"),
                                coords=item_coords_food,
                                schedule_str=_parse_schedule_to_str_general(
                                    item_data_food.get("schedule"), detailed_format=True
                                ),
                                rating_str=_extract_rating(
                                    item_data_food.get("reviews")
                                ),
                                avg_bill_str=_extract_avg_bill(
                                    item_data_food.get("attribute_groups"),
                                    item_data_food.get("adverts"),
                                ),
                                contacts_str=_extract_contacts(
                                    item_data_food.get("contact_groups")
                                ),
                                type_gis=item_data_food.get("type"),
                                rubrics_gis=rubrics_api_food,
                                raw_attribute_groups=item_data_food.get(
                                    "attribute_groups"
                                ),
                                raw_adverts=item_data_food.get("adverts"),
                                raw_contact_groups=item_data_food.get("contact_groups"),
                                raw_reviews=item_data_food.get("reviews"),
                            )
                            result_list_food.append(place_food)
                            if len(result_list_food) >= limit:
                                break
                    else:
                        logger.warning(
                            f"API не вернул 'items' для еды. Ответ: {resp_text_food[:1000]}"
                        )
                elif response.status == 400:
                    logger.error(
                        f"Food Search API Error 400. {await response.text(encoding='utf-8')[:1000]}"
                    )
                elif response.status == 401:
                    logger.error(
                        "Food Search API Error 401 (Unauthorized). Проверьте GIS_API_KEY."
                    )
                elif response.status == 404:
                    logger.info(
                        f"Food Search API не нашел результатов (404) по URL: {response.url}."
                    )
                else:
                    logger.error(
                        f"Food Search API Error {response.status}: {await response.text(encoding='utf-8')[:500]}"
                    )
            return result_list_food[:limit]
        except asyncio.TimeoutError:
            logger.error(f"Food Search Timeout: {url} params: {params}")
            return []
        except aiohttp.ClientError as e_client_food:
            logger.error(
                f"Food Search aiohttp.ClientError: {e_client_food}", exc_info=True
            )
            return []
        except Exception as e_food:
            logger.error(f"Food Search Exception: {e_food}", exc_info=True)
            return []


async def get_route(
    points: List[Dict[str, Any]], transport: str = "driving"
) -> Dict[str, Any]:
    if not settings.GIS_API_KEY:
        return {"status": "error", "message": "API ключ не настроен"}
    url = f"{ROUTING_API_BASE_URL}?key={settings.GIS_API_KEY}"
    api_points = []
    for p_idx, p_val in enumerate(points):
        if not (isinstance(p_val, dict) and "lon" in p_val and "lat" in p_val):
            msg = f"Неверный формат точки {p_idx+1}: {p_val}"
            logger.error(f"2GIS Routing: {msg}")
            return {"status": "error", "message": msg}
        api_points.append(
            {
                "lon": p_val["lon"],
                "lat": p_val["lat"],
                "type": ("pedo" if transport == "walking" else "auto"),
            }
        )

    payload = {"transport": transport, "points": api_points}
    logger.info(
        f"2GIS Routing: Запрос. Транспорт: {transport}, Точек: {len(api_points)}"
    )
    logger.debug(f"2GIS Routing: URL: {url}, Payload: {str(payload)[:300]}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                resp_text_content = await response.text(
                    encoding="utf-8"
                )  # Сначала получаем текст
                logger.debug(
                    f"2GIS Routing: Status: {response.status}, Body(500): {resp_text_content[:500]}"
                )
                if response.status == 200:
                    try:
                        data = json.loads(resp_text_content)
                    except json.JSONDecodeError:  # Исправлено на json.JSONDecodeError
                        logger.error(
                            f"2GIS Routing: API вернул не JSON (status 200): {resp_text_content[:500]}"
                        )
                        return {
                            "status": "error",
                            "message": "Некорректный формат ответа API маршрутов.",
                        }

                    res_list = data.get("result")
                    if res_list and len(res_list) > 0:
                        item = res_list[0]
                        dur_sec, dist_m = item.get("total_duration"), item.get(
                            "total_distance"
                        )
                        if dur_sec is not None and dist_m is not None:
                            return {
                                "status": "success",
                                "duration_seconds": dur_sec,
                                "distance_meters": dist_m,
                                "duration_text": f"~{round(dur_sec / 60)} мин",
                                "distance_text": f"~{round(dist_m / 1000, 1)} км",
                            }
                        else:
                            # Случай, когда result есть, но нет нужных полей (маловероятно для 2GIS, но все же)
                            logger.warning(
                                f"2GIS Routing: Success status but missing duration/distance. Response: {str(data)[:300]}"
                            )
                            return {
                                "status": "error",
                                "message": "Отсутствуют детали маршрута в успешном ответе.",
                            }
                    elif err_info := data.get(
                        "error"
                    ):  # Если API явно вернул ошибку в JSON
                        logger.warning(f"2GIS Routing: API returned error: {err_info}")
                        return {
                            "status": "api_error",  # Более специфичный статус
                            "message": err_info.get(
                                "message", "Ошибка API построения маршрутов."
                            ),
                            "error_details": err_info,
                        }
                    else:  # Успешный статус, но неожиданная структура JSON
                        logger.warning(
                            f"2GIS Routing: Unexpected JSON structure. Response: {str(data)[:300]}"
                        )
                        return {
                            "status": "error",
                            "message": "Неожиданный формат ответа от API маршрутов.",
                        }
                else:  # HTTP ошибка (не 200)
                    # resp_text_content уже получен выше
                    logger.error(
                        f"2GIS Routing: HTTP error {response.status}. Body: {resp_text_content[:200]}"
                    )
                    # Пытаемся распарсить ошибку, если она в JSON
                    try:
                        error_data = json.loads(resp_text_content)
                        api_message = error_data.get("message", resp_text_content[:100])
                        error_type = error_data.get("type")
                        return {
                            "status": "http_error",
                            "code": response.status,
                            "message": f"Ошибка API {response.status}: {api_message}",
                            "error_type_api": error_type,
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "http_error",
                            "code": response.status,
                            "message": f"Ошибка API {response.status}: {resp_text_content[:100]}",
                        }
    except asyncio.TimeoutError:
        logger.error(
            f"2GIS Routing: Timeout for URL {url} with payload {str(payload)[:300]}"
        )
        return {
            "status": "timeout_error",
            "message": "Превышено время ожидания от сервиса маршрутов.",
        }
    except aiohttp.ClientError as e_client:
        logger.error(f"2GIS Routing: ClientError - {e_client}", exc_info=True)
        return {
            "status": "client_error",
            "message": f"Ошибка соединения с сервисом маршрутов: {e_client}",
        }
    except Exception as e_unknown:  # Общий обработчик других исключений
        logger.error(f"2GIS Routing: Unexpected exception - {e_unknown}", exc_info=True)
        return {
            "status": "unknown_error",
            "message": f"Неизвестная ошибка при построении маршрута: {str(e_unknown)}",
        }
