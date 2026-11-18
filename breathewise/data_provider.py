"""Open-Meteo geocoding and air-quality access.

All HTTP lives here. Only two fixed HTTPS hosts are ever called, and nothing
user-supplied is used as a URL. Failures raise ``DataUnavailable`` with a
message that is safe to show to the user; nothing is ever invented.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
PROVIDER = "Open-Meteo Air Quality API (CAMS model data)"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "air_quality_sample.json"

POLLUTANTS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "Nitrogen dioxide (NO₂)",
    "ozone": "Ozone (O₃)",
}
# The provider's own US AQI index (calculated by Open-Meteo from modelled concentrations).
# It is NOT India's National AQI and is never converted to it.
AQI_KEY = "us_aqi"
FIELDS = {AQI_KEY: "US AQI", **POLLUTANTS}  # everything requested from the API
DEFAULT_UNIT = "μg/m³"

REQUEST_TIMEOUT_S = 10
MAX_QUERY_LEN = 100
MAX_FORECAST_HOURS = 48
MAX_PLAUSIBLE_VALUE = 10000.0  # µg/m³; anything above is treated as malformed
STALE_AFTER = timedelta(hours=6)
_USER_AGENT = "BreatheWise/1.0 (educational clean-air assistant)"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class DataUnavailable(Exception):
    """The provider could not give usable data. The message is user-safe."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _num(value) -> float | None:
    """Return a clean non-negative float, or None. Never guesses."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > MAX_PLAUSIBLE_VALUE:
        return None
    return value


def validate_coordinates(latitude, longitude) -> tuple[float, float]:
    """Return (lat, lon) as floats, or raise ValueError."""
    try:
        if isinstance(latitude, bool) or isinstance(longitude, bool):
            raise TypeError
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be numbers.") from None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("Longitude must be between -180 and 180.")
    return lat, lon


def clean_text(value, max_len: int) -> str:
    """Strip control characters, collapse whitespace and cap length."""
    text = _CONTROL_CHARS.sub(" ", str(value))
    return " ".join(text.split())[:max_len]


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict) -> dict:
    """GET with a timeout, a status check and at most one retry."""
    last_problem = "The data service did not respond."
    for _attempt in range(2):
        try:
            response = requests.get(
                url, params=params, timeout=REQUEST_TIMEOUT_S,
                headers={"User-Agent": _USER_AGENT},
            )
        except (requests.Timeout, requests.ConnectionError):
            last_problem = "The data service could not be reached."
            continue
        except requests.RequestException:
            raise DataUnavailable("The data request failed.") from None
        if response.status_code >= 500 or response.status_code == 429:
            last_problem = "The data service is busy or temporarily down."
            continue
        if response.status_code != 200:
            raise DataUnavailable(f"The data service returned an error (HTTP {response.status_code}).")
        try:
            payload = response.json()
        except ValueError:
            raise DataUnavailable("The data service returned an unreadable response.") from None
        if not isinstance(payload, dict):
            raise DataUnavailable("The data service returned an unexpected response.")
        return payload
    raise DataUnavailable(last_problem)


# --------------------------------------------------------------------------- #
# Geocoding
# --------------------------------------------------------------------------- #
def search_places(query: str, count: int = 6) -> list[dict]:
    """Resolve a place name to candidate places. Raises ValueError on bad input."""
    query = clean_text(query or "", MAX_QUERY_LEN)
    if len(query) < 2:
        raise ValueError("Please enter at least 2 characters for the place name.")
    payload = _get_json(
        GEOCODING_URL,
        {"name": query, "count": max(1, min(int(count), 10)), "language": "en", "format": "json"},
    )
    places = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            lat, lon = validate_coordinates(item.get("latitude"), item.get("longitude"))
        except ValueError:
            continue  # skip results with unusable coordinates
        name = clean_text(item["name"], 80)
        region = clean_text(item.get("admin1") or "", 80)
        country = clean_text(item.get("country") or "", 80)
        if not name:
            continue
        parts = []
        for part in (name, region, country):
            if part and part not in parts:
                parts.append(part)
        places.append({
            "name": name, "region": region, "country": country,
            "latitude": lat, "longitude": lon, "label": ", ".join(parts),
        })
    return places


# --------------------------------------------------------------------------- #
# Air quality
# --------------------------------------------------------------------------- #
def parse_air_quality(payload: dict, now_utc: datetime | None = None, allow_past: bool = False) -> dict:
    """Validate an Open-Meteo air-quality payload and normalise it.

    Returns {"provider", "timezone", "units", "current": {"time", "values"},
    "hourly": [{"time", <pollutant>: float|None, ...}], "retrieved_at"}.
    ``hourly`` starts at the current hour and holds at most 48 hours.
    Raises DataUnavailable for missing, malformed or stale data.
    """
    if not isinstance(payload, dict):
        raise DataUnavailable("The data service returned an unexpected response.")
    hourly_raw = payload.get("hourly")
    if not isinstance(hourly_raw, dict) or not isinstance(hourly_raw.get("time"), list):
        raise DataUnavailable("The forecast is missing hourly data.")

    units_raw = payload.get("hourly_units") if isinstance(payload.get("hourly_units"), dict) else {}
    units = {key: (units_raw.get(key) if isinstance(units_raw.get(key), str) else DEFAULT_UNIT)
             if key != AQI_KEY else "" for key in FIELDS}

    # Anchor = the model's "current" hour; the forecast starts from there.
    current_raw = payload.get("current") if isinstance(payload.get("current"), dict) else {}
    current_time = _parse_time(current_raw.get("time"))
    offset = payload.get("utc_offset_seconds")
    now_utc = now_utc or datetime.now(timezone.utc)
    if current_time is not None and isinstance(offset, int) and not isinstance(offset, bool) and not allow_past:
        local_now = now_utc.replace(tzinfo=None) + timedelta(seconds=offset)
        if local_now - current_time > STALE_AFTER:
            raise DataUnavailable("The forecast data looks out of date, so it is not being shown.")
    anchor = current_time.replace(minute=0, second=0, microsecond=0) if current_time else None

    rows = []
    for i, raw_time in enumerate(hourly_raw["time"]):
        t = _parse_time(raw_time)
        if t is None:
            continue
        if anchor is not None and not allow_past and t < anchor:
            continue
        row = {"time": t.strftime("%Y-%m-%dT%H:%M")}
        for key in FIELDS:
            series = hourly_raw.get(key)
            row[key] = _num(series[i]) if isinstance(series, list) and i < len(series) else None
        rows.append(row)
    rows = rows[:MAX_FORECAST_HOURS]
    if not rows:
        raise DataUnavailable("The forecast has no usable time points.")
    if not any(row[key] is not None for row in rows for key in FIELDS):
        raise DataUnavailable("The forecast has no usable pollutant values.")

    return {
        "provider": PROVIDER,
        "timezone": clean_text(payload.get("timezone") or "", 60) or "local time",
        "units": units,
        "current": {
            "time": current_time.strftime("%Y-%m-%dT%H:%M") if current_time else None,
            "values": {key: _num(current_raw.get(key)) for key in FIELDS},
        },
        "hourly": rows,
        "retrieved_at": now_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "is_sample": False,
    }


def fetch_air_quality(latitude, longitude) -> dict:
    """Fetch and validate the forecast for a place. Raises ValueError/DataUnavailable."""
    lat, lon = validate_coordinates(latitude, longitude)
    names = ",".join(FIELDS)
    payload = _get_json(AIR_QUALITY_URL, {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "current": names, "hourly": names, "timezone": "auto", "forecast_days": 2,
    })
    return parse_air_quality(payload)


def load_sample() -> dict:
    """Load the bundled demo data. Always labelled as sample, never used silently."""
    try:
        payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        data = parse_air_quality(payload, allow_past=True)
    except (OSError, ValueError, DataUnavailable):
        raise DataUnavailable("The bundled sample data could not be loaded.") from None
    data.update(provider="Sample data — not live", retrieved_at=None, is_sample=True)
    return data
