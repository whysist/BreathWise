"""Small display helpers shared by the UI and the assistant."""
from __future__ import annotations

from datetime import datetime

NOT_AVAILABLE = "Not available"


def format_value(value, unit: str) -> str:
    """'90.8 μg/m³', or 'Not available' when the value is missing."""
    if value is None:
        return NOT_AVAILABLE
    return f"{value:.1f} {unit}"


def format_time(iso_time: str | None) -> str:
    """'Wed 18 Nov, 14:00' from a local ISO time, or 'Not available'."""
    try:
        return datetime.fromisoformat(iso_time).strftime("%a %d %b, %H:%M")
    except (TypeError, ValueError):
        return NOT_AVAILABLE


def format_retrieved(iso_utc: str | None) -> str:
    """'18 Nov 2026, 04:55 UTC' from the stored retrieval stamp."""
    try:
        return datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").strftime("%d %b %Y, %H:%M UTC")
    except (TypeError, ValueError):
        return NOT_AVAILABLE


def format_metric(key: str, value, unit: str) -> str:
    """Format a forecast field: AQI as a whole number, concentrations with their unit."""
    if value is None:
        return NOT_AVAILABLE
    if key == "us_aqi":
        return f"{value:.0f}"
    return f"{value:.1f} {unit}".strip()


# US EPA AQI scale: upper bound, label, badge colour, standard EPA hex colour, short description,
# and a plain-language "what to do" line. The category is always shown as text, never by colour alone.
_AQI_BANDS = [
    (50, "Good", "green", "#00A651", "Air quality is considered satisfactory.",
     "A good time to open the windows and let fresh air in."),
    (100, "Moderate", "orange", "#D9A400",
     "Acceptable for most people; a few unusually sensitive people may notice effects.",
     "Fine for most people. Anyone unusually sensitive may want to take it easy outdoors."),
    (150, "Unhealthy for sensitive groups", "orange", "#F07F00",
     "Sensitive groups, such as children, older adults and people with heart or lung conditions, may be affected.",
     "Children, older adults and sensitive people should cut down long or hard outdoor activity."),
    (200, "Unhealthy", "red", "#E0262B", "Everyone may begin to be affected; sensitive groups more seriously.",
     "Keep windows closed and cut down long or hard outdoor activity for everyone."),
    (300, "Very unhealthy", "violet", "#8F3F97", "Health-alert level for everyone.",
     "Stay indoors where you can, keep windows shut, and run a purifier if you have one."),
    (float("inf"), "Hazardous", "red", "#7E0023", "Emergency-level conditions.",
     "Stay indoors where you can and follow local public-health advisories."),
]


def aqi_category(value) -> dict | None:
    """US AQI category for a value, or None when the value is missing."""
    if value is None:
        return None
    for upper, label, color, hex_color, note, advice in _AQI_BANDS:
        if value <= upper:
            return {"label": label, "color": color, "hex": hex_color, "note": note, "advice": advice}
    return None
