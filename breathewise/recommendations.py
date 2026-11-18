"""Deterministic recommendation rules. Pure functions: same inputs, same output.

Nothing here calls the network or an LLM. The forecast is only used to say
which part of the supplied window is *relatively* lower or higher in the
model, and what the model shows at the times in the user's own routine.
It never labels a time "safe".
"""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta

CATEGORIES = ("Getting around", "At home", "Daily activities")

COMMUTE_MODES = {
    "car": "Private car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "metro_train": "Metro/train",
    "walk_cycle": "Walking/cycling",
    "wfh": "Work from home",
    "other": "Other",
}
EXERCISE_OPTIONS = {"outdoors": "Outdoors", "indoors": "Indoors", "none": "No regular routine"}
PURIFIER_OPTIONS = {"yes": "Yes", "no": "No", "unsure": "Not sure"}
EXERCISE_TIME_OPTIONS = {"morning": "Morning (6–9)", "midday": "Midday (10–15)",
                         "evening": "Evening (16–20)", "flexible": "Flexible"}
EXERCISE_SLOTS = {"morning": (6, 9), "midday": (10, 15), "evening": (16, 20)}
COOKING_OPTIONS = {"gas": "Gas (LPG/PNG)", "electric": "Electric/induction",
                   "biomass": "Wood, charcoal or other solid fuel", "unsure": "Prefer not to say"}
HOUSEHOLD_OPTIONS = {"children": "Young children", "older_adults": "Older adults",
                     "sensitive": "Someone more sensitive to air pollution"}
INDOOR_SOURCE_OPTIONS = {"smoking": "Indoor smoking", "incense": "Incense, agarbatti or dhoop",
                         "candles": "Candles", "waste_burning": "Burning of waste or leaves at home or nearby"}
STRUGGLE_OPTIONS = {"commute": "Smoggy air on my commute", "cooking": "Smoke or fumes while cooking",
                    "indoor_air": "Dust or smog getting indoors",
                    "exercise": "Staying active without breathing polluted air"}

DEFAULT_PREFS = {
    "commute_mode": "car",
    "departure_time": "08:30",
    "exercise": "outdoors",
    "distance_km": None,
    "purifier": "not_set",
    "exercise_time": "flexible",
    "cooking": "unsure",
    "household": [],
    "indoor_sources": [],
    "struggles": [],
}
MAX_DISTANCE_KM = 300.0
MAX_RECOMMENDATIONS = 5

# Forecast analysis thresholds (see analyze_forecast).
HORIZON_HOURS = 24
WINDOW_HOURS = 3
MIN_VALID_POINTS = 12
DAYTIME_FIRST_START_HOUR = 6    # windows must start 06:00-19:00 so they end by 22:00
DAYTIME_LAST_START_HOUR = 19
MIN_REL_DIFFERENCE = 0.25       # lower window must be >=25% below the higher one
MIN_ABS_DIFFERENCE = {"us_aqi": 10.0, "pm2_5": 5.0}  # in the metric's own units
METRIC_LABELS = {"us_aqi": "US AQI", "pm2_5": "PM2.5"}

GENERAL_LIMITATION = (
    "These are modelled regional forecast estimates (about 45 km grid cells for much of the world), "
    "not readings from a sensor near your home and not an official CPCB/Indian AQI observation. "
    "The AQI shown is the US AQI scale calculated by the data provider, not India's National AQI."
)


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #
def _choice_list(value, options: dict, name: str) -> list:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{name} must be a list.")
    if any(item not in options for item in value):
        raise ValueError(f"Unknown value in {name}.")
    return [key for key in options if key in value]  # canonical order, no duplicates


def validate_preferences(prefs: dict) -> dict:
    """Return a cleaned copy of the household preferences or raise ValueError."""
    merged = {**DEFAULT_PREFS, **(prefs or {})}

    mode = merged["commute_mode"]
    if mode not in COMMUTE_MODES:
        raise ValueError("Unknown commute mode.")
    if merged["exercise"] not in EXERCISE_OPTIONS:
        raise ValueError("Unknown exercise preference.")
    if merged["exercise_time"] not in EXERCISE_TIME_OPTIONS:
        raise ValueError("Unknown exercise time.")
    if merged["cooking"] not in COOKING_OPTIONS:
        raise ValueError("Unknown cooking answer.")
    purifier = merged["purifier"]
    if purifier == "not_set":
        purifier = "unsure"
    if purifier not in PURIFIER_OPTIONS:
        raise ValueError("Unknown air-purifier answer.")

    departure = merged["departure_time"]
    if isinstance(departure, time):
        departure = departure.strftime("%H:%M")
    try:
        departure = datetime.strptime(str(departure), "%H:%M").strftime("%H:%M")
    except ValueError:
        raise ValueError("Departure time must look like HH:MM.") from None

    distance = merged["distance_km"]
    if distance is not None:
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ValueError("Commute distance must be a number.")
        if distance < 0:
            raise ValueError("Commute distance cannot be negative.")
        if distance > MAX_DISTANCE_KM:
            raise ValueError(f"Commute distance must be at most {MAX_DISTANCE_KM:.0f} km.")
        distance = float(distance) or None  # 0 means "not given"

    return {
        "commute_mode": mode, "departure_time": departure, "exercise": merged["exercise"],
        "exercise_time": merged["exercise_time"], "distance_km": distance, "purifier": purifier,
        "cooking": merged["cooking"],
        "household": _choice_list(merged["household"], HOUSEHOLD_OPTIONS, "household"),
        "indoor_sources": _choice_list(merged["indoor_sources"], INDOOR_SOURCE_OPTIONS, "indoor_sources"),
        "struggles": _choice_list(merged["struggles"], STRUGGLE_OPTIONS, "struggles"),
    }


# --------------------------------------------------------------------------- #
# Forecast analysis
# --------------------------------------------------------------------------- #
def _parse_time(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(metric: str, value: float) -> str:
    """'US AQI about 95' or 'PM2.5 about 33.5 μg/m³'."""
    label = METRIC_LABELS[metric]
    return f"{label} about {value:.0f}" if metric == "us_aqi" else f"{label} about {value:.1f} μg/m³"


def _window(start: datetime, avg: float) -> dict:
    end = start + timedelta(hours=WINDOW_HOURS)
    return {"start": start.strftime("%Y-%m-%dT%H:%M"), "end": end.strftime("%Y-%m-%dT%H:%M"),
            "avg": round(avg, 1), "label": f"{start:%a %H:%M}–{end:%H:%M}"}


def _series(hourly, metric: str) -> dict:
    values = {}
    for row in (hourly or [])[:HORIZON_HOURS]:
        if not isinstance(row, dict):
            continue
        t, v = _parse_time(row.get("time")), row.get(metric)
        if t is not None and _is_number(v):
            values[t] = float(v)
    return values


def _analyze(hourly, metric: str) -> dict:
    result = {"metric": metric, "label": METRIC_LABELS[metric], "status": "insufficient",
              "clear_difference": False, "lower": None, "higher": None, "n_valid": 0}
    values = _series(hourly, metric)
    result["n_valid"] = len(values)
    if len(values) < MIN_VALID_POINTS:
        return result

    windows = []
    for start in sorted(values):
        if not DAYTIME_FIRST_START_HOUR <= start.hour <= DAYTIME_LAST_START_HOUR:
            continue
        hours = [start + timedelta(hours=i) for i in range(WINDOW_HOURS)]
        if all(h in values for h in hours):
            windows.append((start, sum(values[h] for h in hours) / WINDOW_HOURS))
    if len(windows) < 2:
        return result

    low = min(windows, key=lambda w: (w[1], w[0]))
    high = max(windows, key=lambda w: (w[1], -w[0].timestamp()))
    diff = high[1] - low[1]
    overlap = abs((high[0] - low[0]).total_seconds()) < WINDOW_HOURS * 3600
    result["status"] = "flat"
    if diff >= MIN_ABS_DIFFERENCE[metric] and diff / high[1] >= MIN_REL_DIFFERENCE and not overlap:
        result.update(status="ok", clear_difference=True,
                      lower=_window(low[0], low[1]), higher=_window(high[0], high[1]))
    return result


def analyze_forecast(hourly: list | None, metric: str | None = None) -> dict:
    """Find the relatively lower and higher 3-hour windows (06:00-22:00) in the next 24 hours.

    The headline metric is the US AQI when enough valid hourly values exist,
    otherwise PM2.5. Status is "insufficient" (too few valid points, so no
    window is named), "flat" (no clear difference) or "ok" (clear difference;
    ``lower`` and ``higher`` are set). Points without a parsable time or a
    numeric value are ignored, so missing timestamps can never produce a window.
    """
    if metric is not None:
        return _analyze(hourly, metric)
    primary = _analyze(hourly, "us_aqi")
    if primary["status"] != "insufficient":
        return primary
    fallback = _analyze(hourly, "pm2_5")
    return fallback if fallback["status"] != "insufficient" else primary


def _value_at_hour(hourly, metric: str, hour: int) -> float | None:
    """First forecast value in the next 24 h at the given clock hour."""
    for t, v in sorted(_series(hourly, metric).items()):
        if t.hour == hour:
            return v
    return None


def _slot_avg(hourly, metric: str, first_hour: int, last_hour: int) -> float | None:
    vals = [v for t, v in _series(hourly, metric).items() if first_hour <= t.hour <= last_hour]
    return sum(vals) / len(vals) if len(vals) >= 2 else None


def _departure_in_higher_window(hourly: list, departure: str, analysis: dict) -> bool:
    if not analysis["clear_difference"]:
        return False
    hour = int(departure[:2])
    start, end = _parse_time(analysis["higher"]["start"]), _parse_time(analysis["higher"]["end"])
    for row in (hourly or [])[:HORIZON_HOURS]:
        t = _parse_time(row.get("time")) if isinstance(row, dict) else None
        if t is not None and t.hour == hour:
            return start <= t < end
    return False


def _is_clearly_above(analysis: dict, value: float) -> bool:
    """Is `value` clearly above the lower window, by the same rule used to find windows?"""
    gap = value - analysis["lower"]["avg"]
    return gap >= MIN_ABS_DIFFERENCE[analysis["metric"]] and value > 0 and gap / value >= MIN_REL_DIFFERENCE


# --------------------------------------------------------------------------- #
# Journey summary: ties the forecast to the user's own routine
# --------------------------------------------------------------------------- #
_TRIP_NOUN = {"car": "drive", "motorcycle": "ride", "bus": "bus journey", "metro_train": "metro/train journey",
              "walk_cycle": "walk or cycle", "other": "trip"}
_SOURCE_SHORT = {"smoking": "indoor smoking", "incense": "incense or dhoop", "candles": "candles",
                 "waste_burning": "burning waste or leaves"}


def _trip(prefs: dict) -> str:
    noun, dist = _TRIP_NOUN[prefs["commute_mode"]], prefs["distance_km"]
    return f"your {dist:g} km {noun}" if dist else f"your {noun}"


def build_journey(prefs: dict, hourly: list | None) -> list[str]:
    """A few Markdown lines describing what the forecast looks like for this household's own day."""
    prefs = validate_preferences(prefs)
    analysis = analyze_forecast(hourly)
    metric = analysis["metric"]
    lines = []

    if prefs["commute_mode"] == "wfh":
        lines.append("**Work from home:** there is no commute to plan around, so most of your hours are indoors "
                     "and the plan focuses on your home air and daily activities.")
    else:
        dist = f", about {prefs['distance_km']:g} km each way" if prefs["distance_km"] else ""
        line = f"**Commute:** {COMMUTE_MODES[prefs['commute_mode']]}{dist}, leaving around {prefs['departure_time']}."
        dep = _value_at_hour(hourly, metric, int(prefs["departure_time"][:2]))
        if dep is not None:
            line += f" The model puts {describe(metric, dep)} at that time"
            if analysis["clear_difference"]:
                line += (f", compared with {describe(metric, analysis['lower']['avg'])} "
                         f"in the relatively lower period ({analysis['lower']['label']}).")
            else:
                line += "."
        lines.append(line)

    if prefs["exercise"] == "outdoors":
        slot = EXERCISE_SLOTS.get(prefs["exercise_time"])
        avg = _slot_avg(hourly, metric, *slot) if slot else None
        if avg is not None:
            lines.append(f"**Outdoor exercise:** your usual {prefs['exercise_time']} session falls where the model "
                         f"shows {describe(metric, avg)}.")
        else:
            lines.append("**Outdoor exercise:** your timing is flexible, so the relatively lower period (if the "
                         "forecast shows one) is the one to aim for.")
    elif prefs["exercise"] == "indoors":
        lines.append("**Exercise:** indoors, so the main exposure to manage is the air in your home and on your trips.")

    home = []
    if prefs["cooking"] == "biomass":
        home.append("cooking with solid fuel")
    home += [_SOURCE_SHORT[k] for k in prefs["indoor_sources"]]
    if home:
        lines.append(f"**At home:** you mentioned {', '.join(home)}; these are indoor sources you can control directly.")
    if prefs["household"]:
        names = ", ".join(HOUSEHOLD_OPTIONS[k].lower() for k in prefs["household"])
        lines.append(f"**Household:** {names}, so the plan leans cautious on outdoor time.")
    if prefs["struggles"]:
        names = "; ".join(STRUGGLE_OPTIONS[k].lower() for k in prefs["struggles"])
        lines.append(f"**You asked us to focus on:** {names}.")
    return lines


# --------------------------------------------------------------------------- #
# Recommendation candidates
# --------------------------------------------------------------------------- #
def _rec(rec_id, category, title, action, reason, impact, effort, cost, caveat, priority):
    rec = {"id": rec_id, "category": category, "title": title, "action": action, "reason": reason,
           "expected_direction_of_impact": impact, "effort": effort, "cost": cost, "caveat": caveat}
    return max(priority, 0), rec


def _boost(prefs: dict, concern: str) -> int:
    """A concern the user selected moves a matching recommendation one place up the list."""
    return -1 if concern in prefs["struggles"] else 0


def _transport_candidates(prefs: dict, hourly: list, analysis: dict) -> list:
    mode, cat = prefs["commute_mode"], "Getting around"
    if mode == "wfh":
        return []  # nothing commute-related for people who do not commute
    metric, label = analysis["metric"], analysis["label"]
    trip, dep = _trip(prefs), prefs["departure_time"]
    boost = _boost(prefs, "commute")
    long_trip = prefs["distance_km"] is not None and prefs["distance_km"] >= 10
    out = []
    in_high = _departure_in_higher_window(hourly, dep, analysis)
    dep_val = _value_at_hour(hourly, metric, int(dep[:2]))
    dep_txt = f" (the model shows {describe(metric, dep_val)} then)" if dep_val is not None else ""

    if in_high:
        out.append(_rec(
            "flex_departure", cat, "Shift trip timing if you can",
            f"If your schedule is flexible, consider travelling closer to the relatively lower forecast period ({analysis['lower']['label']}).",
            f"Your usual {dep} departure{dep_txt} falls in the period the model forecasts as relatively higher for {label}.",
            "May reduce the time you spend travelling during a higher modelled period.",
            "medium", "free", "Only worth doing if work or school hours allow it; the difference is between forecast periods, not a safety guarantee.", 1))

    if mode in ("car", "motorcycle"):
        if long_trip:
            why = f"A long daily trip like {trip} adds up quickly, so swapping even some days for shared transport matters most."
        else:
            why = f"{trip.capitalize()} adds traffic exhaust every day it is made; fewer vehicles and fewer stop-start trips mean less of it."
        out.append(_rec(
            "shared_transport", cat, "Share, switch or combine trips",
            "Where a bus, metro, train or carpool is practical for your route, use it for some trips; combine errands into one loop and skip optional short trips.",
            why,
            "May reduce per-person transport emissions, depending on how full the vehicle is and what it replaces.",
            "medium", "low-cost", "Only if a suitable service exists near you and fits your schedule; keep your current option if it does not.", 2 + boost))
        if mode == "car" and (long_trip or boost):
            out.append(_rec(
                "in_vehicle_air", cat, "Look after the air inside your car",
                "In slow, heavy traffic set the ventilation to recirculate, keep the cabin air filter clean or replaced, and switch back to fresh air once the road clears.",
                f"On {trip}, exhaust from the vehicles around you can enter the cabin through the fresh-air intake.",
                "May reduce how much traffic exhaust gets into the cabin during the drive.",
                "low", "low-cost", "Effect varies by car. Stale air builds up if recirculate is left on for long.", 2 + boost))
        if mode == "motorcycle":
            out.append(_rec(
                "rider_exposure", cat, "Reduce exposure while riding",
                "Where it is safe, use less congested side roads and avoid waiting directly behind buses and trucks. If you wear a mask, a well-fitted N95/FFP2-type respirator filters particles far better than a cloth or surgical mask.",
                f"On {trip} you breathe traffic exhaust directly, with no cabin between you and the road.",
                "May reduce the particles you breathe in during the ride.",
                "low", "low-cost", "Fit matters and respirators can be uncomfortable on long rides. Never take a less safe or much longer route just to cut exposure.", 2 + boost))
        short = prefs["distance_km"] is not None and prefs["distance_km"] <= 3
        if short and not in_high:
            out.append(_rec(
                "short_trip_active", cat, "Consider walking or cycling for very short trips",
                f"For a trip of about 3 km or less like {trip}, walking or cycling on quieter streets could replace the vehicle.",
                "Short vehicle trips contribute emissions without saving much time.",
                "Removes your own vehicle emissions for that trip.",
                "medium", "free", "Only where there is a safe route away from heavy traffic, and not if you are unwell or the weather is poor.", 2))
    elif mode in ("bus", "metro_train"):
        out.append(_rec(
            "keep_public_transport", cat, "Keep using shared transport",
            f"Keep using the {COMMUTE_MODES[mode].lower()} for {trip}. If you wait at a roadside stop, stand back from the kerb and away from idling vehicles where you can.",
            f"Shared transport usually means lower emissions per passenger than a private vehicle, and the wait at the stop is where you are most exposed to roadside air{dep_txt}.",
            "Keeps your per-person transport emissions comparatively low.",
            "low", "low-cost", "Per-person savings depend on how full the service is.", 3 + boost))
    elif mode == "walk_cycle":
        out.append(_rec(
            "quieter_routes", cat, "Choose quieter routes",
            f"For {trip}, use side streets, parks or paths away from heavy traffic wherever a safe alternative exists.",
            "Traffic exhaust is usually higher right beside busy roads, and walking or cycling means you breathe harder along the way.",
            "May reduce how much traffic-related air you breathe on the way.",
            "low", "free", "Do not take an unsafe or much longer route; a route change will not remove regional pollution.", 2 + boost))
    else:  # other
        out.append(_rec(
            "plan_trips", cat, "Plan trips to avoid unnecessary travel",
            f"For {trip}, combine errands and skip trips that are not needed.",
            "Fewer trips means fewer vehicle emissions and less time spent in traffic.",
            "May reduce traffic emissions and your time in traffic.",
            "low", "free", "Only as far as your routine allows.", 3 + boost))
    return out


def _activity_candidates(prefs: dict, hourly: list, analysis: dict) -> list:
    cat, exercise = "Daily activities", prefs["exercise"]
    metric, label = analysis["metric"], analysis["label"]
    clear = analysis["clear_difference"]
    lower = analysis["lower"]["label"] if clear else None
    boost = _boost(prefs, "exercise")
    caveat_cmp = ("This compares forecast periods in a regional model; it does not mean any hour is safe. "
                  "Stop and rest if you feel unwell.")
    out = []

    if exercise == "outdoors":
        slot = EXERCISE_SLOTS.get(prefs["exercise_time"])
        slot_avg = _slot_avg(hourly, metric, *slot) if slot else None
        slot_name = prefs["exercise_time"]
        if clear and slot_avg is not None and not _is_clearly_above(analysis, slot_avg):
            out.append(_rec(
                "exercise_slot_ok", cat, "Your usual exercise time looks comparatively fine",
                f"Your usual {slot_name} session is forecast around {describe(metric, slot_avg)}, close to the lowest waking-hours period ({lower}). "
                "Keep it, pick parks or paths away from busy roads, and have an indoor backup.",
                "Within this model your usual slot is not clearly worse than the lower period.",
                "Keeps your outdoor exercise out of the relatively higher period.",
                "low", "free", caveat_cmp, 2 + boost))
        elif clear:
            if slot_avg is not None:
                action = (f"Your usual {slot_name} session is forecast around {describe(metric, slot_avg)}, versus "
                          f"{describe(metric, analysis['lower']['avg'])} in the relatively lower period ({lower}). "
                          "If you can, move it there, or switch to an indoor session on higher days.")
                reason = (f"The model shows your usual {slot_name} slot clearly above the lower period for {label}, "
                          "and harder exercise increases how much air you breathe.")
            else:
                action = (f"If you exercise outdoors, consider the relatively lower forecast period ({lower}), "
                          "or an indoor alternative on higher-forecast days.")
                reason = (f"The model forecasts a clear difference in {label} across today's waking hours (06:00–22:00), "
                          "and harder exercise increases how much air you breathe.")
            out.append(_rec(
                "shift_exercise", cat, "Plan outdoor exercise for the lower period", action, reason,
                "May reduce time spent outdoors during a higher modelled period.",
                "medium", "free", caveat_cmp, 1 + boost))
        else:
            out.append(_rec(
                "exercise_general", cat, "Keep outdoor exercise flexible",
                "The forecast does not show a clear lower period, so keep outdoor sessions lighter or shorter when you can, "
                "pick parks or paths away from busy roads, and have an indoor option ready.",
                "No clear best time can be named from the data available.",
                "May reduce how much polluted air you breathe during exercise.",
                "low", "free", "General guidance only; follow any local public-health advisory.", 3 + boost))
    elif exercise == "indoors":
        out.append(_rec(
            "indoor_exercise", cat, "Keep exercising indoors",
            "Your indoor routine already avoids exercising beside traffic; keep it and ventilate the room when outdoor conditions look comparatively better.",
            "Harder breathing during exercise increases how much air you take in, so the quality of that air matters.",
            "Avoids outdoor exposure while exercising.",
            "low", "free", "Indoor air can also hold pollution from cooking, smoking or incense; see the home tips.", 3 + boost))
    else:
        if clear:
            out.append(_rec(
                "shift_errands", cat, "Time outdoor errands for the lower period",
                f"Where you can choose, do outdoor errands or walks during the relatively lower forecast period ({lower}).",
                f"The model forecasts a clear difference in {label} across today's waking hours (06:00–22:00).",
                "May reduce time spent outdoors during a higher modelled period.",
                "low", "free", "Only when your schedule is flexible; this is a forecast comparison, not a safety guarantee.", 2))
        else:
            out.append(_rec(
                "errands_general", cat, "Keep outdoor time purposeful",
                "Group outdoor errands together and avoid lingering next to busy roads.",
                "No clear lower period can be identified from the forecast, so conservative everyday habits are the safest guidance.",
                "May reduce unnecessary time near traffic.",
                "low", "free", "General guidance only.", 4))

    if prefs["household"]:
        groups = ", ".join(HOUSEHOLD_OPTIONS[k].lower() for k in prefs["household"])
        when = f" during the relatively lower period ({lower})" if clear else ""
        out.append(_rec(
            "sensitive_household", cat, "Look after the more sensitive people at home",
            f"Plan outdoor play, walks and school or errand trips{when} where you can, and keep a comfortable indoor activity ready for higher-forecast days.",
            f"You mentioned {groups} at home. Children, older adults and people who are more sensitive to air pollution are generally affected sooner by polluted air.",
            "May reduce their time outdoors during a higher modelled period.",
            "low", "free", "General information, not medical advice. Follow your doctor's guidance for any health condition.", 2))
    out.append(_rec(
        "check_advisory", cat, "Check local advisories",
        "Look at your local public-health or pollution-control advisory before planning long outdoor activity.",
        "Official local advisories draw on monitoring this app does not have.",
        "Helps you act on locally verified information.",
        "low", "free", "This app gives general information, not medical advice.", 6))
    return out


def _source_control(prefs: dict) -> tuple:
    chosen = prefs["indoor_sources"]
    caveat = "Does not change outdoor air; you cannot measure the effect with this app."
    if not chosen:
        return _rec(
            "source_control", "At home", "Cut indoor pollution at the source",
            "Avoid indoor smoking, incense, candles and any waste burning; use a kitchen exhaust that vents outdoors when you cook; "
            "damp-dust and damp-mop instead of dry sweeping.",
            "Indoor sources add particles directly to the air you breathe at home, and they are the part a household controls most easily.",
            "Removes avoidable indoor pollution sources, which can lower indoor particle levels.",
            "low", "free", caveat, 1)
    steps = []
    if "smoking" in chosen:
        steps.append("take smoking outside, away from doors and windows")
    if "incense" in chosen:
        steps.append("use incense or dhoop less often, and only in a well-ventilated room")
    if "candles" in chosen:
        steps.append("swap candles for electric lights where you can")
    if "waste_burning" in chosen:
        steps.append("do not burn waste or leaves; use municipal collection or composting and report burning nearby to your local body")
    mentioned = ", ".join(_SOURCE_SHORT[k] for k in chosen)
    return _rec(
        "source_control", "At home", "Cut the indoor sources you mentioned",
        f"You mentioned {mentioned}: " + "; ".join(steps) + ". Also damp-dust and damp-mop instead of dry sweeping.",
        "These add particles directly to the air you breathe at home, and they are the part a household controls most easily.",
        "Removes avoidable pollution sources, which can lower indoor particle levels.",
        "low", "free", caveat, 1)


def _home_candidates(prefs: dict, analysis: dict) -> list:
    cat, label = "At home", analysis["label"]
    clear = analysis["clear_difference"]
    out = [_source_control(prefs)]

    cooking = prefs["cooking"]
    if cooking == "biomass":
        out.append(_rec(
            "cooking_smoke", cat, "Reduce kitchen smoke",
            "Cook with a chimney or exhaust fan that vents outdoors, or next to an open window, and keep children and older family members out of the cooking area. "
            "If a cleaner fuel such as LPG/PNG or induction becomes available or affordable, it is the biggest long-term fix.",
            "Cooking with solid fuel releases a lot of smoke into the home, and it is one of the largest indoor particle sources.",
            "May substantially lower indoor smoke during cooking.",
            "medium", "low-cost", "Switching fuel depends on availability and cost; ventilating the kitchen helps but does not remove the smoke.", 1))
    elif cooking == "gas" or "cooking" in prefs["struggles"]:
        out.append(_rec(
            "cooking_smoke", cat, "Vent the kitchen while you cook",
            "Switch on the exhaust fan or chimney (vented outdoors) or open a window while cooking, especially when frying, and keep it running a few minutes afterwards.",
            "Cooking, particularly frying and high-heat cooking, adds fumes and particles to indoor air.",
            "May lower indoor fumes and particles after cooking.",
            "low", "free", "Only helps if the exhaust actually vents outdoors; a recirculating hood does little for fumes.",
            2 + _boost(prefs, "cooking")))

    boost = _boost(prefs, "indoor_air")
    extra = ""
    if clear and boost:
        extra = f" Keep windows closed during the relatively higher period ({analysis['higher']['label']}) and wipe sills and floors with a damp cloth to catch settled dust."
    elif boost:
        extra = " Wipe window sills and floors with a damp cloth to catch settled dust."
    if clear:
        out.append(_rec(
            "ventilation", cat, "Ventilate in the lower period",
            f"If you need to air out rooms, open windows briefly during the relatively lower forecast period ({analysis['lower']['label']}) rather than keeping everything shut all day." + extra,
            "Stale indoor air can build up from cooking and other indoor sources, so ventilation still matters.",
            "May bring in air that is forecast to be comparatively cleaner than at other waking hours.",
            "low", "free", "Ventilate only when practical; heat, noise, a nearby road or smoke may argue against opening windows.", 2 + boost))
    else:
        out.append(_rec(
            "ventilation", cat, "Ventilate sensibly, not never",
            "Do not keep windows shut all day by default. Air out rooms briefly when conditions outside look comparatively better, "
            "or away from busy roads, and use the kitchen exhaust while cooking." + extra,
            f"The forecast shows no clear lower period for {label}, and stale indoor air can build up from cooking and other indoor sources.",
            "May balance fresh air against bringing in outdoor pollution.",
            "low", "free", "Ventilation decisions depend on heat, nearby sources and any filtration you have.", 2 + boost))

    if prefs["commute_mode"] == "wfh":
        out.append(_rec(
            "wfh_workspace", cat, "Set up your work-from-home space",
            "Keep your desk away from the kitchen and from windows facing a busy road, and, if you have a purifier, run it in the room where you work.",
            "Working from home means most of your day is spent in one room, so that room's air matters most.",
            "May lower the particle levels you breathe during work hours.",
            "low", "free", "Depends on your layout; you cannot measure the effect with this app.", 1))

    if prefs["purifier"] == "yes":
        out.append(_rec(
            "maintain_purifier", cat, "Use and maintain your air purifier",
            "Run it in the room you spend most time in (for example the bedroom at night, or your work room), with doors and windows mostly closed while it runs, "
            "and replace or clean the filter on schedule.",
            "A maintained particle filter can remove particles from the air in a room.",
            "May lower indoor particle levels in that room.",
            "low", "low-cost", "A purifier does not remove all gases, and a clogged filter or an undersized unit does little.", 2))
    else:
        out.append(_rec(
            "consider_purifier", cat, "Consider a correctly sized particle filter",
            "If affordable, a purifier with a particle (HEPA-type) filter sized for the room is a reasonable addition. "
            + ("Check whether yours has one and is sized for the room." if prefs["purifier"] == "unsure" else "It is optional, not essential."),
            "Filtering removes particles that source control and ventilation do not.",
            "May lower indoor particle levels in the room where it runs.",
            "medium", "paid", "Costs money and electricity and needs filter changes. Houseplants and home hacks are not a reliable way to remove PM2.5.", 3))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_recommendations(prefs: dict, hourly: list | None) -> list[dict]:
    """Return 3-5 recommendations, grouped in CATEGORIES order.

    Raises ValueError for invalid preferences. Missing or malformed forecast
    data simply yields general, conservative guidance.
    """
    prefs = validate_preferences(prefs)
    hourly = hourly or []
    analysis = analyze_forecast(hourly)
    candidates = (_transport_candidates(prefs, hourly, analysis)
                  + _home_candidates(prefs, analysis)
                  + _activity_candidates(prefs, hourly, analysis))
    candidates.sort(key=lambda c: (c[0], c[1]["id"]))

    chosen = []
    for category in CATEGORIES:  # best candidate of each category first
        best = next((c for c in candidates if c[1]["category"] == category), None)
        if best:
            chosen.append(best)
    for cand in candidates:      # then fill by priority
        if len(chosen) >= MAX_RECOMMENDATIONS:
            break
        if cand not in chosen:
            chosen.append(cand)
    chosen.sort(key=lambda c: (CATEGORIES.index(c[1]["category"]), c[0], c[1]["id"]))
    return [rec for _, rec in chosen]


def build_checklist(place_label: str, analysis: dict, retrieved_at: str | None, is_sample: bool = False) -> str:
    """Plain-text shareable checklist. General actions only: no coordinates, no household details."""
    label = analysis.get("label", "PM2.5")
    lines = ["BreatheWise household clean-air checklist", f"Area: {place_label}"]
    if is_sample:
        lines.append("NOTE: built from SAMPLE data - not live.")
    lines.append("")
    if analysis.get("clear_difference"):
        lines.append(f"Forecast (modelled {label}, next 24 h): relatively lower around {analysis['lower']['label']}; "
                     f"relatively higher around {analysis['higher']['label']}.")
    else:
        lines.append(f"Forecast (modelled {label}): no clear lower or higher period could be identified.")
    lines += [
        "",
        "[ ] No indoor smoking, incense, candles or waste burning",
        "[ ] Use a kitchen exhaust that vents outdoors while cooking",
        "[ ] Damp-dust and damp-mop instead of dry sweeping",
        "[ ] Air out rooms briefly when outdoor conditions look comparatively better",
        "[ ] Share rides, use public transport or combine errands where practical",
        "[ ] Plan outdoor exercise and errands for the relatively lower period, or choose an indoor option",
        "[ ] If you have a purifier, check its filter and size; houseplants are not a substitute",
        "[ ] Check your local public-health advisory",
        "",
        "Source: Open-Meteo Air Quality API (CAMS model). These are modelled regional forecast estimates, "
        "not readings from a nearby sensor, not official CPCB values and not indoor measurements. "
        "AQI figures use the US AQI scale calculated by the data provider, not India's National AQI.",
        "General information only, not medical advice.",
    ]
    if retrieved_at:
        lines.append(f"Forecast retrieved: {retrieved_at}")
    return "\n".join(lines)
