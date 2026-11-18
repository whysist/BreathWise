import re

import pytest

from breathewise import recommendations as rm

FIELDS = {"id", "category", "title", "action", "reason", "expected_direction_of_impact",
          "effort", "cost", "caveat"}


def hourly_series(values, start_hour=0):
    """Hourly rows from a list of PM2.5 values starting at start_hour on 2026-11-18."""
    rows = []
    for i, v in enumerate(values):
        h = start_hour + i
        day, hh = 18 + h // 24, h % 24
        rows.append({"time": f"2026-11-{day:02d}T{hh:02d}:00", "pm2_5": v, "pm10": v, "nitrogen_dioxide": 20.0,
                     "ozone": 30.0})
    return rows


def varied_day():
    """Clear difference: high in the morning and evening, low 13:00-16:00."""
    values = [100.0] * 24
    for h in (13, 14, 15, 16):
        values[h] = 40.0
    return hourly_series(values)


def flat_day():
    return hourly_series([60.0 + (i % 3) for i in range(24)])


def prefs(**kw):
    return {"commute_mode": "car", "departure_time": "08:30", "exercise": "outdoors",
            "distance_km": None, "purifier": "no", **kw}


def ids(recs):
    return {r["id"] for r in recs}


def text(recs):
    return " ".join(" ".join(str(v) for v in r.values()) for r in recs).lower()


# ---- structure -------------------------------------------------------------
@pytest.mark.parametrize("mode", list(rm.COMMUTE_MODES))
@pytest.mark.parametrize("exercise", list(rm.EXERCISE_OPTIONS))
@pytest.mark.parametrize("forecast", [varied_day(), flat_day(), [], None])
def test_always_three_to_five_well_formed_recommendations(mode, exercise, forecast):
    recs = rm.build_recommendations(prefs(commute_mode=mode, exercise=exercise), forecast)
    assert 3 <= len(recs) <= 5
    assert len({r["id"] for r in recs}) == len(recs)
    for r in recs:
        assert set(r) == FIELDS
        assert r["category"] in rm.CATEGORIES
        assert r["effort"] in ("low", "medium", "high")
        assert r["cost"] in ("free", "low-cost", "paid")
        assert all(isinstance(r[k], str) and r[k] for k in FIELDS)


def test_valid_forecast_gives_concise_list_in_category_order():
    recs = rm.build_recommendations(prefs(), varied_day())
    order = [rm.CATEGORIES.index(r["category"]) for r in recs]
    assert order == sorted(order)
    assert {"Getting around", "At home", "Daily activities"} <= {r["category"] for r in recs}


def test_deterministic():
    assert rm.build_recommendations(prefs(), varied_day()) == rm.build_recommendations(prefs(), varied_day())


# ---- forecast analysis -----------------------------------------------------
def test_analysis_finds_lower_and_higher_windows():
    a = rm.analyze_forecast(varied_day())
    assert a["status"] == "ok" and a["clear_difference"]
    assert a["lower"]["start"] == "2026-11-18T13:00" and a["lower"]["avg"] == 40.0
    assert a["higher"]["avg"] == 100.0
    assert a["lower"]["label"] == "Wed 13:00–16:00"


def test_flat_forecast_names_no_best_time():
    a = rm.analyze_forecast(flat_day())
    assert a["status"] == "flat" and not a["clear_difference"]
    assert a["lower"] is None and a["higher"] is None


@pytest.mark.parametrize("bad", [None, [], "nope", [{}], [None, 3, "x"]])
def test_empty_or_malformed_hourly_does_not_crash(bad):
    a = rm.analyze_forecast(bad)
    assert a["status"] == "insufficient" and a["lower"] is None
    assert 3 <= len(rm.build_recommendations(prefs(), bad)) <= 5


def test_missing_timestamps_never_produce_a_best_time():
    rows = varied_day()
    for row in rows:
        row["time"] = None
    a = rm.analyze_forecast(rows)
    assert a["status"] == "insufficient" and not a["clear_difference"]
    recs = rm.build_recommendations(prefs(), rows)
    assert not ({"shift_exercise", "shift_errands", "flex_departure"} & ids(recs))
    assert "13:00" not in text(recs)


def test_missing_pollutant_values_are_skipped_not_invented():
    rows = varied_day()
    for row in rows[:20]:
        row["pm2_5"] = None  # only 4 valid points remain
    assert rm.analyze_forecast(rows)["status"] == "insufficient"
    assert not rm.analyze_forecast([{**r, "pm2_5": None} for r in rows])["clear_difference"]


def test_other_pollutants_missing_does_not_matter_for_pm25_logic():
    rows = [{k: v for k, v in r.items() if k in ("time", "pm2_5")} for r in varied_day()]
    assert rm.analyze_forecast(rows)["clear_difference"]


def test_overnight_minimum_is_not_suggested_for_activities():
    values = [100.0] * 24
    for h in (2, 3, 4):
        values[h] = 20.0  # lowest, but in the small hours
    a = rm.analyze_forecast(hourly_series(values))
    assert not a["clear_difference"]  # daytime windows are all equal


def test_never_claims_a_time_is_safe():
    for forecast in (varied_day(), flat_day(), []):
        for mode in rm.COMMUTE_MODES:
            body = text(rm.build_recommendations(prefs(commute_mode=mode), forecast))
            body = body.replace("does not mean any hour is safe", "").replace("no clear best time", "")  # negations
            assert not re.search(r"(hour|time|period|window|air|day)s? (is|are) (safe|clean|healthy)", body)
            assert not re.search(r"safest|best time|safe time|safe hour|safe to ", body)


# ---- personalisation -------------------------------------------------------
def test_work_from_home_gets_no_commute_recommendation():
    for forecast in (varied_day(), flat_day(), None):
        recs = rm.build_recommendations(prefs(commute_mode="wfh", distance_km=10), forecast)
        assert "Getting around" not in {r["category"] for r in recs}
        assert 3 <= len(recs) <= 5


@pytest.mark.parametrize("mode", ["car", "motorcycle"])
def test_private_vehicle_gets_feasible_alternatives(mode):
    recs = rm.build_recommendations(prefs(commute_mode=mode), flat_day())
    assert "shared_transport" in ids(recs)
    shared = next(r for r in recs if r["id"] == "shared_transport")
    assert "where" in shared["action"].lower()  # conditional on practicality
    assert "only if" in shared["caveat"].lower()  # never assumes transit exists


def test_vehicle_advice_mentions_combining_errands():
    recs = rm.build_recommendations(prefs(commute_mode="car"), flat_day())
    assert "combine errands" in next(r for r in recs if r["id"] == "shared_transport")["action"]


def test_walk_cycle_and_transit_do_not_get_drive_advice():
    for mode in ("walk_cycle", "bus", "metro_train"):
        recs = rm.build_recommendations(prefs(commute_mode=mode), flat_day())
        assert "shared_transport" not in ids(recs)


def test_departure_in_higher_period_suggests_flexible_timing():
    recs = rm.build_recommendations(prefs(departure_time="08:00"), varied_day())
    assert "flex_departure" in ids(recs)
    assert "if your schedule is flexible" in next(r for r in recs if r["id"] == "flex_departure")["action"].lower()


def test_departure_in_lower_period_gets_no_shift_advice():
    assert "flex_departure" not in ids(rm.build_recommendations(prefs(departure_time="14:00"), varied_day()))


def test_short_trip_walking_only_for_short_known_distance_and_not_unsafe():
    short = rm.build_recommendations(prefs(distance_km=2, departure_time="14:00", exercise="indoors",
                                           purifier="yes"), varied_day())
    assert "short_trip_active" in ids(short)
    assert "safe route" in next(r for r in short if r["id"] == "short_trip_active")["caveat"].lower()
    long_trip = rm.build_recommendations(prefs(distance_km=15, exercise="indoors", purifier="yes"), varied_day())
    assert "short_trip_active" not in ids(long_trip)
    in_high = rm.build_recommendations(prefs(distance_km=2, departure_time="08:00", exercise="indoors",
                                             purifier="yes"), varied_day())
    assert "short_trip_active" not in ids(in_high)


def test_outdoor_exercise_with_clear_difference_compares_periods():
    recs = rm.build_recommendations(prefs(exercise="outdoors"), varied_day())
    rec = next(r for r in recs if r["id"] == "shift_exercise")
    assert "13:00" in rec["action"] and "does not mean any hour is safe" in rec["caveat"]


def test_outdoor_exercise_without_clear_difference_gets_general_guidance():
    recs = rm.build_recommendations(prefs(exercise="outdoors"), flat_day())
    assert "exercise_general" in ids(recs) and "shift_exercise" not in ids(recs)


def test_indoor_exercise_is_not_told_to_go_outdoors():
    recs = rm.build_recommendations(prefs(exercise="indoors"), varied_day())
    assert "shift_exercise" not in ids(recs) and "indoor_exercise" in ids(recs)


def test_purifier_variants():
    assert "maintain_purifier" in ids(rm.build_recommendations(prefs(purifier="yes"), flat_day()))
    no = rm.build_recommendations(prefs(purifier="no"), flat_day())
    assert "consider_purifier" in ids(no)
    unsure = next(r for r in rm.build_recommendations(prefs(purifier="unsure"), flat_day())
                  if r["id"] == "consider_purifier")
    assert "check whether yours" in unsure["action"].lower()


# ---- indoor guidance and wording ------------------------------------------
def test_indoor_source_control_always_present_and_covers_key_items():
    for forecast in (varied_day(), flat_day(), None):
        recs = rm.build_recommendations(prefs(), forecast)
        src = next(r for r in recs if r["id"] == "source_control")["action"].lower()
        for word in ("smoking", "incense", "candles", "waste burning", "exhaust", "damp"):
            assert word in src


def test_does_not_tell_users_to_keep_windows_shut_all_day():
    for forecast in (varied_day(), flat_day()):
        recs = rm.build_recommendations(prefs(), forecast)
        vent = next(r for r in recs if r["id"] == "ventilation")
        action = vent["action"].lower()
        assert not re.search(r"(?<!not )(?<!than )keep(ing)?[^.]*(shut|closed)[^.]*all day", action)
        assert "not keep windows shut all day" in action or "rather than keeping everything shut all day" in action


def test_no_houseplant_claims_or_numeric_risk_claims():
    body = text(rm.build_recommendations(prefs(), varied_day()))
    assert "not a reliable way" in body or "not a substitute" in body or "houseplants" not in body
    assert "reduce your risk by" not in body and "%" not in body


# ---- validation ------------------------------------------------------------
@pytest.mark.parametrize("distance", [-1, -0.1, 301, 1e9, float("nan"), float("inf"), "5", True])
def test_invalid_distance_rejected(distance):
    with pytest.raises(ValueError):
        rm.validate_preferences(prefs(distance_km=distance))
    with pytest.raises(ValueError):
        rm.build_recommendations(prefs(distance_km=distance), varied_day())


@pytest.mark.parametrize("field,value", [("commute_mode", "teleport"), ("exercise", "swimming"),
                                         ("purifier", "maybe"), ("departure_time", "25:99"),
                                         ("departure_time", "late")])
def test_invalid_enums_and_times_rejected(field, value):
    with pytest.raises(ValueError):
        rm.validate_preferences(prefs(**{field: value}))


def test_valid_preferences_normalised():
    from datetime import time
    clean = rm.validate_preferences(prefs(departure_time=time(7, 5), distance_km=0, purifier="not_set"))
    assert clean["departure_time"] == "07:05" and clean["distance_km"] is None and clean["purifier"] == "unsure"
    assert rm.validate_preferences(prefs(distance_km=0))["distance_km"] is None
    assert rm.validate_preferences(prefs(distance_km=12.5))["distance_km"] == 12.5


# ---- checklist -------------------------------------------------------------
def test_checklist_has_no_coordinates_or_household_details():
    a = rm.analyze_forecast(varied_day())
    out = rm.build_checklist("Delhi, India", a, "2026-11-18T05:00Z")
    assert "Delhi, India" in out and "Wed 13:00–16:00" in out
    assert "modelled" in out.lower() and "not medical advice" in out.lower()
    for private in ("latitude", "longitude", "28.6", "77.2", "car", "motorcycle", "departure", "purifier is"):
        assert private not in out.lower().replace("if you have a purifier", "")


def test_checklist_without_clear_forecast_and_sample_label():
    out = rm.build_checklist("Sample", rm.analyze_forecast(None), None, is_sample=True)
    assert "no clear lower or higher" in out and "SAMPLE data - not live" in out


# ---- US AQI as the headline metric -------------------------------------------
def aqi_day():
    rows = varied_day()
    for row in rows:
        row["us_aqi"] = 160.0 if row["pm2_5"] == 100.0 else 90.0  # high vs lower
    return rows


def test_analysis_prefers_aqi_and_falls_back_to_pm25():
    a = rm.analyze_forecast(aqi_day())
    assert a["metric"] == "us_aqi" and a["label"] == "US AQI" and a["clear_difference"]
    assert a["lower"]["avg"] == 90.0 and a["higher"]["avg"] == 160.0
    assert rm.analyze_forecast(varied_day())["metric"] == "pm2_5"


def test_aqi_difference_needs_ten_points_and_25_percent():
    rows = hourly_series([100.0] * 24)
    for r in rows:
        r["us_aqi"] = 100.0
    for r in rows[13:17]:
        r["us_aqi"] = 92.0  # only 8 points lower
    assert not rm.analyze_forecast(rows)["clear_difference"]


def test_recommendations_name_the_metric_in_use():
    recs = rm.build_recommendations(prefs(departure_time="08:00"), aqi_day())
    flex = next(r for r in recs if r["id"] == "flex_departure")
    assert "US AQI" in flex["reason"] and "US AQI about 160" in flex["reason"]


def test_aqi_category_bands_and_missing():
    from breathewise import formatting as f
    assert f.aqi_category(None) is None
    assert f.aqi_category(50)["label"] == "Good"
    assert f.aqi_category(51)["label"] == "Moderate"
    assert f.aqi_category(108)["label"] == "Unhealthy for sensitive groups"
    assert f.aqi_category(160)["label"] == "Unhealthy"
    assert f.aqi_category(250)["label"] == "Very unhealthy"
    assert f.aqi_category(301)["label"] == "Hazardous"
    assert f.format_metric("us_aqi", 107.6, "") == "108"
    assert f.format_metric("us_aqi", None, "") == "Not available"


# ---- personalisation ---------------------------------------------------------
def test_new_preference_fields_validated():
    clean = rm.validate_preferences(prefs(household=["older_adults", "children", "children"],
                                          indoor_sources=["incense"], struggles=["cooking"], cooking="biomass",
                                          exercise_time="evening"))
    assert clean["household"] == ["children", "older_adults"]  # canonical order, deduplicated
    for field, value in (("household", ["pets"]), ("indoor_sources", "smoking"), ("struggles", ["x"]),
                         ("cooking", "coal"), ("exercise_time", "night")):
        with pytest.raises(ValueError):
            rm.validate_preferences(prefs(**{field: value}))


def test_commute_text_uses_the_users_distance_mode_and_departure():
    recs = rm.build_recommendations(prefs(commute_mode="car", distance_km=22, departure_time="08:00"), aqi_day())
    body = text(recs)
    assert "22 km drive" in body and "08:00" in body


def test_long_car_commute_gets_in_vehicle_advice_short_does_not():
    assert "in_vehicle_air" in ids(rm.build_recommendations(prefs(distance_km=25), flat_day()))
    assert "in_vehicle_air" not in ids(rm.build_recommendations(prefs(distance_km=4), flat_day()))
    assert "in_vehicle_air" in ids(rm.build_recommendations(prefs(struggles=["commute"]), flat_day()))


def test_motorcycle_gets_rider_advice_with_safety_caveat():
    recs = rm.build_recommendations(prefs(commute_mode="motorcycle", struggles=["commute"]), flat_day())
    rider = next(r for r in recs if r["id"] == "rider_exposure")
    assert "N95" in rider["action"] and "never take a less safe" in rider["caveat"].lower()


def test_reported_indoor_sources_are_addressed_specifically():
    recs = rm.build_recommendations(prefs(indoor_sources=["incense", "waste_burning"]), flat_day())
    src = next(r for r in recs if r["id"] == "source_control")
    assert "incense" in src["action"].lower() and "burn waste" in src["action"].lower()
    assert "smoking" not in src["action"].lower()  # not mentioned, so not nagged about


def test_solid_fuel_cooking_gets_kitchen_smoke_advice_first():
    recs = rm.build_recommendations(prefs(cooking="biomass"), flat_day())
    cook = next(r for r in recs if r["id"] == "cooking_smoke")
    assert "solid fuel" in cook["reason"] and "chimney" in cook["action"]
    assert "cooking_smoke" not in ids(rm.build_recommendations(prefs(cooking="electric"), flat_day()))
    assert "cooking_smoke" in ids(rm.build_recommendations(prefs(cooking="electric", struggles=["cooking"]),
                                                           flat_day()))


def test_household_members_get_their_own_recommendation_without_medical_claims():
    recs = rm.build_recommendations(prefs(household=["children", "older_adults"]), varied_day())
    rec = next(r for r in recs if r["id"] == "sensitive_household")
    assert "young children" in rec["reason"] and "not medical advice" in rec["caveat"].lower()
    assert "sensitive_household" not in ids(rm.build_recommendations(prefs(), varied_day()))


def test_usual_exercise_slot_is_compared_with_the_lower_period():
    moved = rm.build_recommendations(prefs(exercise_time="morning"), aqi_day())
    rec = next(r for r in moved if r["id"] == "shift_exercise")
    assert "usual morning session" in rec["action"] and "US AQI about 160" in rec["action"]
    ok = rm.build_recommendations(prefs(exercise_time="midday"), aqi_day())  # 10-15 mostly in the lower period
    assert ("exercise_slot_ok" in ids(ok)) != ("shift_exercise" in ids(ok))


def test_wfh_gets_workspace_advice_and_a_matching_journey():
    recs = rm.build_recommendations(prefs(commute_mode="wfh"), flat_day())
    assert "wfh_workspace" in ids(recs)
    assert "no commute" in " ".join(rm.build_journey(prefs(commute_mode="wfh"), flat_day()))


def test_journey_reflects_the_users_inputs():
    lines = " ".join(rm.build_journey(prefs(distance_km=12, departure_time="08:00", exercise_time="morning",
                                            cooking="biomass", household=["children"], struggles=["commute"],
                                            indoor_sources=["candles"]), aqi_day()))
    assert "12 km" in lines and "08:00" in lines and "US AQI about 160" in lines
    assert "solid fuel" in lines and "candles" in lines and "young children" in lines
    assert "smoggy air on my commute" in lines


def test_indoor_air_struggle_adds_window_advice():
    recs = rm.build_recommendations(prefs(struggles=["indoor_air"]), aqi_day())
    assert "wipe sills" in next(r for r in recs if r["id"] == "ventilation")["action"].lower()
