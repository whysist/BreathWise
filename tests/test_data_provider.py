from datetime import datetime, timezone

import pytest
import requests

from breathewise import data_provider as dp

NOW = datetime(2026, 11, 18, 5, 0, tzinfo=timezone.utc)  # 10:30 local at +05:30


def make_payload(**overrides):
    times = [f"2026-11-18T{h:02d}:00" for h in range(24)] + [f"2026-11-19T{h:02d}:00" for h in range(24)]
    payload = {
        "timezone": "Asia/Kolkata", "utc_offset_seconds": 19800,
        "hourly_units": {"pm2_5": "μg/m³", "pm10": "μg/m³", "nitrogen_dioxide": "μg/m³", "ozone": "μg/m³"},
        "current": {"time": "2026-11-18T10:30", "pm2_5": 80.0, "pm10": 90.0, "nitrogen_dioxide": 30.0, "ozone": 40.0},
        "hourly": {"time": times, **{k: [float(50 + i) for i in range(48)] for k in dp.POLLUTANTS}},
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("bad json")
        return self._body


def patch_get(monkeypatch, *responses):
    """Return queued responses/exceptions in order; record call count."""
    queue, calls = list(responses), []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append((url, params, timeout))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(dp.requests, "get", fake_get)
    return calls


# ---- parsing ---------------------------------------------------------------
def test_parse_valid_payload_starts_at_current_hour():
    data = dp.parse_air_quality(make_payload(), now_utc=NOW)
    assert data["hourly"][0]["time"] == "2026-11-18T10:00"
    assert len(data["hourly"]) == 38  # 48 minus the 10 past hours
    assert data["current"]["values"]["pm2_5"] == 80.0
    assert data["units"]["pm2_5"] == "μg/m³"
    assert data["retrieved_at"] == "2026-11-18T05:00Z"
    assert data["is_sample"] is False


def test_missing_pollutant_becomes_none_not_a_made_up_value():
    payload = make_payload()
    del payload["hourly"]["ozone"]
    del payload["current"]["ozone"]
    payload["hourly"]["pm10"][12] = None
    data = dp.parse_air_quality(payload, now_utc=NOW)
    assert data["current"]["values"]["ozone"] is None
    assert all(row["ozone"] is None for row in data["hourly"])
    assert next(r for r in data["hourly"] if r["time"] == "2026-11-18T12:00")["pm10"] is None


@pytest.mark.parametrize("bad", [-5, float("nan"), float("inf"), "12", True, 99999])
def test_invalid_values_are_discarded(bad):
    payload = make_payload()
    payload["hourly"]["pm2_5"][15] = bad
    data = dp.parse_air_quality(payload, now_utc=NOW)
    assert next(r for r in data["hourly"] if r["time"] == "2026-11-18T15:00")["pm2_5"] is None


@pytest.mark.parametrize("payload", [None, [], {}, {"hourly": {}}, {"hourly": {"time": "x"}}, {"hourly": []}])
def test_malformed_payload_is_unavailable(payload):
    with pytest.raises(dp.DataUnavailable):
        dp.parse_air_quality(payload, now_utc=NOW)


def test_empty_or_unparsable_times_are_unavailable():
    for times in ([], ["not-a-time", None, 5]):
        payload = make_payload()
        payload["hourly"]["time"] = times
        with pytest.raises(dp.DataUnavailable):
            dp.parse_air_quality(payload, now_utc=NOW)


def test_all_values_missing_is_unavailable():
    payload = make_payload()
    payload["hourly"] = {"time": payload["hourly"]["time"]}
    with pytest.raises(dp.DataUnavailable):
        dp.parse_air_quality(payload, now_utc=NOW)


def test_stale_forecast_is_unavailable():
    payload = make_payload()
    payload["current"]["time"] = "2026-11-17T01:00"  # a day old
    with pytest.raises(dp.DataUnavailable, match="out of date"):
        dp.parse_air_quality(payload, now_utc=NOW)


def test_sample_data_is_labelled_and_never_live():
    sample = dp.load_sample()
    assert sample["is_sample"] is True
    assert "not live" in sample["provider"].lower()
    assert sample["retrieved_at"] is None


# ---- HTTP ------------------------------------------------------------------
def test_fetch_air_quality_uses_fixed_host_and_params(monkeypatch):
    calls = patch_get(monkeypatch, FakeResponse(200, make_payload()))
    data = dp.fetch_air_quality(28.6, 77.2)
    url, params, timeout = calls[0]
    assert url == dp.AIR_QUALITY_URL and url.startswith("https://air-quality-api.open-meteo.com/")
    assert params["timezone"] == "auto" and params["forecast_days"] == 2
    assert params["hourly"] == params["current"] == "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone"
    assert timeout and timeout <= 12
    assert data["hourly"]


def test_server_error_retries_once_then_controlled_error(monkeypatch):
    calls = patch_get(monkeypatch, FakeResponse(503), FakeResponse(503))
    with pytest.raises(dp.DataUnavailable):
        dp.fetch_air_quality(28.6, 77.2)
    assert len(calls) == 2  # exactly one retry


def test_transient_failure_then_success(monkeypatch):
    patch_get(monkeypatch, requests.ConnectionError("boom"), FakeResponse(200, make_payload()))
    assert dp.fetch_air_quality(28.6, 77.2)["hourly"]


def test_timeout_is_controlled_error(monkeypatch):
    patch_get(monkeypatch, requests.Timeout(), requests.Timeout())
    with pytest.raises(dp.DataUnavailable, match="could not be reached"):
        dp.fetch_air_quality(28.6, 77.2)


def test_client_error_does_not_retry(monkeypatch):
    calls = patch_get(monkeypatch, FakeResponse(400, {}))
    with pytest.raises(dp.DataUnavailable, match="HTTP 400"):
        dp.fetch_air_quality(28.6, 77.2)
    assert len(calls) == 1


def test_invalid_json_is_controlled_error(monkeypatch):
    patch_get(monkeypatch, FakeResponse(200, bad_json=True))
    with pytest.raises(dp.DataUnavailable, match="unreadable"):
        dp.fetch_air_quality(28.6, 77.2)


@pytest.mark.parametrize("lat,lon", [(91, 0), (-91, 0), (0, 181), (0, -181), ("abc", 0), (None, 0),
                                     (float("nan"), 0), (True, 0), (0, float("inf"))])
def test_invalid_coordinates_rejected(lat, lon, monkeypatch):
    calls = patch_get(monkeypatch)
    with pytest.raises(ValueError):
        dp.fetch_air_quality(lat, lon)
    assert calls == []  # no request was made


def test_valid_coordinates_accepted():
    assert dp.validate_coordinates("28.6", 77) == (28.6, 77.0)
    assert dp.validate_coordinates(-90, 180) == (-90.0, 180.0)


# ---- geocoding -------------------------------------------------------------
def test_search_places_parses_and_skips_bad_coordinates(monkeypatch):
    body = {"results": [
        {"name": "Delhi", "admin1": "NCT of Delhi", "country": "India", "latitude": 28.65, "longitude": 77.23},
        {"name": "Delhi", "admin1": "New York", "country": "United States", "latitude": 42.3, "longitude": -74.9},
        {"name": "Broken", "country": "X", "latitude": 999, "longitude": 0},
        {"name": "NoCoords", "country": "X"},
        "junk",
    ]}
    calls = patch_get(monkeypatch, FakeResponse(200, body))
    places = dp.search_places("  Delhi ")
    assert [p["label"] for p in places] == ["Delhi, NCT of Delhi, India", "Delhi, New York, United States"]
    assert calls[0][0] == dp.GEOCODING_URL and calls[0][1]["name"] == "Delhi"


def test_search_places_no_results(monkeypatch):
    patch_get(monkeypatch, FakeResponse(200, {"generationtime_ms": 0.3}))
    assert dp.search_places("zzzzzz") == []


def test_search_places_rejects_bad_input(monkeypatch):
    calls = patch_get(monkeypatch)
    for bad in ("", " ", "a", None):
        with pytest.raises(ValueError):
            dp.search_places(bad)
    assert calls == []


def test_search_places_trims_and_limits_query(monkeypatch):
    calls = patch_get(monkeypatch, FakeResponse(200, {}))
    dp.search_places("x" * 500 + "\n\x00")
    assert len(calls[0][1]["name"]) <= dp.MAX_QUERY_LEN


def test_search_places_api_down(monkeypatch):
    patch_get(monkeypatch, FakeResponse(500), FakeResponse(500))
    with pytest.raises(dp.DataUnavailable):
        dp.search_places("Delhi")


# ---- US AQI ------------------------------------------------------------------
def test_us_aqi_is_parsed_as_a_whole_index_with_no_unit():
    payload = make_payload()
    payload["current"]["us_aqi"] = 108
    payload["hourly"]["us_aqi"] = [100 + i for i in range(48)]
    data = dp.parse_air_quality(payload, now_utc=NOW)
    assert data["current"]["values"]["us_aqi"] == 108.0
    assert data["hourly"][0]["us_aqi"] == 110.0  # hour 10
    assert data["units"]["us_aqi"] == ""


def test_missing_aqi_is_none_not_computed_from_pollutants():
    data = dp.parse_air_quality(make_payload(), now_utc=NOW)
    assert data["current"]["values"]["us_aqi"] is None
    assert all(row["us_aqi"] is None for row in data["hourly"])


def test_sample_data_includes_aqi():
    assert dp.load_sample()["current"]["values"]["us_aqi"] is not None
