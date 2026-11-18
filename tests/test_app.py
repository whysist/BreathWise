"""Streamlit AppTest smoke tests. No network and no LLM key needed."""
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from breathewise import data_provider as dp

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def no_llm_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(dp.requests, "get", boom)


def sample_app():
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.checkbox(key="use_sample").check().run()
    return at


def click_build(at):
    next(b for b in at.button if b.label == "Build my plan").click().run()
    return at


def test_app_starts_without_network_and_prompts_for_a_place(monkeypatch):
    no_network(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert at.title[0].value == "BreatheWise"
    assert any("Search for a place" in i.value for i in at.info)


def test_aqi_is_the_headline_metric_with_clear_labelling(monkeypatch):
    no_network(monkeypatch)
    at = sample_app()
    assert not at.exception
    assert any("Sample data — not live" in w.value for w in at.warning)
    metrics = {m.label: m.value for m in at.metric}
    assert list(metrics)[0] == "US AQI right now (modelled)"
    assert metrics["US AQI right now (modelled)"].isdigit()
    body = " ".join(m.value for m in at.markdown)
    assert "not India's National AQI" in body
    assert any("Modelled forecast estimate" in i.value for i in at.info)


def test_no_plan_until_the_routine_is_submitted(monkeypatch):
    no_network(monkeypatch)
    at = sample_app()
    assert "Getting around" not in [h.value for h in at.subheader]
    assert "Your plan for today" not in [h.value for h in at.header]
    assert any("Build my plan" in i.value for i in at.info)
    click_build(at)
    assert not at.exception
    assert "Your plan for today" in [h.value for h in at.header]
    assert "Your day in the forecast" in [h.value for h in at.subheader]
    assert any("AI explanations are disabled" in i.value for i in at.info)


def test_plan_reflects_the_entered_routine(monkeypatch):
    no_network(monkeypatch)
    at = sample_app()
    at.selectbox(key="pref_mode").select("motorcycle").run()
    at.number_input(key="pref_distance").set_value(18.0).run()
    at.radio(key="pref_cooking").set_value("biomass").run()
    at.multiselect(key="pref_household").set_value(["children"]).run()
    click_build(at)
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "18 km" in body and "Motorcycle" in body and "solid fuel" in body and "young children" in body


def test_work_from_home_hides_commute_section(monkeypatch):
    no_network(monkeypatch)
    at = sample_app()
    at.selectbox(key="pref_mode").select("wfh").run()
    click_build(at)
    assert not at.exception
    assert "Getting around" not in [h.value for h in at.subheader]


def test_unavailable_api_shows_error_and_retry(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(dp.requests, "get", down)
    at = AppTest.from_file(APP, default_timeout=15)
    at.session_state["places"] = [{"name": "X", "region": "", "country": "Y", "latitude": 1.0,
                                   "longitude": 2.0, "label": "X, Y"}]
    at.run()
    assert not at.exception
    assert any("unavailable" in e.value for e in at.error)
    assert any("Retry" in b.label for b in at.button)
