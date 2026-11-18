"""BreatheWise: a small clean-air assistant. Run with: streamlit run app.py

This file only composes the UI. Data access lives in breathewise/data_provider.py,
the rules in breathewise/recommendations.py and the optional LLM call in
breathewise/assistant.py.
"""
import hashlib
import json
import os
from datetime import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from breathewise import assistant, data_provider, formatting
from breathewise import recommendations as recs_mod
from breathewise.data_provider import FIELDS, POLLUTANTS, DataUnavailable

load_dotenv()
st.set_page_config(page_title="BreatheWise", page_icon="🌬️", layout="centered")

CHART_COLOR = "#7A4A2B"
MEDICAL_NOTE = ("This app provides general air-quality information, not medical advice. "
                "Follow local public-health advisories and seek professional care for urgent symptoms.")
MODEL_CAVEAT = ("**Modelled forecast estimate, not a sensor reading.** These values come from a global "
                "atmospheric model (CAMS) via Open-Meteo. For much of the world the model grid is roughly 45 km wide, "
                "so it describes a wide area, not your street or home. It is not an official CPCB reading "
                "and says nothing about indoor air.")
AQI_NOTE = ("**About the AQI:** the AQI index (air quality index) turns pollutant levels into one number. "
            "This one is the **US AQI**, calculated by the data provider from the modelled concentrations. "
            "It is **not India's National AQI (CPCB)**, which uses a different scale, so it can differ from the AQI "
            "you see in Indian apps and news. Scale: 0–50 Good · 51–100 Moderate · 101–150 Unhealthy for sensitive "
            "groups · 151–200 Unhealthy · 201–300 Very unhealthy · 301+ Hazardous.")
GLOSSARY = ("**PM2.5** and **PM10** are tiny airborne particles under 2.5 and 10 micrometres across. "
            "**Nitrogen dioxide (NO₂)** is a gas released mainly by traffic and combustion. "
            "**Ozone (O₃)** is a gas formed when sunlight reacts with other pollutants. "
            "μg/m³ means micrograms per cubic metre of air.")


# Public, non-personal lookups only (geocoding and forecasts). Short TTL; failures are not cached.
@st.cache_data(ttl=1800, show_spinner=False)
def cached_search(query: str) -> list:
    return data_provider.search_places(query)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecast(latitude: float, longitude: float) -> dict:
    return data_provider.fetch_air_quality(latitude, longitude)


def llm_config() -> tuple[str, str]:
    """GEMINI_API_KEY / GEMINI_MODEL from the environment or Streamlit secrets. Never from code."""
    values = {}
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        value = os.environ.get(name, "").strip()
        if not value:
            try:
                value = str(st.secrets.get(name, "")).strip()
            except Exception:  # no secrets file configured
                value = ""
        values[name] = value
    return values["GEMINI_API_KEY"], values["GEMINI_MODEL"]


# --------------------------------------------------------------------------- #
# 1. Title
# --------------------------------------------------------------------------- #
st.title("BreatheWise")
st.write("A simple clean-air plan for your household, based on a regional air-quality forecast.")

# --------------------------------------------------------------------------- #
# 2. Location
# --------------------------------------------------------------------------- #
st.header("Choose a location")
with st.form("place_search"):
    query = st.text_input("City or place name", max_chars=data_provider.MAX_QUERY_LEN,
                          placeholder="For example: Delhi")
    searched = st.form_submit_button("Search for this place")
if searched:
    st.session_state["places"], st.session_state["search_message"] = [], None
    try:
        st.session_state["places"] = cached_search(query)
        if not st.session_state["places"]:
            st.session_state["search_message"] = "No matching places were found. Try a different spelling."
    except ValueError as exc:
        st.session_state["search_message"] = str(exc)
    except DataUnavailable as exc:
        st.session_state["search_message"] = f"Place search is unavailable: {exc}"
if st.session_state.get("search_message"):
    st.warning(st.session_state["search_message"])

places = st.session_state.get("places", [])
place = None
if places:
    index = st.selectbox("Choose the matching place", range(len(places)),
                         format_func=lambda i: places[i]["label"])
    place = places[index]

use_sample = st.checkbox("Show sample data instead (demonstration only, not live)", key="use_sample")

# --------------------------------------------------------------------------- #
# 3. Forecast data
# --------------------------------------------------------------------------- #
forecast, place_label = None, ""
if use_sample:
    try:
        forecast, place_label = data_provider.load_sample(), "Sample location (demonstration only)"
    except DataUnavailable as exc:
        st.error(str(exc))
elif place:
    place_label = place["label"]
    try:
        with st.spinner("Loading the forecast…"):
            forecast = cached_forecast(place["latitude"], place["longitude"])
    except (DataUnavailable, ValueError) as exc:
        st.error(f"The air-quality forecast is unavailable right now. {exc} No values are being shown.")
        if st.button("Retry loading the forecast"):
            cached_forecast.clear()
            st.rerun()

if forecast is None:
    st.info("Search for a place above to see its air-quality outlook and your plan for today.")
    st.stop()

st.markdown(f"Selected location: **{place_label}**")
if forecast["is_sample"]:
    st.warning("**Sample data — not live.** These numbers are made up for demonstration and do not describe any real place.")

hourly, units, current = forecast["hourly"], forecast["units"], forecast["current"]
analysis = recs_mod.analyze_forecast(hourly)
label = analysis["label"]

st.header("Air-quality outlook")
aqi_now = current["values"]["us_aqi"]
aqi_cat = formatting.aqi_category(aqi_now)
with st.container(border=True):
    st.metric("US AQI right now (modelled)", formatting.format_metric("us_aqi", aqi_now, ""))
    if aqi_cat:
        st.badge(aqi_cat["label"], color=aqi_cat["color"])
        st.write(f"**{aqi_cat['label']}.** {aqi_cat['note']} (US EPA scale)")
    else:
        st.write("The AQI is not available in this forecast. The pollutant values below are still shown.")
st.markdown(AQI_NOTE)
st.info(MODEL_CAVEAT)
st.markdown(
    f"- **Location:** {place_label}\n"
    f"- **Source:** {forecast['provider']}\n"
    f"- **Forecast time:** {formatting.format_time(current['time'])} ({forecast['timezone']})\n"
    f"- **Retrieved:** {formatting.format_retrieved(forecast['retrieved_at'])}"
)
st.subheader("What is in the air (modelled)")
keys = list(POLLUTANTS)
for row_keys in (keys[:2], keys[2:]):
    for col, key in zip(st.columns(2), row_keys):
        with col, st.container(border=True):
            st.metric(POLLUTANTS[key], formatting.format_metric(key, current["values"][key], units[key]))
st.markdown(GLOSSARY)

# --------------------------------------------------------------------------- #
# 4. Trend
# --------------------------------------------------------------------------- #
st.header("Forecast trend")
metric = st.selectbox("What to chart", list(FIELDS), format_func=FIELDS.get)
span = st.radio("Time span", ["Next 24 hours", "Next 48 hours (as far as available)"], horizontal=True)
shown = hourly[:24] if span.startswith("Next 24") else hourly
series = [(row["time"], row[metric]) for row in shown if row[metric] is not None]
axis_label = FIELDS[metric] if metric == "us_aqi" else f"{FIELDS[metric]} ({units[metric]})"
if len(series) < 2:
    st.warning(f"{FIELDS[metric]}: Not available in this forecast.")
else:
    # Plain column name on purpose: dots/brackets in a name break Streamlit's chart (field-path parsing).
    frame = pd.DataFrame({"Value": [v for _, v in series]}, index=pd.to_datetime([t for t, _ in series]))
    st.line_chart(frame, x_label=f"Time ({forecast['timezone']})", y_label=axis_label, color=CHART_COLOR)
    lo, hi = min(series, key=lambda p: p[1]), max(series, key=lambda p: p[1])
    st.markdown(
        f"Chart description: modelled hourly **{FIELDS[metric]}** from {formatting.format_time(series[0][0])} "
        f"to {formatting.format_time(series[-1][0])}. The lowest value in this range is "
        f"{formatting.format_metric(metric, lo[1], units[metric])} around {formatting.format_time(lo[0])}; the highest is "
        f"{formatting.format_metric(metric, hi[1], units[metric])} around {formatting.format_time(hi[0])}."
    )
if analysis["clear_difference"]:
    st.markdown(
        f"- **Relatively lower {label} period:** {analysis['lower']['label']} "
        f"({recs_mod.describe(analysis['metric'], analysis['lower']['avg'])})\n"
        f"- **Relatively higher {label} period:** {analysis['higher']['label']} "
        f"({recs_mod.describe(analysis['metric'], analysis['higher']['avg'])})\n\n"
        "These compare 3-hour periods between 06:00 and 22:00 within this model's next 24 hours. "
        "“Lower” does not mean safe."
    )
elif analysis["status"] == "flat":
    st.markdown(f"**No clear lower or higher {label} period** stands out in the next 24 hours, so the plan uses general guidance.")
else:
    st.markdown("**Not enough valid forecast data** to compare periods, so the plan uses general guidance.")

with st.expander("Show the forecast as a table"):
    marks = {}
    if analysis["clear_difference"]:
        for name, mark in (("lower", "Relatively lower period"), ("higher", "Relatively higher period")):
            start, end = analysis[name]["start"], analysis[name]["end"]
            for row in hourly:
                if start <= row["time"] < end:
                    marks[row["time"]] = mark
    table = [{"Time": formatting.format_time(row["time"]),
              **{(FIELDS[k] if k == "us_aqi" else f"{FIELDS[k]} ({units[k]})"):
                 formatting.format_metric(k, row[k], "").strip() for k in FIELDS},
              f"Period marker ({label})": marks.get(row["time"], "")} for row in hourly]
    st.dataframe(table, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 5. Household routine (session state only)
# --------------------------------------------------------------------------- #
st.header("Your household routine")
st.write("Tell us about your day and the plan will be built around it. These answers stay in this browser session "
         "only; they are not saved or logged.")

st.markdown("**Getting around**")
mode = st.selectbox("How do you usually get around?", list(recs_mod.COMMUTE_MODES),
                    format_func=recs_mod.COMMUTE_MODES.get, key="pref_mode")
departure, distance = time(8, 30), 0.0
if mode != "wfh":
    c1, c2 = st.columns(2)
    departure = c1.time_input("Typical departure time", value=time(8, 30), key="pref_departure")
    distance = c2.number_input("Approximate one-way commute distance in km (optional; leave 0 to skip)",
                               min_value=0.0, max_value=recs_mod.MAX_DISTANCE_KM, value=0.0, step=1.0,
                               key="pref_distance")
exercise = st.radio("Exercise preference", list(recs_mod.EXERCISE_OPTIONS), index=2,
                    format_func=recs_mod.EXERCISE_OPTIONS.get, horizontal=True, key="pref_exercise")
exercise_time = "flexible"
if exercise == "outdoors":
    exercise_time = st.radio("When do you usually exercise outdoors?", list(recs_mod.EXERCISE_TIME_OPTIONS),
                             format_func=recs_mod.EXERCISE_TIME_OPTIONS.get, horizontal=True, key="pref_exercise_time")

st.markdown("**At home**")
cooking = st.radio("How do you mostly cook?", list(recs_mod.COOKING_OPTIONS), index=3,
                   format_func=recs_mod.COOKING_OPTIONS.get, horizontal=True, key="pref_cooking")
indoor_sources = st.multiselect("Does any of this happen at home? (choose any that apply)",
                                list(recs_mod.INDOOR_SOURCE_OPTIONS),
                                format_func=recs_mod.INDOOR_SOURCE_OPTIONS.get, key="pref_sources",
                                placeholder="None of these, or choose any")
purifier = st.radio("Does your household have an air purifier?", list(recs_mod.PURIFIER_OPTIONS), index=2,
                    format_func=recs_mod.PURIFIER_OPTIONS.get, horizontal=True, key="pref_purifier")

st.markdown("**Who and what matters most**")
household = st.multiselect("Who lives with you? (optional, general groups only)", list(recs_mod.HOUSEHOLD_OPTIONS),
                           format_func=recs_mod.HOUSEHOLD_OPTIONS.get, key="pref_household",
                           placeholder="Choose any that apply")
struggles = st.multiselect("What is hardest for you right now? (optional)", list(recs_mod.STRUGGLE_OPTIONS),
                           format_func=recs_mod.STRUGGLE_OPTIONS.get, key="pref_struggles",
                           placeholder="Choose any that apply")

if st.button("Build my plan", type="primary"):
    st.session_state["plan_ready"] = True

try:
    prefs = recs_mod.validate_preferences({
        "commute_mode": mode, "departure_time": departure, "exercise": exercise, "exercise_time": exercise_time,
        "distance_km": distance, "purifier": purifier, "cooking": cooking, "household": household,
        "indoor_sources": indoor_sources, "struggles": struggles})
except ValueError as exc:
    st.error(f"Please check your answers: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# 6. Daily action plan (only after the routine has been entered)
# --------------------------------------------------------------------------- #
if not st.session_state.get("plan_ready"):
    st.info("Answer the questions above, then choose “Build my plan” to get recommendations that fit your day.")
else:
    recommendations = recs_mod.build_recommendations(prefs, hourly)
    st.header("Your plan for today")
    st.subheader("Your day in the forecast")
    if (cooking == "unsure" and not (indoor_sources or household or struggles)
            and (mode == "wfh" or not distance)):
        st.caption("Tip: the more you tell us above (distance, cooking, who is at home, what is hardest), "
                   "the more specific this plan becomes.")
    for line in recs_mod.build_journey(prefs, hourly):
        st.markdown(f"- {line}")
    for category in recs_mod.CATEGORIES:
        items = [r for r in recommendations if r["category"] == category]
        if not items:
            continue
        st.subheader(category)
        for r in items:
            with st.container(border=True):
                st.markdown(f"**{r['title']}**")
                st.markdown(
                    f"- **Action:** {r['action']}\n"
                    f"- **Why:** {r['reason']}\n"
                    f"- **Likely direction of impact:** {r['expected_direction_of_impact']}\n"
                    f"- **Effort / cost:** {r['effort']} effort, {r['cost']}\n"
                    f"- **Limitation:** {r['caveat']}"
                )

    context = assistant.build_context(place_label, forecast, prefs, recommendations, analysis)
    st.subheader("Why this plan")
    st.markdown(assistant.template_explanation(context))
    st.caption(MEDICAL_NOTE)

    # ----------------------------------------------------------------------- #
    # 7. Ask a question (optional AI)
    # ----------------------------------------------------------------------- #
    st.header("Ask a question")
    api_key, model = llm_config()
    if not (api_key and model):
        st.info("AI explanations are disabled because no GEMINI_API_KEY and GEMINI_MODEL are configured. "
                "The plan and explanation above come from built-in rules and work without AI.")
    else:
        with st.form("ask_form"):
            question = st.text_area("Ask about your plan or the forecast (optional, up to 500 characters)",
                                    max_chars=assistant.MAX_QUESTION_CHARS)
            asked = st.form_submit_button("Explain my plan with AI")
        signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if asked:
            with st.spinner("Asking the AI model…"):
                result = assistant.explain(context, question, api_key, model)
            st.session_state["ai_result"] = {**result, "sig": signature, "question": assistant.clean_question(question)}
        saved = st.session_state.get("ai_result")
        if saved and saved["sig"] == signature:
            if saved["question"]:
                st.text(f"You asked: {saved['question']}")
            if saved["note"]:
                st.warning(saved["note"])
            st.markdown(saved["text"])  # plain Markdown; raw HTML stays disabled
            st.caption("AI-written explanation of the data and plan above. It may be wrong; it cannot see anything else.")

# --------------------------------------------------------------------------- #
# 8. Checklist, sources and limitations
# --------------------------------------------------------------------------- #
st.header("Share a clean-air checklist")
st.write("A general checklist for family, neighbours or a resident group. It has no coordinates or household details.")
checklist = recs_mod.build_checklist(place_label, analysis, forecast["retrieved_at"], forecast["is_sample"])
st.download_button("Download checklist as a text file", checklist, file_name="breathewise_checklist.txt",
                   mime="text/plain")
st.code(checklist, language=None)
st.write("To copy it, use the copy icon at the top right of the box above.")

st.header("Sources and limitations")
st.markdown(
    "- **Data:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) "
    "(CAMS model data), with [Open-Meteo geocoding](https://open-meteo.com/en/docs/geocoding-api) for place search.\n"
    "- These are modelled forecast estimates, not readings from a sensor at your home and not official CPCB station data.\n"
    "- The AQI shown is the provider's **US AQI**. This app does not calculate or show India's National AQI, "
    "and does not measure indoor air.\n"
    "- Recommendations are general and directional. They cannot measure improvement in your neighbourhood's air "
    "or your personal health risk."
)
st.caption(MEDICAL_NOTE)
