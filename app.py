"""BreatheWise: a small clean-air assistant. Run with: streamlit run app.py

This file only composes the UI. Data access lives in breathewise/data_provider.py,
the rules in breathewise/recommendations.py and the optional LLM call in
breathewise/assistant.py.

Layout follows "answer first, details on demand":
  1. a one-line promise and a compact place search,
  2. a verdict card (AQI, what it means, what to do, cleaner/worse windows),
  3. a short routine questionnaire that turns into a personal plan,
  4. the charts, numbers and caveats for anyone who wants to dig in,
  5. a shareable checklist to finish on.
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

CHART_COLOR = "#1F7A8C"
# Google's auto-updating alias, so the app doesn't break when a pinned model is retired.
# Override with GEMINI_MODEL in secrets (that value is not secret; the key is).
DEFAULT_MODEL = "gemini-flash-latest"
MAX_AI_CALLS_PER_SESSION = 5  # only applies when the app owner's shared key is used
MEDICAL_NOTE = ("This app provides general air-quality information, not medical advice. "
                "Follow local public-health advisories and seek professional care for urgent symptoms.")
MODEL_CAVEAT = ("**Modelled forecast estimate, not a sensor reading.** These values come from a global "
                "atmospheric model (CAMS) via Open-Meteo. For much of the world the model grid is roughly 45 km wide, "
                "so it describes a wide area, not your street or home. It is not an official CPCB reading "
                "and says nothing about indoor air.")
SHORT_CAVEAT = ("Modelled regional forecast covering roughly a 45 km area, not a sensor at your home. "
                "Shown as US AQI, not India's CPCB AQI. More under “Explore the forecast”.")
AQI_NOTE = ("**About the AQI:** the AQI index (air quality index) turns pollutant levels into one number. "
            "This one is the **US AQI**, calculated by the data provider from the modelled concentrations. "
            "It is **not India's National AQI (CPCB)**, which uses a different scale, so it can differ from the AQI "
            "you see in Indian apps and news. Scale: 0–50 Good · 51–100 Moderate · 101–150 Unhealthy for sensitive "
            "groups · 151–200 Unhealthy · 201–300 Very unhealthy · 301+ Hazardous.")
GLOSSARY = ("**PM2.5** and **PM10** are tiny airborne particles under 2.5 and 10 micrometres across. "
            "**Nitrogen dioxide (NO₂)** is a gas released mainly by traffic and combustion. "
            "**Ozone (O₃)** is a gas formed when sunlight reacts with other pollutants. "
            "μg/m³ means micrograms per cubic metre of air.")

BASE_CSS = """
<style>
  /* Tighter, calmer type: headings guide, they don't shout. */
  h1 { font-size: 2.4rem !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }
  h2 { font-size: 1.45rem !important; margin-top: 1.6rem !important; }
  h3 { font-size: 1.1rem !important; }
  .bw-tagline { font-size: 1.15rem; color: #44545F; margin: 0.2rem 0 1.2rem 0; }
  /* Recommendation cards: white on the tinted page so they read as "things to do". */
  .st-key-plan [data-testid="stVerticalBlockBorderWrapper"],
  .st-key-plan div[class*="st-key-rec-"] { background: #FFFFFF; border-radius: 12px; }
</style>
"""


def verdict_css(hex_color: str) -> str:
    """Colour the verdict card by AQI category. hex_color comes from formatting's fixed table."""
    return f"""
<style>
  .st-key-verdict {{
      background: #FFFFFF;
      border: 1px solid #D3DFE4 !important;
      border-left: 10px solid {hex_color} !important;
      border-radius: 16px !important;
      box-shadow: 0 4px 18px rgba(30, 42, 51, 0.07);
      padding: 1.2rem 1.4rem !important;
  }}
  .st-key-verdict [data-testid="stMetricValue"] {{ color: {hex_color}; font-size: 3.4rem; font-weight: 700; }}
  .st-key-verdict h3 {{ font-size: 1.35rem !important; line-height: 1.35; padding-top: 0.2rem; }}
</style>
"""


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


def place_from_url() -> dict | None:
    """Restore the last chosen place from ?place=…&lat=…&lon=… so returning visitors land on their answer."""
    params = st.query_params
    try:
        lat, lon = data_provider.validate_coordinates(params["lat"], params["lon"])
    except (KeyError, ValueError):
        return None
    label = data_provider.clean_text(params.get("place", ""), 200) or f"{lat:.2f}, {lon:.2f}"
    return {"name": label, "region": "", "country": "", "latitude": lat, "longitude": lon, "label": label}


def remember_place(place: dict) -> None:
    """Put the place in the URL (bookmarkable/shareable). Only the public place name and coordinates."""
    wanted = {"place": place["label"], "lat": f"{place['latitude']:.4f}", "lon": f"{place['longitude']:.4f}"}
    if any(st.query_params.get(k) != v for k, v in wanted.items()):
        st.query_params.update(wanted)


st.html(BASE_CSS)

# --------------------------------------------------------------------------- #
# 1. Promise
# --------------------------------------------------------------------------- #
st.title("BreatheWise")
st.html('<p class="bw-tagline">Know when to open the windows, when to head out, '
        'and how to keep the air at home cleaner today.</p>')

# --------------------------------------------------------------------------- #
# 2. Location (compact: one row, the dropdown only when there is a real choice)
# --------------------------------------------------------------------------- #
if "places" not in st.session_state:
    restored = place_from_url()
    if restored:
        st.session_state["places"] = [restored]

with st.form("place_search", border=False):
    c_query, c_button = st.columns([4, 1], vertical_alignment="bottom")
    query = c_query.text_input("Where are you?", max_chars=data_provider.MAX_QUERY_LEN,
                               placeholder="City or town, for example: Hyderabad")
    searched = c_button.form_submit_button("Search", type="primary", width="stretch")
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
if len(places) == 1:
    place = places[0]
elif places:
    index = st.selectbox("Several places match. Which one do you mean?", range(len(places)),
                         format_func=lambda i: places[i]["label"])
    place = places[index]

with st.expander("No place in mind? Preview the app with sample data",
                 expanded=bool(st.session_state.get("use_sample"))):
    use_sample = st.checkbox("Use sample data (demonstration only, not live)", key="use_sample")

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
        remember_place(place)
    except (DataUnavailable, ValueError) as exc:
        st.error(f"The air-quality forecast is unavailable right now. {exc} No values are being shown.")
        if st.button("Retry loading the forecast"):
            cached_forecast.clear()
            st.rerun()

if forecast is None:
    st.info("Search for a place above to see today's air and a simple plan for your household.")
    st.stop()

if forecast["is_sample"]:
    st.warning("**Sample data — not live.** These numbers are made up for demonstration and do not describe any real place.")

hourly, units, current = forecast["hourly"], forecast["units"], forecast["current"]
analysis = recs_mod.analyze_forecast(hourly)
label = analysis["label"]

# --------------------------------------------------------------------------- #
# 4. Verdict card: the answer, before anything else
# --------------------------------------------------------------------------- #
aqi_now = current["values"]["us_aqi"]
aqi_cat = formatting.aqi_category(aqi_now)
st.html(verdict_css(aqi_cat["hex"] if aqi_cat else "#8A9BA5"))

with st.container(key="verdict"):
    st.caption(f"📍 {place_label} · {formatting.format_time(current['time'])} ({forecast['timezone']})")
    c_num, c_text = st.columns([1, 2.2], vertical_alignment="center")
    with c_num:
        st.metric("US AQI right now (modelled)", formatting.format_metric("us_aqi", aqi_now, ""))
    with c_text:
        if aqi_cat:
            st.badge(aqi_cat["label"], color=aqi_cat["color"])
            st.markdown(f"### {aqi_cat['advice']}")
        else:
            st.markdown("### The AQI is not available in this forecast.")
            st.write("The pollutant values further down are still shown.")

    if analysis["clear_difference"]:
        c_low, c_high = st.columns(2)
        with c_low, st.container(border=True):
            st.markdown(f"🌤️ **Relatively lower {label} period**  \n{analysis['lower']['label']}")
            st.caption(recs_mod.describe(analysis["metric"], analysis["lower"]["avg"]))
        with c_high, st.container(border=True):
            st.markdown(f"🌫️ **Relatively higher {label} period**  \n{analysis['higher']['label']}")
            st.caption(recs_mod.describe(analysis["metric"], analysis["higher"]["avg"]))
        st.caption("Compares 3-hour periods between 06:00 and 22:00 in the next 24 hours. "
                   "“Lower” does not mean safe.")
    elif analysis["status"] == "flat":
        st.markdown(f"**No clear lower or higher {label} period** stands out in the next 24 hours.")
    else:
        st.markdown("**Not enough valid forecast data** to compare periods today.")
    st.caption(SHORT_CAVEAT)

# --------------------------------------------------------------------------- #
# 5. Household routine (session state only)
# --------------------------------------------------------------------------- #
st.header("Get a plan for your household")
st.write("About a minute. Your answers stay in this browser session only; nothing is saved or logged.")

st.markdown("**1 · Getting around**")
mode = st.selectbox("How do you usually get around?", list(recs_mod.COMMUTE_MODES),
                    format_func=recs_mod.COMMUTE_MODES.get, key="pref_mode")
departure, distance = time(8, 30), 0.0
if mode != "wfh":
    c1, c2 = st.columns(2)
    departure = c1.time_input("Usual departure time", value=time(8, 30), key="pref_departure")
    distance = c2.number_input("One-way distance in km (optional, 0 to skip)",
                               min_value=0.0, max_value=recs_mod.MAX_DISTANCE_KM, value=0.0, step=1.0,
                               key="pref_distance")
exercise = st.radio("Exercise", list(recs_mod.EXERCISE_OPTIONS), index=2,
                    format_func=recs_mod.EXERCISE_OPTIONS.get, horizontal=True, key="pref_exercise")
exercise_time = "flexible"
if exercise == "outdoors":
    exercise_time = st.radio("When do you usually exercise outdoors?", list(recs_mod.EXERCISE_TIME_OPTIONS),
                             format_func=recs_mod.EXERCISE_TIME_OPTIONS.get, horizontal=True, key="pref_exercise_time")

st.markdown("**2 · At home**")
cooking = st.radio("How do you mostly cook?", list(recs_mod.COOKING_OPTIONS), index=3,
                   format_func=recs_mod.COOKING_OPTIONS.get, horizontal=True, key="pref_cooking")
purifier = st.radio("Do you have an air purifier?", list(recs_mod.PURIFIER_OPTIONS), index=2,
                    format_func=recs_mod.PURIFIER_OPTIONS.get, horizontal=True, key="pref_purifier")

with st.expander("**3 · Optional:** make the plan more specific to your home"):
    indoor_sources = st.multiselect("Does any of this happen at home?",
                                    list(recs_mod.INDOOR_SOURCE_OPTIONS),
                                    format_func=recs_mod.INDOOR_SOURCE_OPTIONS.get, key="pref_sources",
                                    placeholder="None of these, or choose any")
    household = st.multiselect("Who lives with you? (general groups only)", list(recs_mod.HOUSEHOLD_OPTIONS),
                               format_func=recs_mod.HOUSEHOLD_OPTIONS.get, key="pref_household",
                               placeholder="Choose any that apply")
    struggles = st.multiselect("What is hardest for you right now?", list(recs_mod.STRUGGLE_OPTIONS),
                               format_func=recs_mod.STRUGGLE_OPTIONS.get, key="pref_struggles",
                               placeholder="Choose any that apply")

if st.button("Build my plan", type="primary", icon="✨"):
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

    with st.container(key="plan"):
        for category in recs_mod.CATEGORIES:
            items = [r for r in recommendations if r["category"] == category]
            if not items:
                continue
            st.subheader(category)
            for r in items:
                with st.container(border=True, key=f"rec-{r['id']}"):
                    st.markdown(f"**✓ {r['title']}**")
                    st.markdown(r["action"])
                    st.caption(f"{r['effort'].capitalize()} effort · {r['cost']}")
                    with st.expander("Why, and what to expect"):
                        st.markdown(
                            f"- **Why:** {r['reason']}\n"
                            f"- **Likely direction of impact:** {r['expected_direction_of_impact']}\n"
                            f"- **Limitation:** {r['caveat']}"
                        )

    context = assistant.build_context(place_label, forecast, prefs, recommendations, analysis)
    with st.expander("Why this plan"):
        st.markdown(assistant.template_explanation(context))
    st.caption(MEDICAL_NOTE)

    # ----------------------------------------------------------------------- #
    # 7. Ask a question (optional AI)
    # ----------------------------------------------------------------------- #
    st.header("Ask a question")
    server_key, configured_model = llm_config()
    model = configured_model or DEFAULT_MODEL
    if server_key:
        api_key, using_own_key = server_key, False
    else:
        # Bring your own key: kept only in this browser session's memory, sent only to Google's Gemini API,
        # never written to disk, logs, the URL or the cache.
        st.write("Gemini explains your plan in plain language and answers questions about it. "
                 "To use it, paste your own free Gemini API key "
                 "([get one in Google AI Studio](https://aistudio.google.com/apikey)).")
        api_key = st.text_input("Your Gemini API key", type="password", key="user_gemini_key",
                                placeholder="Paste a key starting with AIza…", autocomplete="off").strip()
        using_own_key = True
        st.caption("Your key stays in this browser session only. It is sent to Google to answer your question "
                   "and is never saved or logged. Closing the tab forgets it.")
    if not api_key:
        st.info("AI explanations are disabled because no Gemini API key is available yet. "
                "The plan and explanation above come from built-in rules and work without AI.")
    else:
        used = st.session_state.get("ai_calls", 0)
        limit_hit = not using_own_key and used >= MAX_AI_CALLS_PER_SESSION
        with st.form("ask_form"):
            question = st.text_area("Ask about your plan or the forecast (optional, up to 500 characters)",
                                    max_chars=assistant.MAX_QUESTION_CHARS)
            asked = st.form_submit_button("Explain my plan with AI", disabled=limit_hit)
        if limit_hit:
            st.warning(f"This demo allows {MAX_AI_CALLS_PER_SESSION} AI questions per visit on the shared key. "
                       "The built-in plan above keeps working.")
        signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if asked and not limit_hit:
            st.session_state["ai_calls"] = used + 1
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
# 8. Details on demand: trend, pollutants, caveats
# --------------------------------------------------------------------------- #
st.header("Explore the forecast")
c_metric, c_span = st.columns([3, 2], vertical_alignment="bottom")
metric = c_metric.selectbox("What to chart", list(FIELDS), format_func=FIELDS.get)
span = c_span.segmented_control("Time span", ["Next 24 hours", "Next 48 hours"], default="Next 24 hours") \
    or "Next 24 hours"
shown = hourly[:24] if span == "Next 24 hours" else hourly
series = [(row["time"], row[metric]) for row in shown if row[metric] is not None]
axis_label = FIELDS[metric] if metric == "us_aqi" else f"{FIELDS[metric]} ({units[metric]})"
if len(series) < 2:
    st.warning(f"{FIELDS[metric]}: Not available in this forecast.")
else:
    # Plain column name on purpose: dots/brackets in a name break Streamlit's chart (field-path parsing).
    frame = pd.DataFrame({"Value": [v for _, v in series]}, index=pd.to_datetime([t for t, _ in series]))
    st.line_chart(frame, x_label=f"Time ({forecast['timezone']})", y_label=axis_label, color=CHART_COLOR)
    lo, hi = min(series, key=lambda p: p[1]), max(series, key=lambda p: p[1])
    st.caption(
        f"Chart description: modelled hourly **{FIELDS[metric]}** from {formatting.format_time(series[0][0])} "
        f"to {formatting.format_time(series[-1][0])}. The lowest value in this range is "
        f"{formatting.format_metric(metric, lo[1], units[metric])} around {formatting.format_time(lo[0])}; the highest is "
        f"{formatting.format_metric(metric, hi[1], units[metric])} around {formatting.format_time(hi[0])}."
    )

with st.expander("What is in the air right now (modelled)"):
    keys = list(POLLUTANTS)
    for row_keys in (keys[:2], keys[2:]):
        for col, key in zip(st.columns(2), row_keys):
            with col, st.container(border=True):
                st.metric(POLLUTANTS[key], formatting.format_metric(key, current["values"][key], units[key]))
    st.markdown(GLOSSARY)

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

with st.expander("Where these numbers come from"):
    st.info(MODEL_CAVEAT)
    if aqi_cat:
        st.write(f"**{aqi_cat['label']}** on the US EPA scale: {aqi_cat['note']}")
    st.markdown(AQI_NOTE)
    st.markdown(
        f"- **Location:** {place_label}\n"
        f"- **Source:** {forecast['provider']}\n"
        f"- **Forecast time:** {formatting.format_time(current['time'])} ({forecast['timezone']})\n"
        f"- **Retrieved:** {formatting.format_retrieved(forecast['retrieved_at'])}"
    )

# --------------------------------------------------------------------------- #
# 9. Finish on something useful to keep
# --------------------------------------------------------------------------- #
st.header("Share a clean-air checklist")
st.write("A short checklist for family, neighbours or a resident group. "
         "It has no coordinates or household details.")
checklist = recs_mod.build_checklist(place_label, analysis, forecast["retrieved_at"], forecast["is_sample"])
st.code(checklist, language=None)
st.download_button("Download checklist", checklist, file_name="breathewise_checklist.txt",
                   mime="text/plain", icon="⬇️")

with st.expander("Sources and limitations"):
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
