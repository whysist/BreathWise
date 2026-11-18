# BreatheWise

A small clean-air assistant for households. Search for a city, see the modelled air-quality forecast, answer a few routine questions and get a short, practical plan for the day, plus a shareable checklist. An optional LLM can explain the plan in plain language; the app works fully without it.

> **Not an official AQI app.** BreatheWise shows *modelled forecast estimates* (Open-Meteo, based on the CAMS atmospheric model). It does **not** show readings from a sensor at your home, official CPCB station data, India's National AQI (CPCB) or indoor air measurements. The AQI it leads with is the *US AQI*, calculated by Open-Meteo from the modelled concentrations. It gives general information, not medical advice.

## MVP scope

- Place search (Open-Meteo Geocoding) with a choice among ambiguous results. No GPS, no home address.
- The headline is the provider's **US AQI** (number, category name and a plain-text scale), followed by PM2.5, PM10, nitrogen dioxide and ozone: value, unit, forecast time and time zone, provider, retrieval time. Missing values show "Not available".
- 24 to 48 hour line chart with a data table, plus text labels for the relatively lower and higher periods (by US AQI, or PM2.5 if the AQI is missing) (3-hour averages between 06:00 and 22:00 in the next 24 hours). No hour is ever called "safe".
- Household inputs kept in Streamlit session state only: commute mode, departure time, optional distance, exercise (and usual time), cooking fuel, indoor smoke sources, air purifier, who lives at home (general groups) and what is hardest. The plan is built only after you press **Build my plan**.
- 3 to 5 recommendations under *Getting around*, *At home* and *Daily activities*, each with action, reason, likely direction of impact, effort/cost and a limitation. They use your own answers, for example the forecast at your departure time and exercise slot, your commute distance, your cooking fuel and the indoor sources you report. A short "Your day in the forecast" summary shows what the model says for your routine. Rules are plain Python in `breathewise/recommendations.py`.
- Optional AI explanation and short Q&A (Google Gemini), always layered on top of the deterministic plan.
- Plain-text checklist to download or copy (no coordinates or household details).

Deliberately out of scope: accounts, database, backend server, community posts or chat, analytics, model training, AQI conversion.

## Data source and limitations

- Data: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) and [Geocoding API](https://open-meteo.com/en/docs/geocoding-api). Review the [Open-Meteo terms](https://open-meteo.com/en/terms) before any commercial use.
- For much of the world the CAMS global grid is about 45 km, so values describe a wide area, not a street or a home. The forecast refreshes on a multi-hour cycle.
- The AQI shown is the provider's own **US AQI** (US EPA scale), shown as-is and labelled. It is not India's National AQI (CPCB), which uses different breakpoints, so it can differ from AQI figures in Indian apps. The app never converts or recomputes an Indian AQI.
- If the API fails, returns malformed data or stale timestamps, the app shows an "unavailable" state with a retry button. It never substitutes invented readings.
- Sample data (`sample_data/air_quality_sample.json`) is synthetic. It is only shown when you tick "Show sample data instead", and is labelled "Sample data — not live" everywhere.
- Recommendations describe the *direction* of likely benefit only. The app cannot measure a change in neighbourhood air or a numerical reduction in personal health risk.

## Setup

Requires Python 3.10+ (developed on 3.12).

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

Opens at http://localhost:8501. No key is needed.

## Tests

```
python -m pytest
```

Tests mock all HTTP and the LLM client; they need no network or API key. They cover the recommendation rules, forecast edge cases (missing values, malformed or empty data, missing timestamps), coordinate and distance validation, API failure handling, the LLM fallbacks, and Streamlit `AppTest` smoke tests.

## Optional: AI explanations

Set both variables to enable the AI explanation and question box. Without them the app shows a note that AI explanations are disabled and uses a template explanation.

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Your Google Gemini API key (create one in [Google AI Studio](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL` | Gemini model name to use (not hardcoded, so you can pick a current one) |

Locally: copy `.env.example` to `.env` and fill in the values (`.env` is git-ignored). Or create `.streamlit/secrets.toml` (also git-ignored):

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "..."
```

How the model is used: it is called only after the rules have produced the plan, through the official `google-genai` SDK (`models.generate_content`), with no tools. It receives a compact context (city and country label, pollutant values and times, forecast limitation, routine answers and the generated recommendations). It never receives coordinates. Questions are limited to 500 characters and treated as untrusted text; responses are length-capped, HTML tags and images are stripped, and output is rendered as plain Markdown. On any failure (rate limit, bad key, timeout, empty reply) the built-in explanation is shown with a short note and no error details.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (check that `.env` and `.streamlit/secrets.toml` are not committed).
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), choose **Create app**, select the repository and set the main file to `app.py`.
3. Optional AI: open **Advanced settings → Secrets** (or the app's **Settings → Secrets**) and paste `GEMINI_API_KEY = "..."` and `GEMINI_MODEL = "..."`.
4. Deploy. No Docker or infrastructure is needed.

## Privacy

- Routine answers, selected place and questions live in Streamlit session state only. Nothing is written to files or a database, and private inputs are not logged.
- Usage statistics are switched off in `.streamlit/config.toml`, which also holds the beige/brown colour theme.
- Only public place-name and forecast lookups are cached (30 minutes). LLM replies are not cached globally.
- Place names are sent to Open-Meteo for geocoding, and the chosen place's coordinates are sent to Open-Meteo for the forecast. With AI enabled, the compact context above is sent to Google's Gemini API.

## Project layout

```
app.py                       Streamlit UI composition
breathewise/data_provider.py Open-Meteo HTTP, validation, sample loader
breathewise/recommendations.py Deterministic rules, preference validation, checklist
breathewise/assistant.py     Optional LLM call and template fallback
breathewise/formatting.py    Small display helpers
tests/                       pytest suite (no network)
sample_data/                 Synthetic, clearly labelled demo data
```

## Assumptions and known limitations

- Forecast windows use the US AQI (falling back to PM2.5); "lower/higher" needs at least 12 valid hourly points, a gap of at least 10 AQI points (5 μg/m³ for PM2.5) and at least 25 %, and windows that do not overlap. Otherwise only general guidance is shown.
- The live API returns the current day plus the next; the chart shows what remains from the current hour, so "48 hours" is usually about 38 to 48 hours.
- Chart pollutant selection and recommendations are English-only.
- The LLM path is covered by mocked tests only; it has not been exercised against a live Gemini key in this repository's test run.
- The Streamlit copy-to-clipboard control is the built-in icon on the checklist code block.
