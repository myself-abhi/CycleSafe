# Austin CycleSafe — Streamlit edition

A risk-decision tool for Austin cyclists, built on 2,463 historical bike-crash records (2010–2017). Three tabs: an insights dashboard, a route planner with a Leaflet drawing canvas, and a saved-rides results gallery.

This is the Streamlit version of the HTML app under the same project. Same risk engine, same Modern Minimal visual language, deployable to Streamlit Community Cloud in a few clicks.

## Quick start (local)

```bash
git clone <your-fork-url>
cd cyclist_risk_app
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 — the app should load in under a second.

## Deploy to Streamlit Community Cloud

1. **Push this folder to GitHub** as a public repo (or private if you connected your account).

   ```bash
   cd cyclist_risk_app
   git init
   git add .
   git commit -m "Austin CycleSafe — initial Streamlit deploy"
   git branch -M main
   git remote add origin https://github.com/<your-username>/austin-cyclesafe.git
   git push -u origin main
   ```

2. Go to https://share.streamlit.io and click **New app**.

3. Fill in the form:
   - **Repository:** `<your-username>/austin-cyclesafe`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** 3.13 (Streamlit Cloud's max as of writing — `runtime.txt` requests 3.14 and Streamlit will pick the closest available; the code is compatible with both).

4. Click **Deploy**. First build takes ~2 minutes (pip install). Subsequent restarts are ~5–10 s.

## Performance notes

The app is tuned for fast cold starts:

- **Single-pass data load.** `bike_crash.csv` is read once at first request, parsed, and aggregated. Result is cached in-memory for the lifetime of the process via `@st.cache_data`. Subsequent reruns reuse the cache — interactions are instant.
- **Pre-computed lookup tables.** All risk-engine cells (4-way, 2-way, 1-way) are built once on first access; risk lookup is an O(1) dict access.
- **Lazy heavy imports.** `folium` and `streamlit_folium` are only imported inside the Plan/Results tab handlers, so first paint of the Home tab doesn't pay their import cost.
- **Charts are cached.** Each Plotly figure on the Home tab is built once via `@st.cache_data` and the same figure object is reused on every rerun.
- **`fileWatcherType = "none"`** in `.streamlit/config.toml` reduces file-system polling overhead.

### Keeping the app warm (sub-1-second wake-up)

Streamlit Community Cloud puts free apps to sleep after ~30 minutes of inactivity. The wake-up cold-start is normally 30–60 s. To keep yours under 1 s of perceived latency, ping the public URL every 5–10 minutes from a free uptime monitor:

| Service | URL | Cost |
|---|---|---|
| **UptimeRobot** | https://uptimerobot.com | Free for 50 monitors, 5-minute interval |
| **Cron-job.org** | https://cron-job.org | Free, configurable from 1 minute |
| **GitHub Actions** | (workflow file in your own repo) | Free for public repos |

UptimeRobot setup: create a new HTTP(S) monitor pointing at `https://your-app.streamlit.app`, set interval to 5 minutes, save. The container stays warm 24/7 and your app is effectively never asleep.

(Streamlit's terms of service permit uptime pings for personal projects. If you ever scale this up, host on Railway / Fly.io / Render / a small VM where there's no sleep at all.)

## Files in this folder

```
cyclist_risk_app/
├── app.py                      # Main Streamlit app
├── bike_crash.csv              # 2,463 Austin bike-crash records
├── requirements.txt            # Python 3.13 / 3.14-compatible pins
├── runtime.txt                 # python-3.14 (Streamlit Cloud reads this)
├── README.md                   # This file
├── .gitignore                  # Standard Python ignores
└── .streamlit/
    └── config.toml             # Theme + perf settings
```

## How the risk engine works (TL;DR)

The user's five inputs (day, time band, street type, location, surface, helmet) get joined against a precomputed conditional-rate table.

1. **4-way exact match** (time × street × location × surface) — used if the matched cell has ≥ 30 crashes.
2. **2-way fallback** (time × street) — used if 4-way is too thin.
3. **1-way fallback** (time only) — used if 2-way is also too thin.
4. **Citywide baseline** — last-resort if even time-of-day is thin.

The serious-injury rate from the matched cell is bucketed against the citywide 10.96% baseline:

- **≤ baseline** → green, "below baseline · safer window"
- **baseline → 1.5× baseline** → amber, "above baseline · ride with caution"
- **> 1.5× baseline** → red, "well above baseline · reconsider"

The drawn route on the Plan tab is purely for visualization and distance — it doesn't drive the risk math (the **Street type** dropdown is the user's explicit declaration of road class). Distance is measured along the freehand polyline using the haversine formula.

## Author

**Abhishek Thadem** · ALY 6040 — Data Mining Applications · Faculty: Prof. Justin Grosz
