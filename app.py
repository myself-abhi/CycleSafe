"""
Austin CycleSafe — Streamlit edition
ALY6040 Project · Abhishek Thadem · Faculty: Prof. Justin Grosz

Mirrors the HTML dashboard with three tabs (Home / Plan / Results), a risk
engine over 2,463 historical Austin bike crashes, freehand route drawing on
a Leaflet map (via folium), and Plotly charts.

Performance notes
-----------------
- Data loading + aggregation is wrapped in @st.cache_data so cold starts
  take one disk read; subsequent renders are instant.
- Heavy libraries (folium, plotly) are imported once at module load — the
  Streamlit script reruns from this point on every interaction, so any work
  outside cached functions runs every time. Keep that minimal.
- For Streamlit Community Cloud's sleep behavior, set up a free uptime ping
  (e.g., UptimeRobot pinging the app URL every 5 minutes) to keep the
  container warm and your wake-up time under 1s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(
    page_title="Austin CycleSafe",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Austin CycleSafe — ALY6040 Project · Abhishek Thadem · "
                 "Historical patterns from 2,463 Austin bike crashes (2010–2017).",
    },
)


# ---------- DESIGN TOKENS — Modern Minimal palette ----------
PRIMARY = "#0F766E"          # teal — brand, headings, route line
PRIMARY_HOVER = "#115E59"
PRIMARY_SOFT = "#F0FDFA"
BG = "#FAFAF9"               # warm off-white canvas
SURFACE = "#FFFFFF"
FG = "#111827"
FG_MUTED = "#4B5563"
BORDER = "#E5E7EB"
SUCCESS = "#16A34A"          # below baseline / safer
WARNING = "#D97706"          # caution
DANGER = "#DC2626"           # well above baseline


# ---------- GLOBAL CSS ----------
st.markdown(
    f"""
<style>
  /* Inter from Google Fonts — single family, two weights */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

  html, body, [class*="css"], .stApp {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: {FG};
    background: {BG};
  }}
  .stApp {{ background: {BG}; }}
  .block-container {{ padding-top: 1.4rem !important; padding-bottom: 1rem !important; max-width: 1400px; }}

  h1, h2, h3, h4 {{ color: {FG}; letter-spacing: -0.01em; font-weight: 600; }}

  /* Headline + KPI styles to mirror the HTML version */
  .acs-kicker {{
    text-transform: uppercase; letter-spacing: 0.10em; font-size: 0.72rem;
    font-weight: 600; color: {PRIMARY}; margin: 0 0 0.3rem 0;
  }}
  .acs-question {{
    font-size: clamp(1.6rem, 2.6vw, 2.2rem); font-weight: 600; line-height: 1.18;
    letter-spacing: -0.02em; color: {FG}; margin: 0 0 0.6rem 0;
  }}
  .acs-answer {{
    font-size: clamp(1rem, 1.5vw, 1.15rem); line-height: 1.55; color: {FG_MUTED};
    margin: 0; max-width: 56ch;
  }}
  .acs-answer .accent {{ color: {DANGER}; font-weight: 600; }}
  .acs-yes-tag {{
    display: inline-block; padding: 2px 10px; background: {DANGER}; color: white;
    font-weight: 600; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    border-radius: 6px; margin-right: 8px; vertical-align: middle;
  }}
  /* KPI card */
  .acs-kpi {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 1rem 1.1rem; min-height: 100px;
  }}
  .acs-kpi.dark {{ background: {FG}; color: white; border-color: {FG}; }}
  .acs-kpi .label {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.10em;
    font-weight: 600; color: {FG_MUTED}; margin-bottom: 0.4rem;
  }}
  .acs-kpi.dark .label {{ color: rgba(255,255,255,0.72); }}
  .acs-kpi .num {{
    font-size: clamp(1.6rem, 2.6vw, 2.2rem); font-weight: 600; line-height: 1.05;
    letter-spacing: -0.02em; color: {FG};
  }}
  .acs-kpi.dark .num {{ color: white; }}
  .acs-kpi .num.signal {{ color: {DANGER}; }}
  .acs-kpi .sub {{ font-size: 0.85rem; color: {FG_MUTED}; margin-top: 0.4rem; }}
  .acs-kpi.dark .sub {{ color: rgba(255,255,255,0.72); }}

  /* Hero risk card (Plan tab) */
  .acs-hero {{
    background: {FG}; color: white; padding: 1.4rem 1.6rem; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
  }}
  .acs-hero .verdict {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px; font-weight: 600;
    font-size: 0.78rem; letter-spacing: 0.08em; margin-bottom: 0.5rem;
  }}
  .acs-hero .verdict.go {{ background: {SUCCESS}; }}
  .acs-hero .verdict.caution {{ background: {WARNING}; }}
  .acs-hero .verdict.stop {{ background: {DANGER}; }}
  .acs-hero .verdict.guard {{ background: {PRIMARY}; }}
  .acs-hero .num {{
    font-size: clamp(2.6rem, 5vw, 4rem); font-weight: 600; line-height: 1;
    letter-spacing: -0.025em; margin: 0.2rem 0 0.4rem 0;
  }}
  .acs-hero .num.calm {{ color: {SUCCESS}; }}
  .acs-hero .num.bad  {{ color: {DANGER}; }}
  .acs-hero .num.warn {{ color: {WARNING}; }}
  .acs-hero .band {{
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.10em; color: rgba(255,255,255,0.85); margin-bottom: 0.5rem;
  }}
  .acs-hero .sub {{ font-size: 0.92rem; color: rgba(255,255,255,0.78); line-height: 1.5; }}

  /* Recommendation alert */
  .acs-rec {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {PRIMARY};
    border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 0.8rem;
  }}
  .acs-rec h4 {{
    color: {PRIMARY}; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.4rem 0;
  }}
  .acs-rec p {{ font-size: 0.92rem; line-height: 1.55; color: {FG}; margin: 0; }}
  .acs-rec.calm    {{ border-left-color: {SUCCESS}; background: #F0FDF4; }}
  .acs-rec.calm h4 {{ color: {SUCCESS}; }}
  .acs-rec.caution    {{ border-left-color: {WARNING}; background: #FFFBEB; }}
  .acs-rec.caution h4 {{ color: {WARNING}; }}
  .acs-rec.danger     {{ border-left-color: {DANGER}; background: #FEF2F2; }}
  .acs-rec.danger h4  {{ color: {DANGER}; }}

  /* Insight card */
  .acs-insight {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 1rem 1.1rem;
  }}
  .acs-insight .num {{
    font-size: 0.7rem; font-weight: 600; color: {PRIMARY};
    letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 0.4rem;
  }}
  .acs-insight .stat {{
    font-size: clamp(1.5rem, 2.4vw, 2rem); font-weight: 600; color: {DANGER};
    line-height: 1; margin-bottom: 0.4rem;
  }}
  .acs-insight .h {{ font-size: 0.95rem; font-weight: 600; color: {FG}; line-height: 1.3; }}

  /* Streamlit chrome — hide everything that breaks the "exact replica" feel */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ display: none; height: 0; }}
  div[data-testid="stToolbar"] {{ display: none !important; }}
  div[data-testid="stDecoration"] {{ display: none !important; }}
  div[data-testid="stStatusWidget"] {{ display: none !important; }}
  .stDeployButton, .stAppDeployButton {{ display: none !important; }}
  /* the floating "streamlitApp" page-name badge */
  [data-testid="stSidebarNav"] {{ display: none !important; }}
  [data-testid="stSidebarNavItems"] {{ display: none !important; }}
  ul[data-testid="stSidebarNavItems"] {{ display: none !important; }}
  div[data-testid="stSidebarNav"] {{ display: none !important; }}
  /* legacy and current selectors for the top-left page-name pill */
  .stApp > header {{ display: none !important; }}
  .viewerBadge_container__1QSob {{ display: none !important; }}
  ._terminalButton_rix23_138 {{ display: none !important; }}
  /* Streamlit Cloud "page name" pill — multiple class hashes seen in the wild */
  [class*="viewerBadge"], [class*="ViewerBadge"] {{ display: none !important; }}
  [class*="terminalButton"] {{ display: none !important; }}
  [class*="manageAppButton"] {{ display: none !important; }}
  [data-testid="stPageNav"] {{ display: none !important; }}
  [data-testid="stToolbarActions"] {{ display: none !important; }}
  /* Streamlit Cloud "Manage app" pill at bottom-right + page-name badge */
  iframe[title="streamlitApp"] {{ display: none !important; }}
  div[data-testid="stAppDeployButton"] {{ display: none !important; }}
  /* The page-name badge specifically — it's the only Streamlit element
     positioned with high z-index above the header */
  body > div:not([data-testid]):not([data-stale]):not([class]):not([id]):empty {{ display: none !important; }}
  /* Tighten outer padding so the layout breathes like the HTML preview */
  .block-container {{ padding-top: 0.6rem !important; padding-left: 1.2rem !important; padding-right: 1.2rem !important; }}

  /* Tabs styling */
  .stTabs [role="tablist"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
  .stTabs [role="tab"] {{
    padding: 12px 18px; font-weight: 600; color: {FG_MUTED};
    border-bottom: 2px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{ color: {FG}; border-bottom-color: {PRIMARY}; }}

  /* Buttons */
  .stButton > button {{
    background: {PRIMARY}; color: white; border: 1px solid {PRIMARY};
    font-weight: 600; border-radius: 6px;
  }}
  .stButton > button:hover {{ background: {PRIMARY_HOVER}; border-color: {PRIMARY_HOVER}; }}

  /* Numbered accordion (mirrors HTML <details class="acc-panel">) */
  div[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE};
    margin-top: 0.55rem;
    overflow: hidden;
  }}
  div[data-testid="stExpander"] summary {{
    padding: 0.55rem 0.85rem !important;
    font-weight: 600 !important;
    color: {FG} !important;
    background: #FAFAFA;
  }}
  div[data-testid="stExpander"] summary p {{ margin: 0; font-size: 0.85rem; }}

  /* Hero "Plain-language verdict" card — top of Plan sidebar */
  .acs-plan-kicker {{
    color: rgba(255,255,255,0.85); text-transform: uppercase;
    letter-spacing: 0.10em; font-size: 0.7rem; font-weight: 600;
  }}
  .acs-divider-row {{
    border-top: 1px solid rgba(240,249,255,0.18);
    margin-top: 0.85rem; padding-top: 0.7rem;
  }}
  .acs-distance-num {{
    font-size: 1.35rem; font-weight: 600; color: white;
    letter-spacing: -0.01em; margin-top: 0.2rem;
  }}
  .acs-mini-help {{
    font-size: 0.78rem; color: {FG_MUTED}; line-height: 1.5;
  }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- DATA LOADING (CACHED) ----------
SEVERITY_SCORE = {
    "Killed": 4,
    "Incapacitating Injury": 3,
    "Non-Incapacitating Injury": 2,
    "Possible Injury": 1,
    "Not Injured": 0,
}
SERIOUS_CUTOFF = 3
MIN_CELL_N = 30
DATA_PATH = Path(__file__).parent / "bike_crash.csv"


@st.cache_data(show_spinner=False)
def load_crash_data() -> pd.DataFrame:
    """Load the Austin bike-crash dataset and derive analysis columns.

    Cached: a one-time disk read on cold start; instant on every rerun.
    """
    df = pd.read_csv(DATA_PATH)
    df["sev"] = df["Crash Severity"].map(SEVERITY_SCORE)
    df["serious"] = (df["sev"] >= SERIOUS_CUTOFF).astype(int)
    df["killed"] = (df["Crash Severity"] == "Killed").astype(int)

    # Derive analysis bins
    def _hr(t):
        try:
            return int(str(int(t)).zfill(4)[:-2])
        except Exception:
            return -1

    def _tband(h):
        if h < 0: return None
        if h < 6: return "Late Night (12–6 AM)"
        if h < 10: return "Morning Rush (6–10 AM)"
        if h < 15: return "Midday (10 AM–3 PM)"
        if h < 19: return "Evening Rush (3–7 PM)"
        return "Night (7 PM–12 AM)"

    def _sb(v):
        try:
            v = int(v)
        except Exception:
            return None
        if v <= 0: return None
        if v <= 25: return "Calm street (≤25 mph)"
        if v <= 35: return "Neighborhood arterial (26–35 mph)"
        if v <= 45: return "Major arterial (36–45 mph)"
        return "High-speed road (46+ mph)"

    def _loc(v):
        if v in ("Intersection", "Intersection Related"): return "At or near an intersection"
        if v == "Driveway Access": return "Driveway / parking access"
        if v == "Non Intersection": return "Mid-block (no intersection)"
        return None

    def _srf(v):
        if v == "Dry": return "Dry"
        if v == "Wet": return "Wet"
        return "Other / Unknown"

    df["hour"] = df["Crash Time"].apply(_hr)
    df["tb"] = df["hour"].apply(_tband)
    df["sb"] = df["Speed Limit"].apply(_sb)
    df["loc"] = df["Intersection Related"].apply(_loc)
    df["srf"] = df["Surface Condition"].apply(_srf)
    return df


@st.cache_data(show_spinner=False)
def crash_aggregates() -> dict:
    """Pre-compute every aggregate the app uses, once at first access."""
    df = load_crash_data()
    out: dict = {
        "total": int(len(df)),
        "baseline": float(df["serious"].mean()),
        "killed": int(df["killed"].sum()),
        "year_min": int(df["Crash Year"].min()),
        "year_max": int(df["Crash Year"].max()),
    }

    def marginal(col, order):
        s = df.dropna(subset=[col]).groupby(col).agg(
            n=("serious", "size"), serious=("serious", "sum"),
        )
        s = s.reindex(order)
        return [
            {"label": idx, "n": int(r["n"]), "serious": int(r["serious"]),
             "rate": float(r["serious"] / r["n"]) if r["n"] else 0.0}
            for idx, r in s.iterrows()
        ]

    out["time_pattern"] = marginal(
        "tb",
        ["Morning Rush (6–10 AM)", "Midday (10 AM–3 PM)", "Evening Rush (3–7 PM)",
         "Night (7 PM–12 AM)", "Late Night (12–6 AM)"],
    )
    out["speed_pattern"] = marginal(
        "sb",
        ["Calm street (≤25 mph)", "Neighborhood arterial (26–35 mph)",
         "Major arterial (36–45 mph)", "High-speed road (46+ mph)"],
    )
    out["location_pattern"] = marginal(
        "loc",
        ["At or near an intersection", "Mid-block (no intersection)",
         "Driveway / parking access"],
    )
    out["surface_pattern"] = [
        {"label": idx, "n": int(r["n"]), "serious": int(r["serious"]),
         "rate": float(r["serious"] / r["n"]) if r["n"] else 0.0}
        for idx, r in df[df["srf"].isin(["Dry", "Wet"])]
        .groupby("srf").agg(n=("serious", "size"), serious=("serious", "sum"))
        .iterrows()
    ]

    # Conditional cells for the risk engine — pivoted dictionaries keyed by joins.
    cells4: dict = {}
    for keys, grp in df.dropna(subset=["tb", "sb", "loc", "srf"]).groupby(["tb", "sb", "loc", "srf"]):
        cells4["|".join(keys)] = {
            "n": int(len(grp)), "serious": int(grp["serious"].sum()),
            "killed": int(grp["killed"].sum()),
        }
    out["cells4"] = cells4

    cells2: dict = {}
    for keys, grp in df.dropna(subset=["tb", "sb"]).groupby(["tb", "sb"]):
        cells2["|".join(keys)] = {
            "n": int(len(grp)), "serious": int(grp["serious"].sum()),
            "killed": int(grp["killed"].sum()),
        }
    out["cells2"] = cells2

    cells1: dict = {}
    for tb_, grp in df.groupby("tb"):
        if tb_ is None:
            continue
        cells1[tb_] = {
            "n": int(len(grp)), "serious": int(grp["serious"].sum()),
            "killed": int(grp["killed"].sum()),
        }
    out["cells1"] = cells1

    out["year_trend"] = [
        {"year": int(idx), "n": int(r["serious_count"]),
         "serious": int(r["serious_sum"]), "killed": int(r["killed_sum"])}
        for idx, r in df.groupby("Crash Year").agg(
            serious_count=("serious", "size"),
            serious_sum=("serious", "sum"),
            killed_sum=("killed", "sum"),
        ).iterrows()
    ]
    return out


# ---------- RISK ENGINE ----------
def compute_risk(tb: str, sb: str, loc: str, sf: str, helmet: str) -> dict:
    """Match the user's conditions against the 4-way / 2-way / 1-way cell stack."""
    D = crash_aggregates()
    key4 = f"{tb}|{sb}|{loc}|{sf}"
    cell = D["cells4"].get(key4)
    level, used_fallback, scope = "4-way", False, ""

    if cell and cell["n"] >= MIN_CELL_N:
        scope = f"{cell['n']} crashes match all four conditions you set."
    else:
        original = cell["n"] if cell else 0
        c2 = D["cells2"].get(f"{tb}|{sb}")
        if c2 and c2["n"] >= MIN_CELL_N:
            cell, level, used_fallback = c2, "2-way (time + street type)", True
            scope = (f"Only {original} crashes matched all four — too thin. "
                     f"Falling back to time + street type ({c2['n']} crashes).")
        else:
            c1 = D["cells1"].get(tb)
            if c1 and c1["n"] >= MIN_CELL_N:
                cell, level, used_fallback = c1, "1-way (time only)", True
                scope = (f"Only {original} matched all four; too thin even for time + street. "
                         f"Showing time-of-day rate ({c1['n']} crashes).")
            else:
                cell = {"n": D["total"],
                        "serious": int(round(D["baseline"] * D["total"])),
                        "killed": D["killed"]}
                level, used_fallback = "baseline (citywide)", True
                scope = (f"Too thin to score these conditions. "
                         f"Showing the citywide baseline ({D['total']} crashes).")

    rate = cell["serious"] / cell["n"] if cell["n"] > 0 else D["baseline"]
    if rate <= D["baseline"]:
        band, label = "calm", "Below baseline · safer window"
    elif rate < D["baseline"] * 1.5:
        band, label = "caution", "Above baseline · ride with caution"
    else:
        band, label = "danger", "Well above baseline · reconsider"

    return {
        "rate": rate, "n": cell["n"], "serious": cell["serious"],
        "killed": cell.get("killed", 0),
        "band": band, "band_label": label, "level": level,
        "used_fallback": used_fallback, "scope": scope,
        "inputs": {"tb": tb, "sb": sb, "loc": loc, "sf": sf, "helmet": helmet},
    }


# ---------- CHART HELPERS ----------
def _baseline_shape(baseline_pct: float, x0=0, x1=1, axis="y"):
    """Return a Plotly shape dict for the dashed baseline line."""
    return dict(
        type="line", x0=x0, x1=x1, y0=baseline_pct, y1=baseline_pct,
        xref="paper", yref="y",
        line=dict(color=FG_MUTED, width=1, dash="dash"),
    )


def _chart_layout(title: str, height: int = 220) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=FG, family="Inter"), x=0, xanchor="left"),
        margin=dict(l=8, r=8, t=36, b=24),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, showlegend=False,
        font=dict(family="Inter", size=11, color=FG_MUTED),
        hoverlabel=dict(bgcolor=FG, font=dict(color="white", family="Inter")),
    )


@st.cache_data(show_spinner=False)
def chart_time_of_day():
    D = crash_aggregates()
    pat = D["time_pattern"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["label"].split(" (")[0] for p in pat],
        y=[p["rate"] * 100 for p in pat],
        mode="lines+markers",
        line=dict(color=DANGER, width=2.5, shape="spline"),
        marker=dict(color=DANGER, size=8, line=dict(color="white", width=2)),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.10)",
        hovertemplate="%{x}<br>%{y:.1f}% serious-injury<extra></extra>",
    ))
    fig.add_shape(**_baseline_shape(D["baseline"] * 100))
    fig.update_layout(**_chart_layout("Risk by time of day — evenings 2× midday"))
    fig.update_yaxes(ticksuffix="%", gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


@st.cache_data(show_spinner=False)
def chart_speed():
    D = crash_aggregates()
    pat = D["speed_pattern"]
    colors = [DANGER if p["rate"] > D["baseline"] else SUCCESS for p in pat]
    fig = go.Figure(go.Bar(
        x=[p["rate"] * 100 for p in pat],
        y=[p["label"].split(" (")[1].rstrip(")").replace("mph", "mph") for p in pat],
        orientation="h",
        marker=dict(color=colors), text=[f"{p['rate']*100:.1f}%" for p in pat],
        textposition="outside", textfont=dict(color=FG, size=11),
        hovertemplate="%{y}<br>%{x:.1f}% serious-injury<extra></extra>",
    ))
    fig.update_layout(**_chart_layout("Risk by street speed — 45 mph triples it"))
    fig.update_xaxes(ticksuffix="%", gridcolor=BORDER, range=[0, 24])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


@st.cache_data(show_spinner=False)
def chart_year():
    D = crash_aggregates()
    yrs = D["year_trend"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[y["year"] for y in yrs], y=[y["n"] for y in yrs],
        name="All crashes", marker=dict(color=BORDER), hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=[y["year"] for y in yrs], y=[y["serious"] for y in yrs],
        name="Serious or fatal", marker=dict(color=DANGER),
        hovertemplate="%{x}<br>Serious: %{y}<extra></extra>",
    ))
    fig.update_layout(**_chart_layout("Fatalities by year — 2017 the worst"),
                      barmode="overlay", legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_xaxes(showgrid=False, dtick=1)
    return fig


@st.cache_data(show_spinner=False)
def chart_severity():
    D = crash_aggregates()
    total, killed = D["total"], D["killed"]
    incap = int(round(D["baseline"] * total)) - killed
    nonincap = total - killed - incap
    fig = go.Figure(go.Pie(
        labels=["Killed", "Incapacitating injury", "All other crashes"],
        values=[killed, incap, nonincap],
        marker=dict(colors=[DANGER, WARNING, BORDER], line=dict(color="white", width=2)),
        hole=0.62, textinfo="none",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_chart_layout("Severity breakdown — 1-in-9 serious"),
                      legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center"))
    return fig


# ---------- HEADER + TABS ----------
st.markdown(
    f"""
    <div style="border-bottom: 1px solid {BORDER}; padding-bottom: 10px; margin-bottom: 6px;">
      <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <div>
          <div style="font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em;">
            AUSTIN <span style="color: {PRIMARY};">CYCLESAFE</span>
          </div>
          <div style="font-size: 0.78rem; color: {FG_MUTED};">Plan a ride · see the historical danger · decide</div>
        </div>
        <div style="font-size: 0.72rem; font-weight: 600; color: {PRIMARY};
                    letter-spacing: 0.10em; text-transform: uppercase;">
          v1.0 · 2,463 crashes · 2010–2017
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_plan, tab_results = st.tabs(["01 · Home", "02 · Plan your ride", "03 · Results"])


# ============================================================
# TAB 1 — HOME (insights dashboard)
# ============================================================
with tab_home:
    D = crash_aggregates()
    wet_rate = next(p["rate"] for p in D["surface_pattern"] if p["label"] == "Wet")
    night_rate = next(p["rate"] for p in D["time_pattern"] if p["label"].startswith("Night"))
    midday_rate = next(p["rate"] for p in D["time_pattern"] if p["label"].startswith("Midday"))
    fast_rate = next(p["rate"] for p in D["speed_pattern"] if p["label"].startswith("High-speed"))
    calm_rate = next(p["rate"] for p in D["speed_pattern"] if p["label"].startswith("Calm"))
    midblock_rate = next(p["rate"] for p in D["location_pattern"] if "Mid-block" in p["label"])
    isx_rate = next(p["rate"] for p in D["location_pattern"] if "intersection" in p["label"])

    hero, kpis = st.columns([1.4, 1])
    with hero:
        st.markdown('<p class="acs-kicker">The question Austin&rsquo;s cyclists asked</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="acs-question">Is the city doing enough to protect people on bikes?</h1>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="acs-answer"><span class="acs-yes-tag">No</span>'
            f'Cyclists were right. Eight years of records show '
            f'<span class="accent">serious injuries don&rsquo;t fall</span>, '
            f'and the danger spikes at night, on faster streets, on wet roads, and mid-block.</p>',
            unsafe_allow_html=True,
        )

    with kpis:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f'<div class="acs-kpi dark"><div class="label">Crashes recorded</div>'
                f'<div class="num">{D["total"]:,}</div>'
                f'<div class="sub">{D["year_min"]}–{D["year_max"]}</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="acs-kpi"><div class="label">Serious-injury rate</div>'
                f'<div class="num signal">{D["baseline"]*100:.2f}%</div>'
                f'<div class="sub">{int(round(D["baseline"]*D["total"]))} + {D["killed"]} fatal</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            worst = max(D["year_trend"], key=lambda y: y["killed"])
            st.markdown(
                f'<div class="acs-kpi"><div class="label">Cyclists killed</div>'
                f'<div class="num signal">{D["killed"]}</div>'
                f'<div class="sub">{worst["year"]} was worst ({worst["killed"]} killed)</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Charts grid (2×2)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(chart_time_of_day(), use_container_width=True, config={"displayModeBar": False})
    with c2: st.plotly_chart(chart_speed(),       use_container_width=True, config={"displayModeBar": False})
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(chart_year(),        use_container_width=True, config={"displayModeBar": False})
    with c4: st.plotly_chart(chart_severity(),    use_container_width=True, config={"displayModeBar": False})

    # Insight strip
    i1, i2, i3, i4 = st.columns(4)
    insights = [
        ("01 · TIME", f"~{night_rate/midday_rate:.1f}×", "Ride midday, not at night."),
        ("02 · STREET", f"~{fast_rate/calm_rate:.1f}×", "Take the calm streets."),
        ("03 · LOCATION", f"~{midblock_rate/isx_rate:.1f}×", "Mid-block is the worst."),
        ("04 · WEATHER", f"~{wet_rate/D['baseline']:.1f}×", "Don't ride right after rain."),
    ]
    for col, (num, stat, head) in zip([i1, i2, i3, i4], insights):
        with col:
            st.markdown(
                f'<div class="acs-insight"><div class="num">{num}</div>'
                f'<div class="stat">{stat}</div><div class="h">{head}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 2 — PLAN YOUR RIDE
# ============================================================
# Initialize session state for drawn route + saved rides
if "drawn_route" not in st.session_state:
    st.session_state.drawn_route = None  # GeoJSON Feature or None
if "saved_rides" not in st.session_state:
    st.session_state.saved_rides = []
if "current_risk" not in st.session_state:
    st.session_state.current_risk = None


def _haversine_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    lat1, lng1 = np.radians(a[0]), np.radians(a[1])
    lat2, lng2 = np.radians(b[0]), np.radians(b[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def _route_coords(geojson_feature: dict | None) -> list:
    """Extract a flat [[lng, lat], ...] list from a LineString OR Polygon feature."""
    if not geojson_feature: return []
    geom = geojson_feature.get("geometry", {})
    gt = geom.get("type")
    if gt == "LineString":
        return list(geom.get("coordinates", []))
    if gt == "Polygon":
        # Outer ring only, since cyclists ride the perimeter
        rings = geom.get("coordinates", [])
        return list(rings[0]) if rings else []
    return []


def _route_distance_meters(geojson_feature: dict | None) -> float:
    coords = _route_coords(geojson_feature)
    total = 0.0
    for i in range(1, len(coords)):
        # GeoJSON is [lng, lat]
        a = (coords[i-1][1], coords[i-1][0])
        b = (coords[i][1], coords[i][0])
        total += _haversine_meters(a, b)
    return total


with tab_plan:
    st.markdown('<p class="acs-kicker">Plan your ride</p>', unsafe_allow_html=True)

    # Form-state defaults — initialised once so the verdict can render at the
    # top of the sidebar BEFORE the form expander below it (mirrors HTML).
    _form_defaults = {
        "f_day": "Friday",
        "f_tb": "Evening Rush (3–7 PM)",
        "f_sb": "Neighborhood arterial (26–35 mph)",
        "f_loc": "At or near an intersection",
        "f_sf": "Dry",
        "f_helmet": "Yes",
    }
    for _k, _v in _form_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    sidebar, mapcol = st.columns([1, 2])

    # =========================================================
    # Sidebar — verdict ABOVE form, then numbered accordion sections
    # =========================================================
    with sidebar:
        # Compute risk from CURRENT session-state values
        risk = compute_risk(
            st.session_state.f_tb, st.session_state.f_sb, st.session_state.f_loc,
            st.session_state.f_sf, st.session_state.f_helmet,
        )
        st.session_state.current_risk = risk
        agg = crash_aggregates()
        meters = _route_distance_meters(st.session_state.drawn_route)
        miles = meters / 1609.344
        km = meters / 1000.0

        verdict_map = {
            "calm":    ("go",      "GOOD TIME TO RIDE",   "calm"),
            "caution": ("caution", "RIDE WITH CAUTION",   "warn"),
            "danger":  ("stop",    "WAIT IF YOU CAN",     "bad"),
        }
        v_class, v_text, num_class = verdict_map.get(risk["band"], ("guard", "READING DATA", ""))
        delta = (risk["rate"] - agg["baseline"]) * 100
        delta_color = DANGER if delta >= 0 else SUCCESS

        # ---- (1) Hero risk card — verdict + number + distance row ----
        st.markdown(
            f"""
            <div class="acs-hero">
              <div class="acs-plan-kicker">Plain-language verdict</div>
              <span class="verdict {v_class}" style="margin: 6px 0;">● {v_text}</span>
              <div class="num {num_class}">{risk['rate']*100:.1f}%</div>
              <div class="band">{risk['band_label'].upper()}</div>
              <div class="sub">
                Of <b>{risk['n']:,}</b> matching crashes, <b>{risk['serious']}</b> were serious
                ({risk['killed']} fatal). Baseline: {agg['baseline']*100:.2f}%.
                You sit <b style="color: {delta_color};">{'+' if delta>=0 else ''}{delta:.1f} pts</b>
                from baseline ({risk['rate']/agg['baseline']:.2f}×).
              </div>
              <div class="acs-divider-row">
                <div class="acs-plan-kicker">Route distance</div>
                <div class="acs-distance-num">{miles:.2f} mi</div>
                <div style="font-size: 0.78rem; color: rgba(255,255,255,0.7);">
                  {km:.2f} km · {len(_route_coords(st.session_state.drawn_route))} route points
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- (2) Recommendation banner ----
        rec_class = "guard" if risk["used_fallback"] and risk["level"].startswith("baseline") else risk["band"]
        if risk["band"] == "calm":
            rec_title = "Comparatively safer window"
            rec_body = (f"These conditions sit at or below the {agg['baseline']*100:.2f}% baseline. "
                        f"Helmet on, lights on, stay visible — \"safer\" isn't \"safe\".")
        elif risk["band"] == "caution":
            rec_title = "Above baseline — ride with caution"
            rec_body = (f"Risk is elevated ({risk['rate']*100:.1f}% vs {agg['baseline']*100:.2f}%). "
                        f"Watch for right-hooks, signal turns, assume drivers don't see you.")
        else:
            rec_title = "Well above baseline — reconsider"
            ratio = risk["rate"] / agg["baseline"]
            rec_body = (f"These conditions historically put cyclists in the hospital "
                        f"{ratio:.1f}× as often as the city average. "
                        f"Shift to a calmer street, earlier time, or dry surface if you can.")
        if st.session_state.f_helmet == "No" and risk["band"] != "calm":
            rec_body += "  Helmets correlate with a 5.6% serious-injury rate vs 10.6% unhelmeted."
        st.markdown(
            f'<div class="acs-rec {rec_class}"><h4>{rec_title}</h4><p>{rec_body}</p></div>',
            unsafe_allow_html=True,
        )

        # =====================================================
        # Numbered accordion sections — match the HTML preview
        # =====================================================

        # ---- 01 · Conditions (open) ----
        with st.expander("01 · Conditions", expanded=True):
            ca, cb = st.columns(2)
            with ca:
                st.selectbox("Day",
                    ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"], key="f_day")
            with cb:
                st.selectbox("Time band", [
                    "Morning Rush (6–10 AM)", "Midday (10 AM–3 PM)", "Evening Rush (3–7 PM)",
                    "Night (7 PM–12 AM)", "Late Night (12–6 AM)",
                ], key="f_tb")
            st.selectbox("Street type", [
                "Calm street (≤25 mph)", "Neighborhood arterial (26–35 mph)",
                "Major arterial (36–45 mph)", "High-speed road (46+ mph)",
            ], key="f_sb")
            st.selectbox("Location on road", [
                "At or near an intersection", "Mid-block (no intersection)",
                "Driveway / parking access",
            ], key="f_loc")
            cc, cd = st.columns(2)
            with cc:
                st.selectbox("Surface", ["Dry", "Wet", "Other / Unknown"], key="f_sf")
            with cd:
                st.selectbox("Helmet", ["Yes", "No"], key="f_helmet")

        # ---- 02 · Pace & calories (collapsed) ----
        with st.expander("02 · Pace & calories", expanded=False):
            pa, pb = st.columns(2)
            with pa:
                pace = st.number_input("Pace (min/mi)", min_value=4.0, max_value=20.0,
                                       value=9.0, step=0.5, key="f_pace")
            with pb:
                weight = st.number_input("Weight (lb)", min_value=70, max_value=350,
                                         value=160, step=5, key="f_weight")
            tot_min = pace * miles
            kcal = int(round(0.49 * weight * (tot_min / 60.0) * 10))  # ~10 MET cycling
            mm, ss = int(tot_min), int(round((tot_min - int(tot_min)) * 60))
            ra, rb = st.columns(2)
            with ra:
                st.markdown(
                    f"<div class='acs-kpi'><div class='label'>Time</div>"
                    f"<div class='num' style='font-size:1.3rem;'>{mm}:{ss:02d}</div></div>",
                    unsafe_allow_html=True)
            with rb:
                st.markdown(
                    f"<div class='acs-kpi'><div class='label'>Calories</div>"
                    f"<div class='num' style='font-size:1.3rem;'>{kcal:,}</div></div>",
                    unsafe_allow_html=True)

        # ---- 03 · Save & export (open) ----
        with st.expander("03 · Save & export", expanded=True):
            sv, ex = st.columns(2)
            with sv:
                if st.button("💾 Save ride", use_container_width=True,
                             disabled=meters < 1, key="btn_save_ride"):
                    if st.session_state.drawn_route:
                        st.session_state.saved_rides.append({
                            "name": f"Ride {len(st.session_state.saved_rides)+1}",
                            "geojson": st.session_state.drawn_route,
                            "distance_m": meters,
                            "risk": st.session_state.current_risk,
                        })
                        st.success(f"Saved ride #{len(st.session_state.saved_rides)}")
            with ex:
                export = {
                    "schema": "cyclesafe.streamlit.v1",
                    "current_route": st.session_state.drawn_route,
                    "current_risk": st.session_state.current_risk,
                    "saved_rides": st.session_state.saved_rides,
                }
                st.download_button(
                    "⬇ Export JSON",
                    data=json.dumps(export, indent=2, default=str),
                    file_name="cyclesafe_export.json",
                    mime="application/json",
                    use_container_width=True,
                    key="btn_export_json",
                )
            if st.button("⨯ Clear current route", use_container_width=True,
                         key="btn_clear_route"):
                st.session_state.drawn_route = None
                st.rerun()

            # Real "View results" button — finds the Streamlit tab via JS
            # (parent doc, since components are iframed) and clicks it.
            saved_n = len(st.session_state.saved_rides)
            view_label = (f"📊 View results ({saved_n} saved) →"
                          if saved_n else "📊 View results →")
            import streamlit.components.v1 as components
            components.html(
                f"""
                <button id="view-results-jump"
                        style="width:100%; padding: 8px 12px; margin-top: 8px;
                               background: {PRIMARY}; color: white;
                               border: 1px solid {PRIMARY}; border-radius: 6px;
                               font-weight: 600; font-family: Inter, sans-serif;
                               cursor: pointer;">
                  {view_label}
                </button>
                <script>
                  document.getElementById('view-results-jump').onclick = () => {{
                    const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                    if (tabs.length >= 3) tabs[2].click();
                  }};
                </script>
                """,
                height=52,
            )

    # =========================================================
    # Map column — fills the right 2/3 of the viewport
    # =========================================================
    with mapcol:
        # Brief drawing instructions right above the map
        st.markdown(
            f"""
            <div style="background: {SURFACE}; border: 1px solid {BORDER};
                        border-left: 3px solid {PRIMARY}; border-radius: 6px;
                        padding: 8px 12px; margin-bottom: 8px; font-size: 0.82rem;
                        color: {FG_MUTED}; line-height: 1.45;">
              <b style="color:{FG};">How to draw:</b> click the
              <b style="color:{PRIMARY};">polyline</b> icon (top-left of map) for an
              open path — <i>double-click last point to finish</i>. Or click the
              <b style="color:{PRIMARY};">polygon</b> icon for a closed loop —
              <i>click your start point to close it</i>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium

        center = [30.2672, -97.7431]
        m = folium.Map(location=center, zoom_start=14, control_scale=True,
                       tiles="OpenStreetMap")
        Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": {"shapeOptions": {"color": PRIMARY, "weight": 5,
                                              "opacity": 0.95}},
                # Polygon mode lets the user close a loop by clicking the start point
                "polygon": {"shapeOptions": {"color": PRIMARY, "weight": 5,
                                             "opacity": 0.95, "fillColor": PRIMARY,
                                             "fillOpacity": 0.10}},
                "circle": False, "rectangle": False,
                "marker": False, "circlemarker": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        # Re-render the user's drawn route after Streamlit reruns
        _persist_coords = _route_coords(st.session_state.drawn_route)
        if _persist_coords:
            folium.PolyLine(
                [[c[1], c[0]] for c in _persist_coords],
                color=PRIMARY, weight=5, opacity=0.95,
            ).add_to(m)
        out = st_folium(m, height=720, width=None,
                        returned_objects=["last_active_drawing"], key="plan_map")
        if out and out.get("last_active_drawing"):
            new_route = out["last_active_drawing"]
            # Only update + rerun when the route actually changed, to avoid loops
            if new_route != st.session_state.drawn_route:
                st.session_state.drawn_route = new_route
                st.rerun()


# ============================================================
# Ride-card renderer — defined BEFORE Tab 3 so the calls below work.
# Lazy-imports folium + streamlit_folium so cold-start time stays minimal
# when the user lands on Home and never visits Plan/Results.
# ============================================================
def _render_ride_card(title: str, geojson: dict | None, meters: float, risk: dict | None) -> None:
    """One self-contained ride card: header strip + map + summary + JSON drawer.

    Uses a hand-rolled Leaflet HTML component (not folium) so we have full
    control over fit_bounds — the streamlit-folium pipeline was dropping the
    fit_bounds JS call on small embedded maps and leaving them at world zoom.
    """
    import streamlit.components.v1 as components

    risk = risk or {}
    miles = meters / 1609.344 if meters else 0
    band = risk.get("band", "calm")
    band_label = risk.get("band_label", "—")
    inputs = risk.get("inputs", {})
    num_class = {"calm": "calm", "caution": "warn", "danger": "bad"}.get(band, "")

    st.markdown(
        f"""
        <div style="background:{PRIMARY}; color:white; padding: 10px 16px; border-radius: 8px 8px 0 0;
                    display:flex; justify-content:space-between; align-items:baseline; margin-top: 14px;">
          <span style="font-weight: 600; letter-spacing: 0.04em;">{title}</span>
          <span style="font-size: 0.72rem; letter-spacing: 0.10em; text-transform: uppercase;
                       opacity: 0.85;">
            {inputs.get('tb', '—')} · {inputs.get('sf', '—')}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    map_col, sum_col = st.columns([1, 1])
    with map_col:
        coords = _route_coords(geojson)
        if coords:
            # Build [lat, lng] pairs for Leaflet
            ll = [[float(c[1]), float(c[0])] for c in coords]
            ll_json = json.dumps(ll)
            start = ll[0]; end = ll[-1]
            map_id = ("m_" + "".join(ch for ch in title if ch.isalnum())
                      + str(abs(hash(ll_json)) % 100000))
            map_height = 320
            html = f"""
            <link rel="stylesheet"
                  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
              #{map_id} {{ height: {map_height}px; width: 100%;
                         border-radius: 0 0 8px 8px; }}
            </style>
            <div id="{map_id}"></div>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
              (function() {{
                var coords = {ll_json};
                var map = L.map("{map_id}", {{
                  scrollWheelZoom: false, zoomControl: true
                }});
                L.tileLayer(
                  "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
                  {{ attribution: "&copy; OpenStreetMap contributors",
                     maxZoom: 19 }}
                ).addTo(map);
                var line = L.polyline(coords, {{
                  color: "{PRIMARY}", weight: 5, opacity: 0.95
                }}).addTo(map);
                L.circleMarker({json.dumps(start)}, {{
                  radius: 6, color: "white", weight: 2,
                  fillColor: "{SUCCESS}", fillOpacity: 1.0
                }}).bindTooltip("Start").addTo(map);
                L.circleMarker({json.dumps(end)}, {{
                  radius: 6, color: "white", weight: 2,
                  fillColor: "{DANGER}", fillOpacity: 1.0
                }}).bindTooltip("End").addTo(map);
                // Defer fitBounds slightly so the iframe has its real size
                setTimeout(function() {{
                  map.invalidateSize();
                  map.fitBounds(line.getBounds(), {{ padding: [20, 20] }});
                }}, 50);
              }})();
            </script>
            """
            components.html(html, height=map_height + 4)
        else:
            st.info("No route drawn for this ride yet.")
    with sum_col:
        st.markdown(
            f"""
            <div class="acs-hero" style="padding: 1rem 1.2rem;">
              <span style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.10em;
                           text-transform: uppercase; color: rgba(255,255,255,0.85);">
                Should you ride?
              </span>
              <div class="num {num_class}" style="font-size: 2.2rem;">{risk.get('rate',0)*100:.1f}%</div>
              <div class="band">{band_label.upper()}</div>
              <div class="sub" style="margin-top: 0.4rem;">
                Of <b>{risk.get('n', 0):,}</b> matching crashes,
                <b>{risk.get('serious', 0)}</b> were serious ({risk.get('killed', 0)} fatal).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size: 0.85rem; line-height: 1.55; margin-top: 8px;'>"
            f"<b>Distance:</b> {miles:.2f} mi ({meters/1000:.2f} km)<br>"
            f"<b>Street:</b> {inputs.get('sb', '—')}<br>"
            f"<b>Location:</b> {inputs.get('loc', '—')}<br>"
            f"<b>Helmet:</b> {inputs.get('helmet', '—')}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with st.expander("Full JSON"):
        st.json({
            "title": title, "distance_m": meters, "distance_mi": round(miles, 3),
            "risk": risk, "geojson": geojson,
        })


# ============================================================
# TAB 3 — RESULTS
# ============================================================
with tab_results:
    st.markdown('<p class="acs-kicker">Your rides at a glance</p>', unsafe_allow_html=True)
    saved = st.session_state.saved_rides
    if not saved and not st.session_state.drawn_route:
        st.markdown(
            f'<div class="acs-rec"><h4>Nothing saved yet</h4>'
            f'<p>Switch to the <b>Plan your ride</b> tab, draw a route on the map, '
            f'and click <b>Save ride</b> to add it here for comparison.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # Current in-progress ride first (if any)
        if st.session_state.drawn_route:
            current_meters = _route_distance_meters(st.session_state.drawn_route)
            risk = st.session_state.current_risk or {"rate": 0, "band": "calm",
                                                     "band_label": "—", "n": 0,
                                                     "serious": 0, "killed": 0,
                                                     "inputs": {}}
            _render_ride_card(
                "CURRENT RIDE (in progress)",
                st.session_state.drawn_route, current_meters, risk,
            )

        for i, ride in enumerate(saved):
            _render_ride_card(
                f"SAVED · {ride['name']}",
                ride["geojson"], ride["distance_m"], ride["risk"],
            )

    # Footer
    st.markdown(
        f"<div style='border-top: 1px solid {BORDER}; padding-top: 10px; margin-top: 16px;"
        f"font-size: 0.72rem; color: {FG_MUTED}; letter-spacing: 0.04em;'>"
        f"Austin CycleSafe · ALY6040 Project · Abhishek Thadem · "
        f"Historical patterns from 2010–2017 only.</div>",
        unsafe_allow_html=True,
    )


# (Ride-card renderer is defined above, before Tab 3.)
