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
# Two accent colors only: TEAL (brand + safer + buttons) and RED (risk).
# PRIMARY is the lighter button teal; SUCCESS is the darker teal used for
# chart "below baseline" bars and other "safer" indicators that benefit from
# a touch more contrast against the white card background.
PRIMARY = "#0F766E"          # teal — brand, buttons, headings, route line
PRIMARY_HOVER = "#115E59"    # darker teal — hover states + SUCCESS shade
PRIMARY_SOFT = "#F0FDFA"     # very light teal tint for soft backgrounds
BG = "#FAFAF9"               # warm off-white canvas
SURFACE = "#FFFFFF"
FG = "#111827"
FG_MUTED = "#4B5563"
BORDER = "#E5E7EB"
SUCCESS = "#115E59"          # safer / below baseline — darker teal (per user)
DANGER = "#DC2626"           # above baseline / risk  (Tailwind red-600)
# Soft (10% alpha) variants — derived from the main accents.
SUCCESS_SOFT = "rgba(17,94,89,0.10)"    # #115E59 at 10% alpha (darker teal)
DANGER_SOFT = "rgba(220,38,38,0.10)"    # #DC2626 at 10% alpha (red)
SUCCESS_TINT = "#F0FDFA"     # very light teal tint behind safer banner
DANGER_TINT = "#FEF2F2"      # background tint behind danger recommendations
WARNING = PRIMARY            # alias kept so legacy refs don't break

# ===== Single source of truth for ALL layout heights =====
# Python and CSS both read from these constants. Change a value here and the
# corresponding region resizes across the entire app — no drift, no gaps.
#
# Home tab — chart cards (4× in 2x2 grid)
CHART_INNER_HEIGHT = 270    # Plotly figure pixel height
CHART_CARD_HEIGHT = 280     # outer card pixel height (10px buffer)
#
# Plan tab — map iframe (responsive)
PLAN_MAP_MIN_PX = 640       # floor on small laptops
PLAN_MAP_IDEAL_VH = 80      # target % of viewport on typical desktops
PLAN_MAP_MAX_PX = 900       # ceiling on 4K monitors
#
# Results tab — each ride card (map iframe + dark verdict panel)
RIDE_CARD_HEIGHT = 320      # both map iframe and dark panel use this
RIDE_CARD_IFRAME_BUFFER = 4 # extra pixels for components.html iframe overhead


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
  /* IMPORTANT overrides — Streamlit's default h1/p styles are huge and would
     otherwise win the cascade. Lock the size to a single readable line. */
  .acs-question, h1.acs-question, [data-testid="stMarkdown"] .acs-question {{
    font-size: clamp(1rem, 1.25vw, 1.2rem) !important;
    font-weight: 600 !important; line-height: 1.3 !important;
    letter-spacing: -0.01em !important; color: {FG} !important;
    margin: 0 0 0.4rem 0 !important; padding: 0 !important;
  }}
  .acs-answer, p.acs-answer, [data-testid="stMarkdown"] .acs-answer {{
    font-size: clamp(0.78rem, 0.92vw, 0.88rem) !important;
    line-height: 1.5 !important; color: {FG_MUTED} !important;
    margin: 0 !important; max-width: 70ch !important;
  }}
  .acs-answer .accent {{ color: {DANGER}; font-weight: 600; }}
  .acs-yes-tag {{
    display: inline-block; padding: 2px 10px; background: {DANGER}; color: white;
    font-weight: 600; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    border-radius: 6px; margin-right: 8px; vertical-align: middle;
  }}
  /* KPI card — tightened for single-page fit */
  .acs-kpi {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.7rem 0.9rem; min-height: 88px;
  }}
  .acs-kpi.dark {{ background: {FG}; color: white; border-color: {FG}; }}
  .acs-kpi .label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.10em;
    font-weight: 600; color: {FG_MUTED}; margin-bottom: 0.3rem;
  }}
  .acs-kpi.dark .label {{ color: rgba(255,255,255,0.72); }}
  .acs-kpi .num {{
    font-size: clamp(1.4rem, 2.2vw, 1.9rem); font-weight: 600; line-height: 1.05;
    letter-spacing: -0.02em; color: {FG};
  }}
  .acs-kpi.dark .num {{ color: white; }}
  .acs-kpi .num.signal {{ color: {DANGER}; }}
  .acs-kpi .sub {{ font-size: 0.78rem; color: {FG_MUTED}; margin-top: 0.3rem; }}
  .acs-kpi.dark .sub {{ color: rgba(255,255,255,0.72); }}

  /* Hero risk card (Plan tab) — compact for single-page fit */
  .acs-hero {{
    background: {FG}; color: white; padding: 0.85rem 1.1rem; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
  }}
  .acs-hero .verdict {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 11px; border-radius: 999px; font-weight: 600;
    font-size: 0.7rem; letter-spacing: 0.08em; margin-bottom: 0.3rem;
  }}
  .acs-hero .verdict.go {{ background: {SUCCESS}; }}
  .acs-hero .verdict.caution {{ background: {PRIMARY}; }}
  .acs-hero .verdict.stop {{ background: {DANGER}; }}
  .acs-hero .verdict.guard {{ background: {PRIMARY}; }}
  .acs-hero .num {{
    font-size: clamp(1.8rem, 3.2vw, 2.4rem); font-weight: 600; line-height: 1;
    letter-spacing: -0.025em; margin: 0.1rem 0 0.25rem 0;
  }}
  .acs-hero .num.calm {{ color: {SUCCESS}; }}
  .acs-hero .num.bad  {{ color: {DANGER}; }}
  .acs-hero .num.warn {{ color: white; }}
  .acs-hero .band {{
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.10em; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem;
  }}
  .acs-hero .sub {{ font-size: 0.78rem; color: rgba(255,255,255,0.78); line-height: 1.45; }}

  /* Recommendation alert — compact */
  .acs-rec {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {PRIMARY};
    border-radius: 6px; padding: 0.6rem 0.85rem; margin-top: 0.55rem;
  }}
  .acs-rec h4 {{
    color: {PRIMARY}; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.25rem 0;
  }}
  .acs-rec p {{ font-size: 0.82rem; line-height: 1.45; color: {FG}; margin: 0; }}
  .acs-rec.calm    {{ border-left-color: {SUCCESS}; background: {SUCCESS_TINT}; }}
  .acs-rec.calm h4 {{ color: {SUCCESS}; }}
  /* "caution" uses teal — same brand color, lighter context */
  .acs-rec.caution    {{ border-left-color: {PRIMARY}; background: {PRIMARY_SOFT}; }}
  .acs-rec.caution h4 {{ color: {PRIMARY}; }}
  .acs-rec.danger     {{ border-left-color: {DANGER}; background: {DANGER_TINT}; }}
  .acs-rec.danger h4  {{ color: {DANGER}; }}

  /* Insight card — tightened */
  .acs-insight {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.7rem 0.9rem;
  }}
  .acs-insight .num {{
    font-size: 0.68rem; font-weight: 600; color: {PRIMARY};
    letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 0.3rem;
  }}
  .acs-insight .stat {{
    font-size: clamp(1.3rem, 2vw, 1.7rem); font-weight: 600; color: {DANGER};
    line-height: 1; margin-bottom: 0.3rem;
  }}
  .acs-insight .h {{ font-size: 0.85rem; font-weight: 600; color: {FG}; line-height: 1.3; }}

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
  /* Tight outer padding so the layout breathes like the HTML preview */
  .block-container {{
    padding-top: 0.5rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 1600px;
  }}
  /* Tighten Streamlit's default vertical block gap so cards sit closer */
  div[data-testid="stVerticalBlock"] {{ gap: 0.45rem !important; }}
  div[data-testid="stHorizontalBlock"] {{ gap: 0.6rem !important; }}

  /* On mobile (≤ 768px), Streamlit stacks columns vertically — let map shrink */
  @media (max-width: 768px) {{
    iframe[title^="streamlit_folium"] {{
      height: clamp(360px, 50vh, 520px) !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"]) {{
      min-height: auto !important;
    }}
  }}

  /* Responsive map height — fills the Plan tab vertical space. Reads from
     the PLAN_MAP_* design tokens so any height change happens in one place. */
  iframe[title^="streamlit_folium"] {{
    height: clamp({PLAN_MAP_MIN_PX}px, {PLAN_MAP_IDEAL_VH}vh, {PLAN_MAP_MAX_PX}px) !important;
    margin-bottom: 0 !important;
    display: block;
  }}
  /* Kill the empty space Streamlit injects below an iframe block */
  div[data-testid="stIFrame"] {{ margin-bottom: 0 !important; padding-bottom: 0 !important; }}

  /* ===== Plan tab column-height matching =====
     The Plan tab's two columns (sidebar + map) must end at the same vertical
     point so the action bar can sit flush below both. Three rules below: */
  /* 1. The columns row gets the same height as the map iframe. */
  div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"]) {{
    min-height: clamp({PLAN_MAP_MIN_PX}px, {PLAN_MAP_IDEAL_VH}vh, {PLAN_MAP_MAX_PX}px);
    align-items: stretch !important;
  }}
  /* 2. Each column inside that row stretches to fill the row height. */
  div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"])
    > div[data-testid="column"] {{
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
  }}
  /* 3. The sidebar's vertical block becomes a flex column; its LAST border-
        wrapped container grows to absorb leftover space, keeping all three
        cards visually filling the column. */
  div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"])
    > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {{
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
  }}
  div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"])
    > div[data-testid="column"] > div[data-testid="stVerticalBlock"]
    > div[data-testid="stVerticalBlockBorderWrapper"]:last-of-type {{
    flex: 1 1 auto;
  }}
  /* Tighten the gap between the columns row and the action bar below */
  div[data-testid="stHorizontalBlock"]:has(iframe[title^="streamlit_folium"])
    + div {{
    margin-top: 4px !important;
  }}

  /* Tabs styling */
  .stTabs [role="tablist"] {{
    gap: 0;
    border-bottom: 1px solid {BORDER};
    /* Sticky tab nav — stays visible right below the header on scroll */
    position: sticky;
    top: 64px;
    background: {BG};
    z-index: 99;
    padding: 6px 0 0 0;
    margin: 0 !important;
  }}
  .stTabs [role="tab"] {{
    padding: 12px 18px; font-weight: 600; color: {FG_MUTED};
    border-bottom: 2px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{ color: {FG}; border-bottom-color: {PRIMARY}; }}

  /* Sticky header — Austin CycleSafe wordmark + version badge */
  .acs-sticky-header {{
    position: sticky;
    top: 0;
    background: {BG};
    z-index: 100;
    padding: 8px 0 10px 0;
    margin-bottom: 0 !important;
  }}

  /* Buttons — unified TEAL background across Streamlit + download buttons */
  .stButton > button,
  .stDownloadButton > button {{
    background: {PRIMARY} !important; color: white !important;
    border: 1px solid {PRIMARY} !important;
    font-weight: 600 !important; border-radius: 6px !important;
    height: 40px !important;
    transition: background 140ms ease !important;
  }}
  .stButton > button:hover:not(:disabled),
  .stDownloadButton > button:hover:not(:disabled) {{
    background: {PRIMARY_HOVER} !important;
    border-color: {PRIMARY_HOVER} !important;
  }}
  .stButton > button:disabled,
  .stDownloadButton > button:disabled {{
    background: #E5E7EB !important; color: #9CA3AF !important;
    border-color: #E5E7EB !important; cursor: not-allowed !important;
  }}

  /* Hide Leaflet.Draw's edit + remove toolbar (clear-route lives in sidebar) */
  .leaflet-draw-edit-edit, .leaflet-draw-edit-remove {{ display: none !important; }}
  .leaflet-draw-section:nth-child(2) {{ display: none !important; }}
  /* Hide the "Leaflet | © OpenStreetMap contributors" badge on the Plan map */
  .leaflet-control-attribution {{ display: none !important; }}

  /* ===== Modern polish: shadows, hover, chart cards ===== */
  /* KPI cards lift slightly on hover */
  .acs-kpi {{
    box-shadow: 0 1px 2px rgba(17,24,39,0.04), 0 1px 4px rgba(17,24,39,0.03);
    transition: transform 160ms ease, box-shadow 160ms ease;
  }}
  .acs-kpi:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(17,24,39,0.08), 0 4px 12px rgba(17,24,39,0.04);
  }}
  .acs-insight {{
    box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    transition: transform 160ms ease, box-shadow 160ms ease;
  }}
  .acs-insight:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(17,24,39,0.08), 0 4px 12px rgba(17,24,39,0.04);
  }}
  /* Wrap every Plotly chart in a clean white card. Card height EXACTLY equals
     CHART_CARD_HEIGHT (which is CHART_INNER_HEIGHT + 10px buffer for padding
     and border). Plotly renders at exactly CHART_INNER_HEIGHT — no autosize
     drift, no whitespace inside, no clipping. Same value on every screen size. */
  div[data-testid="stPlotlyChart"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px 6px 2px 6px;
    box-shadow: 0 1px 2px rgba(17,24,39,0.04), 0 1px 4px rgba(17,24,39,0.03);
    margin: 0 0 4px 0 !important;
    height: {CHART_CARD_HEIGHT}px;
    box-sizing: border-box;
    overflow: hidden;
  }}
  /* Reset the inner wrapper so styles don't double-apply */
  div[data-testid="stPlotlyChart"] > div {{
    background: transparent !important; border: 0 !important;
    padding: 0 !important; box-shadow: none !important;
  }}
  /* Acrylic shadow on the dark hero card */
  .acs-hero {{
    box-shadow: 0 4px 12px rgba(17,24,39,0.10), 0 1px 3px rgba(17,24,39,0.06);
  }}

  /* Bottom CTA row on Home — "NEXT: ... Check my ride →" */
  .acs-next-row {{
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 16px; padding: 12px 16px;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    box-shadow: 0 1px 2px rgba(17,24,39,0.04);
  }}
  .acs-next-row .next-text {{ font-size: 0.92rem; color: {FG_MUTED}; }}
  .acs-next-row .next-text b {{
    color: {PRIMARY}; text-transform: uppercase; letter-spacing: 0.1em;
    font-size: 0.72rem; margin-right: 6px;
  }}

  /* Section title — replaces the expander header on the Plan sidebar.
     Always-visible heading rendered just above its bordered container. */
  .acs-section-title {{
    font-weight: 600; color: {FG}; font-size: 0.78rem;
    letter-spacing: 0.04em;
    margin: 0.7rem 0 0.3rem 0;
    padding: 0.45rem 0.75rem;
    background: {FG}; color: white;
    border-radius: 6px;
  }}
  /* Bordered container that follows the section title — tightened paddings */
  .acs-section-title + div[data-testid="stVerticalBlockBorderWrapper"],
  .acs-section-title + div[data-testid="stContainer"] {{
    margin-top: -0.35rem !important;
    border-top-left-radius: 0 !important;
    border-top-right-radius: 0 !important;
  }}

  /* Compact Streamlit form fields inside the Plan sidebar */
  div[data-testid="stSelectbox"] > label,
  div[data-testid="stNumberInput"] > label {{
    font-size: 0.72rem !important;
    color: {FG_MUTED} !important;
    margin-bottom: 0.15rem !important;
    padding: 0 !important;
  }}
  div[data-testid="stSelectbox"] > div > div,
  div[data-testid="stNumberInput"] > div > div {{
    min-height: 34px !important;
  }}

  /* Legacy expander styles kept for any other tabs that still use them */
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
    margin-top: 0.5rem; padding-top: 0.45rem;
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 12px;
  }}
  .acs-divider-row .acs-plan-kicker {{ flex: 0 0 auto; }}
  .acs-distance-num {{
    font-size: 1rem; font-weight: 600; color: white;
    letter-spacing: -0.01em; margin: 0;
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


def _format_chart_title(name: str, subtitle: str) -> str:
    """Bold name + lighter subtitle on the same line. No em-dash separator —
    visual hierarchy comes from weight + color contrast alone.
    """
    return (f"<b>{name}</b>  "
            f"<span style='font-weight:400; color:{FG_MUTED};'>{subtitle}</span>")


def _chart_layout(title: str, height: int = CHART_INNER_HEIGHT,
                  show_legend: bool = False) -> dict:
    """Uniform chart layout. Fixed height matches the CSS card height EXACTLY,
    so there's never internal whitespace inside the card. Charts with a legend
    use the same total height; we just steal more of the bottom margin for it.
    """
    return dict(
        title=dict(text=title, font=dict(size=13, color=FG, family="Inter"),
                   x=0, xanchor="left", y=0.985, yanchor="top",
                   pad=dict(t=0, b=0)),
        margin=dict(l=64, r=20, t=30, b=38 if not show_legend else 56),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, autosize=False, showlegend=show_legend,
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=-0.18, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=10, color=FG_MUTED)),
        font=dict(family="Inter", size=10, color=FG_MUTED),
        hoverlabel=dict(bgcolor=FG, font=dict(color="white", family="Inter")),
    )


# NOTE: @st.cache_data was removed from these chart functions because the
# cache key doesn't include color constants — when SUCCESS/DANGER change,
# cached figures retain the old colors. The build cost is negligible (~5ms).
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
        fill="tozeroy", fillcolor=DANGER_SOFT,
        hovertemplate="%{x}<br>%{y:.1f}% serious-injury<extra></extra>",
    ))
    fig.add_shape(**_baseline_shape(D["baseline"] * 100))
    fig.update_layout(**_chart_layout(
        _format_chart_title("Risk by time of day", "evenings 2× midday")))
    fig.update_yaxes(ticksuffix="%", gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


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
    fig.update_layout(**_chart_layout(
        _format_chart_title("Risk by street speed", "45 mph triples it")))
    fig.update_xaxes(ticksuffix="%", gridcolor=BORDER, range=[0, 24])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


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
    fig.update_layout(**_chart_layout(
        _format_chart_title("Fatalities by year", "2017 the worst"),
        show_legend=True),
        barmode="overlay")
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_xaxes(showgrid=False, dtick=1)
    return fig


def chart_severity():
    D = crash_aggregates()
    total, killed = D["total"], D["killed"]
    incap = int(round(D["baseline"] * total)) - killed
    nonincap = total - killed - incap
    fig = go.Figure(go.Pie(
        labels=["Killed", "Incapacitating injury", "All other crashes"],
        values=[killed, incap, nonincap],
        # 3-color rule: red for fatal, teal for incapacitating, neutral gray
        # for everything else.
        marker=dict(colors=[DANGER, PRIMARY, BORDER],
                    line=dict(color="white", width=2)),
        hole=0.62, textinfo="none",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(
        _format_chart_title("Severity breakdown", "1-in-9 serious")))
    return fig


# ---------- HEADER + TABS ----------
st.markdown(
    f"""
    <div class="acs-sticky-header" style="border-bottom: 1px solid {BORDER};">
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
        st.markdown('<div class="acs-question">Is the city doing enough to protect people on bikes?</div>',
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

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

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

    # Bottom CTA row — mirrors HTML "NEXT: about to ride? ... Check my ride →"
    # The visible row lives in the main Streamlit DOM (st.markdown), and a tiny
    # invisible iframe wires the button up to clicking the Plan tab.
    st.markdown(
        f"""
        <style>
          .acs-cta-row {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 16px; margin-top: 8px;
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
            box-shadow: 0 1px 2px rgba(17,24,39,0.04), 0 4px 10px rgba(17,24,39,0.04);
          }}
          .acs-cta-row .next-text {{ font-size: 0.94rem; color: {FG_MUTED}; }}
          .acs-cta-row .next-text b {{
            color: {PRIMARY}; text-transform: uppercase; letter-spacing: 0.1em;
            font-size: 0.72rem; margin-right: 6px;
          }}
          button.acs-cta-primary {{
            background: {PRIMARY} !important; color: white !important;
            border: 1px solid {PRIMARY} !important;
            font-weight: 600 !important; height: 40px !important;
            padding: 0 18px !important;
            border-radius: 6px !important; cursor: pointer !important;
            font-family: Inter, sans-serif !important; font-size: 0.92rem !important;
            transition: background 140ms ease !important;
          }}
          button.acs-cta-primary:hover {{ background: {PRIMARY_HOVER} !important; }}
        </style>
        <div class="acs-cta-row">
          <div class="next-text">
            <b>Next:</b> about to ride? Let the app check it for you.
          </div>
          <button class="acs-cta-primary" id="acs-check-my-ride">Check my ride →</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tiny iframe — height 0, runs only the click-handler wiring against the
    # parent doc. Has to be in an iframe because <script> tags injected via
    # st.markdown are stripped by Streamlit's sanitizer.
    import streamlit.components.v1 as _components
    _components.html(
        """
        <script>
          (function() {
            const doc = window.parent.document;
            const wire = () => {
              const checkBtn = doc.getElementById('acs-check-my-ride');
              if (checkBtn && !checkBtn.dataset.wired) {
                checkBtn.dataset.wired = '1';
                checkBtn.onclick = () => {
                  const tabs = doc.querySelectorAll('[role="tab"]');
                  if (tabs.length >= 2) tabs[1].click();
                };
              }
            };
            wire();
            setInterval(wire, 800);
          })();
        </script>
        """,
        height=0,
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
                <div class="acs-plan-kicker">Distance</div>
                <div style="text-align: right;">
                  <div class="acs-distance-num">{miles:.2f} mi · {km:.2f} km</div>
                  <div style="font-size: 0.7rem; color: rgba(255,255,255,0.65);">
                    {len(_route_coords(st.session_state.drawn_route))} route points
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 01 · Conditions — always visible, no expander chrome ----
        st.markdown('<div class="acs-section-title">01 · Conditions</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
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
            cc, cd = st.columns(2)
            with cc:
                st.selectbox("Street type", [
                    "Calm street (≤25 mph)", "Neighborhood arterial (26–35 mph)",
                    "Major arterial (36–45 mph)", "High-speed road (46+ mph)",
                ], key="f_sb")
            with cd:
                st.selectbox("Location", [
                    "At or near an intersection", "Mid-block (no intersection)",
                    "Driveway / parking access",
                ], key="f_loc")
            ce, cf = st.columns(2)
            with ce:
                st.selectbox("Surface", ["Dry", "Wet", "Other / Unknown"], key="f_sf")
            with cf:
                st.selectbox("Helmet", ["Yes", "No"], key="f_helmet")

        # ---- 02 · Pace & calories — always visible, no expander chrome ----
        st.markdown('<div class="acs-section-title">02 · Pace & calories</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
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
                    f"<div class='acs-kpi' style='min-height:auto; padding:0.5rem 0.7rem;'>"
                    f"<div class='label'>Time</div>"
                    f"<div class='num' style='font-size:1.15rem;'>{mm}:{ss:02d}</div></div>",
                    unsafe_allow_html=True)
            with rb:
                st.markdown(
                    f"<div class='acs-kpi' style='min-height:auto; padding:0.5rem 0.7rem;'>"
                    f"<div class='label'>Calories</div>"
                    f"<div class='num' style='font-size:1.15rem;'>{kcal:,}</div></div>",
                    unsafe_allow_html=True)

    # =========================================================
    # Map column — fills the right 2/3 of the viewport
    # =========================================================
    with mapcol:
        # Compact one-line drawing hint above the map
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: {FG_MUTED};
                        margin-bottom: 4px; line-height: 1.4;">
              <b style="color:{FG};">Draw:</b> click the
              <b style="color:{PRIMARY};">polyline</b> icon then click points along
              your route — <i>double-click last point to finish</i>. For a closed
              loop, use the <b style="color:{PRIMARY};">polygon</b> icon and click
              your start point to close.
            </div>
            """,
            unsafe_allow_html=True,
        )

        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        from branca.element import MacroElement, Template

        center = [30.2672, -97.7431]
        m = folium.Map(location=center, zoom_start=14, control_scale=True,
                       tiles="OpenStreetMap")
        # Inject CSS INTO the folium iframe — parent-app CSS can't reach there.
        # Hides the Leaflet/OSM attribution badge on the rendered map.
        _hide_attr = MacroElement()
        _hide_attr._template = Template("""
            {% macro html(this, kwargs) %}
            <style>
              .leaflet-control-attribution { display: none !important; }
            </style>
            {% endmacro %}
        """)
        m.get_root().add_child(_hide_attr)
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
            # Hide the edit + delete tool buttons; "Clear current route" in the
            # sidebar handles route removal, and edit-after-the-fact isn't
            # needed (just redraw if it's wrong).
            edit_options={"edit": False, "remove": False},
        ).add_to(m)
        # Re-render the user's drawn route after Streamlit reruns. Polygons get
        # a translucent fill so they keep the same look as during drawing;
        # polylines stay as a plain teal line. Either way the saved route is
        # always visible the moment the user returns to or refreshes Plan.
        _persist_coords = _route_coords(st.session_state.drawn_route)
        _persist_type = ((st.session_state.drawn_route or {})
                         .get("geometry", {}).get("type"))
        if _persist_coords:
            latlngs = [[c[1], c[0]] for c in _persist_coords]
            if _persist_type == "Polygon":
                folium.Polygon(
                    latlngs,
                    color=PRIMARY, weight=5, opacity=0.95,
                    fill=True, fill_color=PRIMARY, fill_opacity=0.10,
                ).add_to(m)
            else:
                folium.PolyLine(
                    latlngs, color=PRIMARY, weight=5, opacity=0.95,
                ).add_to(m)
            # Re-center the map on the route bounds so the user always sees
            # what they drew, even after a Streamlit rerun.
            lats = [c[1] for c in _persist_coords]
            lngs = [c[0] for c in _persist_coords]
            if len(_persist_coords) >= 2:
                lat_pad = max((max(lats) - min(lats)) * 0.20, 0.0015)
                lng_pad = max((max(lngs) - min(lngs)) * 0.20, 0.0015)
                m.fit_bounds([
                    [min(lats) - lat_pad, min(lngs) - lng_pad],
                    [max(lats) + lat_pad, max(lngs) + lng_pad],
                ])
        # Height is overridden by CSS clamp(PLAN_MAP_MIN, PLAN_MAP_IDEAL, PLAN_MAP_MAX);
        # passing min as a sane fallback before the CSS rule kicks in.
        out = st_folium(m, height=PLAN_MAP_MIN_PX, width=None,
                        returned_objects=["last_active_drawing"], key="plan_map",
                        use_container_width=True)
        if out and out.get("last_active_drawing"):
            new_route = out["last_active_drawing"]
            # Only update + rerun when the route actually changed, to avoid loops
            if new_route != st.session_state.drawn_route:
                st.session_state.drawn_route = new_route
                st.rerun()

    # =========================================================
    # FULL-WIDTH action bar — sits flush below both sidebar AND map column.
    # The :has() rule above already tightens the gap; no extra spacer needed.
    # =========================================================
    ab1, ab2, ab3, ab4 = st.columns(4)
    with ab1:
        if st.button("💾 Save ride", use_container_width=True,
                     disabled=meters < 1, key="btn_save_ride"):
            if st.session_state.drawn_route:
                st.session_state.saved_rides.append({
                    "name": f"Ride {len(st.session_state.saved_rides)+1}",
                    "geojson": st.session_state.drawn_route,
                    "distance_m": meters,
                    "risk": st.session_state.current_risk,
                })
                # Toast-style note that auto-disappears, instead of persistent banner
                st.toast(f"Saved ride #{len(st.session_state.saved_rides)}", icon="💾")
    with ab2:
        if st.button("⨯ Clear", use_container_width=True, key="btn_clear_route"):
            st.session_state.drawn_route = None
            st.rerun()
    with ab3:
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
    with ab4:
        saved_n = len(st.session_state.saved_rides)
        view_label = (f"📊 Results ({saved_n}) →"
                      if saved_n else "📊 Results →")
        import streamlit.components.v1 as _comps_view
        _comps_view.html(
            f"""
            <html><body style="margin:0;padding:0;">
            <button id="view-results-jump-plan"
                    style="width:100%; height: 40px; padding: 0 12px;
                           background: {PRIMARY}; color: white;
                           border: 1px solid {PRIMARY}; border-radius: 6px;
                           font-weight: 600; font-family: Inter, sans-serif;
                           font-size: 0.92rem; cursor: pointer;
                           white-space: nowrap;
                           transition: background 140ms ease;"
                    onmouseover="this.style.background='{PRIMARY_HOVER}';"
                    onmouseout="this.style.background='{PRIMARY}';">
              {view_label}
            </button>
            <script>
              document.getElementById('view-results-jump-plan').onclick = () => {{
                const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                if (tabs.length >= 3) tabs[2].click();
              }};
            </script>
            </body></html>
            """,
            height=44,
        )


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
    band_label = risk.get("band_label", "-")
    inputs = risk.get("inputs", {})
    num_class = {"calm": "calm", "caution": "warn", "danger": "bad"}.get(band, "")

    st.markdown(
        f"""
        <div style="background:{PRIMARY}; color:white; padding: 10px 16px; border-radius: 8px 8px 0 0;
                    display:flex; justify-content:space-between; align-items:baseline; margin-top: 14px;">
          <span style="font-weight: 600; letter-spacing: 0.04em;">{title}</span>
          <span style="font-size: 0.72rem; letter-spacing: 0.10em; text-transform: uppercase;
                       opacity: 0.85;">
            {inputs.get('tb', '-')} · {inputs.get('sf', '-')}
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
            map_height = RIDE_CARD_HEIGHT  # single source of truth — see design tokens
            # Compute padded bounds + a max-zoom cap in Python so the JS doesn't
            # get to "decide" — we tell it exactly what view to set.
            lats = [p[0] for p in ll]; lngs = [p[1] for p in ll]
            sw_lat, ne_lat = min(lats), max(lats)
            sw_lng, ne_lng = min(lngs), max(lngs)
            # Pad the box by a generous margin so the polyline doesn't skim the edge.
            lat_span = max(ne_lat - sw_lat, 1e-5)
            lng_span = max(ne_lng - sw_lng, 1e-5)
            lat_pad = max(lat_span * 0.20, 0.0015)
            lng_pad = max(lng_span * 0.20, 0.0015)
            bounds = [
                [sw_lat - lat_pad, sw_lng - lng_pad],
                [ne_lat + lat_pad, ne_lng + lng_pad],
            ]
            bounds_json = json.dumps(bounds)
            html = f"""
            <link rel="stylesheet"
                  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
              html, body {{ margin: 0; padding: 0; }}
              #{map_id} {{ height: {map_height}px; width: 100%;
                         border-radius: 0 0 0 8px; }}
              /* Hide the Leaflet / OSM attribution badge */
              .leaflet-control-attribution {{ display: none !important; }}
            </style>
            <div id="{map_id}"></div>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
              (function() {{
                var coords = {ll_json};
                var bounds = {bounds_json};
                var map = L.map("{map_id}", {{
                  scrollWheelZoom: false, zoomControl: true,
                  maxZoom: 17,
                  attributionControl: false
                }});
                L.tileLayer(
                  "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
                  {{ attribution: "", maxZoom: 19 }}
                ).addTo(map);
                L.polyline(coords, {{
                  color: "{PRIMARY}", weight: 5, opacity: 0.95
                }}).addTo(map);
                L.circleMarker({json.dumps(start)}, {{
                  radius: 7, color: "white", weight: 2,
                  fillColor: "{SUCCESS}", fillOpacity: 1.0
                }}).bindTooltip("Start").addTo(map);
                L.circleMarker({json.dumps(end)}, {{
                  radius: 7, color: "white", weight: 2,
                  fillColor: "{DANGER}", fillOpacity: 1.0
                }}).bindTooltip("End").addTo(map);

                // Set the view immediately with our explicit padded bounds, then
                // fix it again after the iframe sizes itself. maxZoom caps it
                // so a tiny route doesn't zoom to street-level.
                function applyView() {{
                  map.invalidateSize();
                  map.fitBounds(bounds, {{
                    padding: [10, 10], maxZoom: 15, animate: false
                  }});
                }}
                applyView();
                setTimeout(applyView, 100);
                setTimeout(applyView, 350);
              }})();
            </script>
            """
            components.html(html, height=map_height + RIDE_CARD_IFRAME_BUFFER)
        else:
            st.info("No route drawn for this ride yet.")
    with sum_col:
        # Build a unique modal ID per ride so multiple cards on Results don't collide
        ride_json = json.dumps({
            "title": title,
            "distance_m": meters,
            "distance_mi": round(miles, 3),
            "risk": risk,
            "geojson": geojson,
        }, indent=2, default=str)
        ride_json_html = (ride_json
                          .replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;"))
        modal_id = f"jm-{abs(hash(title + ride_json)) % 1000000}"

        # Single cohesive dark panel — fills the full map height (RIDE_CARD_HEIGHT)
        # so the right side aligns flush with the map on the left. The { } JSON
        # button sits absolutely positioned in the bottom-right corner.
        st.markdown(
            f"""
            <div class="acs-hero" style="padding: 1.1rem 1.3rem;
                                          height: {RIDE_CARD_HEIGHT}px;
                                          border-radius: 0 0 8px 0;
                                          margin: 0; box-shadow: none;
                                          display: flex; flex-direction: column;
                                          justify-content: space-between; gap: 0.8rem;
                                          position: relative;">
              <div>
                <span style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.10em;
                             text-transform: uppercase; color: rgba(255,255,255,0.85);">
                  Should you ride?
                </span>
                <div class="num {num_class}" style="font-size: 2.4rem; margin-top: 4px;">
                  {risk.get('rate',0)*100:.1f}%
                </div>
                <div class="band" style="margin-top: 2px;">{band_label.upper()}</div>
                <div class="sub" style="margin-top: 0.5rem;">
                  Of <b>{risk.get('n', 0):,}</b> matching crashes,
                  <b>{risk.get('serious', 0)}</b> were serious ({risk.get('killed', 0)} fatal).
                </div>
              </div>
              <div style="border-top: 1px solid rgba(255,255,255,0.18);
                          padding-top: 0.7rem; font-size: 0.82rem; line-height: 1.55;
                          color: rgba(255,255,255,0.9);">
                <div><b>Distance:</b> {miles:.2f} mi · {meters/1000:.2f} km</div>
                <div><b>Street:</b> {inputs.get('sb', '-')}</div>
                <div><b>Location:</b> {inputs.get('loc', '-')}</div>
                <div><b>Helmet:</b> {inputs.get('helmet', '-')}</div>
              </div>
              <button title="Show JSON"
                      onclick="document.getElementById('{modal_id}').style.display='flex';"
                      style="position: absolute; bottom: 10px; right: 12px;
                             width: 32px; height: 28px;
                             background: rgba(255,255,255,0.10); color: white;
                             border: 1px solid rgba(255,255,255,0.25);
                             border-radius: 5px; cursor: pointer;
                             font-family: 'Courier New', monospace; font-size: 0.95rem;
                             line-height: 1; padding: 0;
                             transition: background 140ms ease;"
                      onmouseover="this.style.background='rgba(255,255,255,0.20)';"
                      onmouseout="this.style.background='rgba(255,255,255,0.10)';">
                {{ }}
              </button>
            </div>
            <div id="{modal_id}"
                 style="display:none; position: fixed; inset: 0;
                        background: rgba(17,24,39,0.55); z-index: 9999;
                        align-items: center; justify-content: center;"
                 onclick="if(event.target===this) this.style.display='none';">
              <div style="background: white; border-radius: 10px; padding: 18px 20px;
                          max-width: 720px; max-height: 80vh; width: 90%;
                          overflow: auto; box-shadow: 0 20px 50px rgba(0,0,0,0.25);
                          font-family: Inter, sans-serif;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom: 10px;">
                  <span style="font-weight:600; color:{FG}; font-size: 0.92rem;">
                    {title} · JSON
                  </span>
                  <button onclick="document.getElementById('{modal_id}').style.display='none';"
                          style="background: transparent; border: 0; cursor: pointer;
                                 font-size: 1.5rem; color: {FG_MUTED}; line-height: 1;">×</button>
                </div>
                <pre style="font-family:'Courier New', monospace; font-size: 0.78rem;
                            line-height: 1.5; color: {FG}; white-space: pre-wrap;
                            word-break: break-word; margin: 0;">{ride_json_html}</pre>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # JSON drawer is now rendered inside map_col directly under the map,
    # labeled "{ } JSON" — see the components.html block above.


# ============================================================
# TAB 3 — RESULTS
# ============================================================
with tab_results:
    # Header row — kicker on the left, "Back to Plan" jump-button on the right.
    # The button uses parent-doc DOM access to click the Plan tab (index 1).
    import streamlit.components.v1 as _components
    _hdr_l, _hdr_r = st.columns([3, 1])
    with _hdr_l:
        st.markdown('<p class="acs-kicker">Your rides at a glance</p>',
                    unsafe_allow_html=True)
    with _hdr_r:
        _components.html(
            f"""
            <button id="back-to-plan-jump"
                    style="width:100%; padding: 7px 12px; margin-top: 2px;
                           background: white; color: {FG};
                           border: 1px solid {BORDER}; border-radius: 6px;
                           font-weight: 600; font-family: Inter, sans-serif;
                           font-size: 0.85rem; cursor: pointer;">
              ← Back to Plan
            </button>
            <script>
              document.getElementById('back-to-plan-jump').onclick = () => {{
                const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                if (tabs.length >= 2) tabs[1].click();
              }};
            </script>
            """,
            height=44,
        )

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
                                                     "band_label": "-", "n": 0,
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
