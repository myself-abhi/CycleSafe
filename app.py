"""
Austin CycleSafe — a go/no-go decision tool for Austin cyclists.

ALY6040 · Module 5 · Building Products
Single-file Streamlit app. Run with:  streamlit run app.py

The user is an Austin cyclist about to leave the house. The decision the
app supports is binary: ride now, or wait / change route / skip. The
indicator is the rate at which historical crashes turned serious (Killed
or Incapacitating Injury — "KSI"), compared against the citywide baseline.

Three colors only:  Charcoal #1E1E1E  ·  Safety Teal #2ECC71  ·  Pure White
Four panels, no vertical scroll on a 14" laptop, single column on mobile.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIG  — must be the first Streamlit call
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Austin CycleSafe",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
CHARCOAL = "#1E1E1E"
TEAL = "#2ECC71"
WHITE = "#FFFFFF"
LINE = "#ECECEC"
MUTED = "#6B6B6B"

KSI_SEVERITIES = {"Killed", "Incapacitating Injury"}
MIN_SAMPLE_SIZE = 30
DATA_PATH = Path(__file__).parent / "bike_crash.csv"


# ---------------------------------------------------------------------
# CSS  —  three-color system, responsive 4-panel grid, footer kill
# Panels are styled by targeting Streamlit's bordered-container DOM,
# not by manual <div> wrappers (those don't nest the way you'd expect
# in Streamlit — children land outside the parent).
# ---------------------------------------------------------------------
def inject_css(mode: str) -> None:
    is_danger = (mode == "danger")
    page_bg = CHARCOAL if is_danger else "#FAFAFA"
    page_fg = WHITE if is_danger else CHARCOAL
    panel_bg = "#262626" if is_danger else WHITE
    panel_border = "#333" if is_danger else LINE
    muted_color = "#9B9B9B" if is_danger else MUTED
    accent = WHITE if is_danger else TEAL
    inner_bg = "#1E1E1E" if is_danger else "#FAFAFA"

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      /* ============== STREAMLIT CHROME — KILL IT ============== */
      #MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
      [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
      [data-testid="stStatusWidget"] {{ display: none !important; }}
      .stDeployButton {{ display: none !important; }}

      /* ============== PAGE SURFACE ============== */
      html, body, .stApp {{
        background: {page_bg} !important;
        color: {page_fg};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        transition: background 220ms ease, color 220ms ease;
      }}

      /* Fluid container — padding scales with viewport */
      .block-container {{
        padding: clamp(0.5rem, 1.2vw, 1rem) clamp(0.75rem, 2vw, 1.75rem) 1rem !important;
        max-width: 1600px !important;
        margin: 0 auto !important;
      }}

      /* ============== BRAND BAR ============== */
      .brand-row {{
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.2rem 0 0.65rem 0;
        border-bottom: 1px solid {panel_border};
        margin-bottom: 0.7rem;
        flex-wrap: wrap;
      }}
      .brand-icon {{
        width: clamp(28px, 3vw, 34px);
        height: clamp(28px, 3vw, 34px);
        border-radius: 8px;
        background: {accent};
        color: {CHARCOAL if is_danger else WHITE};   /* SVG stroke inherits via currentColor */
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
      }}
      .brand-icon svg {{ width: 60%; height: 60%; stroke: currentColor !important; }}
      .brand-name {{
        color: {page_fg};
        font-weight: 800;
        font-size: clamp(0.95rem, 1.2vw, 1.1rem);
        letter-spacing: -0.02em;
      }}
      .brand-tag {{
        color: {muted_color};
        font-size: clamp(0.7rem, 0.85vw, 0.8rem);
        margin-left: auto;
        letter-spacing: 0.02em;
      }}
      @media (max-width: 920px) {{
        .brand-tag {{ display: none; }}
      }}

      /* ============== FILTER STRIP — wraps at narrow widths ============== */
      /* Streamlit's column row */
      [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.6rem !important;
        row-gap: 0.6rem !important;
      }}
      /* Each filter column gets a sensible min-width so 5 cols don't squish */
      [data-testid="stHorizontalBlock"] > [data-testid="column"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        min-width: 0;
        flex: 1 1 180px !important;
      }}
      .stSelectbox label {{
        font-size: 0.68rem !important;
        color: {muted_color} !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
      }}
      .stSelectbox > div > div {{
        background: {panel_bg} !important;
        border: 1px solid {panel_border} !important;
        border-radius: 8px !important;
        color: {page_fg} !important;
      }}
      .stSelectbox div[data-baseweb="select"] > div {{
        color: {page_fg} !important;
      }}

      /* ============== PANELS ============== */
      /* Streamlit's bordered container is the panel surface.
         Children inside this wrapper actually nest properly. */
      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {panel_bg} !important;
        border: 1px solid {panel_border} !important;
        border-radius: 14px !important;
        padding: clamp(0.85rem, 1.4vw, 1.2rem) clamp(0.95rem, 1.6vw, 1.35rem) !important;
        height: 100%;
        transition: background 220ms ease, border-color 220ms ease;
      }}

      /* Panel row spacing */
      [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {{
        margin-bottom: 0.8rem;
      }}

      /* Panel section header */
      .sec-h {{
        font-size: clamp(0.68rem, 0.85vw, 0.76rem);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: {muted_color};
        margin: 0 0 0.85rem 0;
        font-weight: 700;
      }}

      /* Verdict accent bar — :has() identifies the verdict panel */
      [data-testid="stVerticalBlockBorderWrapper"]:has(.verdict-headline) {{
        position: relative;
        overflow: hidden;
      }}
      [data-testid="stVerticalBlockBorderWrapper"]:has(.verdict-headline)::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: {accent};
      }}

      /* ============== VERDICT PANEL CONTENT ============== */
      .verdict-mode {{
        display: inline-block;
        font-size: clamp(0.65rem, 0.8vw, 0.72rem);
        letter-spacing: 0.2em;
        font-weight: 700;
        text-transform: uppercase;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.85rem;
      }}
      .mode-safe .verdict-mode {{ background: {TEAL}; color: {WHITE}; }}
      .mode-danger .verdict-mode {{ background: {WHITE}; color: {CHARCOAL}; }}
      .mode-unknown .verdict-mode {{ background: {CHARCOAL}; color: {WHITE}; }}

      .verdict-headline {{
        font-size: clamp(1.45rem, 2.6vw, 2.3rem);
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
        font-weight: 800;
        color: {page_fg};
      }}
      .verdict-sub {{
        font-size: clamp(0.85rem, 1vw, 0.95rem);
        color: {muted_color};
        margin: 0 0 1rem 0;
        line-height: 1.5;
      }}

      /* Verdict numbers — fluid grid, drops columns gracefully */
      .verdict-numbers {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid {panel_border};
      }}
      .verdict-numbers .cell {{
        display: flex; flex-direction: column; gap: 0.15rem;
        min-width: 0;
      }}
      .verdict-numbers .num {{
        font-size: clamp(1.1rem, 1.7vw, 1.5rem);
        font-weight: 800;
        color: {page_fg};
        letter-spacing: -0.02em;
        line-height: 1.1;
      }}
      .verdict-numbers .lbl {{
        font-size: clamp(0.62rem, 0.75vw, 0.7rem);
        color: {muted_color};
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }}

      /* ============== METRIC CARDS ============== */
      .metric-block {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 0.65rem;
        margin-bottom: 0.85rem;
      }}
      .metric-card {{
        background: {inner_bg};
        border-radius: 10px;
        padding: clamp(0.5rem, 0.9vw, 0.75rem) clamp(0.6rem, 1vw, 0.9rem);
        border: 1px solid {panel_border};
        min-width: 0;
      }}
      .metric-card .num {{
        display: block;
        font-size: clamp(1.1rem, 1.7vw, 1.5rem);
        font-weight: 800;
        color: {page_fg};
        letter-spacing: -0.02em;
        line-height: 1.1;
      }}
      .metric-card .lbl {{
        display: block;
        font-size: clamp(0.6rem, 0.72vw, 0.68rem);
        color: {muted_color};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
      }}

      /* ============== FACTOR LIST ============== */
      .factor {{
        font-size: clamp(0.85rem, 1vw, 0.95rem);
        color: {page_fg};
        padding: 0.55rem 0;
        border-bottom: 1px solid {panel_border};
        line-height: 1.45;
      }}
      .factor:last-of-type {{ border-bottom: none; }}
      .factor b {{ color: {accent}; font-weight: 700; }}
      .mode-danger .factor b {{
        color: {CHARCOAL};
        background: {WHITE};
        padding: 0 0.35rem;
        border-radius: 4px;
      }}

      /* ============== EMPTY STATE ============== */
      .empty-state {{
        color: {muted_color};
        font-size: 0.95rem;
        line-height: 1.55;
        padding: 1.2rem 0.5rem;
        text-align: center;
      }}
      .empty-state .muted {{
        color: {muted_color};
        font-size: 0.82rem;
        opacity: 0.85;
      }}

      /* ============== PROVENANCE FOOTER ============== */
      .prov {{
        margin-top: 0.65rem;
        padding-top: 0.55rem;
        border-top: 1px dashed {panel_border};
        font-size: clamp(0.62rem, 0.75vw, 0.7rem);
        color: {muted_color};
        letter-spacing: 0.04em;
      }}

      /* ============== ALTAIR / VEGA CHART STYLING ============== */
      .vega-embed {{ background: transparent !important; width: 100% !important; }}
      .vega-embed canvas, .vega-embed svg {{ max-width: 100% !important; height: auto !important; }}
      .mode-danger .vega-embed text {{ fill: {WHITE} !important; }}
      .stAltairChart, [data-testid="stAltairChart"] {{
        background: transparent !important;
        width: 100% !important;
      }}

      /* ============== TABLET BREAKPOINT (≤ 900px) ============== */
      /* Force the 2-col panel rows to stack here too */
      @media (max-width: 900px) {{
        [data-testid="stHorizontalBlock"] {{ flex-direction: column !important; }}
        [data-testid="stHorizontalBlock"] > [data-testid="column"],
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
          width: 100% !important;
          flex: 1 1 100% !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
          margin-bottom: 0.7rem !important;
        }}
      }}

      /* ============== MOBILE BREAKPOINT (≤ 600px) ============== */
      @media (max-width: 600px) {{
        .verdict-numbers {{
          grid-template-columns: repeat(3, 1fr);
          gap: 0.5rem;
        }}
        .metric-block {{
          grid-template-columns: repeat(3, 1fr);
          gap: 0.4rem;
        }}
      }}

      /* ============== TINY VIEWPORT (≤ 380px) ============== */
      @media (max-width: 380px) {{
        .verdict-numbers,
        .metric-block {{
          grid-template-columns: 1fr 1fr;
        }}
      }}

      /* ============== REDUCED MOTION ============== */
      @media (prefers-reduced-motion: reduce) {{
        * {{ transition: none !important; }}
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["is_ksi"] = df["Crash Severity"].isin(KSI_SEVERITIES).astype(int)
    df["hour"] = (df["Crash Time"] // 100).clip(0, 23)
    bins = [-1, 5, 10, 15, 20, 24]
    labels = ["Late night (0-5)", "Morning (5-10)", "Midday (10-15)",
              "Evening (15-20)", "Night (20-24)"]
    df["time_of_day"] = pd.cut(df["hour"], bins=bins, labels=labels)
    df["weather_proxy"] = df["Surface Condition"].fillna("Unknown")
    df["speed_limit_clean"] = df["Speed Limit"].where(df["Speed Limit"] > 0)
    return df


def baseline_ksi_rate(df: pd.DataFrame) -> float:
    return df["is_ksi"].mean()


def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    if filters["time_of_day"] != "Any":
        out = out[out["time_of_day"] == filters["time_of_day"]]
    if filters["day_of_week"] != "Any":
        out = out[out["Day of Week"] == filters["day_of_week"]]
    if filters["roadway_part"] != "Any":
        out = out[out["Roadway Part"] == filters["roadway_part"]]
    if filters["surface"] != "Any":
        out = out[out["weather_proxy"] == filters["surface"]]
    if filters["intersection"] != "Any":
        out = out[out["Intersection Related"] == filters["intersection"]]
    return out


def decide(filtered: pd.DataFrame, baseline: float) -> dict:
    n = len(filtered)
    if n < MIN_SAMPLE_SIZE:
        return {
            "mode": "UNKNOWN",
            "headline": "Insufficient data",
            "subline": (
                f"Only {n} crashes match these conditions — fewer than the "
                f"{MIN_SAMPLE_SIZE} we need for a confident verdict. "
                "Proceed with extreme caution and consider loosening a filter."
            ),
            "rate": None, "n": n, "delta_pct": None,
        }

    rate = filtered["is_ksi"].mean()
    delta = (rate - baseline) / baseline * 100 if baseline > 0 else 0

    # Small effects (< 2pt relative) get treated as "in line with baseline"
    # rather than fake-precise SAFE/DANGER calls.
    if abs(delta) < 2:
        return {
            "mode": "SAFE",
            "headline": "In line with the Austin baseline",
            "subline": (
                "Serious-injury rate for these conditions tracks the citywide "
                f"average closely ({rate*100:.1f}% vs {baseline*100:.1f}%). "
                "No condition-specific risk signal — ride with normal caution."
            ),
            "rate": rate, "n": n, "delta_pct": delta,
        }

    if delta > 0:
        return {
            "mode": "DANGER",
            "headline": "Higher than typical risk",
            "subline": (
                f"Serious-injury rate runs {delta:.0f}% above the Austin "
                "baseline for these conditions. Consider waiting, changing "
                "your route, or skipping this ride."
            ),
            "rate": rate, "n": n, "delta_pct": delta,
        }

    return {
        "mode": "SAFE",
        "headline": "Lower than typical risk",
        "subline": (
            f"Serious-injury rate runs {abs(delta):.0f}% below the Austin "
            "baseline for these conditions. Conditions look favorable — "
            "stay alert anyway."
        ),
        "rate": rate, "n": n, "delta_pct": delta,
    }


# ---------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------
def hour_day_heatmap(df: pd.DataFrame, mode: str) -> alt.Chart:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    pivot = (
        df.groupby(["Day of Week", "hour"])
        .agg(crashes=("is_ksi", "size"), ksi=("is_ksi", "sum"))
        .reset_index()
    )
    pivot["ksi_rate"] = pivot["ksi"] / pivot["crashes"]
    pivot.loc[pivot["crashes"] < 5, "ksi_rate"] = np.nan

    is_danger = (mode == "danger")
    low_color = "#3A3A3A" if is_danger else "#F4F4F4"
    high_color = WHITE if is_danger else TEAL
    axis_color = "#9B9B9B" if is_danger else MUTED
    stroke_color = "#262626" if is_danger else WHITE

    chart = (
        alt.Chart(pivot)
        .mark_rect(stroke=stroke_color, strokeWidth=1)
        .encode(
            x=alt.X("hour:O", title="Hour of day",
                    axis=alt.Axis(labelFontSize=10, titleFontSize=11,
                                  labelColor=axis_color, titleColor=axis_color,
                                  domainColor=axis_color, tickColor=axis_color)),
            y=alt.Y("Day of Week:N", sort=day_order, title=None,
                    axis=alt.Axis(labelFontSize=10, labelColor=axis_color,
                                  domainColor=axis_color, tickColor=axis_color)),
            color=alt.Color(
                "ksi_rate:Q",
                scale=alt.Scale(range=[low_color, high_color]),
                legend=alt.Legend(title="Serious-injury rate", format=".0%",
                                  orient="bottom", titleFontSize=10,
                                  labelFontSize=9, titleColor=axis_color,
                                  labelColor=axis_color),
            ),
            tooltip=[
                alt.Tooltip("Day of Week:N", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("crashes:Q", title="Crashes"),
                alt.Tooltip("ksi_rate:Q", format=".1%", title="KSI rate"),
            ],
        )
        .properties(height=240, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    return chart


def severity_breakdown_chart(filtered: pd.DataFrame, mode: str) -> alt.Chart:
    sev = (
        filtered["Crash Severity"]
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="count")
    )
    is_danger = (mode == "danger")
    accent = WHITE if is_danger else TEAL
    neutral = "#555" if is_danger else "#D8D8D8"
    axis_color = "#9B9B9B" if is_danger else MUTED

    return (
        alt.Chart(sev)
        .mark_bar(cornerRadius=3, height=18)
        .encode(
            y=alt.Y("severity:N", sort="-x", title=None,
                    axis=alt.Axis(labelFontSize=10, labelColor=axis_color,
                                  domainColor=axis_color, tickColor=axis_color)),
            x=alt.X("count:Q", title=None,
                    axis=alt.Axis(labelFontSize=9, grid=False,
                                  labelColor=axis_color, domainColor=axis_color,
                                  tickColor=axis_color)),
            color=alt.condition(
                alt.FieldOneOfPredicate(field="severity",
                                        oneOf=list(KSI_SEVERITIES)),
                alt.value(accent),
                alt.value(neutral),
            ),
            tooltip=[alt.Tooltip("severity:N"), alt.Tooltip("count:Q")],
        )
        .properties(height=160, background="transparent")
        .configure_view(stroke=None)
    )


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
BIKE_SVG = '''
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
     stroke-linecap="round" stroke-linejoin="round">
  <circle cx="5.5" cy="17.5" r="3.5"/>
  <circle cx="18.5" cy="17.5" r="3.5"/>
  <path d="M15 6h2l2 5-3 6-4-7-3-1h-2"/>
  <path d="M5.5 17.5l4-7"/>
</svg>
'''


def render_brand():
    """No mode arg — colors are driven entirely by CSS classes so the
    verdict-mode CSS re-injection takes effect without inline-style override."""
    st.markdown(
        f'<div class="brand-row">'
        f'<div class="brand-icon">{BIKE_SVG}</div>'
        f'<div class="brand-name">Austin CycleSafe</div>'
        f'<div class="brand-tag">Go / No-Go decision · 2,463 crashes · 2010–2017 City of Austin</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_filters(df: pd.DataFrame) -> dict:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        time_of_day = st.selectbox(
            "Time of day",
            ["Any", "Morning (5-10)", "Midday (10-15)",
             "Evening (15-20)", "Night (20-24)", "Late night (0-5)"],
        )
    with c2:
        day_of_week = st.selectbox(
            "Day",
            ["Any", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"],
        )
    with c3:
        roadway_part = st.selectbox(
            "Roadway part",
            ["Any"] + sorted(df["Roadway Part"].dropna().unique().tolist()),
        )
    with c4:
        surface = st.selectbox(
            "Surface",
            ["Any"] + sorted(df["weather_proxy"].dropna().unique().tolist()),
        )
    with c5:
        intersection = st.selectbox(
            "Intersection",
            ["Any"] + sorted(df["Intersection Related"].dropna().unique().tolist()),
        )
    return dict(time_of_day=time_of_day, day_of_week=day_of_week,
                roadway_part=roadway_part, surface=surface, intersection=intersection)


def explain_drivers(filtered: pd.DataFrame, baseline: float, verdict: dict) -> list[str]:
    if verdict["mode"] == "UNKNOWN":
        return [
            f"Only <b>{verdict['n']}</b> historical crashes match this exact filter set.",
            "Treat the silence as a warning, not an all-clear — the data simply doesn't cover this case.",
            "Loosen one filter (e.g. drop the surface condition) to recover a usable sample.",
        ]
    out = []
    if len(filtered) > 0:
        h = (filtered.groupby("hour")["is_ksi"]
             .agg(["size", "mean"]).query("size >= 5")
             .sort_values("mean", ascending=False))
        if len(h) > 0:
            top_hour = int(h.index[0])
            out.append(
                f"Worst hour in your filter set is <b>{top_hour:02d}:00</b> "
                f"— KSI rate of <b>{h.iloc[0]['mean']:.0%}</b>."
            )
        rp = filtered["Roadway Part"].value_counts(normalize=True)
        if len(rp) > 0:
            out.append(
                f"<b>{rp.index[0]}</b> dominates this slice "
                f"({rp.iloc[0]:.0%} of matching crashes)."
            )
        helm = filtered["Person Helmet"].value_counts(normalize=True)
        if "Not Worn" in helm.index:
            out.append(
                f"Helmet not worn in <b>{helm['Not Worn']:.0%}</b> of matching "
                "crashes — wear yours."
            )
        if verdict["delta_pct"] is not None and abs(verdict["delta_pct"]) >= 20:
            direction = "above" if verdict["delta_pct"] > 0 else "below"
            out.append(
                f"Your conditions are <b>{abs(verdict['delta_pct']):.0f}% {direction}</b> "
                "the citywide serious-injury rate — a meaningful effect, not noise."
            )
    return out[:4] or ["Verdict tracks the citywide baseline. Add filters to refine."]


# ---------------------------------------------------------------------
# PANEL RENDERERS  — each renders inside an st.container(border=True)
# so the children actually nest in the bordered DOM node.
# ---------------------------------------------------------------------
def render_verdict_panel(verdict: dict, baseline: float, mode_class: str):
    with st.container(border=True):
        nums_html = ""
        if verdict["mode"] != "UNKNOWN":
            nums_html = (
                '<div class="verdict-numbers">'
                f'<div class="cell"><span class="num">{verdict["rate"]*100:.1f}%</span>'
                f'<span class="lbl">Your conditions</span></div>'
                f'<div class="cell"><span class="num">{baseline*100:.1f}%</span>'
                f'<span class="lbl">Austin baseline</span></div>'
                f'<div class="cell"><span class="num">{verdict["n"]:,}</span>'
                f'<span class="lbl">Matching crashes</span></div>'
                '</div>'
            )
        st.markdown(
            f'<div class="mode-{mode_class}">'
            f'<div class="verdict-mode">{verdict["mode"]}</div>'
            f'<h1 class="verdict-headline">{verdict["headline"]}</h1>'
            f'<p class="verdict-sub">{verdict["subline"]}</p>'
            f'{nums_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_heatmap_panel(df: pd.DataFrame, mode_class: str):
    with st.container(border=True):
        st.markdown('<div class="sec-h">When crashes turn serious</div>',
                    unsafe_allow_html=True)
        st.altair_chart(hour_day_heatmap(df, mode_class), use_container_width=True)
        st.markdown(
            '<div class="prov">Hour × day · KSI rate · cells with &lt; 5 crashes shown blank</div>',
            unsafe_allow_html=True,
        )


def render_selection_panel(filtered: pd.DataFrame, mode_class: str):
    with st.container(border=True):
        st.markdown('<div class="sec-h">What\'s in your selection</div>',
                    unsafe_allow_html=True)
        if len(filtered) == 0:
            st.markdown(
                '<div class="empty-state">No crashes match these filters.<br>'
                '<span class="muted">Loosen one to see the breakdown.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            speed_med = filtered["speed_limit_clean"].median()
            speed_str = f"{int(speed_med)}" if pd.notna(speed_med) else "—"
            st.markdown(
                f'<div class="metric-block">'
                f'<div class="metric-card"><span class="num">{len(filtered):,}</span>'
                f'<span class="lbl">Matching crashes</span></div>'
                f'<div class="metric-card"><span class="num">{speed_str}</span>'
                f'<span class="lbl">Median speed limit</span></div>'
                f'<div class="metric-card"><span class="num">{filtered["is_ksi"].sum()}</span>'
                f'<span class="lbl">Serious injuries</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.altair_chart(severity_breakdown_chart(filtered, mode_class),
                            use_container_width=True)
        st.markdown(
            '<div class="prov">K + I bars highlighted · all other severities muted</div>',
            unsafe_allow_html=True,
        )


def render_why_panel(filtered: pd.DataFrame, baseline: float, verdict: dict, mode_class: str):
    with st.container(border=True):
        st.markdown('<div class="sec-h">What\'s driving the call</div>',
                    unsafe_allow_html=True)
        factors_html = "".join(
            f'<div class="factor mode-{mode_class}">{line}</div>'
            for line in explain_drivers(filtered, baseline, verdict)
        )
        st.markdown(factors_html, unsafe_allow_html=True)
        st.markdown(
            f'<div class="prov">Sample-size floor: {MIN_SAMPLE_SIZE} crashes · '
            'KSI = Killed + Incapacitating Injury</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_data()
    baseline = baseline_ksi_rate(df)

    # Render UI shell first so filters paint before we know the verdict
    inject_css("safe")
    render_brand()
    filters = render_filters(df)

    filtered = filter_df(df, filters)
    verdict = decide(filtered, baseline)
    mode_class = verdict["mode"].lower()

    # Re-inject CSS now that we know the mode (drives the color flip)
    inject_css(mode_class)

    # ----- ROW 1: VERDICT + HEATMAP -----
    top_left, top_right = st.columns([1, 1.15], gap="medium")
    with top_left:
        render_verdict_panel(verdict, baseline, mode_class)
    with top_right:
        render_heatmap_panel(df, mode_class)

    # ----- ROW 2: SELECTION + WHY -----
    bot_left, bot_right = st.columns([1, 1.15], gap="medium")
    with bot_left:
        render_selection_panel(filtered, mode_class)
    with bot_right:
        render_why_panel(filtered, baseline, verdict, mode_class)


if __name__ == "__main__":
    main()
