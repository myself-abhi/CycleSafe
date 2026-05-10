"""
Austin CycleSafe — Go/No-Go decision tool for urban cyclists.
Built on Austin Open Data 2010-2017 cyclist crash records.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional viewport detection — graceful fallback if package not installed
try:
    from streamlit_javascript import st_javascript
    _HAS_JS = True
except ImportError:
    _HAS_JS = False

# ---------- CONFIG ----------
DATA_PATH = "bike_crash.csv"
MIN_SAMPLE = 30  # sample-size floor for trustworthy verdict

# Design tokens
INK, BODY, MUTED = "#0A0A0A", "#374151", "#6B7280"
SURFACE, CARD, BORDER = "#FFFFFF", "#F9FAFB", "#E5E7EB"
GREEN, AMBER, RED = "#16A34A", "#F59E0B", "#DC2626"
GREEN_SOFT, AMBER_SOFT, RED_SOFT = "#DCFCE7", "#FEF3C7", "#FEE2E2"
FONT = '-apple-system, "Segoe UI", Inter, Roboto, sans-serif'

st.set_page_config(
    page_title="Austin CycleSafe", page_icon="🚲",
    layout="wide", initial_sidebar_state="collapsed",
)


# ---------- VIEWPORT DETECTION ----------
def get_viewport() -> tuple[int, int]:
    """Read window dimensions on first render, cache in session state.
    Returns (height_px, width_px). Falls back to (900, 1440) if JS unavailable.
    """
    if "viewport_h" in st.session_state and "viewport_w" in st.session_state:
        return st.session_state.viewport_h, st.session_state.viewport_w
    if _HAS_JS:
        try:
            h = st_javascript("window.innerHeight", key="vp_h")
            w = st_javascript("window.innerWidth", key="vp_w")
            if h and w and h > 100 and w > 100:
                st.session_state.viewport_h = int(h)
                st.session_state.viewport_w = int(w)
                return int(h), int(w)
        except Exception:
            pass
    # Fallback: assume typical 1080p laptop with browser chrome
    return 880, 1440


def layout_budget(viewport_h: int) -> dict[str, int]:
    """Distribute the viewport across rows + their internal charts.
    Each ROW has its own height (decision < pattern > drivers, because pattern
    holds the dense 2×3 small-multiples grid), but every PANEL within a row
    is identical to its row-mates. All ratios — no hardcoded pixels.
    """
    FIXED = 180   # app bar + filter rail + footer + paddings
    GAPS  = 28    # ~12 px between each of 3 rows
    available = max(540, viewport_h - FIXED - GAPS)

    # Per-row ratios — pattern row gets more space for its small-multiples grid
    decision_card = max(200, int(available * 0.27))   # 3 cards same height
    pattern_card  = max(300, int(available * 0.40))   # 2 cards same height
    drivers_card  = max(220, int(available * 0.27))   # 4 cards same height

    # Plotly font sizes scale with viewport — small screens get smaller
    # axis labels, large screens get bigger. Stays within readable bounds.
    font_scale = max(0.85, min(1.4, viewport_h / 900))
    return {
        "decision_card": decision_card,
        "pattern_card":  pattern_card,
        "drivers_card":  drivers_card,
        "gauge":         max(110, int(decision_card * 0.55)),
        "hour_curve":    max(180, int(pattern_card  * 0.78)),
        "small_tile":    max(80,  int(pattern_card  * 0.34)),
        "drivers_chart": max(110, int(drivers_card  * 0.72)),
        "donut":         max(80,  int(drivers_card  * 0.50)),
        # Plotly font sizes — used inside _chart_layout / chart functions
        "font_chart_title": max(11, int(13 * font_scale)),
        "font_chart_body":  max(8,  int(10 * font_scale)),
        "font_chart_tick":  max(8,  int(10 * font_scale)),
    }

# ---------- DATA ----------
@st.cache_data(show_spinner=False)
def synthesize_data(n: int = 2463, seed: int = 42) -> pd.DataFrame:
    """Plausible Austin crash rows when CSV is missing."""
    rng = np.random.default_rng(seed)
    hour_choices = (
        [7, 8, 9] * int(n * 0.06) + [17, 18, 19] * int(n * 0.07) +
        list(range(10, 16)) * int(n * 0.05) + [20, 21, 22] * int(n * 0.03) +
        [0, 1, 2, 3, 4, 5, 6, 23] * int(n * 0.02)
    )
    hours = rng.choice(hour_choices, size=n)
    minutes = rng.integers(0, 60, n)
    crash_time = (hours * 100 + minutes).astype(int)

    severities = ["Killed", "Incapacitating Injury", "Non-Incapacitating Injury",
                  "Possible Injury", "Not Injured", "Unknown"]
    sev_p = [0.012, 0.098, 0.32, 0.30, 0.25, 0.02]

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_p = [0.16, 0.16, 0.16, 0.16, 0.18, 0.10, 0.08]

    traffic = rng.integers(500, 35000, n).astype(object)
    no_data_idx = rng.choice(n, size=int(n * 0.04), replace=False)
    traffic[no_data_idx] = "No Data"

    return pd.DataFrame({
        "$1000 Damage to Any One Person's Property": rng.choice(["Yes", "No"], n, p=[0.45, 0.55]),
        "Active School Zone Flag": rng.choice(["Yes", "No"], n, p=[0.04, 0.96]),
        "At Intersection Flag": rng.choice(["TRUE", "FALSE"], n, p=[0.55, 0.45]),
        "Average Daily Traffic Amount": traffic,
        "Construction Zone Flag": rng.choice(["Yes", "No"], n, p=[0.06, 0.94]),
        "Crash Severity": rng.choice(severities, n, p=sev_p),
        "Crash Time": crash_time,
        "Crash Total Injury Count": rng.integers(0, 4, n),
        "Crash Year": rng.integers(2010, 2018, n),
        "Day of Week": rng.choice(days, n, p=day_p),
        "Intersection Related": rng.choice(
            ["Intersection", "Non Intersection", "Driveway Access", "Intersection Related"],
            n, p=[0.45, 0.32, 0.10, 0.13]),
        "Roadway Part": rng.choice(
            ["Main/Proper Lane", "Service/Frontage Road", "Shoulder", "Sidewalk", "Other"],
            n, p=[0.78, 0.06, 0.05, 0.08, 0.03]),
        "Speed Limit": rng.choice([0, 25, 30, 35, 40, 45, 50, 55], n,
                                  p=[0.05, 0.18, 0.30, 0.22, 0.12, 0.08, 0.03, 0.02]),
        "Surface Condition": rng.choice(
            ["Dry", "Wet", "Standing Water", "Other"], n, p=[0.85, 0.11, 0.02, 0.02]),
        "Traffic Control Type": rng.choice(
            ["Marked Lanes", "Center Stripe/Divider", "Signal Light", "Stop Sign",
             "None", "Yield Sign"], n, p=[0.40, 0.18, 0.20, 0.10, 0.08, 0.04]),
        "Person Helmet": rng.choice(
            ["Not Worn", "Worn, Damaged", "Worn, Not Damaged", "Unknown"],
            n, p=[0.55, 0.05, 0.30, 0.10]),
    })


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
    p = Path(DATA_PATH)
    source = ""
    try:
        if p.exists():
            df = pd.read_csv(p)
            source = f"City of Austin Open Data ({p.name})"
        else:
            df, source = synthesize_data(), "synthetic fallback (CSV not found)"
    except Exception:
        df, source = synthesize_data(), "synthetic fallback (CSV unreadable)"

    df["Average Daily Traffic Amount"] = pd.to_numeric(
        df["Average Daily Traffic Amount"], errors="coerce")
    crash_t = pd.to_numeric(df["Crash Time"], errors="coerce").fillna(-1).astype(int)
    df["hour"] = (crash_t // 100).clip(lower=0, upper=23)
    df["is_ksi"] = df["Crash Severity"].isin(["Killed", "Incapacitating Injury"])

    def norm_helmet(v):
        if not isinstance(v, str): return "Unknown"
        if "Worn," in v: return "Worn"
        if v == "Not Worn": return "Not Worn"
        return "Unknown"
    df["helmet"] = df["Person Helmet"].apply(norm_helmet)

    df["Speed Limit"] = pd.to_numeric(df["Speed Limit"], errors="coerce")
    df.loc[df["Speed Limit"] <= 5, "Speed Limit"] = np.nan

    def time_band(h):
        if 5 <= h < 10: return "Morning 5–10"
        if 10 <= h < 15: return "Midday 10–15"
        if 15 <= h < 19: return "Evening 15–19"
        return "Night 19–5"
    df["time_band"] = df["hour"].apply(time_band)

    def isect(row):
        ir = row.get("Intersection Related", "")
        if ir in ("Intersection", "Intersection Related"): return "Intersection"
        if ir == "Non Intersection": return "Non Intersection"
        return "Other"
    df["intersection_bucket"] = df.apply(isect, axis=1)

    def speed_bucket(s):
        if pd.isna(s): return "Unknown"
        if s <= 25: return "≤25"
        if s <= 30: return "30"
        if s <= 35: return "35"
        if s <= 40: return "40"
        return "≥45"
    df["speed_bucket"] = df["Speed Limit"].apply(speed_bucket)

    return df, source


# ---------- VERDICT ----------
@dataclass
class Verdict:
    code: str
    headline: str
    sub: str
    color: str
    n: int
    ksi_rate: float
    baseline: float
    delta_pp: float


def decide(slice_df: pd.DataFrame, all_df: pd.DataFrame) -> Verdict:
    n = len(slice_df)
    baseline = float(all_df["is_ksi"].mean())
    if n < MIN_SAMPLE:
        return Verdict(
            "INSUFFICIENT", "Not enough data, ride with normal caution",
            f"Fewer than {MIN_SAMPLE} matching crashes for these conditions ({n} found). "
            f"The dataset can't tell us if these conditions are safer or riskier than typical.",
            MUTED, n, 0.0, baseline, 0.0,
        )
    ksi = float(slice_df["is_ksi"].mean())
    delta = (ksi - baseline) * 100
    if delta <= 1:
        head = "Conditions look unusually safe" if delta < -1 else "In line with the Austin baseline"
        return Verdict("GO", head,
            f"Serious-injury rate for these conditions is {ksi*100:.1f}% vs the "
            f"{baseline*100:.1f}% citywide baseline ({delta:+.1f} pp). Helmet on, lights on, ride.",
            GREEN, n, ksi, baseline, delta)
    if delta <= 5:
        return Verdict("CAUTION", "Riskier than typical Austin conditions",
            f"Serious-injury rate for these conditions is {ksi*100:.1f}% vs the "
            f"{baseline*100:.1f}% citywide baseline ({delta:+.1f} pp). Ride with extra caution "
            f"or shift one variable.",
            AMBER, n, ksi, baseline, delta)
    return Verdict("NO-GO", "Well above the Austin baseline, reconsider",
        f"Serious-injury rate for these conditions is {ksi*100:.1f}% vs the "
        f"{baseline*100:.1f}% citywide baseline ({delta:+.1f} pp). Consider waiting, driving, "
        f"or changing route.",
        RED, n, ksi, baseline, delta)


def compute_drivers(slice_df: pd.DataFrame, all_df: pd.DataFrame) -> list[tuple[str, str]]:
    if len(slice_df) < MIN_SAMPLE:
        return [
            ("⚠️", "Sample is too small for confident drivers; defaulting to baseline precautions."),
            ("✅", "Helmet on, lights on, bright/visible clothing."),
            ("✅", "Stay off main travel lanes when possible. They account for most crashes citywide."),
        ]
    items: list[tuple[str, str]] = []
    base = float(all_df["is_ksi"].mean())

    by_hour = slice_df.groupby("hour").agg(n=("is_ksi", "size"), ksi=("is_ksi", "mean"))
    worst = by_hour[by_hour.n >= 10].sort_values("ksi", ascending=False)
    if len(worst) and worst.iloc[0].ksi >= 1.5 * base:
        h = int(worst.index[0])
        items.append(("⚠️", f"Avoid {h:02d}:00 to {(h+1) % 24:02d}:00. Worst hour in your filter at "
                            f"{worst.iloc[0].ksi * 100:.0f}% serious-injury rate."))

    rd = slice_df["Roadway Part"].value_counts(normalize=True)
    if len(rd) and rd.iloc[0] > 0.40:
        items.append(("⚠️", f"Stay off {rd.index[0]}. Accounts for {rd.iloc[0] * 100:.0f}% of "
                            f"crashes in your filter."))

    helmet_share = (slice_df["helmet"] == "Not Worn").mean()
    if helmet_share > 0.30:
        items.append(("✅", f"Helmet on. {helmet_share * 100:.0f}% of riders in these crashes "
                            f"weren't wearing one."))

    valid_tr = all_df["Average Daily Traffic Amount"].dropna()
    if len(valid_tr):
        q3 = valid_tr.quantile(0.75)
        slice_tr = slice_df["Average Daily Traffic Amount"].dropna()
        if len(slice_tr):
            high = (slice_tr >= q3).mean()
            if high > 0.25:
                items.append(("⚠️", f"High-traffic streets dominate this slice ({high * 100:.0f}%); "
                                    f"pick lower-traffic alternates."))

    base_c = (all_df["Construction Zone Flag"] == "Yes").mean()
    sl_c = (slice_df["Construction Zone Flag"] == "Yes").mean()
    if base_c > 0 and sl_c >= 2 * base_c and sl_c > 0.05:
        items.append(("⚠️", f"Construction zones overrepresented. {sl_c * 100:.0f}% of these crashes."))

    base_s = (all_df["Active School Zone Flag"] == "Yes").mean()
    sl_s = (slice_df["Active School Zone Flag"] == "Yes").mean()
    if base_s > 0 and sl_s >= 2 * base_s and sl_s > 0.03:
        items.append(("⚠️", f"School zones flagged in {sl_s * 100:.0f}% of these crashes; "
                            f"extra vigilance during arrival/dismissal."))

    if not any("Helmet" in s or "helmet" in s for _, s in items):
        items.append(("✅", f"Helmet check. {helmet_share * 100:.0f}% of riders in these crashes "
                            f"weren't wearing one."))

    while len(items) < 3:
        extras = [
            ("✅", "Lights on, bright clothing, signal turns clearly."),
            ("✅", "Assume drivers don't see you. Eye contact at intersections."),
            ("✅", "Avoid riding right after rain. Wet pavement adds risk."),
        ]
        for e in extras:
            if e not in items and len(items) < 3:
                items.append(e)
    return items[:5]


# ---------- FILTERING ----------
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df
    if f["time_band"] != "Any":
        out = out[out["time_band"] == f["time_band"]]
    if f["day"] != "Any":
        out = out[out["Day of Week"] == f["day"]]
    if f["roadway"] != "Any":
        out = out[out["Roadway Part"] == f["roadway"]]
    if f["surface"] != "Any":
        out = out[out["Surface Condition"] == f["surface"]]
    if f["intersection"] != "Any":
        out = out[out["intersection_bucket"] == f["intersection"]]
    if f["traffic_ctrl"] != "Any":
        out = out[out["Traffic Control Type"] == f["traffic_ctrl"]]
    sl_lo, sl_hi = f["speed_range"]
    speed_mask = out["Speed Limit"].between(sl_lo, sl_hi) | out["Speed Limit"].isna()
    out = out[speed_mask]
    return out


# ---------- CHART HELPERS ----------
def base_layout(height: int = 220, b: int = 28, t: int = 20, l: int = 36, r: int = 8) -> dict:
    # Plotly font sizes pulled from the viewport-aware budget — smaller on
    # phones, larger on 4K — so chart text auto-scales with the device.
    f_body = _h("font_chart_body", 10)
    f_tick = _h("font_chart_tick", 10)
    return dict(
        margin=dict(l=l, r=r, t=t, b=b),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, showlegend=False, autosize=False,
        font=dict(family=FONT, size=f_body, color=BODY),
        hoverlabel=dict(bgcolor=INK, font_color="white", font_family=FONT),
        xaxis=dict(automargin=True, tickfont=dict(size=f_tick, color=BODY)),
        yaxis=dict(automargin=True, tickfont=dict(size=f_tick, color=BODY)),
    )


def empty_fig(height: int = 220, msg: str = "No data for this slice") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=MUTED, size=11))
    fig.update_layout(**base_layout(height=height),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _h(key: str, fallback: int) -> int:
    """Read a budgeted height from session state, fall back if absent."""
    return st.session_state.get("budget", {}).get(key, fallback)


def gauge_chart(v: Verdict) -> tuple[go.Figure, float, str]:
    """Returns the gauge figure AND the score + color so the value can be
    rendered as a separate Streamlit element below the chart. Avoids the
    'gauge+number' overflow that clipped the number on narrow cards.
    """
    if v.code == "INSUFFICIENT" or v.baseline == 0:
        score, color = 0.0, MUTED
    else:
        score = float(min(100, (v.ksi_rate / v.baseline) * 50))
        color = v.color
    fig = go.Figure(go.Indicator(
        mode="gauge", value=score,
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0,
                     "tickfont": {"size": 9, "color": MUTED}, "nticks": 5},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": SURFACE, "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": GREEN_SOFT},
                {"range": [50, 75], "color": AMBER_SOFT},
                {"range": [75, 100], "color": RED_SOFT},
            ],
        }, domain={"x": [0.05, 0.95], "y": [0.1, 1]},
    ))
    fig.update_layout(**base_layout(height=_h("gauge", 130), b=4, t=4, l=4, r=4))
    return fig, score, color


def hour_curve(slice_df: pd.DataFrame, all_df: pd.DataFrame, color: str) -> go.Figure:
    hours_idx = pd.Index(range(24), name="hour")
    if len(slice_df) == 0:
        return empty_fig(height=220)
    cur = slice_df.groupby("hour")["is_ksi"].mean().reindex(hours_idx, fill_value=0) * 100
    base = all_df.groupby("hour")["is_ksi"].mean().reindex(hours_idx, fill_value=0) * 100
    cur_n = slice_df.groupby("hour").size().reindex(hours_idx, fill_value=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours_idx, y=base, mode="lines",
                             line=dict(color=MUTED, width=1, dash="dash"),
                             name="Baseline", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=hours_idx, y=cur, mode="lines+markers",
        line=dict(color=color, width=2.4, shape="spline"),
        marker=dict(color=color, size=6, line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.10)",
        hovertemplate="%{x:02d}:00<br>%{y:.1f}% serious-injury<br>%{customdata} crashes<extra></extra>",
        customdata=cur_n.values, name="Slice",
    ))
    now_h = datetime.now().hour
    fig.add_vline(x=now_h, line=dict(color=INK, width=1, dash="dot"),
                  annotation_text=f"Now: {now_h:02d}:00", annotation_position="top right",
                  annotation_font=dict(size=10, color=INK))
    if cur.max() > 0:
        worst_h = int(cur.idxmax())
        fig.add_annotation(x=worst_h, y=float(cur.max()),
                           text=f"Worst: {cur.max():.0f}% @ {worst_h:02d}:00",
                           showarrow=True, arrowhead=0, ax=0, ay=-22,
                           font=dict(size=9, color=color), bgcolor="white",
                           bordercolor=BORDER, borderwidth=1)
    fig.update_layout(**base_layout(height=_h("hour_curve", 220), l=44, b=30))
    fig.update_yaxes(ticksuffix="%", gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=False, dtick=4, range=[-0.5, 23.5])
    return fig


def small_multiple(slice_df: pd.DataFrame, baseline: float, dim: str,
                   order: Optional[list] = None, color: str = GREEN) -> go.Figure:
    if len(slice_df) == 0:
        return empty_fig(height=120)
    grp = slice_df.groupby(dim).agg(n=("is_ksi", "size"), ksi=("is_ksi", "mean"))
    if order is not None:
        grp = grp.reindex(order).fillna(0)
    else:
        grp = grp.sort_values("ksi", ascending=False).head(8)
    if len(grp) == 0:
        return empty_fig(height=120)
    rates = grp["ksi"].values * 100
    bar_colors = [color if r > baseline * 100 else "#D1D5DB" for r in rates]
    fig = go.Figure(go.Bar(
        x=[str(i)[:10] for i in grp.index], y=rates, marker_color=bar_colors,
        hovertemplate="%{x}<br>%{y:.1f}% KSI<br>%{customdata} crashes<extra></extra>",
        customdata=grp["n"].values, width=0.7,
    ))
    fig.add_hline(y=baseline * 100, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(**base_layout(height=_h("small_tile", 120), l=28, r=4, t=4, b=22))
    fig.update_yaxes(range=[0, max(40, rates.max() * 1.2 if len(rates) else 40)],
                     ticksuffix="%", gridcolor=BORDER, tickfont=dict(size=8), nticks=3)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=8))
    return fig


def hour_curve_small(slice_df: pd.DataFrame, baseline: float, color: str) -> go.Figure:
    if len(slice_df) == 0:
        return empty_fig(height=120)
    hours_idx = pd.Index(range(24), name="hour")
    cur = slice_df.groupby("hour")["is_ksi"].mean().reindex(hours_idx, fill_value=0) * 100
    fig = go.Figure(go.Scatter(
        x=hours_idx, y=cur, mode="lines",
        line=dict(color=color, width=2, shape="spline"),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.10)",
        hovertemplate="%{x:02d}:00<br>%{y:.1f}% KSI<extra></extra>",
    ))
    fig.add_hline(y=baseline * 100, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(**base_layout(height=_h("small_tile", 120), l=28, r=4, t=4, b=22))
    fig.update_yaxes(range=[0, 40], ticksuffix="%", gridcolor=BORDER,
                     tickfont=dict(size=8), nticks=3)
    fig.update_xaxes(showgrid=False, dtick=6, tickfont=dict(size=8))
    return fig


def severity_bar(slice_df: pd.DataFrame, color: str) -> go.Figure:
    if len(slice_df) == 0:
        return empty_fig(height=180)
    order = ["Killed", "Incapacitating Injury", "Non-Incapacitating Injury",
             "Possible Injury", "Not Injured"]
    counts = slice_df["Crash Severity"].value_counts().reindex(order, fill_value=0)
    colors = [color, color, "#D1D5DB", "#D1D5DB", "#D1D5DB"]
    labels = [s.replace("Incapacitating", "Incap.").replace("Non-Incap. Injury", "Non-incap.")
              for s in counts.index]
    fig = go.Figure(go.Bar(
        y=labels, x=counts.values, orientation="h", marker_color=colors,
        text=counts.values, textposition="outside", textfont=dict(size=10, color=INK),
        hovertemplate="%{y}<br>%{x} crashes<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=_h("drivers_chart", 180), l=110, r=24, t=4, b=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def heatmap(slice_df: pd.DataFrame, color: str) -> go.Figure:
    if len(slice_df) == 0:
        return empty_fig(height=180)
    try:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_map = dict(zip(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                            "Saturday", "Sunday"], days))
        df2 = slice_df.copy()
        df2["d_short"] = df2["Day of Week"].map(day_map)
        df2 = df2.dropna(subset=["d_short"])
        if len(df2) == 0:
            return empty_fig(height=180)
        grp = df2.groupby(["d_short", "hour"]).agg(
            n=("is_ksi", "size"), ksi=("is_ksi", "mean")).reset_index()
        z = np.full((7, 24), np.nan)
        counts = np.zeros((7, 24), dtype=int)
        for _, r in grp.iterrows():
            if r["d_short"] not in days:
                continue
            try:
                i = days.index(r["d_short"]); j = int(r["hour"])
            except (ValueError, TypeError):
                continue
            if not (0 <= j < 24):
                continue
            counts[i, j] = int(r["n"])
            if r["n"] >= 5:
                z[i, j] = float(r["ksi"]) * 100
        if np.isnan(z).all():
            zmax_val = 40.0
        else:
            zmax_val = max(40.0, float(np.nanmax(z)))
        rgb = tuple(int(color[k:k + 2], 16) for k in (1, 3, 5))
        colorscale = [[0.0, "rgba(255,255,255,0)"],
                      [0.001, f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)"],
                      [1.0, f"rgba({rgb[0]},{rgb[1]},{rgb[2]},1)"]]
        fig = go.Figure(go.Heatmap(
            z=z, x=list(range(24)), y=days, colorscale=colorscale,
            showscale=False, customdata=counts,
            hovertemplate="%{y} %{x:02d}:00<br>%{z:.0f}% KSI<br>%{customdata} crashes<extra></extra>",
            zmin=0, zmax=zmax_val, xgap=1, ygap=1,
        ))
        fig.update_layout(**base_layout(height=_h("drivers_chart", 180), l=32, r=4, t=4, b=22))
        fig.update_xaxes(dtick=4, tickfont=dict(size=8), showgrid=False)
        fig.update_yaxes(tickfont=dict(size=9), showgrid=False)
        return fig
    except Exception:
        return empty_fig(height=180, msg="Heatmap unavailable for this slice")


def roadway_breakdown(slice_df: pd.DataFrame, color: str) -> tuple[go.Figure, str]:
    if len(slice_df) == 0:
        return empty_fig(height=180), "No data."
    counts = slice_df["Roadway Part"].value_counts().head(5)
    pct = counts / counts.sum() * 100
    bar_colors = [color] + ["#D1D5DB"] * (len(pct) - 1)
    labels = [s.replace("/", " / ")[:24] for s in pct.index]
    fig = go.Figure(go.Bar(
        y=labels, x=pct.values, orientation="h", marker_color=bar_colors,
        text=[f"{v:.0f}%" for v in pct.values], textposition="outside",
        textfont=dict(size=10, color=INK),
        hovertemplate="%{y}<br>%{x:.1f}% of slice<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=_h("drivers_chart", 180), l=140, r=30, t=4, b=12))
    fig.update_xaxes(visible=False, range=[0, max(pct.values) * 1.18])
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    cap = f"{pct.index[0]} dominates this slice ({pct.iloc[0]:.0f}%)."
    return fig, cap


def donut_pair(slice_df: pd.DataFrame, all_df: pd.DataFrame, color: str
               ) -> tuple[go.Figure, go.Figure, str, str]:
    helmet_share = (slice_df["helmet"] == "Not Worn").mean() if len(slice_df) else 0
    valid = all_df["Average Daily Traffic Amount"].dropna()
    if len(valid) and len(slice_df):
        q3 = valid.quantile(0.75)
        sl_t = slice_df["Average Daily Traffic Amount"].dropna()
        traffic_share = (sl_t >= q3).mean() if len(sl_t) else 0
    else:
        traffic_share = 0

    def _donut(share: float, lab: str) -> go.Figure:
        # Defensive: if mean of empty slice returns NaN, treat as 0
        if share is None or (isinstance(share, float) and np.isnan(share)):
            share = 0.0
        share_pct = float(max(0.0, min(100.0, share * 100)))
        fig = go.Figure(go.Pie(
            values=[share_pct, 100 - share_pct], hole=0.7,
            marker=dict(colors=[color, "#E5E7EB"], line=dict(color="white", width=1)),
            textinfo="none", sort=False, direction="clockwise",
            hovertemplate=lab + ": %{value:.0f}%<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{share_pct:.0f}%</b>", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=18, color=INK))
        # base_layout already sets showlegend=False — don't pass it again here
        # or you get TypeError: multiple values for keyword 'showlegend'.
        fig.update_layout(**base_layout(height=_h("donut", 110), l=4, r=4, t=4, b=4))
        return fig

    return (_donut(helmet_share, "Helmet not worn"),
            _donut(traffic_share, "High-traffic exposure"),
            f"{helmet_share * 100:.0f}% of riders in these crashes weren't wearing a helmet.",
            f"{traffic_share * 100:.0f}% of these crashes happened on the busiest streets.")


# ---------- STYLE ----------
def inject_css() -> None:
    st.markdown(f"""
    <style>
    html, body, .stApp {{ background: {SURFACE}; }}
    body, .stApp, [data-testid="stAppViewContainer"] {{
      font-family: {FONT}; color: {BODY};
    }}
    .block-container {{
      padding-top: 0.6rem !important; padding-bottom: 0.4rem !important;
      padding-left: 1.2rem !important; padding-right: 1.2rem !important;
      max-width: 1480px;
    }}
    #MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
    div[data-testid="stToolbar"] {{ display: none !important; }}
    /* Hide Streamlit Cloud "Manage app" + page-name badge */
    .stDeployButton, [class*="viewerBadge"], [class*="ViewerBadge"] {{ display: none !important; }}
    iframe[title="streamlitApp"] {{ display: none !important; }}
    div[data-testid="stStatusWidget"] {{ display: none !important; }}
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    /* Tighten the spacing between bands so the dashboard reads as one frame */
    div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
    /* Streamlit's st.container(border=True) renders this wrapper — style it
       to match our design tokens (16px radius, light border, generous pad). */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
      background: {SURFACE} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 16px !important;
      padding: 16px !important;
      box-sizing: border-box !important;
      height: 100% !important;
      display: flex !important;
      flex-direction: column !important;
    }}
    /* Equal-height rows: every column in a row stretches to match the tallest
       sibling, so the bordered card inside each column ends at the same line. */
    div[data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
      display: flex !important;
      flex-direction: column !important;
    }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]
      > div[data-testid="stVerticalBlock"] {{
      height: 100% !important;
      display: flex !important;
      flex-direction: column !important;
    }}
    /* The Plotly chart inside a flex container — let it grow to fill */
    div[data-testid="stVerticalBlockBorderWrapper"]
      div[data-testid="stPlotlyChart"] {{
      flex: 1 1 auto !important;
      min-height: 0 !important;
    }}

    /* Row markers are 0-height and invisible (kept for backwards compat). */
    .acs-row-marker {{
      height: 0 !important; margin: 0 !important; padding: 0 !important;
      display: block !important;
    }}
    /* Hide the internal scrollbar that st.container(height=N) creates.
       Card heights are explicit; content always fits because charts inside
       are sized to fit. No need for scroll. */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
      overflow: hidden !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]
      > div[data-testid="stVerticalBlock"] {{
      overflow: hidden !important;
    }}
    /* Decision row (3 cards) — taller to accommodate verdict + KPI strip.
       `+` (adjacent sibling) targets ONLY the row block immediately after
       the marker, not every following row. */
    div[data-testid="stElementContainer"]:has(.acs-row-decision)
      + div[data-testid="stHorizontalBlock"]
      div[data-testid="stVerticalBlockBorderWrapper"] {{
      height: 290px !important;
      min-height: 290px !important;
      max-height: 290px !important;
      overflow: hidden;
    }}
    /* Pattern row (2 cards) — tallest, holds small-multiples grid + curve */
    div[data-testid="stElementContainer"]:has(.acs-row-pattern)
      + div[data-testid="stHorizontalBlock"]
      div[data-testid="stVerticalBlockBorderWrapper"] {{
      height: 380px !important;
      min-height: 380px !important;
      max-height: 380px !important;
      overflow: hidden;
    }}
    /* Drivers row (4 cards) — compact for severity/heatmap/breakdown/donuts */
    div[data-testid="stElementContainer"]:has(.acs-row-drivers)
      + div[data-testid="stHorizontalBlock"]
      div[data-testid="stVerticalBlockBorderWrapper"] {{
      height: 260px !important;
      min-height: 260px !important;
      max-height: 260px !important;
      overflow: hidden;
    }}
    /* Compact selectbox + slider — clean borders, no double outlines */
    div[data-baseweb="select"] {{ border-radius: 10px !important; }}
    div[data-baseweb="select"] > div {{
      min-height: 38px !important; font-size: 13px !important;
      border: 1px solid {BORDER} !important; border-radius: 10px !important;
      background: {SURFACE} !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{ border-color: {INK} !important; }}
    .acs-appbar {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 4px 0 12px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 14px;
    }}
    /* FLUID TYPOGRAPHY — every text size uses clamp(min, vw-scaled, max)
       so it shrinks gracefully on phones and grows on 4K, without any
       hardcoded pixel sizes that could break on extreme viewports. */
    .acs-appbar .brand {{
      font-weight: 700; color: {INK};
      font-size: clamp(15px, 1.2vw, 20px); letter-spacing: -0.01em;
    }}
    .acs-appbar .meta {{
      font-size: clamp(9px, 0.75vw, 12px); color: {MUTED};
      text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .acs-flabel {{
      font-size: clamp(9px, 0.75vw, 12px); text-transform: uppercase;
      letter-spacing: 0.08em; color: {MUTED}; margin: 0 0 4px 0; font-weight: 600;
    }}
    .acs-pill {{
      display: inline-block; padding: 4px 10px; border-radius: 6px;
      color: white; font-size: clamp(10px, 0.78vw, 12px); font-weight: 700;
      letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;
    }}
    .acs-hero {{
      font-size: clamp(20px, 2.0vw, 34px); line-height: 1.2; font-weight: 700;
      color: {INK}; letter-spacing: -0.01em; margin: 0 0 6px 0;
    }}
    .acs-sub {{
      font-size: clamp(11px, 0.92vw, 14px); line-height: 1.5;
      color: {BODY}; margin: 0 0 10px 0;
    }}
    .acs-kpi-row {{
      display: flex; gap: clamp(8px, 1vw, 18px);
      padding-top: 10px; border-top: 1px solid {BORDER};
    }}
    .acs-kpi {{ flex: 1; min-width: 0; }}
    .acs-kpi .label {{
      font-size: clamp(8px, 0.7vw, 10.5px); color: {MUTED};
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
      margin-bottom: 4px;
    }}
    .acs-kpi .num {{
      font-size: clamp(16px, 1.55vw, 26px); color: {INK};
      font-weight: 700; line-height: 1;
    }}
    .acs-section {{
      font-size: clamp(12px, 1.0vw, 16px); color: {INK}; font-weight: 700;
      margin: 0 0 6px 0; letter-spacing: -0.005em;
    }}
    .acs-tile-title {{
      font-size: clamp(8px, 0.7vw, 11px); color: {MUTED};
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
      margin: 0 0 4px 0;
    }}
    .acs-check {{ display: flex; flex-direction: column; gap: 8px; }}
    .acs-check .item {{
      display: flex; align-items: flex-start; gap: 8px;
      font-size: clamp(11px, 0.92vw, 14px); line-height: 1.45; color: {BODY};
    }}
    .acs-check .icon {{
      flex: 0 0 auto; font-size: clamp(12px, 1vw, 15px); line-height: 1.4;
    }}
    .acs-gauge-value {{
      text-align: center; font-size: clamp(20px, 1.85vw, 32px);
      font-weight: 700; line-height: 1; margin: 2px 0 4px 0;
      letter-spacing: -0.01em;
    }}
    .acs-gauge-value .suffix {{
      font-size: clamp(10px, 0.85vw, 14px); color: {MUTED}; font-weight: 500;
    }}
    .acs-tier {{
      text-align: center; font-size: clamp(10px, 0.85vw, 13px);
      color: {INK}; font-weight: 600; letter-spacing: 0.04em;
      text-transform: uppercase; margin-top: 2px;
    }}
    .acs-tier .delta {{
      display: block; font-size: clamp(9px, 0.78vw, 12px); font-weight: 500;
      color: {MUTED}; text-transform: none; letter-spacing: 0; margin-top: 2px;
    }}
    .acs-footer {{
      font-size: clamp(9px, 0.78vw, 12px); color: {MUTED};
      padding-top: 8px; border-top: 1px solid {BORDER};
      margin-top: 10px; letter-spacing: 0.02em;
    }}
    .acs-banner {{
      font-size: clamp(9px, 0.78vw, 12px); color: {MUTED};
      padding: 4px 8px; background: {CARD}; border: 1px solid {BORDER};
      border-radius: 6px; display: inline-block; margin-bottom: 8px;
    }}
    .acs-caption {{
      font-size: clamp(9px, 0.78vw, 12px); color: {MUTED}; margin-top: 4px;
    }}
    label[data-testid="stWidgetLabel"] {{ display: none !important; }}
    div[data-testid="stSlider"] {{ padding-top: 4px; }}
    </style>
    """, unsafe_allow_html=True)


# ---------- BANDS ----------
def render_appbar(n_total: int) -> None:
    st.markdown(f"""
    <div class="acs-appbar">
      <div class="brand">🚲 Austin CycleSafe</div>
      <div class="meta">Go / No-Go decision · {n_total:,} crashes · 2010–2017 · City of Austin</div>
    </div>
    """, unsafe_allow_html=True)


def render_filter_rail(df: pd.DataFrame) -> dict:
    cols = st.columns([1, 1, 1, 1, 1, 1, 1.4])
    bands = ["Any", "Morning 5–10", "Midday 10–15", "Evening 15–19", "Night 19–5"]
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_unique = ["Any"] + [d for d in days_order if d in df["Day of Week"].unique()]
    rd_opts = ["Any"] + sorted(df["Roadway Part"].dropna().unique().tolist())
    sf_opts = ["Any"] + sorted(df["Surface Condition"].dropna().unique().tolist())
    isect_opts = ["Any", "Intersection", "Non Intersection"]
    tc_opts = ["Any"] + df["Traffic Control Type"].value_counts().head(8).index.tolist()
    sl_valid = df["Speed Limit"].dropna()
    if len(sl_valid):
        lo, hi = int(sl_valid.quantile(0.05)), int(sl_valid.quantile(0.95))
    else:
        lo, hi = 25, 45

    labels = ["Time of Day", "Day", "Roadway", "Surface", "Intersection", "Traffic Control"]
    options = [bands, days_unique, rd_opts, sf_opts, isect_opts, tc_opts]
    keys = ["time_band", "day", "roadway", "surface", "intersection", "traffic_ctrl"]
    out: dict = {}
    for col, lab, opts, k in zip(cols[:6], labels, options, keys):
        with col:
            st.markdown(f'<div class="acs-flabel">{lab}</div>', unsafe_allow_html=True)
            out[k] = st.selectbox(lab, opts, label_visibility="collapsed", key=f"flt_{k}")
    with cols[6]:
        st.markdown('<div class="acs-flabel">Speed Limit (mph)</div>', unsafe_allow_html=True)
        out["speed_range"] = st.slider("Speed Limit", min_value=lo, max_value=hi,
                                       value=(lo, hi), label_visibility="collapsed")
    return out


def render_decision_band(slice_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    v = decide(slice_df, all_df)
    drivers = compute_drivers(slice_df, all_df)
    pill_color = {"GO": GREEN, "CAUTION": AMBER, "NO-GO": RED, "INSUFFICIENT": MUTED}[v.code]
    your_disp = f"{v.ksi_rate*100:.1f}%" if v.code != "INSUFFICIENT" else "n/a"
    base_disp = f"{v.baseline*100:.1f}%"
    n_disp = f"{v.n:,}"

    # Marker — CSS uses :has() to find this and lock all cards in this row
    # to the same explicit pixel height.
    st.markdown('<div class="acs-row-marker acs-row-decision"></div>',
                unsafe_allow_html=True)
    # Row height pulled from session-state budget — adapts to viewport size
    DECISION_H = _h("decision_card", 220)
    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    with c1:
        with st.container(border=True, height=DECISION_H):
            st.markdown(f"""
            <span class="acs-pill" style="background:{pill_color};">{v.code}</span>
            <div class="acs-hero">{v.headline}</div>
            <div class="acs-sub">{v.sub}</div>
            <div class="acs-kpi-row">
              <div class="acs-kpi"><div class="label">Your conditions</div>
                <div class="num" style="color:{pill_color};">{your_disp}</div></div>
              <div class="acs-kpi"><div class="label">Austin baseline</div>
                <div class="num">{base_disp}</div></div>
              <div class="acs-kpi"><div class="label">Matching crashes</div>
                <div class="num">{n_disp}</div></div>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        with st.container(border=True, height=DECISION_H):
            st.markdown('<div class="acs-tile-title">Risk gauge</div>',
                        unsafe_allow_html=True)
            gauge_fig, score, gauge_color = gauge_chart(v)
            st.plotly_chart(gauge_fig, use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
            # Render score as a separate styled element — guaranteed to fit
            # the card width since it's plain text inside flex layout.
            if v.code == "INSUFFICIENT":
                tier, delta_lbl = "Insufficient", f"{v.n} matching crashes"
                score_disp = "n/a"
            else:
                tier = "Low" if score < 50 else ("Elevated" if score < 75 else "High")
                delta_lbl = f"{v.delta_pp:+.1f} pp vs baseline"
                score_disp = f"{score:.0f}"
            st.markdown(
                f'<div class="acs-gauge-value" style="color:{gauge_color};">{score_disp}'
                f'<span class="suffix">/100</span></div>'
                f'<div class="acs-tier">{tier}'
                f'<span class="delta">{delta_lbl}</span></div>',
                unsafe_allow_html=True)
    with c3:
        with st.container(border=True, height=DECISION_H):
            st.markdown('<div class="acs-tile-title">Before you ride</div>',
                        unsafe_allow_html=True)
            items_html = "".join(
                f'<div class="item"><div class="icon">{ic}</div><div>{msg}</div></div>'
                for ic, msg in drivers
            )
            st.markdown(f'<div class="acs-check">{items_html}</div>',
                        unsafe_allow_html=True)


def render_pattern_band(slice_df: pd.DataFrame, all_df: pd.DataFrame, color: str) -> None:
    baseline = float(all_df["is_ksi"].mean())
    st.markdown('<div class="acs-row-marker acs-row-pattern"></div>',
                unsafe_allow_html=True)
    PATTERN_H = _h("pattern_card", 300)
    left, right = st.columns([1.5, 1.0])

    with left:
        with st.container(border=True, height=PATTERN_H):
            st.markdown('<div class="acs-section">Where the risk lives in your data</div>',
                        unsafe_allow_html=True)
            r1 = st.columns(3); r2 = st.columns(3)
            tiles = [
                ("By hour", lambda: hour_curve_small(slice_df, baseline, color), r1[0]),
                ("By day of week",
                 lambda: small_multiple(slice_df, baseline, "Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    color), r1[1]),
                ("By roadway part",
                 lambda: small_multiple(slice_df, baseline, "Roadway Part", color=color), r1[2]),
                ("By speed limit",
                 lambda: small_multiple(slice_df, baseline, "speed_bucket",
                    ["≤25", "30", "35", "40", "≥45"], color), r2[0]),
                ("By surface",
                 lambda: small_multiple(slice_df, baseline, "Surface Condition", color=color), r2[1]),
                ("By intersection",
                 lambda: small_multiple(slice_df, baseline, "intersection_bucket",
                    ["Intersection", "Non Intersection", "Other"], color), r2[2]),
            ]
            for title, fn, col in tiles:
                with col:
                    st.markdown(f'<div class="acs-tile-title">{title}</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(fn(), use_container_width=True,
                                    config={"displayModeBar": False, "responsive": True})
    with right:
        with st.container(border=True, height=PATTERN_H):
            st.markdown('<div class="acs-section">When risk peaks across the day</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(hour_curve(slice_df, all_df, color), use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
            st.markdown(
                '<div class="acs-caption">Dotted line is the citywide baseline. '
                'Vertical rule marks the current local hour.</div>',
                unsafe_allow_html=True)


def render_drivers_strip(slice_df: pd.DataFrame, all_df: pd.DataFrame, color: str) -> None:
    st.markdown('<div class="acs-row-marker acs-row-drivers"></div>',
                unsafe_allow_html=True)
    DRIVERS_H = _h("drivers_card", 220)
    cols = st.columns(4)
    rd_fig, rd_cap = roadway_breakdown(slice_df, color)
    don_helm, don_traf, _, _ = donut_pair(slice_df, all_df, color)

    with cols[0]:
        with st.container(border=True, height=DRIVERS_H):
            st.markdown('<div class="acs-tile-title">Severity mix</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(severity_bar(slice_df, color), use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
    with cols[1]:
        with st.container(border=True, height=DRIVERS_H):
            st.markdown('<div class="acs-tile-title">Day × hour heatmap</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(heatmap(slice_df, color), use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
    with cols[2]:
        with st.container(border=True, height=DRIVERS_H):
            st.markdown('<div class="acs-tile-title">Roadway breakdown</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(rd_fig, use_container_width=True,
                            config={"displayModeBar": False, "responsive": True})
            st.markdown(f'<div class="acs-caption">{rd_cap}</div>',
                        unsafe_allow_html=True)
    with cols[3]:
        with st.container(border=True, height=DRIVERS_H):
            st.markdown('<div class="acs-tile-title">Helmet & traffic exposure</div>',
                        unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(don_helm, use_container_width=True,
                                config={"displayModeBar": False, "responsive": True})
                st.markdown(
                    '<div class="acs-caption" style="text-align:center;">Helmet not worn</div>',
                    unsafe_allow_html=True)
            with d2:
                st.plotly_chart(don_traf, use_container_width=True,
                                config={"displayModeBar": False, "responsive": True})
                st.markdown(
                    '<div class="acs-caption" style="text-align:center;">High-traffic exposure</div>',
                    unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown("""
    <div class="acs-footer">
      Sample-size floor: 30 crashes · KSI = Killed + Incapacitating Injury ·
      Data: City of Austin Open Data, 2010–2017 ·
      Built for the Week 4 Building Products assignment.
    </div>
    """, unsafe_allow_html=True)


# ---------- MAIN ----------
def main() -> None:
    inject_css()

    # Capture viewport once on first render, cache in session state.
    # Heights below adapt to whatever the user's actual screen is.
    vh, vw = get_viewport()
    budget = layout_budget(vh)
    st.session_state.budget = budget
    st.session_state.is_mobile = vw < 768

    df, source = load_data()
    render_appbar(len(df))
    if "synthetic" in source.lower():
        st.markdown(f'<div class="acs-banner">Data source: {source}. '
                    f'Drop a real <code>bike_crash.csv</code> next to this app to use live data.</div>',
                    unsafe_allow_html=True)

    filters = render_filter_rail(df)
    sliced = apply_filters(df, filters)

    v = decide(sliced, df)
    band_color = {"GO": GREEN, "CAUTION": AMBER, "NO-GO": RED, "INSUFFICIENT": MUTED}[v.code]

    render_decision_band(sliced, df)
    render_pattern_band(sliced, df, band_color)
    render_drivers_strip(sliced, df, band_color)
    render_footer()


if __name__ == "__main__":
    main()