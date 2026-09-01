import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://karachi-aqi-predictor.onrender.com"

st.set_page_config(
    page_title="Pearls AQI Observatory",
    page_icon="🌫️",
    layout="wide"
)

KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011


# ============================================================
# AQI BAND SYSTEM
# ============================================================

BANDS = [
    (50,  "Good",                            "#5FD98A", "☀️"),
    (100, "Moderate",                        "#E8C547", "🌤️"),
    (150, "Unhealthy for Sensitive Groups",  "#F0904A", "😷"),
    (200, "Unhealthy",                       "#E85B4C", "🌫️"),
    (300, "Very Unhealthy",                  "#A366D9", "⚠️"),
    (999, "Hazardous",                       "#7A2E3A", "☠️"),
]

ADVICE = {
    "Good": "Air quality is satisfactory. Outdoor activities are generally safe.",
    "Moderate": "Air quality is acceptable. Sensitive individuals should monitor conditions.",
    "Unhealthy for Sensitive Groups": "Sensitive groups should reduce prolonged or heavy outdoor activity.",
    "Unhealthy": "Everyone may experience health effects. Consider reducing outdoor activity.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health emergency: avoid outdoor exposure and follow official guidance.",
}


def get_band(aqi):
    for limit, label, color, icon in BANDS:
        if aqi <= limit:
            return label, color, icon
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def haze_blur_px(aqi):
    """Signature effect: readout gets visually hazier as AQI rises."""
    return round(min(aqi / 300 * 2.6, 2.6), 2)


ACTION_TIPS = {
    "Good": [
        "Great day for outdoor activity — walks, runs, and exercise are all fine.",
        "Windows can stay open for fresh air.",
        "No mask needed.",
    ],
    "Moderate": [
        "Outdoor activity is fine for most people.",
        "If you have asthma or a heart/lung condition, watch for symptoms during long outdoor exertion.",
        "Windows can stay open; consider a purifier indoors if you're sensitive.",
    ],
    "Unhealthy for Sensitive Groups": [
        "Sensitive groups should limit prolonged or intense outdoor exertion.",
        "Consider an N95 mask outdoors if you're in a sensitive group.",
        "Everyone else can continue normal activity, but take it easier during long exercise.",
    ],
    "Unhealthy": [
        "Limit outdoor exercise, especially cardio.",
        "Wear an N95 mask if you're heading outside.",
        "Close windows and run an air purifier indoors if you have one.",
    ],
    "Very Unhealthy": [
        "Avoid outdoor exertion entirely.",
        "Wear a mask outdoors if you must go out.",
        "Keep windows closed and run an air purifier.",
    ],
    "Hazardous": [
        "Stay indoors as much as possible.",
        "Wear an N95/N99 mask if you absolutely must go outside.",
        "Seal windows, run a purifier, and avoid all outdoor exertion.",
    ],
}


# ============================================================
# STYLE — dusk observatory theme
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg-void: #10141b;
        --bg-panel: #171d27;
        --bg-panel-2: #1e2531;
        --border: #2a313d;
        --text-primary: #edf1f5;
        --text-muted: #8b94a3;
        --haze-gold: #e8a33d;
    }

    .stApp {
        background: radial-gradient(ellipse at top, #1c2431 0%, #10141b 60%);
        color: var(--text-primary);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border);
    }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: var(--haze-gold);
        margin-bottom: 4px;
    }

    .hero {
        background: linear-gradient(135deg, var(--bg-panel-2) 0%, var(--bg-panel) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 34px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 24px;
        margin-bottom: 8px;
    }

    .hero-number {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 6.2rem;
        line-height: 1;
        letter-spacing: -0.02em;
    }

    .hero-band {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        margin-top: 6px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid var(--border);
    }

    .hero-advice {
        max-width: 340px;
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .station-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }

    .station-day {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .station-date {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        margin: 2px 0 14px 0;
    }

    .station-aqi {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1;
    }

    .station-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        margin-top: 10px;
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
    }

    .mono-caption {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-muted);
        font-size: 0.78rem;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 14px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--haze-gold) !important;
    }

    .stMarkdown, .stMarkdown p, .stCaption, p, span, label {
        color: var(--text-primary);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    [data-testid="stDataFrame"] {
        color: var(--text-primary);
    }

    div[data-testid="stButton"] button {
        background: var(--haze-gold) !important;
        color: #10141b !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px #00000022;
    }

    div[data-testid="stButton"] button:hover {
        background: #f4b657 !important;
        color: #10141b !important;
    }

    div[data-testid="stButton"] button p {
        color: #10141b !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 12px !important;
        background: var(--bg-panel) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API REQUEST
# ============================================================

@st.cache_data(ttl=60)
def get_forecast():
    response = requests.get(
        f"{API_URL}/forecast",
        timeout=15
    )

    response.raise_for_status()

    return response.json()

# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown('<div class="eyebrow">Karachi · Air Quality Observatory</div>', unsafe_allow_html=True)
    st.markdown("## 🌫️ Pearls AQI Predictor")
    st.caption("AI-powered 3-day AQI forecast using historical pollution, weather and engineered time-series features.")

with header_right:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

try:
    data = get_forecast()
except requests.exceptions.ConnectionError:
    st.error(
        "❌ Cannot connect to the Karachi AQI API.\n\n"
        "Make sure the Flask API is running with:\n\n"
        "`python src\\api\\app.py`"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Unable to retrieve forecast: {e}")
    st.stop()

forecast = data.get("forecast", [])
if not forecast:
    st.error("No forecast data was returned by the API.")
    st.stop()

df = pd.DataFrame(forecast)

date_column = next((c for c in ["date", "forecast_date", "prediction_date"] if c in df.columns), None)
aqi_column = next((c for c in ["aqi", "AQI", "predicted_aqi", "prediction"] if c in df.columns), None)

if date_column is None or aqi_column is None:
    st.error("The API response format does not contain the expected date and AQI fields.")
    st.json(data)
    st.stop()

df["date"] = pd.to_datetime(df[date_column])
df["AQI"] = pd.to_numeric(df[aqi_column])
df = df.sort_values("date").reset_index(drop=True)

latest_aqi = float(df["AQI"].iloc[0])
category, color, icon = get_band(latest_aqi)
blur = haze_blur_px(latest_aqi)


# ============================================================
# HERO — signature haze-blur readout + map
# ============================================================

hero_col, map_col = st.columns([1.6, 1])

with hero_col:
    st.markdown(
        f"""
        <div class="hero" style="height:260px;">
            <div>
                <div class="eyebrow">Current forecast reading</div>
                <div class="hero-number" style="color:{color}; filter: blur({blur}px); text-shadow: 0 0 24px {color}55;">
                    {latest_aqi:.0f}
                </div>
                <div class="mono-caption">exact value · {latest_aqi:.1f} AQI</div>
                <div class="hero-band" style="color:{color}; border-color:{color}66;">{icon} {category}</div>
            </div>
            <div class="hero-advice">{ADVICE.get(category, "")}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    

with map_col:
    st.markdown('<div class="eyebrow" style="margin-bottom:10px;">📍 Karachi</div>', unsafe_allow_html=True)
    with st.container(height=220):
        st.map(
            pd.DataFrame({"lat": [KARACHI_LAT], "lon": [KARACHI_LON]}),
            size=100,
            color="#e8a33d",
            zoom=10,
            height=200
        )

st.write("")


# ============================================================
# TOP METRICS + GAUGE
# ============================================================

col_gauge, col_metrics = st.columns([1, 1.4])

with col_gauge:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_aqi,
        number={"font": {"color": color, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 350], "tickcolor": "#8b94a3"},
            "bar": {"color": color},
            "bgcolor": "#171d27",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#5FD98A66"},
                {"range": [50, 100], "color": "#E8C54766"},
                {"range": [100, 150], "color": "#F0904A66"},
                {"range": [150, 200], "color": "#E85B4C66"},
                {"range": [200, 300], "color": "#B27FE066"},
                {"range": [300, 350], "color": "#D65C7488"},
            ],
        }
    ))
    gauge.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf1f5"},
        margin=dict(l=20, r=20, t=30, b=10)
    )
    st.plotly_chart(gauge, use_container_width=True)

with col_metrics:
    m1, m2 = st.columns(2)
    with m1:
        st.metric("📍 Location", "Karachi")
        st.metric("Forecast Horizon", "3 Days")
    with m2:
        st.metric("Air Quality", category)
        st.metric("Peak in window", f"{df['AQI'].max():.0f}")

st.divider()


# ============================================================
# 3-DAY STATION CARDS
# ============================================================

st.markdown("### 📅 3-Day Forecast")

forecast_columns = st.columns(len(df))

for i, (_, row) in enumerate(df.iterrows()):
    aqi = float(row["AQI"])
    date = row["date"]
    band, band_color, band_icon = get_band(aqi)

    with forecast_columns[i]:
        st.markdown(
            f"""
            <div class="station-card">
                <div class="station-day">Day {i + 1}</div>
                <div class="station-date">{date.strftime('%a, %d %b')}</div>
                <div class="station-aqi" style="color:{band_color};">{aqi:.1f}</div>
                <div class="station-badge" style="color:{band_color}; border-color:{band_color}66;">
                    {band_icon} {band}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
st.divider()


# ============================================================
# BANDED TREND CHART
# ============================================================

st.markdown("### 📈 AQI Trend")

fig = go.Figure()

band_edges = [0, 50, 100, 150, 200, 300, max(350, df["AQI"].max() + 20)]
band_colors = ["#5FD98A", "#E8C547", "#F0904A", "#E85B4C", "#B27FE0", "#D65C74"]

for i in range(len(band_edges) - 1):
    fig.add_hrect(
        y0=band_edges[i], y1=band_edges[i + 1],
        fillcolor=band_colors[i], opacity=0.22, line_width=0
    )

fig.add_trace(go.Scatter(
    x=df["date"], y=df["AQI"],
    mode="lines+markers+text",
    text=[f"{x:.0f}" for x in df["AQI"]],
    textposition="top center",
    line=dict(width=4, color="#e8a33d"),
    marker=dict(size=11, color="#e8a33d", line=dict(width=2, color="#10141b")),
    fill="tozeroy",
    fillcolor="rgba(232,163,61,0.08)",
    name="Predicted AQI"
))

fig.update_layout(
    height=440,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf1f5", family="IBM Plex Sans"),
    xaxis=dict(title="Date", gridcolor="#2a313d"),
    yaxis=dict(title="AQI", gridcolor="#2a313d"),
    hovermode="x unified",
    margin=dict(l=30, r=30, t=45, b=20)
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# WHAT YOU CAN DO
# ============================================================

st.markdown(f"### ✅ What You Can Do — {category}")

tips = ACTION_TIPS.get(category, [])
tips_html = "".join(f'<div style="margin:8px 0; display:flex; gap:10px;"><span>✅</span><span>{t}</span></div>' for t in tips)

st.markdown(
    f"""
    <div class="station-card" style="border-color:{color}55;">
        {tips_html}
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# HEALTH GUIDANCE
# ============================================================

st.divider()
st.markdown("### 🩺 Health Guidance")

max_aqi = float(df["AQI"].max())
max_category, max_color, max_icon = get_band(max_aqi)

st.info(
    f"{max_icon} The highest predicted AQI during the next 3 days is "
    f"**{max_aqi:.1f} ({max_category})**.\n\n{ADVICE.get(max_category, '')}"
)

st.divider()

source_date = data.get("source_date")

if source_date:
    updated_text = f"Data through {source_date}"
else:
    updated_text = "Latest available data"

st.markdown(
    f'<div class="mono-caption" style="text-align:center;">'
    f'Pearls AQI Predictor · Karachi · {updated_text}'
    f'</div>',
    unsafe_allow_html=True
)
