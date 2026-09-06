import os
from datetime import datetime

import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PEARLS AQI OBSERVATORY — STREAMLIT DASHBOARD
# ============================================================

API_URL = os.getenv("AQI_API_URL", "https://karachi-aqi-predictor.onrender.com").rstrip("/")
SHAP_LOCAL_DIR = os.getenv("SHAP_DIR", "results/shap")
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

st.set_page_config(page_title="Pearls AQI Observatory", page_icon="🌫️", layout="wide", initial_sidebar_state="collapsed")

BANDS = [
    (50, "Good", "#5FD98A", "☀️"),
    (100, "Moderate", "#E8C547", "🌤️"),
    (150, "Unhealthy for Sensitive Groups", "#F0904A", "😷"),
    (200, "Unhealthy", "#E85B4C", "🌫️"),
    (300, "Very Unhealthy", "#A366D9", "⚠️"),
    (999, "Hazardous", "#7A2E3A", "☠️"),
]

ADVICE = {
    "Good": "Air quality is satisfactory. Outdoor activities are generally safe.",
    "Moderate": "Air quality is acceptable. Sensitive individuals should monitor conditions.",
    "Unhealthy for Sensitive Groups": "Sensitive groups should reduce prolonged or heavy outdoor activity.",
    "Unhealthy": "Everyone may experience health effects. Consider reducing outdoor activity.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health emergency: avoid outdoor exposure and follow official guidance.",
}

ACTION_TIPS = {
    "Good": ["Outdoor exercise is generally fine.", "Windows can stay open.", "Normal daily activity is appropriate."],
    "Moderate": ["Outdoor activity is fine for most people.", "Sensitive individuals should monitor symptoms.", "Consider reducing prolonged intense exertion if sensitive."],
    "Unhealthy for Sensitive Groups": ["Sensitive groups should limit prolonged intense outdoor activity.", "Consider a well-fitting respirator for prolonged outdoor exposure.", "Everyone else should take breaks during heavy exertion."],
    "Unhealthy": ["Limit prolonged outdoor exercise.", "Consider a well-fitting respirator outdoors.", "Close windows and use filtration indoors when possible."],
    "Very Unhealthy": ["Avoid outdoor exertion.", "Keep windows closed where practical.", "Use indoor air filtration if available."],
    "Hazardous": ["Stay indoors as much as possible.", "Avoid outdoor exertion.", "Follow local public-health guidance."],
}

# Fun "personality" tag for the day, purely a creative/editorial flourish
PERSONALITY = {
    "Good": ("The Clear-Skies Day", "🌈"),
    "Moderate": ("The Cautiously-Fine Day", "🌥️"),
    "Unhealthy for Sensitive Groups": ("The Sensitive-Souls Day", "🤧"),
    "Unhealthy": ("The Mask-Up Day", "😷"),
    "Very Unhealthy": ("The Stay-In Day", "🚪"),
    "Hazardous": ("The Lockdown-Level Day", "🚨"),
}

FACTS = [
    "AQI is a piecewise-linear index — equal jumps in the number don't mean equal jumps in pollution.",
    "PM2.5 particles are roughly 30x smaller than the width of a human hair.",
    "The WHO's 24-hour PM2.5 guideline is 15 µg/m³ — far stricter than many national AQI 'good' thresholds.",
    "Morning traffic and temperature inversions are a common reason AQI spikes overnight into early morning.",
    "SHAP values explain *how much* a feature moved a prediction — not whether that feature is good or bad for air quality.",
    "Dust storms can push AQI into 'Hazardous' territory in minutes, independent of local traffic or industry.",
]

SHAP_TARGETS = {"Day 1": "AQI_t+1", "Day 2": "AQI_t+2", "Day 3": "AQI_t+3"}


def get_band(aqi):
    for limit, label, color, icon in BANDS:
        if float(aqi) <= limit:
            return label, color, icon
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def safe_num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trend_label(current, future_values):
    if not future_values:
        return "NO FORECAST", "→", "#E8C547"
    delta = max(future_values) - current
    if delta >= 25:
        return "RAPIDLY WORSENING", "↗", "#E85B4C"
    if delta >= 10:
        return "WORSENING", "↗", "#F0904A"
    if delta <= -25:
        return "STRONGLY IMPROVING", "↘", "#5FD98A"
    if delta <= -10:
        return "IMPROVING", "↘", "#5FD98A"
    return "RELATIVELY STABLE", "→", "#E8C547"


def aqi_to_pm25(aqi):
    """Reverse the EPA PM2.5 AQI breakpoint table to estimate PM2.5 (µg/m3)."""
    breakpoints = [
        (0, 50, 0.0, 12.0),
        (51, 100, 12.1, 35.4),
        (101, 150, 35.5, 55.4),
        (151, 200, 55.5, 150.4),
        (201, 300, 150.5, 250.4),
        (301, 400, 250.5, 350.4),
        (401, 500, 350.5, 500.4),
    ]
    aqi = max(0, min(500, aqi))
    for ilo, ihi, plo, phi in breakpoints:
        if ilo <= aqi <= ihi:
            return plo + (aqi - ilo) * (phi - plo) / (ihi - ilo)
    return breakpoints[-1][3]


def cigarette_equivalent(aqi):
    """Berkeley Earth style rule of thumb: ~22 µg/m3 of PM2.5 ≈ 1 cigarette/day."""
    return round(aqi_to_pm25(aqi) / 22, 1)


def who_multiple(aqi):
    """How many times the WHO 24-hr PM2.5 guideline (15 µg/m3) today's air represents."""
    return round(aqi_to_pm25(aqi) / 15, 1)


THEMES = {
    "dark": {
        "void": "#0b1017", "panel": "#121a24", "panel2": "#182331", "border": "#283442",
        "text": "#edf2f7", "muted": "#8e9aaa", "gold": "#e8a33d", "shadow": "rgba(0,0,0,.35)",
    },
    "light": {
        "void": "#fff9f2", "panel": "#ffffff", "panel2": "#fff1de", "border": "#ffd9a8",
        "text": "#1f2430", "muted": "#6b7280", "gold": "#ff7a3d", "shadow": "rgba(255,122,61,.14)",
    },
}


def inject_css(theme):
    t = THEMES.get(theme, THEMES["dark"])
    root_vars = (
        f":root{{--void:{t['void']}; --panel:{t['panel']}; --panel2:{t['panel2']}; "
        f"--border:{t['border']}; --text:{t['text']}; --muted:{t['muted']}; "
        f"--gold:{t['gold']}; --shadow:{t['shadow']};}}"
    )
    static_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stBottomBlockContainer"]{
        background:var(--void)!important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"]{background:transparent!important;}
    .stApp{background:radial-gradient(circle at 15% 0%, var(--panel2) 0%, var(--void) 48%, var(--void) 100%)!important;font-family:'IBM Plex Sans',sans-serif;color:var(--text);}
    body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, .stCaption, p, span, label, li{color:var(--text);}
    h1,h2,h3,h4,h5,h6{font-family:'Space Grotesk',sans-serif!important;color:var(--text);}
    .block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px;}
    .eyebrow{font-family:'IBM Plex Mono',monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);font-size:.68rem;margin-bottom:5px;}

    .hero{background:linear-gradient(135deg,var(--panel2),var(--panel));border:1px solid var(--border);border-radius:22px;padding:30px 34px;min-height:285px;display:flex;justify-content:space-between;align-items:center;gap:25px;box-shadow:0 18px 60px var(--shadow);}
    .hero-label{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;}
    .hero-number{font-family:'Space Grotesk',sans-serif;font-size:6.8rem;line-height:.9;font-weight:700;letter-spacing:-.06em;margin:8px 0;}
    .hero-category{display:inline-block;border:1px solid;border-radius:999px;padding:7px 13px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;}
    .hero-personality{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.95rem;margin-top:10px;color:var(--gold);}
    .hero-note{max-width:340px;color:var(--muted);line-height:1.55;font-size:.92rem;}

    .panel{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--border);border-radius:18px;padding:20px;height:100%;box-shadow:0 12px 40px var(--shadow);}
    .mini-label{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;}
    .mini-value{font-family:'Space Grotesk',sans-serif;font-size:2.1rem;font-weight:700;margin-top:4px;color:var(--text);}

    .forecast-card{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--border);border-radius:17px;padding:19px;min-height:185px;box-shadow:0 10px 35px var(--shadow);}
    .forecast-day{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;}
    .forecast-aqi{font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:700;line-height:1;margin:15px 0 8px;}
    .badge{display:inline-block;padding:5px 9px;border:1px solid;border-radius:999px;font-family:'IBM Plex Mono',monospace;font-size:.67rem;}
    .delta{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);margin-top:10px;}

    .section-title{display:flex;align-items:center;gap:10px;margin:5px 0 3px;}
    .insight{background:var(--panel);border-left:3px solid var(--gold);padding:15px 17px;border-radius:0 12px 12px 0;color:var(--text);line-height:1.55;}
    .risk{background:linear-gradient(135deg,var(--panel2),var(--panel));border:1px solid var(--border);border-radius:18px;padding:22px;}
    .risk-word{font-family:'IBM Plex Mono',monospace;font-size:.9rem;letter-spacing:.08em;}

    .fun-card{background:linear-gradient(135deg,var(--panel2),var(--panel));border:1px dashed var(--gold);border-radius:18px;padding:22px;height:100%;}
    .fun-big{font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:700;color:var(--gold);line-height:1;}
    .fun-caption{color:var(--muted);font-size:.85rem;margin-top:8px;line-height:1.5;}

    .footer{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.68rem;text-align:center;padding-top:22px;letter-spacing:.05em;}

    div[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:13px;padding:10px 14px;}
    div[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'Space Grotesk',sans-serif;}
    div[data-testid="stMetricLabel"]{color:var(--muted)!important;}

    /* Buttons: Streamlit renders these with its own theme-specific classes/specificity, so force it */
    button, .stButton button, [data-testid="stButton"] button, [data-testid^="stBaseButton"], button[kind="secondary"], button[kind="primary"]{
        background:var(--panel)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:10px!important;font-weight:600!important;
    }
    button:hover, .stButton button:hover, [data-testid^="stBaseButton"]:hover{
        background:var(--panel2)!important;border-color:var(--gold)!important;color:var(--text)!important;
    }
    button p, button span, button div{color:var(--text)!important;}

    /* Component toolbar / fullscreen-expand icon buttons that float over charts and the map */
    [data-testid="StyledFullScreenButton"], [data-testid="stElementToolbarButton"], [data-testid="stElementToolbar"] button,
    div[data-testid="stElementToolbar"]{
        background:var(--panel)!important;color:var(--text)!important;border-radius:8px!important;
    }
    [data-testid="StyledFullScreenButton"] svg, [data-testid="stElementToolbarButton"] svg{fill:var(--text)!important;}
    .stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;}
    .stTabs [data-baseweb="tab"]{background:var(--panel);border-radius:9px;border:1px solid var(--border);padding:8px 14px;}
    .stTabs [data-baseweb="tab"] p{color:var(--text);}
    section[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--border);border-radius:14px;color:var(--text);}
    section[data-testid="stExpander"] summary{color:var(--text);}
    div[data-testid="stAlert"]{background:var(--panel);border:1px solid var(--border);border-radius:12px;color:var(--text);}
    div[data-testid="stVerticalBlockBorderWrapper"]{background:transparent;}
    [data-testid="stRadio"] label p{color:var(--text);}
    """
    st.markdown(f"<style>{root_vars}{static_css}</style>", unsafe_allow_html=True)


def render_map(theme):
    style = "dark" if theme == "dark" else "light"
    point_color = [232, 163, 61, 220] if theme == "dark" else [255, 122, 61, 220]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({"lat": [KARACHI_LAT], "lon": [KARACHI_LON]}),
        get_position="[lon, lat]",
        get_fill_color=point_color,
        get_radius=1200,
        pickable=False,
    )
    view_state = pdk.ViewState(latitude=KARACHI_LAT, longitude=KARACHI_LON, zoom=9.5)
    deck = pdk.Deck(map_style=style, initial_view_state=view_state, layers=[layer], tooltip=False)
    st.pydeck_chart(deck, use_container_width=True, height=225)


@st.cache_data(ttl=60)
def get_forecast():
    response = requests.get(f"{API_URL}/forecast", timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def get_shap_remote(horizon):
    response = requests.get(f"{API_URL}/shap/{horizon}", timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def load_shap(horizon):
    target = SHAP_TARGETS[horizon]
    local_path = os.path.join(SHAP_LOCAL_DIR, f"{target}_feature_importance.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        return df.sort_values("mean_abs_shap", ascending=False)
    remote = get_shap_remote({"Day 1": "1_day", "Day 2": "2_days", "Day 3": "3_days"}[horizon])
    if remote and "features" in remote:
        return pd.DataFrame(remote["features"]).sort_values("mean_abs_shap", ascending=False)
    return None


def aqi_gauge(value, color):
    axis_max = max(300, value + 40)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 1}, "valueformat": ""},  # number hidden, shown via HTML instead
        gauge={
            "axis": {"range": [0, axis_max], "tickcolor": "#8e9aaa", "tickfont": {"color": "#8e9aaa"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(95,217,138,.25)"},
                {"range": [50, 100], "color": "rgba(232,197,71,.25)"},
                {"range": [100, 150], "color": "rgba(240,144,74,.25)"},
                {"range": [150, 200], "color": "rgba(232,91,76,.25)"},
                {"range": [200, 300], "color": "rgba(163,102,217,.25)"},
                {"range": [300, axis_max], "color": "rgba(122,46,58,.25)"},
            ],
        },
    ))
    fig.update_layout(height=140, margin=dict(l=15, r=15, t=15, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8e9aaa"))
    return fig


if "theme" not in st.session_state:
    st.session_state.theme = "dark"

inject_css(st.session_state.theme)

# ---------------- HEADER ----------------
header_l, header_r = st.columns([5, 1.3])
with header_l:
    st.markdown('<div class="eyebrow">Karachi · AI Air Quality Intelligence</div>', unsafe_allow_html=True)
    st.title("🌫️ Pearls AQI Predictor")
with header_r:
    st.write("")
    st.write("")
    btn_l, btn_r = st.columns(2)
    with btn_l:
        if st.button("🔄", use_container_width=True, help="Refresh data"):
            st.cache_data.clear()
            st.rerun()
    with btn_r:
        toggle_label = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(toggle_label, use_container_width=True, help="Toggle light / dark mode"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

try:
    data = get_forecast()
except requests.exceptions.ConnectionError:
    st.error(f"Cannot connect to the AQI API at {API_URL}. Start the Flask API or verify the deployed URL.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to retrieve forecast: {exc}")
    st.stop()

forecast = data.get("forecast", [])
if not forecast:
    st.error("No forecast data was returned by the API.")
    st.stop()

fdf = pd.DataFrame(forecast)
fdf["date"] = pd.to_datetime(fdf["date"])
fdf["AQI"] = pd.to_numeric(fdf["aqi"], errors="coerce")
fdf = fdf.dropna(subset=["AQI"]).sort_values("date").reset_index(drop=True).head(3)

if fdf.empty:
    st.error("Forecast data could not be parsed.")
    st.stop()

# "Current" is the Day-1 prediction; the remaining rows are the upcoming outlook.
current_row = fdf.iloc[0]
current_aqi = float(current_row["AQI"])
current_date = current_row["date"]
current_category, current_color, current_icon = get_band(current_aqi)
personality_name, personality_icon = PERSONALITY.get(current_category, ("The Unknown Day", "❔"))

upcoming = fdf.iloc[1:].reset_index(drop=True)
future_values = upcoming["AQI"].tolist()

peak_aqi = safe_num(data.get("peak_aqi"), float(fdf["AQI"].max()))
lowest_aqi = safe_num(data.get("lowest_aqi"), float(fdf["AQI"].min()))
peak_idx = int(fdf["AQI"].idxmax())
lowest_idx = int(fdf["AQI"].idxmin())
peak_date = fdf.loc[peak_idx, "date"]
lowest_date = fdf.loc[lowest_idx, "date"]

# ---------------- HERO ----------------
hero_l, hero_r = st.columns([1.65, 1])
with hero_l:
    st.markdown(f"""
    <div class="hero">
        <div>
            <div class="hero-label">Today · Day 1 prediction · {current_date.strftime('%A, %d %b')}</div>
            <div class="hero-number" style="color:{current_color};text-shadow:0 0 28px {current_color}55">{current_aqi:.0f}</div>
            <div class="hero-category" style="color:{current_color};border-color:{current_color}66">{current_icon} {current_category}</div>
            <div class="hero-personality">{personality_icon} {personality_name}</div>
        </div>
        <div class="hero-note"><b>Air-quality status</b><br>{ADVICE.get(current_category, 'Status unavailable.')}<br><br><span style="color:var(--muted)">Source: Day-1 model forecast</span></div>
    </div>
    """, unsafe_allow_html=True)

with hero_r:
    st.markdown('<div class="eyebrow">📍 Karachi</div>', unsafe_allow_html=True)
    with st.container(border=True):
        render_map(st.session_state.theme)

st.write("")

# ---------------- LIVE AQI DIAL ----------------
st.markdown('<div class="eyebrow">Live reading</div>', unsafe_allow_html=True)
st.markdown("### 🎯 Live AQI Dial")
dial_l, dial_r = st.columns([1.3, 1])
with dial_l:
    with st.container(border=True):
        st.plotly_chart(aqi_gauge(current_aqi, current_color), use_container_width=True, config={"displayModeBar": False})
with dial_r:
    st.markdown(f'<div class="panel"><div class="mini-label">Dial reading</div><div class="mini-value" style="color:{current_color}">{current_aqi:.0f} · {current_category}</div><div style="margin-top:10px;color:var(--muted)">3-day range: {lowest_aqi:.0f} – {peak_aqi:.0f}</div></div>', unsafe_allow_html=True)

# ---------------- RISK RADAR ----------------
st.write("")
st.markdown('<div class="eyebrow">Risk intelligence</div>', unsafe_allow_html=True)
st.markdown("### 🚨 3-Day Risk Radar")

outlook, arrow, outlook_color = trend_label(current_aqi, future_values if future_values else [current_aqi])
delta_peak = peak_aqi - current_aqi
risk_score = min(100, max(0, peak_aqi / 3))

r1, r2, r3 = st.columns([1.2, 1.2, 1.8])
with r1:
    st.markdown(f'<div class="risk"><div class="mini-label">Outlook</div><div class="mini-value" style="color:{outlook_color}">{arrow} {outlook}</div><div class="delta">Peak change vs today: {delta_peak:+.1f} AQI</div></div>', unsafe_allow_html=True)
with r2:
    st.markdown(f'<div class="risk"><div class="mini-label">Peak day</div><div class="mini-value">Day {peak_idx+1}</div><div class="delta">{peak_date.strftime("%A, %d %b")} · AQI {peak_aqi:.0f}</div></div>', unsafe_allow_html=True)
with r3:
    st.markdown(f'<div class="risk"><div class="mini-label">Risk index · visual indicator</div><div class="mini-value">{risk_score:.0f}/100</div><div style="margin-top:12px;background:var(--border);border-radius:999px;height:8px"><div style="width:{risk_score:.0f}%;background:{get_band(peak_aqi)[1]};height:8px;border-radius:999px"></div></div></div>', unsafe_allow_html=True)

# ---------------- 3-DAY PREDICTION ----------------
st.write("")
st.markdown('<div class="eyebrow">AI forecast</div>', unsafe_allow_html=True)
st.markdown("### 📅 3-Day Prediction")

cards = st.columns(len(fdf))
prev = None
for i, (_, row) in enumerate(fdf.iterrows()):
    aqi = float(row["AQI"])
    band, color, icon = get_band(aqi)
    day_num = i + 1
    day_tag = "TODAY" if i == 0 else f"DAY {day_num}"
    delta_html = "" if prev is None else f'<div class="delta">{aqi - prev:+.1f} vs previous day</div>'
    with cards[i]:
        st.markdown(f'<div class="forecast-card"><div class="forecast-day">{day_tag} · {row["date"].strftime("%a, %d %b")}</div><div class="forecast-aqi" style="color:{color}">{aqi:.0f}</div><span class="badge" style="color:{color};border-color:{color}66">{icon} {band}</span>{delta_html}</div>', unsafe_allow_html=True)
    prev = aqi

# ---------------- TRAJECTORY ----------------
st.write("")
st.markdown("### 📈 AQI Trajectory")
fig = go.Figure()
band_edges = [0, 50, 100, 150, 200, 300, max(350, float(fdf["AQI"].max()) + 20)]
band_colors = ["#5FD98A", "#E8C547", "#F0904A", "#E85B4C", "#A366D9", "#7A2E3A"]
for i in range(len(band_edges) - 1):
    fig.add_hrect(y0=band_edges[i], y1=band_edges[i + 1], fillcolor=band_colors[i], opacity=.13, line_width=0)
fig.add_trace(go.Scatter(
    x=fdf["date"], y=fdf["AQI"], mode="lines+markers+text",
    text=[f"{v:.0f}" for v in fdf["AQI"]], textposition="top center",
    line=dict(width=4, color="#e8a33d"),
    marker=dict(size=12, color="#e8a33d", line=dict(width=2, color="#0b1017")),
    fill="tozeroy", fillcolor="rgba(232,163,61,.08)", name="AQI",
))
fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#8e9aaa", family="IBM Plex Sans"),
                   xaxis=dict(title="Date", gridcolor="#283442"),
                   yaxis=dict(title="AQI", gridcolor="#283442", rangemode="tozero"),
                   hovermode="x unified", margin=dict(l=30, r=30, t=30, b=20), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------- BEST / WORST ----------------
b1, b2 = st.columns(2)
with b1:
    band, color, icon = get_band(lowest_aqi)
    st.markdown(f'<div class="insight">🌤️ <b>Best predicted day:</b> {lowest_date.strftime("%A, %d %B")} · AQI <b>{lowest_aqi:.0f}</b> ({band}).</div>', unsafe_allow_html=True)
with b2:
    band, color, icon = get_band(peak_aqi)
    st.markdown(f'<div class="insight">⚠️ <b>Highest predicted exposure:</b> {peak_date.strftime("%A, %d %B")} · AQI <b>{peak_aqi:.0f}</b> ({band}).</div>', unsafe_allow_html=True)

# ---------------- CREATIVE / "WHAT DOES THIS MEAN" STRIP ----------------
st.write("")
st.markdown('<div class="eyebrow">Put it in perspective</div>', unsafe_allow_html=True)
st.markdown("### 🌬️ What Today's Air Actually Means")

cig = cigarette_equivalent(current_aqi)
who_x = who_multiple(current_aqi)
pm25_est = aqi_to_pm25(current_aqi)

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(f'<div class="fun-card"><div class="mini-label">Cigarette equivalent</div><div class="fun-big">🚬 {cig}</div><div class="fun-caption">Breathing today\'s outdoor air in Karachi is roughly like smoking <b>{cig}</b> cigarette(s), using the ~22 µg/m³ PM2.5 per cigarette rule of thumb.</div></div>', unsafe_allow_html=True)
with f2:
    st.markdown(f'<div class="fun-card"><div class="mini-label">vs. WHO guideline</div><div class="fun-big">🌍 {who_x}×</div><div class="fun-caption">Estimated PM2.5 (~{pm25_est:.0f} µg/m³) is about <b>{who_x}×</b> the WHO 24-hour guideline of 15 µg/m³.</div></div>', unsafe_allow_html=True)
with f3:
    st.markdown(f'<div class="fun-card"><div class="mini-label">Today\'s mood</div><div class="fun-big">{personality_icon}</div><div class="fun-caption"><b>{personality_name}</b> — {ADVICE.get(current_category, "")}</div></div>', unsafe_allow_html=True)

with st.expander("💡 Did you know?"):
    fact = FACTS[datetime.now().timetuple().tm_yday % len(FACTS)]
    st.write(fact)

# ---------------- SHAP ----------------
st.write("")
st.markdown('<div class="eyebrow">Explainable AI</div>', unsafe_allow_html=True)
st.markdown("### 🧠 Why is the AI predicting this?")
st.caption("SHAP ranks features by their average absolute contribution to each model's prediction. Higher values mean greater influence, not necessarily a positive or negative direction.")

shap_day = st.radio("Prediction horizon", ["Day 1", "Day 2", "Day 3"], horizontal=True)
shap_df = load_shap(shap_day)

if shap_df is not None and not shap_df.empty:
    shap_top = shap_df.head(15).copy().sort_values("mean_abs_shap")
    shap_fig = go.Figure(go.Bar(x=shap_top["mean_abs_shap"], y=shap_top["feature"], orientation="h", text=shap_top["mean_abs_shap"].round(3), textposition="outside", marker=dict(color="#e8a33d")))
    shap_fig.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8e9aaa"), xaxis_title="Mean |SHAP value|", yaxis_title="Feature", margin=dict(l=20, r=70, t=20, b=20))
    st.plotly_chart(shap_fig, use_container_width=True)
    strongest = shap_df.iloc[0]
    st.markdown(f'<div class="insight">🔎 <b>Strongest influence for {shap_day}:</b> <code>{strongest["feature"]}</code> with mean |SHAP| of <b>{strongest["mean_abs_shap"]:.4f}</b>. This tells you which feature the model relied on most strongly overall.</div>', unsafe_allow_html=True)
else:
    st.info("SHAP feature-importance data is not available yet. Run the SHAP analysis script first, then place the generated CSV files in results/shap/ or deploy them alongside the API.")

# ---------------- HEALTH ACTION PLAN ----------------
st.write("")
st.markdown('<div class="eyebrow">Decision support</div>', unsafe_allow_html=True)
st.markdown("### 🛡️ Air-Quality Action Plan")

h1, h2 = st.columns([1, 1.5])
with h1:
    st.markdown(f'<div class="panel"><div class="mini-label">Today (Day 1)</div><div class="mini-value" style="color:{current_color}">{current_icon} {current_category}</div><div style="margin-top:12px;color:var(--muted)">AQI {current_aqi:.0f}</div></div>', unsafe_allow_html=True)
with h2:
    tips = ACTION_TIPS.get(current_category, [])
    html = "".join(f"<div style='margin:7px 0'>✓ {tip}</div>" for tip in tips)
    st.markdown(f'<div class="panel"><div class="mini-label">Recommended actions</div><div style="margin-top:10px;color:var(--text)">{html}</div></div>', unsafe_allow_html=True)

st.info(f"{current_icon} {ADVICE.get(current_category, 'Monitor the latest AQI and follow official guidance.')}")

st.markdown(f'<div class="footer">PEARLS AQI PREDICTOR · KARACHI · Today\'s reading is the Day-1 model forecast · AI forecasts are model estimates, not official air-quality or health guidance</div>', unsafe_allow_html=True)