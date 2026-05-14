"""
app.py
=======
AgriAssist+ — Tamil Nadu Crop Intelligence System
Streamlit web app

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import date
from dotenv import load_dotenv
from PIL import Image

from weather import get_weather, weather_code_label
from rag_engine import RAGEngine
from database import get_db
from plant_disease_cnn import PlantDiseaseDetector, MangoVarietyDetector

# ── Initialise DB on startup ────────────────────────────────
@st.cache_resource
def init_database():
    db = get_db()
    if db.is_connected:
        db.init_tables()
    return db

db = init_database()

load_dotenv()  # ensure .env is loaded after cache_resource init

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriAssist+ Tamil Nadu",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.metric-card .label { font-size: 0.82em; color: rgba(255,255,255,0.7); margin-bottom: 4px; }
.metric-card .value { font-size: 1.5em; font-weight: 700; color: #a5d6a7; }
.metric-card .delta { font-size: 0.88em; margin-top: 4px; color: rgba(255,255,255,0.85); }
.gain  { color: #69f0ae; font-weight: 600; }
.loss  { color: #ff5252; font-weight: 600; }
.stable{ color: #ffd740; font-weight: 600; }
.banner {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
    line-height: 1.8;
    color: #ffffff !important;
    font-size: 0.97em;
}
.banner b, .banner strong { color: #ffffff !important; }
.banner-green  { background: rgba(46,125,50,0.55);  border-left: 5px solid #69f0ae; }
.banner-red    { background: rgba(183,28,28,0.55);  border-left: 5px solid #ff5252; }
.banner-yellow { background: rgba(230,162,0,0.50);  border-left: 5px solid #ffd740; }
.banner-blue   { background: rgba(13,71,161,0.50);  border-left: 5px solid #82b1ff; }
.banner-orange { background: rgba(180,80,0,0.50);   border-left: 5px solid #ffab40; }
.banner-info   { background: rgba(1,87,155,0.45);   border-left: 5px solid #40c4ff; }
.banner-mango  { background: rgba(180,100,0,0.50);  border-left: 5px solid #ffcc02; }
.section-header {
    font-size: 1.2em;
    font-weight: 700;
    color: #a5d6a7;
    border-bottom: 2px solid rgba(165,214,167,0.4);
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}
.weather-card {
    background: rgba(13,71,161,0.35);
    border: 1px solid rgba(130,177,255,0.3);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: #e3f2fd;
}
.ai-response {
    background: rgba(27,94,32,0.40);
    border-left: 5px solid #69f0ae;
    border-radius: 10px;
    padding: 18px 22px;
    color: #e8f5e9 !important;
    font-size: 0.97em;
    line-height: 1.8;
    white-space: pre-wrap;
}
.ai-response b, .ai-response strong { color: #b9f6ca !important; }
.solution-panel {
    background: rgba(13,71,161,0.40);
    border-left: 5px solid #82b1ff;
    border-radius: 10px;
    padding: 18px 22px;
    color: #e3f2fd !important;
    font-size: 0.97em;
    line-height: 1.85;
}
.solution-panel b, .solution-panel strong { color: #b3d4ff !important; }
.reco-panel {
    background: rgba(27,94,32,0.45);
    border-left: 5px solid #69f0ae;
    border-radius: 10px;
    padding: 18px 22px;
    color: #e8f5e9 !important;
    font-size: 0.97em;
    line-height: 1.85;
}
.reco-panel b, .reco-panel strong { color: #b9f6ca !important; }
.yield-box {
    border-radius: 10px;
    padding: 14px 20px;
    margin: 8px 0;
    color: #ffffff !important;
    line-height: 1.7;
}
.yield-box b { color: #ffffff !important; }
.cnn-result-card {
    background: rgba(13,71,161,0.35);
    border: 2px solid rgba(130,177,255,0.5);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #e3f2fd;
    line-height: 1.8;
}
.cnn-healthy {
    background: rgba(27,94,32,0.45);
    border: 2px solid rgba(105,240,174,0.5);
}
.cnn-disease {
    background: rgba(183,28,28,0.40);
    border: 2px solid rgba(255,82,82,0.5);
}
.cnn-warning {
    background: rgba(180,80,0,0.40);
    border: 2px solid rgba(255,171,64,0.5);
}
.mango-card {
    background: rgba(120,60,0,0.40);
    border: 2px solid rgba(255,204,2,0.6);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #fff8e1;
    line-height: 1.8;
}
.mango-healthy {
    background: rgba(27,94,32,0.45);
    border: 2px solid rgba(255,204,2,0.5);
}
.mango-diseased {
    background: rgba(183,28,28,0.40);
    border: 2px solid rgba(255,82,82,0.5);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SEASON_MAP = {
    1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
    6:"Monsoon",7:"Monsoon",8:"Monsoon",
    9:"Post-Monsoon",10:"Post-Monsoon",11:"Post-Monsoon",12:"Winter"
}
SEASON_CODES = {"Winter":0,"Summer":1,"Monsoon":2,"Post-Monsoon":3}

TN_DISTRICTS = [
    "Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli",
    "Erode","Vellore","Thoothukudi","Dharmapuri","Dindigul","Cuddalore",
    "Thanjavur","Tiruvarur","Nagapattinam","Pudukkottai","Ramanathapuram",
    "Kanyakumari","Krishnagiri","Namakkal","Karur","Perambalur","Ariyalur",
    "Villupuram","Tiruvannamalai","Ranipet","Tirupathur","Chengalpattu",
    "Kallakurichi","Tenkasi","Mayiladuthurai"
]

# ─────────────────────────────────────────────
# Load models & data
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    files = {
        "current":    "current_price_model.joblib",
        "1week":      "future_1w_model.joblib",
        "2week":      "future_2w_model.joblib",
        "1month":     "future_1m_model.joblib",
        "yield_loss": "yield_loss_model.joblib",
    }
    models = {}
    for key, f in files.items():
        if os.path.exists(f):
            models[key] = joblib.load(f)
    return models

@st.cache_resource
def load_meta():
    if os.path.exists("model_meta.joblib"):
        return joblib.load("model_meta.joblib")
    return None

@st.cache_data
def load_dataset():
    for f in ["tn_agri_dataset.csv"]:
        if os.path.exists(f):
            return pd.read_csv(f)
    return None

@st.cache_resource
def load_rag():
    return RAGEngine()

@st.cache_resource
def load_cnn_detector():
    return PlantDiseaseDetector()

@st.cache_resource
def load_mango_detector():
    return MangoVarietyDetector()

models        = load_models()
meta          = load_meta()
df            = load_dataset()
rag           = load_rag()
detector      = load_cnn_detector()
mango_detector = load_mango_detector()

# ─────────────────────────────────────────────
# Startup check
# ─────────────────────────────────────────────
if not models or "current" not in models:
    st.error("⚠️ Models not found. Please run the setup commands below:")
    st.code("python generate_dataset.py\npython train_models.py")
    st.stop()

if df is None:
    st.error("⚠️ Dataset not found. Run: python generate_dataset.py")
    st.stop()

# ─────────────────────────────────────────────
# Build crop/variety dropdowns from dataset
# ─────────────────────────────────────────────
all_crops = sorted(df["Commodity"].dropna().unique().tolist())
encoders  = meta["encoders"] if meta else {}

def encode(col, val):
    d = encoders.get(col, {})
    return d.get(val, d.get(list(d.keys())[0], 0) if d else 0)

# ─────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────
def get_price_stats(crop, district):
    sub = df[(df["Commodity"] == crop) & (df["District"] == district)]
    if sub.empty:
        sub = df[df["Commodity"] == crop]
    if sub.empty:
        return {"modal": 3000.0, "min": 2500.0, "max": 3500.0, "lag1": 3000.0,
                "lag2": 3000.0, "lag4": 3000.0, "lag8": 3000.0, "lag12": 3000.0}
    latest = sub.sort_values("Arrival_Date").iloc[-1]
    return {
        "modal":  float(latest.get("Modal_Price", 3000)),
        "min":    float(latest.get("Min_Price", 2500)),
        "max":    float(latest.get("Max_Price", 3500)),
        "lag1":   float(latest.get("Lag1", 3000)),
        "lag2":   float(latest.get("Lag2", 3000)),
        "lag4":   float(latest.get("Lag4", 3000)),
        "lag8":   float(latest.get("Lag8", 3000)),
        "lag12":  float(latest.get("Lag12", 3000)),
    }

def build_price_row(crop, district, market, variety, sel_date, weather_data):
    dt       = pd.to_datetime(sel_date)
    month    = dt.month
    week     = int(dt.isocalendar()[1])
    season   = SEASON_MAP.get(month, "Monsoon")
    stats    = get_price_stats(crop, district)
    min_p    = stats["min"]
    max_p    = stats["max"]
    spread   = round(max_p - min_p, 2)
    mid      = round((max_p + min_p) / 2, 2)
    spread_r = round(spread / mid if mid > 0 else 0, 4)
    w = weather_data.get("avg_7day", {})
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code":  encode("District", district),
        "Market_Code":    encode("Market", market),
        "Variety_Code":   encode("Variety", variety),
        "Month":          month,
        "Week":           week,
        "Season_Code":    SEASON_CODES.get(season, 0),
        "Min_Price":      min_p,
        "Max_Price":      max_p,
        "Price_Spread":   spread,
        "Price_Mid":      mid,
        "Spread_Ratio":   spread_r,
        "Lag1":           stats["lag1"],
        "Lag2":           stats["lag2"],
        "Lag4":           stats["lag4"],
        "Lag8":           stats["lag8"],
        "Lag12":          stats["lag12"],
        "Rainfall_mm":    w.get("rainfall", 30),
        "Temp_Max_C":     w.get("temp_max", 32),
        "Temp_Min_C":     w.get("temp_min", 24),
        "Humidity_Pct":   w.get("humidity", 60),
    }
    pf = meta["price_features"] if meta else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in pf}])

def build_yield_row(crop, district, sel_date, weather_data):
    dt    = pd.to_datetime(sel_date)
    month = dt.month
    season= SEASON_MAP.get(month, "Monsoon")
    w     = weather_data.get("avg_7day", {})
    stats = get_price_stats(crop, district)
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code":  encode("District", district),
        "Month":          month,
        "Season_Code":    SEASON_CODES.get(season, 0),
        "Rainfall_mm":    w.get("rainfall", 30),
        "Temp_Max_C":     w.get("temp_max", 32),
        "Temp_Min_C":     w.get("temp_min", 24),
        "Humidity_Pct":   w.get("humidity", 60),
        "Lag1":           stats["lag1"],
        "Lag4":           stats["lag4"],
        "Lag8":           stats["lag8"],
    }
    yf = meta["yield_features"] if meta else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in yf}])

def trend_icon(base, compare):
    if not compare: return "➡️"
    d = (compare - base) / base * 100 if base > 0 else 0
    if d > 2:  return f"🔺 +{d:.1f}%"
    if d < -2: return f"🔻 {d:.1f}%"
    return f"➡️ {d:+.1f}%"

def qtl_to_kg(price_qtl):
    return round(price_qtl / 100, 2) if price_qtl else 0

def fmt_price(price_qtl):
    kg = qtl_to_kg(price_qtl)
    return f"Rs.{kg:.2f}/kg", f"Rs.{price_qtl:,.0f}/Qtl"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriAssist+")
    st.caption("Tamil Nadu Crop Intelligence System")
    st.markdown("---")

    selected_district = st.selectbox("📍 District", TN_DISTRICTS, index=1)
    selected_crop     = st.selectbox("🌱 Crop", all_crops)

    varieties = sorted(df[df["Commodity"] == selected_crop]["Variety"].dropna().unique().tolist())
    selected_variety = st.selectbox("🔬 Variety", varieties if varieties else ["Local"])

    markets = sorted(df[
        (df["Commodity"] == selected_crop) & (df["District"] == selected_district)
    ]["Market"].dropna().unique().tolist())
    selected_market = st.selectbox("🏪 Market", markets if markets else [f"{selected_district} APMC"])

    selected_date = st.date_input("📅 Date", value=date.today())
    yield_qty     = st.number_input("🌾 Yield Quantity (Qtl)", min_value=1.0, max_value=10000.0, value=10.0, step=1.0)

    predict_btn = st.button("🔮 Predict & Analyze", use_container_width=True, type="primary")
    st.markdown("---")

    # CNN status in sidebar
    st.markdown("**🧠 AI Detection Models**")
    if detector.cnn_available:
        st.success("🍅 Tomato CNN: Ready")
    else:
        st.info("🍅 Tomato: Claude Vision mode")
    if mango_detector.cnn_available:
        st.success("🥭 Mango CNN: Ready")
    else:
        st.info("🥭 Mango: Claude Vision mode")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌾 AgriAssist+ — Tamil Nadu Crop Intelligence")
st.markdown("*Predict prices · Assess yield risk · Get AI-powered advice · Identify diseases from photos*")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Price Prediction",
    "🌦️ Weather & Yield Risk",
    "🤖 AI Advisor",
    "🧪 Chemical Advisor",
    "📈 Market Trends",
    "📷 Disease Detection (CNN)",
])

# ══════════════════════════════════════════════
# TAB 1 — PRICE PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if predict_btn:
        with st.spinner("Fetching weather & running predictions..."):
            weather_data = get_weather(selected_district)
            st.session_state["weather_data"] = weather_data
            X_price = build_price_row(selected_crop, selected_district, selected_market,
                                       selected_variety, selected_date, weather_data)
            X_yield = build_yield_row(selected_crop, selected_district, selected_date, weather_data)
            cur  = float(models["current"].predict(X_price)[0])
            p1w  = float(models["1week"].predict(X_price)[0])
            p2w  = float(models["2week"].predict(X_price)[0]) if "2week" in models else None
            p1m  = float(models["1month"].predict(X_price)[0]) if "1month" in models else None
            yl   = float(models["yield_loss"].predict(X_yield)[0]) if "yield_loss" in models else 0.0
            st.session_state.update({
                "cur": cur, "p1w": p1w, "p2w": p2w, "p1m": p1m,
                "yield_loss": yl, "crop": selected_crop,
                "district": selected_district, "predicted": True,
            })
            if db.is_connected:
                session_id = st.session_state.get("session_id", "streamlit")
                db.upsert_session(session_id, selected_crop, selected_district)
                db.save_prediction(session_id, {
                    "crop":            selected_crop,
                    "variety":         selected_variety,
                    "district":        selected_district,
                    "market":          selected_market,
                    "prediction_date": str(selected_date),
                    "current_price":   round(cur, 2),
                    "price_1week":     round(p1w, 2),
                    "price_2week":     round(p2w, 2) if p2w else None,
                    "price_1month":    round(p1m, 2) if p1m else None,
                    "yield_loss_pct":  round(yl, 2),
                    "yield_qty_qtl":   yield_qty,
                })
        st.success(f"✅ Prediction complete for **{selected_crop}** in **{selected_district}**")
        st.markdown("---")

    if st.session_state.get("predicted"):
        cur = st.session_state["cur"]
        p1w = st.session_state["p1w"]
        p2w = st.session_state.get("p2w")
        p1m = st.session_state.get("p1m")
        yl  = st.session_state.get("yield_loss", 0)

        cols = st.columns(4)
        for col, (label, price, base) in zip(cols, [
            ("📌 Current Price", cur, None),
            ("📅 1 Week",  p1w,  cur),
            ("📅 2 Weeks", p2w,  cur),
            ("📅 1 Month", p1m,  cur),
        ]):
            if price is None: continue
            kg_price, qtl_price = fmt_price(price)
            delta_qtl = f"{price-cur:+,.0f}" if base else ""
            delta_kg  = f"{qtl_to_kg(price-cur):+.2f}" if base else ""
            icon      = trend_icon(cur, price) if base else ""
            card_cls  = "banner-green" if (price >= cur if base else True) else "banner-red"
            col.markdown(f"""
<div class="metric-card {card_cls}">
  <div class="label">{label}</div>
  <div class="value">{kg_price}</div>
  <div class="delta" style="font-size:0.78em;opacity:0.8;">{qtl_price}</div>
  <div class="delta">{icon} Rs.{delta_kg}/kg</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Price Forecast Chart</div>', unsafe_allow_html=True)
        periods = ["Current"]
        prices_list = [cur]
        if p1w: periods.append("1 Week"); prices_list.append(p1w)
        if p2w: periods.append("2 Weeks"); prices_list.append(p2w)
        if p1m: periods.append("1 Month"); prices_list.append(p1m)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods, y=[qtl_to_kg(p) for p in prices_list],
            mode="lines+markers+text",
            text=[f"Rs.{qtl_to_kg(p):.2f}/kg" for p in prices_list],
            textposition="top center",
            line=dict(color="#2e7d32", width=3),
            marker=dict(size=10, color="#1b5e20"),
            name="Price",
            hoverinfo="skip",
        ))
        fig.add_hline(y=qtl_to_kg(cur), line_dash="dot", line_color="#aaaaaa", annotation_text="Current baseline")
        fig.update_layout(
            title=f"{selected_crop} Price Forecast — {selected_district}",
            xaxis_title="Period", yaxis_title="Price (Rs./kg)",
            height=350, margin=dict(t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.04)",
            font=dict(color="#e0e0e0"),
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True, "scrollZoom": False})

        st.markdown('<div class="section-header">🌿 Estimated Yield Risk & Sell Decision</div>', unsafe_allow_html=True)
        yl_cls   = "banner-red" if yl > 20 else "banner-yellow" if yl > 10 else "banner-green"
        yl_icon  = "🔴" if yl > 30 else "🟠" if yl > 15 else "🟢"
        yl_label = "High Risk" if yl > 30 else "Moderate Risk" if yl > 15 else "Low Risk"
        est_loss = round(yl * yield_qty / 100, 2)
        est_val  = round(est_loss * cur, 2)
        cur_kg   = qtl_to_kg(cur)

        price_trend_1m = ((p1m - cur) / cur * 100) if p1m and cur else 0
        price_trend_1w = ((p1w - cur) / cur * 100) if p1w and cur else 0

        if yl > 25 and price_trend_1m <= 2:
            sell_icon = "🚨"; sell_decision = "SELL NOW"; sell_cls = "banner-red"
            sell_reason = f"High yield risk ({yl:.0f}%) + prices not rising significantly."
        elif yl > 25 and price_trend_1m > 2:
            sell_icon = "⚠️"; sell_decision = "SELL WITHIN 1 WEEK"; sell_cls = "banner-orange"
            sell_reason = f"High yield risk ({yl:.0f}%) — sell soon."
        elif price_trend_1m > 8 and yl <= 15:
            sell_icon = "⏳"; sell_decision = "WAIT — HOLD FOR 1 MONTH"; sell_cls = "banner-green"
            sell_reason = f"Prices forecast to rise {price_trend_1m:.1f}% (Rs.{qtl_to_kg(p1m):.2f}/kg)."
        elif price_trend_1w > 3 and yl <= 20:
            sell_icon = "📅"; sell_decision = "WAIT — SELL IN 1 WEEK"; sell_cls = "banner-blue"
            sell_reason = f"Prices rising {price_trend_1w:.1f}% this week."
        else:
            sell_icon = "✅"; sell_decision = "SELL NOW — PRICES ARE STABLE"; sell_cls = "banner-yellow"
            sell_reason = "Prices stable with no major upside forecast."

        st.markdown(f"""
<div class="banner yield-box {yl_cls}">
{yl_icon} <b>Yield Loss Estimate: {yl:.1f}% — {yl_label}</b><br>
For your {yield_qty:.0f} Qtl crop: Estimated loss = <b>{est_loss:.1f} Qtl</b>
(Rs. <b>{est_val:,.0f}</b> at current price of Rs.{cur_kg:.2f}/kg)
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="banner yield-box {sell_cls}">
{sell_icon} <b>Sell Decision: {sell_decision}</b><br>
{sell_reason}<br><br>
📊 <b>Price outlook:</b> Current Rs.{cur_kg:.2f}/kg → 1 Week Rs.{qtl_to_kg(p1w):.2f}/kg ({price_trend_1w:+.1f}%) → 1 Month Rs.{qtl_to_kg(p1m):.2f}/kg ({price_trend_1m:+.1f}%)
</div>""", unsafe_allow_html=True)

    else:
        st.info("👈 Select inputs in the sidebar and click **Predict & Analyze** to start.")


# ══════════════════════════════════════════════
# TAB 2 — WEATHER & YIELD RISK
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🌦️ Live Weather — Tamil Nadu Districts</div>', unsafe_allow_html=True)

    weather_district = st.selectbox("Select district for weather", TN_DISTRICTS,
                                     index=TN_DISTRICTS.index(selected_district),
                                     key="weather_district_select")

    if st.button("🔄 Fetch Weather", key="fetch_weather_btn"):
        with st.spinner(f"Fetching weather for {weather_district}..."):
            wd = get_weather(weather_district)
            st.session_state["weather_data"] = wd
    else:
        wd = st.session_state.get("weather_data")

    if wd and wd.get("success"):
        cur_w = wd.get("current", {})
        avg   = wd.get("avg_7day", {})
        fcast = wd.get("forecast", [])

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("🌡️ Temperature", f"{cur_w.get('temperature', 'N/A')} °C")
        w2.metric("💧 Humidity",    f"{cur_w.get('humidity', 'N/A')} %")
        w3.metric("🌧️ Rainfall",    f"{cur_w.get('precipitation', 0)} mm")
        w4.metric("💨 Wind Speed",  f"{cur_w.get('wind_speed', 'N/A')} km/h")
        st.markdown(f"**Condition:** {weather_code_label(cur_w.get('weather_code'))}")

        st.markdown('<div class="section-header">📅 7-Day Forecast</div>', unsafe_allow_html=True)
        if fcast:
            fdf = pd.DataFrame(fcast)
            fdf.columns = ["Date","Max Temp (°C)","Min Temp (°C)","Rainfall (mm)","Humidity (%)"]
            st.dataframe(fdf.set_index("Date"), use_container_width=True)

        st.markdown('<div class="section-header">⚠️ Crop Stress Alerts</div>', unsafe_allow_html=True)
        tmax = avg.get("temp_max", 30)
        rain = avg.get("rainfall", 20)
        hum  = avg.get("humidity", 60)

        alerts = []
        if tmax > 38: alerts.append(("🌡️ Heat Stress", f"Max temp {tmax}°C — risk for tomato, brinjal, chilli."))
        if tmax < 15: alerts.append(("❄️ Cold Stress", f"Min temp below 15°C — risk for banana, papaya, coconut."))
        if rain > 150: alerts.append(("🌊 Flood Risk", f"Heavy rainfall {rain:.0f}mm — check drainage immediately."))
        if rain < 5:  alerts.append(("🏜️ Drought Risk", f"Very low rainfall {rain:.0f}mm — irrigation needed."))
        if hum > 80:  alerts.append(("🦠 Fungal Disease Risk", f"Humidity {hum:.0f}% — spray preventive fungicide."))
        if not alerts: alerts.append(("✅ Favourable Conditions", "Weather conditions are normal. No immediate alerts."))

        alert_cls_map = {
            "✅ Favourable Conditions": "banner-green",
            "🌡️ Heat Stress": "banner-red",
            "❄️ Cold Stress": "banner-blue",
            "🌊 Flood Risk": "banner-blue",
            "🏜️ Drought Risk": "banner-yellow",
            "🦠 Fungal Disease Risk": "banner-orange",
        }
        for title, msg in alerts:
            acls = alert_cls_map.get(title, "banner-info")
            st.markdown(f'<div class="banner {acls}"><b>{title}</b><br>{msg}</div>', unsafe_allow_html=True)

    elif wd and not wd.get("success"):
        st.warning(f"Weather fetch failed: {wd.get('error')}. Check internet connection.")
    else:
        st.info("Click **Fetch Weather** or use **Predict & Analyze** in the sidebar to load weather data.")


# ══════════════════════════════════════════════
# TAB 3 — AI ADVISOR
# ══════════════════════════════════════════════
with tab3:
    if not st.session_state.get("predicted", False):
        st.warning("💡 For best results: first run **Predict & Analyze** from the sidebar, then ask here.")

    def build_context():
        return {
            "crop":          st.session_state.get("crop", selected_crop),
            "district":      st.session_state.get("district", selected_district),
            "prices": {
                "current": st.session_state.get("cur"),
                "1week":   st.session_state.get("p1w"),
                "1month":  st.session_state.get("p1m"),
            },
            "weather":       st.session_state.get("weather_data", {}).get("avg_7day", {}),
            "yield_loss_pct": st.session_state.get("yield_loss", 0),
        }

    st.markdown('<div class="section-header">🔍 Describe Your Problem</div>', unsafe_allow_html=True)
    st.caption("Type your crop issue and click **Analyze** — you'll get a solution, recommendations, and all 6 topic answers in one go.")

    problem_input = st.text_area(
        "What problem are you facing with your crop?",
        placeholder="e.g. My tomato leaves have white powder and are turning yellow. Some stems are rotting at the base.",
        height=110,
        key="shared_problem_input",
    )

    diagnose_btn = st.button("🔬 Analyze My Problem", type="primary", key="diagnose_btn", use_container_width=True)

    if diagnose_btn:
        if not problem_input.strip():
            st.warning("⚠️ Please describe your crop problem first.")
        else:
            ctx = build_context()
            with st.spinner("🤖 Diagnosing your problem..."):
                diag = rag.diagnose_problem(problem_input.strip(), ctx)
            st.session_state["diagnosis_result"]  = diag
            st.session_state["diagnosis_problem"] = problem_input.strip()

            TOPICS = [
                ("sell_wait",    "Should I sell now or wait?"),
                ("diseases",     "What diseases should I watch for and how to treat them?"),
                ("profit",       "How can I improve my profit?"),
                ("weather_q",    "Is the weather good for my crop right now?"),
                ("harvest",      "When is the best time to harvest?"),
                ("yield_loss_q", "How can I reduce yield loss?"),
            ]
            bar = st.progress(0, text="Generating topic answers…")
            for i, (key, base_q) in enumerate(TOPICS):
                full_q = f"{base_q} — Farmer's problem: {problem_input.strip()}"
                ans = rag.recommend(full_q, ctx)
                st.session_state[f"qq_ans_{key}"]     = ans
                st.session_state[f"qq_problem_{key}"] = problem_input.strip()
                bar.progress((i + 1) / len(TOPICS), text=f"Topic {i+1}/6 — {base_q[:45]}…")
            bar.empty()

    if st.session_state.get("diagnosis_result"):
        diag    = st.session_state["diagnosis_result"]
        problem = st.session_state.get("diagnosis_problem", "")

        st.markdown("---")
        st.markdown(f'<div class="section-header">🧠 AI Analysis — {st.session_state.get("crop", selected_crop)}</div>', unsafe_allow_html=True)
        if problem:
            st.caption(f"📝 Problem: {problem[:160]}{'...' if len(problem) > 160 else ''}")

        col_sol, col_rec = st.columns(2, gap="medium")
        with col_sol:
            st.markdown("#### 💡 Solution")
            st.markdown(f'<div class="solution-panel">{diag["solution"]}</div>', unsafe_allow_html=True)
        with col_rec:
            st.markdown("#### 📋 Recommendations")
            st.markdown(f'<div class="reco-panel">{diag["recommendations"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📚 Deep-Dive Topic Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.get("diagnosis_result"):
        st.caption("Describe your problem above and click **🔬 Analyze My Problem** — all 6 answers load automatically.")

    TOPIC_LABELS = [
        ("💰 Should I sell now or wait?",       "sell_wait"),
        ("🦠 What diseases should I watch for?", "diseases"),
        ("📈 How can I improve my profit?",      "profit"),
        ("🌤️ Is the weather good for my crop?",  "weather_q"),
        ("🌾 When is the best time to harvest?", "harvest"),
        ("⚠️ How to reduce yield loss?",         "yield_loss_q"),
    ]

    for label, key in TOPIC_LABELS:
        cached_ans     = st.session_state.get(f"qq_ans_{key}")
        cached_problem = st.session_state.get(f"qq_problem_{key}", "")
        exp_label = f"{label}  ✅" if cached_ans else label
        with st.expander(exp_label, expanded=False):
            if cached_ans:
                if cached_problem:
                    st.caption("🔍 Based on: " + cached_problem[:120] + ("…" if len(cached_problem) > 120 else ""))
                st.markdown(f'<div class="ai-response">{cached_ans}</div>', unsafe_allow_html=True)
                with st.expander("📚 Knowledge sources", expanded=False):
                    base_q = label.split(" ", 1)[1]
                    docs = rag.retrieve(f"{base_q}. {cached_problem}".strip(),
                                        crop=st.session_state.get("crop", selected_crop), top_k=3)
                    for d in docs:
                        st.markdown(f"**[{d['crop']} — {d['topic'].replace('_',' ').title()}]**")
                        st.caption(d["text"][:200] + "...")
                        st.markdown("---")
            else:
                st.info("👆 Describe your problem above and click **🔬 Analyze My Problem** to populate this.")


# ══════════════════════════════════════════════
# TAB 4 — CHEMICAL ADVISOR
# ══════════════════════════════════════════════
with tab4:
    from rag_engine import recommend_chemicals

    st.markdown('<div class="section-header">🧪 Weather-Aware Chemical Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        "Based on **current weather conditions**, this advisor tells you which chemicals "
        "to **use**, which to **avoid**, the **exact dosage**, and the **expected yield improvement**."
    )

    wd_chem   = st.session_state.get("weather_data")
    chem_crop = st.session_state.get("crop", selected_crop)
    chem_yl   = st.session_state.get("yield_loss", 0)

    col_cc1, col_cc2 = st.columns([2, 1])
    with col_cc1:
        chem_crop_sel = st.selectbox("Select Crop", all_crops,
                                      index=all_crops.index(chem_crop) if chem_crop in all_crops else 0,
                                      key="chem_crop_sel")
    with col_cc2:
        chem_dist_sel = st.selectbox("District (for live weather)", TN_DISTRICTS,
                                      index=TN_DISTRICTS.index(selected_district),
                                      key="chem_dist_sel")

    chem_disease_input = st.text_area(
        "🦠 Describe the disease or problem with your crop",
        placeholder="e.g. Leaves are turning yellow with brown spots. White powder on stems.",
        height=90,
        key="chem_disease_input",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Load Weather & Analyze Chemicals", key="chem_weather_btn", use_container_width=True):
            with st.spinner(f"Fetching weather for {chem_dist_sel}..."):
                wd_chem = get_weather(chem_dist_sel)
                st.session_state["weather_data"] = wd_chem
    with col_btn2:
        ai_chem_btn = st.button("🤖 Get AI Chemical Advice", key="ai_chem_btn",
                                 use_container_width=True, type="primary",
                                 disabled=(not (st.session_state.get("weather_data") and
                                               st.session_state.get("weather_data", {}).get("success"))))

    if wd_chem and wd_chem.get("success"):
        avg      = wd_chem.get("avg_7day", {})
        chem_res = recommend_chemicals(chem_crop_sel, avg, chem_yl)

        if ai_chem_btn:
            if not chem_disease_input.strip():
                st.warning("⚠️ Please describe your crop disease or problem above.")
            else:
                with st.spinner("🤖 Generating chemical treatment plan..."):
                    ai_chem_result = rag.llm_chemical_advice(
                        crop=chem_crop_sel,
                        disease_input=chem_disease_input.strip(),
                        weather=avg,
                        chem_res=chem_res,
                        yield_loss=chem_yl,
                        district=chem_dist_sel,
                    )
                st.session_state["ai_chem_result"] = ai_chem_result
                st.session_state["ai_chem_problem"] = chem_disease_input.strip()

        if st.session_state.get("ai_chem_result"):
            st.markdown('<div class="section-header">🤖 AI Chemical Treatment Plan</div>', unsafe_allow_html=True)
            st.caption(f"📝 Problem: {st.session_state.get('ai_chem_problem', '')[:160]}")
            st.markdown(f'<div class="ai-response">{st.session_state["ai_chem_result"]}</div>', unsafe_allow_html=True)
            st.markdown("---")

        st.markdown('<div class="section-header">🌦️ Current Weather Snapshot</div>', unsafe_allow_html=True)
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("🌡️ Temperature",   f"{avg.get('temp_max','N/A')} °C")
        wc2.metric("💧 Humidity",      f"{avg.get('humidity','N/A')} %")
        wc3.metric("🌧️ Rainfall (7d)", f"{avg.get('rainfall','N/A')} mm")
        wc4.metric("📍 District",       chem_dist_sel)

        for note in chem_res["weather_notes"]:
            note_cls = "banner-red" if any(x in note for x in ["Heavy","38°C","Avoid"]) \
                  else "banner-yellow" if any(x in note for x in ["High humidity","Dry","Wind"]) \
                  else "banner-green"
            st.markdown(f'<div class="banner {note_cls}">{note}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">✅ Recommended Chemicals for Current Weather</div>', unsafe_allow_html=True)
        if chem_res["recommended"]:
            for chem in chem_res["recommended"]:
                yield_color = "#69f0ae" if chem["yield_gain_pct"] >= 20 \
                         else "#ffd740" if chem["yield_gain_pct"] >= 10 else "#90caf9"
                st.markdown(f"""
<div class="banner banner-green" style="margin-bottom:14px;">
  <div style="font-size:1.05em;font-weight:700;margin-bottom:8px;">✅ {chem['name']} <span style="font-size:0.8em;opacity:0.85;">({chem['type']})</span></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.92em;">
    <div>🎯 <b>Targets:</b> {', '.join(chem['targets'])}</div>
    <div>⏰ <b>Timing:</b> {chem['timing']}</div>
    <div>🧪 <b>Dose/Litre:</b> <span style="color:#b9f6ca;font-weight:700;">{chem['dose_per_litre']}</span></div>
    <div>🌾 <b>Dose/Acre:</b> <span style="color:#b9f6ca;font-weight:700;">{chem['dose_per_acre']}</span></div>
    <div>🔁 <b>Frequency:</b> {chem['frequency']}</div>
    <div style="color:{yield_color};font-weight:700;">📈 Yield Gain: +{chem['yield_gain_pct']}%</div>
  </div>
  <div style="margin-top:8px;font-size:0.85em;opacity:0.85;">⚠️ {chem['safety']}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner banner-yellow">⚠️ No sprays recommended right now. Weather conditions are unfavourable.</div>', unsafe_allow_html=True)

        if chem_res["avoid"]:
            st.markdown('<div class="section-header">❌ Chemicals to AVOID in Current Weather</div>', unsafe_allow_html=True)
            for av in chem_res["avoid"]:
                reasons_html = "<br>".join(f"  • {r}" for r in av["reasons"])
                st.markdown(f"""
<div class="banner banner-red" style="margin-bottom:12px;">
  <div style="font-size:1.0em;font-weight:700;margin-bottom:6px;">❌ {av['name']} ({av['type']})</div>
  <div style="font-size:0.9em;">🚫 <b>Why to avoid now:</b><br>{reasons_html}</div>
  <div style="font-size:0.88em;margin-top:6px;opacity:0.9;">📌 {av['original_avoid_reason']}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 Expected Yield Impact</div>', unsafe_allow_html=True)
        if chem_res["recommended"]:
            best = chem_res["recommended"][0]
            cur_p = st.session_state.get("cur", 0)
            gain_pct = best["yield_gain_pct"]
            gain_qtl = round(yield_qty * gain_pct / 100, 1)
            gain_val = round(gain_qtl * cur_p, 0) if cur_p else 0
            cur_p_kg = qtl_to_kg(cur_p) if cur_p else 0
            st.markdown(f"""
<div class="banner banner-blue">
  <b>🔬 Yield Impact Analysis — {chem_crop_sel}</b><br><br>
  If you apply <b>{best['name']}</b> at the recommended dose:<br>
  • Expected yield gain: <b>+{gain_pct}%</b><br>
  • For your {yield_qty:.0f} Qtl crop: extra <b>+{gain_qtl} Qtl</b> output<br>
  • Estimated extra revenue: <b>Rs.{gain_val:,.0f}</b> (at Rs.{cur_p_kg:.2f}/kg)<br><br>
  ⚠️ <i>Yield gains are estimates. Actual results depend on soil, variety, and application timing.</i>
</div>""", unsafe_allow_html=True)

    elif wd_chem and not wd_chem.get("success"):
        st.warning(f"Weather fetch failed: {wd_chem.get('error')}. Check internet connection.")
    else:
        st.info("👆 Click **Load Weather & Analyze Chemicals** to get weather-based chemical recommendations.")


# ══════════════════════════════════════════════
# TAB 5 — MARKET TRENDS
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📈 Market Price Trends — Tamil Nadu</div>', unsafe_allow_html=True)

    t_crop = st.selectbox("Select Crop for Trend Analysis", all_crops, key="trend_crop")
    t_dist = st.selectbox("Select District", ["All Districts"] + TN_DISTRICTS, key="trend_dist")

    sub = df[df["Commodity"] == t_crop].copy()
    if t_dist != "All Districts":
        sub = sub[sub["District"] == t_dist]

    if not sub.empty:
        sub["Arrival_Date"] = pd.to_datetime(sub["Arrival_Date"])
        trend = sub.groupby("Arrival_Date")[["Modal_Price","Min_Price","Max_Price"]].mean().reset_index()
        trend = trend.sort_values("Arrival_Date").tail(120)
        trend["Modal_kg"] = trend["Modal_Price"] / 100
        trend["Min_kg"]   = trend["Min_Price"]   / 100
        trend["Max_kg"]   = trend["Max_Price"]   / 100

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend["Arrival_Date"], y=trend["Max_kg"],
                                   fill=None, mode="lines", name="Max Price",
                                   line=dict(color="#a5d6a7", width=1)))
        fig2.add_trace(go.Scatter(x=trend["Arrival_Date"], y=trend["Min_kg"],
                                   fill="tonexty", mode="lines", name="Min Price",
                                   line=dict(color="#a5d6a7", width=1),
                                   fillcolor="rgba(165,214,167,0.3)"))
        fig2.add_trace(go.Scatter(x=trend["Arrival_Date"], y=trend["Modal_kg"],
                                   mode="lines", name="Modal Price",
                                   line=dict(color="#2e7d32", width=2.5)))
        fig2.update_layout(
            title=f"{t_crop} Price Trend — {t_dist}",
            xaxis_title="Date", yaxis_title="Price (Rs./kg)",
            height=380,
            plot_bgcolor="rgba(255,255,255,0.04)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            hovermode=False, dragmode=False,
            xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "staticPlot": True, "scrollZoom": False})

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Average Price",    f"Rs.{qtl_to_kg(trend['Modal_Price'].mean()):.2f}/kg")
        s2.metric("Min Price",        f"Rs.{qtl_to_kg(trend['Min_Price'].min()):.2f}/kg")
        s3.metric("Max Price",        f"Rs.{qtl_to_kg(trend['Max_Price'].max()):.2f}/kg")
        s4.metric("Price Volatility", f"Rs.{qtl_to_kg(trend['Modal_Price'].std()):.2f}/kg")

        st.markdown('<div class="section-header">🏆 Top 5 Districts by Average Price</div>', unsafe_allow_html=True)
        top_dist = (df[df["Commodity"] == t_crop]
                    .groupby("District")["Modal_Price"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index())
        top_dist.columns = ["District", "Avg Price (Rs./Qtl)"]
        top_dist["Avg Price (Rs./Qtl)"] = top_dist["Avg Price (Rs./Qtl)"].round(0).astype(int)
        st.dataframe(top_dist.set_index("District"), use_container_width=True)
    else:
        st.info(f"No data found for {t_crop} in {t_dist}.")


# ══════════════════════════════════════════════
# TAB 6 — CNN PLANT DISEASE & MANGO DETECTION
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">📷 AI Plant Disease & Variety Detection</div>', unsafe_allow_html=True)
    st.markdown("Identify crop diseases and mango varieties using CNN + Claude Vision AI.")

    # ── Sub-tab selector ──────────────────────
    detect_mode = st.radio(
        "Select Detection Mode",
        ["🍅 Tomato Early Blight Detection", "🥭 Mango Variety & Disease Detection"],
        horizontal=True,
        key="detect_mode_radio",
    )
    st.markdown("---")

    # ══════════════════════════════
    # TOMATO MODE
    # ══════════════════════════════
    if detect_mode == "🍅 Tomato Early Blight Detection":

        if detector.cnn_available:
            st.markdown('<div class="banner banner-green">🤖 <b>CNN Model Active</b> — MobileNetV2 trained for Tomato Early Blight (2 classes).<br>Upload a tomato leaf photo for instant detection.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner banner-blue">🔍 <b>Claude Vision AI Mode</b> — CNN not trained yet.<br>Upload a tomato leaf and Claude AI will analyze it.<br>💡 Run <code>python train_cnn.py</code> after downloading PlantVillage dataset for offline CNN.</div>', unsafe_allow_html=True)

        col_upload, col_options = st.columns([2, 1])
        with col_upload:
            st.markdown("#### 📸 Upload Tomato Leaf Photo")
            t_uploaded = st.file_uploader("Clear photo of leaf, stem, or fruit",
                                           type=["jpg","jpeg","png","webp"], key="tomato_upload",
                                           help="Best: close-up, natural light, affected area visible")
        with col_options:
            st.markdown("#### ⚙️ Options")
            t_district = st.selectbox("District (weather context)", TN_DISTRICTS,
                                       index=TN_DISTRICTS.index(selected_district), key="t_district")
            t_analyze  = st.button("🔬 Detect Tomato Disease", type="primary",
                                    use_container_width=True, key="t_analyze_btn",
                                    disabled=(t_uploaded is None))

        if t_uploaded:
            t_img = Image.open(t_uploaded)
            c1, c2 = st.columns([1,1])
            with c1: st.image(t_img, caption="Uploaded Tomato Photo", use_container_width=True)
            with c2:
                st.markdown(f"**File:** {t_uploaded.name}")
                st.markdown(f"**Size:** {t_img.size[0]} × {t_img.size[1]} px")
                if not t_analyze: st.info("👆 Click **Detect Tomato Disease** to analyze")

        if t_analyze and t_uploaded:
            with st.spinner("🔬 Analyzing tomato image..."):
                t_img = Image.open(t_uploaded)
                t_weather = {}
                try:
                    wd_t = get_weather(t_district)
                    if wd_t.get("success"): t_weather = wd_t.get("avg_7day", {})
                except Exception: pass
                t_result = detector.predict_from_file(t_img, crop_hint="Tomato")
                st.session_state["tomato_result"]  = t_result
                st.session_state["tomato_weather"] = t_weather
                st.session_state["tomato_district"] = t_district

        if st.session_state.get("tomato_result"):
            result  = st.session_state["tomato_result"]
            t_wx    = st.session_state.get("tomato_weather", {})
            disease = result.get("disease_detected", "Unknown")
            severity= result.get("severity", "Unknown")
            conf    = result.get("confidence_pct", 0)
            is_ok   = detector.is_healthy(result)
            sev_cls = {"None":"cnn-healthy","Low":"cnn-result-card","Medium":"cnn-warning","High":"cnn-disease","Very High":"cnn-disease"}.get(severity, "cnn-result-card")
            sev_icon= {"None":"✅","Low":"🟡","Medium":"🟠","High":"🔴","Very High":"🚨"}.get(severity, "⚠️")
            src_lbl = "🤖 CNN" if result.get("source") == "cnn" else "🔍 Claude Vision"
            tamil   = result.get("tamil_name","")

            st.markdown("---")
            st.markdown('<div class="section-header">🔬 Tomato Detection Results</div>', unsafe_allow_html=True)
            st.markdown(f"""
<div class="cnn-result-card {sev_cls}">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:12px;">
    <div><div style="font-size:0.78em;opacity:0.7;">🌱 Crop</div><div style="font-size:1.15em;font-weight:700;">Tomato</div></div>
    <div><div style="font-size:0.78em;opacity:0.7;">🦠 Disease</div><div style="font-size:1.1em;font-weight:700;">{disease}</div>{'<div style="font-size:0.82em;">🗣️ '+tamil+'</div>' if tamil else ''}</div>
    <div><div style="font-size:0.78em;opacity:0.7;">{sev_icon} Severity</div><div style="font-size:1.15em;font-weight:700;">{severity}</div></div>
    <div><div style="font-size:0.78em;opacity:0.7;">📊 Confidence</div><div style="font-size:1.15em;font-weight:700;">{conf:.0f}%</div><div style="font-size:0.72em;opacity:0.7;">{src_lbl}</div></div>
  </div>
  <div style="font-size:0.84em;opacity:0.8;">📉 Yield risk: <b>{result.get('yield_impact','Unknown')}</b></div>
</div>""", unsafe_allow_html=True)

            top2 = result.get("top2_predictions", [])
            if top2:
                st.markdown("**CNN Class Probabilities:**")
                for i, p in enumerate(top2):
                    lbl = p["label"].replace("___"," → ").replace("_"," ")
                    pc  = p["confidence"] * 100
                    bc  = "#69f0ae" if i == 0 else "#ffd740"
                    st.markdown(f'<div style="margin:3px 0;font-size:0.9em;"><span style="display:inline-block;width:230px;">{i+1}. {lbl}</span><span style="display:inline-block;width:{int(pc)}px;height:11px;background:{bc};border-radius:4px;vertical-align:middle;"></span> <span>{pc:.1f}%</span></div>', unsafe_allow_html=True)

            c_diag, c_act = st.columns(2, gap="medium")
            with c_diag:
                st.markdown("#### 💡 Diagnosis")
                st.markdown(f'<div class="solution-panel">{result.get("diagnosis","")}</div>', unsafe_allow_html=True)
            with c_act:
                st.markdown("#### ✅ Immediate Action")
                st.markdown(f'<div class="reco-panel">{result.get("immediate_action","")}</div>', unsafe_allow_html=True)

            if result.get("chemicals") and not is_ok:
                st.markdown('<div class="section-header">💊 Treatment</div>', unsafe_allow_html=True)
                for ch in result["chemicals"]:
                    st.markdown(f'<div class="banner banner-green"><b>💊 {ch["name"]}</b><br>🧪 {ch["dose"]} &nbsp; 🔁 {ch["frequency"]}</div>', unsafe_allow_html=True)

            if result.get("prevention"):
                st.markdown(f'<div class="banner banner-blue">🛡️ <b>Prevention:</b><br>{result["prevention"]}</div>', unsafe_allow_html=True)

            if t_wx and not is_ok:
                st.markdown('<div class="section-header">🌦️ Weather Impact on Treatment</div>', unsafe_allow_html=True)
                wx_warnings = []
                if t_wx.get("rainfall", 0) > 40: wx_warnings.append(("banner-red", "🌧️ Heavy rainfall — delay foliar sprays."))
                if t_wx.get("temp_max", 30) > 38: wx_warnings.append(("banner-red", f"🌡️ High temp {t_wx['temp_max']}°C — avoid copper/sulphur."))
                if t_wx.get("humidity", 60) > 85: wx_warnings.append(("banner-yellow", f"💧 High humidity {t_wx['humidity']}% — apply fungicide preventively."))
                if not wx_warnings: wx_warnings.append(("banner-green", "✅ Weather suitable for chemical application."))
                for wc, wm in wx_warnings:
                    st.markdown(f'<div class="banner {wc}">{wm}</div>', unsafe_allow_html=True)

            if not is_ok:
                st.markdown("---")
                if st.button("🤖 Get Full AI Advisor Analysis", key="tomato_to_ai"):
                    st.session_state["shared_problem_input"] = f"My Tomato plant has {disease}. {result.get('symptoms_visible','')[:250]}"
                    st.success("✅ Switch to **🤖 AI Advisor** tab and click Analyze My Problem.")

        elif not t_uploaded:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
#### 📸 Photo Tips
- ✅ Natural daylight, close-up of affected leaf
- ✅ Both sides of leaf visible
- ❌ No blurry or dark photos
""")
            with c2:
                st.markdown("""
#### 🍅 Detects
✅ Tomato Early Blight (*Alternaria solani*)
✅ Healthy Tomato Leaf

**Symptoms:** dark concentric-ring spots, yellow halos, defoliation on older leaves
""")
            if not detector.cnn_available:
                with st.expander("🛠️ Train CNN for faster offline detection", expanded=False):
                    st.markdown("""
```bash
# 1. Install TensorFlow
pip install tensorflow pillow numpy

# 2. Download PlantVillage dataset from Kaggle
# https://www.kaggle.com/datasets/emmarex/plantdisease
# Keep only: Tomato___Early_blight/ and Tomato___healthy/ in PlantVillage/

# 3. Train
cd D:/Agri_Assist
python train_cnn.py
```
Expected accuracy: ~98%+ | GPU: ~5 min | CPU: ~30 min | Colab: ~3 min
""")

    # ══════════════════════════════
    # MANGO MODE
    # ══════════════════════════════
    else:
        st.markdown("""
<div class="banner banner-mango">
🥭 <b>Mango Variety & Disease Detection</b><br>
Identifies <b>Alphonso (Hapus)</b> and <b>Imam Pasand (Himayuddin)</b> varieties
and detects diseases like Anthracnose, Powdery Mildew, Hopper Damage, and Die-back.<br>
{}
</div>""".format(
            "🤖 <b>CNN Model Active</b> — Upload a mango leaf or fruit photo for instant CNN detection." if mango_detector.cnn_available
            else "🔍 <b>Claude Vision Mode</b> — Upload a mango photo and Claude AI will identify variety + disease."
        ), unsafe_allow_html=True)

        col_mu, col_mo = st.columns([2, 1])
        with col_mu:
            st.markdown("#### 📸 Upload Mango Photo")
            m_uploaded = st.file_uploader(
                "Upload a clear photo of mango leaf, fruit, or affected area",
                type=["jpg","jpeg","png","webp"], key="mango_upload",
                help="Best: close-up of leaf (both sides) or fruit showing disease symptoms",
            )
        with col_mo:
            st.markdown("#### ⚙️ Options")
            m_district = st.selectbox("District (weather context)", TN_DISTRICTS,
                                       index=TN_DISTRICTS.index(selected_district), key="m_district")
            m_analyze  = st.button("🔬 Detect Mango Variety & Disease", type="primary",
                                    use_container_width=True, key="m_analyze_btn",
                                    disabled=(m_uploaded is None))

        if m_uploaded:
            m_img = Image.open(m_uploaded)
            c1, c2 = st.columns([1, 1])
            with c1: st.image(m_img, caption="Uploaded Mango Photo", use_container_width=True)
            with c2:
                st.markdown(f"**File:** {m_uploaded.name}")
                st.markdown(f"**Size:** {m_img.size[0]} × {m_img.size[1]} px")
                if not m_analyze: st.info("👆 Click **Detect Mango Variety & Disease** to analyze")

        if m_analyze and m_uploaded:
            with st.spinner("🥭 Identifying mango variety and checking for diseases..."):
                m_img = Image.open(m_uploaded)
                m_weather = {}
                try:
                    wd_m = get_weather(m_district)
                    if wd_m.get("success"): m_weather = wd_m.get("avg_7day", {})
                except Exception: pass
                m_result = mango_detector.predict_from_file(m_img, crop_hint="Mango")
                st.session_state["mango_result"]   = m_result
                st.session_state["mango_weather"]  = m_weather
                st.session_state["mango_district"] = m_district

        if st.session_state.get("mango_result"):
            mr       = st.session_state["mango_result"]
            m_wx     = st.session_state.get("mango_weather", {})
            variety  = mr.get("variety_identified", "Unknown")
            health   = mr.get("health_status", "Unknown")
            disease  = mr.get("disease_detected", "Unknown")
            severity = mr.get("severity", "Unknown")
            conf     = mr.get("confidence_pct", 0)
            vconf    = mr.get("variety_confidence", "Medium")
            is_ok    = mango_detector.is_healthy(mr)
            src_lbl  = "🤖 CNN" if mr.get("source") == "cnn" else "🔍 Claude Vision"
            sev_icon = {"None":"✅","Low":"🟡","Medium":"🟠","High":"🔴","Very High":"🚨"}.get(severity,"⚠️")
            card_cls = "mango-healthy" if is_ok else "mango-diseased"
            tamil    = mr.get("tamil_name","")

            st.markdown("---")
            st.markdown('<div class="section-header">🥭 Mango Detection Results</div>', unsafe_allow_html=True)

            # Main result card
            st.markdown(f"""
<div class="mango-card {card_cls}">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:12px;margin-bottom:14px;">
    <div><div style="font-size:0.78em;opacity:0.7;">🥭 Variety</div><div style="font-size:1.15em;font-weight:700;">{variety}</div><div style="font-size:0.72em;opacity:0.7;">Confidence: {vconf}</div></div>
    <div><div style="font-size:0.78em;opacity:0.7;">💚 Status</div><div style="font-size:1.15em;font-weight:700;">{health}</div></div>
    <div><div style="font-size:0.78em;opacity:0.7;">🦠 Disease</div><div style="font-size:1.1em;font-weight:700;">{disease}</div>{'<div style="font-size:0.78em;">🗣️ '+tamil+'</div>' if tamil else ''}</div>
    <div><div style="font-size:0.78em;opacity:0.7;">{sev_icon} Severity</div><div style="font-size:1.15em;font-weight:700;">{severity}</div></div>
    <div><div style="font-size:0.78em;opacity:0.7;">📊 Confidence</div><div style="font-size:1.15em;font-weight:700;">{conf:.0f}%</div><div style="font-size:0.72em;opacity:0.7;">{src_lbl}</div></div>
  </div>
  <div style="font-size:0.84em;opacity:0.8;">📉 Yield impact: <b>{mr.get('yield_impact','Unknown')}</b> &nbsp;|&nbsp; 💰 Market price: <b>{mr.get('market_price_range','—')}</b></div>
</div>""", unsafe_allow_html=True)

            # CNN top predictions bar chart
            top_preds = mr.get("top_predictions", [])
            if top_preds:
                st.markdown("**CNN Class Probabilities:**")
                for i, p in enumerate(top_preds):
                    lbl = p["label"].replace("_", " ")
                    pc  = p["confidence"] * 100
                    bc  = "#ffcc02" if i == 0 else "#ffd740" if i == 1 else "#90caf9"
                    st.markdown(f'<div style="margin:3px 0;font-size:0.88em;"><span style="display:inline-block;width:260px;">{i+1}. {lbl}</span><span style="display:inline-block;width:{int(pc)}px;height:11px;background:{bc};border-radius:4px;vertical-align:middle;"></span> <span>{pc:.1f}%</span></div>', unsafe_allow_html=True)
                st.markdown("")

            # Diagnosis + Action
            c_diag, c_act = st.columns(2, gap="medium")
            with c_diag:
                st.markdown("#### 💡 Diagnosis")
                st.markdown(f'<div class="solution-panel">{mr.get("diagnosis","")}</div>', unsafe_allow_html=True)
            with c_act:
                st.markdown("#### ✅ Immediate Action")
                st.markdown(f'<div class="reco-panel">{mr.get("immediate_action","")}</div>', unsafe_allow_html=True)

            # Variety-specific tips
            if mr.get("variety_tips"):
                st.markdown(f'<div class="banner banner-mango">🥭 <b>Variety Tips — {variety}:</b><br>{mr["variety_tips"]}</div>', unsafe_allow_html=True)

            # Chemicals
            if mr.get("chemicals") and not is_ok:
                st.markdown('<div class="section-header">💊 Treatment Plan</div>', unsafe_allow_html=True)
                for ch in mr["chemicals"]:
                    if ch.get("name"):
                        st.markdown(f'<div class="banner banner-orange"><b>💊 {ch["name"]}</b><br>🧪 Dose: <b>{ch["dose"]}</b> &nbsp; 🔁 {ch["frequency"]}</div>', unsafe_allow_html=True)

            # Prevention
            if mr.get("prevention"):
                st.markdown(f'<div class="banner banner-blue">🛡️ <b>Prevention:</b><br>{mr["prevention"]}</div>', unsafe_allow_html=True)

            # Weather warnings
            if m_wx and not is_ok:
                st.markdown('<div class="section-header">🌦️ Weather Impact on Treatment</div>', unsafe_allow_html=True)
                wx_msgs = []
                if m_wx.get("rainfall", 0) > 40:  wx_msgs.append(("banner-red", "🌧️ Heavy rainfall — delay all foliar sprays. Apply only soil drenches."))
                if m_wx.get("temp_max", 30) > 38:  wx_msgs.append(("banner-red", f"🌡️ High temp ({m_wx['temp_max']}°C) — avoid copper-based sprays (phytotoxic at high temp)."))
                if m_wx.get("humidity", 60) > 80:  wx_msgs.append(("banner-yellow", f"💧 High humidity ({m_wx['humidity']}%) — Anthracnose risk HIGH. Spray preventive fungicide."))
                if not wx_msgs: wx_msgs.append(("banner-green", "✅ Weather conditions are suitable for spraying. Apply in early morning or evening."))
                for wc, wm in wx_msgs:
                    st.markdown(f'<div class="banner {wc}">{wm}</div>', unsafe_allow_html=True)

            if not is_ok:
                st.markdown("---")
                if st.button("🤖 Get Full AI Advisor Analysis for This Disease", key="mango_to_ai"):
                    st.session_state["shared_problem_input"] = f"My {variety} mango has {disease}. {mr.get('symptoms_visible','')[:250]}"
                    st.success("✅ Switch to **🤖 AI Advisor** tab and click Analyze My Problem.")

        elif not m_uploaded:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
#### 📸 Mango Photo Tips
- ✅ Photograph affected **leaf (both sides)**
- ✅ Include affected **fruits** if visible
- ✅ Take in **natural outdoor light**
- ✅ Show **stem/shoot tips** if die-back suspected
- ❌ Avoid very distant shots of whole tree
""")
            with c2:
                st.markdown("""
#### 🥭 What This Detects

**Alphonso (Hapus):**
✅ Healthy leaf/fruit identification
🔴 Anthracnose (black spots on fruit)
🔴 Powdery Mildew (white coating)

**Imam Pasand (Himayuddin):**
✅ Healthy leaf/fruit identification
🔴 Hopper Damage (curled leaves)
🔴 Die-back (brown shoot tips)
🔴 Stem-end Rot
""")

            if not mango_detector.cnn_available:
                with st.expander("🛠️ Train Mango CNN for faster offline detection", expanded=False):
                    st.markdown("""
### Step 1 — Collect Dataset

Create this folder structure in `D:/Agri_Assist/MangoDataset/`:
```
MangoDataset/
  Alphonso_Healthy/      ← 50-200 photos of healthy Alphonso leaves/fruits
  Alphonso_Diseased/     ← 50-200 photos of diseased Alphonso
  ImamPasand_Healthy/    ← 50-200 photos of healthy Imam Pasand
  ImamPasand_Diseased/   ← 50-200 photos of diseased Imam Pasand
```

**Photo sources:**
- Your own orchard photos (best!)
- Kaggle: https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset (rename folders)
- Google Images search for each variety

### Step 2 — Train
```bash
cd D:/Agri_Assist
python train_mango_cnn.py
```

**Training time:**
- GPU (NVIDIA RTX): ~5–8 minutes
- CPU only: ~40–60 minutes
- Google Colab (free GPU): ~5 minutes ← recommended

**Output:** `mango_variety_model.h5` — restart app to activate CNN mode.
""")
