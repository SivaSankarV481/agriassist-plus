"""
whatsapp_bot.py
================
AgriAssist+ WhatsApp Chatbot — Meta Cloud API webhook server.

Run (development):
    pip install flask requests python-dotenv
    python whatsapp_bot.py

Expose publicly for Meta webhook verification:
    ngrok http 5000
    Then set webhook URL in Meta Developer Console to:
    https://<ngrok-id>.ngrok.io/webhook

Environment variables needed (add to .env):
    WHATSAPP_TOKEN        — Meta permanent access token
    WHATSAPP_VERIFY_TOKEN — any secret string you choose (for webhook verification)
    PHONE_NUMBER_ID       — from Meta Developer Console
"""

import os
import re
import json
import joblib
import hashlib
import requests
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WHATSAPP_TOKEN        = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "agriassist_verify")
PHONE_NUMBER_ID       = os.getenv("PHONE_NUMBER_ID", "")
WHATSAPP_API_URL      = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ─────────────────────────────────────────────
# Web App URL — change this to your deployed Streamlit URL
# ─────────────────────────────────────────────
WEB_APP_URL = os.getenv("WEB_APP_URL", "https://your-agriassist-app.streamlit.app")

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load AgriAssist+ models once at startup
# ─────────────────────────────────────────────
print("Loading models…")
MODELS, META, DF, RAG = {}, None, None, None

try:
    for key, fname in {
        "current":    "current_price_model.joblib",
        "1week":      "future_1w_model.joblib",
        "2week":      "future_2w_model.joblib",
        "1month":     "future_1m_model.joblib",
        "yield_loss": "yield_loss_model.joblib",
    }.items():
        if os.path.exists(fname):
            MODELS[key] = joblib.load(fname)
    META = joblib.load("model_meta.joblib") if os.path.exists("model_meta.joblib") else None
    DF   = pd.read_csv("tn_agri_dataset.csv") if os.path.exists("tn_agri_dataset.csv") else None
    from rag_engine import RAGEngine
    RAG = RAGEngine()
    print("✅ Models, dataset and RAGEngine loaded.")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

from weather import get_weather

# ─────────────────────────────────────────────
# Constants (same as app.py)
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
    "Kallakurichi","Tenkasi","Mayiladuthurai",
]

# ─────────────────────────────────────────────
# States
# ─────────────────────────────────────────────
# idle            — first message, no context
# main_menu       — showing top-level 1-4 menu
# post_prediction — after price result, showing 1=Weather 2=AI 3=Chemical
# choose_crop     — waiting for crop name
# choose_district — waiting for district name
# ask_problem     — waiting for free-text disease description

# ─────────────────────────────────────────────
# In-memory session store  {phone: session_dict}
# ─────────────────────────────────────────────
SESSIONS = {}

def get_session(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {
            "state":      "idle",
            "crop":       None,
            "district":   None,
            "weather":    None,
            "prices":     {},
            "yield_loss": 0.0,
        }
    return SESSIONS[phone]

def reset_session(phone: str):
    SESSIONS[phone] = {
        "state": "idle", "crop": None, "district": None,
        "weather": None, "prices": {}, "yield_loss": 0.0,
    }

# ─────────────────────────────────────────────
# Helpers — duplicated from app.py (keep in sync)
# ─────────────────────────────────────────────
def encode(col, val):
    if not META:
        return 0
    d = META["encoders"].get(col, {})
    return d.get(val, d.get(list(d.keys())[0], 0) if d else 0)

def get_price_stats(crop, district):
    if DF is None:
        return {"modal":3000,"min":2500,"max":3500,"lag1":3000,"lag2":3000,"lag4":3000,"lag8":3000,"lag12":3000}
    sub = DF[(DF["Commodity"] == crop) & (DF["District"] == district)]
    if sub.empty:
        sub = DF[DF["Commodity"] == crop]
    if sub.empty:
        return {"modal":3000,"min":2500,"max":3500,"lag1":3000,"lag2":3000,"lag4":3000,"lag8":3000,"lag12":3000}
    latest = sub.sort_values("Arrival_Date").iloc[-1]
    return {
        "modal": float(latest.get("Modal_Price", 3000)),
        "min":   float(latest.get("Min_Price",   2500)),
        "max":   float(latest.get("Max_Price",   3500)),
        "lag1":  float(latest.get("Lag1",  3000)),
        "lag2":  float(latest.get("Lag2",  3000)),
        "lag4":  float(latest.get("Lag4",  3000)),
        "lag8":  float(latest.get("Lag8",  3000)),
        "lag12": float(latest.get("Lag12", 3000)),
    }

def build_price_row(crop, district, weather_data):
    dt     = pd.to_datetime(date.today())
    month  = dt.month
    week   = int(dt.isocalendar()[1])
    season = SEASON_MAP.get(month, "Monsoon")
    stats  = get_price_stats(crop, district)
    min_p  = stats["min"]; max_p = stats["max"]
    spread = round(max_p - min_p, 2)
    mid    = round((max_p + min_p) / 2, 2)
    w      = weather_data.get("avg_7day", {})
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code":  encode("District",  district),
        "Market_Code":    encode("Market",    f"{district} APMC"),
        "Variety_Code":   encode("Variety",   "Local"),
        "Month":          month, "Week": week,
        "Season_Code":    SEASON_CODES.get(season, 0),
        "Min_Price":      min_p, "Max_Price": max_p,
        "Price_Spread":   spread, "Price_Mid": mid,
        "Spread_Ratio":   round(spread / mid if mid > 0 else 0, 4),
        "Lag1":  stats["lag1"],  "Lag2":  stats["lag2"],
        "Lag4":  stats["lag4"],  "Lag8":  stats["lag8"],
        "Lag12": stats["lag12"],
        "Rainfall_mm":  w.get("rainfall", 30),
        "Temp_Max_C":   w.get("temp_max", 32),
        "Temp_Min_C":   w.get("temp_min", 24),
        "Humidity_Pct": w.get("humidity", 60),
    }
    pf = META["price_features"] if META else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in pf}])

def build_yield_row(crop, district, weather_data):
    dt     = pd.to_datetime(date.today())
    month  = dt.month
    season = SEASON_MAP.get(month, "Monsoon")
    w      = weather_data.get("avg_7day", {})
    stats  = get_price_stats(crop, district)
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code":  encode("District",  district),
        "Month":          month,
        "Season_Code":    SEASON_CODES.get(season, 0),
        "Rainfall_mm":    w.get("rainfall", 30),
        "Temp_Max_C":     w.get("temp_max", 32),
        "Temp_Min_C":     w.get("temp_min", 24),
        "Humidity_Pct":   w.get("humidity", 60),
        "Lag1": stats["lag1"], "Lag4": stats["lag4"], "Lag8": stats["lag8"],
    }
    yf = META["yield_features"] if META else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in yf}])

def qtl_to_kg(price_qtl):
    return round(price_qtl / 100, 2) if price_qtl else 0

def trend_icon(base, compare):
    if not compare or not base:
        return "➡️"
    d = (compare - base) / base * 100 if base > 0 else 0
    return f"🔺+{d:.1f}%" if d > 2 else f"🔻{d:.1f}%" if d < -2 else f"➡️{d:+.1f}%"

# ─────────────────────────────────────────────
# Intent parser
# ─────────────────────────────────────────────
CROP_KEYWORDS = {}

def build_crop_keywords():
    global CROP_KEYWORDS
    if DF is None:
        return
    for crop in DF["Commodity"].dropna().unique():
        CROP_KEYWORDS[crop.lower()] = crop
        CROP_KEYWORDS[crop.split()[0].lower()] = crop

def parse_intent(text: str, session: dict) -> str:
    t     = text.strip().lower()
    state = session.get("state", "idle")

    # ── Global shortcuts (work from any state) ──
    if t in ("hi", "hello", "start", "menu", "help", "/start"):
        return "greeting"
    if t in ("0", "reset", "restart", "main menu"):
        return "reset"

    # ── Main menu (1-4) ──────────────────────────
    if state == "main_menu":
        if t in ("1", "price", "price check"):
            return "price_check"
        if t in ("2", "weather", "weather alert"):
            return "weather_alert"
        if t in ("3", "disease", "disease help", "ai advice"):
            return "ai_advice"
        if t in ("4", "chemical", "chemicals", "chemical tips"):
            return "chemical_advice"

    # ── Post-prediction sub-menu (1=Weather 2=AI 3=Chemical) ──
    if state == "post_prediction":
        if t in ("1", "weather", "weather alert"):
            return "weather_alert"
        if t in ("2", "disease", "disease help", "ai advice"):
            return "ai_advice"
        if t in ("3", "chemical", "chemicals", "chemical tips"):
            return "chemical_advice"

    # ── Free-text inputs — check BEFORE crop/district detection
    # so "my tomato leaves are yellow" doesn't get parsed as an inline crop query
    if state == "ask_problem":
        return "problem_description"
    if state == "ask_chemical_problem":
        return "chemical_problem_description"

    # ── State-driven crop / district input ───────
    words = t.split()
    if state == "choose_crop":
        for w in words:
            if w in CROP_KEYWORDS:
                return "set_crop"

    if state == "choose_district":
        for d in TN_DISTRICTS:
            if t in d.lower() or d.lower().startswith(t):
                return "set_district"

    # ── Inline crop+district detection (any state except ask_problem) ──
    build_crop_keywords()
    found_crop, found_dist = None, None
    for w in words:
        if w in CROP_KEYWORDS:
            found_crop = CROP_KEYWORDS[w]
        for d in TN_DISTRICTS:
            if w == d.lower() or d.lower().startswith(w) and len(w) >= 4:
                found_dist = d
    if found_crop:
        session["_inline_crop"] = found_crop
    if found_dist:
        session["_inline_dist"] = found_dist
    if found_crop or found_dist:
        return "inline_query"

    return "unknown"

# ─────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────
ALL_CROPS_LIST = []

def get_all_crops():
    global ALL_CROPS_LIST
    if not ALL_CROPS_LIST and DF is not None:
        ALL_CROPS_LIST = sorted(DF["Commodity"].dropna().unique().tolist())
    return ALL_CROPS_LIST

def welcome_message():
    return (
        "🌾 *AgriAssist+* — Tamil Nadu Crop Intelligence\n\n"
        "Hello Farmer! 👋 I can help you with:\n\n"
        "1️⃣  Price prediction\n"
        "2️⃣  Weather alert\n"
        "3️⃣  Disease & AI advice\n"
        "4️⃣  Chemical recommendations\n\n"
        "Reply with a number *or* type your crop & district directly.\n"
        "_Example: tomato coimbatore_\n\n"
        f"🌐 *Full Web App:* {WEB_APP_URL}\n"
        "_(More features: charts, trends, detailed analysis)_\n\n"
        "Type *0* anytime to return to this menu."
    )

def ask_crop_message():
    crops  = get_all_crops()
    sample = ", ".join(crops[:12]) + "…"
    return (
        f"🌱 *Which crop?*\n\nAvailable: {sample}\n\n"
        "Just type the crop name (e.g. *Tomato*)"
    )

def ask_district_message(crop):
    sample = ", ".join(TN_DISTRICTS[:8]) + "…"
    return (
        f"📍 *Which district?* (for {crop})\n\n"
        f"Available: {sample}\n\n"
        "Type your district name (e.g. *Coimbatore*)"
    )

def price_result_message(crop, district, prices, yield_loss):
    """
    After showing price results, display sub-menu with:
    1=Weather  2=AI advice  3=Chemical tips
    State must be set to 'post_prediction' before returning this.
    """
    cur = prices.get("current", 0)
    p1w = prices.get("1week",   0)
    p1m = prices.get("1month",  0)
    yl_icon = "🔴" if yield_loss > 25 else "🟠" if yield_loss > 10 else "🟢"

    # ── Sell or Wait logic ────────────────────────────────
    price_trend_1m = ((p1m - cur) / cur * 100) if p1m and cur else 0
    price_trend_1w = ((p1w - cur) / cur * 100) if p1w and cur else 0

    if yield_loss > 25 and price_trend_1m <= 2:
        sell_icon     = "🚨"
        sell_decision = "SELL NOW"
        sell_reason   = f"High yield risk ({yield_loss:.0f}%) + prices flat. Every delay risks more loss."
    elif yield_loss > 25 and price_trend_1m > 2:
        sell_icon     = "⚠️"
        sell_decision = "SELL WITHIN 1 WEEK"
        sell_reason   = f"High yield risk ({yield_loss:.0f}%) — crop loss will outweigh price gains. Sell soon."
    elif price_trend_1m > 8 and yield_loss <= 15:
        sell_icon     = "⏳"
        sell_decision = "WAIT — HOLD FOR 1 MONTH"
        sell_reason   = f"Prices rising {price_trend_1m:.1f}% next month → Rs.{qtl_to_kg(p1m):.2f}/kg. Low risk — holding is profitable."
    elif price_trend_1w > 3 and yield_loss <= 20:
        sell_icon     = "📅"
        sell_decision = "WAIT — SELL IN 1 WEEK"
        sell_reason   = f"Prices rising {price_trend_1w:.1f}% this week. Hold a few more days for better return."
    else:
        sell_icon     = "✅"
        sell_decision = "SELL NOW — PRICES STABLE"
        sell_reason   = "No major price upside. Selling now avoids storage costs and yield risk."

    lines = [
        f"📊 *{crop} — {district}*",
        f"📅 {date.today().strftime('%d %b %Y')}",
        "",
        f"💰 Current:  *Rs.{qtl_to_kg(cur):.2f}/kg*",
        f"📅 1 Week:   *Rs.{qtl_to_kg(p1w):.2f}/kg* {trend_icon(cur, p1w)}",
        f"📅 1 Month:  *Rs.{qtl_to_kg(p1m):.2f}/kg* {trend_icon(cur, p1m)}",
        "",
        f"{yl_icon} Yield risk: *{yield_loss:.1f}%*",
        "",
        f"{sell_icon} *{sell_decision}*",
        f"   {sell_reason}",
        "",
        "What next?",
        "1️⃣ Weather alert",
        "2️⃣ Disease & AI advice",
        "3️⃣ Chemical tips",
        "0️⃣ Main menu",
        "",
        f"📱 *Want full charts & trends?*",
        f"👉 {WEB_APP_URL}",
    ]
    return "\n".join(lines)

def weather_message(district, weather_data):
    if not weather_data or not weather_data.get("success"):
        return f"⚠️ Could not fetch weather for {district}. Please try again."

    cur = weather_data.get("current", {})
    avg = weather_data.get("avg_7day", {})

    alerts = []
    tmax = avg.get("temp_max", 30)
    rain = avg.get("rainfall", 20)
    hum  = avg.get("humidity", 60)
    if tmax > 38:
        alerts.append("🌡️ Heat stress risk — check tomato, chilli, brinjal")
    if rain > 150:
        alerts.append("🌊 Flood risk — ensure drainage")
    if rain < 5:
        alerts.append("🏜️ Drought risk — irrigate now")
    if hum > 80:
        alerts.append("🦠 Fungal risk — apply preventive fungicide")
    if not alerts:
        alerts.append("✅ Conditions favourable — no immediate alerts")

    lines = [
        f"🌦️ *Weather — {district}*",
        "",
        f"🌡️ Temp:     {cur.get('temperature','N/A')} °C",
        f"💧 Humidity: {cur.get('humidity','N/A')} %",
        f"🌧️ Rainfall: {cur.get('precipitation', 0)} mm",
        f"💨 Wind:     {cur.get('wind_speed','N/A')} km/h",
        "",
        "⚠️ *Alerts:*",
    ] + [f"  • {a}" for a in alerts] + [
        "",
        "2️⃣ AI advice  |  3️⃣ Chemicals  |  0️⃣ Menu",
    ]
    return "\n".join(lines)

def ask_problem_message(crop, district):
    return (
        f"🤖 *AI Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe your crop problem:*\n"
        "Tell me what you are seeing — symptoms, affected parts, how long it has been happening.\n\n"
        "_Examples:_\n"
        "  • Leaves turning yellow with brown spots\n"
        "  • Fruits rotting before harvest\n\n"
        "Type your problem ↓"
    )

def ask_chemical_problem_message(crop, district):
    return (
        f"🧪 *Chemical Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe the disease or problem:*\n"
        "Tell me what symptoms you are seeing so I can recommend the right chemical.\n\n"
        "_Examples:_\n"
        "  • White powder on leaves and stems\n"
        "  • Insects eating the leaves\n\n"
        "Type your problem ↓"
    )


def chemical_problem_message(crop, district, problem, weather, yield_loss, phone="anon"):
    """
    Calls llm_chemical_advice() for a focused, farmer-friendly chemical recommendation.
    Format: medicine name, dosage, how to apply, price impact.
    """
    if RAG is None:
        return "⚠️ Chemical advisor not available right now.\n\n0️⃣ Main menu"

    weather_avg = weather.get("avg_7day", {}) if weather else {}
    session_id  = hashlib.md5(f"wa_{phone}".encode()).hexdigest()[:16]

    try:
        # Get weather-filtered chemical list first
        from rag_engine import recommend_chemicals
        chem_res = recommend_chemicals(crop, weather_avg, yield_loss)

        # Call LLM for farmer-friendly advice
        advice = RAG.llm_chemical_advice(
            crop       = crop,
            disease_input = problem,
            weather    = weather_avg,
            chem_res   = chem_res,
            yield_loss = yield_loss,
            district   = district,
        )

        # Clean up any HTML/bold markdown
        advice = re.sub(r'<[^>]+>', '', advice).strip()
        advice = re.sub(r'\*\*(.+?)\*\*', r'\1', advice)

        problem_preview = problem[:90] + ('...' if len(problem) > 90 else '')

        body = (
            f"🧪 *Chemical Advisor — {crop}*\n"
            f"📍 {district}\n"
            f"❓ Problem: _{problem_preview}_\n"
            f"─────────────────────\n\n"
            f"{advice[:3800]}\n\n"
            f"─────────────────────\n"
            f"2️⃣ AI advice  |  0️⃣ Main menu"
        )
        return body[:4090]

    except Exception as e:
        return f"⚠️ Chemical advisor error: {e}\n\n0️⃣ Main menu"


def ai_advice_message(crop, district, problem, weather, prices, yield_loss, phone="anon"):
    if RAG is None:
        return "⚠️ AI advisor not available right now. Please try again later.\n\n0️⃣ Main menu"

    session_id = hashlib.md5(f"wa_{phone}".encode()).hexdigest()[:16]
    ctx = {
        "crop":           crop,
        "district":       district,
        "weather":        weather.get("avg_7day", {}) if weather else {},
        "prices":         prices,
        "yield_loss_pct": yield_loss,
        "session_id":     session_id,
    }
    try:
        result   = RAG.diagnose_problem(problem, ctx)
        solution = result.get("solution", "").strip()
        recs     = result.get("recommendations", "").strip()

        # Clean up any HTML/markdown tags that may come from the Streamlit version
        solution = re.sub(r'<[^>]+>', '', solution).strip()
        recs     = re.sub(r'<[^>]+>', '', recs).strip()
        # Remove bold markdown (**text**) for cleaner WhatsApp plain text
        solution = re.sub(r'\*\*(.+?)\*\*', r'\1', solution)
        recs     = re.sub(r'\*\*(.+?)\*\*', r'\1', recs)

        # Truncate problem preview for display
        problem_preview = problem[:100] + ('...' if len(problem) > 100 else '')

        body = (
            f"🧠 *AI Analysis — {crop}*\n"
            f"📍 {district}\n"
            f"❓ You said: _{problem_preview}_\n"
            f"─────────────────────\n\n"
            f"🔍 *What is happening:*\n"
            f"{solution[:500]}\n\n"
            f"─────────────────────\n"
            f"✅ *What you should do:*\n"
            f"{recs[:650]}\n\n"
            f"─────────────────────\n"
            f"3️⃣ Chemical tips  |  0️⃣ Main menu"
        )
        return body[:4090]
    except Exception as e:
        return f"⚠️ AI advisor error: {e}\n\n0️⃣ Main menu"



def unknown_message():
    return (
        "🤔 I didn't understand that.\n\n"
        "Try:\n"
        "• Crop + district (e.g. *Tomato Coimbatore*)\n"
        "• *1* Price  *2* Weather\n"
        "• *3* AI advice  *4* Chemicals\n"
        "• *0* Main menu"
    )

# ─────────────────────────────────────────────
# Core prediction runner
# ─────────────────────────────────────────────
def run_prediction(crop, district, session):
    weather_data = get_weather(district)
    session["weather"] = weather_data

    prices     = {}
    yield_loss = 0.0

    if MODELS and META and DF is not None:
        try:
            X_price = build_price_row(crop, district, weather_data)
            X_yield = build_yield_row(crop, district, weather_data)
            prices = {
                "current": float(MODELS["current"].predict(X_price)[0]),
                "1week":   float(MODELS["1week"].predict(X_price)[0]),
                "1month":  float(MODELS["1month"].predict(X_price)[0]) if "1month" in MODELS else 0,
            }
            yield_loss = float(MODELS["yield_loss"].predict(X_yield)[0]) if "yield_loss" in MODELS else 0.0
        except Exception as e:
            print(f"Prediction error: {e}")

    session["prices"]     = prices
    session["yield_loss"] = yield_loss

    # Save to MySQL
    try:
        from database import get_db
        db = get_db()
        if db.is_connected:
            session_id = hashlib.md5(f"wa_{session.get('phone','anon')}".encode()).hexdigest()[:16]
            db.upsert_session(session_id, crop, district)
            db.save_prediction(session_id, {
                "crop": crop, "variety": "Local", "district": district,
                "market": f"{district} APMC",
                "prediction_date": str(date.today()),
                "current_price":  round(prices.get("current", 0), 2),
                "price_1week":    round(prices.get("1week",   0), 2),
                "price_2week":    None,
                "price_1month":   round(prices.get("1month",  0), 2),
                "yield_loss_pct": round(yield_loss, 2),
                "yield_qty_qtl":  10,
            })
    except Exception:
        pass

    return weather_data, prices, yield_loss

# ─────────────────────────────────────────────
# Main conversation handler
# ─────────────────────────────────────────────
def handle_message(phone: str, text: str) -> str:
    session = get_session(phone)
    session["phone"] = phone
    intent  = parse_intent(text, session)

    # ── Greeting / reset ──────────────────────
    if intent == "greeting":
        session["state"] = "main_menu"
        return welcome_message()

    if intent == "reset":
        reset_session(phone)
        SESSIONS[phone]["state"] = "main_menu"
        return welcome_message()

    # ── Inline crop+district in one message ───
    if intent == "inline_query":
        crop = session.pop("_inline_crop", None) or session.get("crop")
        dist = session.pop("_inline_dist", None) or session.get("district")

        if crop and dist:
            session["crop"]     = crop
            session["district"] = dist
            weather, prices, yl = run_prediction(crop, dist, session)
            session["state"]    = "post_prediction"   # ← key fix
            return price_result_message(crop, dist, prices, yl)

        if crop and not dist:
            session["crop"]  = crop
            session["state"] = "choose_district"
            return ask_district_message(crop)

        if dist and not crop:
            session["district"] = dist
            session["state"]    = "choose_crop"
            return ask_crop_message()

    # ── Main menu selections (1–4) ────────────
    if intent == "price_check":
        if session.get("crop") and session.get("district"):
            weather, prices, yl = run_prediction(session["crop"], session["district"], session)
            session["state"] = "post_prediction"      # ← key fix
            return price_result_message(session["crop"], session["district"], prices, yl)
        session["state"] = "choose_crop"
        return ask_crop_message()

    if intent == "weather_alert":
        if session.get("district"):
            wd = session.get("weather") or get_weather(session["district"])
            session["weather"] = wd
            session["state"]   = "post_prediction"
            return weather_message(session["district"], wd)
        session["state"] = "choose_district"
        return "📍 *Which district?*\nType your district name (e.g. *Coimbatore*)"

    if intent == "ai_advice":
        if session.get("crop") and session.get("district"):
            session["state"] = "ask_problem"          # ← goes to free-text
            return ask_problem_message(session["crop"], session["district"])
        session["state"] = "choose_crop"
        return ask_crop_message()

    if intent == "chemical_advice":
        if session.get("crop") and session.get("district"):
            session["state"] = "ask_chemical_problem"
            return ask_chemical_problem_message(session["crop"], session["district"])
        session["state"] = "choose_crop"
        return ask_crop_message()

    # ── State-driven flows ────────────────────
    if session["state"] == "choose_crop":
        t = text.strip().lower()
        build_crop_keywords()
        for w in t.split():
            if w in CROP_KEYWORDS:
                session["crop"]  = CROP_KEYWORDS[w]
                session["state"] = "choose_district"
                return ask_district_message(session["crop"])
        return f"🌱 I didn't recognise that crop.\n\n{ask_crop_message()}"

    if session["state"] == "choose_district":
        t = text.strip().lower()
        for d in TN_DISTRICTS:
            if t in d.lower() or d.lower().startswith(t):
                session["district"] = d
                weather, prices, yl = run_prediction(session["crop"], d, session)
                session["state"]    = "post_prediction"   # ← key fix
                return price_result_message(session["crop"], d, prices, yl)
        return f"📍 I didn't recognise that district.\n\n{ask_district_message(session.get('crop','your crop'))}"

    if session["state"] == "ask_problem":
        if len(text.strip()) < 5:
            return (
                "⚠️ Please describe the problem in a bit more detail.\n\n"
                "_Example: Leaves turning yellow, white powder on stems_"
            )
        session["state"] = "post_prediction"
        return ai_advice_message(
            session["crop"],
            session["district"],
            text.strip(),
            session.get("weather") or {},
            session.get("prices", {}),
            session.get("yield_loss", 0),
            phone=phone,
        )

    if session["state"] == "ask_chemical_problem":
        if len(text.strip()) < 5:
            return (
                "⚠️ Please describe the disease or problem in a bit more detail.\n\n"
                "_Example: White powder on leaves, insects eating stems_"
            )
        # Fetch weather if not already cached
        if not session.get("weather"):
            session["weather"] = get_weather(session["district"])
        session["state"] = "post_prediction"
        return chemical_problem_message(
            session["crop"],
            session["district"],
            text.strip(),
            session.get("weather") or {},
            session.get("yield_loss", 0),
            phone=phone,
        )

    # ── Idle / first message ──────────────────
    if session["state"] in ("idle", "main_menu", "post_prediction"):
        return unknown_message()

    return unknown_message()

# ─────────────────────────────────────────────
# WhatsApp API — send message
# ─────────────────────────────────────────────
def send_whatsapp_message(to: str, body: str):
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print(f"[MOCK] To {to}:\n{body}\n")
        return

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type":    "individual",
        "to":                to,
        "type":              "text",
        "text":              {"preview_url": False, "body": body},
    }
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":  "application/json",
    }
    try:
        r = requests.post(WHATSAPP_API_URL, json=payload, headers=headers, timeout=15)
        
        # ── Friendly error messages for common failures ────────
        if r.status_code == 401:
            print("")
            print("❌ ERROR 401 — TOKEN EXPIRED!")
            print("─" * 50)
            print("Your WhatsApp token has expired. Fix it now:")
            print("1. Go to developers.facebook.com")
            print("2. Open your app → WhatsApp → API Setup")
            print("3. Copy the new Temporary Access Token")
            print("4. Update WHATSAPP_TOKEN in your .env file")
            print("5. Restart this bot: python whatsapp_bot.py")
            print("─" * 50)
            print("TIP: Use a permanent System User token to avoid this!")
            print("See: business.facebook.com → Settings → System Users")
            print("")
            return
        
        if r.status_code == 403:
            print("❌ ERROR 403 — Permission denied. Check your token has whatsapp_business_messaging permission.")
            return

        r.raise_for_status()
        print(f"✅ Message sent to {to}")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Failed to send to {to}: {e}")
    except Exception as e:
        print(f"❌ Failed to send to {to}: {e}")

# ─────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    print(f"🔍 Verification attempt — mode={mode} token={token} challenge={challenge}")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print("✅ Webhook verified by Meta.")
        return challenge, 200
    print(f"❌ Verification failed — expected token: {WHATSAPP_VERIFY_TOKEN!r}")
    return "Forbidden", 403

@app.route("/", methods=["GET"])
def health_check():
    return "AgriAssist+ WhatsApp Bot is running ✅", 200

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json(silent=True)
    if not data:
        print("⚠️ No JSON data received")
        return jsonify({"status": "no data"}), 400

    # Log the full payload so we can see exactly what Meta sends
    print(f"📦 Raw payload: {json.dumps(data, indent=2)}")

    try:
        entry   = data["entry"][0]
        changes = entry["changes"][0]
        value   = changes["value"]

        if "messages" not in value:
            print(f"ℹ️ No messages in payload — ignoring. Value keys: {list(value.keys())}")
            return jsonify({"status": "ignored"}), 200

        msg      = value["messages"][0]
        phone    = msg["from"]
        msg_type = msg.get("type", "")
        print(f"📩 From {phone} | type={msg_type}")

        if msg_type == "text":
            text = msg["text"]["body"]
        elif msg_type == "interactive":
            interactive = msg["interactive"]
            if interactive["type"] == "button_reply":
                text = interactive["button_reply"]["id"]
            elif interactive["type"] == "list_reply":
                text = interactive["list_reply"]["id"]
            else:
                text = ""
        else:
            print(f"⚠️ Unsupported message type: {msg_type}")
            send_whatsapp_message(phone, "Sorry, I only understand text messages right now.")
            return jsonify({"status": "unsupported type"}), 200

        print(f"💬 Message text: {text!r}")
        reply = handle_message(phone, text)
        print(f"🤖 Reply: {reply[:100]}…")
        send_whatsapp_message(phone, reply)

    except (KeyError, IndexError) as e:
        print(f"❌ Webhook parse error: {e} | body: {json.dumps(data)}")

    return jsonify({"status": "ok"}), 200

# ─────────────────────────────────────────────
# CLI test mode — test the full flow locally
# ─────────────────────────────────────────────
def cli_test():
    print("\n" + "="*55)
    print("  AgriAssist+ WhatsApp Bot — CLI Test Mode")
    print("="*55)
    print("Type messages as if you were a farmer on WhatsApp.")
    print("Type 'quit' to exit.\n")
    build_crop_keywords()
    phone = "test_farmer_1234"
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() == "quit":
            break
        reply = handle_message(phone, text)
        print(f"\nBot:\n{reply}\n")
        print("-" * 45)

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    build_crop_keywords()
    if "--cli" in sys.argv:
        cli_test()
    else:
        if not WHATSAPP_TOKEN:
            print("⚠️  WHATSAPP_TOKEN not set — bot will receive messages but cannot send replies.")
            print("    Add WHATSAPP_TOKEN to your .env file to enable sending.\n")
        print("🚀 Starting AgriAssist+ WhatsApp webhook server on port 8000…")
        print("   Test locally: http://localhost:8000")
        app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
