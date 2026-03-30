"""
whatsapp_bot.py
================
AgriAssist+ WhatsApp Chatbot — Meta Cloud API webhook server.

Run (development):
    pip install -r requirements.txt
    python whatsapp_bot.py

Expose publicly for Meta webhook verification:
    ngrok http 8000
    Then set webhook URL in Meta Developer Console to:
    https://<ngrok-id>.ngrok.io/webhook
"""

import os
import re
import joblib
import hashlib
import requests
import pandas as pd
from datetime import date
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ── Translation ──────────────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("⚠️  deep-translator not installed. Run: pip install deep-translator")

try:
    from langdetect import detect as _langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not installed. Run: pip install langdetect")

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WHATSAPP_TOKEN        = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "agriassist_verify")
PHONE_NUMBER_ID       = os.getenv("PHONE_NUMBER_ID", "")
WHATSAPP_API_URL      = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ── Web App URL — your Streamlit Cloud deployment ────────────────
WEB_APP_URL = os.getenv(
    "WEB_APP_URL",
    "https://siddharth02122004-agri-assist-app.streamlit.app"   # ← update after deploy
)

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
    "Kallakurichi","Tenkasi","Mayiladuthurai",
]

# ─────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": ("English",   "1"),
    "ta": ("Tamil",     "2"),
    "hi": ("Hindi",     "3"),
    "te": ("Telugu",    "4"),
    "kn": ("Kannada",   "5"),
    "ml": ("Malayalam", "6"),
}
LANG_BY_NUMBER = {"1":"en", "2":"ta", "3":"hi", "4":"te", "5":"kn", "6":"ml"}

def language_menu() -> str:
    return (
        "Select your language / Mozhi Thervuseiyungal:\n\n"
        "1  English\n"
        "2  Tamil (தமிழ்)\n"
        "3  Hindi (हिंदी)\n"
        "4  Telugu (తెలుగు)\n"
        "5  Kannada (ಕನ್ನಡ)\n"
        "6  Malayalam (മലയാളം)\n\n"
        "Reply with a number (1-6)"
    )

def detect_language(text: str) -> str:
    """Auto-detect language using langdetect."""
    if not LANGDETECT_AVAILABLE or len(text.strip()) < 5:
        return "en"
    try:
        lang = _langdetect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except Exception:
        return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate farmer's message to English for processing."""
    if not TRANSLATION_AVAILABLE or source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as e:
        print(f"⚠️  Translation error (→EN): {e}")
        return text

def translate_reply(text: str, target_lang: str) -> str:
    """Translate bot reply back to farmer's language, preserving URLs."""
    if not TRANSLATION_AVAILABLE or not target_lang or target_lang == "en":
        return text
    try:
        url_pat = re.compile(r'https?://\S+')
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = line.strip()
            if (not stripped
                    or set(stripped).issubset(set("─━─ "))
                    or url_pat.fullmatch(stripped)
                    or len(stripped) <= 3):
                out.append(line)
                continue
            urls = url_pat.findall(line)
            safe = url_pat.sub("URLPH", line)
            try:
                t = GoogleTranslator(source="en", target=target_lang).translate(safe)
                for url in urls:
                    t = t.replace("URLPH", url, 1)
                out.append(t if t else line)
            except Exception:
                out.append(line)
        return "\n".join(out)
    except Exception as e:
        print(f"⚠️  Translation error (→{target_lang}): {e}")
        return text

# ─────────────────────────────────────────────
# Session store  {phone: session_dict}
# ─────────────────────────────────────────────
SESSIONS = {}

def get_session(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {
            "state": "choose_language", "crop": None, "district": None,
            "weather": None, "prices": {}, "yield_loss": 0.0,
            "lang": None,  # None = not yet chosen
        }
    return SESSIONS[phone]

def reset_session(phone: str):
    lang = SESSIONS.get(phone, {}).get("lang", None)
    SESSIONS[phone] = {
        "state": "choose_language", "crop": None, "district": None,
        "weather": None, "prices": {}, "yield_loss": 0.0,
        "lang": lang,  # keep language preference on reset
    }

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def encode(col, val):
    if not META: return 0
    d = META["encoders"].get(col, {})
    return d.get(val, d.get(list(d.keys())[0], 0) if d else 0)

def get_price_stats(crop, district):
    if DF is None:
        return {"modal":3000,"min":2500,"max":3500,"lag1":3000,"lag2":3000,"lag4":3000,"lag8":3000,"lag12":3000}
    sub = DF[(DF["Commodity"] == crop) & (DF["District"] == district)]
    if sub.empty: sub = DF[DF["Commodity"] == crop]
    if sub.empty:
        return {"modal":3000,"min":2500,"max":3500,"lag1":3000,"lag2":3000,"lag4":3000,"lag8":3000,"lag12":3000}
    latest = sub.sort_values("Arrival_Date").iloc[-1]
    return {k: float(latest.get(v, d)) for k,v,d in [
        ("modal","Modal_Price",3000),("min","Min_Price",2500),("max","Max_Price",3500),
        ("lag1","Lag1",3000),("lag2","Lag2",3000),("lag4","Lag4",3000),
        ("lag8","Lag8",3000),("lag12","Lag12",3000),
    ]}

def build_price_row(crop, district, weather_data):
    dt = pd.to_datetime(date.today())
    month = dt.month; week = int(dt.isocalendar()[1])
    season = SEASON_MAP.get(month, "Monsoon")
    stats = get_price_stats(crop, district)
    min_p = stats["min"]; max_p = stats["max"]
    spread = round(max_p - min_p, 2); mid = round((max_p + min_p) / 2, 2)
    w = weather_data.get("avg_7day", {})
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code":  encode("District", district),
        "Market_Code":    encode("Market", f"{district} APMC"),
        "Variety_Code":   encode("Variety", "Local"),
        "Month": month, "Week": week,
        "Season_Code": SEASON_CODES.get(season, 0),
        "Min_Price": min_p, "Max_Price": max_p,
        "Price_Spread": spread, "Price_Mid": mid,
        "Spread_Ratio": round(spread / mid if mid > 0 else 0, 4),
        "Lag1": stats["lag1"], "Lag2": stats["lag2"], "Lag4": stats["lag4"],
        "Lag8": stats["lag8"], "Lag12": stats["lag12"],
        "Rainfall_mm": w.get("rainfall", 30), "Temp_Max_C": w.get("temp_max", 32),
        "Temp_Min_C": w.get("temp_min", 24), "Humidity_Pct": w.get("humidity", 60),
    }
    pf = META["price_features"] if META else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in pf}])

def build_yield_row(crop, district, weather_data):
    dt = pd.to_datetime(date.today())
    month = dt.month; season = SEASON_MAP.get(month, "Monsoon")
    w = weather_data.get("avg_7day", {})
    stats = get_price_stats(crop, district)
    row = {
        "Commodity_Code": encode("Commodity", crop),
        "District_Code": encode("District", district),
        "Month": month, "Season_Code": SEASON_CODES.get(season, 0),
        "Rainfall_mm": w.get("rainfall", 30), "Temp_Max_C": w.get("temp_max", 32),
        "Temp_Min_C": w.get("temp_min", 24), "Humidity_Pct": w.get("humidity", 60),
        "Lag1": stats["lag1"], "Lag4": stats["lag4"], "Lag8": stats["lag8"],
    }
    yf = META["yield_features"] if META else list(row.keys())
    return pd.DataFrame([{col: row.get(col, 0) for col in yf}])

def qtl_to_kg(p): return round(p / 100, 2) if p else 0

def trend_icon(base, compare):
    if not compare or not base: return "➡️"
    d = (compare - base) / base * 100 if base > 0 else 0
    return f"🔺+{d:.1f}%" if d > 2 else f"🔻{d:.1f}%" if d < -2 else f"➡️{d:+.1f}%"

def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return text.strip()

# ─────────────────────────────────────────────
# Intent parser
# ─────────────────────────────────────────────
CROP_KEYWORDS = {}

def build_crop_keywords():
    global CROP_KEYWORDS
    if DF is None: return
    for crop in DF["Commodity"].dropna().unique():
        CROP_KEYWORDS[crop.lower()] = crop
        CROP_KEYWORDS[crop.split()[0].lower()] = crop

def parse_intent(text: str, session: dict) -> str:
    t = text.strip().lower()
    state = session.get("state", "idle")

    if t in ("hi","hello","start","menu","help","/start"): return "greeting"
    if t in ("0","reset","restart","main menu"):            return "reset"
    if t in ("lang","language","change language"):         return "change_language"
    if t == "web" or t == "webapp" or t == "website":      return "show_webapp"

    if state == "choose_language":
        if t in LANG_BY_NUMBER:                             return "set_language"

    if state == "main_menu":
        if t in ("1","price","price check"):   return "price_check"
        if t in ("2","weather"):               return "weather_alert"
        if t in ("3","disease","ai advice"):   return "ai_advice"
        if t in ("4","chemical","chemicals"):  return "chemical_advice"
        if t in ("5","web","webapp","website","full web app","app"): return "show_webapp"
        

    if state == "post_prediction":
        if t in ("1","weather"):               return "weather_alert"
        if t in ("2","disease","ai advice"):   return "ai_advice"
        if t in ("3","chemical","chemicals"):  return "chemical_advice"
        if t in ("4","web","webapp","website","full report","full web app","app"): return "show_webapp"

    if state == "ask_problem":              return "problem_description"
    if state == "ask_chemical_problem":    return "chemical_problem_description"

    words = t.split()
    if state == "choose_crop":
        for w in words:
            if w in CROP_KEYWORDS: return "set_crop"
    if state == "choose_district":
        for d in TN_DISTRICTS:
            if t in d.lower() or d.lower().startswith(t): return "set_district"

    build_crop_keywords()
    found_crop, found_dist = None, None
    for w in words:
        if w in CROP_KEYWORDS: found_crop = CROP_KEYWORDS[w]
        for d in TN_DISTRICTS:
            if w == d.lower() or (d.lower().startswith(w) and len(w) >= 4):
                found_dist = d
    if found_crop: session["_inline_crop"] = found_crop
    if found_dist: session["_inline_dist"] = found_dist
    if found_crop or found_dist: return "inline_query"

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

def web_app_message(crop=None, district=None, context=""):
    """
    Sends a rich link to the Streamlit web app.
    If crop/district are known, includes a note that the full analysis
    is ready there.
    """
    if crop and district:
        detail = (
            f"Your *{crop}* analysis for *{district}* is available in full detail on the web app:\n\n"
            f"✅ Interactive price charts\n"
            f"✅ 7-day weather forecast\n"
            f"✅ Full AI advisor with 6 topic analysis\n"
            f"✅ Chemical dosage cards\n"
            f"✅ Market trend graphs\n"
        )
    else:
        detail = (
            "The full AgriAssist+ web app includes:\n\n"
            "✅ Price prediction with charts\n"
            "✅ Live weather for all 31 TN districts\n"
            "✅ AI crop advisor (6 topics)\n"
            "✅ Chemical advisor with dosage\n"
            "✅ Market trend graphs\n"
        )

    return (
        f"🌐 *AgriAssist+ Web App*\n"
        f"──────────────────────\n\n"
        f"{detail}\n"
        f"👉 *Open here:*\n"
        f"{WEB_APP_URL}\n\n"
        f"──────────────────────\n"
        f"0️⃣ Main menu"
    )

def welcome_message():
    return (
        "🌾 AgriAssist+ — Tamil Nadu Crop Intelligence\n"
        "Hello Farmer! 👋 I can help you with:\n"
        "1️⃣  Price prediction\n"
        "2️⃣  Weather alert\n"
        "3️⃣  Disease & AI advice\n"
        "4️⃣  Chemical recommendations\n"
        "5️⃣  Open full web app\n\n"
        "Reply with a number or type your crop & district.\n"
        "Example: tomato coimbatore\n\n"
        f"🌐 Full Web App (charts & more):\n"
        f"{WEB_APP_URL}\n\n"
        "Type 0 anytime to return to this menu."
    )

def ask_crop_message():
    crops = get_all_crops()
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
    cur = prices.get("current", 0)
    p1w = prices.get("1week", 0)
    p1m = prices.get("1month", 0)
    yl_icon = "🔴" if yield_loss > 25 else "🟠" if yield_loss > 10 else "🟢"

    price_trend_1m = ((p1m - cur) / cur * 100) if p1m and cur else 0
    price_trend_1w = ((p1w - cur) / cur * 100) if p1w and cur else 0

    if yield_loss > 25 and price_trend_1m <= 2:
        sell_icon = "🚨"; sell_decision = "SELL NOW"
        sell_reason = f"High yield risk ({yield_loss:.0f}%) + prices flat."
    elif yield_loss > 25 and price_trend_1m > 2:
        sell_icon = "⚠️"; sell_decision = "SELL WITHIN 1 WEEK"
        sell_reason = f"High yield risk ({yield_loss:.0f}%) — sell soon."
    elif price_trend_1m > 8 and yield_loss <= 15:
        sell_icon = "⏳"; sell_decision = "WAIT — HOLD FOR 1 MONTH"
        sell_reason = f"Prices rising {price_trend_1m:.1f}% → Rs.{qtl_to_kg(p1m):.2f}/kg. Low risk."
    elif price_trend_1w > 3 and yield_loss <= 20:
        sell_icon = "📅"; sell_decision = "WAIT — SELL IN 1 WEEK"
        sell_reason = f"Prices rising {price_trend_1w:.1f}% this week."
    else:
        sell_icon = "✅"; sell_decision = "SELL NOW — PRICES STABLE"
        sell_reason = "No major price upside forecast."

    return "\n".join([
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
        "─────────────────────",
        "What next?",
        "1️⃣ Weather alert",
        "2️⃣ Disease & AI advice",
        "3️⃣ Chemical tips",
        f"4️⃣ Full Web App (charts & more)",
        f"   {WEB_APP_URL}",
        "0️⃣ Main menu",
    ])

def weather_message(district, weather_data):
    if not weather_data or not weather_data.get("success"):
        return f"⚠️ Could not fetch weather for {district}. Please try again."
    cur = weather_data.get("current", {})
    avg = weather_data.get("avg_7day", {})
    alerts = []
    if avg.get("temp_max", 30) > 38: alerts.append("🌡️ Heat stress — check tomato, chilli, brinjal")
    if avg.get("rainfall", 20) > 150: alerts.append("🌊 Flood risk — ensure drainage")
    if avg.get("rainfall", 20) < 5:  alerts.append("🏜️ Drought risk — irrigate now")
    if avg.get("humidity", 60) > 80: alerts.append("🦠 Fungal risk — apply preventive fungicide")
    if not alerts: alerts.append("✅ Conditions favourable — no immediate alerts")

    return "\n".join([
        f"🌦️ *Weather — {district}*", "",
        f"🌡️ Temp:     {cur.get('temperature','N/A')} °C",
        f"💧 Humidity: {cur.get('humidity','N/A')} %",
        f"🌧️ Rainfall: {cur.get('precipitation', 0)} mm",
        f"💨 Wind:     {cur.get('wind_speed','N/A')} km/h",
        "", "⚠️ *Alerts:*",
    ] + [f"  • {a}" for a in alerts] + [
        "",
        "─────────────────────",
        "2️⃣ AI advice",
        "3️⃣ Chemicals",
        f"4️⃣ Full Web App (charts & more)",
        f"   {WEB_APP_URL}",
        "0️⃣ Main menu",
    ])

def ask_problem_message(crop, district):
    return (
        f"🤖 *AI Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe your crop problem:*\n"
        "Tell me the symptoms — affected parts, colour, how long.\n\n"
        "_Examples:_\n"
        "  • Leaves turning yellow with brown spots\n"
        "  • Fruits rotting before harvest\n\n"
        "Type your problem ↓"
    )

def ask_chemical_problem_message(crop, district):
    return (
        f"🧪 *Chemical Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe the disease or problem:*\n\n"
        "_Examples:_\n"
        "  • White powder on leaves and stems\n"
        "  • Insects eating the leaves\n\n"
        "Type your problem ↓"
    )

def ai_advice_message(crop, district, problem, weather, prices, yield_loss, phone="anon"):
    if RAG is None:
        return "⚠️ AI advisor not available right now.\n\n0️⃣ Main menu"
    ctx = {
        "crop": crop, "district": district,
        "weather": weather.get("avg_7day", {}) if weather else {},
        "prices": prices, "yield_loss_pct": yield_loss,
    }
    try:
        result = RAG.diagnose_problem(problem, ctx)
        solution = clean_html(result.get("solution", ""))
        recs     = clean_html(result.get("recommendations", ""))
        problem_preview = problem[:100] + ('...' if len(problem) > 100 else '')
        return "\n".join([
            f"🧠 *AI Analysis — {crop}*",
            f"📍 {district}",
            f"❓ _{problem_preview}_",
            "─────────────────────", "",
            "🔍 *What is happening:*",
            solution[:500], "",
            "─────────────────────",
            "✅ *What you should do:*",
            recs[:650], "",
            "─────────────────────",
            "3️⃣ Chemical tips",
            f"4️⃣ Full Web App (charts & more)",
            f"   {WEB_APP_URL}",
            "0️⃣ Main menu",
        ])[:4090]
    except Exception as e:
        return f"⚠️ AI advisor error: {e}\n\n0️⃣ Main menu"

def chemical_problem_message(crop, district, problem, weather, yield_loss, phone="anon"):
    if RAG is None:
        return "⚠️ Chemical advisor not available right now.\n\n0️⃣ Main menu"
    weather_avg = weather.get("avg_7day", {}) if weather else {}
    try:
        from rag_engine import recommend_chemicals
        chem_res = recommend_chemicals(crop, weather_avg, yield_loss)
        advice = clean_html(RAG.llm_chemical_advice(
            crop=crop, disease_input=problem, weather=weather_avg,
            chem_res=chem_res, yield_loss=yield_loss, district=district,
        ))
        problem_preview = problem[:90] + ('...' if len(problem) > 90 else '')
        return "\n".join([
            f"🧪 *Chemical Advisor — {crop}*",
            f"📍 {district}",
            f"❓ _{problem_preview}_",
            "─────────────────────", "",
            advice[:3600], "",
            "─────────────────────",
            "2️⃣ AI advice",
            f"4️⃣ Full Web App (charts & more)",
            f"   {WEB_APP_URL}",
            "0️⃣ Main menu",
        ])[:4090]
    except Exception as e:
        return f"⚠️ Chemical advisor error: {e}\n\n0️⃣ Main menu"

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
    prices = {}; yield_loss = 0.0
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
    session["prices"] = prices
    session["yield_loss"] = yield_loss
    try:
        from database import get_db
        db = get_db()
        if db.is_connected:
            sid = hashlib.md5(f"wa_{session.get('phone','anon')}".encode()).hexdigest()[:16]
            db.upsert_session(sid, crop, district)
            db.save_prediction(sid, {
                "crop": crop, "variety": "Local", "district": district,
                "market": f"{district} APMC", "prediction_date": str(date.today()),
                "current_price": round(prices.get("current", 0), 2),
                "price_1week":   round(prices.get("1week", 0), 2),
                "price_2week":   None,
                "price_1month":  round(prices.get("1month", 0), 2),
                "yield_loss_pct": round(yield_loss, 2), "yield_qty_qtl": 10,
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

    # ── Step 1: Language selection gate (first contact) ──────────
    # If lang not yet chosen, show menu — UNLESS farmer is picking a language right now
    t_stripped = text.strip()
    if session.get("lang") is None:
        if t_stripped in LANG_BY_NUMBER:
            # Farmer chose a language
            chosen = LANG_BY_NUMBER[t_stripped]
            session["lang"] = chosen
            session["state"] = "main_menu"
            return translate_reply(welcome_message(), chosen)
        elif t_stripped in ("lang", "language", "change language"):
            return language_menu()
        else:
            # Any other message → show language menu
            session["state"] = "choose_language"
            return language_menu()

    # ── Step 2: Change language intent (anytime) ─────────────────
    if t_stripped.lower() in ("lang", "language", "change language"):
        session["lang"] = None
        session["state"] = "choose_language"
        return language_menu()

    # ── Step 3: Detect & translate incoming message ───────────────
    lang = session["lang"]  # already set at this point
    detected_lang = detect_language(text)
    if detected_lang != "en" and detected_lang != lang:
        print(f"🌐 Detected language: {SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}")
        text_for_processing = translate_to_english(text, detected_lang)
        print(f"   Translated: {text!r} → {text_for_processing!r}")
    elif lang != "en":
        text_for_processing = translate_to_english(text, lang)
    else:
        text_for_processing = text

    intent = parse_intent(text_for_processing, session)

    if intent == "greeting":
        session["state"] = "main_menu"
        return translate_reply(welcome_message(), lang)

    if intent == "reset":
        reset_session(phone)
        SESSIONS[phone]["state"] = "main_menu"
        SESSIONS[phone]["lang"] = lang  # keep language preference after reset
        return translate_reply(welcome_message(), lang)

    if intent == "show_webapp":
        return translate_reply(web_app_message(session.get("crop"), session.get("district")), lang)


    if intent == "inline_query":
        crop = session.pop("_inline_crop", None) or session.get("crop")
        dist = session.pop("_inline_dist", None) or session.get("district")
        if crop and dist:
            session["crop"] = crop; session["district"] = dist
            weather, prices, yl = run_prediction(crop, dist, session)
            session["state"] = "post_prediction"
            return translate_reply(price_result_message(crop, dist, prices, yl), lang)
        if crop and not dist:
            session["crop"] = crop; session["state"] = "choose_district"
            return translate_reply(ask_district_message(crop), lang)
        if dist and not crop:
            session["district"] = dist; session["state"] = "choose_crop"
            return translate_reply(ask_crop_message(), lang)

    if intent == "price_check":
        if session.get("crop") and session.get("district"):
            weather, prices, yl = run_prediction(session["crop"], session["district"], session)
            session["state"] = "post_prediction"
            return translate_reply(price_result_message(session["crop"], session["district"], prices, yl), lang)
        session["state"] = "choose_crop"
        return translate_reply(ask_crop_message(), lang)

    if intent == "weather_alert":
        if session.get("district"):
            wd = session.get("weather") or get_weather(session["district"])
            session["weather"] = wd; session["state"] = "post_prediction"
            return translate_reply(weather_message(session["district"], wd), lang)
        session["state"] = "choose_district"
        return translate_reply("📍 *Which district?*\nType your district name (e.g. *Coimbatore*)", lang)

    if intent == "ai_advice":
        if session.get("crop") and session.get("district"):
            session["state"] = "ask_problem"
            return translate_reply(ask_problem_message(session["crop"], session["district"]), lang)
        session["state"] = "choose_crop"
        return translate_reply(ask_crop_message(), lang)

    if intent == "chemical_advice":
        if session.get("crop") and session.get("district"):
            session["state"] = "ask_chemical_problem"
            return translate_reply(ask_chemical_problem_message(session["crop"], session["district"]), lang)
        session["state"] = "choose_crop"
        return translate_reply(ask_crop_message(), lang)

    if session["state"] == "choose_crop":
        t = text_for_processing.strip().lower(); build_crop_keywords()
        for w in t.split():
            if w in CROP_KEYWORDS:
                session["crop"] = CROP_KEYWORDS[w]; session["state"] = "choose_district"
                return translate_reply(ask_district_message(session["crop"]), lang)
        return translate_reply(f"🌱 I didn't recognise that crop.\n\n{ask_crop_message()}", lang)

    if session["state"] == "choose_district":
        t = text_for_processing.strip().lower()
        for d in TN_DISTRICTS:
            if t in d.lower() or d.lower().startswith(t):
                session["district"] = d
                weather, prices, yl = run_prediction(session["crop"], d, session)
                session["state"] = "post_prediction"
                return translate_reply(price_result_message(session["crop"], d, prices, yl), lang)
        return translate_reply(f"📍 I didn't recognise that district.\n\n{ask_district_message(session.get('crop','your crop'))}", lang)

    if session["state"] == "ask_problem":
        if len(text_for_processing.strip()) < 5:
            return translate_reply("⚠️ Please describe the problem in more detail.\n\n_Example: Leaves turning yellow, white powder on stems_", lang)
        session["state"] = "post_prediction"
        return translate_reply(ai_advice_message(
            session["crop"], session["district"], text_for_processing.strip(),
            session.get("weather") or {}, session.get("prices", {}),
            session.get("yield_loss", 0), phone=phone,
        ), lang)

    if session["state"] == "ask_chemical_problem":
        if len(text_for_processing.strip()) < 5:
            return translate_reply("⚠️ Please describe the problem in more detail.\n\n_Example: White powder on leaves, insects eating stems_", lang)
        if not session.get("weather"):
            session["weather"] = get_weather(session["district"])
        session["state"] = "post_prediction"
        return translate_reply(chemical_problem_message(
            session["crop"], session["district"], text_for_processing.strip(),
            session.get("weather") or {}, session.get("yield_loss", 0), phone=phone,
        ), lang)

    if session["state"] in ("idle", "main_menu", "post_prediction"):
        return translate_reply(unknown_message(), lang)

    return translate_reply(unknown_message(), lang)

# ─────────────────────────────────────────────
# WhatsApp API — send message
# ─────────────────────────────────────────────
def send_whatsapp_message(to: str, body: str):
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print(f"[MOCK] To {to}:\n{body}\n"); return
    payload = {
        "messaging_product": "whatsapp", "recipient_type": "individual",
        "to": to, "type": "text", "text": {"preview_url": False, "body": body},
    }
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(WHATSAPP_API_URL, json=payload, headers=headers, timeout=15)
        if r.status_code == 401:
            print("❌ ERROR 401 — Token expired. Regenerate at developers.facebook.com"); return
        if r.status_code == 403:
            print("❌ ERROR 403 — Permission denied. Check token permissions."); return
        r.raise_for_status()
        print(f"✅ Message sent to {to}")
    except Exception as e:
        print(f"❌ Failed to send to {to}: {e}")

# ─────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print("✅ Webhook verified."); return challenge, 200
    return "Forbidden", 403

@app.route("/", methods=["GET"])
def health_check():
    return f"AgriAssist+ WhatsApp Bot ✅ | Web App: {WEB_APP_URL}", 200

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json(silent=True)
    if not data: return jsonify({"status": "no data"}), 400
    try:
        value = data["entry"][0]["changes"][0]["value"]
        if "messages" not in value: return jsonify({"status": "ignored"}), 200
        msg = value["messages"][0]
        phone = msg["from"]; msg_type = msg.get("type", "")
        if msg_type == "text":
            text = msg["text"]["body"]
        elif msg_type == "interactive":
            it = msg["interactive"]
            text = it.get("button_reply", it.get("list_reply", {})).get("id", "")
        else:
            send_whatsapp_message(phone, "Sorry, I only understand text messages right now.")
            return jsonify({"status": "unsupported"}), 200
        print(f"📩 From {phone}: {text!r}")
        reply = handle_message(phone, text)
        send_whatsapp_message(phone, reply)
    except (KeyError, IndexError) as e:
        print(f"❌ Parse error: {e}")
    return jsonify({"status": "ok"}), 200

# ─────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────
def cli_test():
    print("\n" + "="*55)
    print("  AgriAssist+ WhatsApp Bot — CLI Test Mode")
    print("="*55)
    print(f"Web App URL: {WEB_APP_URL}")
    print("Type 'quit' to exit.\n")
    build_crop_keywords()
    phone = "test_farmer_1234"
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() == "quit": break
        reply = handle_message(phone, text)
        print(f"\nBot:\n{reply}\n" + "-"*45)

if __name__ == "__main__":
    import sys
    build_crop_keywords()
    if "--cli" in sys.argv:
        cli_test()
    else:
        if not WHATSAPP_TOKEN:
            print("⚠️  WHATSAPP_TOKEN not set — bot will receive but cannot send messages.")
        print(f"🚀 Starting AgriAssist+ WhatsApp bot on port 8000…")
        print(f"   Web App URL configured: {WEB_APP_URL}")
        app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
