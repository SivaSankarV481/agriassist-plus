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

TOKEN MANAGEMENT:
    Meta temporary tokens expire every 24 hours.
    To update without restarting the bot, use the /token endpoint:
        curl -X POST http://localhost:8000/token \
             -H "Content-Type: application/json" \
             -d '{"token": "YOUR_NEW_TOKEN_HERE"}'
    Or just edit .env and restart the bot.
    For a permanent token: see https://developers.facebook.com → Business Settings → System Users
"""

import os
import re
import joblib
import requests
import pandas as pd
from datetime import datetime
from datetime import date
from flask import Flask, request, jsonify
from dotenv import load_dotenv, set_key

load_dotenv()

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

# ─────────────────────────────────────────────
# Base directory
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

# ─────────────────────────────────────────────
# Token manager — supports hot-reload without restart
# ─────────────────────────────────────────────
class TokenManager:
    """
    Manages the WhatsApp API token.
    - Stores the active token in memory
    - Can hot-reload from .env without restarting the bot
    - Detects 401 expired token errors and prints clear instructions
    - Provides update_token() to change token at runtime via /token endpoint
    """
    def __init__(self):
        self._token = os.getenv("WHATSAPP_TOKEN", "").strip()
        self.phone_number_id = os.getenv("PHONE_NUMBER_ID", "").strip()
        self.verify_token    = os.getenv("WHATSAPP_VERIFY_TOKEN", "agriassist_verify")
        self.api_url = f"https://graph.facebook.com/v21.0/{self.phone_number_id}/messages"
        self.media_url = "https://graph.facebook.com/v21.0/{media_id}"
        self._token_valid = None  # None=unchecked, True=valid, False=invalid

    @property
    def token(self) -> str:
        return self._token

    def is_configured(self) -> bool:
        return bool(self._token and self.phone_number_id)

    def is_placeholder(self) -> bool:
        t = self._token.strip()
        return not t or t in ("PASTE_YOUR_NEW_TOKEN_HERE", "your_token_here", "")

    def reload_from_env(self) -> bool:
        """Re-read token from .env file without restarting."""
        load_dotenv(ENV_FILE, override=True)
        new_token = os.getenv("WHATSAPP_TOKEN", "").strip()
        if new_token and new_token != self._token:
            self._token = new_token
            self._token_valid = None
            print(f"🔄 Token reloaded from .env ({new_token[:20]}…)")
            return True
        return False

    def update_token(self, new_token: str) -> bool:
        """Update token at runtime AND persist to .env file."""
        new_token = new_token.strip()
        if not new_token:
            return False
        self._token = new_token
        self._token_valid = None
        # Persist to .env so it survives restarts
        try:
            set_key(ENV_FILE, "WHATSAPP_TOKEN", new_token)
            print(f"✅ Token updated in memory and saved to .env ({new_token[:20]}…)")
        except Exception as e:
            print(f"⚠️  Could not save token to .env: {e} — token updated in memory only")
        return True

    def auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def handle_401(self):
        """Print clear instructions when token expires."""
        self._token_valid = False
        print("\n" + "="*65)
        print("  ❌  WHATSAPP TOKEN EXPIRED — ACTION REQUIRED")
        print("="*65)
        print("""
OPTION A — Quick fix (get fresh 24-hour token):
  1. Open: https://developers.facebook.com/apps/
  2. Your App → WhatsApp → API Setup
  3. Copy the "Temporary access token"
  4. Paste into .env as WHATSAPP_TOKEN=<token>
  5. Either restart the bot OR call the /token endpoint:
       POST http://localhost:8000/token
       Body: {"token": "YOUR_NEW_TOKEN_HERE"}

OPTION B — Permanent token (NEVER expires — do this once):
  1. Open: https://business.facebook.com/settings/
  2. Left menu → Users → System Users → Add
  3. Create System User with Admin role
  4. Click "Generate New Token" → select your app
  5. Permissions needed:
       ✅ whatsapp_business_messaging
       ✅ whatsapp_business_management
  6. Copy the token → paste into .env → restart bot
  This token NEVER expires.
""")
        print("="*65 + "\n")

# Global token manager instance
TM = TokenManager()

# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
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
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            MODELS[key] = joblib.load(fpath)
    meta_path = os.path.join(BASE_DIR, "model_meta.joblib")
    data_path = os.path.join(BASE_DIR, "tn_agri_dataset.csv")
    META = joblib.load(meta_path) if os.path.exists(meta_path) else None
    DF   = pd.read_csv(data_path) if os.path.exists(data_path) else None
    from rag_engine import RAGEngine
    RAG = RAGEngine()
    print("✅ Models, dataset and RAGEngine loaded.")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

# ── Load CNN plant disease detector ─────────────────────────────
DETECTOR = None
try:
    from plant_disease_cnn import PlantDiseaseDetector
    DETECTOR = PlantDiseaseDetector(os.path.join(BASE_DIR, "tomato_early_blight_model.h5"))
    cnn_mode = "CNN + Claude Vision" if DETECTOR.cnn_available else "Claude Vision only"
    print(f"✅ Plant disease detector loaded ({cnn_mode})")
except Exception as e:
    print(f"⚠️ Disease detector load error: {e}")

from weather import get_weather
from database import get_db

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

SUPPORTED_LANGUAGES = {
    "en": ("English",   "1"),
    "ta": ("Tamil",     "2"),
    "hi": ("Hindi",     "3"),
    "te": ("Telugu",    "4"),
    "kn": ("Kannada",   "5"),
    "ml": ("Malayalam", "6"),
}
LANG_BY_NUMBER = {"1":"en", "2":"ta", "3":"hi", "4":"te", "5":"kn", "6":"ml"}

WEB_APP_URL = os.getenv("WEB_APP_URL", "https://agri-assist481.streamlit.app")

# ─────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────
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
    if not LANGDETECT_AVAILABLE or len(text.strip()) < 5:
        return "en"
    try:
        lang = _langdetect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except Exception:
        return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    if not TRANSLATION_AVAILABLE or source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as e:
        print(f"⚠️  Translation error (→EN): {e}")
        return text

def translate_reply(text: str, target_lang: str) -> str:
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
                out.append(line); continue
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
# Session store
# ─────────────────────────────────────────────
SESSIONS = {}

def get_session(phone: str) -> dict:
    if phone not in SESSIONS:
        db = get_db()
        session_id = db.get_or_create_session_id(phone)
        SESSIONS[phone] = {
            "state":      "choose_language",
            "session_id": session_id,
            "crop":       None,
            "district":   None,
            "weather":    None,
            "prices":     {},
            "yield_loss": 0.0,
            "lang":       None,
        }
    return SESSIONS[phone]

def reset_session(phone: str):
    existing   = SESSIONS.get(phone, {})
    lang       = existing.get("lang", None)
    session_id = existing.get("session_id")
    if session_id is None:
        session_id = get_db().get_or_create_session_id(phone)
    SESSIONS[phone] = {
        "state":      "choose_language",
        "session_id": session_id,
        "crop":       None,
        "district":   None,
        "weather":    None,
        "prices":     {},
        "yield_loss": 0.0,
        "lang":       lang,
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
        "District_Code":  encode("District", district),
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
# WhatsApp API — send message (with retry + 401 handling)
# ─────────────────────────────────────────────
def send_whatsapp_message(to: str, body: str):
    """Send a WhatsApp text message. Handles 401, 429, timeouts with clear logging."""
    if not TM.is_configured():
        print(f"[MOCK — token not configured] To {to}:\n{body}\n")
        return

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type":    "individual",
        "to":                to,
        "type":              "text",
        "text":              {"preview_url": False, "body": body},
    }

    for attempt in range(1, 3):   # max 2 attempts
        try:
            r = requests.post(
                TM.api_url,
                json=payload,
                headers=TM.auth_headers(),
                timeout=15,
            )

            if r.status_code == 401:
                TM.handle_401()
                return   # No point retrying — token is expired

            if r.status_code == 403:
                print(f"❌ ERROR 403 — Permission denied. Check token permissions.\nResponse: {r.text}")
                return

            if r.status_code == 429:
                import time
                wait = 2 ** attempt
                print(f"⚠️  Rate limited (429). Waiting {wait}s before retry {attempt}…")
                time.sleep(wait)
                continue

            if not r.ok:
                print(f"❌ ERROR {r.status_code} sending to {to}:\n{r.text}")
                if attempt < 2:
                    continue
                return

            print(f"✅ Sent to {to}")
            return

        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout on attempt {attempt}/2 sending to {to}")
            if attempt < 2:
                continue
        except Exception as e:
            print(f"❌ Exception sending to {to}: {e}")
            return

    print(f"❌ All send attempts failed for {to}")

# ─────────────────────────────────────────────
# Image handler — WhatsApp photo → CNN/Claude disease detection
# ─────────────────────────────────────────────
def handle_image_message(phone: str, media_id: str, caption: str = "") -> str:
    session = get_session(phone)
    lang    = session.get("lang", "en") or "en"

    if DETECTOR is None:
        return translate_reply(
            "📷 Image received!\n\n"
            "⚠️ Disease detection module not loaded.\n\n"
            "Please describe your crop problem in text instead:\n"
            "_Example: My tomato leaves have yellow spots_\n\n"
            "0️⃣ Main menu",
            lang
        )

    # Get media download URL from Meta
    try:
        meta_resp = requests.get(
            TM.media_url.format(media_id=media_id),
            headers={"Authorization": f"Bearer {TM.token}"},
            timeout=15,
        )
        if meta_resp.status_code == 401:
            TM.handle_401()
            return translate_reply(
                "⚠️ Bot token expired — could not download your image.\n"
                "The bot owner needs to update the WhatsApp token.\n\n"
                "Please describe your problem in text for now.\n\n0️⃣ Main menu",
                lang
            )
        meta_resp.raise_for_status()
        media_url = meta_resp.json().get("url", "")
        if not media_url:
            raise ValueError("No URL returned from Meta media API")
    except Exception as e:
        print(f"❌ Media URL fetch error: {e}")
        return translate_reply(
            "📷 Image received but could not be downloaded. Please try again.\n\n0️⃣ Main menu",
            lang
        )

    # Run disease detection
    crop_hint = session.get("crop", "") or ""
    try:
        result = DETECTOR.predict_from_url(
            url=media_url,
            headers={"Authorization": f"Bearer {TM.token}"},
            crop_hint=crop_hint,
        )
    except Exception as e:
        print(f"❌ Disease detection error: {e}")
        return translate_reply(
            "📷 Image analysis failed. Please try a clearer, well-lit photo.\n\n0️⃣ Main menu",
            lang
        )

    # Get weather context for spray recommendations
    weather_ctx = {}
    district = session.get("district", "")
    if district:
        try:
            wd = get_weather(district)
            if wd.get("success"):
                weather_ctx = wd.get("avg_7day", {})
        except Exception:
            pass

    reply = DETECTOR.format_whatsapp_reply(result, weather=weather_ctx, prices=session.get("prices", {}))

    # Auto-set crop in session if identified
    identified_crop = result.get("crop_identified", "")
    if identified_crop and identified_crop not in ("Unknown", "Plant") and not session.get("crop"):
        if DF is not None:
            for known_crop in DF["Commodity"].dropna().unique():
                if identified_crop.lower() in known_crop.lower() or known_crop.lower() in identified_crop.lower():
                    session["crop"] = known_crop
                    break

    return translate_reply(reply, lang)

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
    t     = text.strip().lower()
    state = session.get("state", "idle")

    if t in ("hi","hello","start","menu","help","/start"): return "greeting"
    if t in ("0","reset","restart","main menu"):            return "reset"
    if t in ("lang","language","change language"):          return "change_language"
    if t in ("web","webapp","website"):                     return "show_webapp"

    if state == "choose_language" and t in LANG_BY_NUMBER: return "set_language"

    if state == "main_menu":
        if t in ("1","price","price check"):                return "price_check"
        if t in ("2","weather"):                            return "weather_alert"
        if t in ("3","disease","ai advice"):                return "ai_advice"
        if t in ("4","chemical","chemicals"):               return "chemical_advice"
        if t in ("5","web","webapp","website","full web app","app"): return "show_webapp"

    if state == "post_prediction":
        if t in ("1","weather"):                            return "weather_alert"
        if t in ("2","disease","ai advice"):                return "ai_advice"
        if t in ("3","chemical","chemicals"):               return "chemical_advice"
        if t in ("4","web","webapp","website","full report","full web app","app"): return "show_webapp"

    if state == "ask_problem":          return "problem_description"
    if state == "ask_chemical_problem": return "chemical_problem_description"

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

def web_app_message(crop=None, district=None):
    detail = (
        f"Your *{crop}* analysis for *{district}* is on the web app:\n\n"
        if crop and district else
        "The full AgriAssist+ web app includes:\n\n"
    ) + (
        "✅ Interactive price charts\n"
        "✅ 7-day weather forecast\n"
        "✅ Full AI advisor (6 topics)\n"
        "✅ Chemical dosage cards\n"
        "✅ Market trend graphs\n"
        "✅ 📷 Plant Disease Detection (photo upload)\n"
    )
    return (
        f"🌐 *AgriAssist+ Web App*\n──────────────────────\n\n"
        f"{detail}\n👉 *Open here:*\n{WEB_APP_URL}\n\n"
        f"──────────────────────\n0️⃣ Main menu"
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
        "📷 *Send a photo* of your plant to identify disease!\n\n"
        "Reply with a number or type: crop + district\n"
        "Example: *tomato coimbatore*\n\n"
        f"🌐 Web App: {WEB_APP_URL}\n\n"
        "Type *0* anytime to return to this menu."
    )

def ask_crop_message():
    crops  = get_all_crops()
    sample = ", ".join(crops[:12]) + "…"
    return f"🌱 *Which crop?*\n\nAvailable: {sample}\n\nJust type the crop name (e.g. *Tomato*)"

def ask_district_message(crop):
    sample = ", ".join(TN_DISTRICTS[:8]) + "…"
    return (
        f"📍 *Which district?* (for {crop})\n\n"
        f"Available: {sample}\n\n"
        "Type your district name (e.g. *Coimbatore*)"
    )

def price_result_message(crop, district, prices, yield_loss):
    cur = prices.get("current", 0)
    p1w = prices.get("1week",  0)
    p2w = prices.get("2week",  0)
    p1m = prices.get("1month", 0)
    yl_icon = "🔴" if yield_loss > 25 else "🟠" if yield_loss > 10 else "🟢"

    pt1m = ((p1m - cur) / cur * 100) if p1m and cur else 0
    pt1w = ((p1w - cur) / cur * 100) if p1w and cur else 0

    if   yield_loss > 25 and pt1m <= 2:  sell_icon = "🚨"; sell_d = "SELL NOW";           sell_r = f"High yield risk ({yield_loss:.0f}%) + prices flat."
    elif yield_loss > 25 and pt1m > 2:   sell_icon = "⚠️"; sell_d = "SELL WITHIN 1 WEEK"; sell_r = f"High yield risk ({yield_loss:.0f}%) — sell soon."
    elif pt1m > 8 and yield_loss <= 15:  sell_icon = "⏳"; sell_d = "WAIT — HOLD 1 MONTH";sell_r = f"Prices rising {pt1m:.1f}% → Rs.{qtl_to_kg(p1m):.2f}/kg. Low risk."
    elif pt1w > 3 and yield_loss <= 20:  sell_icon = "📅"; sell_d = "WAIT — SELL IN 1 WK"; sell_r = f"Prices rising {pt1w:.1f}% this week."
    else:                                 sell_icon = "✅"; sell_d = "SELL NOW — STABLE";   sell_r = "No major price upside forecast."

    return "\n".join([
        f"📊 *{crop} — {district}*",
        f"📅 {date.today().strftime('%d %b %Y')}",
        "",
        f"💰 Current:  *Rs.{qtl_to_kg(cur):.2f}/kg*",
        f"📅 1 Week:   *Rs.{qtl_to_kg(p1w):.2f}/kg* {trend_icon(cur, p1w)}",
        f"📅 2 Weeks:  *Rs.{qtl_to_kg(p2w):.2f}/kg* {trend_icon(cur, p2w)}",
        f"📅 1 Month:  *Rs.{qtl_to_kg(p1m):.2f}/kg* {trend_icon(cur, p1m)}",
        "",
        f"{yl_icon} Yield risk: *{yield_loss:.1f}%*",
        "",
        f"{sell_icon} *{sell_d}*",
        f"   {sell_r}",
        "",
        "─────────────────────",
        "What next?",
        "1️⃣ Weather alert",
        "2️⃣ Disease & AI advice",
        "3️⃣ Chemical tips",
        "4️⃣ Full Web App",
        f"   {WEB_APP_URL}",
        "📷 Send a plant photo for disease detection",
        "0️⃣ Main menu",
    ])

def weather_message(district, weather_data):
    if not weather_data or not weather_data.get("success"):
        return f"⚠️ Could not fetch weather for {district}. Please try again."
    cur = weather_data.get("current", {})
    avg = weather_data.get("avg_7day", {})
    alerts = []
    if avg.get("temp_max", 30) > 38:   alerts.append("🌡️ Heat stress — check tomato, chilli, brinjal")
    if avg.get("rainfall", 20) > 150:  alerts.append("🌊 Flood risk — ensure drainage")
    if avg.get("rainfall", 20) < 5:    alerts.append("🏜️ Drought risk — irrigate now")
    if avg.get("humidity", 60) > 80:   alerts.append("🦠 Fungal risk — apply preventive fungicide")
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
        "2️⃣ AI advice  3️⃣ Chemicals",
        "4️⃣ Full Web App",
        f"   {WEB_APP_URL}",
        "📷 Send a plant photo",
        "0️⃣ Main menu",
    ])

def ask_problem_message(crop, district):
    return (
        f"🤖 *AI Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe your crop problem:*\n"
        "Symptoms — affected parts, colour, how long.\n\n"
        "_Examples:_\n"
        "  • Leaves turning yellow with brown spots\n"
        "  • Fruits rotting before harvest\n\n"
        "💡 *Or send a 📷 photo* — I'll identify the disease automatically!\n\n"
        "Type your problem ↓"
    )

def ask_chemical_problem_message(crop, district):
    return (
        f"🧪 *Chemical Advisor — {crop}, {district}*\n\n"
        "🔍 *Describe the disease or problem:*\n\n"
        "_Examples:_\n"
        "  • White powder on leaves and stems\n"
        "  • Insects eating the leaves\n\n"
        "💡 *Or send a 📷 photo* of the affected plant!\n\n"
        "Type your problem ↓"
    )

def ai_advice_message(crop, district, problem, weather, prices, yield_loss, session_id="anon"):
    if RAG is None:
        return "⚠️ AI advisor not available right now.\n\n0️⃣ Main menu"
    ctx = {
        "crop": crop, "district": district, "session_id": session_id,
        "weather": weather.get("avg_7day", {}) if weather else {},
        "prices": prices, "yield_loss_pct": yield_loss,
    }
    try:
        result   = RAG.diagnose_problem(problem, ctx)
        solution = clean_html(result.get("solution", ""))
        recs     = clean_html(result.get("recommendations", ""))
        preview  = problem[:100] + ('...' if len(problem) > 100 else '')
        return "\n".join([
            f"🧠 *AI Analysis — {crop}*",
            f"📍 {district}", f"❓ _{preview}_",
            "─────────────────────", "",
            "🔍 *What is happening:*", solution[:500], "",
            "─────────────────────",
            "✅ *What you should do:*", recs[:650], "",
            "─────────────────────",
            "3️⃣ Chemical tips  4️⃣ Web App",
            f"   {WEB_APP_URL}",
            "📷 Send a plant photo",
            "0️⃣ Main menu",
        ])[:4090]
    except Exception as e:
        return f"⚠️ AI advisor error: {e}\n\n0️⃣ Main menu"

def chemical_problem_message(crop, district, problem, weather, yield_loss, session_id="anon"):
    if RAG is None:
        return "⚠️ Chemical advisor not available right now.\n\n0️⃣ Main menu"
    weather_avg = weather.get("avg_7day", {}) if weather else {}
    try:
        from rag_engine import recommend_chemicals
        chem_res = recommend_chemicals(crop, weather_avg, yield_loss)
        advice   = clean_html(RAG.llm_chemical_advice(
            crop=crop, disease_input=problem, weather=weather_avg,
            chem_res=chem_res, yield_loss=yield_loss, district=district,
        ))
        preview = problem[:90] + ('...' if len(problem) > 90 else '')
        return "\n".join([
            f"🧪 *Chemical Advisor — {crop}*",
            f"📍 {district}", f"❓ _{preview}_",
            "─────────────────────", "",
            advice[:3600], "",
            "─────────────────────",
            "2️⃣ AI advice  4️⃣ Web App",
            f"   {WEB_APP_URL}",
            "📷 Send a plant photo",
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
        "• 📷 Send a photo to identify plant disease\n"
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
                "1week":   float(MODELS["1week"].predict(X_price)[0])   if "1week"  in MODELS else 0,
                "2week":   float(MODELS["2week"].predict(X_price)[0])   if "2week"  in MODELS else 0,
                "1month":  float(MODELS["1month"].predict(X_price)[0])  if "1month" in MODELS else 0,
            }
            yield_loss = float(MODELS["yield_loss"].predict(X_yield)[0]) if "yield_loss" in MODELS else 0.0
        except Exception as e:
            print(f"Prediction error: {e}")
    session["prices"]     = prices
    session["yield_loss"] = yield_loss

    session_id = session.get("session_id", "unknown")
    try:
        db = get_db()
        if db.is_connected:
            db.upsert_session(session_id, crop, district)
            db.save_prediction(session_id, {
                "crop": crop, "variety": "Local", "district": district,
                "market": f"{district} APMC", "prediction_date": str(date.today()),
                "current_price":  round(prices.get("current", 0), 2),
                "price_1week":    round(prices.get("1week", 0), 2),
                "price_2week":    None,
                "price_1month":   round(prices.get("1month", 0), 2),
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
    session   = get_session(phone)
    t_stripped = text.strip()

    # ── Language gate ─────────────────────────────────────────
    if session.get("lang") is None:
        session["state"] = "choose_language"
        if t_stripped in LANG_BY_NUMBER:
            chosen = LANG_BY_NUMBER[t_stripped]
            session["lang"] = chosen
            session["state"] = "main_menu"
            return translate_reply(welcome_message(), chosen)
        if t_stripped.lower() in ("lang","language","change language"):
            return language_menu()
        return language_menu()

    # ── Change language ───────────────────────────────────────
    if t_stripped.lower() in ("lang","language","change language"):
        session["lang"] = None
        session["state"] = "choose_language"
        return language_menu()

    lang = session["lang"]

    # ── Translate to English for processing ───────────────────
    detected = detect_language(text)
    if detected != "en" and detected != lang:
        text_en = translate_to_english(text, detected)
    elif lang != "en":
        text_en = translate_to_english(text, lang)
    else:
        text_en = text

    intent     = parse_intent(text_en, session)
    session_id = session["session_id"]

    if intent == "greeting":
        session["state"] = "main_menu"
        return translate_reply(welcome_message(), lang)

    if intent == "reset":
        reset_session(phone)
        SESSIONS[phone]["state"] = "main_menu"
        SESSIONS[phone]["lang"]  = lang
        return translate_reply(welcome_message(), lang)

    if intent == "set_language":
        chosen = LANG_BY_NUMBER.get(text.strip(), "en")
        session["lang"] = chosen
        session["state"] = "main_menu"
        return translate_reply(welcome_message(), chosen)

    if intent == "show_webapp":
        return translate_reply(web_app_message(session.get("crop"), session.get("district")), lang)

    if intent == "inline_query":
        crop = session.pop("_inline_crop", None) or session.get("crop")
        dist = session.pop("_inline_dist", None) or session.get("district")
        if crop and dist:
            session["crop"] = crop; session["district"] = dist
            w, p, yl = run_prediction(crop, dist, session)
            session["state"] = "post_prediction"
            return translate_reply(price_result_message(crop, dist, p, yl), lang)
        if crop and not dist:
            session["crop"] = crop; session["state"] = "choose_district"
            return translate_reply(ask_district_message(crop), lang)
        if dist and not crop:
            session["district"] = dist; session["state"] = "choose_crop"
            return translate_reply(ask_crop_message(), lang)

    if intent == "price_check":
        if session.get("crop") and session.get("district"):
            w, p, yl = run_prediction(session["crop"], session["district"], session)
            session["state"] = "post_prediction"
            return translate_reply(price_result_message(session["crop"], session["district"], p, yl), lang)
        session["state"] = "choose_crop"
        return translate_reply(ask_crop_message(), lang)

    if intent == "weather_alert":
        if session.get("district"):
            wd = session.get("weather") or get_weather(session["district"])
            session["weather"] = wd; session["state"] = "post_prediction"
            return translate_reply(weather_message(session["district"], wd), lang)
        session["state"] = "choose_district"
        return translate_reply("📍 *Which district?*\nType district name (e.g. *Coimbatore*)", lang)

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
        build_crop_keywords()
        for w in text_en.strip().lower().split():
            if w in CROP_KEYWORDS:
                session["crop"] = CROP_KEYWORDS[w]; session["state"] = "choose_district"
                return translate_reply(ask_district_message(session["crop"]), lang)
        return translate_reply(f"🌱 Crop not recognised.\n\n{ask_crop_message()}", lang)

    if session["state"] == "choose_district":
        t = text_en.strip().lower()
        for d in TN_DISTRICTS:
            if t in d.lower() or d.lower().startswith(t):
                session["district"] = d
                w, p, yl = run_prediction(session["crop"], d, session)
                session["state"] = "post_prediction"
                return translate_reply(price_result_message(session["crop"], d, p, yl), lang)
        return translate_reply(f"📍 District not recognised.\n\n{ask_district_message(session.get('crop','your crop'))}", lang)

    if session["state"] == "ask_problem":
        if len(text_en.strip()) < 5:
            return translate_reply("⚠️ Please describe the problem in more detail.\nOr send a 📷 photo!", lang)
        session["state"] = "post_prediction"
        return translate_reply(ai_advice_message(
            session["crop"], session["district"], text_en.strip(),
            session.get("weather") or {}, session.get("prices", {}),
            session.get("yield_loss", 0), session_id=session_id,
        ), lang)

    if session["state"] == "ask_chemical_problem":
        if len(text_en.strip()) < 5:
            return translate_reply("⚠️ Please describe the problem in more detail.\nOr send a 📷 photo!", lang)
        if not session.get("weather"):
            session["weather"] = get_weather(session["district"])
        session["state"] = "post_prediction"
        return translate_reply(chemical_problem_message(
            session["crop"], session["district"], text_en.strip(),
            session.get("weather") or {}, session.get("yield_loss", 0),
            session_id=session_id,
        ), lang)

    return translate_reply(unknown_message(), lang)

# ─────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == TM.verify_token:
        print("✅ Webhook verified.")
        return challenge, 200
    return "Forbidden", 403

@app.route("/", methods=["GET"])
def health_check():
    cnn_status  = "CNN+Vision" if (DETECTOR and DETECTOR.cnn_available) else "Vision only"
    token_short = (TM.token[:12] + "…") if TM.token else "NOT SET"
    return (
        f"AgriAssist+ WhatsApp Bot ✅\n"
        f"Disease Detector: {cnn_status}\n"
        f"Token: {token_short}\n"
        f"Web App: {WEB_APP_URL}"
    ), 200

@app.route("/status", methods=["GET"])
def status():
    """Quick token validity check — GET http://localhost:8000/status"""
    if not TM.is_configured():
        return jsonify({"status": "error", "message": "Token or Phone Number ID not configured"}), 400
    if TM.is_placeholder():
        return jsonify({"status": "error", "message": "Token is still the placeholder — replace PASTE_YOUR_NEW_TOKEN_HERE"}), 400
    # Ping Meta API to verify token
    try:
        r = requests.get(
            f"https://graph.facebook.com/v21.0/{TM.phone_number_id}",
            headers={"Authorization": f"Bearer {TM.token}"},
            timeout=10,
        )
        if r.status_code == 200:
            return jsonify({
                "status":       "ok",
                "token_valid":  True,
                "token_prefix": TM.token[:15] + "…",
                "phone_number_id": TM.phone_number_id,
            })
        elif r.status_code == 401:
            return jsonify({
                "status":      "error",
                "token_valid": False,
                "message":     "Token expired — get a new one from developers.facebook.com",
                "meta_error":  r.json(),
            }), 401
        else:
            return jsonify({
                "status":      "warning",
                "http_status": r.status_code,
                "meta_response": r.json(),
            }), r.status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/token", methods=["POST"])
def update_token():
    """
    Hot-update the WhatsApp token without restarting.

    Usage:
        curl -X POST http://localhost:8000/token \\
             -H "Content-Type: application/json" \\
             -d '{"token": "YOUR_NEW_TOKEN_HERE"}'

    Or with the verify_token as a security check:
        curl -X POST http://localhost:8000/token \\
             -H "Content-Type: application/json" \\
             -d '{"token": "EAAxx...", "verify": "agriassist_verify"}'
    """
    data = request.get_json(silent=True) or {}
    new_token = data.get("token", "").strip()

    if not new_token:
        return jsonify({"status": "error", "message": "No token provided. Send: {\"token\": \"your_token_here\"}"}), 400

    # Optional security: require the verify token to prevent unauthorized updates
    provided_verify = data.get("verify", "")
    if provided_verify and provided_verify != TM.verify_token:
        return jsonify({"status": "error", "message": "Invalid verify token"}), 403

    if TM.update_token(new_token):
        return jsonify({
            "status":  "ok",
            "message": "Token updated successfully. Bot will use new token immediately.",
            "token_prefix": new_token[:15] + "…",
        })
    return jsonify({"status": "error", "message": "Failed to update token"}), 500

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json(silent=True)
    if not data: return jsonify({"status": "no data"}), 400
    try:
        value = data["entry"][0]["changes"][0]["value"]
        if "messages" not in value: return jsonify({"status": "ignored"}), 200

        msg      = value["messages"][0]
        phone    = msg["from"]
        msg_type = msg.get("type", "")

        if msg_type == "text":
            text = msg["text"]["body"]
            print(f"📩 {phone}: {text!r}")
            reply = handle_message(phone, text)
            send_whatsapp_message(phone, reply)

        elif msg_type == "image":
            image_data = msg.get("image", {})
            media_id   = image_data.get("id", "")
            caption    = image_data.get("caption", "")
            print(f"📷 Image from {phone} — media_id={media_id}")

            if media_id:
                session = get_session(phone)
                lang    = session.get("lang", "en") or "en"
                send_whatsapp_message(phone, translate_reply(
                    "📷 Photo received! Analyzing your plant...\n⏳ Please wait 10–20 seconds.",
                    lang
                ))
                reply = handle_image_message(phone, media_id, caption)
                send_whatsapp_message(phone, reply)
            else:
                send_whatsapp_message(phone, "📷 Image received but could not be processed. Please try again.")

        elif msg_type == "interactive":
            it   = msg["interactive"]
            text = it.get("button_reply", it.get("list_reply", {})).get("id", "")
            print(f"📩 Interactive from {phone}: {text!r}")
            reply = handle_message(phone, text)
            send_whatsapp_message(phone, reply)

        else:
            session = get_session(phone)
            lang    = session.get("lang", "en") or "en"
            send_whatsapp_message(phone, translate_reply(
                "Sorry, I only understand text messages and plant photos.\n"
                "📷 Send a photo to identify plant disease!\n0️⃣ Main menu",
                lang
            ))

    except (KeyError, IndexError) as e:
        print(f"❌ Parse error: {e}")
    return jsonify({"status": "ok"}), 200

# ─────────────────────────────────────────────
# CLI test mode
# ─────────────────────────────────────────────
def cli_test():
    print("\n" + "="*55)
    print("  AgriAssist+ — CLI Test Mode")
    print("="*55)
    print(f"Web App URL : {WEB_APP_URL}")
    print(f"CNN Detector: {'CNN + Claude Vision' if (DETECTOR and DETECTOR.cnn_available) else 'Claude Vision only'}")
    print("Type 'quit' to exit.\n")
    build_crop_keywords()
    phone = "test_farmer_cli_0001"
    while True:
        try: text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if text.lower() == "quit": break
        reply = handle_message(phone, text)
        print(f"\nBot:\n{reply}\n" + "-"*45)

if __name__ == "__main__":
    import sys
    build_crop_keywords()

    # ── Startup check ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  AgriAssist+ WhatsApp Bot — Startup Check")
    print("="*60)

    ok = True
    if not TM.phone_number_id:
        print("❌ PHONE_NUMBER_ID not set in .env"); ok = False
    else:
        print(f"✅ PHONE_NUMBER_ID : {TM.phone_number_id}")

    if TM.is_placeholder():
        print("❌ WHATSAPP_TOKEN  : placeholder detected — must be replaced!")
        print("   → Open .env and set WHATSAPP_TOKEN=<your real token>")
        print("   → Get token: developers.facebook.com → Your App → WhatsApp → API Setup")
        ok = False
    elif not TM.token:
        print("❌ WHATSAPP_TOKEN  : not set in .env"); ok = False
    else:
        print(f"✅ WHATSAPP_TOKEN  : {TM.token[:20]}…")
        print(f"   Check validity  : http://localhost:8000/status")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key or anthropic_key == "your_anthropic_api_key_here":
        print("⚠️  ANTHROPIC_API_KEY: not set — AI advice uses rule-based fallback")
    else:
        print("✅ ANTHROPIC_API_KEY: loaded")

    print(f"\n   To update token without restart:")
    print(f'   curl -X POST http://localhost:8000/token -H "Content-Type: application/json" -d \'{{"token":"NEW_TOKEN_HERE"}}\'')
    print("="*60 + "\n")

    if "--cli" in sys.argv:
        cli_test()
    else:
        print(f"🚀 Starting on port 8000…  Web App: {WEB_APP_URL}")
        app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
