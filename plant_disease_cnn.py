"""
plant_disease_cnn.py
=====================
CNN-based PLANT DISEASE & VARIETY DETECTION for AgriAssist+.

Supports TWO independent CNN models:
  1. TOMATO EARLY BLIGHT  (tomato_early_blight_model.h5)  — 2 classes
  2. MANGO VARIETY        (mango_variety_model.h5)         — 4 classes:
       Alphonso_Healthy / Alphonso_Diseased
       ImamPasand_Healthy / ImamPasand_Diseased

Both fall back to Claude Vision API when CNN confidence < 60%.

Usage:
    from plant_disease_cnn import PlantDiseaseDetector, MangoVarietyDetector
    detector = PlantDiseaseDetector()          # tomato
    mango    = MangoVarietyDetector()          # mango varieties
    result   = detector.predict_from_file(img)
    result   = mango.predict_from_file(img)
"""

import os
import io
import json
import base64
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
IMAGE_SIZE   = (224, 224)

# ══════════════════════════════════════════════
# ── TOMATO DISEASE INFO ────────────────────────
# ══════════════════════════════════════════════
PLANT_VILLAGE_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___healthy",
]

TOMATO_DISEASE_INFO = {
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early Blight",
        "tamil_name": "தக்காளி ஆரம்ப கருகல் நோய்",
        "severity": "High",
        "symptoms": "Dark brown spots with concentric rings (target board pattern) on older leaves. Yellow area around spots. Defoliation.",
        "cause": "Alternaria solani fungus. Thrives in humid conditions (>80% humidity) at 24–29°C.",
        "solution": "Spray Mancozeb 75WP immediately. Remove and burn infected leaves. Improve air circulation.",
        "chemicals": [
            {"name": "Mancozeb 75WP",    "dose": "2.5g per litre water", "frequency": "Every 7 days"},
            {"name": "Carbendazim 50WP", "dose": "1g per litre water",   "frequency": "Every 10 days"},
        ],
        "prevention": "Avoid overhead irrigation. Mulch around plants. Rotate crops. Use resistant varieties like CO-3.",
        "yield_impact": "20–50% if untreated",
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy",
        "tamil_name": "தக்காளி ஆரோக்கியமான செடி",
        "severity": "None",
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "cause": "N/A",
        "solution": "Your tomato plant looks healthy! Continue regular monitoring and preventive care.",
        "chemicals": [],
        "prevention": "Continue current practices. Monitor weekly for early signs of disease.",
        "yield_impact": "0% — healthy plant",
    },
}

# ══════════════════════════════════════════════
# ── MANGO VARIETY & DISEASE INFO ──────────────
# ══════════════════════════════════════════════
MANGO_CLASSES = [
    "Alphonso_Healthy",
    "Alphonso_Diseased",
    "ImamPasand_Healthy",
    "ImamPasand_Diseased",
]

MANGO_DISEASE_INFO = {
    "Alphonso_Healthy": {
        "crop": "Mango — Alphonso (Hapus)",
        "variety": "Alphonso",
        "disease": "Healthy",
        "tamil_name": "ஆல்போன்சோ மாம்பழம் — ஆரோக்கியமானது",
        "severity": "None",
        "symptoms": "No disease detected. Leaves are vibrant green with no spots, curling, or discolouration.",
        "cause": "N/A",
        "solution": "Your Alphonso mango tree is healthy! Maintain current care practices.",
        "chemicals": [],
        "prevention": (
            "Continue monthly foliar spray of micronutrients (Zinc + Boron). "
            "Irrigate at flower and fruit set stages. "
            "Apply 20kg FYM + 500g NPK per tree annually before monsoon."
        ),
        "variety_tips": (
            "Alphonso (Hapus) is TN's premium export variety. "
            "Harvest at 90–100 days from fruit set. "
            "Fruit turns yellow-orange when ripe. Store at 10–12°C for up to 3 weeks."
        ),
        "yield_impact": "0% — healthy tree",
        "market_price_range": "Rs. 80–200/kg (Alphonso commands premium in export markets)",
    },
    "Alphonso_Diseased": {
        "crop": "Mango — Alphonso (Hapus)",
        "variety": "Alphonso",
        "disease": "Disease Detected",
        "tamil_name": "ஆல்போன்சோ மாம்பழம் — நோய் கண்டறியப்பட்டது",
        "severity": "High",
        "symptoms": (
            "Possible symptoms: black/brown spots on leaves or fruits (Anthracnose), "
            "white powdery coating (Powdery Mildew), yellowing and tip-burn (nutrient deficiency), "
            "gummosis on stem, or fruit drop."
        ),
        "cause": (
            "Common causes: Colletotrichum gloeosporioides (Anthracnose), "
            "Oidium mangiferae (Powdery Mildew), bacterial infection, or nutrient imbalance."
        ),
        "solution": (
            "1. For Anthracnose: Spray Carbendazim 50WP (1g/L) + Mancozeb 75WP (2.5g/L) mix.\n"
            "2. For Powdery Mildew: Spray Sulphur 80WP (3g/L) or Hexaconazole (1ml/L).\n"
            "3. Remove and burn infected leaves/fruits.\n"
            "4. Improve canopy ventilation by pruning dense branches."
        ),
        "chemicals": [
            {"name": "Carbendazim 50WP + Mancozeb 75WP", "dose": "1g + 2.5g per litre", "frequency": "Every 10 days (3 sprays)"},
            {"name": "Hexaconazole 5EC",                  "dose": "1ml per litre",        "frequency": "Every 15 days for powdery mildew"},
            {"name": "Copper Oxychloride 50WP",           "dose": "3g per litre",          "frequency": "Preventive, every 21 days"},
        ],
        "prevention": (
            "Prune after harvest to allow air flow. "
            "Apply Bordeaux mixture (1%) before monsoon. "
            "Avoid water logging near the root zone. "
            "Apply balanced NPK (10:26:26) at fruit development stage."
        ),
        "variety_tips": (
            "Alphonso is highly susceptible to Anthracnose — protect during flowering. "
            "Apply pre-harvest fungicide spray 3 weeks before harvest. "
            "Diseased fruits lose export grade — treat early to protect premium market value."
        ),
        "yield_impact": "30–70% loss in untreated Alphonso orchards during high humidity seasons",
        "market_price_range": "Rs. 80–200/kg healthy; diseased fruit fetches Rs. 10–30/kg (waste grade)",
    },
    "ImamPasand_Healthy": {
        "crop": "Mango — Imam Pasand (Himayuddin)",
        "variety": "Imam Pasand",
        "disease": "Healthy",
        "tamil_name": "இமாம் பசந்த் மாம்பழம் — ஆரோக்கியமானது",
        "severity": "None",
        "symptoms": "No disease detected. Tree is in good health with strong green foliage.",
        "cause": "N/A",
        "solution": "Your Imam Pasand mango tree is healthy! Maintain current care.",
        "chemicals": [],
        "prevention": (
            "Imam Pasand trees need deep irrigation (once every 10–14 days in dry season). "
            "Apply potassium sulphate (1kg/tree) at pre-flowering to improve fruit sweetness. "
            "Protect from stem borers — check trunk monthly for holes and apply Chlorpyrifos paste."
        ),
        "variety_tips": (
            "Imam Pasand (Himayuddin) is popular in Andhra Pradesh and Tamil Nadu. "
            "Large fruit (400–600g), very sweet (22–26 Brix). "
            "Harvest window: June–July. Ripens uniformly on tree — no post-harvest ethylene needed."
        ),
        "yield_impact": "0% — healthy tree",
        "market_price_range": "Rs. 60–120/kg (local + regional markets; lower export demand than Alphonso)",
    },
    "ImamPasand_Diseased": {
        "crop": "Mango — Imam Pasand (Himayuddin)",
        "variety": "Imam Pasand",
        "disease": "Disease Detected",
        "tamil_name": "இமாம் பசந்த் மாம்பழம் — நோய் கண்டறியப்பட்டது",
        "severity": "High",
        "symptoms": (
            "Possible symptoms: Hopper damage (curled tender leaves, honeydew drip), "
            "Anthracnose (dark sunken spots on fruit), "
            "stem-end rot, sooty mould on leaves, or die-back of shoot tips."
        ),
        "cause": (
            "Common causes: Idioscopus clypealis (Mango Hopper), "
            "Colletotrichum gloeosporioides (Anthracnose), "
            "Lasiodiplodia theobromae (Die-back / Stem-end rot)."
        ),
        "solution": (
            "1. For Hopper: Spray Imidacloprid 17.8SL (0.5ml/L) at bud burst stage.\n"
            "2. For Anthracnose: Spray Carbendazim (1g/L) + Mancozeb (2.5g/L).\n"
            "3. For Die-back: Prune 10cm below infected area; apply Copper Oxychloride paste on cut ends.\n"
            "4. Drench soil with Trichoderma viride (10g/L) around root zone."
        ),
        "chemicals": [
            {"name": "Imidacloprid 17.8SL",               "dose": "0.5ml per litre",      "frequency": "2 sprays — bud burst & flower stage"},
            {"name": "Carbendazim 50WP + Mancozeb 75WP",  "dose": "1g + 2.5g per litre",  "frequency": "Every 10 days (3 sprays)"},
            {"name": "Trichoderma viride (bio-fungicide)", "dose": "10g per litre (drench)","frequency": "Once at root zone — monthly"},
            {"name": "Copper Oxychloride 50WP",            "dose": "3g per litre",          "frequency": "Protective spray before monsoon"},
        ],
        "prevention": (
            "Prune canopy after harvest to reduce hopper breeding. "
            "Apply light irrigation to trigger uniform flushing (avoids pest hotspots). "
            "Spray Neem oil (5ml/L) as preventive at bud break. "
            "Avoid excessive nitrogen — promotes soft flushes attractive to hoppers."
        ),
        "variety_tips": (
            "Imam Pasand is relatively hardy but susceptible to Hopper in dry hot conditions. "
            "Monitor new flushes carefully in Jan–March (pre-flowering). "
            "Apply pre-harvest fungicide (Carbendazim) 21 days before harvest to prevent stem-end rot."
        ),
        "yield_impact": "20–60% loss if hopper or anthracnose goes untreated at flowering stage",
        "market_price_range": "Rs. 60–120/kg healthy; diseased fruits unsaleable",
    },
}

# ══════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) * 2.0   # MobileNetV2 normalization
    return np.expand_dims(arr, axis=0)

def image_to_base64(img: Image.Image, max_size: int = 1024) -> str:
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _vision_fallback(reason: str) -> dict:
    return {
        "source": "fallback", "error": reason,
        "crop_identified": "Unknown", "disease_detected": "Analysis failed",
        "severity": "Unknown", "confidence": "Low",
        "symptoms_visible": "Could not analyze image.",
        "diagnosis": f"Image analysis failed: {reason}. Please try a clearer photo.",
        "immediate_action": "Take a clearer photo in good natural light and try again.",
        "chemical_treatment": "Cannot recommend without successful diagnosis.",
        "prevention": "Maintain general crop hygiene.", "yield_impact": "Unknown",
    }


# ══════════════════════════════════════════════
# CLAUDE VISION — TOMATO
# ══════════════════════════════════════════════
def analyze_tomato_with_claude(img: Image.Image, crop_hint: str = "") -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        return _vision_fallback("API key not configured")
    img_b64 = image_to_base64(img)
    prompt = f"""{'The farmer says this is a tomato plant. ' if crop_hint else ''}Analyze this tomato plant image specifically for EARLY BLIGHT disease.

Early Blight symptoms: dark brown concentric-ring spots on older leaves, yellow halos, defoliation.

Respond ONLY in this exact JSON format (no markdown fences):
{{
  "crop_identified": "Tomato",
  "disease_detected": "Early Blight" or "Healthy" or "Other Disease",
  "severity": "None / Low / Medium / High / Very High",
  "confidence": "High / Medium / Low",
  "symptoms_visible": "Describe what you see",
  "diagnosis": "2-3 sentences on whether this is early blight",
  "immediate_action": "Most urgent treatment step",
  "chemical_treatment": "Specific chemical name + dose + timing",
  "prevention": "1-2 sentences on early blight prevention",
  "yield_impact": "Estimated yield loss if untreated"
}}"""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": 1000, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}]},
            timeout=45,
        )
        resp.raise_for_status()
        raw = "".join(b["text"] for b in resp.json().get("content", []) if b.get("type") == "text").strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        parsed["source"] = "claude_vision"
        return parsed
    except Exception as e:
        return _vision_fallback(str(e))


# ══════════════════════════════════════════════
# CLAUDE VISION — MANGO
# ══════════════════════════════════════════════
def analyze_mango_with_claude(img: Image.Image, crop_hint: str = "") -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        return _vision_fallback("API key not configured")
    img_b64 = image_to_base64(img)
    prompt = """You are an expert agricultural AI specializing in Tamil Nadu mango cultivation.
Analyze this mango plant image and identify:
1. The mango VARIETY — is this Alphonso (Hapus) or Imam Pasand (Himayuddin)?
2. The HEALTH STATUS — is it healthy or showing disease symptoms?

Variety identification clues:
- Alphonso: smaller leaves, yellow-orange fruit, slightly curved shape, golden skin when ripe
- Imam Pasand: larger rounded fruits (400-600g), light green/yellow skin, less fibrous flesh

Disease clues: anthracnose (black spots), powdery mildew (white coating), hopper damage (curled leaves), die-back (brown shoot tips)

Respond ONLY in this exact JSON format (no markdown fences):
{
  "crop_identified": "Mango",
  "variety_identified": "Alphonso" or "Imam Pasand" or "Unknown Mango Variety",
  "health_status": "Healthy" or "Diseased",
  "disease_detected": "Healthy" or disease name (e.g. "Anthracnose", "Powdery Mildew", "Hopper Damage", "Die-back"),
  "severity": "None / Low / Medium / High / Very High",
  "confidence": "High / Medium / Low",
  "variety_confidence": "High / Medium / Low",
  "symptoms_visible": "Detailed description of what you observe",
  "diagnosis": "3-4 sentences explaining the variety identification and health assessment",
  "immediate_action": "Most urgent step if diseased, or maintenance tip if healthy",
  "chemical_treatment": "Specific chemical name + dose per litre + timing (or 'None needed' if healthy)",
  "prevention": "2 sentences on disease prevention or orchard management",
  "yield_impact": "Estimated yield impact if untreated (or 0% if healthy)"
}"""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": 1200, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}]},
            timeout=45,
        )
        resp.raise_for_status()
        raw = "".join(b["text"] for b in resp.json().get("content", []) if b.get("type") == "text").strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        parsed["source"] = "claude_vision"
        return parsed
    except Exception as e:
        return _vision_fallback(str(e))


# ══════════════════════════════════════════════
# TOMATO DETECTOR CLASS  (unchanged API)
# ══════════════════════════════════════════════
class PlantDiseaseDetector:
    def __init__(self, model_path: str = "tomato_early_blight_model.h5"):
        self.model_path    = model_path
        self.model         = None
        self.cnn_available = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"ℹ️  Tomato CNN model not found at '{self.model_path}'. Using Claude Vision.")
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.cnn_available = True
            print(f"✅ Tomato CNN model loaded from '{self.model_path}'")
        except ImportError:
            print("⚠️  TensorFlow not installed — pip install tensorflow")
        except Exception as e:
            print(f"⚠️  Failed to load Tomato CNN: {e}")

    def predict_from_file(self, img_input, crop_hint: str = "") -> dict:
        if isinstance(img_input, str):         img = Image.open(img_input)
        elif isinstance(img_input, Image.Image): img = img_input
        else:                                   img = Image.open(img_input)
        return self._predict(img, crop_hint)

    def predict_from_bytes(self, img_bytes: bytes, crop_hint: str = "") -> dict:
        return self._predict(Image.open(io.BytesIO(img_bytes)), crop_hint)

    def predict_from_url(self, url: str, headers: dict = None, crop_hint: str = "") -> dict:
        try:
            resp = requests.get(url, headers=headers or {}, timeout=30, stream=True)
            resp.raise_for_status()
            return self._predict(Image.open(io.BytesIO(resp.content)), crop_hint)
        except Exception as e:
            return {"error": str(e), "source": "url_download_failed",
                    "crop_identified": "Tomato", "disease_detected": "Download failed",
                    "tamil_name": "", "severity": "Unknown", "confidence_pct": 0,
                    "diagnosis": f"Could not download photo: {e}",
                    "immediate_action": "Please resend the photo.", "chemicals": [],
                    "prevention": "", "yield_impact": "Unknown", "top2_predictions": []}

    def _predict(self, img: Image.Image, crop_hint: str = "") -> dict:
        cnn_result = None
        cnn_label  = None
        cnn_conf   = 0.0
        use_claude = True

        if self.cnn_available and self.model is not None:
            try:
                preds    = self.model.predict(preprocess_image(img), verbose=0)[0]
                top_idx  = int(np.argmax(preds))
                cnn_conf = float(preds[top_idx])
                cnn_label = PLANT_VILLAGE_CLASSES[top_idx]
                if cnn_conf >= 0.60:
                    use_claude = False
                    cnn_result = {
                        "label": cnn_label, "confidence": cnn_conf,
                        "top2": [{"label": PLANT_VILLAGE_CLASSES[i], "confidence": float(preds[i])}
                                 for i in np.argsort(preds)[::-1][:2]],
                    }
            except Exception as e:
                print(f"⚠️  Tomato CNN error: {e}")

        claude_result = analyze_tomato_with_claude(img, crop_hint) if use_claude else None
        return self._build_result(cnn_result, claude_result, cnn_label, cnn_conf, crop_hint)

    def _build_result(self, cnn_result, claude_result, cnn_label, cnn_conf, crop_hint):
        result = {
            "image_analyzed": True,
            "cnn_available":  self.cnn_available,
            "cnn_label":      cnn_label,
            "source":         "cnn" if cnn_result else "claude_vision",
            "raw_claude":     claude_result,
        }
        if cnn_result and cnn_conf >= 0.60:
            info = TOMATO_DISEASE_INFO.get(cnn_label, {})
            result.update({
                "crop_identified":  info.get("crop", "Tomato"),
                "disease_detected": info.get("disease", "Unknown"),
                "tamil_name":       info.get("tamil_name", ""),
                "severity":         info.get("severity", "Unknown"),
                "confidence_pct":   round(cnn_conf * 100, 1),
                "confidence_label": "High" if cnn_conf > 0.85 else "Medium",
                "symptoms_visible": info.get("symptoms", ""),
                "diagnosis":        f"CNN detected: {info.get('disease')} ({cnn_conf*100:.0f}% confidence).\n\n{info.get('symptoms')}\n\nCause: {info.get('cause','')}",
                "immediate_action": info.get("solution", ""),
                "chemicals":        info.get("chemicals", []),
                "prevention":       info.get("prevention", ""),
                "yield_impact":     info.get("yield_impact", "Unknown"),
                "top2_predictions": cnn_result.get("top2", []),
            })
        elif claude_result and claude_result.get("source") != "fallback":
            result.update({
                "crop_identified":  claude_result.get("crop_identified", crop_hint or "Tomato"),
                "disease_detected": claude_result.get("disease_detected", "Unknown"),
                "tamil_name":       "",
                "severity":         claude_result.get("severity", "Unknown"),
                "confidence_pct":   85 if claude_result.get("confidence") == "High" else 60 if claude_result.get("confidence") == "Medium" else 40,
                "confidence_label": claude_result.get("confidence", "Medium"),
                "symptoms_visible": claude_result.get("symptoms_visible", ""),
                "diagnosis":        claude_result.get("diagnosis", ""),
                "immediate_action": claude_result.get("immediate_action", ""),
                "chemicals":        [{"name": claude_result.get("chemical_treatment", "See diagnosis"), "dose": "As per label", "frequency": "As directed"}] if claude_result.get("chemical_treatment") else [],
                "prevention":       claude_result.get("prevention", ""),
                "yield_impact":     claude_result.get("yield_impact", "Unknown"),
                "top2_predictions": [],
            })
        else:
            result.update({
                "crop_identified": crop_hint or "Tomato", "disease_detected": "Analysis failed",
                "tamil_name": "", "severity": "Unknown", "confidence_pct": 0,
                "confidence_label": "Low", "symptoms_visible": "",
                "diagnosis": "Image analysis failed. Try a clearer photo.",
                "immediate_action": "Retake photo in natural light.", "chemicals": [],
                "prevention": "", "yield_impact": "Unknown", "top2_predictions": [],
            })
        return result

    def is_healthy(self, result: dict) -> bool:
        return "healthy" in result.get("disease_detected", "").lower() or result.get("severity") == "None"

    def format_whatsapp_reply(self, result: dict, **kwargs) -> str:
        lines = [
            "🔬 *Tomato Early Blight Detection*", "──────────────────────",
            f"🌱 *Crop:* {result.get('crop_identified','Tomato')}",
            f"🦠 *Disease:* {result.get('disease_detected','Unknown')}",
            f"⚠️ *Severity:* {result.get('severity','Unknown')}",
            f"📊 *Confidence:* {result.get('confidence_pct',0):.0f}%",
            "", f"✅ *Action:* {result.get('immediate_action','')[:300]}",
            "", "📸 Send another photo | 0️⃣ Main menu",
        ]
        return "\n".join(lines)[:4090]


# ══════════════════════════════════════════════
# MANGO VARIETY DETECTOR CLASS  (new)
# ══════════════════════════════════════════════
class MangoVarietyDetector:
    """
    Detects mango variety (Alphonso / Imam Pasand) and health status.
    Uses CNN if trained model is present, otherwise Claude Vision.
    """

    def __init__(self, model_path: str = "mango_variety_model.h5"):
        self.model_path    = model_path
        self.model         = None
        self.cnn_available = False
        self.class_indices = self._load_class_indices()
        self._load_model()

    def _load_class_indices(self) -> dict:
        idx_path = "mango_class_indices.json"
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                return json.load(f)
        # default order when trained with sorted class names
        return {str(i): c for i, c in enumerate(sorted(MANGO_CLASSES))}

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"ℹ️  Mango CNN model not found at '{self.model_path}'. Using Claude Vision.")
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.cnn_available = True
            print(f"✅ Mango CNN model loaded from '{self.model_path}'")
        except ImportError:
            print("⚠️  TensorFlow not installed — pip install tensorflow")
        except Exception as e:
            print(f"⚠️  Failed to load Mango CNN: {e}")

    # ── Public API ─────────────────────────────
    def predict_from_file(self, img_input, crop_hint: str = "") -> dict:
        if isinstance(img_input, str):           img = Image.open(img_input)
        elif isinstance(img_input, Image.Image): img = img_input
        else:                                    img = Image.open(img_input)
        return self._predict(img, crop_hint)

    def predict_from_bytes(self, img_bytes: bytes, crop_hint: str = "") -> dict:
        return self._predict(Image.open(io.BytesIO(img_bytes)), crop_hint)

    def predict_from_url(self, url: str, headers: dict = None, crop_hint: str = "") -> dict:
        try:
            resp = requests.get(url, headers=headers or {}, timeout=30, stream=True)
            resp.raise_for_status()
            return self._predict(Image.open(io.BytesIO(resp.content)), crop_hint)
        except Exception as e:
            return self._error_result(str(e))

    # ── Core prediction ────────────────────────
    def _predict(self, img: Image.Image, crop_hint: str = "") -> dict:
        cnn_result = None
        cnn_label  = None
        cnn_conf   = 0.0
        use_claude = True

        if self.cnn_available and self.model is not None:
            try:
                preds    = self.model.predict(preprocess_image(img), verbose=0)[0]
                top_idx  = int(np.argmax(preds))
                cnn_conf = float(preds[top_idx])
                # map index → class name using saved class_indices
                num_classes = len(preds)
                idx_to_class = {int(k): v for k, v in self.class_indices.items()}
                cnn_label = idx_to_class.get(top_idx, MANGO_CLASSES[top_idx] if top_idx < len(MANGO_CLASSES) else "Unknown")
                if cnn_conf >= 0.60:
                    use_claude = False
                    cnn_result = {
                        "label": cnn_label, "confidence": cnn_conf,
                        "top_predictions": [
                            {"label": idx_to_class.get(i, f"class_{i}"), "confidence": float(preds[i])}
                            for i in np.argsort(preds)[::-1][:min(4, num_classes)]
                        ],
                    }
            except Exception as e:
                print(f"⚠️  Mango CNN error: {e}")

        claude_result = analyze_mango_with_claude(img, crop_hint) if use_claude else None
        return self._build_result(cnn_result, claude_result, cnn_label, cnn_conf, crop_hint)

    def _build_result(self, cnn_result, claude_result, cnn_label, cnn_conf, crop_hint):
        result = {
            "detector_type": "mango",
            "image_analyzed": True,
            "cnn_available":  self.cnn_available,
            "source":         "cnn" if cnn_result else "claude_vision",
        }

        if cnn_result and cnn_conf >= 0.60:
            info = MANGO_DISEASE_INFO.get(cnn_label, {})
            variety = "Alphonso" if "Alphonso" in cnn_label else "Imam Pasand" if "ImamPasand" in cnn_label else "Unknown"
            health  = "Healthy" if "Healthy" in cnn_label else "Diseased"
            result.update({
                "crop_identified":    info.get("crop", f"Mango — {variety}"),
                "variety_identified": variety,
                "health_status":      health,
                "disease_detected":   info.get("disease", health),
                "tamil_name":         info.get("tamil_name", ""),
                "severity":           info.get("severity", "Unknown"),
                "confidence_pct":     round(cnn_conf * 100, 1),
                "confidence_label":   "High" if cnn_conf > 0.85 else "Medium",
                "variety_confidence": "High" if cnn_conf > 0.80 else "Medium",
                "symptoms_visible":   info.get("symptoms", ""),
                "diagnosis":          f"CNN identified: {variety} mango — {health} ({cnn_conf*100:.0f}% confidence).\n\n{info.get('symptoms','')}",
                "immediate_action":   info.get("solution", ""),
                "chemicals":          info.get("chemicals", []),
                "prevention":         info.get("prevention", ""),
                "variety_tips":       info.get("variety_tips", ""),
                "market_price_range": info.get("market_price_range", ""),
                "yield_impact":       info.get("yield_impact", "Unknown"),
                "top_predictions":    cnn_result.get("top_predictions", []),
            })

        elif claude_result and claude_result.get("source") != "fallback":
            variety = claude_result.get("variety_identified", "Unknown")
            health  = claude_result.get("health_status", "Unknown")
            # lookup MANGO_DISEASE_INFO for enriched static data
            key = None
            if "Alphonso" in variety and health == "Healthy":    key = "Alphonso_Healthy"
            elif "Alphonso" in variety and health == "Diseased": key = "Alphonso_Diseased"
            elif "Imam" in variety and health == "Healthy":      key = "ImamPasand_Healthy"
            elif "Imam" in variety and health == "Diseased":     key = "ImamPasand_Diseased"
            static = MANGO_DISEASE_INFO.get(key, {})

            result.update({
                "crop_identified":    f"Mango — {variety}" if variety != "Unknown" else "Mango",
                "variety_identified": variety,
                "health_status":      health,
                "disease_detected":   claude_result.get("disease_detected", "Unknown"),
                "tamil_name":         static.get("tamil_name", ""),
                "severity":           claude_result.get("severity", "Unknown"),
                "confidence_pct":     85 if claude_result.get("confidence") == "High" else 60 if claude_result.get("confidence") == "Medium" else 40,
                "confidence_label":   claude_result.get("confidence", "Medium"),
                "variety_confidence": claude_result.get("variety_confidence", "Medium"),
                "symptoms_visible":   claude_result.get("symptoms_visible", ""),
                "diagnosis":          claude_result.get("diagnosis", ""),
                "immediate_action":   claude_result.get("immediate_action", ""),
                "chemicals":          static.get("chemicals") or (
                    [{"name": claude_result.get("chemical_treatment", ""), "dose": "As per label", "frequency": "As directed"}]
                    if claude_result.get("chemical_treatment") and claude_result.get("chemical_treatment") != "None needed" else []
                ),
                "prevention":         claude_result.get("prevention", "") or static.get("prevention", ""),
                "variety_tips":       static.get("variety_tips", ""),
                "market_price_range": static.get("market_price_range", ""),
                "yield_impact":       claude_result.get("yield_impact", "Unknown"),
                "top_predictions":    [],
            })
        else:
            result.update(self._error_result("Analysis failed — please try a clearer photo."))

        return result

    def is_healthy(self, result: dict) -> bool:
        return result.get("health_status") == "Healthy" or result.get("severity") == "None"

    def _error_result(self, msg: str) -> dict:
        return {
            "detector_type": "mango", "image_analyzed": True,
            "cnn_available": self.cnn_available, "source": "error",
            "crop_identified": "Mango", "variety_identified": "Unknown",
            "health_status": "Unknown", "disease_detected": "Analysis failed",
            "tamil_name": "", "severity": "Unknown", "confidence_pct": 0,
            "confidence_label": "Low", "variety_confidence": "Low",
            "symptoms_visible": "", "diagnosis": msg,
            "immediate_action": "Try a clearer, well-lit close-up photo of the leaf or fruit.",
            "chemicals": [], "prevention": "", "variety_tips": "",
            "market_price_range": "", "yield_impact": "Unknown", "top_predictions": [],
        }
