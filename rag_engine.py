"""
rag_engine.py
==============
RAG + LLM engine for AgriAssist+.
Features:
  - Weather-aware chemical recommendation system
  - Dosage guidance per acre / per litre water
  - Yield impact prediction if chemical is used
  - Chemicals to AVOID based on weather conditions
  - Rule-based fallback when Anthropic API key not set
"""

import os
import re
import json
import hashlib
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CHEMICAL DATABASE — weather-aware, with dosage + yield impact
# ══════════════════════════════════════════════════════════════════
CHEMICAL_DB = {
    # ── FUNGICIDES ──────────────────────────────────────────────
    "Mancozeb 75WP": {
        "type": "Fungicide",
        "targets": ["early blight","leaf spot","downy mildew","anthracnose","blast"],
        "crops":   ["Tomato","Potato","Onion","Chilli","Paddy","Banana","Mango","Groundnut"],
        "dose_per_litre": "2.5 g",
        "dose_per_acre":  "500 g in 200 L water",
        "frequency":      "Every 7–10 days",
        "yield_gain_pct": 15,
        "best_weather":   {"humidity_min": 60, "humidity_max": 85, "temp_max": 35},
        "avoid_when": {
            "rain_heavy":   True,
            "temp_above":   38,
            "humidity_above": 90,
        },
        "avoid_reason":   "Washes off in heavy rain (>15 mm); phytotoxic above 38°C",
        "timing":         "Early morning or evening; NOT when rain expected within 4 hrs",
        "safety":         "Wear gloves + mask. PHI: 7 days before harvest.",
    },
    "Carbendazim 50WP": {
        "type": "Fungicide",
        "targets": ["powdery mildew","wilt","stem rot","sheath blight","leaf blight"],
        "crops":   ["Tomato","Chilli","Paddy","Banana","Beans","Onion","Groundnut"],
        "dose_per_litre": "1 g",
        "dose_per_acre":  "200 g in 200 L water",
        "frequency":      "Every 10–14 days",
        "yield_gain_pct": 18,
        "best_weather":   {"humidity_min": 55, "humidity_max": 80, "temp_max": 36},
        "avoid_when": {
            "rain_heavy":   True,
            "temp_above":   40,
        },
        "avoid_reason":   "Ineffective if washed off; decomposes rapidly above 40°C",
        "timing":         "Morning spray preferred",
        "safety":         "PHI: 14 days. Rotate with other fungicides to avoid resistance.",
    },
    "Hexaconazole 5EC": {
        "type": "Fungicide (Systemic)",
        "targets": ["sheath blight","powdery mildew","rust","leaf blight"],
        "crops":   ["Paddy","Banana","Mango","Chilli","Groundnut"],
        "dose_per_litre": "2 ml",
        "dose_per_acre":  "400 ml in 200 L water",
        "frequency":      "Every 14 days",
        "yield_gain_pct": 20,
        "best_weather":   {"humidity_min": 60, "humidity_max": 85, "temp_max": 36},
        "avoid_when": {
            "rain_heavy":   True,
            "temp_above":   38,
            "wind_above":   20,
        },
        "avoid_reason":   "Systemic — needs 2 hrs dry after spraying to absorb; avoid high wind drift",
        "timing":         "Early morning; ensure no rain for 4 hrs after application",
        "safety":         "PHI: 21 days. Do not spray during flowering (bee hazard).",
    },
    "Metalaxyl + Mancozeb": {
        "type": "Fungicide (Systemic + Contact)",
        "targets": ["late blight","downy mildew","damping off","pythium"],
        "crops":   ["Tomato","Potato","Chilli","Onion","Banana"],
        "dose_per_litre": "2.5 g",
        "dose_per_acre":  "500 g in 200 L water",
        "frequency":      "Every 7 days during high disease pressure",
        "yield_gain_pct": 25,
        "best_weather":   {"humidity_min": 70, "humidity_max": 90, "temp_max": 30},
        "avoid_when": {
            "temp_above":   35,
            "humidity_below": 50,
        },
        "avoid_reason":   "Most effective in cool humid conditions; less effective in dry heat",
        "timing":         "Best applied when humidity is high (monsoon early stage)",
        "safety":         "PHI: 10 days. Do not apply more than 3 times per season.",
    },
    "Copper Oxychloride 50WP": {
        "type": "Fungicide + Bactericide",
        "targets": ["bacterial blight","anthracnose","leaf curl","downy mildew","shot hole"],
        "crops":   ["Tomato","Mango","Banana","Chilli","Coconut","Brinjal"],
        "dose_per_litre": "3 g",
        "dose_per_acre":  "600 g in 200 L water",
        "frequency":      "Every 10 days",
        "yield_gain_pct": 12,
        "best_weather":   {"humidity_min": 50, "humidity_max": 80, "temp_max": 35},
        "avoid_when": {
            "temp_above":   35,
            "rain_heavy":   True,
        },
        "avoid_reason":   "Causes phytotoxicity (leaf burn) above 35°C or under bright sun",
        "timing":         "Evening application strongly preferred to avoid leaf burn",
        "safety":         "PHI: 7 days. Avoid mixing with alkaline pesticides.",
    },
    "Sulphur 80WP": {
        "type": "Fungicide (Contact)",
        "targets": ["powdery mildew","mites","rust"],
        "crops":   ["Mango","Chilli","Tomato","Grapes","Onion"],
        "dose_per_litre": "3 g",
        "dose_per_acre":  "600 g in 200 L water",
        "frequency":      "Every 10 days",
        "yield_gain_pct": 10,
        "best_weather":   {"humidity_min": 40, "humidity_max": 70, "temp_max": 35},
        "avoid_when": {
            "temp_above":   35,
            "rain_heavy":   True,
            "humidity_above": 85,
        },
        "avoid_reason":   "STRICTLY AVOID above 35°C — causes severe leaf scorch and crop damage",
        "timing":         "Use ONLY in cool morning (before 9 AM); never in afternoon",
        "safety":         "PHI: 3 days. Do not mix with oil-based pesticides.",
    },

    # ── INSECTICIDES ────────────────────────────────────────────
    "Imidacloprid 17.8SL": {
        "type": "Insecticide (Systemic)",
        "targets": ["whitefly","aphid","thrips","brown plant hopper","leafhopper"],
        "crops":   ["Tomato","Chilli","Brinjal","Paddy","Cotton","Banana"],
        "dose_per_litre": "0.5 ml",
        "dose_per_acre":  "100 ml in 200 L water",
        "frequency":      "Once; repeat after 15 days if needed",
        "yield_gain_pct": 20,
        "best_weather":   {"humidity_min": 40, "humidity_max": 80, "temp_max": 38},
        "avoid_when": {
            "rain_heavy":   True,
            "flowering":    True,
        },
        "avoid_reason":   "HIGHLY TOXIC TO BEES — never spray during flowering or when bees active",
        "timing":         "Early morning or late evening; avoid flowering period entirely",
        "safety":         "PHI: 7 days. Do NOT spray near water bodies. Bee hazard.",
    },
    "Chlorpyrifos 20EC": {
        "type": "Insecticide (Contact + Stomach)",
        "targets": ["stem borer","leaf folder","cutworm","termite","pod borer"],
        "crops":   ["Paddy","Sugarcane","Maize","Groundnut","Cotton","Ragi"],
        "dose_per_litre": "2.5 ml",
        "dose_per_acre":  "500 ml in 200 L water",
        "frequency":      "Once per infestation cycle",
        "yield_gain_pct": 22,
        "best_weather":   {"humidity_min": 40, "humidity_max": 80, "temp_max": 38},
        "avoid_when": {
            "rain_heavy":   True,
            "temp_above":   40,
            "wind_above":   15,
        },
        "avoid_reason":   "Volatilizes rapidly above 40°C; wash-off in rain reduces efficacy",
        "timing":         "Evening application; avoid windy conditions",
        "safety":         "PHI: 15 days. Highly toxic to fish — do not spray near ponds.",
    },
    "Spinosad 45SC": {
        "type": "Insecticide (Bio-derived)",
        "targets": ["thrips","leaf miner","fruit borer","caterpillar"],
        "crops":   ["Tomato","Chilli","Onion","Brinjal","Cauliflower","Beans"],
        "dose_per_litre": "0.3 ml",
        "dose_per_acre":  "60 ml in 200 L water",
        "frequency":      "Every 7 days during infestation",
        "yield_gain_pct": 17,
        "best_weather":   {"humidity_min": 40, "humidity_max": 85, "temp_max": 38},
        "avoid_when": {
            "rain_heavy":   True,
            "temp_above":   40,
        },
        "avoid_reason":   "UV degradation in strong sunlight; less effective in very high heat",
        "timing":         "Evening spray for best results",
        "safety":         "PHI: 1 day. Low mammalian toxicity. Safe for beneficial insects.",
    },
    "Buprofezin 25SC": {
        "type": "Insecticide (IGR)",
        "targets": ["brown plant hopper","white backed plant hopper","scale insects"],
        "crops":   ["Paddy","Sugarcane","Coconut"],
        "dose_per_litre": "1 ml",
        "dose_per_acre":  "200 ml in 200 L water",
        "frequency":      "Once per crop season",
        "yield_gain_pct": 25,
        "best_weather":   {"humidity_min": 60, "humidity_max": 90, "temp_max": 38},
        "avoid_when": {
            "rain_heavy":   True,
        },
        "avoid_reason":   "Rain within 2 hrs reduces efficacy significantly",
        "timing":         "Morning; ensure no rain for at least 2 hrs after spraying",
        "safety":         "PHI: 14 days. Do not mix with fungicides.",
    },
    "Neem Oil 1500ppm": {
        "type": "Bio-Pesticide",
        "targets": ["aphid","whitefly","mealybug","mites","powdery mildew","leaf spot"],
        "crops":   ["All crops"],
        "dose_per_litre": "5 ml + 1 ml liquid soap (emulsifier)",
        "dose_per_acre":  "1000 ml neem oil + 200 ml soap in 200 L water",
        "frequency":      "Every 5–7 days",
        "yield_gain_pct": 8,
        "best_weather":   {"humidity_min": 40, "humidity_max": 80, "temp_max": 36},
        "avoid_when": {
            "temp_above":   38,
            "rain_heavy":   True,
            "humidity_above": 90,
        },
        "avoid_reason":   "Breaks down quickly in heavy rain; less effective in very humid or very hot conditions",
        "timing":         "Evening only — degrades rapidly in sunlight",
        "safety":         "PHI: 0 days (organic). Safe for humans and beneficial insects.",
    },

    # ── FERTILIZERS / GROWTH PROMOTERS ─────────────────────────
    "Urea (46% N)": {
        "type": "Fertilizer (Nitrogen)",
        "targets": ["nitrogen deficiency","yellowing","slow growth"],
        "crops":   ["Paddy","Maize","Sugarcane","Banana","Vegetables"],
        "dose_per_litre": "2 g (foliar spray)",
        "dose_per_acre":  "25–35 kg (soil application); OR 400 g in 200 L water (foliar)",
        "frequency":      "Split dose: basal + 30 days + 60 days",
        "yield_gain_pct": 20,
        "best_weather":   {"humidity_min": 50, "humidity_max": 80, "temp_max": 35},
        "avoid_when": {
            "temp_above":   38,
            "rain_heavy":   True,
            "humidity_above": 90,
        },
        "avoid_reason":   "Volatilizes as ammonia above 38°C (up to 40% N loss); heavy rain leaches N from soil",
        "timing":         "Apply after rain when soil is moist but NOT waterlogged",
        "safety":         "Do not over-apply — excess N promotes pest & disease attack.",
    },
    "DAP (18:46:0)": {
        "type": "Fertilizer (Nitrogen + Phosphorus)",
        "targets": ["root development","early growth","phosphorus deficiency"],
        "crops":   ["All crops"],
        "dose_per_litre": "N/A (soil application only)",
        "dose_per_acre":  "50 kg at planting (basal dose)",
        "frequency":      "Once at sowing / transplanting",
        "yield_gain_pct": 15,
        "best_weather":   {"humidity_min": 40, "humidity_max": 85, "temp_max": 40},
        "avoid_when": {
            "flood":        True,
        },
        "avoid_reason":   "Phosphorus lost through runoff during heavy flooding",
        "timing":         "Apply before or at planting; mix into soil",
        "safety":         "Avoid contact with seeds during sowing.",
    },
    "Potassium Nitrate (13:0:45)": {
        "type": "Fertilizer (Potassium + Nitrogen)",
        "targets": ["fruit development","drought stress","quality improvement"],
        "crops":   ["Tomato","Chilli","Banana","Mango","Grapes"],
        "dose_per_litre": "5 g (foliar)",
        "dose_per_acre":  "1 kg in 200 L water (foliar); or 25 kg soil",
        "frequency":      "Every 15 days during fruiting",
        "yield_gain_pct": 12,
        "best_weather":   {"humidity_min": 40, "humidity_max": 80, "temp_max": 38},
        "avoid_when": {
            "rain_heavy":   True,
        },
        "avoid_reason":   "Washes off leaves in rain; foliar uptake requires dry conditions",
        "timing":         "Morning or evening; avoid afternoon in summer",
        "safety":         "PHI: 3 days. Store away from flammable material.",
    },
    "Zinc Sulphate (21% Zn)": {
        "type": "Micronutrient",
        "targets": ["zinc deficiency","white bud","khaira disease","yellowing"],
        "crops":   ["Paddy","Maize","Sugarcane","Groundnut","Vegetables"],
        "dose_per_litre": "3 g",
        "dose_per_acre":  "10 kg (soil); OR 600 g in 200 L water (foliar)",
        "frequency":      "Once per season (soil); or twice (foliar)",
        "yield_gain_pct": 10,
        "best_weather":   {"humidity_min": 40, "humidity_max": 85, "temp_max": 40},
        "avoid_when": {
            "rain_heavy":   True,
        },
        "avoid_reason":   "Foliar spray washed off in rain; soil application unaffected",
        "timing":         "Foliar: evening; Soil: before irrigation",
        "safety":         "Do not mix with phosphate fertilizers — forms insoluble precipitate.",
    },
}

# ── Weather condition thresholds ──────────────────────────────
def get_weather_condition(weather: dict) -> dict:
    temp     = weather.get("temp_max", 30)
    rain     = weather.get("rainfall", 10)
    humidity = weather.get("humidity", 60)
    wind     = weather.get("wind_speed", 10)

    return {
        "temp":           temp,
        "rain":           rain,
        "humidity":       humidity,
        "wind":           wind,
        "rain_heavy":     rain > 40,
        "rain_light":     5 < rain <= 40,
        "drought":        rain < 5,
        "temp_above_35":  temp > 35,
        "temp_above_38":  temp > 38,
        "temp_above_40":  temp > 40,
        "high_humidity":  humidity > 80,
        "very_high_humidity": humidity > 90,
        "low_humidity":   humidity < 40,
        "high_wind":      wind > 15,
    }

def recommend_chemicals(crop: str, weather: dict, yield_loss_pct: float) -> dict:
    wc = get_weather_condition(weather)
    recommended = []
    avoid_list  = []

    for chem_name, chem in CHEMICAL_DB.items():
        crop_match = (
            crop in chem["crops"] or
            "All crops" in chem["crops"] or
            any(c.lower() in crop.lower() for c in chem["crops"])
        )
        if not crop_match:
            continue

        should_avoid = False
        avoid_reasons = []

        av = chem.get("avoid_when", {})
        if av.get("rain_heavy") and wc["rain_heavy"]:
            should_avoid = True
            avoid_reasons.append(f"Heavy rainfall ({wc['rain']:.0f}mm) — chemical will wash off")
        if av.get("temp_above") and wc["temp"] > av["temp_above"]:
            should_avoid = True
            avoid_reasons.append(f"Temperature {wc['temp']}°C exceeds safe limit of {av['temp_above']}°C")
        if av.get("humidity_above") and wc["humidity"] > av["humidity_above"]:
            should_avoid = True
            avoid_reasons.append(f"Humidity {wc['humidity']}% too high for this chemical")
        if av.get("humidity_below") and wc["humidity"] < av["humidity_below"]:
            should_avoid = True
            avoid_reasons.append(f"Humidity {wc['humidity']}% too low for this chemical")
        if av.get("wind_above") and wc["wind"] > av["wind_above"]:
            should_avoid = True
            avoid_reasons.append(f"Wind speed {wc['wind']} km/h too high — drift risk")

        if should_avoid:
            avoid_list.append({
                "name":    chem_name,
                "type":    chem["type"],
                "reasons": avoid_reasons,
                "original_avoid_reason": chem["avoid_reason"],
            })
        else:
            bw = chem.get("best_weather", {})
            score = 0
            if bw.get("humidity_min", 0) <= wc["humidity"] <= bw.get("humidity_max", 100):
                score += 2
            if wc["temp"] <= bw.get("temp_max", 40):
                score += 2
            if yield_loss_pct > 15:
                score += 1

            recommended.append({
                "name":           chem_name,
                "type":           chem["type"],
                "targets":        chem["targets"][:3],
                "dose_per_litre": chem["dose_per_litre"],
                "dose_per_acre":  chem["dose_per_acre"],
                "frequency":      chem["frequency"],
                "yield_gain_pct": chem["yield_gain_pct"],
                "timing":         chem["timing"],
                "safety":         chem["safety"],
                "score":          score,
            })

    recommended.sort(key=lambda x: x["score"], reverse=True)

    weather_notes = []
    if wc["rain_heavy"]:
        weather_notes.append("🌧️ Heavy rainfall detected — avoid all foliar sprays. Apply only soil-based treatments.")
    if wc["temp_above_38"]:
        weather_notes.append("🌡️ Temperature above 38°C — avoid sulphur and copper-based chemicals to prevent leaf burn.")
    if wc["high_humidity"]:
        weather_notes.append("💧 High humidity (>80%) — fungal disease pressure is HIGH. Prioritize fungicide application.")
    if wc["drought"]:
        weather_notes.append("🏜️ Dry conditions — increase irrigation before chemical application for better absorption.")
    if wc["high_wind"]:
        weather_notes.append("💨 High wind speed — delay spraying to prevent chemical drift and uneven coverage.")
    if not weather_notes:
        weather_notes.append("✅ Weather conditions are suitable for most chemical applications.")

    return {
        "recommended":    recommended[:5],
        "avoid":          avoid_list[:4],
        "weather_notes":  weather_notes,
        "weather_condition": wc,
    }


# ══════════════════════════════════════════════════════════════════
# CROP KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    {"id":"paddy_01","crop":"Paddy","topic":"season",
     "text":"Paddy in Tamil Nadu: Kuruvai (June–Sep) and Samba (Aug–Jan). Best varieties: IR-20, Ponni, ADT-43, Samba Masonali. Kuruvai commands premium price. High humidity during Samba increases blast and sheath blight risk."},
    {"id":"paddy_02","crop":"Paddy","topic":"disease_chemical",
     "text":"Paddy diseases: Blast — Tricyclazole 75WP 0.6g/L every 10 days. Brown Plant Hopper — Buprofezin 25SC 1ml/L. Sheath blight — Hexaconazole 5EC 2ml/L. Stem borer — Chlorpyrifos 20EC 2.5ml/L. Apply when humidity >80% as preventive measure."},
    {"id":"paddy_03","crop":"Paddy","topic":"yield",
     "text":"Paddy yield factors: Drought during flowering -30% yield. Blast fungus -25%. BPH infestation -40%. With Hexaconazole treatment yield improves 20%. Proper NPK + Zinc Sulphate boosts yield by 15-20%. Optimal yield: 20-25 Qtl/acre."},
    {"id":"tomato_01","crop":"Tomato","topic":"disease_chemical",
     "text":"Tomato diseases: Early blight — Mancozeb 75WP 2.5g/L. Late blight (humid, cool) — Metalaxyl+Mancozeb 2.5g/L. Leaf curl virus — Imidacloprid 0.5ml/L for whitefly. Fruit borer — Spinosad 45SC 0.3ml/L. Bacterial wilt — no chemical cure, use CO-3 or PKM-1 varieties."},
    {"id":"tomato_02","crop":"Tomato","topic":"weather_chemical",
     "text":"Tomato chemical use by weather: High humidity (>75%) — preventive Mancozeb spray critical. Hot dry (>38°C) — avoid Sulphur, use Neem oil in evenings only. Heavy rain — delay all sprays, check drainage. Fruit drop above 38°C — spray Potassium Nitrate 5g/L to improve set."},
    {"id":"onion_01","crop":"Onion","topic":"disease_chemical",
     "text":"Onion diseases: Purple blotch (high humidity) — Mancozeb 75WP 2.5g/L every 7 days. Thrips — Spinosad 0.3ml/L or Imidacloprid 0.5ml/L. Downy mildew (cool wet) — Metalaxyl+Mancozeb 2.5g/L. Avoid spraying 2 weeks before harvest."},
    {"id":"chilli_01","crop":"Chilli","topic":"disease_chemical",
     "text":"Chilli diseases: Leaf curl (virus via aphids/thrips) — Imidacloprid 0.5ml/L weekly. Anthracnose fruit rot (humid) — Carbendazim 1g/L + Mancozeb 2g/L. Powdery mildew — Sulphur 80WP 3g/L (below 35°C only). Bacterial wilt — Copper Oxychloride 3g/L soil drench."},
    {"id":"mango_01","crop":"Mango","topic":"disease_chemical",
     "text":"Mango spray schedule: Pre-flowering — Carbendazim 1g/L for anthracnose. At bud break — Sulphur 80WP 3g/L for powdery mildew. At flowering — NO sprays (bee activity). Post fruit set — Mancozeb 2.5g/L. Mango hopper — Imidacloprid 0.5ml/L before flowering only."},
    {"id":"banana_01","crop":"Banana","topic":"disease_chemical",
     "text":"Banana diseases: Sigatoka leaf spot (high humidity) — Mancozeb 75WP 2.5g/L every 3 weeks. Bunchy top virus — remove plant immediately, no chemical cure. Fusarium wilt — no cure, use Cavendish varieties. Aphids — Neem oil 5ml/L. Rhizome weevil — Chlorpyrifos soil drench 2.5ml/L."},
    {"id":"coconut_01","crop":"Coconut","topic":"disease_chemical",
     "text":"Coconut treatment: Bud rot — Bordeaux mixture (copper sulphate 1%) pour into crown. Root wilt — no cure, nutrition management. Rhinoceros beetle — Chlorpyrifos 0.1% pour in crown + pheromone traps. Red palm weevil — inject Monocrotophos 10ml per palm + pheromone traps. Spray Neem oil for leaf eating caterpillars."},
    {"id":"groundnut_01","crop":"Groundnut","topic":"disease_chemical",
     "text":"Groundnut diseases: Tikka leaf spot — Mancozeb 75WP 2.5g/L every 10 days from 30 DAS. Stem rot — Carbendazim 1g/L soil drench. Aphids — Imidacloprid 0.5ml/L. Zinc deficiency — Zinc Sulphate 10kg/acre soil or 3g/L foliar. Collar rot — Thiram seed treatment 3g/kg seed."},
    {"id":"weather_01","crop":"General","topic":"weather_chemical",
     "text":"Weather-based chemical rules: Heavy rain (>40mm/week) — skip all foliar sprays, wait 48 hrs after rain. High humidity (>80%) — high fungal risk, spray preventive fungicide immediately. Temperature >38°C — avoid Sulphur compounds and Copper-based fungicides (leaf burn). Wind >15 km/h — delay spraying (drift and uneven coverage). Apply chemicals in early morning (6–9 AM) or evening (4–6 PM)."},
    {"id":"weather_02","crop":"General","topic":"yield_chemical",
     "text":"Yield improvement through timely chemical use: Fungicide applied at disease onset improves yield 15-25%. Insecticide for vector control prevents 20-30% yield loss. Foliar nutrition (Urea 2%, Zinc Sulphate 0.5%) at critical stages improves yield 10-15%. Excessive chemical use (>recommended dose) reduces yield due to phytotoxicity. IPM approach combining bio and chemical gives best results."},
    {"id":"turmeric_01","crop":"Turmeric","topic":"disease_chemical",
     "text":"Turmeric diseases: Rhizome rot (Pythium) — Metalaxyl+Mancozeb 2.5g/L soil drench. Leaf blotch — Mancozeb 75WP 2.5g/L. Shoot borer — Chlorpyrifos 2.5ml/L. Thrips — Spinosad 0.3ml/L. Apply lime before planting for soil pH correction. Rhizome treatment with Carbendazim 1g/L before storage reduces post-harvest loss 30%."},
    {"id":"market_01","crop":"General","topic":"market",
     "text":"Tamil Nadu market intelligence: Avoid selling during harvest peak. Chennai Koyambedu benchmark price. Direct to Uzhavar Sandhai for 10-15% premium. e-NAM online platform available. PM-KISAN Rs.6000/year benefit. Crop insurance under CMCCCI available."},
    {"id":"yield_01","crop":"General","topic":"yield_improvement",
     "text":"Yield improvement tips: Certified seeds +30%. Soil testing every 3 years. Balanced NPK + micronutrients. Drip irrigation saves 40% water, improves yield 20-30%. Timely chemical application at disease onset critical — delay by 5 days reduces efficacy by 40%. IPM reduces input cost 25% while maintaining yield."},
]


# ══════════════════════════════════════════════════════════════════
# VECTORIZER (TF-IDF style, no GPU needed)
# ══════════════════════════════════════════════════════════════════
class SimpleVectorizer:
    def __init__(self, docs):
        self.docs  = docs
        self.vocab = self._build_vocab(docs)

    def _build_vocab(self, docs):
        from collections import Counter
        words = []
        for d in docs:
            words.extend(self._tok(d["text"] + " " + d.get("crop","") + " " + d.get("topic","")))
        counts = Counter(words)
        return {w: i for i, (w, _) in enumerate(counts.most_common(600))}

    def _tok(self, text):
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def vec(self, text):
        v = np.zeros(len(self.vocab))
        for w in self._tok(text):
            if w in self.vocab:
                v[self.vocab[w]] += 1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def search(self, query, top_k=3):
        qv = self.vec(query)
        scores = [(float(np.dot(qv, self.vec(
            d["text"] + " " + d.get("crop","") + " " + d.get("topic","")
        ))), i) for i, d in enumerate(self.docs)]
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]


# ══════════════════════════════════════════════════════════════════
# RAG ENGINE  —  powered by Anthropic Claude
# ══════════════════════════════════════════════════════════════════
class RAGEngine:
    def __init__(self):
        self.vectorizer = SimpleVectorizer(KNOWLEDGE_BASE)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        # ── Bootstrap DB knowledge base on first run ──────────
        try:
            from database import get_db
            db = get_db()
            if db.is_connected:
                db.sync_knowledge_base(KNOWLEDGE_BASE)
        except Exception:
            pass  # DB unavailable — use in-memory fallback

    def retrieve(self, query: str, crop: str = "", top_k: int = 3) -> list:
        return self.vectorizer.search(f"{crop} {query}".strip(), top_k=top_k)

    # ── NEW: diagnose a farmer's problem → separate solution + recommendations ──
    def diagnose_problem(self, problem: str, context: dict) -> dict:
        """
        Given the farmer's problem description, returns a dict with:
          { "solution": str, "recommendations": str }
        Both are plain text / markdown, displayed in separate panels in the UI.
        Falls back to rule-based if no API key.
        """
        crop       = context.get("crop", "")
        district   = context.get("district", "")
        weather    = context.get("weather", {})
        prices     = context.get("prices", {})
        yield_loss = context.get("yield_loss_pct", 0)

        docs      = self.retrieve(problem, crop, top_k=3)
        knowledge = "\n\n".join([f"[{d['topic'].upper()}] {d['text']}" for d in docs])

        chem_recs = recommend_chemicals(crop, weather, yield_loss)
        chem_lines = []
        for c in chem_recs["recommended"][:3]:
            chem_lines.append(
                f"- {c['name']} ({c['type']}): {c['dose_per_litre']} per litre | "
                f"{c['dose_per_acre']} | Timing: {c['timing']} | +{c['yield_gain_pct']}% yield"
            )
        avoid_lines = [f"- {a['name']}: {'; '.join(a['reasons'])}" for a in chem_recs["avoid"][:2]]

        prompt = f"""You are AgriAssist+, a friendly agricultural advisor helping Tamil Nadu farmers via WhatsApp.
A farmer has described a crop problem in simple words. Respond in simple, clear Tamil Nadu farmer language — like a trusted local agricultural officer explaining to a village farmer.

IMPORTANT RULES:
- Use simple words. No scientific jargon. No complex English.
- Write like you are talking directly to the farmer ("Your crop has...", "You should...")
- Keep it short and practical — farmers read this on WhatsApp
- Give exact steps, not vague advice
- Mention dose clearly: "mix X in 1 litre water" style

FARMER CONTEXT:
- Crop: {crop} | District: {district}
- Yield Loss Risk: {yield_loss:.1f}%
- Weather: Temp {weather.get('temp_max','N/A')}°C | Humidity {weather.get('humidity','N/A')}% | Rainfall {weather.get('rainfall','N/A')}mm

WEATHER-SAFE CHEMICALS (safe to use now):
{chr(10).join(chem_lines) if chem_lines else 'No sprays recommended — weather not suitable'}

CHEMICALS TO AVOID NOW:
{chr(10).join(avoid_lines) if avoid_lines else 'None'}

KNOWLEDGE BASE:
{knowledge}

FARMER'S PROBLEM:
{problem}

Respond ONLY in this exact JSON format (no markdown fences, no extra text):
{{
  "solution": "In 2-3 simple sentences: Tell the farmer WHAT disease/pest this is (use common name), WHY it happened (simple reason like weather/humidity), and WHAT they should do TODAY as first step. Example style: 'Your tomato has early blight disease. This happens when weather is humid and leaves stay wet. Today, remove the badly affected leaves and prepare to spray medicine tomorrow morning.'",
  "recommendations": "Give 3-4 numbered action steps the farmer can follow easily. Step 1 should be the most urgent. Include: medicine name + how much to mix in 1 litre water + when to spray (morning/evening). Last step: one simple tip to prevent this problem next time. Write like giving instructions to a neighbour."
}}"""

        if not self.api_key or self.api_key.strip() in ("", "your_anthropic_api_key_here"):
            result = self._rule_based_diagnose(problem, context, chem_recs)
        else:
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      "claude-sonnet-4-6",
                        "max_tokens": 1200,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=45,
                )
                resp.raise_for_status()
                data = resp.json()
                text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text").strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                parsed = json.loads(text)
                result = {
                    "solution":        parsed.get("solution", "").strip(),
                    "recommendations": parsed.get("recommendations", "").strip(),
                }
            except Exception:
                result = self._rule_based_diagnose(problem, context, chem_recs)

        # ── Save query + response to DB ───────────────────────
        try:
            from database import get_db
            db = get_db()
            if db.is_connected:
                session_id = context.get("session_id") or hashlib.md5(f"{crop}_{district}".encode()).hexdigest()[:16]
                db.save_query(session_id, {
                    "crop":               crop,
                    "district":           district,
                    "query_type":         "diagnose",
                    "user_query":         problem,
                    "ai_solution":        result.get("solution"),
                    "ai_recommendations": result.get("recommendations"),
                    "weather_snapshot":   weather,
                    "price_snapshot":     prices,
                    "yield_loss_pct":     yield_loss,
                })
        except Exception:
            pass

        return result

    def _rule_based_diagnose(self, problem: str, context: dict, chem_recs: dict = None) -> dict:
        """Fallback when no API key is set."""
        crop       = context.get("crop", "your crop")
        weather    = context.get("weather", {})
        yield_loss = context.get("yield_loss_pct", 0)
        prices     = context.get("prices", {})

        if chem_recs is None:
            chem_recs = recommend_chemicals(crop, weather, yield_loss)

        # ── Solution block ──
        prob_lower = problem.lower()
        if any(w in prob_lower for w in ["yellow","yellowing","pale"]):
            cause = "Yellowing is typically caused by nitrogen deficiency, fungal leaf spot, or waterlogging. Check roots for rot and soil moisture first."
        elif any(w in prob_lower for w in ["white powder","powdery","mildew"]):
            cause = "White powder on leaves is powdery mildew, a fungal disease that thrives in humid conditions with poor air circulation."
        elif any(w in prob_lower for w in ["rot","rotting","stem rot","root rot"]):
            cause = "Stem or root rot is caused by Fusarium or Pythium fungi, usually triggered by waterlogged soil or poor drainage."
        elif any(w in prob_lower for w in ["wilt","wilting","drooping"]):
            cause = "Wilting can be due to bacterial wilt, Fusarium wilt, or water stress. Check for brown discoloration inside the stem."
        elif any(w in prob_lower for w in ["insect","pest","bug","holes","borer"]):
            cause = "Insect damage detected. Identify the pest (chewing vs sucking) before selecting treatment. Inspect undersides of leaves."
        else:
            cause = f"Based on the symptoms described, this appears to be a disease or stress condition affecting your {crop}. Immediate inspection and treatment is advised."

        solution = (
            f"🔍 **Diagnosis for {crop}:** {cause} "
            f"Current weather (Humidity: {weather.get('humidity','N/A')}%, Temp: {weather.get('temp_max','N/A')}°C) "
            f"{'is increasing disease pressure.' if weather.get('humidity',0) > 75 else 'is currently manageable.'} "
            f"Act within 24–48 hours to prevent further spread."
        )

        # ── Recommendations block ──
        rec_lines = []
        if chem_recs["recommended"]:
            top = chem_recs["recommended"][0]
            rec_lines.append(
                f"1. **Apply {top['name']}** ({top['type']}) — "
                f"Dose: {top['dose_per_litre']} per litre | {top['dose_per_acre']}. "
                f"Apply {top['timing']}. Expected yield recovery: +{top['yield_gain_pct']}%."
            )
        else:
            rec_lines.append("1. **No chemical spray recommended right now** due to current weather. Wait for dry conditions.")

        rec_lines.append("2. **Remove severely affected plant parts** immediately to reduce spread of infection.")
        rec_lines.append("3. **Improve drainage** if soil is waterlogged — standing water accelerates most fungal and bacterial diseases.")
        rec_lines.append("4. **Monitor daily for 7 days** — if symptoms persist or worsen, escalate to a systemic fungicide.")

        current_p = prices.get("current", 0) or 0
        future_p  = prices.get("1month", 0) or 0
        if future_p > current_p:
            rec_lines.append(f"5. **Price tip:** {crop} prices are forecast to rise. Protect your crop now to maximise revenue at Rs.{future_p:,.0f}/Qtl next month.")
        else:
            rec_lines.append(f"5. **Price tip:** Consider selling soon — current price is Rs.{current_p:,.0f}/Qtl. Visit Uzhavar Sandhai for 10–15% premium over APMC.")

        recommendations = "\n\n".join(rec_lines)

        return {"solution": solution, "recommendations": recommendations}

    # ── existing recommend() method (used by the topic dropdowns) ──
    def recommend(self, query: str, context: dict) -> str:
        crop       = context.get("crop", "")
        district   = context.get("district", "")
        weather    = context.get("weather", {})
        prices     = context.get("prices", {})
        yield_loss = context.get("yield_loss_pct", 0)

        docs      = self.retrieve(query, crop, top_k=3)
        knowledge = "\n\n".join([f"[{d['topic'].upper()}] {d['text']}" for d in docs])

        chem_recs = recommend_chemicals(crop, weather, yield_loss)
        chem_text = ""
        if chem_recs["recommended"]:
            lines = []
            for c in chem_recs["recommended"][:3]:
                lines.append(
                    f"- {c['name']} ({c['type']}): {c['dose_per_litre']} per litre / "
                    f"{c['dose_per_acre']}. Timing: {c['timing']}. "
                    f"Expected yield gain: +{c['yield_gain_pct']}%"
                )
            chem_text = "RECOMMENDED CHEMICALS FOR CURRENT WEATHER:\n" + "\n".join(lines)

        avoid_text = ""
        if chem_recs["avoid"]:
            lines = [f"- {a['name']}: {'; '.join(a['reasons'])}" for a in chem_recs["avoid"][:2]]
            avoid_text = "CHEMICALS TO AVOID NOW:\n" + "\n".join(lines)

        # ── Build a focused prompt based on the topic of the query ──
        q_lower = query.lower()

        if any(k in q_lower for k in ["sell", "wait", "price", "market"]):
            topic_instruction = f"""Answer ONLY the question: Should the farmer sell now or wait?

Base your answer strictly on these price numbers:
- Current price: Rs.{prices.get('current', 'N/A')}/Qtl
- 1 Week forecast: Rs.{prices.get('1week', 'N/A')}/Qtl
- 1 Month forecast: Rs.{prices.get('1month', 'N/A')}/Qtl
- Yield loss risk: {yield_loss:.1f}%

Give:
1. A clear SELL NOW or WAIT recommendation with reason
2. Best time/market to sell (e.g. Uzhavar Sandhai for premium)
3. How yield loss risk affects the decision

Do NOT talk about chemicals, diseases, or weather unless directly relevant to the sell decision.
Keep under 180 words."""

        elif any(k in q_lower for k in ["disease", "watch", "treat", "pest", "symptom"]):
            topic_instruction = f"""Answer ONLY the question: What diseases or pests should this farmer watch for and how to treat them?

Focus on:
1. Top 2-3 diseases/pests most common for {crop} in {district} given current weather (Temp {weather.get('temp_max','N/A')}°C, Humidity {weather.get('humidity','N/A')}%, Rainfall {weather.get('rainfall','N/A')}mm)
2. Key symptoms to watch for (simple descriptions)
3. Treatment for each: chemical name + dose per litre + timing

Use the knowledge base and chemical data provided.
Do NOT talk about price, selling decisions, or yield loss percentage.
Keep under 200 words."""

        elif any(k in q_lower for k in ["profit", "improve", "income", "earn", "revenue"]):
            topic_instruction = f"""Answer ONLY the question: How can this farmer improve their profit for {crop}?

Focus on:
1. Price strategies (best market, best time to sell, grading/quality tips)
2. Input cost reduction tips
3. One practical step to increase yield or reduce loss

Use these price forecasts:
- Current: Rs.{prices.get('current', 'N/A')}/Qtl → 1 Month: Rs.{prices.get('1month', 'N/A')}/Qtl

Do NOT talk about specific chemical doses or disease names unless directly tied to profit improvement.
Keep under 180 words."""

        elif any(k in q_lower for k in ["weather", "rain", "temperature", "humid", "climate"]):
            topic_instruction = f"""Answer ONLY the question: Is the current weather good for this crop and what should the farmer do?

Current weather data:
- Temperature: {weather.get('temp_max','N/A')}°C
- Humidity: {weather.get('humidity','N/A')}%
- Rainfall (7-day): {weather.get('rainfall','N/A')}mm

Focus on:
1. Whether these conditions are good or risky for {crop} right now
2. Specific weather-related risks (e.g. fungal disease from humidity, heat stress, waterlogging)
3. 2-3 actions the farmer should take based on current weather

Do NOT talk about prices, selling, or general crop advice unrelated to weather.
Keep under 180 words."""

        elif any(k in q_lower for k in ["harvest", "when", "time to harvest", "ready"]):
            topic_instruction = f"""Answer ONLY the question: When is the best time to harvest {crop} and how should the farmer decide?

Focus on:
1. Signs that {crop} is ready to harvest (visual/physical indicators)
2. Best time of day/season to harvest for quality
3. How current weather ({weather.get('temp_max','N/A')}°C, Humidity {weather.get('humidity','N/A')}%) affects harvest timing
4. Post-harvest handling tip to preserve quality

Do NOT talk about chemicals, diseases, or selling prices.
Keep under 180 words."""

        elif any(k in q_lower for k in ["yield", "loss", "reduce", "production"]):
            topic_instruction = f"""Answer ONLY the question: How can this farmer reduce yield loss for {crop}?

Current yield loss risk: {yield_loss:.1f}%

Focus on:
1. The most likely causes of yield loss for {crop} right now given the weather
2. Top 3 practical steps to reduce yield loss immediately
3. One preventive measure for the next growing season

Do NOT give general crop advice, price info, or selling tips.
Keep under 180 words."""

        else:
            topic_instruction = f"""Answer this specific question for the farmer: {query}

Be direct and practical. Give only information relevant to the question asked.
Farmer context: {crop} in {district}, yield loss risk {yield_loss:.1f}%.
Keep under 180 words."""

        prompt = f"""You are AgriAssist+, a senior agricultural expert for Tamil Nadu farmers.
Give practical, specific advice a farmer can act on today. Use simple language.

FARMER CONTEXT:
- Crop: {crop} | District: {district}
- Current Price: Rs.{prices.get('current','N/A')}/Qtl | 1-Month Forecast: Rs.{prices.get('1month','N/A')}/Qtl
- Predicted Yield Loss: {yield_loss:.1f}%
- Weather: {weather.get('temp_max','N/A')}°C, Humidity {weather.get('humidity','N/A')}%, Rainfall {weather.get('rainfall','N/A')}mm (7-day)

{chem_text}

{avoid_text}

KNOWLEDGE BASE:
{knowledge}

INSTRUCTION:
{topic_instruction}"""

        if not self.api_key or self.api_key.strip() in ("", "your_anthropic_api_key_here"):
            answer = self._rule_based(context, docs, query, chem_recs)
        else:
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      "claude-sonnet-4-6",
                        "max_tokens": 900,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=45,
                )
                resp.raise_for_status()
                data = resp.json()
                text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                answer = "\n".join(text_blocks).strip()
            except Exception:
                answer = self._rule_based(context, docs, query, chem_recs)

        # ── Save query + response to DB ───────────────────────
        try:
            from database import get_db
            db = get_db()
            if db.is_connected:
                session_id = context.get("session_id") or hashlib.md5(f"{crop}_{district}".encode()).hexdigest()[:16]
                db.save_query(session_id, {
                    "crop":           crop,
                    "district":       district,
                    "query_type":     "recommend",
                    "user_query":     query,
                    "ai_solution":    answer,
                    "weather_snapshot": weather,
                    "price_snapshot":   prices,
                    "yield_loss_pct": yield_loss,
                })
        except Exception:
            pass

        return answer

    def _rule_based(self, context: dict, docs: list, query: str, chem_recs: dict = None) -> str:
        crop       = context.get("crop", "your crop")
        prices     = context.get("prices", {})
        weather    = context.get("weather", {})
        yield_loss = context.get("yield_loss_pct", 0)

        if chem_recs is None:
            chem_recs = recommend_chemicals(crop, weather, yield_loss)

        current = prices.get("current", 0) or 0
        p1w     = prices.get("1week", 0) or 0
        future  = prices.get("1month", 0) or 0
        diff    = future - current
        pct     = (diff / current * 100) if current > 0 else 0
        q_lower = query.lower()

        # ── SELL / WAIT ──
        if any(k in q_lower for k in ["sell", "wait", "price", "market"]):
            if pct > 8 and yield_loss <= 15:
                decision = f"⏳ **WAIT — prices rising {pct:.1f}% next month** (Rs.{future:,.0f}/Qtl). Low yield risk means you can hold safely."
            elif pct > 3 and yield_loss <= 20:
                decision = f"📅 **WAIT 1 WEEK** — prices forecast up {pct:.1f}% (Rs.{p1w:,.0f}/Qtl in 1 week). Moderate risk — sell within 7 days."
            elif yield_loss > 25:
                decision = f"🚨 **SELL NOW** — high yield loss risk ({yield_loss:.0f}%) means delay costs more than any price gain."
            else:
                decision = f"✅ **SELL NOW** — prices stable at Rs.{current:,.0f}/Qtl with no major rise forecast."
            return (
                f"{decision}\n\n"
                f"📊 **Price Summary:**\n"
                f"  • Current: Rs.{current:,.0f}/Qtl\n"
                f"  • 1 Week:  Rs.{p1w:,.0f}/Qtl\n"
                f"  • 1 Month: Rs.{future:,.0f}/Qtl\n\n"
                f"🛒 **Best markets:** Uzhavar Sandhai gives 10–15% premium over APMC. e-NAM platform available for online selling."
            )

        # ── DISEASES ──
        elif any(k in q_lower for k in ["disease", "watch", "treat", "pest", "symptom"]):
            parts = [f"🦠 **Common diseases to watch for in {crop}:**\n"]
            for doc in docs:
                if "disease" in doc.get("topic", "") or "chemical" in doc.get("topic", ""):
                    parts.append(doc["text"][:300])
                    break
            if chem_recs["recommended"]:
                top = chem_recs["recommended"][0]
                parts.append(f"\n💊 **Treatment now:** {top['name']} — Mix {top['dose_per_litre']} per litre. {top['timing']}.")
            return "\n".join(parts)

        # ── PROFIT ──
        elif any(k in q_lower for k in ["profit", "improve", "income", "earn", "revenue"]):
            tips = [
                f"💰 **Profit tips for {crop}:**",
                f"1. Sell at Uzhavar Sandhai for 10–15% more than APMC price.",
                f"2. Grade your produce — A-grade fetches Rs.{int(current*1.15):,}/Qtl vs Rs.{current:,.0f}/Qtl for mixed grade.",
                f"3. Reduce input cost: IPM (neem oil + targeted spray) cuts chemical cost by 25%.",
            ]
            if pct > 5:
                tips.append(f"4. Hold stock if possible — prices rising to Rs.{future:,.0f}/Qtl next month (+{pct:.1f}%).")
            else:
                tips.append(f"4. Sell now — prices stable, no major rise expected.")
            return "\n".join(tips)

        # ── WEATHER ──
        elif any(k in q_lower for k in ["weather", "rain", "temperature", "humid", "climate"]):
            notes = "\n".join(f"  • {n}" for n in chem_recs["weather_notes"])
            temp = weather.get("temp_max", "N/A")
            hum  = weather.get("humidity", "N/A")
            rain = weather.get("rainfall", "N/A")
            status = "✅ Favourable" if not chem_recs["weather_notes"][0].startswith("✅") is False else "⚠️ Caution needed"
            return (
                f"🌦️ **Weather for {crop} — {context.get('district','')}:**\n"
                f"  • Temp: {temp}°C | Humidity: {hum}% | Rainfall: {rain}mm\n\n"
                f"**Weather alerts:**\n{notes}"
            )

        # ── HARVEST ──
        elif any(k in q_lower for k in ["harvest", "when", "time to harvest", "ready"]):
            return (
                f"🌾 **Harvest timing for {crop}:**\n\n"
                f"1. Harvest when crop shows maturity signs (colour change, firmness, dry outer layers depending on crop).\n"
                f"2. Best time: **early morning** when temperature is coolest — reduces moisture loss and spoilage.\n"
                f"3. Current weather ({weather.get('temp_max','N/A')}°C, {weather.get('humidity','N/A')}% humidity): "
                + ("High humidity — harvest quickly and dry produce to prevent mould." if (weather.get('humidity') or 0) > 75
                   else "Conditions are suitable for harvesting now.") +
                f"\n4. Post-harvest: store in cool, dry, ventilated area. Sort and grade immediately to get better price."
            )

        # ── YIELD LOSS ──
        elif any(k in q_lower for k in ["yield", "loss", "reduce", "production"]):
            icon = "🔴" if yield_loss > 25 else "🟠" if yield_loss > 10 else "🟢"
            steps = [
                f"{icon} **Yield loss risk: {yield_loss:.1f}%**\n",
                f"**Top steps to reduce yield loss:**",
                f"1. Apply preventive fungicide/insecticide at first sign of disease — delay of 5 days reduces efficacy 40%.",
                f"2. Ensure proper drainage — waterlogging causes 20–30% yield loss in most crops.",
                f"3. Use drip irrigation — saves 40% water and improves yield 20–30%.",
            ]
            if chem_recs["recommended"]:
                top = chem_recs["recommended"][0]
                steps.append(f"4. **Apply {top['name']} now** ({top['dose_per_litre']} per litre) — expected yield recovery: +{top['yield_gain_pct']}%.")
            steps.append(f"5. **Next season:** use certified seeds (+30% yield) and get soil tested every 3 years.")
            return "\n".join(steps)

        # ── GENERIC FALLBACK ──
        else:
            doc_text = docs[0]["text"][:300] if docs else "No specific data found."
            return (
                f"📋 **Advice for {crop} — {context.get('district','')}:**\n\n"
                f"{doc_text}\n\n"
                f"Current price: Rs.{current:,.0f}/Qtl | Yield risk: {yield_loss:.1f}%"
            )

    def get_chemical_recommendations(self, crop: str, weather: dict, yield_loss: float) -> dict:
        """Public method for direct chemical recommendation from app.py."""
        return recommend_chemicals(crop, weather, yield_loss)

    def llm_chemical_advice(self, crop: str, disease_input: str, weather: dict,
                             chem_res: dict, yield_loss: float, district: str) -> str:
        """
        Uses Claude Sonnet to generate intelligent, contextual chemical advice
        based on the farmer's described disease/problem + weather + rule-based chemical results.
        Falls back to a formatted rule-based summary if no API key.
        """
        if not self.api_key or self.api_key.strip() in ("", "your_anthropic_api_key_here"):
            return self._rule_based_chemical_summary(crop, disease_input, chem_res, weather, yield_loss)

        rec_lines = []
        for c in chem_res["recommended"][:4]:
            rec_lines.append(
                f"- {c['name']} ({c['type']}): {c['dose_per_litre']} per litre | "
                f"{c['dose_per_acre']} | Timing: {c['timing']} | +{c['yield_gain_pct']}% yield gain"
            )
        avoid_lines = [
            f"- {a['name']}: {'; '.join(a['reasons'])}" for a in chem_res["avoid"][:3]
        ]
        weather_notes = "\n".join(chem_res["weather_notes"])

        prompt = f"""You are AgriAssist+, a helpful chemical advisor for Tamil Nadu farmers using WhatsApp.
A farmer described a crop problem. Give ONLY the chemical recommendation in simple farmer language.

FARMER CONTEXT:
- Crop: {crop} | District: {district}
- Problem: {disease_input}
- Yield Loss Risk: {yield_loss:.1f}%
- Weather: Temp {weather.get('temp_max', 'N/A')}°C | Humidity {weather.get('humidity', 'N/A')}% | Rainfall {weather.get('rainfall', 'N/A')}mm

WEATHER-SAFE CHEMICALS (use only from this list):
{chr(10).join(rec_lines) if rec_lines else 'None safe right now — advise waiting for better weather.'}

WRITE EXACTLY IN THIS FORMAT (4 sections, no extra text):

💊 *Medicine to Use:*
[Name of the chemical. One line only. If no safe chemical available, say 'Wait for weather to improve before spraying'.]

🧪 *How to Mix & Apply:*
[Tell the farmer exactly: how much to mix in 1 litre of water, when to spray (morning/evening), how many times per week. Use simple words like 'Mix 2.5g in 1 litre water. Spray in the morning before 9am. Repeat every 7 days.']

💰 *Will this help your price?:*
[One clear sentence. If treating now can protect yield and help the farmer get a better price or avoid loss, say so simply. Example: 'Yes — treating now protects your crop, so you can sell at full price instead of selling damaged crop cheap.']

⚠️ *Safety Reminder:*
[One practical safety tip. Example: 'Wear gloves and mask while spraying. Do not sell crop for 7 days after spraying.']

Rules:
- Simple English only. No bullet sub-lists. No jargon.
- Keep total response under 200 words.
- Do NOT repeat the farmer's problem back to them.
- Do NOT add any section beyond these 4."""

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-sonnet-4-6",
                    "max_tokens": 900,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=45,
            )
            resp.raise_for_status()
            data = resp.json()
            return "\n".join(
                b["text"] for b in data.get("content", []) if b.get("type") == "text"
            ).strip()
        except Exception:
            return self._rule_based_chemical_summary(crop, disease_input, chem_res, weather, yield_loss)

    def _rule_based_chemical_summary(self, crop: str, disease_input: str,
                                      chem_res: dict, weather: dict, yield_loss: float) -> str:
        """Fallback 4-section summary when LLM is unavailable."""
        lines = []

        # Section 1 — Medicine
        if chem_res["recommended"]:
            top = chem_res["recommended"][0]
            lines.append(f"💊 *Medicine to Use:*")
            lines.append(f"{top['name']} ({top['type']})")
        else:
            lines.append("💊 *Medicine to Use:*")
            lines.append("Wait for weather to improve before spraying. Conditions are not suitable right now.")

        lines.append("")

        # Section 2 — How to Mix & Apply
        lines.append("🧪 *How to Mix & Apply:*")
        if chem_res["recommended"]:
            top = chem_res["recommended"][0]
            lines.append(
                f"Mix {top['dose_per_litre']} in 1 litre of water. "
                f"{top['timing']}. "
                f"Repeat {top['frequency'].lower()}."
            )
        else:
            lines.append("Do not spray now. Wait until rainfall stops and weather is dry for 2 days.")

        lines.append("")

        # Section 3 — Price impact
        lines.append("💰 *Will this help your price?:*")
        if chem_res["recommended"]:
            top = chem_res["recommended"][0]
            gain = top.get("yield_gain_pct", 10)
            if yield_loss > 15:
                lines.append(
                    f"Yes — treating now can recover up to {gain}% of your yield. "
                    f"A healthier crop sells at full market price instead of being sold cheap as damaged crop."
                )
            else:
                lines.append(
                    f"Yes — this spray protects your crop quality. "
                    f"Good quality {crop} always gets a better price at the market."
                )
        else:
            lines.append("Skipping spray now protects crop quality. Spraying in bad weather can damage leaves and reduce your price.")

        lines.append("")

        # Section 4 — Safety
        lines.append("⚠️ *Safety Reminder:*")
        if chem_res["recommended"]:
            top = chem_res["recommended"][0]
            safety = top.get("safety", "Wear gloves and mask while spraying.")
            lines.append(safety)
        else:
            lines.append("Wear gloves and mask whenever handling chemicals. Store chemicals away from children.")

        return "\n".join(lines)


if __name__ == "__main__":
    rag = RAGEngine()
    result = rag.get_chemical_recommendations(
        "Tomato",
        {"temp_max": 36, "humidity": 82, "rainfall": 50, "wind_speed": 8},
        yield_loss=18,
    )
    print("RECOMMENDED:")
    for r in result["recommended"]:
        print(f"  {r['name']} — {r['dose_per_litre']} | yield gain +{r['yield_gain_pct']}%")
    print("\nAVOID:")
    for a in result["avoid"]:
        print(f"  {a['name']} — {a['reasons']}")
    print("\nWEATHER NOTES:")
    for n in result["weather_notes"]:
        print(f"  {n}")
