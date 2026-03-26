"""
weather.py
===========
Weather integration using Open-Meteo API (FREE, no key needed).
Returns current + 7-day forecast for any Tamil Nadu district.
Weather responses are cached in MySQL for 1 hour.
"""

import requests
from datetime import datetime

# District lat/lon for Tamil Nadu
TN_COORDS = {
    "Chennai":          (13.0827, 80.2707),
    "Coimbatore":       (11.0168, 76.9558),
    "Madurai":          (9.9252, 78.1198),
    "Tiruchirappalli":  (10.7905, 78.7047),
    "Salem":            (11.6643, 78.1460),
    "Tirunelveli":      (8.7139, 77.7567),
    "Erode":            (11.3410, 77.7172),
    "Vellore":          (12.9165, 79.1325),
    "Thoothukudi":      (8.7642, 78.1348),
    "Dharmapuri":       (12.1277, 78.1579),
    "Dindigul":         (10.3673, 77.9803),
    "Cuddalore":        (11.7480, 79.7714),
    "Thanjavur":        (10.7870, 79.1378),
    "Tiruvarur":        (10.7733, 79.6365),
    "Nagapattinam":     (10.7631, 79.8420),
    "Pudukkottai":      (10.3797, 78.8214),
    "Ramanathapuram":   (9.3762, 78.8302),
    "Kanyakumari":      (8.0883, 77.5385),
    "Krishnagiri":      (12.5186, 78.2137),
    "Namakkal":         (11.2189, 78.1674),
    "Karur":            (10.9601, 78.0766),
    "Perambalur":       (11.2333, 78.8833),
    "Ariyalur":         (11.1400, 79.0800),
    "Villupuram":       (11.9390, 79.4926),
    "Tiruvannamalai":   (12.2253, 79.0747),
    "Ranipet":          (12.9229, 79.3325),
    "Tirupathur":       (12.4961, 78.5638),
    "Chengalpattu":     (12.6921, 79.9765),
    "Kallakurichi":     (11.7395, 78.9593),
    "Tenkasi":          (8.9603, 77.3152),
    "Mayiladuthurai":   (11.1015, 79.6548),
}

WEATHER_API = "https://api.open-meteo.com/v1/forecast"

def get_weather(district: str, use_cache: bool = True) -> dict:
    """
    Fetch current + 7-day forecast for a Tamil Nadu district.
    Checks MySQL cache first (TTL 1 hour) before hitting the API.
    Set use_cache=False to force a fresh fetch.
    """
    # ── Try DB cache first ────────────────────────────────────
    if use_cache:
        try:
            from database import get_db
            db = get_db()
            if db.is_connected:
                cached = db.get_cached_weather(district)
                if cached:
                    cached["from_cache"] = True
                    return cached
        except Exception:
            pass  # DB unavailable — fall through to live fetch

    # ── Live fetch from Open-Meteo ────────────────────────────
    coords = TN_COORDS.get(district, (11.0, 78.0))
    lat, lon = coords

    params = {
        "latitude":  lat,
        "longitude": lon,
        "current": [
            "temperature_2m","relative_humidity_2m",
            "precipitation","wind_speed_10m","weather_code",
        ],
        "daily": [
            "temperature_2m_max","temperature_2m_min",
            "precipitation_sum","relative_humidity_2m_max",
        ],
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    try:
        resp = requests.get(WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        daily   = data.get("daily", {})

        forecast_days = []
        dates    = daily.get("time", [])
        tmax     = daily.get("temperature_2m_max", [])
        tmin     = daily.get("temperature_2m_min", [])
        rain     = daily.get("precipitation_sum", [])
        humidity = daily.get("relative_humidity_2m_max", [])

        for i in range(len(dates)):
            forecast_days.append({
                "date":     dates[i],
                "temp_max": tmax[i] if i < len(tmax) else None,
                "temp_min": tmin[i] if i < len(tmin) else None,
                "rainfall": rain[i] if i < len(rain) else 0,
                "humidity": humidity[i] if i < len(humidity) else None,
            })

        avg_rain = sum(d["rainfall"] or 0 for d in forecast_days) / max(len(forecast_days), 1)
        avg_tmax = sum(d["temp_max"] or 30 for d in forecast_days) / max(len(forecast_days), 1)
        avg_tmin = sum(d["temp_min"] or 22 for d in forecast_days) / max(len(forecast_days), 1)
        avg_hum  = sum(d["humidity"] or 60 for d in forecast_days) / max(len(forecast_days), 1)

        result = {
            "success":    True,
            "district":   district,
            "lat":        lat,
            "lon":        lon,
            "from_cache": False,
            "current": {
                "temperature":   current.get("temperature_2m"),
                "humidity":      current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "wind_speed":    current.get("wind_speed_10m"),
                "weather_code":  current.get("weather_code"),
            },
            "forecast": forecast_days,
            "avg_7day": {
                "rainfall": round(avg_rain, 2),
                "temp_max": round(avg_tmax, 1),
                "temp_min": round(avg_tmin, 1),
                "humidity": round(avg_hum, 1),
            },
        }

        # ── Save to DB cache ──────────────────────────────────
        if use_cache:
            try:
                from database import get_db
                db = get_db()
                if db.is_connected:
                    db.save_weather_cache(district, result, ttl_minutes=60)
            except Exception:
                pass

        return result

    except Exception as e:
        return {
            "success":    False,
            "error":      str(e),
            "district":   district,
            "from_cache": False,
            "avg_7day":   {"rainfall": 30, "temp_max": 32, "temp_min": 24, "humidity": 60},
        }


def weather_code_label(code: int) -> str:
    """Convert WMO weather code to readable label."""
    if code is None: return "Unknown"
    if code == 0:    return "☀️ Clear Sky"
    if code <= 3:    return "⛅ Partly Cloudy"
    if code <= 9:    return "🌫️ Foggy"
    if code <= 19:   return "🌦️ Drizzle"
    if code <= 29:   return "🌧️ Rain"
    if code <= 39:   return "❄️ Snow"
    if code <= 49:   return "🌫️ Freezing Fog"
    if code <= 59:   return "🌦️ Drizzle"
    if code <= 69:   return "🌧️ Rain"
    if code <= 79:   return "🌨️ Snow"
    if code <= 84:   return "🌧️ Heavy Rain"
    if code <= 94:   return "⛈️ Thunderstorm"
    return "⛈️ Severe Storm"


if __name__ == "__main__":
    result = get_weather("Coimbatore")
    print(result)
