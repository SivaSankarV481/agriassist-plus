"""
generate_dataset.py
====================
Generates Tamil Nadu crop dataset with price history, weather features,
yield loss estimates, and market intelligence for AgriAssist+.

Run:
    python generate_dataset.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────
# Tamil Nadu Crop Varieties (exhaustive local list)
# ─────────────────────────────────────────────────────────────────
TN_CROPS = {
    "Paddy": ["IR-20", "IR-36", "CO-43", "ADT-36", "ADT-43", "BPT-5204", "Ponni", "Samba Masonali", "Kuruvai"],
    "Banana": ["Nendran", "Robusta", "Poovan", "Rasthali", "Karpooravalli", "Monthan", "Red Banana"],
    "Mango":  ["Alphonso", "Banganapalli", "Neelam", "Totapuri", "Malgova", "Imam Pasand", "Rumani"],
    "Coconut":["Tall", "Dwarf", "Hybrid"],
    "Groundnut": ["TMV-2", "TMV-7", "VRI-2", "CO-2", "K-6"],
    "Sugarcane": ["CO-86032", "CO-62175", "COC-671", "CO-7704"],
    "Cotton":  ["MCU-5", "MCU-7", "LRA-5166", "Surabhi"],
    "Tomato":  ["Hybrid", "CO-3", "PKM-1", "Local Red"],
    "Brinjal": ["CO-1", "PLR-1", "Local Purple", "Green Long"],
    "Onion":   ["CO-4", "Aggregatum", "Bellary Red", "Small Onion"],
    "Chilli":  ["CO-1", "CO-4", "K-2", "PKM-1", "Guntur"],
    "Turmeric":["CO-1", "BSR-2", "Salem", "Erode Local"],
    "Tamarind":["Local", "Urigam", "PKM-1"],
    "Drumstick":["PKM-1", "PKM-2", "Local"],
    "Moringa": ["PKM-1", "Local"],
    "Amaranthus":["Local", "CO-1"],
    "Cluster Beans":["CO-1", "Pusa Navbahar", "Local"],
    "Bhindi": ["CO-1", "Arka Anamika", "Local"],
    "Bitter Gourd":["CO-1", "MDU-1", "Local"],
    "Bottle Gourd":["CO-1", "Local"],
    "Snake Gourd":["CO-1", "Local"],
    "Cauliflower":["Improved Japanese", "Snowball", "Local"],
    "Cabbage":["CO-1", "Pride of India", "Local"],
    "Beans":  ["CO-1", "CO-8", "Local"],
    "Potato": ["Kufri Jyoti", "Kufri Badshah", "Local"],
    "Ginger": ["Rio-de-Janeiro", "Maran", "Himachal Local", "Varada"],
    "Garlic": ["Ooty-1", "Local White", "G-1"],
    "Pineapple":["Kew", "Queen", "Mauritius"],
    "Papaya": ["CO-7", "Red Lady", "Local"],
    "Guava":  ["Allahabad Safeda", "CO-2", "Local"],
    "Lemon":  ["PKM-1", "Assam Lemon", "Local"],
    "Sweet Potato":["CO-1", "CO-2", "Local"],
    "Tapioca":["H-97", "CO-2", "Sri Visakham", "Local"],
    "Soybean":["CO-1", "JS-335", "PKV-J-27"],
    "Maize":  ["CO-6", "Hybrid", "Local Yellow"],
    "Ragi":   ["CO-12", "CO-13", "GPU-28", "Local"],
    "Cumbu":  ["CO-9", "CO-10", "TNBH-0449"],
    "Horsegram":["CO-1", "CO-4", "PLR-1"],
    "Cowpea": ["CO-6", "CO-7", "Local"],
    "Black Gram":["VBN-4", "VBN-6", "CO-6"],
    "Green Gram":["CO-6", "CO-7", "VBN-2"],
    "Red Gram": ["CO-6", "ICPL-87109", "APK-1"],
}

TN_DISTRICTS = [
    "Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli",
    "Erode","Vellore","Thoothukudi","Dharmapuri","Dindigul","Cuddalore",
    "Thanjavur","Tiruvarur","Nagapattinam","Pudukkottai","Ramanathapuram",
    "Kanyakumari","Krishnagiri","Namakkal","Karur","Perambalur","Ariyalur",
    "Villupuram","Tiruvannamalai","Ranipet","Tirupathur","Chengalpattu",
    "Kallakurichi","Tenkasi","Mayiladuthurai"
]

TN_MARKETS = {
    "Coimbatore": ["Mettupalayam APMC", "Pollachi APMC", "Coimbatore Main"],
    "Salem":      ["Salem Main APMC", "Attur APMC", "Mettur APMC"],
    "Thanjavur":  ["Thanjavur APMC", "Kumbakonam APMC", "Papanasam APMC"],
    "Madurai":    ["Madurai Main APMC", "Usilampatti APMC"],
    "Erode":      ["Erode Main APMC", "Bhavani APMC", "Gobichettipalayam APMC"],
    "Dharmapuri": ["Karimangalam APMC", "Palacode APMC"],
    "Dindigul":   ["Dindigul Main APMC", "Palani APMC"],
    "Tirunelveli":["Tirunelveli APMC", "Nanguneri APMC"],
    "Krishnagiri":["Krishnagiri APMC", "Hosur APMC", "Bargur APMC"],
    "Namakkal":   ["Namakkal APMC", "Rasipuram APMC"],
    "Villupuram": ["Villupuram APMC", "Tindivanam APMC"],
    "Cuddalore":  ["Cuddalore APMC", "Panruti APMC"],
    "Kanyakumari":["Nagercoil APMC", "Marthandam APMC"],
    "Pudukkottai":["Pudukkottai APMC", "Alangudi APMC"],
    "Tiruchirappalli":["Trichy Main APMC", "Musiri APMC"],
}

SEASON_MAP = {1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
              6:"Monsoon",7:"Monsoon",8:"Monsoon",
              9:"Post-Monsoon",10:"Post-Monsoon",11:"Post-Monsoon",12:"Winter"}
SEASON_CODES = {"Winter":0,"Summer":1,"Monsoon":2,"Post-Monsoon":3}

# Base price ranges per crop category (Rs/Qtl)
BASE_PRICES = {
    "Paddy":(1600,2200),"Banana":(1500,4000),"Mango":(2000,6000),
    "Coconut":(8000,18000),"Groundnut":(4500,7000),"Sugarcane":(280,350),
    "Cotton":(5500,7500),"Tomato":(500,4000),"Brinjal":(500,2500),
    "Onion":(800,3500),"Chilli":(4000,12000),"Turmeric":(6000,14000),
    "Tamarind":(4000,9000),"Drumstick":(2000,6000),"Moringa":(2000,5000),
    "Amaranthus":(800,2500),"Cluster Beans":(1500,5000),"Bhindi":(1000,3500),
    "Bitter Gourd":(1000,3000),"Bottle Gourd":(500,2000),"Snake Gourd":(800,2500),
    "Cauliflower":(600,2500),"Cabbage":(300,1500),"Beans":(1500,5000),
    "Potato":(1200,3000),"Ginger":(3000,8000),"Garlic":(3000,10000),
    "Pineapple":(1500,3500),"Papaya":(800,2500),"Guava":(1000,3000),
    "Lemon":(2000,6000),"Sweet Potato":(1000,2500),"Tapioca":(500,1500),
    "Soybean":(3000,5000),"Maize":(1400,2200),"Ragi":(1800,3000),
    "Cumbu":(1500,2500),"Horsegram":(4000,8000),"Cowpea":(3000,6000),
    "Black Gram":(4500,8000),"Green Gram":(5000,9000),"Red Gram":(5000,9000),
}

def generate_price_series(base_min, base_max, n=500, seed=42):
    rng = np.random.default_rng(seed)
    mid = (base_min + base_max) / 2
    prices = []
    p = mid
    for _ in range(n):
        shock = rng.normal(0, mid * 0.03)
        mean_rev = (mid - p) * 0.05
        p = max(base_min * 0.7, min(base_max * 1.3, p + shock + mean_rev))
        prices.append(round(p, 2))
    return prices

def build_dataset():
    rng = np.random.default_rng(2024)
    rows = []
    start_date = datetime(2020, 1, 1)

    for crop, varieties in TN_CROPS.items():
        bp = BASE_PRICES.get(crop, (1000, 5000))
        prices = generate_price_series(bp[0], bp[1], n=600, seed=hash(crop) % 9999)

        for i in range(500):
            dt = start_date + timedelta(days=i * 3)
            month = dt.month
            week  = int(dt.isocalendar()[1])
            season = SEASON_MAP.get(month, "Winter")

            district = rng.choice(TN_DISTRICTS)
            markets  = TN_MARKETS.get(district, [f"{district} APMC"])
            market   = rng.choice(markets)
            variety  = rng.choice(varieties)

            modal = prices[i]
            min_p = round(modal * rng.uniform(0.85, 0.97), 2)
            max_p = round(modal * rng.uniform(1.03, 1.18), 2)

            # Rainfall (mm) — seasonal pattern
            rainfall = {
                "Monsoon": rng.uniform(80, 200),
                "Post-Monsoon": rng.uniform(30, 120),
                "Winter": rng.uniform(5, 40),
                "Summer": rng.uniform(2, 20),
            }.get(season, 10)
            rainfall = round(rainfall, 1)

            # Temperature
            temp_max = {
                "Summer": rng.uniform(36, 42),
                "Monsoon": rng.uniform(28, 35),
                "Post-Monsoon": rng.uniform(25, 32),
                "Winter": rng.uniform(22, 30),
            }.get(season, 30)
            temp_min = round(temp_max - rng.uniform(5, 10), 1)
            temp_max = round(temp_max, 1)

            # Humidity
            humidity = {
                "Monsoon": rng.uniform(75, 95),
                "Post-Monsoon": rng.uniform(60, 80),
                "Winter": rng.uniform(50, 70),
                "Summer": rng.uniform(30, 55),
            }.get(season, 60)
            humidity = round(humidity, 1)

            # Yield loss % (due to weather/pest stress)
            disease_risk = min(1.0, max(0.0, (humidity - 50) / 50 + rainfall / 300))
            heat_stress  = max(0.0, (temp_max - 38) / 10)
            yield_loss_pct = round(min(80, (disease_risk * 25 + heat_stress * 15 + rng.uniform(0, 10))), 2)

            # Spread features
            spread      = round(max_p - min_p, 2)
            mid_price   = round((max_p + min_p) / 2, 2)
            spread_ratio= round(spread / mid_price if mid_price > 0 else 0, 4)

            # Lag prices
            lag1  = prices[max(0, i-1)]
            lag2  = prices[max(0, i-2)]
            lag4  = prices[max(0, i-4)]
            lag8  = prices[max(0, i-8)]
            lag12 = prices[max(0, i-12)]

            # Future prices (targets)
            fut_1w  = prices[min(len(prices)-1, i+2)]
            fut_2w  = prices[min(len(prices)-1, i+5)]
            fut_1m  = prices[min(len(prices)-1, i+10)]

            rows.append({
                "State":           "Tamil Nadu",
                "District":        district,
                "Market":          market,
                "Commodity":       crop,
                "Variety":         variety,
                "Arrival_Date":    dt.strftime("%Y-%m-%d"),
                "Month":           month,
                "Week":            week,
                "Season":          season,
                "Season_Code":     SEASON_CODES[season],
                "Min_Price":       min_p,
                "Max_Price":       max_p,
                "Modal_Price":     modal,
                "Price_Spread":    spread,
                "Price_Mid":       mid_price,
                "Spread_Ratio":    spread_ratio,
                "Lag1":            lag1,
                "Lag2":            lag2,
                "Lag4":            lag4,
                "Lag8":            lag8,
                "Lag12":           lag12,
                "Rainfall_mm":     rainfall,
                "Temp_Max_C":      temp_max,
                "Temp_Min_C":      temp_min,
                "Humidity_Pct":    humidity,
                "Yield_Loss_Pct":  yield_loss_pct,
                "Future_1Week":    fut_1w,
                "Future_2Week":    fut_2w,
                "Future_1Month":   fut_1m,
            })

    df = pd.DataFrame(rows)
    df.to_csv("tn_agri_dataset.csv", index=False)
    print(f"✅ Dataset saved: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Crops    : {df['Commodity'].nunique()}")
    print(f"   Districts: {df['District'].nunique()}")
    print(f"   Date range: {df['Arrival_Date'].min()} → {df['Arrival_Date'].max()}")
    return df

if __name__ == "__main__":
    build_dataset()
