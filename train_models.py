"""
train_models.py
================
Trains XGBoost models for AgriAssist+:
  - current_price_model.joblib   → current Modal Price
  - future_1w_model.joblib       → 1-week ahead price
  - future_2w_model.joblib       → 2-week ahead price
  - future_1m_model.joblib       → 1-month ahead price
  - yield_loss_model.joblib      → yield loss % prediction

Run:
    python generate_dataset.py   (first time only)
    python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

DATASET = "tn_agri_dataset.csv"
RANDOM_STATE = 42

PRICE_FEATURES = [
    "Commodity_Code","District_Code","Market_Code","Variety_Code",
    "Month","Week","Season_Code",
    "Min_Price","Max_Price","Price_Spread","Price_Mid","Spread_Ratio",
    "Lag1","Lag2","Lag4","Lag8","Lag12",
    "Rainfall_mm","Temp_Max_C","Temp_Min_C","Humidity_Pct",
]

YIELD_FEATURES = [
    "Commodity_Code","District_Code","Month","Season_Code",
    "Rainfall_mm","Temp_Max_C","Temp_Min_C","Humidity_Pct",
    "Lag1","Lag4","Lag8",
]

TARGETS = {
    "current_price_model.joblib": "Modal_Price",
    "future_1w_model.joblib":     "Future_1Week",
    "future_2w_model.joblib":     "Future_2Week",
    "future_1m_model.joblib":     "Future_1Month",
    "yield_loss_model.joblib":    "Yield_Loss_Pct",
}

def encode_and_save(df):
    encoders = {}
    for col in ["Commodity","District","Market","Variety"]:
        le = LabelEncoder()
        df[f"{col}_Code"] = le.fit_transform(df[col].astype(str))
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    joblib.dump(encoders, "encoders.joblib")
    print("  Encoders saved → encoders.joblib")
    return df, encoders

def train_one(df, features, target, out_file):
    X = df[features]
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    preds = model.predict(X_te)
    r2   = r2_score(y_te, preds)
    mape = mean_absolute_percentage_error(y_te, preds) * 100
    joblib.dump(model, out_file)
    print(f"  {out_file:<35} R²={r2:.4f}  MAPE={mape:.2f}%")
    return model

def main():
    if not os.path.exists(DATASET):
        print(f"Dataset '{DATASET}' not found. Running generator...")
        import subprocess, sys
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)

    print(f"\n[LOAD] {DATASET}")
    df = pd.read_csv(DATASET)
    print(f"  {len(df):,} rows  {df['Commodity'].nunique()} crops")

    df, encoders = encode_and_save(df)

    print("\n[TRAIN] Models")
    for out_file, target in TARGETS.items():
        feats = YIELD_FEATURES if "yield" in out_file else PRICE_FEATURES
        train_one(df, feats, target, out_file)

    # Save feature column lists
    joblib.dump({
        "price_features": PRICE_FEATURES,
        "yield_features": YIELD_FEATURES,
        "encoders": encoders,
    }, "model_meta.joblib")
    print("\n✅ All models trained and saved.")
    print("   Run:  streamlit run app.py")

if __name__ == "__main__":
    main()
