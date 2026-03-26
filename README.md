# 🌾 AgriAssist+ — Tamil Nadu Crop Intelligence System

A Streamlit AI crop advisory system for Tamil Nadu farmers.  
Predict prices · Assess yield risk · Get AI-powered recommendations — all weather-aware.

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

```bash
cd "D:\projects\agri assist 1"

# First time only
git init
git add .
git commit -m "Initial commit"

# Create a repo on github.com named: agriassist-plus
git remote add origin https://github.com/YOUR_USERNAME/agriassist-plus.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Select your repo: `agriassist-plus`
4. Branch: `main`
5. Main file: `app.py`
6. Click **"Advanced settings"** → paste your secrets (see Step 3)
7. Click **"Deploy!"**

### Step 3 — Add Secrets (API Keys)

In the Streamlit Cloud dashboard:  
Go to **App → Settings → Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"
```

> MySQL is optional. Leave those fields blank and the app will run without a database.

---

## 🏗️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset and train models (first time only)
python generate_dataset.py
python train_models.py

# Run
streamlit run app.py
```

---

## 📦 Project Structure

```
agri assist 1/
├── app.py                    ← Main Streamlit app (5 tabs)
├── weather.py                ← Open-Meteo weather integration
├── rag_engine.py             ← RAG + Claude AI recommendation engine
├── database.py               ← MySQL integration (optional)
├── generate_dataset.py       ← Generates TN crop dataset
├── train_models.py           ← Trains XGBoost models
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   ├── config.toml           ← Streamlit theme config
│   └── secrets.toml.example  ← Secrets template
├── .env.example              ← Local env template
├── .gitignore                ← Keeps .env and secrets out of GitHub
└── README.md
```

---

## 🌾 Features

| Tab | Feature |
|-----|---------|
| 📊 Price Prediction | Current + 1w/2w/1m price forecast, yield risk, sell decision |
| 🌦️ Weather & Yield Risk | Live weather, 7-day forecast, crop stress alerts |
| 🤖 AI Advisor | Diagnose problems, 6-topic deep analysis with Claude AI |
| 🧪 Chemical Advisor | Weather-aware chemical recommendations with dosage |
| 📈 Market Trends | Historical price trends, district comparison |

---

## 🔑 APIs Used

| API | Purpose | Cost |
|-----|---------|------|
| Open-Meteo | Live weather + forecast | **FREE** |
| Anthropic Claude | AI crop advice | API key required |
| MySQL (optional) | Save predictions & history | Optional |
