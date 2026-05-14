    meta_path = os.path.join(BASE_DIR, "model_meta.joblib")
    data_path = os.path.join(BASE_DIR, "tn_agri_dataset.csv")
    META = joblib.load(meta_path) if os.path.exists(meta_path) else None
    DF   = pd.read_csv(data_path) if os.path.exists(data_path) else None
    from rag_engine import RAGEngine
    RAG = RAGEngine()
    print("✅ Models, dataset and RAGEngine loaded.")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

# ── Load CNN detector ────────────────────────────────────────────
try:
    from plant_disease_cnn import PlantDiseaseDetector
    DETECTOR = PlantDiseaseDetector(os.path.join(BASE_DIR, "plant_disease_cnn_model.h5"))
    print(f"✅ Plant disease detector loaded (CNN: {DETECTOR.cnn_available})")
except Exception as e:
    DETECTOR = None
    print(f"⚠️ Disease detector load error: {e}")

from weather import get_weather
from database import get_db