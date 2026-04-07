from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from src.data_loader import load_dataset
from src.model import build_logistic_regression_model
from src.preprocessing import clean_text


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "sample_sentiment.csv"
MODEL_PATH = Path("/tmp/sentiment_model.joblib")

_model_cache = None


def get_or_train_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if MODEL_PATH.exists():
        _model_cache = joblib.load(MODEL_PATH)
        return _model_cache

    df = load_dataset(str(DATASET_PATH))
    model = build_logistic_regression_model()
    model.fit(df["clean_text"], df["sentiment"])
    joblib.dump(model, MODEL_PATH)
    _model_cache = model
    return model


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Sentiment API is running.",
            "usage": "POST /api/predict with JSON: {\"text\": \"your sentence\"}",
        }
    )


@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
    else:
        text = request.args.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing required 'text' value."}), 400

    model = get_or_train_model()
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.classes_
    confidence = {label: round(float(prob), 4) for label, prob in zip(classes, probabilities)}

    return jsonify(
        {
            "input_text": text,
            "clean_text": cleaned,
            "sentiment": str(prediction),
            "confidence": confidence,
        }
    )

