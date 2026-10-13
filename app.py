"""
app.py - Module 10: Flask backend.   Run:  python app.py   then open http://127.0.0.1:5000

REAL MODEL MODE : models/dyslexia_model.joblib exists (created by train_model.py)
DEMO MODE       : no trained model yet -> a clearly labelled demo estimate is shown instead
"""
import os
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from config import (MODEL_PATH, FEATURE_COLUMNS_PATH, META_PATH, FEATURE_COLUMNS, GAME_SKILL_MAP,
                    SKILL_LABELS, RISK_THRESHOLDS, WEAK_SKILL_THRESHOLD,
                    STRONG_SKILL_THRESHOLD, DISCLAIMER)
from feature_engineering import engineer_game, game_summary

app = Flask(__name__, static_folder="public/static", static_url_path="/static")
CORS(app)
app.json.sort_keys = False  # keep skill order as defined in config

MODEL, META = None, {}


def load_model():
    """Load the saved pipeline once at start-up (never retrain inside the app)."""
    global MODEL, META
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        saved_cols = joblib.load(FEATURE_COLUMNS_PATH)
        assert saved_cols == FEATURE_COLUMNS, "Feature columns changed - retrain the model."
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                META = json.load(f)
        print(f"REAL MODEL MODE - loaded {META.get('model_name', 'model')}")
    else:
        MODEL, META = None, {}
        print("DEMO MODE - no trained model found. Run: python train_model.py")


def risk_level(prob):
    if prob < RISK_THRESHOLDS["LOW_MAX"]:
        return "Low"
    if prob < RISK_THRESHOLDS["MODERATE_MAX"]:
        return "Moderate"
    return "High"


def skill_analysis(features):
    scores = {SKILL_LABELS[s]: round(features[f"{s}_score"], 3) for s in SKILL_LABELS}
    weak = [k for k, v in sorted(scores.items(), key=lambda kv: kv[1]) if v < WEAK_SKILL_THRESHOLD]
    strong = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= STRONG_SKILL_THRESHOLD]
    return scores, weak, strong


def predict_from_features(features):
    """Returns (probability, mode). Uses the real model when available."""
    if MODEL is not None:
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        return float(MODEL.predict_proba(X)[0, 1]), "real"
    # DEMO MODE: simple transparent rule, NOT machine learning and NOT meaningful.
    skills = [features[f"{s}_score"] for s in SKILL_LABELS]
    return round(1 - sum(skills) / len(skills), 3), "demo"


def build_report(features, summary=None):
    prob, mode = predict_from_features(features)
    scores, weak, strong = skill_analysis(features)
    return {
        "mode": mode,
        "model_name": META.get("model_name") if mode == "real" else None,
        "risk_level": risk_level(prob),
        "probability": round(prob, 3),
        "skill_scores": scores,
        "weak_skills": weak,
        "strong_skills": strong,
        "features": {k: round(v, 3) for k, v in features.items()},
        "summary": summary,
        "thresholds": RISK_THRESHOLDS,
        "threshold_note": "Prototype screening thresholds - not clinically validated.",
        "message": DISCLAIMER,
        "demo_note": ("DEMO MODE: no trained model is loaded. This estimate comes from a simple "
                      "rule, not machine learning, and has no screening meaning.") if mode == "demo" else None,
    }


def validate_game_payload(data):
    if not isinstance(data, dict):
        return "Request body must be JSON."
    age = data.get("age")
    if not isinstance(age, (int, float)) or not 5 <= age <= 18:
        return "Age must be a number between 5 and 18."
    results = data.get("results")
    if not isinstance(results, list) or len(results) == 0:
        return "Results must be a non-empty list."
    for i, r in enumerate(results):
        if r.get("game_id") not in GAME_SKILL_MAP:
            return f"Result {i}: unknown game_id '{r.get('game_id')}'."
        for k in ("clicks", "hits", "misses"):
            if not isinstance(r.get(k), (int, float)) or r[k] < 0:
                return f"Result {i}: '{k}' must be a non-negative number."
        if r["hits"] + r["misses"] > r["clicks"]:
            return f"Result {i}: hits + misses cannot exceed clicks."
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "real" if MODEL is not None else "demo",
                    "model_name": META.get("model_name")})


@app.route("/api/submit-game", methods=["POST"])
def submit_game():
    """Raw game results -> same feature engineering as training -> prediction."""
    data = request.get_json(silent=True)
    error = validate_game_payload(data)
    if error:
        return jsonify({"error": error}), 400
    features = engineer_game(data["results"], data["age"])
    return jsonify(build_report(features, game_summary(data["results"])))


@app.route("/api/predict", methods=["POST"])
def predict():
    """Already-engineered features -> prediction (useful for testing with Postman/curl)."""
    data = request.get_json(silent=True) or {}
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400
    try:
        features = {c: float(data[c]) for c in FEATURE_COLUMNS}
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numbers."}), 400
    return jsonify(build_report(features))


load_model()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
