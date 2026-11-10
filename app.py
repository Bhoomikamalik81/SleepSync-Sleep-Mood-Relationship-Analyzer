"""
Mindora Sleep Check API

Loads the trained model (best_model.pkl) and the feature description
(model_info.json) that were created by Sleep_Mood_Analysis.ipynb.

Run:   python app.py
Test:  python test_api.py
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------------------------------------------------------------- load model
model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
with open(os.path.join(BASE_DIR, "model_info.json")) as f:
    info = json.load(f)

FEATURES = info["features"]
SLEEP_COLS = info["sleep_questions"]["columns"]
SLEEP_RANGE = info["sleep_questions"]["answer_range"]
STRESS_COLS = info["stress_questions"]["columns"]
STRESS_RANGE = info["stress_questions"]["answer_range"]
AGE_MIN, AGE_MAX = info["age_range"]
CATEGORIES = info["categories"]
STEADY_MAX = info["risk_levels"]["disturbed"][0]            # 40
ATTENTION_MIN = info["risk_levels"]["needs_attention"][0]   # 60

# ---------------------------------------------------------------- text used in the result
SLEEP_CONCERN = {
    "isi_1": "Difficulty falling asleep",
    "isi_2": "Difficulty staying asleep",
    "isi_3": "Waking up too early",
    "isi_4": "Dissatisfaction with sleep",
    "isi_5": "Sleep problem noticeable to others",
    "isi_6": "Worry about sleep",
    "isi_7": "Sleep problem affects daily life",
}

MESSAGES = {
    "steady": {
        "title": "Your sleep and stress look steady",
        "message": "Your answers are similar to students who usually report good mood. Keep your routine going.",
    },
    "disturbed": {
        "title": "Your sleep or stress needs a little care",
        "message": "Your answers are similar to some students who report low mood. A fixed bedtime and less screen time at night may help.",
    },
    "needs_attention": {
        "title": "Your sleep and stress need some attention",
        "message": "Your answers are similar to students who often report low mood. Try a fixed sleep time this week and look at the coping tips.",
    },
}

SUPPORT_MESSAGE = "If you have felt low for a while, consider talking to someone you trust or a counsellor."

# Show this on every result, not only on "needs_attention".
# Please confirm the number on the official Tele-MANAS page before launch.
HELPLINE = {
    "name": "Tele-MANAS",
    "number": "14416",
    "number_alternative": "1-800-891-4416",
    "hours": "24x7, free and confidential",
    "emergency_number": "112",
}

DISCLAIMER = (
    "This is a wellness check, not a medical diagnosis. "
    "The score shows how your answers compare with other students."
)


# ---------------------------------------------------------------- CORS (so the frontend can call the API)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("ALLOWED_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------- helpers
def is_whole_number(value):
    """True for 3 or 3.0, False for True/False, strings and None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value).is_integer()


def validate(data):
    """Returns a list of error messages. Empty list means the input is fine."""
    errors = []

    for cols, (low, high) in [(SLEEP_COLS, SLEEP_RANGE), (STRESS_COLS, STRESS_RANGE)]:
        for col in cols:
            if col not in data:
                errors.append(f"{col} is missing")
            elif not is_whole_number(data[col]) or not (low <= data[col] <= high):
                errors.append(f"{col} must be a whole number from {low} to {high}")

    if "age" not in data:
        errors.append("age is missing")
    elif isinstance(data["age"], bool) or not isinstance(data["age"], (int, float)) \
            or not (AGE_MIN <= data["age"] <= AGE_MAX):
        errors.append(f"age must be a number from {AGE_MIN} to {AGE_MAX}")

    for col, allowed in CATEGORIES.items():
        if col not in data:
            errors.append(f"{col} is missing")
        elif data[col] not in allowed:
            errors.append(f"{col} must be one of the values from /questions")

    return errors


def level_for(score):
    if score >= ATTENTION_MIN:
        return "needs_attention"
    if score >= STEADY_MAX:
        return "disturbed"
    return "steady"


# ---------------------------------------------------------------- routes
@app.route("/")
def home():
    return jsonify({
        "name": "Mindora Sleep Check API",
        "endpoints": {
            "GET /health": "check that the API is running",
            "GET /questions": "questions, answer ranges and allowed options",
            "POST /predict": "send the answers, get the risk score and message",
        },
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": info["model"]})


@app.route("/questions")
def questions():
    return jsonify({
        "sleep_questions": info["sleep_questions"],
        "stress_questions": info["stress_questions"],
        "age_range": info["age_range"],
        "categories": CATEGORIES,
        "risk_levels": info["risk_levels"],
        "helpline": HELPLINE,
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":      # browser preflight request
        return ("", 204)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON body with the answers."}), 400

    errors = validate(data)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    row = pd.DataFrame([[data[col] for col in FEATURES]], columns=FEATURES)
    score = int(round(float(model.predict_proba(row)[0][1]) * 100))
    level = level_for(score)

    # biggest sleep concern = the sleep question with the highest answer
    worst = max(SLEEP_COLS, key=lambda col: data[col])
    concern = None
    if data[worst] > 0:
        concern = {"question": worst, "text": SLEEP_CONCERN[worst]}

    result = {
        "risk_score": score,
        "level": level,
        "title": MESSAGES[level]["title"],
        "message": MESSAGES[level]["message"],
        "biggest_sleep_concern": concern,
        "support_message": SUPPORT_MESSAGE if level == "needs_attention" else None,
        "helpline": HELPLINE,
        "disclaimer": DISCLAIMER,
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )