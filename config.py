"""
config.py - ALL project settings live here, so you only change things in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "dyslexia_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")
SELECTED_FEATURES_PATH = os.path.join(MODELS_DIR, "selected_features.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# SKILL GROUPS FOR THE KAGGLE DATASET (Rello et al., 2020, PLOS ONE)
# The dataset has 32 questions (Q1..Q32). We group them into 4 skill areas.
#   * Q1-Q9   : verified in the paper - reading prerequisites
#               (phonological, alphabetic, syllabic awareness)
#   * Q14-Q17 : verified in the paper - visual search / visual discrimination
#   * Q30-Q32 : verified in the paper - sequential working memory
#   * Q10-Q13 and Q18-Q29 : APPROXIMATE grouping chosen for this prototype.
#               Check the "Data sets" section of the paper and edit if needed.
# ---------------------------------------------------------------------------
SKILL_QUESTION_MAP = {
    "phonological": list(range(1, 10)),                        # Q1-Q9
    "visual": list(range(10, 18)),                             # Q10-Q17
    "executive": list(range(18, 30)),                          # Q18-Q29
    "memory": [30, 31, 32],                                    # Q30-Q32
}

# Each browser mini-game feeds exactly one skill area (same 4 groups as above).
GAME_SKILL_MAP = {
    "letter_hunt": "visual",        # Game 1: find the target letter (b/d, p/q, m/n)
    "symbol_memory": "memory",      # Game 2: remember a symbol sequence
    "odd_one_out": "visual",        # Game 3: spot the different letter
    "pattern_next": "executive",    # Game 4: what comes next in the pattern
    "sound_match": "phonological",  # Game 5: same or different rhythm (auditory)
    "rhyme_time": "phonological",   # Game 6: pick the rhyming word
}

SKILL_LABELS = {
    "phonological": "Phonological Awareness",
    "visual": "Visual Discrimination",
    "memory": "Working Memory",
    "executive": "Executive Function",
}

# The EXACT feature columns the model is trained on (and that the game must produce).
FEATURE_COLUMNS = [
    "age",
    "phonological_score",
    "visual_score",
    "memory_score",
    "executive_score",
    "overall_accuracy",
    "miss_rate",
]
TARGET_COLUMN = "dyslexia"

# ---------------------------------------------------------------------------
# PROTOTYPE SCREENING THRESHOLDS - NOT clinically validated.
# probability < LOW_MAX -> Low,  < MODERATE_MAX -> Moderate,  else High
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {"LOW_MAX": 0.40, "MODERATE_MAX": 0.70}

# A skill score (0-1) below WEAK is listed as a weaker area, at/above STRONG as stronger.
WEAK_SKILL_THRESHOLD = 0.60
STRONG_SKILL_THRESHOLD = 0.75

# Model selection: highest cross-validated recall, among models whose
# cross-validated precision is at least this value (avoids "flag everyone" models).
MIN_CV_PRECISION = 0.20

DISCLAIMER = ("This result is a screening indication and not a diagnosis. "
              "Professional assessment is recommended if concerns persist.")
