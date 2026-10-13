"""
feature_engineering.py - turns raw question results into skill-level features.

The SAME core function (build_features) is used for:
  A. every row of the training dataset, and
  B. the live results sent by the browser game.
This guarantees the model sees identical features in training and deployment.

Formulas (identical to how the Rello dataset defines them):
  skill_score      = hits in that skill / clicks in that skill
  overall_accuracy = total hits   / total clicks
  miss_rate        = total misses / total clicks
"""
import re
import pandas as pd

from config import SKILL_QUESTION_MAP, GAME_SKILL_MAP, FEATURE_COLUMNS


def safe_div(a, b):
    return float(a) / b if b and b > 0 else 0.0


def build_features(records, age):
    """
    records: list of dicts -> {"skill": str, "clicks": int, "hits": int, "misses": int}
    age:     child's age in years
    returns: dict with exactly FEATURE_COLUMNS as keys
    """
    per_skill = {s: {"clicks": 0, "hits": 0, "misses": 0} for s in SKILL_QUESTION_MAP}
    for r in records:
        s = r["skill"]
        for k in ("clicks", "hits", "misses"):
            per_skill[s][k] += float(r.get(k, 0) or 0)

    total_clicks = sum(v["clicks"] for v in per_skill.values())
    total_hits = sum(v["hits"] for v in per_skill.values())
    total_misses = sum(v["misses"] for v in per_skill.values())

    feats = {"age": float(age)}
    for s, v in per_skill.items():
        feats[f"{s}_score"] = safe_div(v["hits"], v["clicks"])
    feats["overall_accuracy"] = safe_div(total_hits, total_clicks)
    feats["miss_rate"] = safe_div(total_misses, total_clicks)
    return {c: feats[c] for c in FEATURE_COLUMNS}


# ---------------------------------------------------------------- A. DATASET
def _find_col(df, prefix, q):
    """Find e.g. 'Clicks12' regardless of upper/lower case."""
    pattern = re.compile(rf"^{prefix}{q}$", re.IGNORECASE)
    for c in df.columns:
        if pattern.match(c):
            return c
    return None


def dataset_row_to_records(row, df_columns_lookup):
    records = []
    for skill, questions in SKILL_QUESTION_MAP.items():
        for q in questions:
            cols = df_columns_lookup.get(q)
            if not cols:
                continue
            vals = [row[cols["clicks"]], row[cols["hits"]], row[cols["misses"]]]
            if any(pd.isna(v) for v in vals):
                continue  # missing answer for this question -> skip it
            records.append({
                "skill": skill,
                "clicks": row[cols["clicks"]],
                "hits": row[cols["hits"]],
                "misses": row[cols["misses"]],
            })
    return records


def engineer_dataset(df, age_col="Age"):
    """Apply build_features to every participant in the Kaggle dataset."""
    lookup = {}
    for skill, questions in SKILL_QUESTION_MAP.items():
        for q in questions:
            c, h, m = (_find_col(df, "clicks", q), _find_col(df, "hits", q),
                       _find_col(df, "misses", q))
            if c and h and m:
                lookup[q] = {"clicks": c, "hits": h, "misses": m}
    if not lookup:
        raise ValueError("Could not find Clicks/Hits/Misses columns. Check data/dataset.csv.")

    rows = []
    for _, row in df.iterrows():
        records = dataset_row_to_records(row, lookup)
        rows.append(build_features(records, row[age_col]))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# ------------------------------------------------------------- B. LIVE GAME
def game_results_to_records(results):
    """
    results: list of question results sent by the browser, e.g.
      {"game_id": "letter_hunt", "question_id": 1, "clicks": 5, "hits": 4,
       "misses": 1, "score": 4, "response_time_ms": 8200}
    The skill is looked up on the SERVER from GAME_SKILL_MAP (never trusted from the browser).
    """
    return [{"skill": GAME_SKILL_MAP[r["game_id"]], "clicks": r["clicks"],
             "hits": r["hits"], "misses": r["misses"]} for r in results]


def engineer_game(results, age):
    return build_features(game_results_to_records(results), age)


def game_summary(results):
    """Descriptive statistics for the report (NOT model inputs - the dataset has no timing data)."""
    clicks = sum(r["clicks"] for r in results)
    hits = sum(r["hits"] for r in results)
    misses = sum(r["misses"] for r in results)
    times = [r.get("response_time_ms", 0) for r in results if r.get("response_time_ms")]
    return {
        "questions_answered": len(results),
        "correct_answers": int(hits),
        "incorrect_answers": int(misses),
        "total_clicks": int(clicks),
        "total_score": int(sum(r.get("score", r["hits"]) for r in results)),
        "click_efficiency": round(safe_div(hits, max(clicks, 1)), 3),
        "avg_response_time_sec": round(sum(times) / len(times) / 1000, 2) if times else None,
    }
