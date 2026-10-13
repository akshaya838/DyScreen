"""
train_model.py - Modules 4, 5, 7, 12, 13: balancing, training, tuning, evaluation,
explainability and saving.   Run:  python train_model.py

Every number written to results/ comes from actually running this script on data/dataset.csv.
"""
import os
import json
import time
import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, make_scorer, fbeta_score)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from config import (RESULTS_DIR, MODELS_DIR, MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH,
                    SELECTED_FEATURES_PATH, META_PATH, RANDOM_STATE, CV_FOLDS, FEATURE_COLUMNS,
                    RISK_THRESHOLDS, MIN_CV_PRECISION)
from preprocess import build_engineered_dataset, split_data, make_imputer_and_scaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def get_models(scale_pos_weight):
    """
    Each model is trained twice:
      'class_weight' -> the algorithm itself gives more weight to the dyslexia class
      'smote'        -> SMOTE creates synthetic dyslexia examples (TRAINING folds only)
    Grids are kept small so the script runs on a normal laptop.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10], "model__solver": ["lbfgs", "liblinear"]},
            {"model__class_weight": ["balanced"]}),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"],
             "model__metric": ["euclidean", "manhattan"]},
            None),  # KNN has no class_weight option -> only the SMOTE version is trained
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"],
             "model__gamma": ["scale"]},
            {"model__class_weight": ["balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 8], "model__min_samples_split": [2, 10, 30]},
            {"model__class_weight": ["balanced"]}),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [5, 10, None],
             "model__min_samples_split": [2, 10], "model__max_features": ["sqrt"]},
            {"model__class_weight": ["balanced", "balanced_subsample"]}),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8],
             "model__colsample_bytree": [0.8]},
            {"model__scale_pos_weight": [scale_pos_weight]}),
    }


def build_pipeline(model, use_smote):
    imputer, scaler = make_imputer_and_scaler()
    steps = [("imputer", imputer), ("scaler", scaler)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def main():
    start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---------------- Module 1 + 2: preprocessing and feature engineering
    X, y, _ = build_engineered_dataset(verbose=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nStratified split -> train: {len(X_train)}, test: {len(X_test)}")

    # ---------------- Module 4: class imbalance (real counts)
    before = y_train.value_counts().sort_index()
    imputer, scaler = make_imputer_and_scaler()
    Xtr_s = scaler.fit_transform(imputer.fit_transform(X_train))
    _, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr_s, y_train)
    after = pd.Series(y_res).value_counts().sort_index()
    balance_text = (
        "MODULE 4 - CLASS IMBALANCE (training set only; SMOTE is never applied before the split)\n"
        f"Before balancing:  Class 0 = {before[0]}   Class 1 = {before[1]}\n"
        f"After SMOTE:       Class 0 = {after[0]}   Class 1 = {after[1]}\n"
        f"Class-weight approach: data unchanged, minority errors weighted about "
        f"{before[0] / before[1]:.2f}x more.\n")
    print("\n" + balance_text)
    with open(os.path.join(RESULTS_DIR, "class_balance.txt"), "w") as f:
        f.write(balance_text)
    scale_pos_weight = round(before[0] / before[1], 2)

    # ---------------- Module 5: training + tuning with stratified k-fold CV
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Tuning score: F2 = F-score that weights recall 2x more than precision.
    f2 = make_scorer(fbeta_score, beta=2, zero_division=0)
    rows, fitted = [], {}

    for name, (model, grid, cw_grid) in get_models(scale_pos_weight).items():
        variants = [("SMOTE", True, grid)]
        if cw_grid is not None:
            variants.insert(0, ("Class weight", False, {**grid, **cw_grid}))
        for strategy, use_smote, param_grid in variants:
            label = f"{name} ({strategy})"
            print(f"Training {label} ...", end=" ", flush=True)
            search = GridSearchCV(build_pipeline(model, use_smote), param_grid, scoring=f2,
                                  cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            cvres = cross_validate(best, X_train, y_train, cv=cv, n_jobs=-1,
                                   scoring=["recall", "precision", "f1", "roc_auc"])
            y_prob = best.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            m = evaluate(y_test, y_pred, y_prob)
            rows.append({"Model": label, **{k: round(v, 4) for k, v in m.items()},
                         "CV Recall": round(cvres["test_recall"].mean(), 4),
                         "CV Precision": round(cvres["test_precision"].mean(), 4),
                         "CV F1": round(cvres["test_f1"].mean(), 4),
                         "CV ROC-AUC": round(cvres["test_roc_auc"].mean(), 4),
                         "Best params": {k.replace("model__", ""): v
                                         for k, v in search.best_params_.items()}})
            fitted[label] = (best, y_prob, y_pred)
            print(f"test recall={m['Recall']:.3f}  CV recall={cvres['test_recall'].mean():.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    # ---------------- Model selection: highest CV recall with CV precision >= minimum
    eligible = results[results["CV Precision"] >= MIN_CV_PRECISION]
    if eligible.empty:
        eligible = results
    best_row = eligible.sort_values(["CV Recall", "CV F1"], ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model, best_prob, best_pred = fitted[best_name]
    print(f"\nSELECTED MODEL: {best_name}  (chosen on cross-validated recall, "
          f"not on test accuracy)")

    print("\nClassification report (test set, selected model):")
    report = classification_report(y_test, best_pred, target_names=["No dyslexia", "Dyslexia"])
    print(report)

    # ---------------- Graphs (Module 13)
    show_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(results[show_cols].to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot.bar(ax=ax)
    ax.set_title("Model comparison on the test set (recall is the primary metric)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=130)
    plt.close(fig)

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No dyslexia", "Dyslexia"], yticklabels=["No dyslexia", "Dyslexia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (_, prob, _) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, lw=2.5 if label == best_name else 1,
                label=f"{label} (AUC={roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=130)
    plt.close(fig)

    # ---------------- Module 7: explainability
    perm = permutation_importance(best_model, X_test, y_test, scoring="recall",
                                  n_repeats=10, random_state=RANDOM_STATE)
    imp = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax, color="#E8744C")
    ax.set_title(f"Permutation feature importance (drop in recall) - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=130)
    plt.close(fig)
    imp.sort_values(ascending=False).round(4).to_csv(
        os.path.join(RESULTS_DIR, "feature_importance.csv"), header=["importance"])

    try:
        import shap
        est = best_model.named_steps["model"]
        Xt_test = best_model.named_steps["scaler"].transform(
            best_model.named_steps["imputer"].transform(X_test))
        Xt_df = pd.DataFrame(Xt_test, columns=FEATURE_COLUMNS)
        if hasattr(est, "estimators_") or est.__class__.__name__ in ("XGBClassifier", "DecisionTreeClassifier"):
            sv = shap.TreeExplainer(est).shap_values(Xt_df)
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
        elif est.__class__.__name__ == "LogisticRegression":
            sv = shap.LinearExplainer(est, Xt_df).shap_values(Xt_df)
        else:
            background = shap.sample(Xt_df, 50, random_state=RANDOM_STATE)
            sv = shap.KernelExplainer(lambda d: est.predict_proba(d)[:, 1], background)\
                .shap_values(Xt_df.iloc[:100], nsamples=100, silent=True)
            Xt_df = Xt_df.iloc[:100]
        plt.figure()
        shap.summary_plot(sv, Xt_df, show=False)
        plt.title(f"SHAP summary - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=130)
        plt.close("all")
        print("SHAP summary saved.")
    except Exception as e:  # SHAP is optional - permutation importance always works
        print("SHAP skipped:", e)

    # ---------------- Module 12: save model + preprocessing objects
    joblib.dump(best_model, MODEL_PATH)                           # full pipeline (imputer+scaler+model)
    joblib.dump(best_model.named_steps["scaler"], SCALER_PATH)    # scaler on its own (for viva/demo)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)
    joblib.dump(FEATURE_COLUMNS, SELECTED_FEATURES_PATH)
    meta = {
        "model_name": best_name,
        "trained_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_metrics": {k: float(best_row[k]) for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]},
        "cv_recall": float(best_row["CV Recall"]),
        "risk_thresholds": RISK_THRESHOLDS,
        "top_features": imp.sort_values(ascending=False).index.tolist(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    write_analysis(results, best_row, cm, imp, balance_text, report)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Done in {time.time() - start:.0f} s. Now run:  python app.py")


def write_analysis(results, best, cm, imp, balance_text, report):
    """Module 13 + 14: result analysis written ONLY from measured values."""
    tn, fp, fn, tp = cm.ravel()
    top_acc = results.sort_values("Accuracy", ascending=False).iloc[0]
    table = results[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    md = ["# DyScreen - Result Analysis (auto-generated from real results)\n",
          "## Model comparison (test set)\n",
          "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
          "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r['Model']} | {r['Accuracy']:.3f} | {r['Precision']:.3f} | "
                  f"{r['Recall']:.3f} | {r['F1']:.3f} | {r['ROC-AUC']:.3f} |")
    md += ["\n## Class balance\n", "```", balance_text.strip(), "```",
           "\n## Selected model\n",
           f"The model was selected on **cross-validated recall** (among models with CV precision "
           f"of at least 0.20), not on accuracy. Among the evaluated models, **{best['Model']}** "
           f"achieved a cross-validated dyslexia-class recall of **{best['CV Recall']:.3f}**. "
           f"On the held-out test set its recall was **{best['Recall']:.3f}**, precision "
           f"**{best['Precision']:.3f}**, F1-score **{best['F1']:.3f}** and ROC-AUC "
           f"**{best['ROC-AUC']:.3f}**.",
           f"\nOn the test set it correctly flagged **{tp}** of **{tp + fn}** dyslexic participants "
           f"and missed **{fn}**. It also produced **{fp}** false positives out of **{fp + tn}** "
           f"non-dyslexic participants, which remain a consideration: a false positive means a "
           f"child is referred for assessment unnecessarily.",
           f"\nFor comparison, the highest *accuracy* was **{top_acc['Accuracy']:.3f}** "
           f"({top_acc['Model']}) - this shows why accuracy alone is misleading on imbalanced data.",
           "\n## Most influential features (permutation importance, drop in recall)\n"]
    for feat, val in imp.sort_values(ascending=False).items():
        md.append(f"- {feat}: {val:.4f}")
    md += ["\n## Classification report (selected model)\n", "```", report, "```",
           "\n## Practical insights\n",
           "1. The system can provide an inexpensive first-level screening aid.",
           "2. Browser-based games reduce the need for specialised hardware such as eye trackers.",
           "3. Skill-level features make the output more interpretable than a simple yes/no.",
           "4. Recall is prioritised because a screening system should reduce missed at-risk cases.",
           "5. Class imbalance strongly affects evaluation - compare the accuracy and recall columns above.",
           "6. The system can help identify children who may benefit from professional assessment.",
           "7. It does not replace teachers, special educators, psychologists or clinicians.",
           "8. Only anonymous game-performance data is collected (no name, contact, photo or video).",
           "\n## Limitations\n",
           "- Trained on Spanish-speaking participants; transfer to Indian children is not yet validated.",
           "- Risk thresholds are prototype values, not clinically validated.",
           "- The browser game produces the same feature types as the dataset, but on different tasks.",
           ]
    with open(os.path.join(RESULTS_DIR, "results_analysis.md"), "w") as f:
        f.write("\n".join(md))
    print("Result analysis written to results/results_analysis.md")


if __name__ == "__main__":
    main()
