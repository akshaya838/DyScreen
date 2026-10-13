"""
eda.py - Module 3: Exploratory Data Analysis.  Run:  python eda.py
All graphs are saved inside results/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import RESULTS_DIR, TARGET_COLUMN, SKILL_LABELS
from preprocess import load_dataset, clean_dataset, find_column
from feature_engineering import engineer_dataset

sns.set_theme(style="whitegrid")
CLASS_NAMES = {0: "No dyslexia", 1: "Dyslexia"}


def save(fig, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  saved", path)


def run_eda():
    raw = load_dataset()
    print("=" * 60, "\nMODULE 3 - EXPLORATORY DATA ANALYSIS\n" + "=" * 60)
    print("1. Shape:", raw.shape)
    print("2. Data types:\n", raw.dtypes.value_counts().to_string())
    print("3. Missing values (total):", int(raw.isnull().sum().sum()))
    print("4. Duplicate rows:", int(raw.duplicated().sum()))

    df = clean_dataset(raw, verbose=False)
    age_col = find_column(df, "Age")
    gender_col = find_column(df, "Gender")
    feats = engineer_dataset(df, age_col=age_col)
    feats[TARGET_COLUMN] = df[TARGET_COLUMN].values
    feats["label"] = feats[TARGET_COLUMN].map(CLASS_NAMES)

    # 5. Class distribution
    counts = feats[TARGET_COLUMN].value_counts().sort_index()
    print("5. Class distribution:", {CLASS_NAMES[k]: int(v) for k, v in counts.items()})
    ratio = counts[0] / counts[1]
    print(f"   Imbalance ratio = {ratio:.1f} : 1  (majority : minority)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="label", data=feats, ax=ax, order=["No dyslexia", "Dyslexia"],
                  palette=["#4C9BE8", "#E8744C"])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Class distribution (imbalanced)")
    ax.set_xlabel("")
    save(fig, "class_distribution.png")

    # 6. Age distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=feats, x="age", hue="label", multiple="stack", bins=11, ax=ax)
    ax.set_title("Age distribution by class")
    save(fig, "age_distribution.png")

    # 7. Gender distribution
    if gender_col:
        fig, ax = plt.subplots(figsize=(5, 4))
        g = pd.DataFrame({"gender": df[gender_col].values, "label": feats["label"]})
        sns.countplot(x="gender", hue="label", data=g, ax=ax)
        ax.set_title("Gender distribution by class")
        save(fig, "gender_distribution.png")

    # 8 + 9. Accuracy and miss-rate distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.kdeplot(data=feats, x="overall_accuracy", hue="label", common_norm=False, fill=True, ax=axes[0])
    axes[0].set_title("Overall accuracy by class")
    sns.kdeplot(data=feats, x="miss_rate", hue="label", common_norm=False, fill=True, ax=axes[1])
    axes[1].set_title("Miss rate by class")
    save(fig, "accuracy_missrate_distribution.png")

    # 10. Skill-score distributions (box plots by class)
    skill_cols = [f"{s}_score" for s in SKILL_LABELS]
    long = feats.melt(id_vars="label", value_vars=skill_cols, var_name="skill", value_name="score")
    long["skill"] = long["skill"].str.replace("_score", "").map(SKILL_LABELS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=long, x="skill", y="score", hue="label", ax=ax)
    ax.set_title("Skill-score distributions by class")
    ax.set_xlabel("")
    save(fig, "feature_distribution.png")

    # 11. Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = feats.drop(columns=["label"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (engineered features + target)")
    save(fig, "correlation_heatmap.png")

    # 12. Mean feature values by class (table)
    means = feats.drop(columns=["label"]).groupby(TARGET_COLUMN).mean().T
    means.columns = [CLASS_NAMES[c] for c in means.columns]
    print("\n12. Mean engineered feature value by class:\n", means.round(3))
    means.round(4).to_csv(os.path.join(RESULTS_DIR, "feature_means_by_class.csv"))
    print("\nNote: EDA shows statistical patterns in this dataset only; it makes no medical claims.")


if __name__ == "__main__":
    run_eda()
