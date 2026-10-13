"""
feature_selection.py - Module 6: which features matter most?  Run: python feature_selection.py
Uses the TRAINING split only, so the test set stays unseen.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from config import RESULTS_DIR, RANDOM_STATE, TARGET_COLUMN
from preprocess import build_engineered_dataset, split_data, encode_raw_features


def run_feature_selection():
    X, y, df = build_engineered_dataset(verbose=False)
    X_train, X_test, y_train, y_test = split_data(X, y)
    Xt = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_train), columns=X.columns)
    yt = y_train.reset_index(drop=True)

    print("=" * 60, "\nMODULE 6 - FEATURE SELECTION (training data only)\n" + "=" * 60)

    # 1. Correlation with the target
    corr = Xt.assign(target=yt).corr()["target"].drop("target")
    # 2. Chi-square (needs non-negative values -> MinMax scale first)
    chi_scores, chi_p = chi2(MinMaxScaler().fit_transform(Xt), yt)
    # 3. Mutual information
    mi = mutual_info_classif(Xt, yt, random_state=RANDOM_STATE)
    # 4. Recursive Feature Elimination with Logistic Regression
    rfe = RFE(LogisticRegression(max_iter=2000, class_weight="balanced"), n_features_to_select=4)
    rfe.fit(StandardScaler().fit_transform(Xt), yt)

    table = pd.DataFrame({
        "correlation_with_target": corr.values,
        "chi2_score": chi_scores,
        "chi2_p_value": chi_p,
        "mutual_information": mi,
        "rfe_rank (1=kept)": rfe.ranking_,
    }, index=X.columns).sort_values("mutual_information", ascending=False)
    print(table.round(4))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.round(5).to_csv(os.path.join(RESULTS_DIR, "feature_selection_scores.csv"))
    print("\nRFE selected features:", list(X.columns[rfe.support_]))

    fig, ax = plt.subplots(figsize=(7, 4))
    table["mutual_information"].sort_values().plot.barh(ax=ax, color="#4C9BE8")
    ax.set_title("Mutual information with dyslexia label")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_selection_mi.png"), dpi=130)
    plt.close(fig)

    # 5. Mutual information on the RAW 196 question-level features (top 15)
    X_raw, _ = encode_raw_features(df)
    X_raw = X_raw.loc[X_train.index] if len(X_raw) == len(X) else X_raw
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    mi_raw = pd.Series(mutual_info_classif(X_raw, yt, random_state=RANDOM_STATE),
                       index=X_raw.columns).sort_values(ascending=False).head(15)
    print("\nTop 15 RAW question-level features by mutual information:\n", mi_raw.round(4))
    mi_raw.round(5).to_csv(os.path.join(RESULTS_DIR, "raw_top15_mutual_information.csv"))

    # 6. PCA - ONLY for 2D visualisation (final model keeps interpretable features)
    Xs = StandardScaler().fit_transform(Xt)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    comps = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, name in [(0, "#4C9BE8", "No dyslexia"), (1, "#E8744C", "Dyslexia")]:
        m = (yt == label).values
        ax.scatter(comps[m, 0], comps[m, 1], s=8, alpha=0.5, c=color, label=name)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.set_title("PCA visualisation of engineered features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pca_visualisation.png"), dpi=130)
    plt.close(fig)
    print(f"\nPCA: first 2 components explain {ev.sum():.1f}% of variance (visualisation only).")


if __name__ == "__main__":
    run_feature_selection()
