# steps/eda.py
# WHY: Understand the data before modelling anything.
# Every downstream decision — imputation strategy, PCA, feature selection —
# is informed by what we find here.
# Output is a typed EDASummary that the Flow uses for routing decisions.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Why: non-interactive backend, safe for pipelines
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config.settings import EDA_DIR, TARGET_COLUMN, MAX_NULL_RATE
from state.models import EDASummary
from guardrails.provenance import log_event


def _plot_target_distribution(df: pd.DataFrame, run_id: str) -> None:
    """
    Why: The target distribution tells us immediately whether log(rent)
    is approximately normal. If it is heavily skewed even after the log
    transform, we may need a different transformation strategy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["rent"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Raw Rent Distribution")
    axes[0].set_xlabel("Rent ($)")

    axes[1].hist(df[TARGET_COLUMN], bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Log(Rent) Distribution")
    axes[1].set_xlabel("log(Rent)")

    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{run_id}_target_dist.png", dpi=100)
    plt.close()


def _plot_correlations(df: pd.DataFrame, numeric_cols: list,
                       run_id: str) -> list[dict]:
    """
    Why: Correlation with the target tells us which features are
    most likely to be useful to the model. High inter-feature
    correlation warns us about multicollinearity.
    Returns the top correlations as structured data for EDASummary.
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return []

    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    plt.colorbar(im)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{run_id}_correlations.png", dpi=100)
    plt.close()

    # Extract top correlations with the target
    if TARGET_COLUMN not in corr_matrix.columns:
        return []

    target_corr = (corr_matrix[TARGET_COLUMN]
                   .drop(TARGET_COLUMN, errors="ignore")
                   .abs()
                   .sort_values(ascending=False))

    return [{"feature": f, "correlation": round(v, 4)}
            for f, v in target_corr.head(10).items()]


def _check_multicollinearity(df: pd.DataFrame,
                              numeric_cols: list) -> bool:
    """
    Why VIF: Variance Inflation Factor measures how much a feature's
    variance is inflated by its correlation with other features.
    VIF > 10 means that feature is almost entirely explained by
    others — it adds noise, not signal, to a linear model.
    We flag this so the preprocessing step knows to handle it.
    """
    cols = [c for c in numeric_cols
            if c in df.columns and c != TARGET_COLUMN]
    clean = df[cols].dropna()

    if len(clean.columns) < 2 or len(clean) < 10:
        return False

    try:
        vifs = [variance_inflation_factor(clean.values, i)
                for i in range(clean.shape[1])]
        return any(v > 10 for v in vifs if np.isfinite(v))
    except Exception:
        return False


def run(gold_path: Path, run_id: str) -> EDASummary:
    """
    Main entry point called by the Flow.
    Loads the gold table, computes all EDA metrics, saves plots,
    and returns a typed EDASummary.
    """
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gold_path)

    log_event(run_id, "EDA_START", {
        "rows": len(df), "cols": list(df.columns)
    })

    # Classify features by type
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # High null features — will need imputation or dropping
    high_null = [
        c for c in df.columns
        if df[c].isna().mean() > MAX_NULL_RATE
    ]

    # High cardinality categoricals — may need encoding or grouping
    high_card = [
        c for c in categorical_cols
        if df[c].nunique() > 50
    ]

    # Target skewness — ideally close to 0 after log transform
    target_skew = (float(df[TARGET_COLUMN].skew())
                   if TARGET_COLUMN in df.columns else 0.0)

    # Generate plots
    if "rent" in df.columns and TARGET_COLUMN in df.columns:
        _plot_target_distribution(df, run_id)

    top_correlations = _plot_correlations(df, numeric_cols, run_id)

    # Multicollinearity check
    multicollinearity = _check_multicollinearity(df, numeric_cols)

    summary = EDASummary(
        feature_count=len(df.columns),
        row_count=len(df),
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        high_null_features=high_null,
        high_cardinality_features=high_card,
        multicollinearity_flag=multicollinearity,
        target_skew=round(target_skew, 4),
    )

    log_event(run_id, "EDA_COMPLETE", {
        "feature_count":       summary.feature_count,
        "high_null_features":  high_null,
        "multicollinearity":   multicollinearity,
        "target_skew":         target_skew,
        "top_correlations":    top_correlations,
    })

    return summary