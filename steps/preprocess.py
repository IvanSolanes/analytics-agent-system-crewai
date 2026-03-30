# steps/preprocess.py
# WHY: Prepares data for modelling in a leak-proof way.
# The golden rule: fit only on train, transform both train and test.
# Every transformation is captured in a sklearn Pipeline so it can
# be applied identically to new data at inference time.

import numpy as np
import pandas as pd
import joblib

from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from config.settings import (
    TEST_SIZE, RANDOM_SEED, CALIBRATION_SIZE,
    TARGET_COLUMN, MODELS_DIR,
    PCA_FEATURE_THRESHOLD, PCA_VARIANCE_TARGET
)
from state.models import EDASummary, PreprocessResult, SelectedFeatures
from guardrails.provenance import log_event


def _detect_outliers(df: pd.DataFrame,
                     numeric_cols: list[str]) -> list[int]:
    """
    Why Mahalanobis distance for outlier detection:
    Unlike z-score which checks one feature at a time, Mahalanobis
    distance measures how far a point is from the centre of the
    distribution accounting for correlations between features.
    A point can look normal on every individual feature but be
    an extreme outlier in the multivariate space.
    We flag outliers but do not auto-drop them — the analyst
    can inspect and decide. They are recorded in PreprocessResult.
    """
    cols = [c for c in numeric_cols
            if c in df.columns and c != TARGET_COLUMN]
    clean = df[cols].dropna()

    if len(clean) < 10 or len(cols) < 2:
        return []

    try:
        cov = np.cov(clean.values, rowvar=False)
        inv_cov = np.linalg.pinv(cov)   # pinv handles singular matrices
        mean = clean.mean().values
        distances = clean.apply(
            lambda row: mahalanobis(row.values, mean, inv_cov), axis=1
        )
        # Threshold: chi-squared 97.5th percentile with df=n_features
        from scipy.stats import chi2
        threshold = chi2.ppf(0.975, df=len(cols))
        outlier_mask = distances ** 2 > threshold
        return clean[outlier_mask].index.tolist()
    except Exception:
        return []


def _build_pipeline(numeric_cols: list[str],
                    categorical_cols: list[str],
                    imputation_log: dict) -> ColumnTransformer:
    """
    Why ColumnTransformer: numeric and categorical features need
    different treatment. This applies the right steps to the right
    columns automatically — no manual column slicing needed.

    Why RobustScaler over StandardScaler:
    RobustScaler uses median and IQR instead of mean and std.
    It is far less affected by the outliers we just detected.
    Rent data often has extreme values in expensive cities.

    Why SimpleImputer with median:
    Median imputation is robust — it is not pulled by extreme values.
    For categorical features we use the most frequent value.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Record what imputation strategy each feature gets
    for c in numeric_cols:
        imputation_log[c] = "median"
    for c in categorical_cols:
        imputation_log[c] = "most_frequent"

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers,
                             remainder="drop")


def build_pipeline(gold_path, eda: EDASummary,
                   run_id: str) -> PreprocessResult:
    """
    Main entry point called by the Flow.
    Splits data, detects outliers, builds and fits the pipeline,
    optionally applies PCA, saves everything, returns PreprocessResult.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gold_path)

    # ── Feature columns ──────────────────────────────────────────────
    # Drop the raw rent — we only use log_rent as the target
    feature_cols = [c for c in df.columns
                    if c not in [TARGET_COLUMN, "rent"]]

    numeric_cols     = [c for c in eda.numeric_features
                        if c in feature_cols]
    categorical_cols = [c for c in eda.categorical_features
                        if c in feature_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # ── Train / test split ───────────────────────────────────────────
    # Why split first, before anything else:
    # The test set must never influence any fitted parameter —
    # not the scaler mean, not the imputer median, nothing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    log_event(run_id, "SPLIT_DONE", {
        "train_rows": len(X_train),
        "test_rows":  len(X_test),
    })

    # ── Outlier detection on training set only ───────────────────────
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    outlier_indices = _detect_outliers(train_df, numeric_cols)

    log_event(run_id, "OUTLIERS_DETECTED", {"count": len(outlier_indices)})

    # ── Build and fit the pipeline ───────────────────────────────────
    imputation_log = {}
    preprocessor = _build_pipeline(numeric_cols, categorical_cols,
                                   imputation_log)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # ── Optional PCA ─────────────────────────────────────────────────
    reduction_applied = False
    reduction_method  = None
    n_components      = None

    if eda.feature_count > PCA_FEATURE_THRESHOLD:
        pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full")
        X_train_processed = pca.fit_transform(X_train_processed)
        X_test_processed  = pca.transform(X_test_processed)
        reduction_applied = True
        reduction_method  = "PCA"
        n_components      = X_train_processed.shape[1]

        log_event(run_id, "PCA_APPLIED", {"n_components": n_components})

    # ── Save pipeline to disk ────────────────────────────────────────
    pipeline_path = MODELS_DIR / f"{run_id}_pipeline.joblib"
    joblib.dump(preprocessor, pipeline_path)

    # ── Save train / test sets ───────────────────────────────────────
    from config.settings import GOLD_DIR
    train_path = GOLD_DIR / f"{run_id}_train.parquet"
    test_path  = GOLD_DIR / f"{run_id}_test.parquet"

    pd.DataFrame(X_train_processed).assign(
        target=y_train.values).to_parquet(train_path, index=False)
    pd.DataFrame(X_test_processed).assign(
        target=y_test.values).to_parquet(test_path, index=False)

    log_event(run_id, "PREPROCESS_COMPLETE", {
        "pipeline_path":      str(pipeline_path),
        "reduction_applied":  reduction_applied,
        "outlier_count":      len(outlier_indices),
    })

    return PreprocessResult(
        train_path=train_path,
        test_path=test_path,
        pipeline_path=pipeline_path,
        outlier_count=len(outlier_indices),
        imputation_log=imputation_log,
        reduction_applied=reduction_applied,
        reduction_method=reduction_method,
        n_components=n_components,
    )


def select_features(preprocess_result: PreprocessResult,
                    run_id: str) -> SelectedFeatures:
    """
    Why feature selection after preprocessing:
    After encoding, we may have many near-zero-variance columns
    from one-hot encoding rare categories. VarianceThreshold
    removes features that carry almost no information.
    We log exactly what was dropped and why.
    """
    df = pd.read_parquet(preprocess_result.train_path)
    feature_cols = [c for c in df.columns if c != "target"]

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df[feature_cols])

    kept    = [feature_cols[i] for i, s in
               enumerate(selector.get_support()) if s]
    dropped = [feature_cols[i] for i, s in
               enumerate(selector.get_support()) if not s]

    drop_reasons = {c: "near-zero variance" for c in dropped}

    log_event(run_id, "FEATURE_SELECTION", {
        "kept": len(kept), "dropped": len(dropped)
    })

    return SelectedFeatures(kept=kept, dropped=dropped,
                            drop_reasons=drop_reasons)