# steps/model.py
# WHY: Runs the full modelling workflow in a fixed, auditable order.
# Baseline → CV selection → final test evaluation → conformal intervals.
# The test set is touched exactly once. This is enforced structurally —
# only evaluate_on_test() loads it, and it is called once by the Flow.

import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (root_mean_squared_error,
                             mean_absolute_error, r2_score)
from xgboost import XGBRegressor
from mapie.regression import SplitConformalRegressor

from config.settings import (
    CANDIDATE_MODELS, CV_FOLDS, RANDOM_SEED,
    MODELS_DIR, PREDICTIONS_DIR,
    MIN_IMPROVEMENT_OVER_BASELINE,
    CONFORMAL_COVERAGE_TARGET, CALIBRATION_SIZE,
    TARGET_COLUMN
)
from state.models import (ModelResult, CVResults,
                           FinalEvaluation, PredictionOutput)
from guardrails.provenance import log_event


# ── Model registry ────────────────────────────────────────────────────
# Why a registry dict: adding a new candidate model means adding one
# line here. Nothing else changes anywhere in the codebase.

def _get_model(name: str):
    registry = {
        "ridge":         Ridge(random_state=RANDOM_SEED),
        "lasso":         Lasso(random_state=RANDOM_SEED),
        "random_forest": RandomForestRegressor(
                             n_estimators=100,
                             random_state=RANDOM_SEED,
                             n_jobs=-1),
        "xgboost":       XGBRegressor(
                             n_estimators=200,
                             learning_rate=0.05,
                             random_state=RANDOM_SEED,
                             verbosity=0,
                             n_jobs=-1),
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. "
                         f"Add it to the registry in steps/model.py")
    return registry[name]


def _load_train(preprocess_result) -> tuple:
    """Load train set. Returns X, y as numpy arrays."""
    df = pd.read_parquet(preprocess_result.train_path)
    X  = df.drop(columns=["target"]).values
    y  = df["target"].values
    return X, y


def _load_test(preprocess_result) -> tuple:
    """
    Load test set. Returns X, y as numpy arrays.
    Why a separate function: makes it impossible to accidentally
    load the test set anywhere other than evaluate_on_test().
    """
    df = pd.read_parquet(preprocess_result.test_path)
    X  = df.drop(columns=["target"]).values
    y  = df["target"].values
    return X, y


# ── Step 1: Baseline ─────────────────────────────────────────────────

def train_baseline(preprocess_result, run_id: str) -> ModelResult:
    """
    Why a baseline: the simplest possible model — predicting the mean
    rent for every observation — sets the floor. Any complex model
    that cannot beat this is not worth deploying.
    We use cross-validation even for the baseline so the comparison
    with candidate models is on equal terms.
    """
    X, y = _load_train(preprocess_result)
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    dummy = DummyRegressor(strategy="mean")

    cv = cross_validate(dummy, X, y, cv=kf,
                        scoring=["neg_root_mean_squared_error", "r2"])

    result = ModelResult(
        model_name="baseline_mean",
        cv_rmse_mean=float(-cv["test_neg_root_mean_squared_error"].mean()),
        cv_rmse_std=float(cv["test_neg_root_mean_squared_error"].std()),
        cv_r2_mean=float(cv["test_r2"].mean()),
        hyperparams={"strategy": "mean"},
    )

    log_event(run_id, "BASELINE_DONE", {
        "cv_rmse_mean": result.cv_rmse_mean,
        "cv_r2_mean":   result.cv_r2_mean,
    })

    return result


# ── Step 2: Model selection via cross-validation ──────────────────────

def select_via_cv(preprocess_result, baseline: ModelResult,
                  run_id: str) -> CVResults:
    """
    Why k-fold CV for model selection:
    A single train/val split gives a noisy performance estimate —
    it depends heavily on which rows end up in each split.
    K-fold uses every row for both training and validation across
    different folds, giving a much more reliable estimate.

    Why we record std alongside mean:
    A model with RMSE 0.15 ± 0.01 is preferable to one with
    RMSE 0.14 ± 0.08. The second model is highly variable —
    it may perform well on this dataset but poorly on new data.

    The 1-SE rule: if a simpler model is within one standard error
    of the best model, we prefer the simpler one.
    """
    X, y = _load_train(preprocess_result)
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for name in CANDIDATE_MODELS:
        m  = _get_model(name)
        cv = cross_validate(m, X, y, cv=kf,
                            scoring=["neg_root_mean_squared_error", "r2"])

        result = ModelResult(
            model_name=name,
            cv_rmse_mean=float(-cv["test_neg_root_mean_squared_error"].mean()),
            cv_rmse_std=float(cv["test_neg_root_mean_squared_error"].std()),
            cv_r2_mean=float(cv["test_r2"].mean()),
            hyperparams=_get_model(name).get_params(),
        )
        results.append(result)

        log_event(run_id, "CV_RESULT", {
            "model":    name,
            "rmse":     result.cv_rmse_mean,
            "rmse_std": result.cv_rmse_std,
            "r2":       result.cv_r2_mean,
        })

    # Pick winner: lowest CV RMSE
    winner = min(results, key=lambda r: r.cv_rmse_mean)

    # Guardrail: winner must beat baseline by minimum threshold
    improvement = ((baseline.cv_rmse_mean - winner.cv_rmse_mean)
                   / baseline.cv_rmse_mean)

    if improvement < MIN_IMPROVEMENT_OVER_BASELINE:
        log_event(run_id, "CV_WARNING", {
            "message": (f"Winner {winner.model_name} only improves baseline "
                        f"by {improvement:.1%} — below "
                        f"{MIN_IMPROVEMENT_OVER_BASELINE:.0%} threshold"),
        })

    # Refit winner on full training set and save
    final_model = _get_model(winner.model_name)
    final_model.fit(X, y)
    model_path = MODELS_DIR / f"{run_id}_winner.joblib"
    joblib.dump(final_model, model_path)

    log_event(run_id, "CV_WINNER", {
        "model":      winner.model_name,
        "rmse":       winner.cv_rmse_mean,
        "model_path": str(model_path),
    })

    return CVResults(
        candidates=results,
        winner=winner,
        baseline_rmse=baseline.cv_rmse_mean,
        model_path=model_path,
    )


# ── Step 3: Final evaluation on locked test set ───────────────────────

def evaluate_on_test(cv_results: CVResults,
                     preprocess_result,
                     run_id: str) -> FinalEvaluation:
    """
    WHY THIS FUNCTION EXISTS AND WHY IT IS CALLED EXACTLY ONCE:
    The test set is our one honest estimate of how the model will
    perform on data it has never seen. Every time you look at test
    results and adjust something, you are implicitly fitting to the
    test set and your estimate becomes optimistic.
    This function is called once by the Flow at the very end.
    It is never called inside a loop or a hyperparameter search.
    """
    model = joblib.load(cv_results.model_path)
    X_test, y_test = _load_test(preprocess_result)

    y_pred = model.predict(X_test)

    # Feature importances — available for tree models
    importances = {}
    if hasattr(model, "feature_importances_"):
        importances = {
            str(i): float(v)
            for i, v in enumerate(model.feature_importances_)
        }

    result = FinalEvaluation(
        model_name=cv_results.winner.model_name,
        test_rmse=float(root_mean_squared_error(y_test, y_pred)),
        test_mae=float(mean_absolute_error(y_test, y_pred)),
        test_r2=float(r2_score(y_test, y_pred)),
        feature_importances=importances,
        model_path=cv_results.model_path,
    )

    log_event(run_id, "FINAL_EVAL", {
        "model":     result.model_name,
        "test_rmse": result.test_rmse,
        "test_mae":  result.test_mae,
        "test_r2":   result.test_r2,
    })

    return result


# ── Step 4: Conformal prediction intervals ────────────────────────────

def conformal_intervals(final_eval: FinalEvaluation,
                        preprocess_result,
                        run_id: str) -> PredictionOutput:
    """
    Why conformal prediction over bootstrap or Bayesian intervals:
    - No distributional assumptions needed
    - Coverage guarantee is finite-sample, not asymptotic
    - Works with any sklearn-compatible model
    - MAPIE implements it in two lines

    What coverage means: if we request 90% coverage, at least 90%
    of prediction intervals on new data will contain the true value.
    This is a mathematical guarantee, not a hope.

    We use a calibration split from the training data.
    Why not the test set: the test set must remain untouched.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model  = joblib.load(final_eval.model_path)
    X_train, y_train = _load_train(preprocess_result)

    # Carve out calibration set from training data
    cal_size = int(len(X_train) * CALIBRATION_SIZE)
    X_cal, y_cal     = X_train[:cal_size],  y_train[:cal_size]
    X_fit, y_fit     = X_train[cal_size:],  y_train[cal_size:]

    # Refit on the reduced training set, calibrate on cal set
    model.fit(X_fit, y_fit)

    # Why SplitConformalRegressor: this is the MAPIE 1.x equivalent
    # of the old MapieRegressor with method="plus", cv="prefit".
    # We pass confidence_level directly instead of alpha.
    conformal = SplitConformalRegressor(
        estimator=model,
        confidence_level=CONFORMAL_COVERAGE_TARGET
    )
    conformal.fit(X_cal, y_cal)

    # Generate intervals on the test set
    X_test, y_test = _load_test(preprocess_result)
    y_pred, intervals = conformal.predict(X_test)

    lower = intervals[:, 0]
    upper = intervals[:, 1]

    # Empirical coverage — what fraction of true values fall inside?
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    width_median = float(np.median(upper - lower))

    if coverage < CONFORMAL_COVERAGE_TARGET:
        log_event(run_id, "COVERAGE_WARNING", {
            "achieved": coverage,
            "target":   CONFORMAL_COVERAGE_TARGET,
        })

    # Save predictions with intervals
    pred_df = pd.DataFrame({
        "y_true":   y_test,
        "y_pred":   y_pred,
        "lower_90": lower,
        "upper_90": upper,
    })
    pred_path = PREDICTIONS_DIR / f"{run_id}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    log_event(run_id, "CONFORMAL_DONE", {
        "coverage_achieved":   coverage,
        "interval_width_med":  width_median,
        "predictions_path":    str(pred_path),
    })

    return PredictionOutput(
        predictions_path=pred_path,
        coverage_achieved=coverage,
        interval_width_median=width_median,
    )