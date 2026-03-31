# config/settings.py
# WHY THIS FILE EXISTS:
# Every tunable value lives here. Nothing is hard-coded anywhere else.
# Change behaviour by editing this file, not by hunting through code.

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
# Why Path() instead of strings: Path works on Windows, Mac, and Linux
# without worrying about forward vs backward slashes.

ROOT        = Path(__file__).parent.parent   # the project root folder
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

BRONZE_DIR      = DATA_DIR    / "bronze"
SILVER_DIR      = DATA_DIR    / "silver"
GOLD_DIR        = DATA_DIR    / "gold"

EDA_DIR         = OUTPUTS_DIR / "eda"
MODELS_DIR      = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR     = OUTPUTS_DIR / "reports"
PROVENANCE_DIR  = OUTPUTS_DIR / "provenance"

# ── Reproducibility ──────────────────────────────────────────────────
# Why: Every random operation in the pipeline uses this seed.
# Same seed = same results every run. Essential for auditability.
RANDOM_SEED = 42

# ── Data split ───────────────────────────────────────────────────────
# Why 80/20: Standard split. The 20% test set is locked away and
# touched exactly once at the very end to measure true generalisation.
TEST_SIZE = 0.20

# ── Cross-validation ─────────────────────────────────────────────────
# Why 5 folds: Balances bias vs variance in performance estimates.
# More folds = more reliable but slower. 5 is the standard choice.
CV_FOLDS = 5

# ── Candidate models ─────────────────────────────────────────────────
# Why these four: They cover the spectrum from simple+interpretable
# (ridge, lasso) to powerful+complex (random_forest, xgboost).
# We let cross-validation pick the winner — not our assumptions.
CANDIDATE_MODELS = ["ridge", "lasso", "random_forest", "xgboost"]

# ── Guardrail thresholds ─────────────────────────────────────────────
# Why explicit thresholds: Every guardrail check compares against
# one of these values. Changing a threshold here changes it everywhere.

MIN_ROWS_BRONZE  = 100    # fewer rows than this = likely a failed download
MIN_ROWS_SILVER  = 500    # fewer rows after cleaning = pipeline halts
MAX_NULL_RATE    = 0.20   # features with >20% nulls are flagged
PCA_FEATURE_THRESHOLD = 30  # run PCA only if we have more than 30 features
PCA_VARIANCE_TARGET   = 0.90  # keep enough PCA components for 90% variance

# Why 5%: A complex model must beat the baseline by at least this much
# to justify its added complexity and reduced interpretability.
MIN_IMPROVEMENT_OVER_BASELINE = 0.05

# ── Uncertainty quantification ───────────────────────────────────────
# Why 90%: Our prediction intervals must contain the true value
# at least 90% of the time. This is the coverage guarantee.
CONFORMAL_COVERAGE_TARGET = 0.90

# ── LLM ──────────────────────────────────────────────────────────────
# Why gpt-4o-mini: Our agents produce structured JSON, not essays.
# A smaller, cheaper model is fully sufficient for this task.
LLM_MODEL = "gpt-4o-mini"

# ── Target ────────────────────────────────────────────────────────────
# Why log transform: rent is right-skewed. Modelling log(rent) makes
# the target more normally distributed and improves model fit.
# RMSE on log scale = approximate percentage error. More interpretable.
TARGET_COLUMN = "log_rent"

# ── Calibration split ─────────────────────────────────────────────────
# Why a separate calibration set: conformal prediction needs a held-out
# set to calibrate the interval widths. This must be separate from both
# the training set and the test set.
CALIBRATION_SIZE = 0.10   # 10% of training data reserved for calibration

# ── Expected columns per source ───────────────────────────────────────
# Why here: schema_checks.py needs to know what columns to expect from
# each source. Defining them here means one place to update if a source
# changes its schema.
EXPECTED_COLUMNS = {
    "zillow":           ["RegionID", "RegionName", "RegionType", "StateName"],
    "zillow_inventory": ["RegionID", "RegionName", "RegionType", "StateName"],
}

# ── V1 Verified data sources ──────────────────────────────────────────
# WHY HARD-CODED HERE: The source discovery agent returned plausible
# but broken URLs on first run. For v1 we seed it with verified working
# URLs so extraction actually succeeds. The agent still runs — it now
# has these as reference anchors in its prompt.
VERIFIED_SOURCES = {
    "zillow_rents": {
        "url": (
            "https://files.zillowstatic.com/research/public_csvs/zori/"
            "Metro_zori_uc_sfrcondomfr_sm_month.csv"
        ),
        "description": "Zillow Observed Rent Index by metro, monthly"
    },
    "hud_fmr": {
        "url": (
            "https://www.huduser.gov/portal/datasets/fmr/"
            "fmr2024r/FY2024_4050_FMRs.csv"
        ),
        "description": "HUD Fair Market Rents 2024 by metro area"
    },
}