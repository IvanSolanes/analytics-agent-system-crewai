# state/models.py
# WHY: Every piece of data that moves between pipeline steps
# is defined here as a typed Pydantic model.
# Nothing passes between steps as a raw dict or plain string.

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from datetime import datetime
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────
# Why enums: Constrain values to a fixed set.
# "api" is valid. "API" or "rest" is not. No ambiguity.

class AccessMethod(str, Enum):
    api      = "api"
    csv      = "csv_download"
    scrape   = "web_scrape"

class DataType(str, Enum):
    listing       = "listing"
    socioeconomic = "socioeconomic"
    geographic    = "geographic"

class Severity(str, Enum):
    warning  = "WARNING"
    critical = "CRITICAL"

class SupportLevel(str, Enum):
    supported   = "SUPPORTED"
    weak        = "WEAK"
    unsupported = "UNSUPPORTED"


# ── Parsed brief ──────────────────────────────────────────────────────
# Produced by: ingest_brief step (pure Python, no agent needed)
# Consumed by: source discovery agent, insight narrator agent
# Why: Forces the brief into structured fields so agents receive
# precise inputs rather than a blob of free text.

class ParsedBrief(BaseModel):
    goal:        str        # what the team wants to achieve
    target:      str        # what we are predicting
    geographies: list[str]  # cities, regions, or "national"
    data_needs:  list[str]  # types of data required
    raw:         str        # original brief, kept for provenance


# ── Source discovery ──────────────────────────────────────────────────
# Produced by: SourceDiscoveryAgent
# Consumed by: extract step

class DataSource(BaseModel):
    name:          str
    url:           str
    data_type:     DataType
    access_method: AccessMethod
    justification: str          # one sentence — why is this source relevant?

class DataSourceList(BaseModel):
    sources: list[DataSource]


# ── Bronze layer ──────────────────────────────────────────────────────
# Produced by: extract step
# Consumed by: validate step
# Why checksum: proves the raw file was never modified after download.

class BronzeFile(BaseModel):
    source_name:     str
    local_path:      Path
    row_count:       int
    downloaded_at:   datetime
    checksum:        str        # SHA-256 of the raw file
    columns:         list[str]  # column names at the moment of download

class BronzeManifest(BaseModel):
    run_id: str
    files:  list[BronzeFile]


# ── Validation ────────────────────────────────────────────────────────
# Produced by: validate step
# Consumed by: flow router — CRITICAL failures halt the pipeline

class ValidationFailure(BaseModel):
    source:   str
    check:    str       # e.g. "missing_columns", "row_count", "null_rate"
    severity: Severity
    detail:   str

class ValidationReport(BaseModel):
    run_id:   str
    failures: list[ValidationFailure] = []

    @property
    def has_critical_failures(self) -> bool:
        # Why property: the flow calls this to decide whether to halt.
        # Logic stays here, not scattered across flow steps.
        return any(f.severity == Severity.critical for f in self.failures)


# ── EDA ───────────────────────────────────────────────────────────────
# Produced by: eda step
# Consumed by: flow router (PCA decision), preprocess step, insight agent

class EDASummary(BaseModel):
    feature_count:              int
    row_count:                  int
    numeric_features:           list[str]
    categorical_features:       list[str]
    high_null_features:         list[str]   # > 20% null
    high_cardinality_features:  list[str]   # > 50 unique values
    multicollinearity_flag:     bool        # True if VIF > 10 detected
    target_skew:                float       # skewness of the target column


# ── Preprocessing ─────────────────────────────────────────────────────
# Produced by: preprocess step
# consumed by: model step
# Why pipeline_path: the sklearn Pipeline object is saved to disk so
# it can be applied identically to new data at inference time.

class PreprocessResult(BaseModel):
    train_path:         Path
    test_path:          Path
    pipeline_path:      Path            # serialised sklearn Pipeline (joblib)
    outlier_count:      int
    imputation_log:     dict[str, str]  # feature → strategy used
    reduction_applied:  bool
    reduction_method:   Optional[str]   # "PCA" | None
    n_components:       Optional[int]

class SelectedFeatures(BaseModel):
    kept:         list[str]
    dropped:      list[str]
    drop_reasons: dict[str, str]  # feature → reason for dropping


# ── Modelling ─────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    model_name:    str
    cv_rmse_mean:  float
    cv_rmse_std:   float
    cv_r2_mean:    float
    hyperparams:   dict

class CVResults(BaseModel):
    candidates:    list[ModelResult]
    winner:        ModelResult
    baseline_rmse: float # DummyRegressor RMSE — the floor every model must beat
    model_path:    Path 

class FinalEvaluation(BaseModel):
    model_name:          str
    test_rmse:           float
    test_mae:            float
    test_r2:             float
    feature_importances: dict[str, float]
    model_path:          Path 

class PredictionOutput(BaseModel):
    predictions_path:      Path    # parquet: id | y_pred | lower_90 | upper_90
    coverage_achieved:     float   # empirical coverage on test set
    interval_width_median: float   # narrower = more useful to decision-makers


# ── Insights ──────────────────────────────────────────────────────────
# Produced by: InsightNarratorAgent
# Consumed by: ReviewerAgent

class Insight(BaseModel):
    statement:       str
    evidence_cited:  str   # the exact metric that supports this claim
    confidence:      str   # "HIGH" | "MEDIUM" | "LOW"
    category:        str   # "rent_driver" | "undervalued_area" | "prediction"

class InsightReport(BaseModel):
    executive_summary:       str
    insights:                list[Insight]
    assumptions:             list[str]
    limitations:             list[str]
    recommended_next_steps:  list[str]


# ── Review ────────────────────────────────────────────────────────────
# Produced by: ReviewerAgent
# Consumed by: approval gate router

class InsightReview(BaseModel):
    original_statement: str
    support_level:      SupportLevel
    reviewer_comment:   str

class ReviewResult(BaseModel):
    reviews:              list[InsightReview]
    requires_human_review: bool
    overall_verdict:      str  # "APPROVED" | "APPROVED_WITH_WARNINGS" | "REJECTED"


# ── Top-level Flow state ───────────────────────────────────────────────
# WHY: This is the single object the Flow carries from step to step.
# Every step reads from it and writes back to it.
# Nothing is passed as function arguments between steps.

class AnalyticsState(BaseModel):
    # Set at the start
    run_id:    str = ""
    raw_brief: str = ""

    # Filled in step by step — all Optional because they start empty
    parsed_brief:       Optional[ParsedBrief]      = None
    calibration_path:   Optional[Path]             = None
    sources:            Optional[DataSourceList]   = None
    bronze_manifest:    Optional[BronzeManifest]   = None
    validation_report:  Optional[ValidationReport] = None
    silver_path:        Optional[Path]             = None
    gold_path:          Optional[Path]             = None
    active_path:        Optional[Path]             = None
    eda_summary:        Optional[EDASummary]       = None
    preprocess_result:  Optional[PreprocessResult] = None
    selected_features:  Optional[SelectedFeatures] = None
    baseline_result:    Optional[ModelResult]      = None
    cv_results:         Optional[CVResults]        = None
    final_eval:         Optional[FinalEvaluation]  = None
    prediction_output:  Optional[PredictionOutput] = None
    insight_draft:      Optional[InsightReport]    = None
    review_result:      Optional[ReviewResult]     = None
    report_path:        Optional[Path]             = None