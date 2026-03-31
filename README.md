# Analytics Agent System — CrewAI

A deterministic, guardrailed multi-agent analytics pipeline built with CrewAI.

## What it does

Takes a plain-English business brief, discovers relevant public data sources, runs a full data engineering and data science workflow end to end, and produces a decision-ready insight report with assumptions, limitations, and uncertainty quantification — all in under 60 seconds.

## V1 Use Case: Rent & Housing Analytics

> *"A real-estate analytics team wants to understand rent drivers across cities, identify undervalued areas, and predict listing prices using listing data plus socioeconomic and geographic indicators."*

**V1 results on Zillow data (700 US metro areas):**
- Winner model: Ridge Regression
- CV RMSE: 0.211 ± 0.022 (log scale)
- Test R²: 0.509 (no overfitting)
- Conformal coverage: 97.1% (target: 90%) ✅
- Total runtime: ~34 seconds

## Design Principles

| Principle | How it is implemented |
|---|---|
| Deterministic backbone | CrewAI Flow with `@start` / `@listen` / `@router` — fixed execution order |
| Agents only where needed | 3 LLM agents only (16% of steps). 84% is pure Python |
| Typed state everywhere | Pydantic `AnalyticsState` carries all data between steps |
| Fail fast and loudly | Schema checks, row count guards, checksum verification at bronze layer |
| Test set touched once | `evaluate_on_test()` is called exactly once, never in a loop |
| Uncertainty quantification | Conformal prediction intervals with 90% coverage guarantee |
| Adversarial review | Reviewer agent challenges every insight claim before the report is written |
| Full audit trail | JSONL provenance log for every step of every run |

## Architecture: 19 Steps, 3 Agents

```
Step 1   ingest_brief           DETERMINISTIC  Parse brief, generate run ID
Step 2   discover_sources       ── AGENT ──    SourceDiscoveryAgent (LLM)
Step 3   extract_data           DETERMINISTIC  Download CSV/API → bronze layer
Step 4   validate_bronze        DETERMINISTIC  Schema, nulls, checksums
Step 5   transform_silver       DETERMINISTIC  Clean, type-cast, deduplicate
Step 6   transform_gold         DETERMINISTIC  Join, feature engineering, log(rent)
Step 7   run_eda                DETERMINISTIC  Stats, correlations, VIF, plots
Step 8   preprocess_data        DETERMINISTIC  Split, impute, scale, outlier detect
Step 9   select_features        DETERMINISTIC  Variance threshold, drop log
Step 10  train_baseline         DETERMINISTIC  DummyRegressor, 5-fold CV
Step 11  select_model           DETERMINISTIC  CV across ridge/lasso/RF/XGBoost
Step 12  evaluate_final         DETERMINISTIC  Test set — evaluated ONCE only
Step 13  compute_intervals      DETERMINISTIC  Conformal prediction intervals
Step 14  generate_insights      ── AGENT ──    InsightNarratorAgent (LLM)
Step 15  review_insights        ── AGENT ──    ReviewerAgent (LLM — adversarial)
Step 16  approval_gate          DETERMINISTIC  Router: approved or human review
Step 17  wait_for_approval      DETERMINISTIC  Halt if reviewer flagged issues
Step 18  produce_report         DETERMINISTIC  Render markdown report
```

## Data Layers (Medallion Architecture)

```
Bronze  Raw files exactly as downloaded. Never modified. SHA-256 checksummed.
Silver  Cleaned, typed, deduplicated. Schema enforced.
Gold    Joined, feature-engineered, ML-ready. Parquet format.
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | CrewAI 1.12 Flows |
| Data | Pandas 3.0, Parquet |
| ML | scikit-learn 1.8, XGBoost 3.2 |
| Uncertainty | MAPIE 1.3 (conformal prediction) |
| State | Pydantic 2.x |
| LLM | OpenAI gpt-4o-mini |

## Setup

```powershell
# Python 3.12 required (not 3.14)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
```

## Running

```powershell
# Clean all generated data for a fresh run
python cleanup.py

# Run the full pipeline
python main.py
```

## Outputs

| Location | Contents |
|---|---|
| `outputs/reports/` | Final markdown insight report |
| `outputs/predictions/` | Parquet: y_pred, lower_90, upper_90 per city |
| `outputs/eda/` | Correlation heatmap, target distribution plots |
| `outputs/models/` | Serialised sklearn pipeline + winning model |
| `outputs/provenance/` | Full JSONL audit trail for every run |

## Inspecting a Failed Review

```powershell
python check_review.py
```