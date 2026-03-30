# steps/report.py
# WHY: Renders the final decision-ready report from the completed
# pipeline state. Pure Python — no LLM needed. The insights were
# already generated and reviewed. This step just formats them.

from pathlib import Path
from datetime import datetime, timezone

from config.settings import REPORTS_DIR
from guardrails.provenance import log_event, read_log


def render(state, run_id: str) -> Path:
    """
    Render the full pipeline output as a markdown report.
    WHY MARKDOWN: Human-readable, version-controllable, and renderable
    in any browser, notebook, or documentation system.
    WHY INCLUDE PROVENANCE: Every section links back to the data and
    decisions that produced it. The report is self-auditing.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []

    # ── Header ───────────────────────────────────────────────────────
    lines += [
        "# Rent Analytics Report",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Review Verdict:** {state.review_result.overall_verdict}",
        "",
    ]

    # ── Brief ─────────────────────────────────────────────────────────
    lines += [
        "## Business Brief",
        state.raw_brief.strip(),
        "",
    ]

    # ── Data sources ──────────────────────────────────────────────────
    lines += ["## Data Sources Used", ""]
    for src in state.sources.sources:
        lines.append(
            f"- **{src.name}** ({src.data_type.value}) — {src.justification}"
        )
    lines.append("")

    # ── Data quality ─────────────────────────────────────────────────
    lines += ["## Data Quality", ""]
    if state.validation_report.failures:
        for f in state.validation_report.failures:
            lines.append(f"- `{f.severity.value}` | {f.source} | {f.detail}")
    else:
        lines.append("No validation failures detected.")
    lines.append("")

    # ── EDA summary ──────────────────────────────────────────────────
    eda = state.eda_summary
    lines += [
        "## Dataset Summary",
        f"- **Rows:** {eda.row_count:,}",
        f"- **Features:** {eda.feature_count}",
        f"- **Target skew (log rent):** {eda.target_skew}",
        f"- **Multicollinearity detected:** {eda.multicollinearity_flag}",
        f"- **High-null features:** "
        f"{', '.join(eda.high_null_features) or 'None'}",
        "",
    ]

    # ── Preprocessing ─────────────────────────────────────────────────
    pre = state.preprocess_result
    lines += [
        "## Preprocessing",
        f"- **Outliers detected:** {pre.outlier_count}",
        f"- **Dimensionality reduction:** "
        f"{pre.reduction_method or 'None'}",
        f"- **Features kept after selection:** "
        f"{len(state.selected_features.kept)}",
        f"- **Features dropped:** "
        f"{len(state.selected_features.dropped)}",
        "",
    ]

    # ── Model results ─────────────────────────────────────────────────
    cv  = state.cv_results
    fe  = state.final_eval
    po  = state.prediction_output

    lines += [
        "## Model Results",
        f"- **Baseline RMSE (CV):** {cv.baseline_rmse:.4f}",
        f"- **Winner:** {cv.winner.model_name}",
        f"- **Winner CV RMSE:** "
        f"{cv.winner.cv_rmse_mean:.4f} ± {cv.winner.cv_rmse_std:.4f}",
        f"- **Winner CV R²:** {cv.winner.cv_r2_mean:.4f}",
        "",
        "### Final Test Set Evaluation *(evaluated once, on locked test set)*",
        f"- **Test RMSE:** {fe.test_rmse:.4f}",
        f"- **Test MAE:** {fe.test_mae:.4f}",
        f"- **Test R²:** {fe.test_r2:.4f}",
        "",
        "### Prediction Intervals (Conformal)",
        f"- **Coverage achieved:** {po.coverage_achieved:.1%}",
        f"- **Median interval width:** {po.interval_width_median:.4f}",
        "",
    ]

    # ── Insights ──────────────────────────────────────────────────────
    draft = state.insight_draft
    lines += [
        "## Executive Summary",
        draft.executive_summary,
        "",
        "## Key Insights",
        "",
    ]
    for insight in draft.insights:
        lines += [
            f"### {insight.category.replace('_', ' ').title()}",
            f"**{insight.statement}**",
            f"*Evidence: {insight.evidence_cited}*",
            f"*Confidence: {insight.confidence}*",
            "",
        ]

    lines += ["## Assumptions", ""]
    for a in draft.assumptions:
        lines.append(f"- {a}")
    lines += ["", "## Limitations", ""]
    for lim in draft.limitations:
        lines.append(f"- {lim}")
    lines += ["", "## Recommended Next Steps", ""]
    for step in draft.recommended_next_steps:
        lines.append(f"- {step}")
    lines.append("")

    # ── Review ────────────────────────────────────────────────────────
    review = state.review_result
    lines += [
        "## Insight Review",
        f"**Overall verdict:** {review.overall_verdict}",
        f"**Requires human review:** {review.requires_human_review}",
        "",
    ]
    for r in review.reviews:
        lines.append(
            f"- `{r.support_level.value}` — {r.original_statement[:80]}... "
            f"| {r.reviewer_comment}"
        )
    lines.append("")

    # ── Provenance trail ──────────────────────────────────────────────
    lines += ["## Provenance Trail", ""]
    events = read_log(run_id)
    for e in events:
        lines.append(f"- `{e['timestamp']}` — **{e['event']}**")
    lines.append("")

    # ── Write report ──────────────────────────────────────────────────
    report_path = REPORTS_DIR / f"{run_id}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log_event(run_id, "REPORT_WRITTEN", {"path": str(report_path)})

    return report_path