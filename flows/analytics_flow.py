# flows/analytics_flow.py
from datetime import datetime
from crewai.flow.flow import Flow, start, listen, router

from state.models import AnalyticsState, ParsedBrief
from guardrails.provenance import generate_run_id, log_event
from steps import extract, validate, transform, eda, preprocess, model, report
from agents.source_discovery import source_crew
from agents.insight_narrator import narrator_crew
from agents.reviewer import reviewer_crew


def _log(msg: str):
    """Timestamped console progress — so we know the pipeline is alive."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


class RentAnalyticsFlow(Flow[AnalyticsState]):

    @start()
    def ingest_brief(self):
        _log("STEP 1/19 — Ingesting brief and generating run ID...")
        self.state.run_id = generate_run_id()
        self.state.parsed_brief = ParsedBrief(
            goal="understand rent drivers and predict listing prices",
            target="listing price / rent",
            geographies=["US cities"],
            data_needs=["listing", "socioeconomic", "geographic"],
            raw=self.state.raw_brief,
        )
        log_event(self.state.run_id, "FLOW_START",
                  {"brief_length": len(self.state.raw_brief)})
        _log(f"Run ID: {self.state.run_id}")

    @listen(ingest_brief)
    def discover_sources(self):
        _log("STEP 2/19 — Calling SourceDiscovery agent (LLM)...")
        _log("  This may take 15-30 seconds...")
        log_event(self.state.run_id, "AGENT_CALL",
                  {"agent": "source_discovery"})
        result = source_crew.kickoff(
            inputs={"brief": self.state.parsed_brief.model_dump()}
        )
        self.state.sources = result.pydantic
        names = [s.name for s in self.state.sources.sources]
        _log(f"  Found {len(names)} sources: {names}")
        log_event(self.state.run_id, "SOURCES_FOUND", {
            "count": len(self.state.sources.sources),
            "names": names,
        })

    @listen(discover_sources)
    def extract_data(self):
        _log("STEP 3/19 — Extracting raw data to bronze layer...")
        _log("  Downloading from external sources (may take 30-60s)...")
        self.state.bronze_manifest = extract.run(
            sources=self.state.sources.sources,
            run_id=self.state.run_id,
        )
        n = len(self.state.bronze_manifest.files)
        _log(f"  Downloaded {n} file(s) to bronze layer.")

    @listen(extract_data)
    def validate_bronze(self):
        _log("STEP 4/19 — Validating bronze files (schema, nulls, checksums)...")
        self.state.validation_report = validate.bronze(
            manifest=self.state.bronze_manifest,
        )
        r = self.state.validation_report
        _log(f"  Validation complete — "
             f"{sum(1 for f in r.failures if f.severity.value == 'CRITICAL')} critical, "
             f"{sum(1 for f in r.failures if f.severity.value == 'WARNING')} warnings.")
        if r.has_critical_failures:
            failures = [f.detail for f in r.failures]
            log_event(self.state.run_id, "FLOW_HALT",
                      {"reason": "critical bronze validation failures",
                       "failures": failures})
            raise ValueError(f"Bronze validation failed: {failures}")

    @listen(validate_bronze)
    def transform_silver(self):
        _log("STEP 5/19 — Transforming bronze → silver (clean, type, dedup)...")
        self.state.silver_path = transform.to_silver(
            manifest=self.state.bronze_manifest,
            run_id=self.state.run_id,
        )
        _log("  Silver layer written.")

    @listen(transform_silver)
    def transform_gold(self):
        _log("STEP 6/19 — Transforming silver → gold (join, feature engineering)...")
        self.state.gold_path = transform.to_gold(
            silver_path=self.state.silver_path,
            run_id=self.state.run_id,
        )
        _log("  Gold layer written.")

    @listen(transform_gold)
    def run_eda(self):
        _log("STEP 7/19 — Running EDA (distributions, correlations, VIF)...")
        self.state.eda_summary = eda.run(
            gold_path=self.state.gold_path,
            run_id=self.state.run_id,
        )
        e = self.state.eda_summary
        _log(f"  EDA complete — {e.row_count} rows, "
             f"{e.feature_count} features, "
             f"target skew: {e.target_skew:.3f}, "
             f"multicollinearity: {e.multicollinearity_flag}")

    @listen(run_eda)
    def preprocess_data(self):
        _log("STEP 8/19 — Preprocessing (split, impute, scale, outlier detection)...")
        _log("  Fitting pipeline on training data only...")
        self.state.preprocess_result = preprocess.build_pipeline(
            gold_path=self.state.gold_path,
            eda=self.state.eda_summary,
            run_id=self.state.run_id,
        )
        p = self.state.preprocess_result
        _log(f"  Done — {p.outlier_count} outliers flagged, "
             f"PCA: {p.reduction_method or 'None'}")

    @listen(preprocess_data)
    def select_features(self):
        _log("STEP 9/19 — Selecting features (variance threshold)...")
        self.state.selected_features = preprocess.select_features(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )
        s = self.state.selected_features
        _log(f"  Kept {len(s.kept)} features, dropped {len(s.dropped)}.")

    @listen(select_features)
    def train_baseline(self):
        _log("STEP 10/19 — Training baseline model (DummyRegressor, 5-fold CV)...")
        self.state.baseline_result = model.train_baseline(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )
        b = self.state.baseline_result
        _log(f"  Baseline CV RMSE: {b.cv_rmse_mean:.4f}")

    @listen(train_baseline)
    def select_model(self):
        _log("STEP 11/19 — Cross-validating candidate models (may take 1-2 min)...")
        _log(f"  Candidates: ridge, lasso, random_forest, xgboost")
        self.state.cv_results = model.select_via_cv(
            preprocess_result=self.state.preprocess_result,
            baseline=self.state.baseline_result,
            run_id=self.state.run_id,
        )
        cv = self.state.cv_results
        _log(f"  Winner: {cv.winner.model_name} "
             f"| CV RMSE: {cv.winner.cv_rmse_mean:.4f} "
             f"± {cv.winner.cv_rmse_std:.4f} "
             f"| R²: {cv.winner.cv_r2_mean:.4f}")

    @listen(select_model)
    def evaluate_final(self):
        _log("STEP 12/19 — Final evaluation on locked test set (evaluated ONCE)...")
        self.state.final_eval = model.evaluate_on_test(
            cv_results=self.state.cv_results,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )
        fe = self.state.final_eval
        _log(f"  Test RMSE: {fe.test_rmse:.4f} "
             f"| MAE: {fe.test_mae:.4f} "
             f"| R²: {fe.test_r2:.4f}")

    @listen(evaluate_final)
    def compute_intervals(self):
        _log("STEP 13/19 — Computing conformal prediction intervals...")
        self.state.prediction_output = model.conformal_intervals(
            final_eval=self.state.final_eval,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )
        po = self.state.prediction_output
        _log(f"  Coverage: {po.coverage_achieved:.1%} "
             f"(target: 90%) "
             f"| Median interval width: {po.interval_width_median:.4f}")

    @listen(compute_intervals)
    def generate_insights(self):
        _log("STEP 14/19 — Calling InsightNarrator agent (LLM)...")
        _log("  Converting model results to business language...")
        log_event(self.state.run_id, "AGENT_CALL",
                  {"agent": "insight_narrator"})
        result = narrator_crew.kickoff(inputs={
            "brief":      self.state.parsed_brief.model_dump(mode='json'),
            "eda":        self.state.eda_summary.model_dump(mode='json'),
            "cv_results": self.state.cv_results.model_dump(mode='json'),
            "final_eval": self.state.final_eval.model_dump(mode='json'),
        })
        self.state.insight_draft = result.pydantic
        _log(f"  Generated {len(self.state.insight_draft.insights)} insights.")

    @listen(generate_insights)
    def review_insights(self):
        _log("STEP 15/19 — Calling Reviewer agent (LLM — adversarial check)...")
        _log("  Challenging every claim against the evidence bundle...")
        log_event(self.state.run_id, "AGENT_CALL", {"agent": "reviewer"})
        result = reviewer_crew.kickoff(inputs={
            "draft": self.state.insight_draft.model_dump(mode='json'),
            "evidence": {
                "eda":        self.state.eda_summary.model_dump(mode='json'),
                "cv_results": self.state.cv_results.model_dump(mode='json'),
                "final_eval": self.state.final_eval.model_dump(mode='json'),
            },
        })
        self.state.review_result = result.pydantic
        rv = self.state.review_result
        supported = sum(1 for r in rv.reviews
                        if r.support_level == "SUPPORTED")
        weak      = sum(1 for r in rv.reviews
                        if r.support_level == "WEAK")
        unsup     = sum(1 for r in rv.reviews
                        if r.support_level == "UNSUPPORTED")
        _log(f"  Review: {supported} SUPPORTED, {weak} WEAK, "
             f"{unsup} UNSUPPORTED → {rv.overall_verdict}")

        # Show each claim result in the console immediately
        for r in rv.reviews:
            symbol = "✓" if r.support_level == "SUPPORTED" \
                     else "~" if r.support_level == "WEAK" else "✗"
            print(f"    [{symbol}] {r.original_statement[:90]}")
            print(f"         {r.reviewer_comment}")
        log_event(self.state.run_id, "REVIEW_COMPLETE", {
            "verdict":        rv.overall_verdict,
            "requires_human": rv.requires_human_review,
        })

    @router(review_insights)
    def approval_gate(self):
        _log("STEP 16/19 — Approval gate...")
        if self.state.review_result.requires_human_review:
            _log("  ⚠ Human review required — halting before report.")
            log_event(self.state.run_id, "HUMAN_REVIEW_REQUIRED",
                      {"verdict": self.state.review_result.overall_verdict})
            return "await_approval"
        _log("  ✓ All insights approved — proceeding to report.")
        return "report_approved"

    @listen("await_approval")
    def wait_for_approval(self):
        _log("STEP 17/19 — PAUSED: awaiting human approval.")
        log_event(self.state.run_id, "AWAITING_HUMAN_APPROVAL",
                  {"verdict": self.state.review_result.overall_verdict})

        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║           ⚠  HUMAN REVIEW REQUIRED               ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print(f"  Verdict: {self.state.review_result.overall_verdict}")
        print()
        print("  INSIGHT REVIEW DETAILS:")
        print("  " + "─" * 60)

        label_colors = {
            "SUPPORTED":   "✓",
            "WEAK":        "~",
            "UNSUPPORTED": "✗",
        }

        for r in self.state.review_result.reviews:
            symbol = label_colors.get(r.support_level, "?")
            print(f"  [{symbol} {r.support_level}]")
            print(f"  Claim:   {r.original_statement[:120]}")
            print(f"  Review:  {r.reviewer_comment}")
            print("  " + "─" * 60)

        print()
        print("  To inspect the full provenance log:")
        print(f"  python check_review.py")
        print()

    @listen("report_approved")
    def produce_report(self):
        _log("STEP 18/19 — Rendering final markdown report...")
        self.state.report_path = report.render(
            state=self.state,
            run_id=self.state.run_id,
        )
        log_event(self.state.run_id, "FLOW_COMPLETE",
                  {"report": str(self.state.report_path)})
        _log("STEP 19/19 — Pipeline complete.")
        print(f"\n  ✓ Report: {self.state.report_path}")
        print(f"  Verdict: {self.state.review_result.overall_verdict}\n")