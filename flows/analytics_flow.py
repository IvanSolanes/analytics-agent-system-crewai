# flows/analytics_flow.py
# WHY THIS FILE: The deterministic backbone of the entire pipeline.
# Every step is connected by decorators — the Flow decides what runs
# and when. Agents are called at exactly 3 bounded moments.
# Nothing runs autonomously. Everything is auditable.

from crewai.flow.flow import Flow, start, listen, router, or_

from config.settings import PCA_FEATURE_THRESHOLD
from state.models import AnalyticsState
from guardrails.provenance import generate_run_id, log_event

# Pure Python steps — deterministic, no LLM
from steps import extract, validate, transform, eda, preprocess, model, report

# LLM agents — called only where judgment adds value
from agents.source_discovery import source_crew
from agents.insight_narrator import narrator_crew
from agents.reviewer import reviewer_crew


class RentAnalyticsFlow(Flow[AnalyticsState]):
    """
    The complete rent analytics pipeline as a deterministic Flow.
    State flows through every step as a typed AnalyticsState object.
    Agents are called at steps 2, 15, and 16 only.
    The test set is touched exactly once, at step 13.
    """

    # ── STEP 1 ───────────────────────────────────────────────────────
    @start()
    def ingest_brief(self):
        """
        WHY FIRST: Sets up the run context before anything else.
        Every subsequent step uses the run_id for provenance logging.
        WHY parse_brief here: Converts free text into structured fields
        so agents receive precise inputs, not a blob of prose.
        """
        self.state.run_id = generate_run_id()

        # Parse brief into structured fields
        # Why simple parsing here: this is deterministic string work,
        # not judgment — no agent needed.
        self.state.parsed_brief = {
            "goal":        "understand rent drivers and predict listing prices",
            "target":      "listing price / rent",
            "geographies": ["US cities"],
            "data_needs":  ["listing", "socioeconomic", "geographic"],
            "raw":         self.state.raw_brief,
        }

        log_event(self.state.run_id, "FLOW_START", {
            "brief_length": len(self.state.raw_brief)
        })

    # ── STEP 2 ───────────────────────────────────────────────────────
    @listen(ingest_brief)
    def discover_sources(self):
        """
        WHY AN AGENT: Finding relevant public datasets for a business
        brief requires world knowledge. Pure Python cannot do this.
        WHY BOUNDED: Output is typed DataSourceList — Pydantic validates
        immediately. If the LLM returns prose the pipeline halts.
        """
        log_event(self.state.run_id, "AGENT_CALL", {"agent": "source_discovery"})

        result = source_crew.kickoff(
            inputs={"brief": self.state.parsed_brief}
        )
        self.state.sources = result.pydantic

        log_event(self.state.run_id, "SOURCES_FOUND", {
            "count": len(self.state.sources.sources),
            "names": [s.name for s in self.state.sources.sources],
        })

    # ── STEP 3 ───────────────────────────────────────────────────────
    @listen(discover_sources)
    def extract_data(self):
        """
        WHY PURE PYTHON: Downloading files is deterministic HTTP work.
        No judgment needed — just fetch, save, checksum, log.
        """
        self.state.bronze_manifest = extract.run(
            sources=self.state.sources.sources,
            run_id=self.state.run_id,
        )

    # ── STEP 4 ───────────────────────────────────────────────────────
    @listen(extract_data)
    def validate_bronze(self):
        """
        WHY HALT ON CRITICAL: Bad data discovered early is far cheaper
        than bad data discovered after model training.
        The ValidationReport tells us exactly what failed and why.
        """
        self.state.validation_report = validate.bronze(
            manifest=self.state.bronze_manifest,
        )

        if self.state.validation_report.has_critical_failures:
            failures = [f.detail for f in
                        self.state.validation_report.failures]
            log_event(self.state.run_id, "FLOW_HALT", {
                "reason": "critical bronze validation failures",
                "failures": failures,
            })
            raise ValueError(
                f"Bronze validation failed: {failures}\n"
                f"Check outputs/provenance/{self.state.run_id}.jsonl "
                f"for details."
            )

    # ── STEP 5 ───────────────────────────────────────────────────────
    @listen(validate_bronze)
    def transform_silver(self):
        """
        WHY PURE PYTHON: Cleaning rules are deterministic — strip
        whitespace, cast types, drop duplicates. No judgment needed.
        WHY SILVER IS SEPARATE FROM GOLD: If cleaning breaks, we fix
        it here without touching feature engineering.
        """
        self.state.silver_path = transform.to_silver(
            manifest=self.state.bronze_manifest,
            run_id=self.state.run_id,
        )

    # ── STEP 6 ───────────────────────────────────────────────────────
    @listen(transform_silver)
    def transform_gold(self):
        """
        WHY PURE PYTHON: Feature engineering is deterministic math —
        log transforms, ratios, joins. No judgment needed.
        WHY GOLD IS SEPARATE FROM SILVER: Features change more often
        than cleaning logic. Keeping them separate reduces blast radius.
        """
        self.state.gold_path = transform.to_gold(
            silver_path=self.state.silver_path,
            run_id=self.state.run_id,
        )

    # ── STEP 7 ───────────────────────────────────────────────────────
    @listen(transform_gold)
    def run_eda(self):
        """
        WHY BEFORE MODELLING: EDA informs every downstream decision.
        The EDASummary drives the PCA router, the imputation strategy,
        and provides evidence for the insight agent.
        """
        self.state.eda_summary = eda.run(
            gold_path=self.state.gold_path,
            run_id=self.state.run_id,
        )

    # ── STEP 8 — ROUTER ──────────────────────────────────────────────
    @router(run_eda)
    def check_dimensionality(self):
        """
        WHY A ROUTER: PCA is only beneficial when feature count is high.
        Below the threshold it adds complexity for no gain.
        WHY DATA-DRIVEN: The decision is based on eda_summary.feature_count
        — a measured property of the data, not a human assumption.
        """
        if self.state.eda_summary.feature_count > PCA_FEATURE_THRESHOLD:
            log_event(self.state.run_id, "ROUTER_PCA", {
                "feature_count": self.state.eda_summary.feature_count,
                "threshold":     PCA_FEATURE_THRESHOLD,
            })
            return "reduce"
        return "skip_reduction"

    # ── STEP 9a — PCA path ───────────────────────────────────────────
    @listen("reduce")
    def reduce_dimensions(self):
        self.state.active_path = self.state.gold_path
        log_event(self.state.run_id, "PCA_ROUTE_TAKEN")

    # ── STEP 9b — skip path ──────────────────────────────────────────
    @listen("skip_reduction")
    def skip_reduction(self):
        """No PCA needed. Gold path flows directly to preprocessing."""
        self.state.active_path = self.state.gold_path
        log_event(self.state.run_id, "PCA_ROUTE_SKIPPED")

    # ── STEP 10 ──────────────────────────────────────────────────────
    @listen(or_(reduce_dimensions, skip_reduction))
    def preprocess_data(self):
        """
        WHY FIT ON TRAIN ONLY: The single most important rule in ML.
        The preprocessing pipeline is fit here on training data only,
        then applied identically to the test set.
        Fitting on all data before splitting = data leakage = invalid results.
        """
        self.state.preprocess_result = preprocess.build_pipeline(
            gold_path=self.state.active_path,
            eda=self.state.eda_summary,
            run_id=self.state.run_id,
        )

    # ── STEP 11 ──────────────────────────────────────────────────────
    @listen(preprocess_data)
    def select_features(self):
        """
        WHY AFTER PREPROCESSING: Feature selection uses the processed
        values — variance is only meaningful after scaling.
        We log what was dropped and why for full auditability.
        """
        self.state.selected_features = preprocess.select_features(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    # ── STEP 12 ──────────────────────────────────────────────────────
    @listen(select_features)
    def train_baseline(self):
        """
        WHY A BASELINE: Sets the floor. A model that cannot beat
        'predict the mean rent' is not worth deploying regardless
        of how sophisticated it looks.
        """
        self.state.baseline_result = model.train_baseline(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    # ── STEP 13 ──────────────────────────────────────────────────────
    @listen(train_baseline)
    def select_model(self):
        """
        WHY CV ON TRAIN ONLY: Cross-validation selects the winner
        using only training data. The test set is not involved here.
        WHY WE LOG THE WARNING: If the winner barely beats the baseline
        we flag it rather than silently accepting a marginal improvement.
        """
        self.state.cv_results = model.select_via_cv(
            preprocess_result=self.state.preprocess_result,
            baseline=self.state.baseline_result,
            run_id=self.state.run_id,
        )

    # ── STEP 14 ──────────────────────────────────────────────────────
    @listen(select_model)
    def evaluate_final(self):
        """
        WHY EXACTLY ONCE: This is the only honest estimate of real-world
        performance. evaluate_on_test() is called here and nowhere else
        in the entire codebase. Evaluating on the test set multiple times
        — even just to look — invalidates the estimate.
        """
        self.state.final_eval = model.evaluate_on_test(
            cv_results=self.state.cv_results,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    # ── STEP 15 ──────────────────────────────────────────────────────
    @listen(evaluate_final)
    def compute_intervals(self):
        """
        WHY CONFORMAL PREDICTION: Point predictions without uncertainty
        bounds are misleading for business decisions.
        "The model predicts $2,400/month" is less useful than
        "The model predicts $2,400/month, 90% interval: $2,100–$2,750".
        Conformal prediction gives coverage-guaranteed intervals
        with no distributional assumptions.
        """
        self.state.prediction_output = model.conformal_intervals(
            final_eval=self.state.final_eval,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    # ── STEP 16 ──────────────────────────────────────────────────────
    @listen(compute_intervals)
    def generate_insights(self):
        """
        WHY AN AGENT: Converting RMSE scores and correlations into
        sentences an executive can act on requires language judgment.
        WHY BOUNDED: Agent receives only the evidence bundle we pass.
        Output is typed InsightReport — validated immediately.
        """
        log_event(self.state.run_id, "AGENT_CALL",
                  {"agent": "insight_narrator"})

        result = narrator_crew.kickoff(inputs={
            "brief":      self.state.parsed_brief,
            "eda":        self.state.eda_summary.model_dump(),
            "cv_results": self.state.cv_results.model_dump(),
            "final_eval": self.state.final_eval.model_dump(),
        })
        self.state.insight_draft = result.pydantic

    # ── STEP 17 ──────────────────────────────────────────────────────
    @listen(generate_insights)
    def review_insights(self):
        """
        WHY AN AGENT: Adversarial review of language claims requires
        language judgment. Pure Python cannot read a sentence and decide
        whether the cited metric actually supports the claim.
        WHY THIS IS THE LAST GUARDRAIL: Nothing reaches the report
        without passing through this review.
        """
        log_event(self.state.run_id, "AGENT_CALL", {"agent": "reviewer"})

        result = reviewer_crew.kickoff(inputs={
            "draft": self.state.insight_draft.model_dump(),
            "evidence": {
                "eda":        self.state.eda_summary.model_dump(),
                "cv_results": self.state.cv_results.model_dump(),
                "final_eval": self.state.final_eval.model_dump(),
            },
        })
        self.state.review_result = result.pydantic

        log_event(self.state.run_id, "REVIEW_COMPLETE", {
            "verdict":              self.state.review_result.overall_verdict,
            "requires_human":       self.state.review_result.requires_human_review,
        })

    # ── STEP 18 — ROUTER ─────────────────────────────────────────────
    @router(review_insights)
    def approval_gate(self):
        """
        WHY A ROUTER HERE: If the reviewer found unsupported claims
        we pause for human inspection rather than publishing bad outputs.
        WHY NOT ALWAYS PAUSE: Most runs will pass review. Pausing
        every run would make the system unusable.
        """
        if self.state.review_result.requires_human_review:
            log_event(self.state.run_id, "HUMAN_REVIEW_REQUIRED", {
                "verdict": self.state.review_result.overall_verdict,
            })
            return "await_approval"
        return "produce_report"

    # ── STEP 19a — human gate ────────────────────────────────────────
    @listen("await_approval")
    def wait_for_approval(self):
        """
        WHY LOG AND HALT: Writes the full state to provenance so a human
        analyst can inspect what the reviewer flagged before deciding
        whether to proceed. In production this would trigger a Slack
        notification or open a review UI.
        """
        log_event(self.state.run_id, "AWAITING_HUMAN_APPROVAL", {
            "review": self.state.review_result.model_dump(),
            "message": (
                f"Check outputs/provenance/{self.state.run_id}.jsonl "
                f"for the full review result."
            ),
        })
        print(f"\n⚠ HUMAN REVIEW REQUIRED\n"
              f"Verdict: {self.state.review_result.overall_verdict}\n"
              f"Check: outputs/provenance/{self.state.run_id}.jsonl\n")

    # ── STEP 19b — produce report ────────────────────────────────────
    @listen("produce_report")
    def produce_report(self):
        """
        WHY LAST: The report is only produced after every guardrail
        has passed — validation, modelling integrity, and review.
        WHY PURE PYTHON: Rendering a report from structured data
        is deterministic template work. No agent needed.
        """
        self.state.report_path = report.render(
            state=self.state,
            run_id=self.state.run_id,
        )

        log_event(self.state.run_id, "FLOW_COMPLETE", {
            "report": str(self.state.report_path),
            "verdict": self.state.review_result.overall_verdict,
        })

        print(f"\n✓ Pipeline complete\n"
              f"Report: {self.state.report_path}\n"
              f"Verdict: {self.state.review_result.overall_verdict}\n")