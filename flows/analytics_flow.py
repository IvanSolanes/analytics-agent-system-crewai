# flows/analytics_flow.py
from crewai.flow.flow import Flow, start, listen, router

from state.models import AnalyticsState, ParsedBrief
from guardrails.provenance import generate_run_id, log_event
from steps import extract, validate, transform, eda, preprocess, model, report
from agents.source_discovery import source_crew
from agents.insight_narrator import narrator_crew
from agents.reviewer import reviewer_crew


class RentAnalyticsFlow(Flow[AnalyticsState]):

    @start()
    def ingest_brief(self):
        self.state.run_id = generate_run_id()
        # WHY ParsedBrief model not dict: avoids Pydantic serialization
        # warnings and keeps state fully typed end to end.
        self.state.parsed_brief = ParsedBrief(
            goal="understand rent drivers and predict listing prices",
            target="listing price / rent",
            geographies=["US cities"],
            data_needs=["listing", "socioeconomic", "geographic"],
            raw=self.state.raw_brief,
        )
        log_event(self.state.run_id, "FLOW_START",
                  {"brief_length": len(self.state.raw_brief)})

    @listen(ingest_brief)
    def discover_sources(self):
        log_event(self.state.run_id, "AGENT_CALL",
                  {"agent": "source_discovery"})
        result = source_crew.kickoff(
            inputs={"brief": self.state.parsed_brief.model_dump()}
        )
        self.state.sources = result.pydantic
        log_event(self.state.run_id, "SOURCES_FOUND", {
            "count": len(self.state.sources.sources),
            "names": [s.name for s in self.state.sources.sources],
        })

    @listen(discover_sources)
    def extract_data(self):
        self.state.bronze_manifest = extract.run(
            sources=self.state.sources.sources,
            run_id=self.state.run_id,
        )

    @listen(extract_data)
    def validate_bronze(self):
        self.state.validation_report = validate.bronze(
            manifest=self.state.bronze_manifest,
        )
        if self.state.validation_report.has_critical_failures:
            failures = [f.detail for f in
                        self.state.validation_report.failures]
            log_event(self.state.run_id, "FLOW_HALT",
                      {"reason": "critical bronze validation failures",
                       "failures": failures})
            raise ValueError(f"Bronze validation failed: {failures}")

    @listen(validate_bronze)
    def transform_silver(self):
        self.state.silver_path = transform.to_silver(
            manifest=self.state.bronze_manifest,
            run_id=self.state.run_id,
        )

    @listen(transform_silver)
    def transform_gold(self):
        self.state.gold_path = transform.to_gold(
            silver_path=self.state.silver_path,
            run_id=self.state.run_id,
        )

    @listen(transform_gold)
    def run_eda(self):
        self.state.eda_summary = eda.run(
            gold_path=self.state.gold_path,
            run_id=self.state.run_id,
        )

    # WHY NO ROUTER HERE: The PCA decision is made inside build_pipeline
    # based on eda.feature_count. A router would create parallel async
    # chains in CrewAI — every downstream step would run multiple times
    # concurrently, corrupting shared files. One linear path is correct.
    @listen(run_eda)
    def preprocess_data(self):
        self.state.preprocess_result = preprocess.build_pipeline(
            gold_path=self.state.gold_path,
            eda=self.state.eda_summary,
            run_id=self.state.run_id,
        )

    @listen(preprocess_data)
    def select_features(self):
        self.state.selected_features = preprocess.select_features(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    @listen(select_features)
    def train_baseline(self):
        self.state.baseline_result = model.train_baseline(
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    @listen(train_baseline)
    def select_model(self):
        self.state.cv_results = model.select_via_cv(
            preprocess_result=self.state.preprocess_result,
            baseline=self.state.baseline_result,
            run_id=self.state.run_id,
        )

    @listen(select_model)
    def evaluate_final(self):
        self.state.final_eval = model.evaluate_on_test(
            cv_results=self.state.cv_results,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    @listen(evaluate_final)
    def compute_intervals(self):
        self.state.prediction_output = model.conformal_intervals(
            final_eval=self.state.final_eval,
            preprocess_result=self.state.preprocess_result,
            run_id=self.state.run_id,
        )

    @listen(compute_intervals)
    def generate_insights(self):
        log_event(self.state.run_id, "AGENT_CALL",
                  {"agent": "insight_narrator"})
        result = narrator_crew.kickoff(inputs={
            "brief":      self.state.parsed_brief.model_dump(mode='json'),
            "eda":        self.state.eda_summary.model_dump(mode='json'),
            "cv_results": self.state.cv_results.model_dump(mode='json'),
            "final_eval": self.state.final_eval.model_dump(mode='json'),
        })
        self.state.insight_draft = result.pydantic

    @listen(generate_insights)
    def review_insights(self):
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
        log_event(self.state.run_id, "REVIEW_COMPLETE", {
            "verdict":        self.state.review_result.overall_verdict,
            "requires_human": self.state.review_result.requires_human_review,
        })

    @router(review_insights)
    def approval_gate(self):
        if self.state.review_result.requires_human_review:
            log_event(self.state.run_id, "HUMAN_REVIEW_REQUIRED",
                      {"verdict": self.state.review_result.overall_verdict})
            return "await_approval"
        return "report_approved"

    @listen("await_approval")
    def wait_for_approval(self):
        log_event(self.state.run_id, "AWAITING_HUMAN_APPROVAL",
                  {"review": self.state.review_result.model_dump()})
        print(f"\n⚠ HUMAN REVIEW REQUIRED\n"
              f"Verdict: {self.state.review_result.overall_verdict}\n"
              f"Check: outputs/provenance/{self.state.run_id}.jsonl\n")

    @listen("report_approved")
    def produce_report(self):
        self.state.report_path = report.render(
            state=self.state,
            run_id=self.state.run_id,
        )
        log_event(self.state.run_id, "FLOW_COMPLETE",
                  {"report": str(self.state.report_path)})
        print(f"\n✓ Pipeline complete\n"
              f"Report: {self.state.report_path}\n"
              f"Verdict: {self.state.review_result.overall_verdict}\n")