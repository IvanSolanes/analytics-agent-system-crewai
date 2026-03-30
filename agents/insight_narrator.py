# agents/insight_narrator.py
# WHY AN AGENT HERE: Translating model outputs into business language
# requires judgment that pure Python cannot provide.
# WHY BOUNDED: The agent receives only the evidence bundle we give it.
# It cannot access anything else. Output is typed InsightReport — if
# the LLM produces free text the pipeline halts immediately.

from crewai import Agent, Task, Crew
from state.models import InsightReport
from config.settings import LLM_MODEL


narrator_agent = Agent(
    role="Analytics Translator",

    goal=(
        "Convert statistical outputs into concise, evidence-backed "
        "business insights. Every single claim must cite the specific "
        "metric or test result that supports it. "
        "Never speculate beyond the evidence you are given."
    ),

    backstory=(
        "You translate data science outputs into plain language for "
        "real-estate executives and investment teams. You are rigorous "
        "and conservative — you would rather say less and be right "
        "than say more and be wrong. You always flag areas where "
        "the evidence is weak or absent rather than filling gaps "
        "with assumptions."
    ),

    llm=LLM_MODEL,
    verbose=False,
    allow_delegation=False,
)


narrator_task = Task(
    description=(
        "You have the following evidence bundle:\n\n"
        "Original brief:\n{brief}\n\n"
        "EDA summary:\n{eda}\n\n"
        "Cross-validation results:\n{cv_results}\n\n"
        "Final test evaluation:\n{final_eval}\n\n"
        "Your job:\n"
        "1. Write a short executive_summary (3-4 sentences max).\n"
        "2. Write 4 to 6 insights. For each insight:\n"
        "   - statement: the business claim in plain English\n"
        "   - evidence_cited: the exact metric that supports it "
        "(e.g. 'test_r2=0.81', 'rent_to_income correlation=0.74')\n"
        "   - confidence: HIGH if evidence is strong and direct, "
        "MEDIUM if indirect, LOW if the signal is weak\n"
        "   - category: one of 'rent_driver', 'undervalued_area', "
        "'prediction', 'data_quality'\n"
        "3. List 3 to 5 key assumptions the analysis rests on.\n"
        "4. List 3 to 5 honest limitations of the analysis.\n"
        "5. List 2 to 3 concrete recommended next steps.\n\n"
        "Return ONLY valid JSON matching the InsightReport schema. "
        "No prose. No markdown fences. No text before or after the JSON."
    ),

    expected_output=(
        "A JSON object matching the InsightReport schema with fields: "
        "executive_summary, insights, assumptions, "
        "limitations, recommended_next_steps."
    ),

    output_pydantic=InsightReport,
    agent=narrator_agent,
)


narrator_crew = Crew(
    agents=[narrator_agent],
    tasks=[narrator_task],
    verbose=False,
)