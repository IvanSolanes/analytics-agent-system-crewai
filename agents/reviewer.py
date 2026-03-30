# agents/reviewer.py
# WHY AN AGENT HERE: Adversarial review of insight claims requires
# language judgment — reading a statement, finding the evidence it
# cites, and deciding whether the evidence actually supports the claim.
# WHY THIS IS THE MOST CRITICAL AGENT: It is the last guardrail before
# outputs reach decision-makers. It must be explicitly sceptical.
# WHY BOUNDED: Receives only the draft and evidence bundle.
# Output is typed ReviewResult — Pydantic validates immediately.

from crewai import Agent, Task, Crew
from state.models import ReviewResult
from config.settings import LLM_MODEL


reviewer_agent = Agent(
    role="Critical Analytics Reviewer",

    goal=(
        "Challenge every claim in the insight draft. "
        "Mark any claim that cannot be traced to a specific metric "
        "in the evidence bundle as UNSUPPORTED. "
        "Your job is to protect decision-makers from overconfident "
        "or unsupported conclusions."
    ),

    backstory=(
        "You are a senior data scientist whose entire job is to find "
        "flaws in analytical outputs before they reach executives. "
        "You have seen countless cases where confident-sounding insights "
        "turned out to be unsupported by the actual data. "
        "You are not adversarial for its own sake — you want the final "
        "output to be genuinely trustworthy. But you demand evidence "
        "for every single claim. 'The data suggests' is not evidence. "
        "A specific metric with a specific value is evidence."
    ),

    llm=LLM_MODEL,
    verbose=False,
    allow_delegation=False,
)


reviewer_task = Task(
    description=(
        "You have two inputs:\n\n"
        "DRAFT INSIGHTS:\n{draft}\n\n"
        "EVIDENCE BUNDLE (the only valid sources of truth):\n{evidence}\n\n"
        "For each insight in the draft, you must:\n"
        "1. Read the statement carefully.\n"
        "2. Find the evidence_cited field in that insight.\n"
        "3. Check whether that evidence actually exists in the "
        "evidence bundle with the value claimed.\n"
        "4. Assign a support_level:\n"
        "   - SUPPORTED: the cited metric exists and directly supports "
        "the claim\n"
        "   - WEAK: the metric exists but the claim overstates what "
        "it proves\n"
        "   - UNSUPPORTED: the cited metric does not exist in the "
        "evidence bundle, or the claim is not supported by any metric\n"
        "5. Write a reviewer_comment explaining your decision in "
        "one sentence.\n\n"
        "Then set:\n"
        "- requires_human_review: true if ANY insight is UNSUPPORTED, "
        "false otherwise\n"
        "- overall_verdict: 'APPROVED' if all SUPPORTED, "
        "'APPROVED_WITH_WARNINGS' if any WEAK but none UNSUPPORTED, "
        "'REJECTED' if any UNSUPPORTED\n\n"
        "Return ONLY valid JSON matching the ReviewResult schema. "
        "No prose. No markdown fences. No text before or after the JSON."
    ),

    expected_output=(
        "A JSON object matching the ReviewResult schema with fields: "
        "reviews, requires_human_review, overall_verdict."
    ),

    output_pydantic=ReviewResult,
    agent=reviewer_agent,
)


reviewer_crew = Crew(
    agents=[reviewer_agent],
    tasks=[reviewer_task],
    verbose=False,
)