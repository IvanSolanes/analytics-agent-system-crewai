# main.py
# WHY SO SHORT: Every decision lives in the Flow, steps, and agents.
# This file has one job — load the API key, define the brief,
# hand it to the Flow, and start it.
# The entire 19-step pipeline runs from this single call.

from dotenv import load_dotenv
load_dotenv()
# WHY LOAD_DOTENV FIRST: The OpenAI client reads the API key at import
# time. If we import the Flow before loading .env the key will be
# missing and every agent call will fail with an authentication error.

from state.models import AnalyticsState
from flows.analytics_flow import RentAnalyticsFlow

BRIEF = """
A real-estate analytics team wants to understand rent drivers across
cities, identify undervalued areas, and predict listing prices using
listing data plus socioeconomic and geographic indicators.
"""

def main():
    print("Starting Rent Analytics Pipeline...")
    print(f"Brief: {BRIEF.strip()[:80]}...")

    initial_state = AnalyticsState(raw_brief=BRIEF)
    flow = RentAnalyticsFlow()
    flow.kickoff(inputs=initial_state.model_dump())

if __name__ == "__main__":
    main()