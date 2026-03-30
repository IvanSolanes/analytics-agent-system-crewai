# main.py
# WHY: The single entry point for the entire pipeline.
# You run this file. It hands the brief to the Flow. Everything else
# happens automatically from there.

from dotenv import load_dotenv
load_dotenv()   # reads OPENAI_API_KEY from .env before anything else runs

from state.models import AnalyticsState

BRIEF = """
A real-estate analytics team wants to understand rent drivers across cities,
identify undervalued areas, and predict listing prices using listing data
plus socioeconomic and geographic indicators.
"""

def main():
    # Why we import the Flow here and not at the top:
    # The Flow imports agents, which import the OpenAI client.
    # The OpenAI client reads the API key at import time.
    # load_dotenv() must run first or the key will be missing.
    from flows.analytics_flow import RentAnalyticsFlow

    initial_state = AnalyticsState(raw_brief=BRIEF)
    flow = RentAnalyticsFlow()
    flow.kickoff(inputs=initial_state.model_dump())

if __name__ == "__main__":
    main()