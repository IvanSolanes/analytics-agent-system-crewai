# agents/source_discovery.py
# WHY AN AGENT HERE: Mapping a business brief to real public datasets
# requires world knowledge that changes over time. An LLM knows about
# Zillow research data, Census APIs, HUD publications, etc.
# WHY BOUNDED: The agent returns structured JSON only. No free text
# passes to the next step. Output is validated by Pydantic immediately.

from crewai import Agent, Task, Crew
from state.models import DataSourceList
from config.settings import LLM_MODEL


source_agent = Agent(
    role="Public Data Source Specialist",

    goal=(
        "Identify the 4 to 6 most relevant, publicly accessible data sources "
        "for the given business brief. Return structured JSON only. "
        "Every source you recommend must actually exist and be freely accessible."
    ),

    backstory=(
        "You are an expert in public datasets, government open data portals, "
        "and real-estate data APIs. You know sources like Zillow Research, "
        "the US Census Bureau ACS API, HUD Fair Market Rents, and the "
        "Bureau of Labor Statistics. You are precise and sceptical — "
        "you only recommend sources you are highly confident exist and "
        "are accessible without a paid subscription."
    ),

    llm=LLM_MODEL,
    verbose=False,
    allow_delegation=False,   # no sub-agents — this agent works alone
)


source_task = Task(
    description=(
        "You have received this business brief:\n\n"
        "{brief}\n\n"
        "For v1 return ONLY these two verified sources:\n\n"
        "1. name: 'zillow'\n"
        "   url: 'https://files.zillowstatic.com/research/public_csvs/"
        "zori/Metro_zori_uc_sfrcondomfr_sm_month.csv'\n"
        "   data_type: 'listing'\n"
        "   access_method: 'csv_download'\n"
        "   justification: 'Zillow Observed Rent Index provides monthly "
        "median rent by metro area across the US'\n\n"
        "2. name: 'zillow_inventory'\n"
        "   url: 'https://files.zillowstatic.com/research/public_csvs/"
        "invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv'\n"
        "   data_type: 'listing'\n"
        "   access_method: 'csv_download'\n"
        "   justification: 'Zillow for-sale inventory by metro — a supply "
        "signal that drives rent prices'\n\n"
        "Return ONLY valid JSON with key 'sources'. No prose. "
        "No markdown fences. No text before or after the JSON."
    ),

    expected_output=(
        "A JSON object with key 'sources' containing a list of "
        "DataSource objects."
    ),

    output_pydantic=DataSourceList,
    agent=source_agent,
)


source_crew = Crew(
    agents=[source_agent],
    tasks=[source_task],
    verbose=False,
)