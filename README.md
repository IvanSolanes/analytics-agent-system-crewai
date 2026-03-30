# Analytics Agent System — CrewAI

A deterministic, guardrailed multi-agent analytics pipeline built with CrewAI.

## What it does
Takes a business brief, discovers relevant data sources, runs a full
data engineering and data science workflow, and produces decision-ready
insights with assumptions, limitations, and uncertainty quantification.

## Design principles
- Flow as the deterministic backbone — not an autonomous swarm
- LLM agents called only where judgment adds value (3 agents total)
- Typed Pydantic state passed between every step
- Bronze / Silver / Gold data layers with provenance tracking
- Test set evaluated exactly once at the end

## Use case: Rent & Housing Analytics
Predict listing prices across cities using listing data,
socioeconomic indicators, and geographic features.

## Stack
Python 3.12 · CrewAI 1.12 · scikit-learn · XGBoost · MAPIE · Pandas

## Setup
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```
Run:
```powershell
python main.py
```