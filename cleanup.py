# cleanup.py
# WHY: Wipes all generated data and outputs so the pipeline
# runs fresh from scratch. Never deletes .gitkeep files.

import os
from pathlib import Path

def delete_pattern(folder: str, extension: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        return
    files = [f for f in folder_path.glob(f"*{extension}")
             if f.name != ".gitkeep"]
    for f in files:
        f.unlink()
        print(f"  Deleted: {f.name}")
    if files:
        print(f"  → {len(files)} file(s) removed from {folder}")
    else:
        print(f"  → Nothing to clean in {folder}")

print("Cleaning project outputs...")
print()

delete_pattern("data/bronze",         ".csv")
delete_pattern("data/silver",         ".csv")
delete_pattern("data/gold",           ".parquet")
delete_pattern("outputs/eda",         ".png")
delete_pattern("outputs/models",      ".json")
delete_pattern("outputs/models",      ".joblib")
delete_pattern("outputs/predictions", ".parquet")
delete_pattern("outputs/provenance",  ".jsonl")
delete_pattern("outputs/reports",     ".md")

print()
print("Done. Ready for a fresh run.")