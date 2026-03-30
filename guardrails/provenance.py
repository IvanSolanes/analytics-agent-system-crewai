# guardrails/provenance.py
# WHY: Writes a timestamped log entry for every significant pipeline event.
# Creates an immutable audit trail from brief to final report.

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from config.settings import PROVENANCE_DIR


def log_event(run_id: str, event: str, data: dict = None) -> None:
    """
    Write one line to the run's provenance log.
    Why JSONL (one JSON object per line): easy to append, easy to parse,
    human-readable, and works with any log analysis tool.
    """
    entry = {
        "run_id":    run_id,
        "event":     event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data":      data or {}
    }

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROVENANCE_DIR / f"{run_id}.jsonl"

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def file_checksum(path: Path) -> str:
    """
    Return the SHA-256 hash of a file.
    Why: Proves the raw file has not been modified since it was downloaded.
    If the checksum at bronze validation matches the one recorded at
    download time, the file is untouched.
    """
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def generate_run_id() -> str:
    """
    Create a unique ID for this pipeline run.
    Why timestamp-based: human-readable, sortable, and unique enough
    for a pipeline that won't run thousands of times per second.
    """
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

def read_log(run_id: str) -> list[dict]:
    """
    Read back all events for a given run.
    Why: The report step reads this to include a full audit trail.
    Also useful for debugging — see exactly what happened and when.
    """
    log_path = PROVENANCE_DIR / f"{run_id}.jsonl"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]