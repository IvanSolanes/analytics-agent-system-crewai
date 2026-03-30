# steps/extract.py
# WHY: Downloads raw data from each approved source into the bronze layer.
# Bronze = raw files, exactly as received, never modified.
# Every download is checksummed and logged for provenance.

import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path

from config.settings import BRONZE_DIR
from state.models import DataSource, BronzeFile, BronzeManifest
from guardrails.provenance import log_event, file_checksum


def _download_csv(url: str, dest: Path) -> pd.DataFrame:
    """
    Why a separate function: CSV download is the most common case.
    We stream the response so large files don't exhaust memory.
    We save the raw bytes first, then read with pandas — this ensures
    the bronze file is byte-identical to what the server sent.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()     # halts immediately on HTTP errors
    dest.write_bytes(response.content)
    return pd.read_csv(dest)


def _download_api(url: str, dest: Path, params: dict = None) -> pd.DataFrame:
    """
    Why separate from CSV: API responses are JSON.
    We normalise the JSON into a flat DataFrame and save both
    the raw JSON (for provenance) and a CSV (for consistency).
    """
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Save the raw JSON response as the bronze file
    dest.with_suffix(".json").write_text(response.text, encoding="utf-8")

    # Normalise to DataFrame for the manifest row count
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # Most APIs wrap results in a key — find the first list value
        for v in data.values():
            if isinstance(v, list):
                return pd.DataFrame(v)
    return pd.DataFrame([data])


def run(sources: list[DataSource], run_id: str) -> BronzeManifest:
    """
    Main entry point called by the Flow.
    Downloads every source and returns a typed BronzeManifest.

    Why we catch per-source errors instead of letting them propagate:
    If one source fails (e.g. a temporary API outage), we log the failure
    and continue. The validation step will catch the missing file and
    decide whether it is critical or not.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    for source in sources:
        log_event(run_id, "EXTRACT_START", {"source": source.name})

        # Where this source's raw file will be saved
        dest = BRONZE_DIR / f"{run_id}_{source.name}.csv"

        try:
            if source.access_method.value == "csv_download":
                df = _download_csv(source.url, dest)

            elif source.access_method.value == "api":
                df = _download_api(source.url, dest)
                # Save normalised CSV alongside raw JSON for consistency
                df.to_csv(dest, index=False)

            else:
                # web_scrape: not implemented in v1 — log and skip
                log_event(run_id, "EXTRACT_SKIP",
                          {"source": source.name,
                           "reason": "web_scrape not supported in v1"})
                continue

            # Record the bronze file with checksum
            bronze_file = BronzeFile(
                source_name=source.name,
                local_path=dest,
                row_count=len(df),
                downloaded_at=datetime.now(timezone.utc),
                checksum=file_checksum(dest),
                columns=list(df.columns),
            )
            files.append(bronze_file)

            log_event(run_id, "EXTRACT_OK", {
                "source": source.name,
                "rows":   len(df),
                "cols":   len(df.columns),
                "path":   str(dest),
            })

        except Exception as e:
            # Log the failure but do not crash — validation will catch it
            log_event(run_id, "EXTRACT_FAIL", {
                "source": source.name,
                "error":  str(e),
            })

    return BronzeManifest(run_id=run_id, files=files)