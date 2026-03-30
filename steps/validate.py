# steps/validate.py
# WHY: Checks every bronze file before any transformation begins.
# Fail fast and loudly rather than letting bad data corrupt downstream results.
# Returns a ValidationReport — never raises. The Flow decides what to halt.

import pandas as pd

from config.settings import BRONZE_DIR, EXPECTED_COLUMNS
from state.models import BronzeManifest, ValidationReport, ValidationFailure, Severity
from guardrails.schema_checks import run_all_checks
from guardrails.provenance import log_event, file_checksum


def _check_checksum(bronze_file, run_id: str) -> list[ValidationFailure]:
    """
    Why: Recompute the checksum and compare it to what was recorded
    at download time. If they differ, the file was modified after
    ingestion — which should never happen in a bronze layer.
    This protects the entire audit trail.
    """
    current = file_checksum(bronze_file.local_path)
    if current == bronze_file.checksum:
        return []
    return [ValidationFailure(
        source=bronze_file.source_name,
        check="checksum_mismatch",
        severity=Severity.critical,
        detail=(f"Checksum changed after download. "
                f"Expected {bronze_file.checksum[:12]}... "
                f"got {current[:12]}...")
    )]


def bronze(manifest: BronzeManifest) -> ValidationReport:
    """
    Main entry point called by the Flow.
    Validates every file in the manifest and returns a ValidationReport.

    Why we validate all files before returning:
    We want the full picture of what is wrong, not just the first failure.
    The analyst can fix all issues at once rather than discovering them
    one by one.
    """
    all_failures = []

    for bronze_file in manifest.files:
        log_event(manifest.run_id, "VALIDATE_START",
                  {"source": bronze_file.source_name})

        # Load the raw file back from disk
        try:
            df = pd.read_csv(bronze_file.local_path)
        except Exception as e:
            all_failures.append(ValidationFailure(
                source=bronze_file.source_name,
                check="unreadable_file",
                severity=Severity.critical,
                detail=str(e)
            ))
            continue

        # What columns do we expect for this source?
        expected = EXPECTED_COLUMNS.get(bronze_file.source_name, [])

        # Numeric columns we expect for dtype checking
        numeric = [c for c in expected
                   if any(k in c for k in
                          ["rent", "income", "rate", "population", "fmr"])]

        # Run all schema checks
        failures = run_all_checks(df, bronze_file.source_name,
                                  expected, numeric_cols=numeric)

        # Run checksum integrity check
        failures += _check_checksum(bronze_file, manifest.run_id)

        all_failures.extend(failures)

        # Log per-source result
        source_critical = [f for f in failures if f.severity == Severity.critical]
        source_warnings  = [f for f in failures if f.severity == Severity.warning]

        log_event(manifest.run_id, "VALIDATE_RESULT", {
            "source":   bronze_file.source_name,
            "critical": len(source_critical),
            "warnings": len(source_warnings),
        })

    report = ValidationReport(run_id=manifest.run_id, failures=all_failures)

    log_event(manifest.run_id, "VALIDATE_COMPLETE", {
        "total_critical": sum(1 for f in all_failures
                              if f.severity == Severity.critical),
        "total_warnings": sum(1 for f in all_failures
                              if f.severity == Severity.warning),
        "has_critical_failures": report.has_critical_failures,
    })

    return report