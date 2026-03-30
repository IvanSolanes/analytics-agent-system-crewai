# guardrails/schema_checks.py
# WHY: Reusable validation functions that check raw data quality.
# Called by the validate step. Return structured failures, never raise.
# The flow decides what to do with failures — not this file.

import pandas as pd
from state.models import ValidationFailure, Severity
from config.settings import MIN_ROWS_BRONZE, MAX_NULL_RATE


def check_columns(df: pd.DataFrame, source: str,
                  expected: list[str]) -> list[ValidationFailure]:
    """Check that all expected columns are present."""
    missing = set(expected) - set(df.columns)
    if not missing:
        return []
    return [ValidationFailure(
        source=source,
        check="missing_columns",
        severity=Severity.critical,
        detail=f"Missing columns: {sorted(missing)}"
    )]


def check_row_count(df: pd.DataFrame, source: str) -> list[ValidationFailure]:
    """Check that the file has enough rows to be meaningful."""
    if len(df) >= MIN_ROWS_BRONZE:
        return []
    return [ValidationFailure(
        source=source,
        check="row_count",
        severity=Severity.critical,
        detail=f"Only {len(df)} rows — likely a failed or empty download"
    )]


def check_null_rates(df: pd.DataFrame, source: str) -> list[ValidationFailure]:
    """Flag any column whose null rate exceeds the threshold."""
    failures = []
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > MAX_NULL_RATE:
            failures.append(ValidationFailure(
                source=source,
                check="null_rate",
                severity=Severity.warning,
                detail=f"Column '{col}' is {rate:.0%} null"
            ))
    return failures


def check_dtypes(df: pd.DataFrame, source: str,
                 numeric_cols: list[str]) -> list[ValidationFailure]:
    """
    Check that columns expected to be numeric actually are.
    Why: A rent column full of strings like '$1,200' looks valid
    on a column-presence check but breaks every arithmetic operation.
    """
    failures = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            failures.append(ValidationFailure(
                source=source,
                check="dtype",
                severity=Severity.warning,
                detail=f"Column '{col}' expected numeric, got {df[col].dtype}"
            ))
    return failures

def run_all_checks(df: pd.DataFrame, source: str,
                   expected_cols: list[str],
                   numeric_cols: list[str] = None) -> list[ValidationFailure]:
    return (
        check_columns(df, source, expected_cols) +
        check_row_count(df, source) +
        check_null_rates(df, source) +
        check_dtypes(df, source, numeric_cols or [])
    )