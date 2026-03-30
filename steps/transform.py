# steps/transform.py
# WHY: Two-stage transformation.
# Bronze → Silver: clean and type each source independently.
# Silver → Gold:   join everything, engineer features, create target.
# Keeping stages separate means failures are easier to isolate and fix.

import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import SILVER_DIR, GOLD_DIR, TARGET_COLUMN
from guardrails.provenance import log_event


# ── Bronze → Silver ───────────────────────────────────────────────────

def _clean_zillow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why: Zillow CSVs have inconsistent column names and mixed types.
    We standardise to lowercase snake_case, cast rent to float,
    and drop rows where the rent value is missing — they are useless
    for training a rent prediction model.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={"median_rent": "rent", "regionname": "city"})
    df["rent"] = pd.to_numeric(df["rent"], errors="coerce")
    df = df.dropna(subset=["rent"])
    df = df.drop_duplicates(subset=["city", "state"])
    return df[["city", "state", "rent"]].copy()


def _clean_census(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why: Census data uses FIPS codes and has suppressed values marked
    as '-' or null strings. We convert those to NaN so imputation
    can handle them properly later.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in ["median_income", "unemployment_rate",
                "population", "education_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["geo_id"])
    return df.copy()


def _clean_hud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why: HUD Fair Market Rents are the government's benchmark for what
    rent should be in each area. The gap between actual rent and FMR
    is a signal for whether an area is over or undervalued.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in ["fmr_0br", "fmr_1br", "fmr_2br"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["metro_area"])
    return df.copy()


# Map source names to their cleaning functions
_CLEANERS = {
    "zillow": _clean_zillow,
    "census": _clean_census,
    "hud":    _clean_hud,
}


def to_silver(manifest, run_id: str) -> Path:
    """
    Clean each bronze file and save it as a silver CSV.
    Why one file per source: keeps cleaning logic isolated.
    A failure in census cleaning does not affect zillow.
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    for bronze_file in manifest.files:
        name = bronze_file.source_name
        cleaner = _CLEANERS.get(name)

        if cleaner is None:
            log_event(run_id, "SILVER_SKIP",
                      {"source": name, "reason": "no cleaner defined"})
            continue

        df_raw = pd.read_csv(bronze_file.local_path)
        df_clean = cleaner(df_raw)

        dest = SILVER_DIR / f"{run_id}_{name}.csv"
        df_clean.to_csv(dest, index=False)

        log_event(run_id, "SILVER_OK", {
            "source": name,
            "rows":   len(df_clean),
            "cols":   list(df_clean.columns),
            "path":   str(dest),
        })

    return SILVER_DIR


# ── Silver → Gold ─────────────────────────────────────────────────────

def to_gold(silver_path: Path, run_id: str) -> Path:
    """
    Join all silver tables into one ML-ready gold table.
    Engineer features and create the target column.

    Why log(rent) as target:
    Rent is right-skewed — a few very expensive cities pull the
    distribution. Log-transforming compresses the tail, makes
    the distribution more normal, and means our RMSE is interpretable
    as approximate percentage error rather than raw dollar error.

    Why these features:
    - rent_to_income:  affordability signal, strong rent driver
    - fmr_gap:         distance from government benchmark = over/undervalued
    - log_population:  controls for city size effects
    - city_tier:       captures metro vs mid-size vs small city effects
    """
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Load silver tables
    zillow = _load_silver(silver_path, run_id, "zillow")
    census = _load_silver(silver_path, run_id, "census")
    hud    = _load_silver(silver_path, run_id, "hud")

    # Start with zillow as the base — it has rent, city, state
    df = zillow.copy()

    # Join census on city/state if available
    if census is not None and "city" in census.columns:
        df = df.merge(census, on=["city", "state"], how="left")

    # Join HUD on a fuzzy metro match — metro_area often contains city name
    if hud is not None:
        df = _join_hud(df, hud)

    # ── Feature engineering ──────────────────────────────────────────
    # Why here and not in preprocess: these are domain features, not
    # statistical transformations. They encode business knowledge.

    if "median_income" in df.columns:
        df["rent_to_income"] = df["rent"] / df["median_income"].replace(0, np.nan)

    if "fmr_1br" in df.columns:
        df["fmr_gap"] = df["rent"] - df["fmr_1br"]

    if "population" in df.columns:
        df["log_population"] = np.log1p(df["population"])

    # City tier: simple bucketing by population
    # Why: City size has a strong non-linear effect on rent.
    # A large city dummy captures what a linear population term misses.
    if "population" in df.columns:
        df["city_tier"] = pd.cut(
            df["population"],
            bins=[0, 100_000, 500_000, float("inf")],
            labels=["small", "mid", "large"]
        )

    # ── Target column ────────────────────────────────────────────────
    df[TARGET_COLUMN] = np.log(df["rent"])

    # Drop rows where the target is missing or infinite
    df = df[np.isfinite(df[TARGET_COLUMN])].copy()

    dest = GOLD_DIR / f"{run_id}_gold.parquet"
    df.to_parquet(dest, index=False)

    log_event(run_id, "GOLD_OK", {
        "rows":     len(df),
        "cols":     list(df.columns),
        "path":     str(dest),
        "target":   TARGET_COLUMN,
    })

    return dest


def _load_silver(silver_path: Path, run_id: str,
                 source: str) -> pd.DataFrame | None:
    """Load one silver file. Return None if it does not exist."""
    path = silver_path / f"{run_id}_{source}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _join_hud(df: pd.DataFrame, hud: pd.DataFrame) -> pd.DataFrame:
    """
    Why fuzzy join: HUD uses metro area names, not city names.
    'New York-Newark' matches cities 'New York' and 'Newark'.
    We check if the city name appears anywhere in the metro_area string.
    This is imprecise but far better than losing all HUD data.
    """
    hud_cols = ["fmr_0br", "fmr_1br", "fmr_2br"]
    available = [c for c in hud_cols if c in hud.columns]
    if not available:
        return df

    def find_fmr(city: str) -> dict:
        match = hud[hud["metro_area"].str.contains(city, case=False, na=False)]
        if match.empty:
            return {c: np.nan for c in available}
        return match.iloc[0][available].to_dict()

    fmr_df = pd.DataFrame(df["city"].apply(find_fmr).tolist())
    return pd.concat([df.reset_index(drop=True),
                      fmr_df.reset_index(drop=True)], axis=1)