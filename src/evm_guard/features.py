from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def add_evm_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EVM-derived metrics, progress proxy, and lag/trend features."""
    d = df.copy()

    d["BAC"] = d.groupby("ProjectID")["CumPlanned_USD"].transform("max")
    d["CV"] = d["EV"] - d["AC"]
    d["SV"] = d["EV"] - d["PV"]

    d["progress_plan_pct"] = np.where(d["BAC"] > 0, d["PV"] / d["BAC"], np.nan)
    d["progress_plan_pct"] = d["progress_plan_pct"].clip(0, 1)

    for col in ["CPI", "SPI", "CV", "SV", "EV", "PV", "AC"]:
        d[f"{col}_lag1"] = d.groupby("ProjectID")[col].shift(1)
        d[f"{col}_lag2"] = d.groupby("ProjectID")[col].shift(2)
        d[f"{col}_delta1"] = d[col] - d[f"{col}_lag1"]

    for col in ["CPI", "SPI"]:
        d[f"{col}_roll3_mean"] = (
            d.groupby("ProjectID")[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        d[f"{col}_roll3_std"] = (
            d.groupby("ProjectID")[col].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
        )

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def compute_eac_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute classic EAC formulations."""
    d = df.copy()
    bac = d["BAC"]
    cpi = d["CPI"].replace(0, np.nan)
    spi = d["SPI"].replace(0, np.nan)

    d["EAC_cpi"] = d["AC"] + (bac - d["EV"]) / cpi
    d["EAC_cpi_spi"] = d["AC"] + (bac - d["EV"]) / (cpi * spi)
    d["EAC_simple"] = bac / cpi
    return d


def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select ML feature columns (only those that exist)."""
    cols = [
        "CPI", "SPI", "CV", "SV",
        "EV", "PV", "AC",
        "progress_plan_pct",
        "CPI_delta1", "SPI_delta1",
        "CPI_lag1", "SPI_lag1",
        "CPI_lag2", "SPI_lag2",
        "CPI_roll3_mean", "SPI_roll3_mean",
        "CPI_roll3_std", "SPI_roll3_std",
    ]
    return [c for c in cols if c in df.columns]
