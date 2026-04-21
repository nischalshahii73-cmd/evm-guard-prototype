from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


REQUIRED_SHEET = "Budget_Costs"
REQUIRED_COLUMNS = [
    "Month",
    "CostCategory",
    "PlannedCost_USD",
    "ActualCost_USD",
    "ForecastCost_USD",
    "CumPlanned_USD",
    "CumActual_USD",
    "CumForecast_USD",
    "EV",
    "PV",
    "AC",
    "CPI",
    "SPI",
]


@dataclass
class LoadResult:
    monthly: pd.DataFrame
    warnings: List[str]


def _validate_budget_costs(df: pd.DataFrame, source_name: str) -> List[str]:
    warnings: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: Budget_Costs is missing required columns: {missing}")

    if df["Month"].isna().any():
        warnings.append(f"{source_name}: Month has missing values; those rows will be dropped.")

    for col in ["PlannedCost_USD", "ActualCost_USD", "EV", "PV", "AC", "CPI", "SPI"]:
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            warnings.append(f"{source_name}: Column {col} has non-numeric values; coerced to NaN.")
    return warnings


def load_setA_workbooks(files: List[Tuple[str, bytes]]) -> LoadResult:
    """Load SetA_Project*.xlsx workbooks and return a tidy (ProjectID, Month) table."""
    all_rows: List[pd.DataFrame] = []
    warnings: List[str] = []

    for fname, content in files:
        xl = pd.ExcelFile(content)
        if REQUIRED_SHEET not in xl.sheet_names:
            raise ValueError(f"{fname}: Missing required sheet '{REQUIRED_SHEET}'.")

        overview = pd.read_excel(content, sheet_name="Project_Overview")
        if "ProjectID" not in overview.columns or overview.empty:
            raise ValueError(f"{fname}: Project_Overview must contain ProjectID.")
        project_id = str(overview.loc[0, "ProjectID"])

        budget = pd.read_excel(content, sheet_name=REQUIRED_SHEET)
        warnings.extend(_validate_budget_costs(budget, fname))

        budget = budget.copy()
        budget["Month"] = pd.to_datetime(budget["Month"], errors="coerce")
        budget = budget.dropna(subset=["Month"])

        # Aggregate category rows to month level: cost columns summed; EVM columns first (they should match within month)
        cost_sum_cols = ["PlannedCost_USD", "ActualCost_USD", "ForecastCost_USD"]
        agg = {c: "sum" for c in cost_sum_cols}
        first_cols = ["CumPlanned_USD", "CumActual_USD", "CumForecast_USD", "EV", "PV", "AC", "CPI", "SPI"]
        for c in first_cols:
            agg[c] = "first"

        monthly = (
            budget.sort_values(["Month", "CostCategory"])
            .groupby("Month", as_index=False)
            .agg(agg)
        )
        monthly.insert(0, "ProjectID", project_id)
        all_rows.append(monthly)

    out = pd.concat(all_rows, ignore_index=True).sort_values(["ProjectID", "Month"]).reset_index(drop=True)
    return LoadResult(monthly=out, warnings=warnings)
