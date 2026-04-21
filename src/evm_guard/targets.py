from __future__ import annotations

import numpy as np
import pandas as pd


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create supervised targets from end-of-project outcomes.

    Targets:
    - y_cost_overrun_ratio: (Final CumActual - BAC) / BAC
    - y_schedule_slip_proxy: 1 - Final SPI (finish-date variance not provided)
    - y_risk_level: Low/Medium/High (quantile-based operational definition)

    Notes:
    - BAC is treated as the final cumulative planned cost (proxy baseline budget).
    - Schedule outcome is proxied using final SPI due to missing finish-date fields.
    - Risk levels are computed using quantiles to avoid single-class learning when
      datasets are small or highly skewed.
    """
    d = df.copy()

    # End-of-project outcomes (project-level, applied to all months for that project)
    final_actual = d.groupby("ProjectID")["CumActual_USD"].transform("max")
    bac = d.groupby("ProjectID")["BAC"].transform("max")
    final_spi = d.groupby("ProjectID")["SPI"].transform("last")

    # Magnitude targets
    d["y_cost_overrun_ratio"] = np.where(bac > 0, (final_actual - bac) / bac, np.nan)
    d["y_schedule_slip_proxy"] = 1.0 - final_spi

    # -------------------------
    # Quantile-based risk level
    # -------------------------
    # Combine cost and schedule proxy into one risk score (equal weighting)
    risk_score = (
        0.5 * d["y_cost_overrun_ratio"].fillna(0.0)
        + 0.5 * d["y_schedule_slip_proxy"].fillna(0.0)
    )

    # Quantile cut points (33% and 66%)
    q1 = risk_score.quantile(0.33)
    q2 = risk_score.quantile(0.66)

    d["y_risk_level"] = np.where(
        risk_score <= q1,
        "Low",
        np.where(risk_score <= q2, "Medium", "High"),
    )

    return d