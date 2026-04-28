from __future__ import annotations

from typing import Dict
import pandas as pd


def build_markdown_report(
    df_filtered: pd.DataFrame,
    stage_range: tuple[float, float],
    comparison_rows: list[Dict[str, str]],
    selected_classifier: str,
    selected_regressor: str,
) -> str:
    """
    Create a simple reproducible report in Markdown (downloadable).
    """
    a, b = stage_range
    n_projects = int(df_filtered["ProjectID"].nunique()) if "ProjectID" in df_filtered.columns else 0
    n_rows = int(len(df_filtered))

    # Target distribution if present
    target_dist_md = ""
    if "y_risk_level" in df_filtered.columns:
        counts = df_filtered["y_risk_level"].value_counts(dropna=False)
        target_dist_md = "\n".join([f"- {k}: {int(v)}" for k, v in counts.items()])
    else:
        target_dist_md = "- (Target not available in the filtered dataset)"

    # Comparison table
    table_header = "| Model | Classifier (Macro-F1 / AUC) | Regressor (MAE / RMSE) | Notes |\n|---|---|---|---|\n"
    table_rows = ""
    for r in comparison_rows:
        table_rows += f"| {r['model']} | {r['clf']} | {r['reg']} | {r['note']} |\n"

    md = f"""
# EVM-Guard Report (Stage Window PV/BAC {a:.2f}–{b:.2f})

## Dataset summary
- Projects: **{n_projects}**
- Project-month rows: **{n_rows}**

## Risk target distribution (y_risk_level)
{target_dist_md}

## Model comparison (GroupKFold by ProjectID)
{table_header}{table_rows}

## Selected operational models (for dashboard predictions)
- Classifier: **{selected_classifier}**
- Regressor: **{selected_regressor}**

## Notes and limitations
- If the stage filter retains fewer than **2 projects**, grouped cross-validation cannot be computed. In that case, models may still be fitted for pipeline demonstration, but evaluation metrics are treated as undefined.
- AUC may appear as **NULL/None** in small grouped folds when one-vs-rest AUC cannot be computed due to missing classes in at least one fold.
"""
    return md.strip()