# streamlit_app.py
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.evm_guard.loader import load_setA_workbooks
from src.evm_guard.features import (
    add_evm_derived_features,
    compute_eac_baselines,
    get_model_feature_columns,
)
from src.evm_guard.targets import add_targets
from src.evm_guard.model import train_models

# Explainability helpers (tree-focused SHAP + permutation importance)
from src.evm_guard.explain import (
    global_permutation_importance,
    try_shap_global_tree,
    try_shap_local_row,
)

# -------------------------
# Helper: create a clean, shareable report (Markdown)
# -------------------------
def build_markdown_report(df_filtered: pd.DataFrame, models, feature_cols: List[str], pv_range: tuple[float, float]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_projects = int(df_filtered["ProjectID"].nunique())
    n_rows = int(len(df_filtered))

    # small, readable distribution
    label_counts = df_filtered.get("y_risk_level", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()

    # model selection
    sel = models.metrics.get("selected_models", {})
    clf_name = sel.get("classifier", "N/A")
    reg_name = sel.get("regressor", "N/A")

    # include key numeric metrics for selected models (if CV ran)
    clf_block = models.metrics.get("classifier", {}).get(clf_name, {})
    reg_block = models.metrics.get("regressor", {}).get(reg_name, {})

    def _safe_get(d, path, default="N/A"):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    clf_macro_f1 = _safe_get(clf_block, ["classification_report", "macro avg", "f1-score"], default="N/A")
    clf_acc = _safe_get(clf_block, ["classification_report", "accuracy"], default="N/A")
    clf_auc = clf_block.get("auc_macro_ovr", None)

    rmse = reg_block.get("rmse_cost_overrun_ratio", "N/A")
    mae = reg_block.get("mae_cost_overrun_ratio", "N/A")

    # convert ratio errors to “percentage points”
    def _pp(x):
        try:
            return f"{float(x)*100:.2f} pp of BAC"
        except Exception:
            return "N/A"

    md = f"""# EVM-Guard Auto-Generated Report

**Generated:** {now}  
**PV/BAC (planned progress) filter:** {pv_range[0]:.2f} – {pv_range[1]:.2f}  
**Rows (project-months):** {n_rows}  
**Projects:** {n_projects}

## 1) Dataset summary
This report was generated from uploaded SetA project workbooks after schema validation, feature engineering (EVM indicators + lag/trend features), and target construction (risk level classification + cost overrun magnitude regression).

**Risk label distribution (current filter):**
{json.dumps(label_counts, indent=2)}

## 2) Model selection (3-model comparison)
The prototype trains three candidate models for each task and selects a default model for predictions:

- **Classifier candidates:** LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier  
- **Regressor candidates:** Ridge, RandomForestRegressor, HistGradientBoostingRegressor

**Selected classifier:** {clf_name}  
**Selected regressor:** {reg_name}

## 3) Performance summary (GroupKFold by ProjectID)
**Classifier (selected):**
- macro-F1: {clf_macro_f1}
- accuracy: {clf_acc}
- AUC (macro-OVR): {clf_auc}

**Regressor (selected):**
- RMSE (cost overrun ratio): {rmse}  (~{_pp(rmse)})
- MAE (cost overrun ratio): {mae}   (~{_pp(mae)})

> Interpretation tip: An RMSE of 0.015 on the cost-overrun ratio is roughly **1.5 percentage points of BAC**.  
> Example: if BAC ≈ 1,000,000 then 0.015 ≈ 15,000 budget units.

## 4) Key drivers (explainability)
Global permutation importance and SHAP are provided (when compatible) to identify which EVM indicators most influence:
- **risk classification** (Low/Medium/High), and
- **cost overrun magnitude** (continuous ratio).

## 5) Known limitations
- If a PV/BAC filter leaves too few projects (e.g., 1 project), **GroupKFold cannot run**. The app will mark CV as “skipped” and still train models on all rows to keep the UI working.
- AUC can appear as `null` when a fold lacks one or more classes (one-vs-rest AUC not computable).
- With very small samples, permutation importance for the classifier may show near-zero values because model performance is unstable.

"""
    return md


# -------------------------
# Helper: Offline AI assistant (no API key required)
# -------------------------
def offline_assistant_answer(report_md: str, user_q: str) -> str:
    q = user_q.lower().strip()

    if "data" in q and ("need" in q or "required" in q):
        return (
            "For this prototype, each uploaded Excel workbook must contain the 'Budget_Costs' sheet with "
            "monthly EV, PV, AC, CPI, and SPI (and ProjectID + Month). The app then derives CV, SV, "
            "CPI/SPI lags and rolling trends, and computes targets for risk level and cost overrun ratio."
        )
    if "model" in q and ("which" in q or "used" in q):
        return (
            "The prototype trains 3 models for classification and 3 for regression. "
            "It automatically selects the best model (by macro-F1 for classification and lowest RMSE for regression) "
            "using GroupKFold split by ProjectID."
        )
    if "pv/bac" in q or "stage" in q:
        return (
            "PV/BAC is used as a proxy for planned progress (planned value as a fraction of the baseline budget). "
            "Filtering PV/BAC creates early/mid/late stage windows to test how early the warning can be made."
        )
    if "shap" in q:
        return (
            "SHAP explains predictions by showing how each feature pushes the model output up or down. "
            "Dots to the right increase the predicted risk/overrun; dots to the left decrease it."
        )
    if "report" in q:
        return (
            "The Report tab generates a structured summary including dataset size, PV/BAC window, model selection, "
            "GroupKFold metrics, and key explainability notes. You can download it as a Markdown file."
        )

    # fallback
    return (
        "Tell me exactly what you want to know (e.g., 'Explain Global SHAP classifier' or 'Why AUC is null'), "
        "and I will answer using the generated report and your current PV/BAC filter."
    )


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="EVM-Guard", layout="wide")
st.title("EVM-Guard: Explainable EVM + ML Early-Warning Prototype")
st.caption("Upload SetA_Project*.xlsx files (must include 'Budget_Costs' with EV/PV/AC/CPI/SPI by Month).")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
progress_filter = st.sidebar.slider(
    "Filter by planned progress (PV/BAC)",
    0.0, 1.0, (0.0, 1.0), 0.05
)
train_now = st.sidebar.button("Train / Retrain Models")

if st.sidebar.button("Clear trained models (force re-train)"):
    for k in ["models", "report_md"]:
        if k in st.session_state:
            del st.session_state[k]
    st.sidebar.success("Cleared. Now click Train / Retrain Models.")

# -------------------------
# File upload
# -------------------------
uploaded = st.file_uploader("Upload SetA Excel workbooks", type=["xlsx"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload your SetA_Project*.xlsx files to begin.")
    st.stop()

files: List[Tuple[str, bytes]] = [(f.name, f.read()) for f in uploaded]

# -------------------------
# Load + validate
# -------------------------
try:
    load_res = load_setA_workbooks(files)
    df = load_res.monthly
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

if load_res.warnings:
    with st.expander("Load warnings"):
        for w in load_res.warnings:
            st.warning(w)

# -------------------------
# Feature engineering + baselines + targets
# -------------------------
df = add_evm_derived_features(df)
df = compute_eac_baselines(df)
df = add_targets(df)

# Apply stage filter
df_filtered = df[
    (df["progress_plan_pct"] >= progress_filter[0]) &
    (df["progress_plan_pct"] <= progress_filter[1])
].copy()

# -------------------------
# Tabs (NEW: Report + AI Assistant)
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Upload/Input", "Predictions", "Explanations", "Charts", "Export", "Report", "AI Assistant"]
)

# -------------------------
# Tab 1: Upload/Input
# -------------------------
with tab1:
    st.subheader("Loaded data (project-month level)")
    st.write(f"Rows: {len(df_filtered)} | Projects: {df_filtered['ProjectID'].nunique()}")
    st.dataframe(df_filtered, use_container_width=True)

# -------------------------
# Tab 2: Predictions
# -------------------------
with tab2:
    st.subheader("Predictions (trained on uploaded projects)")

    feature_cols = get_model_feature_columns(df_filtered)
    if not feature_cols:
        st.error("No feature columns available. Check uploaded files or widen PV/BAC filter.")
        st.stop()

    st.caption(
        f"Current filter PV/BAC: {progress_filter[0]:.2f}–{progress_filter[1]:.2f} | "
        f"Rows: {len(df_filtered)} | Projects: {df_filtered['ProjectID'].nunique()}"
    )

    train_df = df_filtered.dropna(subset=feature_cols + ["y_risk_level", "y_cost_overrun_ratio"]).copy()
    if train_df.empty:
        st.error("No training rows available after filtering. Widen PV/BAC range or upload more projects.")
        st.stop()

    st.markdown("### Target distribution (y_risk_level)")
    st.write(train_df["y_risk_level"].value_counts(dropna=False))

    # Train (3-model comparison handled inside train_models)
    if train_now or ("models" not in st.session_state):
        with st.spinner("Training 3 candidate models (GroupKFold by ProjectID where possible)..."):
            st.session_state["models"] = train_models(train_df, feature_cols)

    models = st.session_state["models"]

    st.markdown("### Cross-validated metrics (GroupKFold by ProjectID)")
    st.json(models.metrics)

    # Optional selector: which model to use for predictions (default is selected best)
    clf_choices = list(models.clf_candidates.keys())
    reg_choices = list(models.reg_candidates.keys())
    default_clf = models.metrics.get("selected_models", {}).get("classifier", clf_choices[0])
    default_reg = models.metrics.get("selected_models", {}).get("regressor", reg_choices[0])

    c1, c2 = st.columns(2)
    with c1:
        chosen_clf_name = st.selectbox("Classifier used for predictions", clf_choices, index=clf_choices.index(default_clf) if default_clf in clf_choices else 0)
    with c2:
        chosen_reg_name = st.selectbox("Regressor used for predictions", reg_choices, index=reg_choices.index(default_reg) if default_reg in reg_choices else 0)

    clf_model = models.clf_candidates[chosen_clf_name]
    reg_model = models.reg_candidates[chosen_reg_name]

    X_all = df_filtered[feature_cols].copy()
    pred_risk = clf_model.predict(X_all)
    pred_overrun = reg_model.predict(X_all)

    pred_df = df_filtered[["ProjectID", "Month", "BAC", "EV", "PV", "AC", "CPI", "SPI", "progress_plan_pct"]].copy()
    pred_df["pred_risk_level"] = pred_risk
    pred_df["pred_cost_overrun_ratio"] = pred_overrun

    st.markdown("### Predicted outputs (project-month level)")
    st.dataframe(pred_df, use_container_width=True)

# -------------------------
# Tab 3: Explanations
# -------------------------
with tab3:
    st.subheader("Explainability")

    models = st.session_state.get("models")
    if models is None:
        st.info("Train the models in the Predictions tab first.")
        st.stop()

    feature_cols = get_model_feature_columns(df_filtered)
    df_mod = df_filtered.dropna(subset=feature_cols + ["y_risk_level", "y_cost_overrun_ratio"]).copy()
    if df_mod.empty:
        st.warning("No rows available for explainability after filtering. Widen PV/BAC range or upload more projects.")
        st.stop()

    # use selected best models for explanations
    sel = models.metrics.get("selected_models", {})
    best_clf_name = sel.get("classifier", list(models.clf_candidates.keys())[0])
    best_reg_name = sel.get("regressor", list(models.reg_candidates.keys())[0])
    clf_pipe = models.clf_candidates[best_clf_name]
    reg_pipe = models.reg_candidates[best_reg_name]

    st.caption(f"Using best models for explainability: classifier={best_clf_name}, regressor={best_reg_name}")

    # Global permutation importance (works for any sklearn model)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Global permutation importance — classifier**")
        imp_clf = global_permutation_importance(clf_pipe, df_mod[feature_cols], df_mod["y_risk_level"])
        st.dataframe(imp_clf, use_container_width=True, height=420)

    with c2:
        st.markdown("**Global permutation importance — regressor**")
        imp_reg = global_permutation_importance(reg_pipe, df_mod[feature_cols], df_mod["y_cost_overrun_ratio"])
        st.dataframe(imp_reg, use_container_width=True, height=420)

    st.divider()
    st.markdown("## SHAP explanations (global + local)")
    st.caption("SHAP is shown only when the model is compatible (tree-based models).")

    # SHAP availability
    try:
        import shap  # type: ignore
        shap_ok = True
    except Exception:
        shap_ok = False

    if not shap_ok:
        st.info("SHAP not available in this environment. (Install with: pip install shap)")
        st.stop()

    # Global SHAP — classifier (tree only)
    st.markdown("### Global SHAP — Classifier (risk level)")
    class_options = ["Low", "Medium", "High"]
    class_selected = st.selectbox("Select class to explain (multiclass)", class_options, index=2)

    shap_clf = try_shap_global_tree(
        model_pipeline=clf_pipe,
        X=df_mod[feature_cols],
        task="classifier",
        class_name=class_selected,
        max_rows=300,
    )

    if shap_clf is None:
        st.warning("Classifier SHAP could not be computed for the selected model (not tree-based or incompatible).")
    else:
        st.write(f"SHAP enabled for: {shap_clf['model_type']} | Class explained: {shap_clf.get('class_selected')}")
        feature_names_np = np.array(shap_clf["feature_names"])
        fig = plt.figure()
        shap.summary_plot(shap_clf["shap_values"], shap_clf["X_trans"], feature_names=feature_names_np, show=False)
        st.pyplot(fig, clear_figure=True)

    # Global SHAP — regressor (tree only)
    st.markdown("### Global SHAP — Regressor (cost overrun magnitude)")
    shap_reg = try_shap_global_tree(
        model_pipeline=reg_pipe,
        X=df_mod[feature_cols],
        task="regressor",
        max_rows=300,
    )

    if shap_reg is None:
        st.warning("Regressor SHAP could not be computed for the selected model (not tree-based or incompatible).")
    else:
        st.write(f"SHAP enabled for: {shap_reg['model_type']}")
        feature_names_np = np.array(shap_reg["feature_names"])
        fig = plt.figure()
        shap.summary_plot(shap_reg["shap_values"], shap_reg["X_trans"], feature_names=feature_names_np, show=False)
        st.pyplot(fig, clear_figure=True)

    # Local SHAP
    st.markdown("### Local explanation — select a project-month")
    pid = st.selectbox("ProjectID for local explanation", sorted(df_mod["ProjectID"].unique().tolist()))
    pdf = df_mod[df_mod["ProjectID"] == pid].sort_values("Month").reset_index(drop=True)

    if pdf.empty:
        st.warning("This project has no rows under the current PV/BAC filter. Widen PV/BAC or choose another project.")
        st.stop()

    if len(pdf) == 1:
        st.info("Only one month is available for this project under the current PV/BAC filter.")
        month_idx = 0
    else:
        month_idx = st.slider("Month index (0 = earliest)", 0, len(pdf) - 1, 0)

    row = pdf.iloc[[month_idx]].copy()

    show_cols = [c for c in ["ProjectID", "Month", "EV", "PV", "AC", "CPI", "SPI", "CV", "SV", "progress_plan_pct"] if c in row.columns]
    st.dataframe(row[show_cols], use_container_width=True)

    st.markdown("**Local SHAP — Classifier**")
    try:
        local_clf = try_shap_local_row(clf_pipe, row[feature_cols], task="classifier", class_name=class_selected)
    except Exception as e:
        local_clf = None
        st.warning(f"Local classifier SHAP could not be computed (shape mismatch). Details: {e}")

    if local_clf is None:
        st.info("Local classifier SHAP not available for this configuration.")
    else:
        st.dataframe(local_clf.head(15), use_container_width=True)

    st.markdown("**Local SHAP — Regressor**")
    try:
        local_reg = try_shap_local_row(reg_pipe, row[feature_cols], task="regressor")
    except Exception as e:
        local_reg = None
        st.warning(f"Local regressor SHAP could not be computed (shape mismatch). Details: {e}")

    if local_reg is None:
        st.info("Local regressor SHAP not available for this configuration.")
    else:
        st.dataframe(local_reg.head(15), use_container_width=True)

# -------------------------
# Tab 4: Charts
# -------------------------
with tab4:
    st.subheader("Charts")

    if df_filtered.empty:
        st.warning("No rows available under the current PV/BAC filter.")
        st.stop()

    pid_chart = st.selectbox("Select project", sorted(df_filtered["ProjectID"].unique().tolist()))
    pdfc = df_filtered[df_filtered["ProjectID"] == pid_chart].sort_values("Month")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CPI and SPI over time**")
        fig = plt.figure()
        plt.plot(pdfc["Month"], pdfc["CPI"], label="CPI")
        plt.plot(pdfc["Month"], pdfc["SPI"], label="SPI")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.markdown("**Cumulative planned vs actual cost**")
        fig = plt.figure()
        plt.plot(pdfc["Month"], pdfc["CumPlanned_USD"], label="CumPlanned (BAC proxy)")
        plt.plot(pdfc["Month"], pdfc["CumActual_USD"], label="CumActual")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("**Baseline EAC forecasts**")
    fig = plt.figure()
    plt.plot(pdfc["Month"], pdfc["EAC_cpi"], label="EAC (CPI)")
    plt.plot(pdfc["Month"], pdfc["EAC_cpi_spi"], label="EAC (CPI*SPI)")
    plt.plot(pdfc["Month"], pdfc["BAC"], label="BAC")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -------------------------
# Tab 5: Export
# -------------------------
with tab5:
    st.subheader("Export results")

    models = st.session_state.get("models")
    if models is None:
        st.info("Train the models in the Predictions tab first.")
        st.stop()

    feature_cols = get_model_feature_columns(df_filtered)
    if not feature_cols:
        st.error("No feature columns available. Check your uploaded files and PV/BAC filter.")
        st.stop()

    X = df_filtered[feature_cols].copy()
    df_out = df_filtered.copy()

    sel = models.metrics.get("selected_models", {})
    clf_name = sel.get("classifier", list(models.clf_candidates.keys())[0])
    reg_name = sel.get("regressor", list(models.reg_candidates.keys())[0])
    df_out["pred_risk_level"] = models.clf_candidates[clf_name].predict(X)
    df_out["pred_cost_overrun_ratio"] = models.reg_candidates[reg_name].predict(X)

    st.download_button(
        "Download predictions CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="evm_guard_predictions.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download metrics JSON",
        data=json.dumps(models.metrics, indent=2).encode("utf-8"),
        file_name="evm_guard_metrics.json",
        mime="application/json",
    )

# -------------------------
# Tab 6: Report (NEW)
# -------------------------
with tab6:
    st.subheader("Auto-generated report (insights + findings)")

    models = st.session_state.get("models")
    if models is None:
        st.info("Train the models in the Predictions tab first.")
        st.stop()

    feature_cols = get_model_feature_columns(df_filtered)
    if not feature_cols:
        st.error("No feature columns available. Widen PV/BAC range or upload more data.")
        st.stop()

    if st.button("Generate / Refresh report"):
        st.session_state["report_md"] = build_markdown_report(df_filtered, models, feature_cols, progress_filter)

    report_md = st.session_state.get("report_md")
    if not report_md:
        st.info("Click **Generate / Refresh report** to create the report.")
    else:
        st.markdown(report_md)
        st.download_button(
            "Download report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name="evm_guard_report.md",
            mime="text/markdown",
        )

# -------------------------
# Tab 7: AI Assistant (NEW, offline by default)
# -------------------------
with tab7:
    st.subheader("AI Assistant (uses the generated report)")

    report_md = st.session_state.get("report_md")
    if not report_md:
        st.info("Generate the report first in the **Report** tab. The assistant uses that report as context.")
        st.stop()

    st.caption("No API key is required — this prototype includes an offline assistant for common questions.")

    user_q = st.text_input("Ask a question about this project/data/report:", "")
    if user_q.strip():
        st.write(offline_assistant_answer(report_md, user_q))