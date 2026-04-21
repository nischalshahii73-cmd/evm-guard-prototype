from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def global_permutation_importance(
    model_pipeline,
    X: pd.DataFrame,
    y,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None,
) -> pd.DataFrame:
    """
    Model-agnostic permutation importance.

    Returns a DataFrame with:
    feature, importance_mean, importance_std
    """
    from sklearn.inspection import permutation_importance

    # sklearn will call pipeline.predict / predict_proba as needed based on scoring
    r = permutation_importance(
        model_pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    out = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def _unwrap_pipeline(model_pipeline):
    """
    Returns (preprocess, model) from an sklearn Pipeline-like object.
    If not a Pipeline, preprocess=None, model=model_pipeline.
    """
    preprocess = None
    model = model_pipeline

    if hasattr(model_pipeline, "named_steps"):
        steps = model_pipeline.named_steps
        preprocess = steps.get("preprocess", None)

        # fallback: first step if preprocess key not present
        if preprocess is None and len(steps) >= 2:
            preprocess = list(steps.values())[0]

        model = steps.get("model", None)
        # fallback: last step
        if model is None:
            model = list(steps.values())[-1]

    return preprocess, model


def _transform_X(preprocess, X: pd.DataFrame):
    """
    Transform X and also return feature names after transformation.
    """
    if preprocess is None:
        X_trans = X.values
        feature_names = np.array(X.columns, dtype=object)
        return X_trans, feature_names

    X_trans = preprocess.transform(X)

    if hasattr(preprocess, "get_feature_names_out"):
        feature_names = preprocess.get_feature_names_out()
    else:
        feature_names = np.array(X.columns, dtype=object)

    feature_names = np.array(feature_names, dtype=object)

    # sparse -> dense (SHAP usually prefers dense)
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(X_trans):
            X_trans = X_trans.toarray()
    except Exception:
        pass

    return np.asarray(X_trans), feature_names


def try_shap_global_tree(
    model_pipeline,
    X: pd.DataFrame,
    task: str,
    class_name: Optional[str] = None,
    max_rows: int = 300,
) -> Optional[Dict[str, Any]]:
    """
    Compute global SHAP (summary plot data) for tree-based models inside a pipeline.

    Returns dict with:
    - shap_values
    - X_trans
    - feature_names
    - model_type
    - (classifier only) class_selected
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    preprocess, model = _unwrap_pipeline(model_pipeline)

    X_use = X.copy()
    if len(X_use) > max_rows:
        X_use = X_use.sample(max_rows, random_state=42)

    X_trans, feature_names = _transform_X(preprocess, X_use)

    # only tree explainer here (works with RF / GBM / XGB etc)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        return None

    if task == "regressor":
        shap_vals = explainer.shap_values(X_trans)
        # shap_vals typically (n_rows, n_features)
        return {
            "shap_values": shap_vals,
            "X_trans": X_trans,
            "feature_names": feature_names,
            "model_type": type(model).__name__,
        }

    if task == "classifier":
        if class_name is None:
            class_name = "High"

        # choose class index safely
        class_idx = 0
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if class_name in classes:
                class_idx = classes.index(class_name)

        shap_vals = explainer.shap_values(X_trans)

        # multiclass outputs:
        # - list of arrays [ (n_rows,n_feat), ... ]
        # - ndarray (n_rows,n_feat,n_classes)
        if isinstance(shap_vals, list):
            shap_selected = shap_vals[class_idx]
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 3:
                shap_selected = arr[:, :, class_idx]
            else:
                shap_selected = arr

        return {
            "shap_values": shap_selected,
            "X_trans": X_trans,
            "feature_names": feature_names,
            "model_type": type(model).__name__,
            "class_selected": class_name,
        }

    return None


def try_shap_local_row(
    model_pipeline,
    row_X: pd.DataFrame,
    task: str,
    class_name: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Local SHAP explanation for a single row.

    Returns DataFrame:
    feature, shap_contribution, abs_contribution, feature_value
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    preprocess, model = _unwrap_pipeline(model_pipeline)
    X_trans, feature_names = _transform_X(preprocess, row_X)

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        return None

    feature_names = np.array(feature_names, dtype=object)
    feat_vals = X_trans[0]

    if task == "regressor":
        shap_vals = explainer.shap_values(X_trans)
        arr = np.asarray(shap_vals)
        shap_vec = arr[0] if arr.ndim == 2 else arr

        out = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_contribution": shap_vec,
                "abs_contribution": np.abs(shap_vec),
                "feature_value": feat_vals,
            }
        ).sort_values("abs_contribution", ascending=False).reset_index(drop=True)
        return out

    if task == "classifier":
        if class_name is None:
            class_name = "High"

        class_idx = 0
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if class_name in classes:
                class_idx = classes.index(class_name)

        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            # list of (1,n_features)
            shap_vec = np.asarray(shap_vals[class_idx])[0]
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 3:
                shap_vec = arr[0, :, class_idx]
            elif arr.ndim == 2:
                shap_vec = arr[0]
            else:
                shap_vec = arr

        out = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_contribution": shap_vec,
                "abs_contribution": np.abs(shap_vec),
                "feature_value": feat_vals,
            }
        ).sort_values("abs_contribution", ascending=False).reset_index(drop=True)
        return out

    return None