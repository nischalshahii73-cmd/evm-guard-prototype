# src/evm_guard/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass
class CandidateModel:
    """One trained candidate model (pipeline)."""
    name: str
    pipeline: Pipeline


@dataclass
class TrainedModels:
    """Returned by train_models()."""
    # Best models (used for prediction/explainability by default)
    clf: Pipeline
    reg: Pipeline

    # All candidates (for comparison + optional selection in UI)
    clf_candidates: Dict[str, Pipeline]
    reg_candidates: Dict[str, Pipeline]

    # Evaluation metrics (per-candidate + selection summary)
    metrics: Dict


def _make_preprocess(feature_cols: List[str], scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # All features are numeric in this prototype
    return ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _safe_groupkfold_splits(groups: pd.Series, desired_splits: int = 5) -> int:
    n_groups = int(pd.Series(groups).nunique())
    # GroupKFold requires n_splits <= n_groups, and at least 2 splits
    return max(0, min(desired_splits, n_groups))


def _eval_classifier_cv(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
) -> Dict:
    """GroupKFold evaluation for classifier. Skips CV safely when not possible."""
    n_splits = _safe_groupkfold_splits(groups, desired_splits=5)

    if n_splits < 2:
        # Not enough projects under the current PV/BAC filter
        # Train-only metrics are not honest, so return a clear flag.
        return {
            "cv_skipped": True,
            "reason": f"Not enough groups/projects for GroupKFold (n_projects={int(pd.Series(groups).nunique())}).",
        }

    cv = GroupKFold(n_splits=n_splits)
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    y_proba_all: List[np.ndarray] = []
    fold_class_lists: List[List[str]] = []

    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        y_true_all.extend(list(y_te.astype(str)))
        y_pred_all.extend(list(pd.Series(y_pred).astype(str)))

        # probability for AUC (if available)
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)
                y_proba_all.append(np.asarray(proba))
                # classes for this fitted fold
                if hasattr(pipe[-1], "classes_"):
                    fold_class_lists.append(list(pipe[-1].classes_))
            except Exception:
                pass

    report = classification_report(
        y_true_all, y_pred_all, output_dict=True, zero_division=0
    )

    # Macro AUC (one-vs-rest) can fail if a class is missing in y_true in some folds.
    auc_macro_ovr: Optional[float] = None
    try:
        if y_proba_all:
            # For AUC we need probabilities aligned to a single class order.
            # Simplest safe approach: compute AUC on the concatenated fold outputs ONLY
            # when class set is stable across folds.
            if fold_class_lists and all(fold_class_lists[0] == c for c in fold_class_lists):
                classes = fold_class_lists[0]
                proba_concat = np.vstack(y_proba_all)
                y_true = pd.Series(y_true_all)
                # binarise
                y_bin = np.vstack([(y_true == c).astype(int).values for c in classes]).T
                # roc_auc_score expects shape (n_samples, n_classes)
                auc_macro_ovr = float(
                    roc_auc_score(y_bin, proba_concat, average="macro", multi_class="ovr")
                )
    except Exception:
        auc_macro_ovr = None

    return {
        "cv_skipped": False,
        "classification_report": report,
        "auc_macro_ovr": auc_macro_ovr,
        "cv_strategy": f"GroupKFold(n_splits={n_splits}) by ProjectID",
        "random_state": random_state,
    }


def _eval_regressor_cv(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
) -> Dict:
    """GroupKFold evaluation for regressor. Skips CV safely when not possible."""
    n_splits = _safe_groupkfold_splits(groups, desired_splits=5)

    if n_splits < 2:
        return {
            "cv_skipped": True,
            "reason": f"Not enough groups/projects for GroupKFold (n_projects={int(pd.Series(groups).nunique())}).",
        }

    cv = GroupKFold(n_splits=n_splits)
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)

        y_true_all.extend(list(np.asarray(y_te, dtype=float)))
        y_pred_all.extend(list(np.asarray(y_hat, dtype=float)))

    y_true = np.asarray(y_true_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "cv_skipped": False,
        "rmse_cost_overrun_ratio": rmse,
        "mae_cost_overrun_ratio": mae,
        "cv_strategy": f"GroupKFold(n_splits={n_splits}) by ProjectID",
        "random_state": random_state,
    }


def train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> TrainedModels:
    """
    Train 3 candidate models for:
    - Classification: y_risk_level
    - Regression: y_cost_overrun_ratio

    Returns:
      - best clf/reg pipelines (used by the app for predictions)
      - candidate dictionaries for comparison
      - metrics per candidate + selected model names
    """
    # -----------------------
    # Prepare train matrices
    # -----------------------
    X = df[feature_cols].copy()
    y_clf = df["y_risk_level"].astype(str).copy()
    y_reg = df["y_cost_overrun_ratio"].astype(float).copy()
    groups = df["ProjectID"].astype(str).copy()

    # -----------------------
    # Candidate models
    # -----------------------
    # Linear models benefit from scaling; tree models do not require it.
    pre_lin = _make_preprocess(feature_cols, scale=True)
    pre_tree = _make_preprocess(feature_cols, scale=False)

    clf_candidates: List[CandidateModel] = [
        CandidateModel(
            "LogisticRegression",
            Pipeline(
                steps=[
                    ("preprocess", pre_lin),
                    ("model", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    )),
                ]
            ),
        ),
        CandidateModel(
            "RandomForestClassifier",
            Pipeline(
                steps=[
                    ("preprocess", pre_tree),
                    ("model", RandomForestClassifier(
                        n_estimators=400,
                        random_state=random_state,
                        class_weight="balanced",
                    )),
                ]
            ),
        ),
        CandidateModel(
            "HistGradientBoostingClassifier",
            Pipeline(
                steps=[
                    ("preprocess", pre_tree),
                    ("model", HistGradientBoostingClassifier(
                        random_state=random_state,
                        max_depth=6,
                        learning_rate=0.08,
                    )),
                ]
            ),
        ),
    ]

    reg_candidates: List[CandidateModel] = [
        CandidateModel(
            "Ridge",
            Pipeline(
                steps=[
                    ("preprocess", pre_lin),
                    ("model", Ridge(random_state=random_state)),
                ]
            ),
        ),
        CandidateModel(
            "RandomForestRegressor",
            Pipeline(
                steps=[
                    ("preprocess", pre_tree),
                    ("model", RandomForestRegressor(
                        n_estimators=400,
                        random_state=random_state,
                    )),
                ]
            ),
        ),
        CandidateModel(
            "HistGradientBoostingRegressor",
            Pipeline(
                steps=[
                    ("preprocess", pre_tree),
                    ("model", HistGradientBoostingRegressor(
                        random_state=random_state,
                        max_depth=6,
                        learning_rate=0.08,
                    )),
                ]
            ),
        ),
    ]

    # -----------------------
    # Evaluate candidates
    # -----------------------
    metrics: Dict = {
        "n_projects": int(groups.nunique()),
        "n_rows": int(len(df)),
        "feature_cols": list(feature_cols),
        "classifier": {},
        "regressor": {},
        "selected_models": {},
    }

    clf_dict: Dict[str, Pipeline] = {}
    reg_dict: Dict[str, Pipeline] = {}

    # Classifier
    clf_scores: Dict[str, float] = {}
    for cand in clf_candidates:
        clf_dict[cand.name] = cand.pipeline
        m = _eval_classifier_cv(cand.pipeline, X, y_clf, groups, random_state=random_state)
        metrics["classifier"][cand.name] = m

        # selection score: macro F1 if CV ran; otherwise -inf
        if not m.get("cv_skipped", False):
            clf_scores[cand.name] = float(m["classification_report"]["macro avg"]["f1-score"])
        else:
            clf_scores[cand.name] = float("-inf")

    # Regressor
    reg_scores: Dict[str, float] = {}
    for cand in reg_candidates:
        reg_dict[cand.name] = cand.pipeline
        m = _eval_regressor_cv(cand.pipeline, X, y_reg, groups, random_state=random_state)
        metrics["regressor"][cand.name] = m

        # selection score: negative RMSE if CV ran; otherwise -inf
        if not m.get("cv_skipped", False):
            reg_scores[cand.name] = -float(m["rmse_cost_overrun_ratio"])
        else:
            reg_scores[cand.name] = float("-inf")

    # Pick best by score; if CV skipped for all, default to RF for stability
    best_clf_name = max(clf_scores, key=clf_scores.get) if any(np.isfinite(list(clf_scores.values()))) else "RandomForestClassifier"
    best_reg_name = max(reg_scores, key=reg_scores.get) if any(np.isfinite(list(reg_scores.values()))) else "RandomForestRegressor"

    metrics["selected_models"] = {
        "classifier": best_clf_name,
        "regressor": best_reg_name,
        "selection_rule": "Classifier: best macro-F1 (GroupKFold). Regressor: lowest RMSE (GroupKFold). If CV skipped, default RF.",
    }

    # Fit best models on ALL data (final training)
    best_clf = clf_dict[best_clf_name]
    best_reg = reg_dict[best_reg_name]
    best_clf.fit(X, y_clf)
    best_reg.fit(X, y_reg)

    return TrainedModels(
        clf=best_clf,
        reg=best_reg,
        clf_candidates=clf_dict,
        reg_candidates=reg_dict,
        metrics=metrics,
    )