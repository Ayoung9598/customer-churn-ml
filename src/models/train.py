from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass
class TrainingResult:
    model_path: Path
    metrics: Dict[str, float]


def build_classifier(model_name: str, random_state: int = 42, **kwargs: Any):
    name = model_name.lower()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=None, **kwargs)
    if name in {"xgb", "xgboost"}:
        return XGBClassifier(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=0,
            objective="binary:logistic",
            eval_metric="logloss",
            **kwargs,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Use 'logreg' or 'xgb'.")


def evaluate_binary_classifier(model: Pipeline, X_valid, y_valid) -> Dict[str, float]:
    proba = model.predict_proba(X_valid)[:, 1]
    auc_roc = roc_auc_score(y_valid, proba)
    fpr, tpr, _ = roc_curve(y_valid, proba)
    metrics = {"roc_auc": float(auc_roc), "auc": float(auc(fpr, tpr))}
    return metrics


def fit_and_save(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_valid,
    y_valid,
    output_dir: str | Path,
    model_name: str = "model",
) -> TrainingResult:
    pipeline.fit(X_train, y_train)
    metrics = evaluate_binary_classifier(pipeline, X_valid, y_valid)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / f"{model_name}.joblib"
    joblib.dump(pipeline, model_file)

    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainingResult(model_path=model_file, metrics=metrics)
