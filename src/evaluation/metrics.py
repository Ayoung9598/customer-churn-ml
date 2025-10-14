from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def compute_classification_report(y_true, y_pred) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True)
    # Flatten to top-level summary
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "support": report["weighted avg"]["support"],
    }


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_calibration(y_true, y_proba, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return prob_true, prob_pred
