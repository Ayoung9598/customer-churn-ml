from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.data.processing import create_preprocessor


def build_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    k_best: Optional[int] = None,
) -> Pipeline:
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    steps = [("preprocess", preprocessor)]

    if k_best is not None and k_best > 0:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

    return Pipeline(steps=steps)
