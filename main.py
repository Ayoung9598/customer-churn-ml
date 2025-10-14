from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from src.data.processing import DatasetSplits, load_csv_dataset, train_valid_split
from src.features.engineering import build_feature_pipeline
from src.models.train import TrainingResult, build_classifier, fit_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate churn prediction models")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--config", default=None, help="Path to YAML with target/numeric/categorical")
    parser.add_argument("--target", default=None, help="Target column name (overrides config)")
    parser.add_argument(
        "--num-cols", nargs="*", default=[], help="List of numeric feature columns (overrides config)"
    )
    parser.add_argument(
        "--cat-cols", nargs="*", default=[], help="List of categorical feature columns (overrides config)"
    )
    parser.add_argument(
        "--model",
        default="xgb",
        choices=["logreg", "xgb"],
        help="Classifier to use",
    )
    parser.add_argument("--k-best", type=int, default=None, help="Optional k for SelectKBest")
    parser.add_argument("--out", default="artifacts", help="Output directory for model and metrics")
    parser.add_argument("--name", default="churn_model", help="Model name prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration if provided
    cfg_target: Optional[str] = None
    cfg_numeric: List[str] = []
    cfg_categorical: List[str] = []
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg_target = cfg.get("target")
        cfg_numeric = list(cfg.get("numeric", []) or [])
        cfg_categorical = list(cfg.get("categorical", []) or [])

    # CLI overrides config when provided
    target: Optional[str] = args.target or cfg_target
    if not target:
        raise ValueError("Target column must be provided via --target or in --config YAML")

    num_cols: List[str] = args.num_cols if args.num_cols else cfg_numeric
    cat_cols: List[str] = args.cat_cols if args.cat_cols else cfg_categorical

    X, y = load_csv_dataset(args.data, target)

    splits: DatasetSplits = train_valid_split(X, y, random_state=args.seed)

    feature_pipeline = build_feature_pipeline(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        k_best=args.k_best,
    )

    clf = build_classifier(args.model, random_state=args.seed)

    from sklearn.pipeline import Pipeline

    pipeline: Pipeline = Pipeline(
        steps=[
            ("features", feature_pipeline),
            ("clf", clf),
        ]
    )

    result: TrainingResult = fit_and_save(
        pipeline,
        splits.X_train,
        splits.y_train,
        splits.X_valid,
        splits.y_valid,
        args.out,
        args.name,
    )

    print("Saved model:", result.model_path)
    print("Metrics:", result.metrics)


if __name__ == "__main__":
    main()
