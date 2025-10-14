from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import yaml

from src.utils.io import load_model


@st.cache_resource
def get_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        return None
    return load_model(path)


@st.cache_data
def load_columns_config(config_path: str | Path) -> Dict[str, List[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "target": cfg.get("target"),
        "numeric": list(cfg.get("numeric", []) or []),
        "categorical": list(cfg.get("categorical", []) or []),
    }


@st.cache_data
def infer_categorical_values(csv_path: str | Path, cat_cols: List[str]) -> Dict[str, List[str]]:
    if not csv_path or not Path(csv_path).exists():
        return {c: [] for c in cat_cols}
    df = pd.read_csv(csv_path)
    # Drop target if present
    return {c: sorted([v for v in df[c].dropna().unique().tolist() if v != ""]) if c in df.columns else [] for c in cat_cols}


def main() -> None:
    st.set_page_config(page_title="Customer Churn - Streamlit UI", layout="centered")
    st.title("Customer Churn Prediction")
    st.caption("Interactive UI on top of the trained scikit-learn/XGBoost pipeline")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model path", value="artifacts/churn_model.joblib")
    config_path = st.sidebar.text_input("Columns config", value="configs/columns.yaml")
    sample_csv = st.sidebar.text_input("Sample CSV (for choices)", value="data/churn_data.csv")

    cfg = load_columns_config(config_path) if Path(config_path).exists() else {"numeric": [], "categorical": [], "target": None}
    model = get_model(model_path)

    if model is None:
        st.warning("Model not found. Train first or update the model path.")
    else:
        st.success("Model loaded")

    # Tabs: Single prediction | Batch prediction
    tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

    # Prepare choices for categorical fields (best effort from CSV)
    cat_choices = infer_categorical_values(sample_csv, cfg["categorical"]) if cfg["categorical"] else {}

    with tab_single:
        st.subheader("Enter feature values")
        cols = st.columns(2)

        inputs: Dict[str, object] = {}
        # Numeric inputs
        for col in cfg["numeric"]:
            with cols[0]:
                default_val = 0.0
                if col.lower() == "monthlycharges":
                    default_val = 50.0
                inputs[col] = st.number_input(col, value=float(default_val))

        # Categorical inputs
        for col in cfg["categorical"]:
            with cols[1]:
                options = cat_choices.get(col, [])
                if options:
                    default_opt = options[0]
                    inputs[col] = st.selectbox(col, options=options, index=0)
                else:
                    inputs[col] = st.text_input(col, value="")

        if st.button("Predict"):
            if model is None:
                st.error("Model not loaded")
            else:
                df = pd.DataFrame([inputs])
                proba = float(model.predict_proba(df)[:, 1][0])
                pred = int(proba >= 0.5)
                st.metric("Churn probability", f"{proba:.3f}")
                st.write({"prediction": pred})

    with tab_batch:
        st.subheader("Upload CSV for batch prediction")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(batch_df.head())
            if model is None:
                st.error("Model not loaded")
            else:
                proba = model.predict_proba(batch_df)[:, 1]
                preds = (proba >= 0.5).astype(int)
                out = batch_df.copy()
                out["churn_proba"] = proba
                out["churn_pred"] = preds
                st.download_button(
                    label="Download predictions",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()


