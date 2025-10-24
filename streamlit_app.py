from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import streamlit as st
import yaml
import plotly.graph_objects as go

from src.utils.io import load_model


# ---- Cached utilities ----
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
    return {
        c: sorted(
            [v for v in df[c].dropna().unique().tolist() if v != ""]
        ) if c in df.columns else []
        for c in cat_cols
    }


@st.cache_data
def load_model_metrics(metrics_path: str | Path):
    """Load model performance metrics if available."""
    path = Path(metrics_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- Streamlit main app ----
def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
    st.title("📊 Telco Customer Churn Prediction")
    st.caption("Predict the likelihood of a customer leaving the telecom service")

    # ---- About / Overview ----
    with st.expander("ℹ️ About this project", expanded=True):
        st.markdown("""
        **Goal:**  
        This interactive app predicts customer churn — whether a telecom customer is likely to leave the service — 
        using a trained **XGBoost** model built on structured customer data.

        **Dataset Overview:**  
        Each row represents a customer, and each column contains attributes about:
        - **Churn:** Whether the customer left within the last month  
        - **Services:** Phone, multiple lines, internet, online security, backup, device protection, streaming TV/movies  
        - **Account Info:** Tenure, contract type, payment method, paperless billing, monthly & total charges  
        - **Demographics:** Gender, senior citizen status, partner, dependents  

        **Use Case:**  
        Businesses can use this to identify customers at high churn risk and take proactive retention actions
        (e.g., loyalty offers or service quality improvements).
        """)

    # ---- Sidebar configuration ----
    st.sidebar.header("⚙️ Configuration")
    model_path = st.sidebar.text_input("Model path", value="artifacts/churn_model.joblib")
    config_path = st.sidebar.text_input("Columns config", value="configs/columns.yaml")
    sample_csv = st.sidebar.text_input("Sample CSV (for category inference)", value="data/churn_data.csv")
    metrics_path = st.sidebar.text_input("Metrics file", value="artifacts/churn_model_metrics.json")

    cfg = load_columns_config(config_path) if Path(config_path).exists() else {
        "numeric": [], "categorical": [], "target": None
    }
    model = get_model(model_path)
    metrics = load_model_metrics(metrics_path)

    if model is None:
        st.warning("⚠️ Model not found. Train first or update the model path.")
    else:
        st.success("✅ Model loaded successfully!")

    # ---- Model Information ----
    with st.expander("🧠 Model Information", expanded=True):
        st.markdown("""
        - **Algorithm:** XGBoost Classifier  
        - **Framework:** scikit-learn / XGBoost pipeline  
        - **Input Features:** Numeric + Categorical columns (encoded)
        - **Output:** Probability of churn (0–1) and binary prediction (0 = stay, 1 = churn)
        - **Metric Used:** Accuracy / ROC-AUC on validation data
        """)

        # ---- Model Performance Metrics Section ----
        if metrics:
            # Normalize values to float (important for Plotly)
            metrics = {k: float(v) for k, v in metrics.items()}

            st.markdown("#### 📈 Model Performance Metrics")

            # Show metrics
            cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                cols[i].metric(label=k.upper(), value=f"{v:.3f}")

            # ---- Display ROC-AUC Gauge ----
            roc_auc = metrics.get("roc_auc") or metrics.get("auc")
            if roc_auc is not None:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=roc_auc,
                        title={"text": "ROC-AUC Score"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "green" if roc_auc > 0.75 else "orange"},
                            "steps": [
                                {"range": [0, 0.6], "color": "#ffcccc"},
                                {"range": [0.6, 0.8], "color": "#ffe699"},
                                {"range": [0.8, 1.0], "color": "#c6efce"},
                            ],
                        },
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No ROC-AUC value found in metrics JSON.")
        else:
            st.info("No metrics file found (expected at `artifacts/churn_model_metrics.json`).")

    # ---- Tabs ----
    tab_single, tab_batch = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction"])
    cat_choices = infer_categorical_values(sample_csv, cfg["categorical"]) if cfg["categorical"] else {}

    # ---- Single Prediction ----
    with tab_single:
        st.subheader("Enter Feature Values")
        cols = st.columns(2)
        inputs: Dict[str, object] = {}

        for col in cfg["numeric"]:
            with cols[0]:
                default_val = 0.0
                if col.lower() == "monthlycharges":
                    default_val = 50.0
                inputs[col] = st.number_input(col, value=float(default_val))

        for col in cfg["categorical"]:
            with cols[1]:
                options = cat_choices.get(col, [])
                if options:
                    inputs[col] = st.selectbox(col, options=options, index=0)
                else:
                    inputs[col] = st.text_input(col, value="")

        if st.button("🔍 Predict Churn"):
            if model is None:
                st.error("Model not loaded")
            else:
                df = pd.DataFrame([inputs])
                proba = float(model.predict_proba(df)[:, 1][0])
                pred = int(proba >= 0.5)
                st.metric("Churn Probability", f"{proba:.3f}")
                st.write(f"**Predicted class:** {'Churn' if pred == 1 else 'No Churn'}")

    # ---- Batch Prediction ----
    with tab_batch:
        st.subheader("Upload a CSV for Batch Prediction")
        uploaded = st.file_uploader("Select a CSV file", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            if model is None:
                st.error("Model not loaded")
            else:
                proba = model.predict_proba(batch_df)[:, 1]
                preds = (proba >= 0.5).astype(int)
                out = batch_df.copy()
                out["churn_proba"] = proba
                out["churn_pred"] = preds

                st.success("✅ Predictions generated successfully!")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
