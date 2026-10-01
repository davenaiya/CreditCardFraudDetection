"""Streamlit frontend for the credit-card fraud notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = next((path for path in (ROOT / "creditcard.csv", ROOT / "creditcard.xls") if path.exists()), ROOT / "creditcard.csv")
MODEL_PATH = ROOT / "best_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
TARGET = "Class"
FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]

st.set_page_config(page_title="Transaction Risk Lab", page_icon="💳", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 960px; padding-top: 2rem;}
  .hero {padding:24px 28px;border-radius:18px;background:linear-gradient(115deg,#eff6ff,#f0fdfa);border:1px solid #dbeafe;margin-bottom:22px;}
  .hero h1 {color:#16324f;margin:0 0 8px 0;}
  .hero p {color:#53677d;margin:0;}
  .result-card {padding:20px 24px;border-radius:16px;margin:18px 0;background:#f0fdf4;border:1px solid #bbf7d0;}
  .result-card.fraud {background:#fff1f2;border-color:#fecdd3;}
  .result-card h2 {margin:0;color:#166534;}
  .result-card.fraud h2 {color:#be123c;}
  .result-card p {margin:6px 0 0;color:#475569;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the fraud model from the labeled dataset…")
def load_model_and_data(data_path: str, model_path: str, scaler_path: str, modified: float):
    # The project file is named .xls but its contents are CSV (header starts with "Time").
    data = pd.read_csv(data_path)
    missing = set([*FEATURES, TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    data = data[[*FEATURES, TARGET]].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=[TARGET])
    data[FEATURES] = data[FEATURES].fillna(data[FEATURES].median())
    data[TARGET] = data[TARGET].astype(int)
    if data[TARGET].nunique() != 2:
        raise ValueError("The Class column must contain both 0 (normal) and 1 (fraud) examples.")

    # Use the model and scaler saved by the notebook when available.
    if Path(model_path).is_file() and Path(scaler_path).is_file():
        import joblib

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, data, "Notebook-selected model"

    # Fresh checkout fallback: fit on all labeled rows, matching the notebook's scaler + classifier flow.
    scaler = StandardScaler().fit(data[FEATURES])
    model = RandomForestClassifier(
        n_estimators=180,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(scaler.transform(data[FEATURES]), data[TARGET])
    return model, scaler, data, "Random Forest trained on the full dataset"


def generate_synthetic_rows(data: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Create feature rows from balanced normal/fraud patterns; never pass Class to the model."""
    rng = np.random.default_rng(seed)
    feature_data = data[FEATURES]
    spread = feature_data.std().fillna(0).to_numpy()
    # The source dataset is very imbalanced. Choose a class pattern evenly so the demo
    # can show both kinds of model response. Class is used only to choose source features.
    source_labels = rng.integers(0, 2, size=count)
    base_rows = []
    for source_label in source_labels:
        candidates = data.loc[data[TARGET] == source_label, FEATURES]
        base_rows.append(candidates.iloc[int(rng.integers(0, len(candidates)))].to_numpy(dtype=float))
    base = np.asarray(base_rows)
    synthetic = base + rng.normal(0, 0.04, size=base.shape) * spread
    batch = pd.DataFrame(synthetic, columns=FEATURES)
    batch["Time"] = batch["Time"].clip(0, 172800)
    batch["Amount"] = batch["Amount"].clip(lower=0)
    return batch


def predict_new_transaction(data: pd.DataFrame, model, scaler) -> None:
    """Generate and score one new transaction in the same button callback."""
    generation_number = st.session_state.get("generation_number", 0) + 1
    previous = st.session_state.get("prediction_features")
    seed_base = int(np.random.SeedSequence().entropy)
    for attempt in range(10):
        batch = generate_synthetic_rows(data, 1, seed_base + generation_number + attempt)
        rounded_batch = batch.round(5)
        if previous is None or not rounded_batch.equals(previous):
            break
    else:
        # Last-resort uniqueness guard if a row happens to repeat after rounding.
        batch.loc[0, "V28"] += 0.001
        rounded_batch = batch.round(5)

    probability = float(model.predict_proba(scaler.transform(batch[FEATURES]))[0, 1])
    st.session_state["generation_number"] = generation_number
    st.session_state["generation_label"] = generation_number
    st.session_state["fraud_probability"] = probability
    st.session_state["prediction_features"] = rounded_batch


st.markdown("""
<div class="hero">
  <h1>💳 Transaction Fraud Check</h1>
  <p>Generate a synthetic transaction and instantly check whether the model predicts fraud or not fraud.</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("Training dataset not found. Place creditcard.csv beside app.py (the app also accepts the existing creditcard.xls filename).")
    st.stop()

try:
    model, scaler, data, model_name = load_model_and_data(
        str(DATA_PATH), str(MODEL_PATH), str(SCALER_PATH), DATA_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

st.caption(f"Trained on {len(data):,} labeled transactions · A new synthetic transaction is created for every prediction.")

st.button(
    "✨ Generate transaction and predict",
    type="primary",
    use_container_width=True,
    on_click=predict_new_transaction,
    args=(data, model, scaler),
)

if "fraud_probability" in st.session_state:
    probability = st.session_state["fraud_probability"]
    is_fraud = probability >= 0.5
    st.markdown("---")
    label = "LIKELY FRAUD" if is_fraud else "LIKELY NOT FRAUD"
    card_class = "result-card fraud" if is_fraud else "result-card"
    st.markdown(
        f'<div class="{card_class}"><h2>{label}</h2><p>Fraud probability: <b>{probability:.1%}</b></p></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Prediction probabilities")
    chart = go.Figure(go.Bar(
        x=[probability, 1 - probability],
        y=["Fraud", "Not fraud"],
        orientation="h",
        marker={"color": ["#ef4444", "#22c55e"], "line": {"width": 0}},
        text=[f"{probability:.1%}", f"{(1-probability):.1%}"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    chart.update_layout(
        height=190,
        margin={"l": 10, "r": 55, "t": 10, "b": 25},
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": None, "gridcolor": "#e9eef5"},
        yaxis={"title": None, "autorange": "reversed"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font={"color": "#334155", "size": 13},
    )
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader(f"Generated transaction #{st.session_state['generation_label']} features")
    features = st.session_state["prediction_features"].iloc[0]
    feature_table = pd.DataFrame({"Feature": features.index, "Value": features.values})
    left_features, right_features = st.columns(2)
    midpoint = (len(feature_table) + 1) // 2
    left_features.dataframe(feature_table.iloc[:midpoint], hide_index=True, use_container_width=True)
    right_features.dataframe(feature_table.iloc[midpoint:], hide_index=True, use_container_width=True)

st.caption("The workbook is used to train the model. Each click generates a new synthetic transaction; predictions are demonstration estimates.")
