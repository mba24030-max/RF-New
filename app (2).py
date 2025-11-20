import streamlit as st
import json
import pickle
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Random Forest Demo", layout="centered")
st.title("Random Forest - Client Demo")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.pkl"
FEAT_PATH = BASE / "feature.column.json"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        return model
    return None

@st.cache_data
def load_features():
    if FEAT_PATH.exists():
        with open(FEAT_PATH, "r", encoding="utf-8") as f:
            feats = json.load(f)
        return feats
    return []

model = load_model()
features = load_features()

if model is None:
    st.error("model.pkl not found or could not be loaded. Place your trained model at 'model.pkl'.")
    st.stop()

st.write("Loaded model:", type(model).__name__)

st.header("Inputs")
inputs = {}
cols = st.columns(2)
categorical_kw = ("cat", "dept", "role", "type", "status", "gender", "marital", "travel", "education", "job")

for i, feat in enumerate(features):
    col = cols[i % 2]
    if any(k in feat.lower() for k in categorical_kw):
        # Placeholder categorical options - update these to the real domain values
        val = col.selectbox(feat, options=["option_1", "option_2", "option_3"])
    else:
        val = col.number_input(feat, value=0.0, format="%.6f")
    inputs[feat] = val

if st.button("Predict"):
    X = pd.DataFrame([inputs])
    try:
        preds = model.predict(X)
        st.subheader("Prediction")
        st.write(preds)
        if hasattr(model, "predict_proba"):
            st.subheader("Prediction probabilities")
            st.write(model.predict_proba(X))
    except Exception as e:
        st.error(f"Prediction failed: {e}\nEnsure the saved model is a pipeline that accepts a DataFrame with columns matching feature.column.json.")
