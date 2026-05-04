"""
Streamlit Web App — Fake News Detector
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import os
import numpy as np
from fake_news_detector import clean_text, load_data, train_and_evaluate, save_best_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake News Detector")
st.markdown(
    "Paste any news article or headline below. "
    "The model will classify it as **Real** or **Fake**."
)

# ── Load / train model ─────────────────────────────────────────────────────────
MODEL_PATH = "outputs/best_model.joblib"

@st.cache_resource(show_spinner="Training model on first run…")
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Train from scratch using synthetic data
    df = load_data()
    results, _, _ = train_and_evaluate(df)
    save_best_model(results)
    return joblib.load(MODEL_PATH)

model = get_model()

# ── Input ─────────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "News article / headline:",
    height=180,
    placeholder="Paste article text here…"
)

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)
        label   = model.predict([cleaned])[0]

        try:
            proba      = model.predict_proba([cleaned])[0]
            confidence = float(max(proba))
            fake_prob  = float(proba[1])
        except AttributeError:
            score      = model.decision_function([cleaned])[0]
            fake_prob  = float(1 / (1 + np.exp(-score)))
            confidence = float(1 / (1 + np.exp(-abs(score))))

        if label == 1:
            st.error(f"🚨 **FAKE NEWS** — Confidence: {confidence:.1%}")
        else:
            st.success(f"✅ **REAL NEWS** — Confidence: {confidence:.1%}")

        st.progress(fake_prob, text=f"Fake probability: {fake_prob:.1%}")

        with st.expander("Details"):
            st.write(f"**Cleaned input:** {cleaned[:300]}…")
            st.write(f"**Fake probability:** {fake_prob:.4f}")
            st.write(f"**Real probability:** {1 - fake_prob:.4f}")

st.divider()
st.caption(
    "Model trained with TF-IDF + Logistic Regression / SVC / Naive Bayes. "
    "Provide your own dataset in `data/` for best results."
)
