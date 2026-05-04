import streamlit as st
import joblib
import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below. The model will classify it as **Real** or **Fake**.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource(show_spinner="Training model...")
def get_model():
    # Synthetic training data for demo
    real = [
        "The Federal Reserve raised interest rates to combat inflation",
        "Scientists publish new study on climate change effects worldwide",
        "Stock markets rose sharply after positive jobs report released",
        "Government announces new infrastructure spending plan for roads",
        "Researchers discover new treatment for common heart disease",
        "Parliament passes new legislation on data privacy and security",
        "Tech company reports strong quarterly earnings and revenue growth",
        "Astronauts complete successful mission on international space station",
        "United Nations holds emergency meeting on global food crisis",
        "Central bank cuts interest rates to stimulate economic growth",
        "New vaccine approved by health authorities after clinical trials",
        "Police arrest suspect in connection with major financial fraud",
        "Olympic committee announces host city for upcoming games event",
        "Scientists detect gravitational waves from distant black hole merger",
        "Company recalls products after safety concerns raised by regulators",
        "Election results certified after independent audit confirms accuracy",
    ]
    fake = [
        "BREAKING secret cure for cancer hidden by big pharma revealed",
        "Aliens confirmed to have landed at area by anonymous insider source",
        "Vaccines cause autism says shocking study funded by unknown group",
        "Government secretly spraying mind control chemicals from airplanes",
        "This miracle food burns belly fat in three days guaranteed results",
        "Famous celebrity arrested for treason by shadow government agents",
        "Scientists admit the earth is actually flat in shocking leaked memo",
        "Towers causing mysterious illness worldwide according to viral post",
        "Doctors hate this man because he discovered one weird health trick",
        "Illuminati confirmed to be running world governments secret leaked",
        "Drinking bleach cures all diseases doctors do not want you knowing",
        "President secretly replaced by clone says whistleblower insider source",
        "Moon landing was faked in hollywood studio new evidence proves it",
        "Bill gates implanting microchips in people through medicine shots",
        "New world order planning to reduce population through water supply",
        "Robot army being secretly built underground to control citizens soon",
    ]

    texts = real + fake
    labels = [0] * len(real) + [1] * len(fake)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english")),
        ("clf", PassiveAggressiveClassifier(max_iter=1000, C=0.5))
    ])
    cleaned = [clean_text(t) for t in texts]
    pipeline.fit(cleaned, labels)
    return pipeline

model = get_model()

text_input = st.text_area("News article / headline:", height=180,
                           placeholder="Paste article text here…")

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)
        label = model.predict([cleaned])[0]

        try:
            proba = model.predict_proba([cleaned])[0]
            confidence = float(max(proba))
            fake_prob = float(proba[1])
        except AttributeError:
            score = model.decision_function([cleaned])[0]
            fake_prob = float(1 / (1 + np.exp(-score)))
            confidence = float(1 / (1 + np.exp(-abs(score))))

        if label == 1:
            st.error(f"🚨 **FAKE NEWS** — Confidence: {confidence:.1%}")
        else:
            st.success(f"✅ **REAL NEWS** — Confidence: {confidence:.1%}")

        st.progress(fake_prob, text=f"Fake probability: {fake_prob:.1%}")

        with st.expander("Details"):
            st.write(f"**Fake probability:** {fake_prob:.4f}")
            st.write(f"**Real probability:** {1 - fake_prob:.4f}")

st.divider()
st.caption("Model: TF-IDF + Passive Aggressive Classifier | For best results provide your own dataset")
