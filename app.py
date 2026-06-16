import html
import os
import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


st.set_page_config(page_title="Fake News Detector",
                   page_icon="📰", layout="wide")


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 42%, #1f2937 100%);
            color: #e5e7eb;
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-right: 0.5rem;
        }
        .pill-fake { background: rgba(248, 113, 113, 0.18); color: #fecaca; }
        .pill-real { background: rgba(74, 222, 128, 0.16); color: #bbf7d0; }
        .highlight-support {
            background: rgba(74, 222, 128, 0.24);
            color: #f8fafc;
            padding: 0.08rem 0.18rem;
            border-radius: 0.3rem;
            border-bottom: 1px solid rgba(74, 222, 128, 0.55);
        }
        .highlight-contradict {
            background: rgba(248, 113, 113, 0.24);
            color: #f8fafc;
            padding: 0.08rem 0.18rem;
            border-radius: 0.3rem;
            border-bottom: 1px solid rgba(248, 113, 113, 0.55);
        }
        .muted { color: #cbd5e1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_artifact_signature():
    paths = [
        MODELS_DIR / "model.joblib",
        MODELS_DIR / "vectorizer.joblib",
        MODELS_DIR / "metadata.joblib",
    ]
    signature = []
    for path in paths:
        signature.append(path.stat().st_mtime_ns if path.exists() else 0)
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def load_artifacts(signature):
    model_path = MODELS_DIR / "model.joblib"
    vectorizer_path = MODELS_DIR / "vectorizer.joblib"
    metadata_path = MODELS_DIR / "metadata.joblib"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else None
    return model, vectorizer, metadata


def tokenize_words(text: str):
    return re.findall(r"\b\w+\b", text)


def score_tokens(model, vectorizer, text: str):
    vectorized_text = vectorizer.transform([text])
    classes = list(model.classes_)
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]

    if not hasattr(model, "coef_"):
        return prediction, probabilities, classes, {}, vectorized_text

    coefficients = model.coef_
    feature_names = vectorizer.get_feature_names_out()
    feature_values = vectorized_text.toarray()[0]

    if coefficients.shape[0] == 1:
        positive_class = classes[1] if len(classes) > 1 else prediction
        selected_coefficients = coefficients[0] if prediction == positive_class else -coefficients[0]
    else:
        class_to_coef = {cls: coefficients[idx]
                         for idx, cls in enumerate(classes)}
        selected_coefficients = class_to_coef.get(prediction, coefficients[0])
    contributions = selected_coefficients * feature_values
    present_indices = np.where(feature_values > 0)[0]

    ranked = sorted(
        ((feature_names[index], float(contributions[index]))
         for index in present_indices),
        key=lambda item: -abs(item[1]),
    )
    top_support = [item for item in ranked if item[1] > 0][:10]
    top_contra = [item for item in ranked if item[1] < 0][:10]
    importance = {token: value for token, value in ranked}
    return prediction, probabilities, classes, {
        "support": top_support,
        "contra": top_contra,
        "importance": importance,
    }, vectorized_text


def highlight_text(text: str, positive_tokens, negative_tokens):
    positive_set = {token.lower() for token, _ in positive_tokens}
    negative_set = {token.lower() for token, _ in negative_tokens}
    if not positive_set and not negative_set:
        return html.escape(text)

    all_tokens = sorted({token for token in positive_set |
                        negative_set if token}, key=len, reverse=True)
    if not all_tokens:
        return html.escape(text)

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(token)
                               for token in all_tokens) + r")(?!\w)",
        re.IGNORECASE,
    )

    output = []
    last_index = 0
    for match in pattern.finditer(text):
        output.append(html.escape(text[last_index:match.start()]))
        token = match.group(0)
        token_lower = token.lower()
        if token_lower in positive_set:
            output.append(
                f'<span class="highlight-support">{html.escape(token)}</span>')
        elif token_lower in negative_set:
            output.append(
                f'<span class="highlight-contradict">{html.escape(token)}</span>')
        else:
            output.append(html.escape(token))
        last_index = match.end()

    output.append(html.escape(text[last_index:]))
    return "".join(output)


load_css()
model, vectorizer, metadata = load_artifacts(get_artifact_signature())

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0; color:#f8fafc;">Fake News Detector</h1>
        <p class="muted" style="margin-bottom:0;">Paste a news article and get a fast FAKE / REAL prediction with word-level evidence from your offline Kaggle dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model")
    st.caption("Offline TF-IDF + Logistic Regression")
    if metadata:
        st.write(f"Rows trained: {metadata.get('rows', 'unknown')}")
        st.write(f"Source: {metadata.get('dataset_path', 'unknown')}")
        counts = metadata.get("class_counts", {})
        if counts:
            st.write("Class balance:")
            st.json(counts)
    else:
        st.info(
            "Model not trained yet. Run `python train.py` after adding the Kaggle CSV files.")

    st.divider()
    if model is not None and vectorizer is not None:
        st.success("Model loaded from `models/`.")

input_mode = st.radio(
    "Input mode", ["Paste text", "Load sample"], horizontal=True)
sample_text = (
    "The government announced a new support program for small businesses after months of consultation."
)
news_text = st.text_area(
    "News text",
    key="news_text",
    value=sample_text if input_mode == "Load sample" else "",
    height=260,
    placeholder="Paste the news article here...",
)

col1, col2 = st.columns([1, 1])
with col1:
    check_clicked = st.button(
        "Analyze news", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["news_text"] = ""
    st.rerun()

if check_clicked:
    if not news_text.strip():
        st.info("Please paste some news text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model not available. Train the model first (see README).")
    else:
        prediction, probabilities, classes, explanation, vectorized_text = score_tokens(
            model, vectorizer, news_text
        )
        probability_map = {label: float(prob)
                           for label, prob in zip(classes, probabilities)}
        top_label = max(probability_map, key=probability_map.get)
        top_probability = probability_map[top_label]

        badge_class = "pill-real" if prediction == "REAL" else "pill-fake"
        st.markdown(
            f"""
            <div class="result-card">
                <span class="pill {badge_class}">{prediction}</span>
                <span style="font-size:1.1rem;color:#f8fafc;font-weight:700;">Confidence: {top_probability:.1%}</span>
                <div class="muted" style="margin-top:0.4rem;">Top score: {top_label} · Probabilities: {probability_map}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        support = explanation.get("support", []) if explanation else []
        contra = explanation.get("contra", []) if explanation else []

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Highlighted explanation")
            highlighted_html = highlight_text(news_text, support, contra)
            st.markdown(
                f'<div class="result-card" style="line-height:1.9; font-size:1.02rem;">{highlighted_html}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Green words push toward the prediction. Red words push against it.")

        with right:
            st.subheader("Why it was predicted this way")
            if support:
                st.markdown("**Words supporting the result**")
                for word, value in support:
                    st.write(f"- {word}: {value:.4f}")
            else:
                st.write("No strong supporting words were found.")

            if contra:
                st.markdown("**Words pushing against it**")
                for word, value in contra:
                    st.write(f"- {word}: {value:.4f}")
            else:
                st.write("No strong contradicting words were found.")
