from pathlib import Path
import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from utils import (
    MODEL_PATH,
    load_metadata,
    load_trained_model,
    predict_lesion,
)

APP_TITLE = "DermaSense AI"
DB_PATH = Path("data") / "analysis_history.db"

st.set_page_config(
    page_title=f"{APP_TITLE} | Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg:#07111f;
  --panel:#0b1728;
  --panel2:#0f2035;
  --border:#1d3853;
  --text:#edf7ff;
  --muted:#9fb6c9;
  --cyan:#38d6d2;
  --blue:#5fa8ff;
  --amber:#ffce6b;
  --red:#ff7b8a;
  --green:#76e2a8;
}
.stApp {
  background:
    radial-gradient(circle at 10% 5%, rgba(56,214,210,.10), transparent 28%),
    radial-gradient(circle at 95% 15%, rgba(95,168,255,.10), transparent 25%),
    linear-gradient(180deg, #06101d 0%, #081422 100%);
  color: var(--text);
}
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #071321, #091a2b);
  border-right: 1px solid var(--border);
}
.hero {
  padding: 26px 28px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(12,31,51,.95), rgba(7,20,34,.95));
  box-shadow: 0 16px 45px rgba(0,0,0,.20);
  margin-bottom: 1rem;
}
.hero-kicker {
  color: var(--cyan); letter-spacing: .14em; font-size: .75rem;
  font-weight: 800; text-transform: uppercase;
}
.hero h1 { margin:.25rem 0 .4rem; font-size:2.45rem; line-height:1.05; }
.hero p { color:var(--muted); margin:0; max-width:800px; font-size:1.02rem; }
.card {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 19px 20px;
  background: rgba(11,23,40,.92);
  min-height: 100%;
}
.card-title { font-weight:800; font-size:1.0rem; margin-bottom:.35rem; }
.muted { color:var(--muted); }
.score {
  font-size:2.1rem; font-weight:900; line-height:1;
  margin:.2rem 0 .5rem;
}
.tag {
  display:inline-block; border-radius:999px; padding:7px 11px;
  border:1px solid var(--border); background:#10243a; font-weight:800;
}
.good { color:var(--green); }
.warn { color:var(--amber); }
.alert { color:var(--red); }
.info-strip {
  border:1px solid #244767; border-left:4px solid var(--blue);
  border-radius:14px; padding:12px 14px; background:#0d1d30; color:#cfe6f7;
}
.disclaimer {
  border:1px solid #664d1f; border-left:4px solid var(--amber);
  border-radius:14px; padding:12px 14px; background:rgba(87,62,14,.20); color:#ffe7ad;
}
div.stButton > button {
  border-radius:14px;
  border:1px solid #286c7b;
  background: linear-gradient(135deg, #126c79, #245a96);
  color:white;
  font-weight:800;
  min-height:46px;
}
div.stButton > button:hover { border-color:#48e4dd; color:white; }
[data-testid="stFileUploader"] {
  border:1px dashed #2c607a; border-radius:18px; padding:10px;
  background:rgba(10,29,47,.75);
}
[data-testid="stMetric"] {
  background:rgba(11,23,40,.9);
  border:1px solid var(--border);
  padding:13px 15px;
  border-radius:16px;
}
hr { border-color:#17324a; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                melanoma_score REAL NOT NULL,
                score_band TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_history(filename, prediction, confidence, melanoma_score, score_band):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO history
            (timestamp, filename, prediction, confidence, melanoma_score, score_band)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                prediction,
                float(confidence),
                float(melanoma_score),
                score_band,
            ),
        )
        conn.commit()


def read_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM history ORDER BY id DESC LIMIT 100",
            conn,
        )


def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history")
        conn.commit()


def get_band(score):
    if score < 0.35:
        return "Lower model score", "good"
    if score < 0.65:
        return "Uncertain model score", "warn"
    return "Higher model score", "alert"


init_db()

with st.sidebar:
    st.markdown("## 🔬 DermaSense AI")
    st.caption("Skin Lesion Classifier • CNN")
    page = st.radio(
        "Navigation",
        ["Analyze", "History"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if MODEL_PATH.exists():
        st.success("Trained model detected")
    else:
        st.warning("Model not trained yet")
    st.caption("Educational prototype. Not a clinical diagnostic device.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI-Powered Image Classification</div>
      <h1>Skin Lesion Classifier</h1>
      <p>Upload a skin-lesion image and let a trained MobileNetV2-based CNN estimate whether the image looks benign-like or melanoma-suspicious.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Analyze":
    st.markdown(
        '<div class="disclaimer"><b>Important:</b> This project is for education and AI demonstration only. It cannot diagnose melanoma or replace a dermatologist. A suspicious or changing skin lesion should be assessed by a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    model = None
    metadata = {}
    if MODEL_PATH.exists():
        try:
            model = load_trained_model()
            metadata = load_metadata()
        except Exception as exc:
            st.error(f"Model exists but could not be loaded: {exc}")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### 1. Upload image")
        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG skin-lesion image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        image = None
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                st.image(image, caption=uploaded.name, use_container_width=True)
            except (UnidentifiedImageError, OSError):
                st.error("That file could not be read as a valid image.")

    with right:
        st.markdown("### 2. AI analysis")

        if model is None:
            st.info(
                "No trained model is available yet. Put labeled images in "
                "`dataset/benign` and `dataset/melanoma`, then run `python train_model.py`."
            )
            st.code("python train_model.py", language="bash")
        elif image is None:
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">Waiting for an image</div>
                  <div class="muted">Upload a skin-lesion image on the left. The analysis button will appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.button("🔎 Analyze Lesion", use_container_width=True):
                with st.spinner("CNN is analyzing the image..."):
                    result = predict_lesion(model, image, metadata)

                band, css_class = get_band(result["melanoma_score"])
                prediction = result["prediction"]
                confidence = result["confidence"]
                melanoma_score = result["melanoma_score"]

                add_history(
                    uploaded.name,
                    prediction,
                    confidence,
                    melanoma_score,
                    band,
                )

                st.markdown(
                    f"""
                    <div class="card">
                      <div class="card-title">Prediction</div>
                      <div class="score">{prediction}</div>
                      <span class="tag {css_class}">{band}</span>
                      <br><br>
                      <div class="muted">Classification confidence</div>
                      <div style="font-size:1.45rem;font-weight:900">{confidence*100:.1f}%</div>
                      <div class="muted" style="margin-top:10px">Melanoma-class model score</div>
                      <div style="font-size:1.25rem;font-weight:800">{melanoma_score*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(melanoma_score))

                if 0.35 <= melanoma_score < 0.65:
                    st.warning(
                        "The model score is uncertain. Do not interpret this as a diagnosis."
                    )
                elif melanoma_score >= 0.65:
                    st.warning(
                        "The model produced a higher melanoma-class score. This still does not confirm melanoma."
                    )
                else:
                    st.info(
                        "The model produced a lower melanoma-class score. This does not rule out melanoma."
                    )

                st.caption(
                    f"Decision threshold: {result['threshold']:.2f} • "
                    f"Image size: {result['image_size']}×{result['image_size']}"
                )

elif page == "History":
    st.markdown("### Analysis history")
    st.caption(
        "Only filename and model result metadata are stored locally. Uploaded images are not saved by this project."
    )
    history = read_history()

    if history.empty:
        st.info("No analyses have been recorded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyses", len(history))
        c2.metric(
            "Benign-like",
            int((history["prediction"] == "Benign-like").sum()),
        )
        c3.metric(
            "Melanoma-suspicious",
            int((history["prediction"] == "Melanoma-suspicious").sum()),
        )

        display = history[
            ["timestamp", "filename", "prediction", "confidence", "melanoma_score", "score_band"]
        ].copy()
        display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
        display["melanoma_score"] = (
            display["melanoma_score"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(display, use_container_width=True, hide_index=True)

        if st.button("Clear history"):
            clear_history()
            st.success("History cleared.")
            st.rerun()

st.markdown("---")
st.caption("DermaSense AI • Educational use only")
