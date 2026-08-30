from pathlib import Path
import io
import sqlite3
from datetime import datetime

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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=f"{APP_TITLE} | Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

CUSTOM_CSS = """
<style>

:root {
    --bg: #07111f;
    --panel: #0b1728;
    --panel2: #0f2035;
    --border: #1d3853;
    --text: #edf7ff;
    --muted: #9fb6c9;
    --cyan: #38d6d2;
    --blue: #5fa8ff;
    --amber: #ffce6b;
    --red: #ff7b8a;
    --green: #76e2a8;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(56,214,210,.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 15%,
            rgba(95,168,255,.10),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #06101d 0%,
            #081422 100%
        );

    color: var(--text);
}


.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #071321,
            #091a2b
        );

    border-right: 1px solid var(--border);
}


.hero {
    padding: 26px 28px;

    border: 1px solid var(--border);

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(12,31,51,.95),
            rgba(7,20,34,.95)
        );

    box-shadow:
        0 16px 45px rgba(0,0,0,.20);

    margin-bottom: 1rem;
}


.hero-kicker {
    color: var(--cyan);

    letter-spacing: .14em;

    font-size: .75rem;

    font-weight: 800;

    text-transform: uppercase;
}


.hero h1 {
    margin: .25rem 0 .4rem;

    font-size: 2.45rem;

    line-height: 1.05;
}


.hero p {
    color: var(--muted);

    margin: 0;

    max-width: 850px;

    font-size: 1.02rem;
}


.card {
    border: 1px solid var(--border);

    border-radius: 20px;

    padding: 19px 20px;

    background: rgba(11,23,40,.92);

    min-height: 100%;
}


.card-title {
    font-weight: 800;

    font-size: 1rem;

    margin-bottom: .35rem;
}


.muted {
    color: var(--muted);
}


.score {
    font-size: 2rem;

    font-weight: 900;

    line-height: 1.1;

    margin: .3rem 0 .6rem;
}


.tag {
    display: inline-block;

    border-radius: 999px;

    padding: 7px 11px;

    border: 1px solid var(--border);

    background: #10243a;

    font-weight: 800;
}


.good {
    color: var(--green);
}


.warn {
    color: var(--amber);
}


.alert {
    color: var(--red);
}


.blue {
    color: var(--blue);
}


.info-strip {
    border: 1px solid #244767;

    border-left: 4px solid var(--blue);

    border-radius: 14px;

    padding: 12px 14px;

    background: #0d1d30;

    color: #cfe6f7;
}


.disclaimer {
    border: 1px solid #664d1f;

    border-left: 4px solid var(--amber);

    border-radius: 14px;

    padding: 12px 14px;

    background: rgba(87,62,14,.20);

    color: #ffe7ad;
}


div.stButton > button {
    border-radius: 14px;

    border: 1px solid #286c7b;

    background:
        linear-gradient(
            135deg,
            #126c79,
            #245a96
        );

    color: white;

    font-weight: 800;

    min-height: 46px;
}


div.stButton > button:hover {
    border-color: #48e4dd;
    color: white;
}


[data-testid="stFileUploader"] {
    border: 1px dashed #2c607a;

    border-radius: 18px;

    padding: 10px;

    background:
        rgba(10,29,47,.75);
}


[data-testid="stMetric"] {
    background:
        rgba(11,23,40,.9);

    border: 1px solid var(--border);

    padding: 13px 15px;

    border-radius: 16px;
}


hr {
    border-color: #17324a;
}

</style>
"""


st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# =========================================================
# DATABASE
# =========================================================

def init_db():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def add_history(
    filename,
    prediction,
    confidence,
    melanoma_score,
    score_band,
):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            INSERT INTO history
            (
                timestamp,
                filename,
                prediction,
                confidence,
                melanoma_score,
                score_band
            )

            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

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
            """
            SELECT *
            FROM history
            ORDER BY id DESC
            LIMIT 100
            """,
            conn,
        )


def clear_history():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            "DELETE FROM history"
        )

        conn.commit()


init_db()


# =========================================================
# RESULT HELPERS
# =========================================================

def result_style(
    prediction,
    confidence,
):

    if prediction == "Benign-like":

        return (
            "Benign-like pattern",
            "good",
        )

    if prediction == "Melanoma-suspicious":

        return (
            "Suspicious model pattern",
            "alert",
        )

    if prediction == "Other skin lesion":

        return (
            "Different lesion type detected",
            "blue",
        )

    if prediction == "Unsupported image":

        return (
            "Non-skin / unsupported image",
            "warn",
        )

    return (
        "Low-confidence result",
        "warn",
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🔬 DermaSense AI"
    )

    st.caption(
        "4-Class Skin Image Classifier"
    )

    page = st.radio(
        "Navigation",
        [
            "Analyze",
            "History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if MODEL_PATH.exists():

        st.success(
            "Trained model detected"
        )

    else:

        st.warning(
            "Model not trained yet"
        )

    st.caption(
        "Custom CNN trained from scratch."
    )

    st.caption(
        "Educational AI prototype."
    )

    st.caption(
        "Not a medical diagnostic device."
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            AI-Powered Skin Image Analysis
        </div>

        <h1>
            Skin Lesion Classifier
        </h1>

        <p>
            Upload an image and let our custom CNN,
            trained from scratch, classify it as
            benign-like, melanoma-suspicious,
            another skin-lesion type, or an
            unsupported non-skin image.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# ANALYZE PAGE
# =========================================================

if page == "Analyze":

    st.markdown(
        """
        <div class="disclaimer">

        <b>Important:</b>

        DermaSense AI is an educational machine-learning
        prototype. Its output must not be treated as a
        diagnosis or used instead of evaluation by a
        qualified healthcare professional.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = None
    metadata = {}

    if MODEL_PATH.exists():

        try:

            model = load_trained_model()

            metadata = load_metadata()

        except Exception as exc:

            st.error(
                f"Model exists but could not be loaded: {exc}"
            )


    left, right = st.columns(
        [1.05, 0.95],
        gap="large",
    )


    # -----------------------------------------------------
    # IMAGE UPLOAD
    # -----------------------------------------------------

    with left:

        st.markdown(
            "### 1. Upload image"
        )

        st.caption(
            "Upload a clear image for AI analysis."
        )

        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            accept_multiple_files=False,
        )

        image = None

        if uploaded is not None:

            try:

                raw = uploaded.getvalue()

                image = Image.open(
                    io.BytesIO(raw)
                ).convert("RGB")

                st.image(
                    image,
                    caption=uploaded.name,
                    use_container_width=True,
                )

            except (
                UnidentifiedImageError,
                OSError,
            ):

                st.error(
                    "This file could not be read "
                    "as a valid image."
                )


    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    with right:

        st.markdown(
            "### 2. AI analysis"
        )

        if model is None:

            st.info(
                "The trained CNN model is not available."
            )


        elif image is None:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Waiting for an image
                    </div>

                    <div class="muted">
                        Upload an image on the left.
                        The Analyze button will appear here.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        else:

            if st.button(
                "🔎 Analyze Image",
                use_container_width=True,
            ):

                with st.spinner(
                    "DermaSense AI is analyzing the image..."
                ):

                    result = predict_lesion(
                        model,
                        image,
                        metadata,
                    )


                prediction = result[
                    "prediction"
                ]

                class_key = result[
                    "class_key"
                ]

                confidence = result[
                    "confidence"
                ]

                melanoma_score = result[
                    "melanoma_score"
                ]

                probabilities = result[
                    "probabilities"
                ]

                low_confidence = result[
                    "low_confidence"
                ]


                band, css_class = result_style(
                    prediction,
                    confidence,
                )


                # -------------------------------------------------
                # SAVE HISTORY
                # -------------------------------------------------

                add_history(
                    uploaded.name,
                    prediction,
                    confidence,
                    melanoma_score,
                    band,
                )


                # -------------------------------------------------
                # RESULT CARD
                # -------------------------------------------------

                st.markdown(
                    f"""
                    <div class="card">

                        <div class="card-title">
                            AI Prediction
                        </div>

                        <div class="score">
                            {prediction}
                        </div>

                        <span class="tag {css_class}">
                            {band}
                        </span>

                        <br><br>

                        <div class="muted">
                            Model confidence
                        </div>

                        <div style="
                            font-size:1.5rem;
                            font-weight:900;
                        ">
                            {confidence * 100:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                st.write("")


                # -------------------------------------------------
                # CLASS-SPECIFIC MESSAGE
                # -------------------------------------------------

                if prediction == "Unsupported image":

                    st.warning(
                        "This image does not appear similar "
                        "to the supported skin-lesion classes. "
                        "Please upload a clear skin-lesion image."
                    )


                elif prediction == "Uncertain / Unsupported":

                    st.warning(
                        "The model is not confident enough "
                        "to provide a lesion classification. "
                        "Try a clearer and closer image."
                    )


                elif prediction == "Other skin lesion":

                    st.info(
                        "The image appears to represent a skin "
                        "lesion, but its learned pattern is closer "
                        "to the model's 'other lesion' category "
                        "than to benign or melanoma."
                    )


                elif prediction == "Melanoma-suspicious":

                    st.warning(
                        "The model found visual patterns that "
                        "are more similar to its melanoma "
                        "training examples. This result does "
                        "not confirm melanoma."
                    )


                elif prediction == "Benign-like":

                    st.info(
                        "The model found visual patterns that "
                        "are more similar to its benign "
                        "training examples. This result does "
                        "not rule out a medical condition."
                    )


                # -------------------------------------------------
                # PROBABILITY BREAKDOWN
                # -------------------------------------------------

                st.markdown(
                    "### Model probability breakdown"
                )


                labels = {
                    "benign":
                        "Benign-like",

                    "melanoma":
                        "Melanoma",

                    "other":
                        "Other skin lesion",

                    "non_skin":
                        "Non-skin / unsupported",
                }


                for key in [
                    "benign",
                    "melanoma",
                    "other",
                    "non_skin",
                ]:

                    value = float(
                        probabilities.get(
                            key,
                            0.0,
                        )
                    )

                    st.write(
                        f"**{labels[key]}:** "
                        f"{value * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                value,
                                1.0,
                            ),
                        )
                    )


                st.caption(
                    f"Processed image size: "
                    f"{result['image_size']} × "
                    f"{result['image_size']} pixels"
                )


                if low_confidence:

                    st.caption(
                        "Low-confidence safety rule applied."
                    )


# =========================================================
# HISTORY PAGE
# =========================================================

elif page == "History":

    st.markdown(
        "### Analysis History"
    )

    st.caption(
        "DermaSense stores the filename and model-result "
        "information only. The uploaded image itself "
        "is not stored in the history database."
    )


    history = read_history()


    if history.empty:

        st.info(
            "No analyses have been recorded yet."
        )


    else:

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Total",
            len(history),
        )


        c2.metric(
            "Benign-like",
            int(
                (
                    history["prediction"]
                    == "Benign-like"
                ).sum()
            ),
        )


        c3.metric(
            "Melanoma",
            int(
                (
                    history["prediction"]
                    == "Melanoma-suspicious"
                ).sum()
            ),
        )


        c4.metric(
            "Other / Unsupported",
            int(
                history["prediction"]
                .isin(
                    [
                        "Other skin lesion",
                        "Unsupported image",
                        "Uncertain / Unsupported",
                    ]
                )
                .sum()
            ),
        )


        display = history[
            [
                "timestamp",
                "filename",
                "prediction",
                "confidence",
                "melanoma_score",
            ]
        ].copy()


        display["confidence"] = (
            (
                display["confidence"]
                * 100
            )
            .round(1)
            .astype(str)
            + "%"
        )


        display["melanoma_score"] = (
            (
                display["melanoma_score"]
                * 100
            )
            .round(1)
            .astype(str)
            + "%"
        )


        display = display.rename(
            columns={
                "timestamp":
                    "Time",

                "filename":
                    "Image",

                "prediction":
                    "Prediction",

                "confidence":
                    "Confidence",

                "melanoma_score":
                    "Melanoma Score",
            }
        )


        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


        if st.button(
            "Clear History"
        ):

            clear_history()

            st.success(
                "History cleared."
            )

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "DermaSense AI • Custom 4-Class CNN • "
    "Trained from scratch • Educational use only"
)