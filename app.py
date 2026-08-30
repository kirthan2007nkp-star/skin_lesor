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
# THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(56,214,210,.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 15%,
                rgba(95,168,255,.08),
                transparent 25%
            ),
            linear-gradient(
                180deg,
                #06101d 0%,
                #081422 100%
            );
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #071321,
                #091a2b
            );

        border-right: 1px solid #1d3853;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #2c607a;
        border-radius: 18px;
        padding: 10px;
        background: rgba(10,29,47,.75);
    }

    [data-testid="stMetric"] {
        background: rgba(11,23,40,.9);
        border: 1px solid #1d3853;
        padding: 13px 15px;
        border-radius: 16px;
    }

    div.stButton > button {
        border-radius: 14px;
        border: 1px solid #286c7b;
        background: linear-gradient(
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

    </style>
    """,
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔬 DermaSense AI")

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

    st.divider()

    if MODEL_PATH.exists():

        st.success(
            "✅ Trained model detected"
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
# HEADER
# =========================================================

with st.container(border=True):

    st.caption(
        "AI-POWERED SKIN IMAGE ANALYSIS"
    )

    st.title(
        "Skin Lesion Classifier"
    )

    st.write(
        """
        Upload an image and let our custom CNN,
        trained from scratch, classify it as
        **Benign-like**, **Melanoma-suspicious**,
        **Other skin lesion**, or
        **Unsupported image**.
        """
    )


# =========================================================
# ANALYZE
# =========================================================

if page == "Analyze":

    st.warning(
        """
        **Important:** DermaSense AI is an educational
        machine-learning prototype.

        Its output is not a medical diagnosis and should
        not replace evaluation by a qualified healthcare
        professional.
        """
    )

    st.write("")

    model = None
    metadata = {}

    if MODEL_PATH.exists():

        try:

            model = load_trained_model()
            metadata = load_metadata()

        except Exception as exc:

            st.error(
                f"Model could not be loaded: {exc}"
            )


    left, right = st.columns(
        [1.05, 0.95],
        gap="large",
    )


    # =====================================================
    # UPLOAD
    # =====================================================

    with left:

        st.header(
            "1. Upload image"
        )

        st.caption(
            "Upload a clear JPG, JPEG, or PNG image."
        )

        uploaded = st.file_uploader(
            "Choose image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            accept_multiple_files=False,
            label_visibility="collapsed",
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
                    "The uploaded file is not a valid image."
                )


    # =====================================================
    # ANALYSIS
    # =====================================================

    with right:

        st.header(
            "2. AI Analysis"
        )

        if model is None:

            st.error(
                "The trained CNN model is not available."
            )


        elif image is None:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Waiting for an image"
                )

                st.write(
                    """
                    Upload an image on the left.

                    The **Analyze Image** button will
                    appear here.
                    """
                )


        else:

            if st.button(
                "🔎 Analyze Image",
                use_container_width=True,
            ):

                with st.spinner(
                    "DermaSense AI is analyzing..."
                ):

                    result = predict_lesion(
                        model,
                        image,
                        metadata,
                    )


                prediction = result[
                    "prediction"
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


                # =========================================
                # HISTORY LABEL
                # =========================================

                if prediction == "Benign-like":

                    history_band = (
                        "Benign-like pattern"
                    )

                elif prediction == (
                    "Melanoma-suspicious"
                ):

                    history_band = (
                        "Suspicious model pattern"
                    )

                elif prediction == (
                    "Other skin lesion"
                ):

                    history_band = (
                        "Other lesion detected"
                    )

                elif prediction == (
                    "Unsupported image"
                ):

                    history_band = (
                        "Unsupported image"
                    )

                else:

                    history_band = (
                        "Low-confidence result"
                    )


                add_history(
                    uploaded.name,
                    prediction,
                    confidence,
                    melanoma_score,
                    history_band,
                )


                # =========================================
                # RESULT
                # =========================================

                with st.container(
                    border=True
                ):

                    st.caption(
                        "AI PREDICTION"
                    )

                    st.title(
                        prediction
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence * 100:.1f}%",
                    )


                # =========================================
                # MESSAGE
                # =========================================

                if prediction == (
                    "Unsupported image"
                ):

                    st.warning(
                        """
                        This image appears to be outside
                        the supported skin-lesion classes.

                        Please upload a clear skin-lesion
                        image.
                        """
                    )


                elif prediction == (
                    "Uncertain / Unsupported"
                ):

                    st.warning(
                        """
                        The model is not confident enough
                        to give a reliable classification.

                        Try a clearer or closer image.
                        """
                    )


                elif prediction == (
                    "Other skin lesion"
                ):

                    st.info(
                        """
                        The image appears closer to the
                        model's **other skin lesion**
                        category than to benign or melanoma.
                        """
                    )


                elif prediction == (
                    "Melanoma-suspicious"
                ):

                    st.warning(
                        """
                        The model found patterns similar
                        to its melanoma training examples.

                        **This does not confirm melanoma.**
                        """
                    )


                elif prediction == (
                    "Benign-like"
                ):

                    st.success(
                        """
                        The model found patterns similar
                        to its benign training examples.

                        **This does not rule out a medical
                        condition.**
                        """
                    )


                # =========================================
                # PROBABILITIES
                # =========================================

                st.subheader(
                    "Probability Breakdown"
                )


                labels = {
                    "benign":
                        "Benign-like",

                    "melanoma":
                        "Melanoma",

                    "other":
                        "Other skin lesion",

                    "non_skin":
                        "Unsupported / Non-skin",
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
                    f"Processed at "
                    f"{result['image_size']} × "
                    f"{result['image_size']} pixels"
                )


                if low_confidence:

                    st.caption(
                        "⚠️ Low-confidence safety rule applied."
                    )


# =========================================================
# HISTORY
# =========================================================

elif page == "History":

    st.header(
        "Analysis History"
    )

    st.caption(
        """
        DermaSense stores the filename and prediction
        information only.

        Uploaded images themselves are not stored
        in the history database.
        """
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
            "🗑️ Clear History"
        ):

            clear_history()

            st.success(
                "History cleared."
            )

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DermaSense AI • Custom 4-Class CNN • "
    "Trained from scratch • Educational use only"
)