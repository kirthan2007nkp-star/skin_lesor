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

        background:
            linear-gradient(
                135deg,
                #126c79,
                #245a96
            );

        color: white;
        font-weight: 800;
        min-height: 42px;
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
    """
    Adds a history record.

    If the exact same result was stored in the
    last few seconds, it will not be stored again.
    """

    current_time = datetime.now()

    with sqlite3.connect(DB_PATH) as conn:

        last_record = conn.execute(
            """
            SELECT
                timestamp,
                filename,
                prediction,
                confidence,
                melanoma_score
            FROM history
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()


        if last_record is not None:

            (
                last_timestamp,
                last_filename,
                last_prediction,
                last_confidence,
                last_melanoma_score,
            ) = last_record


            try:

                previous_time = datetime.strptime(
                    last_timestamp,
                    "%Y-%m-%d %H:%M:%S",
                )

                seconds_difference = (
                    current_time - previous_time
                ).total_seconds()

            except ValueError:

                seconds_difference = 999


            same_result = (
                last_filename == filename
                and last_prediction == prediction
                and abs(
                    float(last_confidence)
                    - float(confidence)
                ) < 0.0001
                and abs(
                    float(last_melanoma_score)
                    - float(melanoma_score)
                ) < 0.0001
            )


            if (
                same_result
                and seconds_difference <= 10
            ):

                return False


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
                current_time.strftime(
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

    return True


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


def delete_history(history_id):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            DELETE FROM history
            WHERE id = ?
            """,
            (
                int(history_id),
            ),
        )

        conn.commit()


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

    st.title(
        "🔬 DermaSense AI"
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

with st.container(
    border=True
):

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
# ANALYZE PAGE
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
                f"Model could not be loaded: {exc}"
            )


    left, right = st.columns(
        [
            1.05,
            0.95,
        ],
        gap="large",
    )


    # =====================================================
    # IMAGE UPLOAD
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


                history_saved = add_history(
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
                # CLASS MESSAGE
                # =========================================

                if prediction == (
                    "Unsupported image"
                ):

                    st.warning(
                        """
                        This image appears outside the
                        supported skin-lesion classes.

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
                        to provide a reliable classification.

                        Try a clearer or closer skin image.
                        """
                    )


                elif prediction == (
                    "Other skin lesion"
                ):

                    st.info(
                        """
                        The image appears closer to the
                        model's **Other skin lesion**
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
                # PROBABILITY BREAKDOWN
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
# HISTORY PAGE
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

        # =================================================
        # SUMMARY
        # =================================================

        c1, c2, c3, c4 = st.columns(
            4
        )


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


        st.write("")


        # =================================================
        # HISTORY LIST
        # =================================================

        st.subheader(
            "Saved Analyses"
        )


        header = st.columns(
            [
                1.8,
                2.3,
                2,
                1.2,
                1.2,
                0.8,
            ]
        )


        header[0].markdown(
            "**Time**"
        )

        header[1].markdown(
            "**Image**"
        )

        header[2].markdown(
            "**Prediction**"
        )

        header[3].markdown(
            "**Confidence**"
        )

        header[4].markdown(
            "**Melanoma**"
        )

        header[5].markdown(
            "**Action**"
        )


        st.divider()


        for _, row in history.iterrows():

            cols = st.columns(
                [
                    1.8,
                    2.3,
                    2,
                    1.2,
                    1.2,
                    0.8,
                ]
            )


            cols[0].write(
                row["timestamp"]
            )


            cols[1].write(
                row["filename"]
            )


            cols[2].write(
                row["prediction"]
            )


            cols[3].write(
                f"{float(row['confidence']) * 100:.1f}%"
            )


            cols[4].write(
                f"{float(row['melanoma_score']) * 100:.1f}%"
            )


            if cols[5].button(
                "🗑️",
                key=f"delete_history_{int(row['id'])}",
                help="Delete this history record",
            ):

                delete_history(
                    int(row["id"])
                )

                st.rerun()


            st.divider()


        # =================================================
        # CLEAR ALL
        # =================================================

        st.write("")


        if st.button(
            "🗑️ Clear All History",
            type="primary",
        ):

            clear_history()

            st.success(
                "All history records deleted."
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