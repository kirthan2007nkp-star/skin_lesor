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
):

    current_time = datetime.now()

    with sqlite3.connect(DB_PATH) as conn:

        last = conn.execute(
            """
            SELECT
                timestamp,
                filename,
                prediction,
                confidence
            FROM history
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        if last:

            try:

                previous_time = datetime.strptime(
                    last[0],
                    "%Y-%m-%d %H:%M:%S",
                )

                seconds = (
                    current_time - previous_time
                ).total_seconds()

            except ValueError:

                seconds = 999

            same_result = (
                last[1] == filename
                and last[2] == prediction
                and abs(
                    float(last[3])
                    - float(confidence)
                ) < 0.0001
            )

            if (
                same_result
                and seconds <= 10
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
                prediction,
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
        "Skin Image Classifier"
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
        Upload an image and let the custom CNN classify it as
        **Benign-like**, **Melanoma-suspicious**, or **Other**.
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

        It does not provide a medical diagnosis and should
        not replace examination by a qualified healthcare
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
                f"Model loading error: {exc}"
            )

    left, right = st.columns(
        [
            1.05,
            0.95,
        ],
        gap="large",
    )


    # =====================================================
    # UPLOAD IMAGE
    # =====================================================

    with left:

        st.header(
            "1. Upload Image"
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

                image = Image.open(
                    io.BytesIO(
                        uploaded.getvalue()
                    )
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
    # AI ANALYSIS
    # =====================================================

    with right:

        st.header(
            "2. AI Analysis"
        )

        if model is None:

            st.error(
                "The trained model is unavailable."
            )

        elif image is None:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Waiting for an image"
                )

                st.write(
                    "Upload an image on the left to begin analysis."
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

                confidence = float(
                    result["confidence"]
                )

                melanoma_score = float(
                    result["melanoma_score"]
                )

                probabilities = result[
                    "probabilities"
                ]

                # -----------------------------------------
                # SAVE HISTORY
                # -----------------------------------------

                add_history(
                    uploaded.name,
                    prediction,
                    confidence,
                    melanoma_score,
                )

                # -----------------------------------------
                # MAIN RESULT
                # -----------------------------------------

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
                # OTHER
                # =========================================

                if prediction == "Other":

                    st.info(
                        """
                        **Other image detected.**

                        This image does not match the model's
                        Benign-like or Melanoma-suspicious
                        categories.
                        """
                    )

                    # IMPORTANT:
                    # Do NOT show Benign or Melanoma
                    # percentages for an Other result.

                    st.caption(
                        f"Processed at "
                        f"{result['image_size']} × "
                        f"{result['image_size']} pixels"
                    )


                # =========================================
                # BENIGN
                # =========================================

                elif prediction == "Benign-like":

                    st.success(
                        """
                        The model found visual patterns similar
                        to its benign training examples.

                        **This is not a medical diagnosis.**
                        """
                    )

                    benign = float(
                        probabilities.get(
                            "benign",
                            0.0,
                        )
                    )

                    melanoma = float(
                        probabilities.get(
                            "melanoma",
                            0.0,
                        )
                    )

                    st.subheader(
                        "Probability Breakdown"
                    )

                    st.write(
                        f"**Benign-like:** "
                        f"{benign * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                benign,
                                1.0,
                            ),
                        )
                    )

                    st.write(
                        f"**Melanoma:** "
                        f"{melanoma * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                melanoma,
                                1.0,
                            ),
                        )
                    )

                    st.caption(
                        f"Processed at "
                        f"{result['image_size']} × "
                        f"{result['image_size']} pixels"
                    )


                # =========================================
                # MELANOMA
                # =========================================

                elif prediction == "Melanoma-suspicious":

                    st.warning(
                        """
                        The model found visual patterns similar
                        to its melanoma training examples.

                        **This does not confirm melanoma.**
                        """
                    )

                    benign = float(
                        probabilities.get(
                            "benign",
                            0.0,
                        )
                    )

                    melanoma = float(
                        probabilities.get(
                            "melanoma",
                            0.0,
                        )
                    )

                    st.subheader(
                        "Probability Breakdown"
                    )

                    st.write(
                        f"**Benign-like:** "
                        f"{benign * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                benign,
                                1.0,
                            ),
                        )
                    )

                    st.write(
                        f"**Melanoma:** "
                        f"{melanoma * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                melanoma,
                                1.0,
                            ),
                        )
                    )

                    st.caption(
                        f"Processed at "
                        f"{result['image_size']} × "
                        f"{result['image_size']} pixels"
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
        DermaSense stores the image filename and prediction
        information only. Uploaded images themselves are not
        stored in the history database.
        """
    )

    history = read_history()

    if history.empty:

        st.info(
            "No saved history."
        )

    else:

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

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
            "Other",
            int(
                (
                    history["prediction"]
                    == "Other"
                ).sum()
            ),
        )

        st.write("")

        st.subheader(
            "Saved Analyses"
        )

        # -------------------------------------------------
        # COLUMN HEADINGS
        # -------------------------------------------------

        header = st.columns(
            [
                1.8,
                2.5,
                2,
                1.3,
                0.7,
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
            "**Delete**"
        )

        st.divider()

        # -------------------------------------------------
        # HISTORY ROWS
        # -------------------------------------------------

        for _, row in history.iterrows():

            cols = st.columns(
                [
                    1.8,
                    2.5,
                    2,
                    1.3,
                    0.7,
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

            if cols[4].button(
                "🗑️",
                key=(
                    f"delete_"
                    f"{int(row['id'])}"
                ),
                help="Delete this history",
            ):

                delete_history(
                    row["id"]
                )

                st.rerun()

            st.divider()

        # -------------------------------------------------
        # CLEAR ALL
        # -------------------------------------------------

        if st.button(
            "🗑️ Clear All History",
            type="primary",
        ):

            clear_history()

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DermaSense AI • Custom CNN • "
    "Trained from scratch • Educational use only"
)