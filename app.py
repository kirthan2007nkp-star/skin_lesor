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


# =========================================================
# APP SETTINGS
# =========================================================

APP_TITLE = "DermaSense AI"

DB_PATH = (
    Path("data")
    / "analysis_history.db"
)


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
# CUSTOM THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(56, 214, 210, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 15%,
                rgba(95, 168, 255, 0.08),
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

        border-right:
            1px solid #1d3853;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stFileUploader"] {
        border:
            1px dashed #2c607a;

        border-radius: 18px;

        padding: 10px;

        background:
            rgba(10, 29, 47, 0.75);
    }

    [data-testid="stMetric"] {
        background:
            rgba(11, 23, 40, 0.90);

        border:
            1px solid #1d3853;

        padding:
            13px 15px;

        border-radius:
            16px;
    }

    div.stButton > button {
        border-radius: 14px;

        border:
            1px solid #286c7b;

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

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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


        # Prevent accidental duplicate history
        if last:

            try:

                previous_time = (
                    datetime.strptime(
                        last[0],
                        "%Y-%m-%d %H:%M:%S",
                    )
                )

                seconds = (
                    current_time
                    - previous_time
                ).total_seconds()

            except ValueError:

                seconds = 999


            same_result = (

                last[1] == filename

                and

                last[2] == prediction

                and

                abs(
                    float(last[3])
                    - float(confidence)
                ) < 0.0001
            )


            if (
                same_result
                and
                seconds <= 10
            ):

                return


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


def read_history():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        return pd.read_sql_query(
            """
            SELECT *

            FROM history

            ORDER BY id DESC

            LIMIT 100
            """,
            conn,
        )


def delete_history(
    history_id
):

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.execute(
            """
            DELETE FROM history
            """
        )

        conn.commit()


init_db()


# =========================================================
# CACHE MODEL
# =========================================================

@st.cache_resource
def get_model():

    return load_trained_model()


@st.cache_data
def get_metadata():

    return load_metadata()


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
            "Model not found"
        )


    st.caption(
        "MobileNetV2 transfer learning model."
    )

    st.caption(
        "HAM10000 skin lesion dataset."
    )

    st.caption(
        "Educational AI prototype."
    )

    st.caption(
        "Not a medical diagnostic device."
    )


# =========================================================
# MAIN HEADER
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
        Upload an image and DermaSense AI will classify it as
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


    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = None
    metadata = {}


    if MODEL_PATH.exists():

        try:

            model = get_model()

            metadata = (
                get_metadata()
            )

        except Exception as exc:

            st.error(
                f"Model loading error: {exc}"
            )


    # =====================================================
    # TWO-COLUMN LAYOUT
    # =====================================================

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
            "1. Upload Image"
        )

        st.caption(
            "Upload a clear JPG, JPEG, or PNG image."
        )


        uploaded = (
            st.file_uploader(
                "Choose image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
        )


        image = None


        if uploaded is not None:

            try:

                image = Image.open(
                    io.BytesIO(
                        uploaded.getvalue()
                    )
                ).convert(
                    "RGB"
                )


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
                    """
                    Upload an image on the left
                    to begin analysis.
                    """
                )


        else:

            analyze_button = (
                st.button(
                    "🔎 Analyze Image",
                    use_container_width=True,
                )
            )


            if analyze_button:

                with st.spinner(
                    "DermaSense AI is analyzing..."
                ):

                    result = (
                        predict_lesion(
                            model,
                            image,
                            metadata,
                        )
                    )


                prediction = (
                    result[
                        "prediction"
                    ]
                )


                confidence = float(
                    result[
                        "confidence"
                    ]
                )


                melanoma_score = float(
                    result[
                        "melanoma_score"
                    ]
                )


                probabilities = (
                    result[
                        "probabilities"
                    ]
                )


                # =========================================
                # SAVE HISTORY
                # =========================================

                add_history(
                    uploaded.name,
                    prediction,
                    confidence,
                    melanoma_score,
                )


                # =========================================
                # OTHER RESULT
                # =========================================

                if prediction == "Other":

                    with st.container(
                        border=True
                    ):

                        st.caption(
                            "AI PREDICTION"
                        )

                        st.title(
                            "Other"
                        )


                    st.info(
                        """
                        **Other image detected.**

                        This image does not match the
                        model's supported Benign-like
                        or Melanoma-suspicious categories.
                        """
                    )


                    # No confidence shown.
                    # No percentage breakdown shown.


                    st.caption(
                        f"Processed at "
                        f"{result['image_size']} × "
                        f"{result['image_size']} pixels"
                    )


                # =========================================
                # BENIGN RESULT
                # =========================================

                elif prediction == "Benign-like":

                    with st.container(
                        border=True
                    ):

                        st.caption(
                            "AI PREDICTION"
                        )

                        st.title(
                            "Benign-like"
                        )

                        st.metric(
                            "Model Confidence",
                            (
                                f"{confidence * 100:.1f}%"
                            ),
                        )


                    st.success(
                        """
                        The model found visual patterns
                        similar to its benign training
                        examples.

                        **This is not a medical diagnosis.**
                        """
                    )


                    benign_score = float(
                        probabilities.get(
                            "benign",
                            0.0,
                        )
                    )


                    melanoma_probability = float(
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
                        f"{benign_score * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                benign_score,
                                1.0,
                            ),
                        )
                    )


                    st.write(
                        f"**Melanoma:** "
                        f"{melanoma_probability * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                melanoma_probability,
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
                # MELANOMA RESULT
                # =========================================

                elif (
                    prediction
                    == "Melanoma-suspicious"
                ):

                    with st.container(
                        border=True
                    ):

                        st.caption(
                            "AI PREDICTION"
                        )

                        st.title(
                            "Melanoma-suspicious"
                        )

                        st.metric(
                            "Model Confidence",
                            (
                                f"{confidence * 100:.1f}%"
                            ),
                        )


                    st.warning(
                        """
                        The model found visual patterns
                        similar to its melanoma training
                        examples.

                        **This does not confirm melanoma.**
                        """
                    )


                    benign_score = float(
                        probabilities.get(
                            "benign",
                            0.0,
                        )
                    )


                    melanoma_probability = float(
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
                        f"{benign_score * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                benign_score,
                                1.0,
                            ),
                        )
                    )


                    st.write(
                        f"**Melanoma:** "
                        f"{melanoma_probability * 100:.1f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                melanoma_probability,
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
        View previous DermaSense AI predictions.
        Uploaded image files themselves are not
        stored in this history database.
        """
    )


    history = read_history()


    if history.empty:

        st.info(
            "No saved history."
        )


    else:

        # =================================================
        # HISTORY SUMMARY
        # =================================================

        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(
            "Total",
            len(history),
        )


        c2.metric(
            "Benign-like",
            int(
                (
                    history[
                        "prediction"
                    ]
                    == "Benign-like"
                ).sum()
            ),
        )


        c3.metric(
            "Melanoma",
            int(
                (
                    history[
                        "prediction"
                    ]
                    == "Melanoma-suspicious"
                ).sum()
            ),
        )


        c4.metric(
            "Other",
            int(
                (
                    history[
                        "prediction"
                    ]
                    == "Other"
                ).sum()
            ),
        )


        st.write("")


        st.subheader(
            "Saved Analyses"
        )


        # =================================================
        # HEADINGS
        # =================================================

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


        # =================================================
        # HISTORY RECORDS
        # =================================================

        for _, row in (
            history.iterrows()
        ):

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
                row[
                    "timestamp"
                ]
            )


            cols[1].write(
                row[
                    "filename"
                ]
            )


            cols[2].write(
                row[
                    "prediction"
                ]
            )


            # ---------------------------------------------
            # DO NOT SHOW PERCENTAGE FOR OTHER
            # ---------------------------------------------

            if (
                row[
                    "prediction"
                ]
                == "Other"
            ):

                cols[3].write(
                    "—"
                )

            else:

                cols[3].write(
                    (
                        f"{float(row['confidence']) * 100:.1f}%"
                    )
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
                    row[
                        "id"
                    ]
                )

                st.rerun()


            st.divider()


        # =================================================
        # CLEAR HISTORY
        # =================================================

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
    "DermaSense AI • MobileNetV2 • "
    "Transfer Learning • Educational use only"
)