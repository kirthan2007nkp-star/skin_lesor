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

from gradcam import create_gradcam_result


# =========================================================
# SETTINGS
# =========================================================

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
# DESIGN
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
            rgba(10,29,47,.75);
    }

    [data-testid="stMetric"] {
        background:
            rgba(11,23,40,.9);

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


        # Prevent duplicate history
        if last:

            try:

                previous_time = datetime.strptime(
                    last[0],
                    "%Y-%m-%d %H:%M:%S",
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
                and seconds <= 10
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
# LOAD MODEL
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
        "Smart Skin Image Analysis"
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


    st.markdown(
        "### ✨ Explainable ML"
    )

    st.caption(
        "⚡ Multi-image analysis"
    )

    st.caption(
        "🧾 Prediction history"
    )


# =========================================================
# HEADER
# =========================================================

with st.container(
    border=True
):

    st.caption(
        "EXPLAINABLE MACHINE LEARNING FOR SKIN IMAGE ANALYSIS"
    )

    st.title(
        "DermaSense AI"
    )

    st.write(
        """
        Upload one or multiple images and DermaSense AI
        classifies each image as **Benign-like**,
        **Melanoma-suspicious**, or **Other**.

        For supported lesion predictions, the system also
        explains why the model produced that result.
        """
    )


# =========================================================
# ANALYZE PAGE
# =========================================================

if page == "Analyze":

    st.warning(
        """
        **Educational use only.**

        DermaSense AI does not provide a medical diagnosis
        and should not replace assessment by a qualified
        healthcare professional.
        """
    )


    model = None
    metadata = {}


    if MODEL_PATH.exists():

        try:

            model = get_model()
            metadata = get_metadata()

        except Exception as exc:

            st.error(
                f"Unable to load model: {exc}"
            )


    # =====================================================
    # UPLOAD
    # =====================================================

    st.header(
        "1. Upload Images"
    )


    st.caption(
        """
        Upload one image or select multiple images
        for batch analysis.
        """
    )


    uploaded_files = st.file_uploader(
        "Choose images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )


    valid_images = []


    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) selected"
        )


        preview_columns = st.columns(
            min(
                len(uploaded_files),
                4,
            )
        )


        for index, uploaded in enumerate(
            uploaded_files
        ):

            try:

                image = Image.open(
                    io.BytesIO(
                        uploaded.getvalue()
                    )
                ).convert(
                    "RGB"
                )


                valid_images.append(
                    (
                        uploaded,
                        image,
                    )
                )


                with preview_columns[
                    index
                    % len(
                        preview_columns
                    )
                ]:

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
                    f"{uploaded.name} is not a valid image."
                )


    # =====================================================
    # ANALYZE
    # =====================================================

    if (
        valid_images
        and model is not None
    ):

        if st.button(
            "🔎 Analyze Selected Images",
            use_container_width=True,
        ):

            st.divider()

            st.header(
                "2. Analysis Results"
            )


            for image_number, (
                uploaded,
                image,
            ) in enumerate(
                valid_images,
                start=1,
            ):

                with st.container(
                    border=True
                ):

                    st.subheader(
                        f"Image {image_number}: "
                        f"{uploaded.name}"
                    )


                    image_col, result_col = st.columns(
                        [
                            0.9,
                            1.1,
                        ],
                        gap="large",
                    )


                    # =====================================
                    # IMAGE
                    # =====================================

                    with image_col:

                        st.image(
                            image,
                            use_container_width=True,
                        )


                    # =====================================
                    # PREDICTION
                    # =====================================

                    with result_col:

                        with st.spinner(
                            "Analyzing..."
                        ):

                            result = predict_lesion(
                                model,
                                image,
                                metadata,
                            )


                        prediction = (
                            result["prediction"]
                        )


                        confidence = float(
                            result["confidence"]
                        )


                        melanoma_score = float(
                            result["melanoma_score"]
                        )


                        probabilities = (
                            result["probabilities"]
                        )


                        add_history(
                            uploaded.name,
                            prediction,
                            confidence,
                            melanoma_score,
                        )


                        # =================================
                        # OTHER
                        # =================================

                        if prediction == "Other":

                            st.caption(
                                "ML PREDICTION"
                            )


                            st.title(
                                "Other"
                            )


                            st.info(
                                """
                                The uploaded image does not
                                match the model's supported
                                Benign-like or
                                Melanoma-suspicious categories.
                                """
                            )


                            with st.expander(
                                "💡 Why this prediction?"
                            ):

                                st.write(
                                    """
                                    DermaSense includes internal
                                    categories for **other skin
                                    lesions** and **non-skin
                                    images**.

                                    The combined response from
                                    those categories was stronger
                                    than the Benign and Melanoma
                                    responses.

                                    Therefore, the final result is
                                    shown as **Other**.
                                    """
                                )


                        # =================================
                        # BENIGN / MELANOMA
                        # =================================

                        else:

                            st.caption(
                                "ML PREDICTION"
                            )


                            st.title(
                                prediction
                            )


                            st.metric(
                                "Model Confidence",
                                f"{confidence * 100:.1f}%",
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


                            # ---------------------------------
                            # BENIGN MESSAGE
                            # ---------------------------------

                            if (
                                prediction
                                == "Benign-like"
                            ):

                                st.success(
                                    """
                                    The model found visual
                                    patterns more similar to
                                    benign examples from its
                                    training data.

                                    **This is not a medical
                                    diagnosis.**
                                    """
                                )


                            # ---------------------------------
                            # MELANOMA MESSAGE
                            # ---------------------------------

                            elif (
                                prediction
                                ==
                                "Melanoma-suspicious"
                            ):

                                st.warning(
                                    """
                                    The model found visual
                                    patterns more similar to
                                    melanoma examples from its
                                    training data.

                                    **This does not confirm
                                    melanoma.**
                                    """
                                )


                            # =================================
                            # SCORES
                            # =================================

                            st.markdown(
                                "#### Prediction Scores"
                            )


                            score1, score2 = (
                                st.columns(2)
                            )


                            score1.metric(
                                "Benign-like",
                                f"{benign_score * 100:.1f}%",
                            )


                            score2.metric(
                                "Melanoma",
                                f"{melanoma_probability * 100:.1f}%",
                            )


                            # =================================
                            # WHY THIS PREDICTION
                            # =================================

                            with st.expander(
                                "💡 Why this prediction?",
                                expanded=True,
                            ):

                                if (
                                    prediction
                                    == "Benign-like"
                                ):

                                    st.write(
                                        """
                                        The model's
                                        **Benign-like response**
                                        was stronger than its
                                        Melanoma response.

                                        The neural network
                                        recognized visual patterns
                                        that were more similar to
                                        benign examples learned
                                        during training.
                                        """
                                    )


                                else:

                                    st.write(
                                        """
                                        The model's
                                        **Melanoma-suspicious
                                        response** was stronger
                                        than its Benign-like
                                        response.

                                        The neural network
                                        recognized visual patterns
                                        that were more similar to
                                        melanoma examples learned
                                        during training.
                                        """
                                    )


                                # =============================
                                # GRAD-CAM
                                # =============================

                                class_names = metadata.get(
                                    "class_names",
                                    [
                                        "benign",
                                        "melanoma",
                                        "non_skin",
                                        "other_skin",
                                    ],
                                )


                                if (
                                    prediction
                                    == "Benign-like"
                                ):

                                    target_class = "benign"

                                else:

                                    target_class = "melanoma"


                                try:

                                    class_index = (
                                        class_names.index(
                                            target_class
                                        )
                                    )


                                    with st.spinner(
                                        "Generating explanation map..."
                                    ):

                                        gradcam_result = (
                                            create_gradcam_result(
                                                model=model,
                                                image=image,
                                                class_index=class_index,
                                                image_size=int(
                                                    metadata.get(
                                                        "image_size",
                                                        224,
                                                    )
                                                ),
                                            )
                                        )


                                    st.markdown(
                                        "#### 🔥 AI Attention Map"
                                    )


                                    st.caption(
                                        """
                                        The highlighted region
                                        shows which part of the
                                        image had more influence
                                        on this model prediction.
                                        """
                                    )


                                    grad_original, grad_attention = (
                                        st.columns(2)
                                    )


                                    with grad_original:

                                        st.markdown(
                                            "**Original**"
                                        )

                                        st.image(
                                            image,
                                            use_container_width=True,
                                        )


                                    with grad_attention:

                                        st.markdown(
                                            "**Model Attention**"
                                        )

                                        st.image(
                                            gradcam_result[
                                                "overlay"
                                            ],
                                            use_container_width=True,
                                        )


                                    st.caption(
                                        """
                                        This Grad-CAM visualization
                                        explains model attention.
                                        It does not identify or
                                        medically localize cancer.
                                        """
                                    )


                                except Exception:

                                    st.caption(
                                        """
                                        Prediction completed.
                                        Attention visualization
                                        unavailable for this image.
                                        """
                                    )


                st.write("")


# =========================================================
# HISTORY PAGE
# =========================================================

elif page == "History":

    st.header(
        "Analysis History"
    )


    st.caption(
        """
        Review previous DermaSense predictions.
        Uploaded image files themselves are not
        stored in the history database.
        """
    )


    history = read_history()


    if history.empty:

        st.info(
            "No saved analysis history."
        )


    else:

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


        heading = st.columns(
            [
                1.7,
                2.3,
                2,
                1.2,
                0.6,
            ]
        )


        heading[0].markdown(
            "**Time**"
        )

        heading[1].markdown(
            "**Image**"
        )

        heading[2].markdown(
            "**Prediction**"
        )

        heading[3].markdown(
            "**Confidence**"
        )

        heading[4].markdown(
            "**Delete**"
        )


        st.divider()


        for _, row in history.iterrows():

            cols = st.columns(
                [
                    1.7,
                    2.3,
                    2,
                    1.2,
                    0.6,
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


            # Other -> no confidence shown
            if (
                row["prediction"]
                == "Other"
            ):

                cols[3].write(
                    "—"
                )


            else:

                cols[3].write(
                    f"{float(row['confidence']) * 100:.1f}%"
                )


            if cols[4].button(
                "🗑️",
                key=(
                    f"delete_"
                    f"{int(row['id'])}"
                ),
                help="Delete this result",
            ):

                delete_history(
                    row["id"]
                )

                st.rerun()


            st.divider()


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
    "DermaSense AI • "
    "Machine Learning • "
    "Explainable Image Analysis"
)