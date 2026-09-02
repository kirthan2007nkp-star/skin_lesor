from pathlib import Path
import base64
import html
import io
import sqlite3
import time
from datetime import datetime

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    MODEL_PATH,
    load_metadata,
    load_trained_model,
    predict_lesion,
)

from gradcam import create_gradcam_result
from dermaguide import dermaguide_reply


# =========================================================
# SETTINGS
# =========================================================

APP_TITLE = "DermaSense AI"

DB_PATH = Path("data") / "analysis_history.db"

PROCESSING_SECONDS = 5.8


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
# SESSION STATE
# =========================================================

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "latest_probabilities" not in st.session_state:
    st.session_state.latest_probabilities = {}

if "latest_filename" not in st.session_state:
    st.session_state.latest_filename = None

if "dermaguide_messages" not in st.session_state:
    st.session_state.dermaguide_messages = []


# =========================================================
# GLOBAL DESIGN
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 12% 4%,
                rgba(38, 208, 218, .07),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(107, 85, 255, .07),
                transparent 27%
            ),
            linear-gradient(
                180deg,
                #050e1a 0%,
                #071625 100%
            );
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #04111f,
                #071929
            );

        border-right:
            1px solid #17344a;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    [data-testid="stFileUploader"] {
        border:
            1px dashed rgba(64, 206, 232, .42);

        border-radius: 18px;
        padding: 8px;

        background:
            rgba(7, 28, 46, .72);
    }

    [data-testid="stCameraInput"] {
        border:
            1px solid rgba(64, 206, 232, .34);

        border-radius: 18px;
        padding: 8px;

        background:
            rgba(7, 28, 46, .72);
    }

    [data-testid="stMetric"] {
        background:
            rgba(8, 28, 46, .88);

        border:
            1px solid #173a52;

        padding: 12px 14px;
        border-radius: 15px;
    }

    div.stButton > button {
        border-radius: 13px;

        border:
            1px solid rgba(58, 208, 229, .45);

        background:
            linear-gradient(
                135deg,
                #0e7284,
                #285c9f
            );

        color: white;
        font-weight: 750;
        min-height: 43px;
    }

    div.stButton > button:hover {
        border-color: #4cecff;
        color: white;
    }

    .derma-result-chip {
        display: inline-block;

        padding: 7px 13px;

        border:
            1px solid rgba(64, 222, 255, .42);

        border-radius: 18px;

        color: #5eeaff;

        background:
            rgba(23, 92, 119, .22);

        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# IMAGE -> BASE64
# =========================================================

def image_to_base64(image):

    img = image.copy()

    img.thumbnail(
        (
            520,
            520,
        )
    )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=91,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# =========================================================
# PROCESSING ANIMATION
# =========================================================

def render_processing_animation(
    image,
    filename,
):

    image_data = image_to_base64(image)

    safe_filename = html.escape(
        str(filename)
    )

    animation = """
<!DOCTYPE html>

<html>

<head>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    overflow: hidden;
    background: transparent;

    font-family:
        Inter,
        Arial,
        Helvetica,
        sans-serif;
}


/* ======================================================
   ROOT
   ====================================================== */

.root {
    width: 100%;
    height: 365px;

    display: flex;
    align-items: center;
    justify-content: center;
}


/* ======================================================
   CARD
   ====================================================== */

.card {
    position: relative;

    width: 830px;
    max-width: 98%;

    height: 335px;

    overflow: hidden;

    border-radius: 25px;

    background:
        radial-gradient(
            circle at 18% 30%,
            rgba(41,213,231,.13),
            transparent 40%
        ),
        radial-gradient(
            circle at 89% 30%,
            rgba(176,78,255,.10),
            transparent 36%
        ),
        linear-gradient(
            145deg,
            #051625,
            #08243a
        );

    border:
        1px solid rgba(59,216,241,.30);

    box-shadow:
        0 20px 60px
        rgba(0,0,0,.34);
}


/* ======================================================
   HEADER
   ====================================================== */

.header {
    height: 57px;

    display: flex;
    align-items: center;

    padding:
        0 21px;

    border-bottom:
        1px solid rgba(110,175,205,.14);
}

.live {
    width: 10px;
    height: 10px;

    margin-right: 10px;

    border-radius: 50%;

    background:
        #45f4bb;

    box-shadow:
        0 0 14px #45f4bb;

    animation:
        livePulse 1s infinite;
}

.title {
    color:
        #f1f9ff;

    font-weight: 750;
    font-size: 16px;
}

.live-label {
    margin-left: auto;

    color:
        #6d8ea5;

    font-size: 10px;
    letter-spacing: 1.2px;
}


/* ======================================================
   IMAGE
   ====================================================== */

.scene-container {
    position: absolute;

    left: 34px;
    top: 78px;

    width: 300px;
    height: 220px;

    display: flex;
    align-items: center;
    justify-content: center;

    perspective: 850px;
}

.scene {
    position: relative;

    width: 225px;
    height: 185px;

    transform-style: preserve-3d;

    animation:
        sceneMotion 5.8s ease-in-out forwards;
}

.shadow-plane {
    position: absolute;

    left: 15px;
    top: 17px;

    width: 200px;
    height: 157px;

    border-radius: 17px;

    background:
        rgba(21,63,83,.34);

    border:
        1px solid rgba(56,215,239,.17);
}

.back1 {
    transform:
        translateZ(-38px)
        translate(16px,13px);
}

.back2 {
    transform:
        translateZ(-19px)
        translate(8px,6px);
}

.image-plane {
    position: absolute;

    left: 15px;
    top: 17px;

    width: 200px;
    height: 157px;

    border-radius: 17px;

    overflow: hidden;

    transform:
        translateZ(28px);

    border:
        1px solid rgba(68,226,246,.60);

    box-shadow:
        0 16px 34px rgba(0,0,0,.38),
        0 0 25px rgba(50,219,243,.17);
}

.image-plane img {
    width: 100%;
    height: 100%;

    object-fit: cover;

    animation:
        imageProcess 5.8s linear forwards;
}


/* ======================================================
   GRID
   ====================================================== */

.grid {
    position: absolute;
    inset: 0;

    opacity: 0;

    background-image:
        linear-gradient(
            rgba(60,224,244,.15) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(60,224,244,.15) 1px,
            transparent 1px
        );

    background-size:
        22px 22px;

    animation:
        gridAppear 5.8s linear forwards;
}


/* ======================================================
   SCAN
   ====================================================== */

.scan {
    position: absolute;

    left: -5px;
    top: 4px;

    width: 210px;
    height: 5px;

    z-index: 20;

    background:
        linear-gradient(
            90deg,
            transparent,
            #3ee8ff,
            white,
            #ff64c2,
            transparent
        );

    box-shadow:
        0 0 17px
        rgba(55,230,255,.94);

    animation:
        scanMove 1.25s
        ease-in-out infinite alternate;
}


/* ======================================================
   FEATURES
   ====================================================== */

.feature {
    position: absolute;

    width: 9px;
    height: 9px;

    border-radius: 50%;

    background:
        #42e8ff;

    box-shadow:
        0 0 13px #42e8ff;

    opacity: 0;

    z-index: 30;

    animation:
        featureAppear
        5.8s linear forwards;
}

.f1 {
    left: 37%;
    top: 36%;
}

.f2 {
    left: 56%;
    top: 43%;
}

.f3 {
    left: 45%;
    top: 62%;
}

.f4 {
    left: 68%;
    top: 56%;
}

.f5 {
    left: 29%;
    top: 67%;
}

.filename {
    position: absolute;

    bottom: -6px;

    width: 100%;

    text-align: center;

    color:
        #7893a7;

    font-size: 9px;

    overflow: hidden;

    white-space: nowrap;

    text-overflow: ellipsis;
}


/* ======================================================
   PIPELINE
   ====================================================== */

.pipeline {
    position: absolute;

    left: 365px;
    right: 24px;

    top: 78px;

    height: 220px;

    padding: 15px;

    border-radius: 18px;

    background:
        rgba(4,26,44,.78);

    border:
        1px solid rgba(90,155,185,.17);
}

.pipeline-title {
    margin-bottom: 11px;

    color:
        #eaf7ff;

    font-size: 14px;
    font-weight: 700;
}

.step {
    height: 30px;

    display: flex;
    align-items: center;

    margin-bottom: 5px;

    padding:
        0 10px;

    color:
        #718b9f;

    border-radius: 9px;

    background:
        rgba(8,39,61,.64);

    border:
        1px solid rgba(100,150,180,.12);

    font-size: 10px;
}

.number {
    width: 20px;
    height: 20px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-right: 8px;

    border-radius: 50%;

    border:
        1px solid rgba(67,220,245,.25);
}

.step-icon {
    width: 25px;
    margin-right: 5px;
}

.done {
    margin-left: auto;

    color:
        #4affbb;

    opacity: 0;
}


/* ======================================================
   STAGES
   ====================================================== */

.s1 {
    animation:
        stage1 5.8s linear forwards;
}

.s2 {
    animation:
        stage2 5.8s linear forwards;
}

.s3 {
    animation:
        stage3 5.8s linear forwards;
}

.s4 {
    animation:
        stage4 5.8s linear forwards;
}

.s5 {
    animation:
        stage5 5.8s linear forwards;
}


.s1 .done {
    animation:
        done1 5.8s linear forwards;
}

.s2 .done {
    animation:
        done2 5.8s linear forwards;
}

.s3 .done {
    animation:
        done3 5.8s linear forwards;
}

.s4 .done {
    animation:
        done4 5.8s linear forwards;
}

.s5 .done {
    animation:
        done5 5.8s linear forwards;
}


/* ======================================================
   PROGRESS
   ====================================================== */

.progress {
    height: 7px;

    margin-top: 11px;

    border-radius: 20px;

    overflow: hidden;

    background:
        #102a3d;
}

.progress-bar {
    height: 100%;
    width: 0;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #40e6ff,
            #8a78ff,
            #ff5fbd
        );

    animation:
        progress 5.8s linear forwards;
}

.status {
    position: relative;

    height: 22px;

    margin-top: 7px;

    text-align: center;

    color:
        #809caf;

    font-size: 9px;
}

.msg {
    position: absolute;

    width: 100%;

    opacity: 0;
}

.m1 {
    animation:
        msg1 5.8s linear forwards;
}

.m2 {
    animation:
        msg2 5.8s linear forwards;
}

.m3 {
    animation:
        msg3 5.8s linear forwards;
}

.m4 {
    animation:
        msg4 5.8s linear forwards;
}

.m5 {
    animation:
        msg5 5.8s linear forwards;
}


/* ======================================================
   KEYFRAMES
   ====================================================== */

@keyframes livePulse {
    0%,
    100% {
        opacity: .4;
        transform: scale(.75);
    }

    50% {
        opacity: 1;
        transform: scale(1.35);
    }
}


@keyframes sceneMotion {

    0% {
        transform:
            rotateX(3deg)
            rotateY(-12deg);
    }

    22% {
        transform:
            rotateX(-5deg)
            rotateY(13deg);
    }

    45% {
        transform:
            rotateX(6deg)
            rotateY(-9deg)
            scale(1.04);
    }

    67% {
        transform:
            rotateX(-3deg)
            rotateY(11deg)
            scale(1.04);
    }

    85% {
        transform:
            rotateX(2deg)
            rotateY(-5deg);
    }

    100% {
        transform:
            rotateX(0deg)
            rotateY(0deg);
    }
}


@keyframes scanMove {

    from {
        top: 5px;
    }

    to {
        top: 147px;
    }
}


@keyframes imageProcess {

    0%,
    20% {
        filter: none;
    }

    25%,
    42% {
        filter:
            brightness(1.09)
            contrast(1.08);
    }

    46%,
    67% {
        filter:
            contrast(1.23)
            saturate(.80);
    }

    72%,
    88% {
        filter:
            contrast(1.13);
    }

    92%,
    100% {
        filter: none;
    }
}


@keyframes gridAppear {

    0%,
    30% {
        opacity: 0;
    }

    38%,
    70% {
        opacity: .82;
    }

    82%,
    100% {
        opacity: .12;
    }
}


@keyframes featureAppear {

    0%,
    37% {
        opacity: 0;
        transform: scale(.3);
    }

    46%,
    72% {
        opacity: 1;
        transform: scale(1.35);
    }

    82%,
    100% {
        opacity: .15;
        transform: scale(.7);
    }
}


@keyframes progress {

    0% {
        width: 3%;
    }

    20% {
        width: 20%;
    }

    40% {
        width: 42%;
    }

    60% {
        width: 64%;
    }

    80% {
        width: 84%;
    }

    100% {
        width: 100%;
    }
}


/* ======================================================
   ACTIVE STEPS
   ====================================================== */

@keyframes stage1 {

    0%,
    18% {
        color: white;
        border-color: #41e9ff;
        background:
            rgba(29,134,168,.22);
    }

    20%,
    100% {
        color: #9ab2c1;
    }
}


@keyframes stage2 {

    0%,
    19% {
        opacity: .45;
    }

    21%,
    38% {
        opacity: 1;
        color: white;
        border-color: #4ebeff;
        background:
            rgba(35,112,174,.20);
    }

    40%,
    100% {
        color: #9ab2c1;
    }
}


@keyframes stage3 {

    0%,
    39% {
        opacity: .45;
    }

    41%,
    58% {
        opacity: 1;
        color: white;
        border-color: #8b72ff;
        background:
            rgba(101,71,185,.19);
    }

    60%,
    100% {
        color: #9ab2c1;
    }
}


@keyframes stage4 {

    0%,
    59% {
        opacity: .45;
    }

    61%,
    78% {
        opacity: 1;
        color: white;
        border-color: #d760ff;
        background:
            rgba(151,60,184,.18);
    }

    80%,
    100% {
        color: #9ab2c1;
    }
}


@keyframes stage5 {

    0%,
    79% {
        opacity: .45;
    }

    81%,
    100% {
        opacity: 1;
        color: white;
        border-color: #ff65ad;
        background:
            rgba(185,58,122,.18);
    }
}


/* ======================================================
   DONE
   ====================================================== */

@keyframes done1 {
    0%, 19% { opacity: 0; }
    20%, 100% { opacity: 1; }
}

@keyframes done2 {
    0%, 39% { opacity: 0; }
    40%, 100% { opacity: 1; }
}

@keyframes done3 {
    0%, 59% { opacity: 0; }
    60%, 100% { opacity: 1; }
}

@keyframes done4 {
    0%, 79% { opacity: 0; }
    80%, 100% { opacity: 1; }
}

@keyframes done5 {
    0%, 96% { opacity: 0; }
    97%, 100% { opacity: 1; }
}


/* ======================================================
   MESSAGES
   ====================================================== */

@keyframes msg1 {
    0%, 18% { opacity: 1; }
    20%, 100% { opacity: 0; }
}

@keyframes msg2 {
    0%, 19% { opacity: 0; }
    21%, 38% { opacity: 1; }
    40%, 100% { opacity: 0; }
}

@keyframes msg3 {
    0%, 39% { opacity: 0; }
    41%, 58% { opacity: 1; }
    60%, 100% { opacity: 0; }
}

@keyframes msg4 {
    0%, 59% { opacity: 0; }
    61%, 78% { opacity: 1; }
    80%, 100% { opacity: 0; }
}

@keyframes msg5 {
    0%, 79% { opacity: 0; }
    81%, 100% { opacity: 1; }
}

</style>

</head>


<body>


<div class="root">


<div class="card">


<div class="header">

<div class="live"></div>

<div class="title">
DermaSense ML Analysis
</div>

<div class="live-label">
LIVE PROCESSING
</div>

</div>


<div class="scene-container">


<div class="scene">


<div class="shadow-plane back1"></div>

<div class="shadow-plane back2"></div>


<div class="image-plane">


<img
src="data:image/jpeg;base64,__IMAGE__"
/>


<div class="grid"></div>

<div class="scan"></div>

<div class="feature f1"></div>
<div class="feature f2"></div>
<div class="feature f3"></div>
<div class="feature f4"></div>
<div class="feature f5"></div>


</div>


<div class="filename">

__FILENAME__

</div>


</div>


</div>


<div class="pipeline">


<div class="pipeline-title">
Machine-Learning Pipeline
</div>


<div class="step s1">

<div class="number">
1
</div>

<div class="step-icon">
🖼️
</div>

Image input

<div class="done">
✓
</div>

</div>


<div class="step s2">

<div class="number">
2
</div>

<div class="step-icon">
⚙️
</div>

Preprocessing & resize

<div class="done">
✓
</div>

</div>


<div class="step s3">

<div class="number">
3
</div>

<div class="step-icon">
🧠
</div>

MobileNetV2 feature extraction

<div class="done">
✓
</div>

</div>


<div class="step s4">

<div class="number">
4
</div>

<div class="step-icon">
◈
</div>

Class response calculation

<div class="done">
✓
</div>

</div>


<div class="step s5">

<div class="number">
5
</div>

<div class="step-icon">
✨
</div>

Prediction selection

<div class="done">
✓
</div>

</div>


<div class="progress">

<div class="progress-bar"></div>

</div>


<div class="status">

<div class="msg m1">
Reading uploaded image...
</div>

<div class="msg m2">
Preparing 224 × 224 model input...
</div>

<div class="msg m3">
Extracting learned visual features...
</div>

<div class="msg m4">
Calculating class responses...
</div>

<div class="msg m5">
Prediction ready.
</div>

</div>


</div>


</div>


</div>


</body>

</html>
    """

    animation = animation.replace(
        "__IMAGE__",
        image_data,
    )

    animation = animation.replace(
        "__FILENAME__",
        safe_filename,
    )

    components.html(
        animation,
        height=375,
        scrolling=False,
    )


# =========================================================
# PROFESSIONAL 3D VISUAL EXPLANATION
# =========================================================

def render_3d_prediction_explainer(
    image,
    filename,
    prediction,
    confidence,
    probabilities,
):

    image_data = image_to_base64(
        image
    )

    safe_prediction = html.escape(
        str(prediction)
    )


    benign_score = float(
        probabilities.get(
            "benign",
            0.0,
        )
    )


    melanoma_score = float(
        probabilities.get(
            "melanoma",
            0.0,
        )
    )


    other_score = float(
        probabilities.get(
            "other",
            0.0,
        )
    )


    # =====================================================
    # PREDICTION STYLE
    # =====================================================

    if prediction == "Melanoma-suspicious":

        accent = "#ff5d9e"
        accent_soft = "rgba(255,93,158,.18)"

        result_short = (
            "Melanoma response was strongest"
        )

        result_reason = (
            "MobileNetV2 extracted learned visual features "
            "from the uploaded image. The final classifier "
            "produced a stronger melanoma response than the "
            "Benign-like response."
        )


    elif prediction == "Benign-like":

        accent = "#45e4c4"
        accent_soft = "rgba(69,228,196,.17)"

        result_short = (
            "Benign-like response was strongest"
        )

        result_reason = (
            "MobileNetV2 extracted learned visual features "
            "from the uploaded image. The final classifier "
            "produced a stronger Benign-like response than "
            "the melanoma response."
        )


    else:

        accent = "#55dfff"
        accent_soft = "rgba(85,223,255,.16)"

        result_short = (
            "Other response was strongest"
        )

        result_reason = (
            "The model's combined response for other-skin "
            "and non-skin patterns was stronger than its "
            "supported Benign-like and melanoma responses."
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if prediction == "Other":

        confidence_html = """
        <div class="other-tag">
            Rejection / Other category
        </div>
        """

    else:

        confidence_html = f"""
        <div class="confidence-row">

            <div>

                <div class="confidence-label">
                    Model confidence
                </div>

                <div class="confidence-value">
                    {confidence * 100:.1f}%
                </div>

            </div>


            <div class="confidence-ring">

                <svg viewBox="0 0 42 42">

                    <circle
                    class="ring-bg"
                    cx="21"
                    cy="21"
                    r="16"
                    ></circle>

                    <circle
                    class="ring-value"
                    cx="21"
                    cy="21"
                    r="16"
                    pathLength="100"
                    style="
                        stroke-dasharray:
                        {confidence * 100:.1f}
                        100;
                    "
                    ></circle>

                </svg>

                <span>
                    AI
                </span>

            </div>

        </div>
        """


    # =====================================================
    # SCORES
    # =====================================================

    if prediction == "Other":

        score_html = f"""
        <div class="response-box">

            <div class="response-title">
                Model response
            </div>


            <div class="response-line">

                <span>
                    Other
                </span>

                <div class="bar-track">

                    <div
                    class="bar other-bar"
                    style="
                        width:
                        {other_score * 100:.1f}%;
                    "
                    ></div>

                </div>

                <b>
                    strongest
                </b>

            </div>


            <div class="other-explain">

                DermaSense intentionally does not show a
                confidence percentage to users when the
                final prediction is Other.

            </div>

        </div>
        """

    else:

        score_html = f"""
        <div class="response-box">

            <div class="response-title">
                Class responses
            </div>


            <div class="response-line">

                <span>
                    Benign-like
                </span>

                <div class="bar-track">

                    <div
                    class="bar benign-bar"
                    style="
                        width:
                        {benign_score * 100:.1f}%;
                    "
                    ></div>

                </div>

                <b>
                    {benign_score * 100:.1f}%
                </b>

            </div>


            <div class="response-line">

                <span>
                    Melanoma
                </span>

                <div class="bar-track">

                    <div
                    class="bar melanoma-bar"
                    style="
                        width:
                        {melanoma_score * 100:.1f}%;
                    "
                    ></div>

                </div>

                <b>
                    {melanoma_score * 100:.1f}%
                </b>

            </div>


            <div class="response-line">

                <span>
                    Other
                </span>

                <div class="bar-track">

                    <div
                    class="bar other-bar"
                    style="
                        width:
                        {other_score * 100:.1f}%;
                    "
                    ></div>

                </div>

                <b>
                    {other_score * 100:.1f}%
                </b>

            </div>

        </div>
        """


    visual = """
<!DOCTYPE html>

<html>

<head>

<meta
name="viewport"
content="width=device-width, initial-scale=1"
/>


<style>

* {
    box-sizing: border-box;
}


body {
    margin: 0;

    background:
        transparent;

    overflow:
        hidden;

    font-family:
        Inter,
        Arial,
        Helvetica,
        sans-serif;
}


/* ======================================================
   ROOT
   ====================================================== */

.root {
    width: 100%;
    height: 530px;

    display: flex;
    justify-content: center;
    align-items: center;
}


/* ======================================================
   DASHBOARD
   ====================================================== */

.dashboard {
    position: relative;

    width: 1030px;
    max-width: 99%;

    height: 500px;

    overflow: hidden;

    border-radius: 25px;

    background:
        radial-gradient(
            circle at 12% 30%,
            rgba(27,214,235,.09),
            transparent 32%
        ),
        radial-gradient(
            circle at 86% 20%,
            rgba(155,77,255,.08),
            transparent 35%
        ),
        linear-gradient(
            145deg,
            #051522,
            #071f33
        );

    border:
        1px solid
        rgba(69,208,233,.28);

    box-shadow:
        0 24px 70px
        rgba(0,0,0,.32);
}


/* ======================================================
   HEADER
   ====================================================== */

.header {
    height: 62px;

    display: flex;
    align-items: center;

    padding:
        0 22px;

    border-bottom:
        1px solid
        rgba(90,155,185,.13);
}


.brand-icon {
    width: 31px;
    height: 31px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-right: 10px;

    border-radius: 10px;

    color:
        #48e9ff;

    background:
        rgba(28,111,140,.20);

    border:
        1px solid
        rgba(69,221,246,.28);
}


.header-title {
    color:
        #f1f8fd;

    font-size: 16px;
    font-weight: 760;
}


.header-sub {
    color:
        #65869b;

    font-size: 8px;

    margin-top: 2px;
}


.badges {
    margin-left: auto;

    display: flex;

    gap: 7px;
}


.badge {
    padding:
        5px 9px;

    border-radius: 20px;

    color:
        #86a5b8;

    font-size: 7px;

    border:
        1px solid
        rgba(85,157,190,.18);

    background:
        rgba(8,38,59,.55);
}


.live-dot {
    display: inline-block;

    width: 6px;
    height: 6px;

    margin-right: 5px;

    border-radius: 50%;

    background:
        #4ff6bd;

    box-shadow:
        0 0 7px
        #4ff6bd;

    animation:
        pulse 1.2s infinite;
}


/* ======================================================
   PIPELINE
   ====================================================== */

.pipeline {
    position: absolute;

    left: 18px;
    right: 18px;
    top: 75px;

    height: 71px;

    display: grid;

    grid-template-columns:
        1fr 34px
        1fr 34px
        1fr 34px
        1fr 34px
        1fr;

    align-items: center;

    padding:
        7px 11px;

    border-radius: 16px;

    background:
        rgba(5,28,45,.69);

    border:
        1px solid
        rgba(70,155,188,.16);
}


.pipe-step {
    height: 53px;

    display: flex;
    align-items: center;

    padding:
        7px 9px;

    border-radius: 11px;

    transition:
        .25s ease;
}


.pipe-step:hover {
    background:
        rgba(22,74,101,.30);
}


.circle {
    width: 34px;
    height: 34px;

    min-width: 34px;

    margin-right: 8px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    color:
        #53eaff;

    font-size: 12px;
    font-weight: 750;

    border:
        1px solid
        rgba(70,226,250,.55);

    background:
        rgba(16,89,116,.18);
}


.step-title {
    color:
        #dcebf4;

    font-weight: 700;
    font-size: 9px;
}


.step-small {
    color:
        #5e7e92;

    font-size: 7px;

    margin-top: 3px;
}


.arrow {
    text-align: center;

    color:
        #477c96;

    font-size: 18px;
}


/* ======================================================
   CONTENT
   ====================================================== */

.content {
    position: absolute;

    left: 18px;
    right: 18px;

    top: 159px;
    bottom: 18px;

    display: grid;

    grid-template-columns:
        1.28fr
        .78fr
        1.13fr;

    gap: 12px;
}


/* ======================================================
   COMMON PANEL
   ====================================================== */

.panel {
    position: relative;

    overflow: hidden;

    border-radius: 17px;

    background:
        linear-gradient(
            180deg,
            rgba(5,29,47,.89),
            rgba(5,25,41,.74)
        );

    border:
        1px solid
        rgba(73,150,181,.16);
}


.panel-header {
    height: 41px;

    display: flex;
    align-items: center;

    padding:
        0 13px;

    color:
        #dfeef7;

    font-size: 10px;
    font-weight: 700;
}


.panel-header-icon {
    width: 23px;
    height: 23px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-right: 7px;

    border-radius: 7px;

    background:
        rgba(32,112,141,.18);

    color:
        #4de7ff;
}


/* ======================================================
   INTERACTIVE 3D IMAGE
   ====================================================== */

.three-panel {
    background:
        radial-gradient(
            circle at 50% 48%,
            rgba(36,132,167,.14),
            transparent 50%
        ),
        rgba(5,27,44,.82);
}


.scene-area {
    position: relative;

    height: 230px;

    display: flex;
    align-items: center;
    justify-content: center;

    perspective: 950px;

    cursor: grab;

    user-select: none;
}


.scene-area:active {
    cursor: grabbing;
}


/* FLOOR */

.floor {
    position: absolute;

    width: 275px;
    height: 145px;

    top: 96px;

    transform:
        rotateX(72deg);

    opacity: .30;

    background-image:
        linear-gradient(
            rgba(51,160,185,.26) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(51,160,185,.26) 1px,
            transparent 1px
        );

    background-size:
        22px 22px;
}


/* MODEL */

.model {
    position: relative;

    width: 250px;
    height: 185px;

    transform-style:
        preserve-3d;

    transform:
        rotateX(53deg)
        rotateY(-8deg)
        rotateZ(-4deg);

    transition:
        transform .14s ease-out;

    animation:
        autoFloat 7s
        ease-in-out infinite;
}


/* DEPTH */

.depth {
    position: absolute;

    left: 12px;

    width: 225px;

    border-radius:
        7px 7px 14px 14px;

    box-shadow:
        0 12px 23px
        rgba(0,0,0,.20);
}


.layer-one {
    top: 127px;
    height: 21px;

    transform:
        translateZ(29px);

    background:
        linear-gradient(
            90deg,
            #e99a88,
            #d17372
        );
}


.layer-two {
    top: 141px;
    height: 29px;

    transform:
        translateZ(13px);

    background:
        linear-gradient(
            90deg,
            #c55e71,
            #9c4f68
        );
}


.layer-three {
    top: 162px;
    height: 23px;

    transform:
        translateZ(-3px);

    background:
        linear-gradient(
            90deg,
            #e1a844,
            #b97824
        );
}


/* IMAGE FACE */

.image-face {
    position: absolute;

    left: 6px;
    top: 1px;

    width: 237px;
    height: 155px;

    overflow: hidden;

    border-radius: 16px;

    transform:
        translateZ(44px);

    border:
        1px solid
        rgba(65,225,246,.55);

    box-shadow:
        0 16px 32px
        rgba(0,0,0,.32),
        0 0 20px
        rgba(50,215,238,.14);
}


.image-face img {
    width: 100%;
    height: 100%;

    object-fit: cover;
}


/* SCANNER */

.scanner {
    position: absolute;

    left: 0;
    top: 5px;

    width: 100%;
    height: 5px;

    z-index: 20;

    background:
        linear-gradient(
            90deg,
            transparent,
            #3ce8ff,
            white,
            __ACCENT__,
            transparent
        );

    box-shadow:
        0 0 17px
        rgba(58,226,255,.86);

    animation:
        scanner 1.7s
        ease-in-out infinite alternate;
}


/* SCAN CORNERS */

.scan-corner {
    position: absolute;

    width: 22px;
    height: 22px;

    z-index: 25;
}


.tl {
    top: 5px;
    left: 5px;

    border-top:
        2px solid #43e9ff;

    border-left:
        2px solid #43e9ff;
}


.tr {
    top: 5px;
    right: 5px;

    border-top:
        2px solid __ACCENT__;

    border-right:
        2px solid __ACCENT__;
}


.bl {
    bottom: 5px;
    left: 5px;

    border-bottom:
        2px solid __ACCENT__;

    border-left:
        2px solid __ACCENT__;
}


.br {
    bottom: 5px;
    right: 5px;

    border-bottom:
        2px solid #43e9ff;

    border-right:
        2px solid #43e9ff;
}


/* ======================================================
   CONTROLS
   ====================================================== */

.controls {
    height: 39px;

    display: flex;
    align-items: center;
    justify-content: center;

    gap: 6px;

    border-top:
        1px solid
        rgba(73,139,168,.11);
}


.control {
    min-width: 62px;

    padding:
        6px 9px;

    cursor: pointer;

    color:
        #91acbd;

    background:
        rgba(9,40,60,.66);

    border:
        1px solid
        rgba(70,180,208,.18);

    border-radius: 8px;

    font-size: 7px;
}


.control:hover {
    color: white;

    border-color:
        #48e7ff;
}


.model-caption {
    padding:
        3px 15px 0 15px;

    text-align: center;

    color:
        #527389;

    font-size: 6.5px;

    line-height: 1.35;
}


/* ======================================================
   ANALYSIS PANEL
   ====================================================== */

.analysis-content {
    padding:
        2px 13px 10px 13px;
}


.analysis-step {
    position: relative;

    min-height: 54px;

    display: flex;
    align-items: center;

    padding: 7px;

    margin-bottom: 8px;

    border-radius: 11px;

    background:
        rgba(8,37,58,.55);

    border:
        1px solid
        rgba(76,138,166,.12);
}


.analysis-icon {
    width: 35px;
    height: 35px;

    min-width: 35px;

    margin-right: 9px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 10px;

    color:
        #51e7ff;

    background:
        rgba(29,103,133,.20);
}


.analysis-name {
    color:
        #d9e8f1;

    font-size: 8px;
    font-weight: 700;
}


.analysis-desc {
    color:
        #637f92;

    font-size: 6.5px;

    margin-top: 3px;
}


/* FEATURE MAP */

.feature-map {
    height: 49px;

    margin-top: 5px;

    padding: 7px;

    border-radius: 10px;

    background:
        linear-gradient(
            135deg,
            rgba(24,106,151,.25),
            rgba(103,62,174,.22)
        );
}


.feature-line {
    height: 3px;

    margin-bottom: 4px;

    border-radius: 6px;

    background:
        linear-gradient(
            90deg,
            #44e5ff,
            #7b79ff,
            #e95fff
        );

    animation:
        featurePulse
        1.4s ease-in-out infinite;
}


.fa {
    width: 80%;
}

.fb {
    width: 59%;
}

.fc {
    width: 91%;
}

.fd {
    width: 68%;
}


/* ======================================================
   RESULT PANEL
   ====================================================== */

.result-content {
    padding:
        2px 13px 13px 13px;
}


.prediction-card {
    position: relative;

    padding: 13px;

    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            __SOFT__,
            rgba(6,33,52,.70)
        );

    border:
        1px solid __ACCENT__;
}


.prediction-label {
    color:
        #708da0;

    font-size: 7px;
    letter-spacing: 1px;
}


.prediction-value {
    margin-top: 5px;

    color:
        __ACCENT__;

    font-size: 18px;
    font-weight: 800;
}


/* CONFIDENCE */

.confidence-row {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-top: 10px;
}


.confidence-label {
    color:
        #718da0;

    font-size: 7px;
}


.confidence-value {
    margin-top: 2px;

    color:
        #f0f7fb;

    font-size: 15px;
    font-weight: 750;
}


.confidence-ring {
    position: relative;

    width: 45px;
    height: 45px;
}


.confidence-ring svg {
    width: 45px;
    height: 45px;

    transform:
        rotate(-90deg);
}


.confidence-ring circle {
    fill: none;
    stroke-width: 4;
}


.ring-bg {
    stroke:
        #123146;
}


.ring-value {
    stroke:
        __ACCENT__;

    stroke-linecap:
        round;
}


.confidence-ring span {
    position: absolute;

    inset: 0;

    display: flex;
    align-items: center;
    justify-content: center;

    color:
        #8aa9bb;

    font-size: 7px;
}


.other-tag {
    margin-top: 10px;

    display: inline-block;

    padding:
        5px 8px;

    border-radius: 12px;

    color:
        #61e7ff;

    font-size: 7px;

    border:
        1px solid
        rgba(80,221,255,.25);
}


/* WHY */

.reason-box {
    margin-top: 11px;

    padding: 9px;

    border-radius: 11px;

    background:
        rgba(7,30,47,.62);

    border:
        1px solid
        rgba(79,139,168,.13);
}


.reason-head {
    color:
        #d9e8f1;

    font-size: 8px;
    font-weight: 700;
}


.reason-highlight {
    margin-top: 6px;

    color:
        __ACCENT__;

    font-size: 8px;
    font-weight: 700;
}


.reason-copy {
    margin-top: 4px;

    color:
        #718b9d;

    font-size: 6.6px;

    line-height: 1.5;
}


/* RESPONSE */

.response-box {
    margin-top: 10px;

    padding: 9px;

    border-radius: 11px;

    background:
        rgba(5,25,40,.55);

    border:
        1px solid
        rgba(71,135,163,.13);
}


.response-title {
    color:
        #d7e6ef;

    font-size: 8px;
    font-weight: 700;

    margin-bottom: 8px;
}


.response-line {
    display: grid;

    grid-template-columns:
        62px
        1fr
        36px;

    gap: 6px;

    align-items: center;

    margin-top: 7px;

    color:
        #748fa1;

    font-size: 6.5px;
}


.response-line b {
    color:
        #d8e7ef;

    text-align: right;

    font-size: 6.5px;
}


.bar-track {
    height: 6px;

    overflow: hidden;

    border-radius: 10px;

    background:
        #102c40;
}


.bar {
    height: 100%;

    border-radius: 10px;
}


.benign-bar {
    background:
        linear-gradient(
            90deg,
            #40d7ba,
            #56ebcc
        );
}


.melanoma-bar {
    background:
        linear-gradient(
            90deg,
            #ff6b84,
            #ff5cae
        );
}


.other-bar {
    background:
        linear-gradient(
            90deg,
            #3baed0,
            #55ddff
        );
}


.other-explain {
    margin-top: 7px;

    color:
        #617f92;

    line-height: 1.4;

    font-size: 6.5px;
}


/* ======================================================
   ANIMATION
   ====================================================== */

@keyframes pulse {

    0%,
    100% {
        opacity: .45;
        transform: scale(.75);
    }

    50% {
        opacity: 1;
        transform: scale(1.25);
    }
}


@keyframes autoFloat {

    0%,
    100% {
        translate: 0 0;
    }

    50% {
        translate: 0 -5px;
    }
}


@keyframes scanner {

    from {
        top: 6px;
    }

    to {
        top: 144px;
    }
}


@keyframes featurePulse {

    0%,
    100% {
        opacity: .5;

        transform:
            scaleX(.72);

        transform-origin:
            left;
    }

    50% {
        opacity: 1;

        transform:
            scaleX(1);

        transform-origin:
            left;
    }
}


/* ======================================================
   MOBILE
   ====================================================== */

@media (max-width: 720px) {

    .root {
        height: 930px;
    }

    .dashboard {
        height: 900px;
    }

    .pipeline {
        display: none;
    }

    .content {
        top: 74px;

        grid-template-columns:
            1fr;

        overflow-y: auto;
    }
}

</style>

</head>


<body>


<div class="root">


<div class="dashboard">


<!-- HEADER -->

<div class="header">


<div class="brand-icon">
⬡
</div>


<div>

<div class="header-title">

DermaSense AI Analysis

</div>


<div class="header-sub">

Explainable visual prediction workflow

</div>

</div>


<div class="badges">


<div class="badge">

<span class="live-dot"></span>

Analysis complete

</div>


<div class="badge">

MobileNetV2

</div>


<div class="badge">

Explainable AI

</div>


</div>


</div>


<!-- PIPELINE -->

<div class="pipeline">


<div class="pipe-step">

<div class="circle">
1
</div>

<div>

<div class="step-title">
Image Input
</div>

<div class="step-small">
Uploaded image
</div>

</div>

</div>


<div class="arrow">
›
</div>


<div class="pipe-step">

<div class="circle">
2
</div>

<div>

<div class="step-title">
Preprocessing
</div>

<div class="step-small">
224 × 224 input
</div>

</div>

</div>


<div class="arrow">
›
</div>


<div class="pipe-step">

<div class="circle">
3
</div>

<div>

<div class="step-title">
Feature Encoder
</div>

<div class="step-small">
MobileNetV2
</div>

</div>

</div>


<div class="arrow">
›
</div>


<div class="pipe-step">

<div class="circle">
4
</div>

<div>

<div class="step-title">
Class Response
</div>

<div class="step-small">
Compare outputs
</div>

</div>

</div>


<div class="arrow">
›
</div>


<div class="pipe-step">

<div class="circle">
5
</div>

<div>

<div class="step-title">
Prediction
</div>

<div class="step-small">
Strongest response
</div>

</div>

</div>


</div>


<!-- CONTENT -->

<div class="content">


<!-- INTERACTIVE IMAGE -->

<div class="panel three-panel">


<div class="panel-header">

<div class="panel-header-icon">
⬡
</div>

Interactive Image View

</div>


<div
class="scene-area"
id="sceneArea"
>


<div class="floor"></div>


<div
class="model"
id="model"
>


<div class="depth layer-three"></div>

<div class="depth layer-two"></div>

<div class="depth layer-one"></div>


<div class="image-face">


<img
src="data:image/jpeg;base64,__IMAGE__"
alt="Uploaded image"
/>


<div class="scanner"></div>


<div class="scan-corner tl"></div>

<div class="scan-corner tr"></div>

<div class="scan-corner bl"></div>

<div class="scan-corner br"></div>


</div>


</div>


</div>


<div class="controls">


<button
class="control"
onclick="rotateLeft()"
>
↶ Rotate
</button>


<button
class="control"
onclick="resetView()"
>
Reset
</button>


<button
class="control"
onclick="rotateRight()"
>
Rotate ↷
</button>


<button
class="control"
onclick="zoomView()"
>
＋ Zoom
</button>


</div>


<div class="model-caption">

Drag to rotate the uploaded image visualization.
The depth effect is illustrative and is not reconstructed anatomy.

</div>


</div>


<!-- MODEL ANALYSIS -->

<div class="panel">


<div class="panel-header">

<div class="panel-header-icon">
◎
</div>

Model Analysis

</div>


<div class="analysis-content">


<div class="analysis-step">


<div class="analysis-icon">
01
</div>


<div>

<div class="analysis-name">
Image preparation
</div>

<div class="analysis-desc">
Resize image to 224 × 224 pixels.
</div>

</div>


</div>


<div class="analysis-step">


<div class="analysis-icon">
02
</div>


<div style="width:100%">


<div class="analysis-name">
Feature extraction
</div>


<div class="analysis-desc">

MobileNetV2 converts the image into learned visual features.

</div>


<div class="feature-map">

<div class="feature-line fa"></div>

<div class="feature-line fb"></div>

<div class="feature-line fc"></div>

<div class="feature-line fd"></div>

</div>


</div>


</div>


<div class="analysis-step">


<div class="analysis-icon">
03
</div>


<div>

<div class="analysis-name">
Classification
</div>

<div class="analysis-desc">
Final layers calculate class responses.
</div>

</div>


</div>


<div class="analysis-step">


<div class="analysis-icon">
✓
</div>


<div>

<div class="analysis-name">
Result generated
</div>

<div class="analysis-desc">
Strongest response becomes the displayed result.
</div>

</div>


</div>


</div>


</div>


<!-- RESULT -->

<div class="panel">


<div class="panel-header">

<div class="panel-header-icon">
◉
</div>

Prediction Summary

</div>


<div class="result-content">


<div class="prediction-card">


<div class="prediction-label">
MODEL RESULT
</div>


<div class="prediction-value">

__PREDICTION__

</div>


__CONFIDENCE__


</div>


<div class="reason-box">


<div class="reason-head">

Why this prediction?

</div>


<div class="reason-highlight">

__SHORT_REASON__

</div>


<div class="reason-copy">

__REASON__

</div>


</div>


__SCORES__


</div>


</div>


</div>


</div>


</div>


<script>

let rotateY = -8;

let rotateX = 53;

let zoom = 1;

let dragging = false;

let lastX = 0;

let lastY = 0;


const model =
    document.getElementById(
        "model"
    );


const scene =
    document.getElementById(
        "sceneArea"
    );


function updateModel() {

    model.style.animation =
        "none";


    model.style.transform =
        "rotateX("
        + rotateX
        + "deg) "
        +
        "rotateY("
        + rotateY
        + "deg) "
        +
        "rotateZ(-4deg) "
        +
        "scale("
        + zoom
        + ")";
}


function rotateLeft() {

    rotateY -= 18;

    updateModel();
}


function rotateRight() {

    rotateY += 18;

    updateModel();
}


function resetView() {

    rotateY = -8;

    rotateX = 53;

    zoom = 1;

    updateModel();
}


function zoomView() {

    if (
        zoom < 1.18
    ) {

        zoom += .08;
    }

    else {

        zoom = 1;
    }

    updateModel();
}


scene.addEventListener(
    "mousedown",
    function(event) {

        dragging = true;

        lastX =
            event.clientX;

        lastY =
            event.clientY;

        updateModel();
    }
);


window.addEventListener(
    "mouseup",
    function() {

        dragging = false;
    }
);


window.addEventListener(
    "mousemove",
    function(event) {

        if (
            !dragging
        ) {

            return;
        }


        const dx =
            event.clientX
            -
            lastX;


        const dy =
            event.clientY
            -
            lastY;


        rotateY +=
            dx
            * .35;


        rotateX -=
            dy
            * .20;


        rotateX =
            Math.max(
                25,
                Math.min(
                    72,
                    rotateX
                )
            );


        lastX =
            event.clientX;


        lastY =
            event.clientY;


        updateModel();
    }
);


scene.addEventListener(
    "touchstart",
    function(event) {

        if (
            event.touches.length
            !== 1
        ) {

            return;
        }


        dragging = true;


        lastX =
            event.touches[0].clientX;


        lastY =
            event.touches[0].clientY;


        updateModel();
    }
);


scene.addEventListener(
    "touchmove",
    function(event) {

        if (
            !dragging
        ) {

            return;
        }


        const dx =
            event.touches[0].clientX
            -
            lastX;


        const dy =
            event.touches[0].clientY
            -
            lastY;


        rotateY +=
            dx
            * .35;


        rotateX -=
            dy
            * .20;


        rotateX =
            Math.max(
                25,
                Math.min(
                    72,
                    rotateX
                )
            );


        lastX =
            event.touches[0].clientX;


        lastY =
            event.touches[0].clientY;


        updateModel();
    }
);


scene.addEventListener(
    "touchend",
    function() {

        dragging = false;
    }
);

</script>


</body>

</html>
    """


    visual = visual.replace(
        "__IMAGE__",
        image_data,
    )


    visual = visual.replace(
        "__PREDICTION__",
        safe_prediction,
    )


    visual = visual.replace(
        "__CONFIDENCE__",
        confidence_html,
    )


    visual = visual.replace(
        "__SHORT_REASON__",
        html.escape(
            result_short
        ),
    )


    visual = visual.replace(
        "__REASON__",
        html.escape(
            result_reason
        ),
    )


    visual = visual.replace(
        "__SCORES__",
        score_html,
    )


    visual = visual.replace(
        "__ACCENT__",
        accent,
    )


    visual = visual.replace(
        "__SOFT__",
        accent_soft,
    )


    components.html(
        visual,
        height=540,
        scrolling=False,
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

    now = datetime.now()

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


        if last:

            try:

                previous_time = datetime.strptime(
                    last[0],
                    "%Y-%m-%d %H:%M:%S",
                )

                seconds = (
                    now - previous_time
                ).total_seconds()

            except ValueError:

                seconds = 999


            duplicate = (
                last[1] == filename

                and
                last[2] == prediction

                and
                abs(
                    float(last[3])
                    -
                    float(confidence)
                )
                < 0.0001
            )


            if (
                duplicate
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
                now.strftime(
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
    history_id,
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
            "DELETE FROM history"
        )

        conn.commit()


init_db()


# =========================================================
# MODEL CACHE
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
            "DermaGuide AI",
        ],
        label_visibility="collapsed",
    )


    st.divider()


    st.markdown(
        "### ✨ Smart Features"
    )

    st.caption(
        "📷 Live camera capture"
    )

    st.caption(
        "⚡ Multi-image analysis"
    )

    st.caption(
        "⬡ Interactive 3D visualization"
    )

    st.caption(
        "🔥 Grad-CAM explainability"
    )

    st.caption(
        "🤖 DermaGuide assistant"
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
        Upload a skin image or capture one using your camera.
        DermaSense analyzes the image using a
        **MobileNetV2 transfer-learning model** and displays
        **Benign-like**, **Melanoma-suspicious**, or **Other**.
        """
    )


# =========================================================
# ANALYZE PAGE
# =========================================================

if page == "Analyze":

    st.warning(
        """
        **Educational ML prototype — not a medical diagnosis.**
        Concerning skin changes should be assessed by a
        qualified healthcare professional.
        """
    )


    # =====================================================
    # MODEL
    # =====================================================

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
    # INPUT
    # =====================================================

    st.header(
        "1. Add Image"
    )


    input_mode = st.radio(
        "Choose input method",
        [
            "📁 Upload Images",
            "📷 Use Camera",
        ],
        horizontal=True,
    )


    valid_images = []


    # =====================================================
    # UPLOAD
    # =====================================================

    if input_mode == "📁 Upload Images":

        uploaded_files = st.file_uploader(
            "Choose skin image(s)",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )


        if uploaded_files:

            st.success(
                f"✅ {len(uploaded_files)} image(s) selected"
            )


            preview_columns = st.columns(
                min(
                    len(uploaded_files),
                    5,
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
                            uploaded.name,
                            image,
                        )
                    )


                    with preview_columns[
                        index
                        %
                        len(preview_columns)
                    ]:

                        st.image(
                            image,
                            caption=uploaded.name,
                            width=145,
                        )


                except (
                    UnidentifiedImageError,
                    OSError,
                ):

                    st.error(
                        f"{uploaded.name} is not a valid image."
                    )


    # =====================================================
    # CAMERA
    # =====================================================

    else:

        captured_file = st.camera_input(
            "Capture Image",
            label_visibility="collapsed",
        )


        if captured_file is not None:

            try:

                camera_image = Image.open(
                    io.BytesIO(
                        captured_file.getvalue()
                    )
                ).convert(
                    "RGB"
                )


                capture_name = (
                    "camera_"
                    +
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    +
                    ".jpg"
                )


                valid_images.append(
                    (
                        capture_name,
                        camera_image,
                    )
                )


                st.success(
                    "✅ Image captured"
                )


                st.image(
                    camera_image,
                    caption="Captured image",
                    width=210,
                )


            except (
                UnidentifiedImageError,
                OSError,
            ):

                st.error(
                    "Unable to process captured image."
                )


    # =====================================================
    # ANALYZE
    # =====================================================

    if (
        valid_images
        and
        model is not None
    ):

        if len(valid_images) == 1:

            button_label = (
                "🔎 Analyze Image"
            )

        else:

            button_label = (
                f"🔎 Analyze {len(valid_images)} Images"
            )


        if st.button(
            button_label,
            use_container_width=True,
        ):

            results = []


            # =================================================
            # PROCESS EACH IMAGE
            # =================================================

            for (
                filename,
                image,
            ) in valid_images:

                processing_holder = st.empty()


                with processing_holder.container():

                    render_processing_animation(
                        image=image,
                        filename=filename,
                    )


                # REAL MODEL PREDICTION
                result = predict_lesion(
                    model,
                    image,
                    metadata,
                )


                time.sleep(
                    PROCESSING_SECONDS
                )


                processing_holder.empty()


                results.append(
                    (
                        filename,
                        image,
                        result,
                    )
                )


            # =================================================
            # RESULTS
            # =================================================

            st.header(
                "2. Analysis Result"
            )


            for number, (
                filename,
                image,
                result,
            ) in enumerate(
                results,
                start=1,
            ):


                prediction = result[
                    "prediction"
                ]


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


                probabilities = result[
                    "probabilities"
                ]


                # =============================================
                # SAVE LATEST FOR CHATBOT
                # =============================================

                st.session_state.latest_prediction = (
                    prediction
                )

                st.session_state.latest_probabilities = (
                    probabilities
                )

                st.session_state.latest_filename = (
                    filename
                )


                # =============================================
                # HISTORY
                # =============================================

                add_history(
                    filename,
                    prediction,
                    confidence,
                    melanoma_score,
                )


                # =============================================
                # RESULT CARD
                # =============================================

                with st.container(
                    border=True
                ):


                    if len(results) > 1:

                        st.subheader(
                            f"Image {number}"
                        )


                    col_img, col_result = st.columns(
                        [
                            .68,
                            1.32,
                        ],
                        gap="large",
                    )


                    with col_img:

                        st.image(
                            image,
                            caption=filename,
                            width=250,
                        )


                    with col_result:

                        st.caption(
                            "MODEL PREDICTION"
                        )

                        st.title(
                            prediction
                        )


                        if prediction == "Other":

                            st.info(
                                """
                                The model's combined
                                **Other / rejection response**
                                was stronger than the supported
                                Benign-like and
                                Melanoma-suspicious responses.
                                """
                            )


                        else:

                            st.metric(
                                "Model Confidence",
                                f"{confidence * 100:.1f}%",
                            )


                            if prediction == "Benign-like":

                                st.success(
                                    """
                                    The strongest supported
                                    model response was
                                    **Benign-like**.
                                    """
                                )

                            else:

                                st.warning(
                                    """
                                    The strongest supported
                                    model response was
                                    **Melanoma-suspicious**.

                                    This does not confirm melanoma.
                                    """
                                )


                # =============================================
                # VISUAL EXPLANATION
                # =============================================

                st.markdown(
                    "### ⬡ Visual Prediction Explanation"
                )


                render_3d_prediction_explainer(
                    image=image,
                    filename=filename,
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                )


                # =============================================
                # DETAILED EXPLANATION
                # =============================================

                with st.expander(
                    "💡 Detailed prediction explanation"
                ):


                    if prediction == "Benign-like":

                        st.write(
                            """
                            The uploaded image is resized to the
                            model's 224 × 224 input size.

                            MobileNetV2 then converts the image
                            into learned visual features.

                            The final classification layers
                            produced a stronger **Benign-like**
                            response than the melanoma response.

                            This explains the machine-learning
                            output only and is not a medical diagnosis.
                            """
                        )


                    elif prediction == "Melanoma-suspicious":

                        st.write(
                            """
                            The uploaded image is resized to the
                            model's 224 × 224 input size.

                            MobileNetV2 then extracts learned visual
                            features from the image.

                            The final classifier produced a stronger
                            **melanoma response** than the Benign-like
                            response.

                            DermaSense therefore displays
                            **Melanoma-suspicious**. This does not
                            confirm melanoma.
                            """
                        )


                    else:

                        st.write(
                            """
                            DermaSense internally includes categories
                            for other skin patterns and non-skin
                            patterns.

                            Their combined response was stronger than
                            the supported benign and melanoma
                            responses.

                            Therefore the app displays **Other**
                            instead of forcing the image into one
                            of the supported lesion categories.
                            """
                        )


                # =============================================
                # GRAD-CAM
                # =============================================

                if prediction != "Other":

                    st.markdown(
                        "### 🔥 AI Attention Map"
                    )


                    st.caption(
                        """
                        Grad-CAM highlights image regions that
                        influenced the neural-network prediction.
                        """
                    )


                    class_names = metadata.get(
                        "class_names",
                        [
                            "benign",
                            "melanoma",
                            "non_skin",
                            "other_skin",
                        ],
                    )


                    if prediction == "Benign-like":

                        target_class = "benign"

                    else:

                        target_class = "melanoma"


                    try:

                        class_index = class_names.index(
                            target_class
                        )


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


                        left_space, original_col, attention_col, right_space = (
                            st.columns(
                                [
                                    .15,
                                    1,
                                    1,
                                    .15,
                                ]
                            )
                        )


                        with original_col:

                            st.markdown(
                                "**Original**"
                            )

                            st.image(
                                image,
                                use_container_width=True,
                            )


                        with attention_col:

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
                            Grad-CAM visualizes model attention only.
                            It does not medically identify or
                            localize cancer.
                            """
                        )


                    except Exception:

                        st.caption(
                            """
                            Prediction completed successfully,
                            but the attention map is unavailable.
                            """
                        )


                st.success(
                    """
                    🤖 **Need more information?**

                    Open **DermaGuide AI** to ask why the result
                    occurred, what it generally means, possible
                    effects, risk-reduction information, treatment
                    information, or what to do next.
                    """
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
        Prediction metadata is saved locally.
        Uploaded image files themselves are not stored.
        """
    )


    history = read_history()


    if history.empty:

        st.info(
            "No analysis history yet."
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


        headings = st.columns(
            [
                1.7,
                2.3,
                2,
                1.2,
                .6,
            ]
        )


        headings[0].markdown(
            "**Time**"
        )

        headings[1].markdown(
            "**Image**"
        )

        headings[2].markdown(
            "**Prediction**"
        )

        headings[3].markdown(
            "**Confidence**"
        )

        headings[4].markdown(
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
                    .6,
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
                    f"{float(row['confidence']) * 100:.1f}%"
                )


            if cols[4].button(
                "🗑️",
                key=f"delete_{int(row['id'])}",
            ):

                delete_history(
                    row[
                        "id"
                    ]
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
# DERMAGUIDE AI
# =========================================================

elif page == "DermaGuide AI":

    st.header(
        "🤖 DermaGuide AI"
    )


    st.caption(
        """
        Educational assistant for understanding the
        latest DermaSense machine-learning result.
        """
    )


    if (
        st.session_state.latest_prediction
        is None
    ):

        st.info(
            """
            Analyze an image first.

            DermaGuide will then use your latest
            DermaSense result as context.
            """
        )


    else:

        prediction = (
            st.session_state.latest_prediction
        )


        probabilities = (
            st.session_state.latest_probabilities
        )


        filename = (
            st.session_state.latest_filename
        )


        # =============================================
        # CURRENT RESULT
        # =============================================

        with st.container(
            border=True
        ):

            left, right = st.columns(
                [
                    1.5,
                    1,
                ]
            )


            with left:

                st.caption(
                    "LATEST ANALYSIS"
                )

                st.subheader(
                    prediction
                )

                st.caption(
                    filename
                )


            with right:

                st.markdown(
                    """
                    <div class="derma-result-chip">
                    🤖 DermaGuide Ready
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        st.warning(
            """
            DermaGuide provides general educational information
            only. It cannot diagnose a skin condition or prescribe
            a personal treatment.
            """
        )


        # =============================================
        # QUICK QUESTIONS
        # =============================================

        st.subheader(
            "Quick Questions"
        )


        q1, q2, q3 = st.columns(3)

        quick_question = None


        if q1.button(
            "🧠 Why predicted?",
            use_container_width=True,
        ):

            quick_question = (
                "Why did the model predict this?"
            )


        if q2.button(
            "⚠️ Possible effects",
            use_container_width=True,
        ):

            quick_question = (
                "What effects can it have?"
            )


        if q3.button(
            "🛡️ Risk reduction",
            use_container_width=True,
        ):

            quick_question = (
                "How can risk be reduced?"
            )


        q4, q5, q6 = st.columns(3)


        if q4.button(
            "🏥 Treatment info",
            use_container_width=True,
        ):

            quick_question = (
                "What treatments are generally used?"
            )


        if q5.button(
            "👩‍⚕️ What next?",
            use_container_width=True,
        ):

            quick_question = (
                "What should I do next?"
            )


        if q6.button(
            "ℹ️ Explain result",
            use_container_width=True,
        ):

            quick_question = (
                "What does this result mean?"
            )


        if quick_question:

            answer = dermaguide_reply(
                quick_question,
                prediction=prediction,
                probabilities=probabilities,
            )


            st.session_state.dermaguide_messages.extend(
                [
                    {
                        "role": "user",
                        "content": quick_question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            )


            st.rerun()


        st.divider()


        # =============================================
        # CHAT
        # =============================================

        if not st.session_state.dermaguide_messages:

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    f"""
                    👋 I'm **DermaGuide AI**.

                    Your latest DermaSense result is
                    **{prediction}**.

                    Ask me about:

                    - Why the model predicted it
                    - What the result means
                    - Possible effects
                    - General risk reduction
                    - General treatment information
                    - What to do next
                    """
                )


        for message in (
            st.session_state.dermaguide_messages
        ):

            with st.chat_message(
                message[
                    "role"
                ]
            ):

                st.markdown(
                    message[
                        "content"
                    ]
                )


        question = st.chat_input(
            "Ask DermaGuide..."
        )


        if question:

            st.session_state.dermaguide_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            answer = dermaguide_reply(
                question,
                prediction=prediction,
                probabilities=probabilities,
            )


            st.session_state.dermaguide_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


            st.rerun()


        if st.button(
            "🗑️ Clear DermaGuide Chat"
        ):

            st.session_state.dermaguide_messages = []

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "DermaSense AI • "
    "MobileNetV2 Transfer Learning • "
    "Grad-CAM Explainability • "
    "DermaGuide AI"
)