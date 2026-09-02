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
# APP CONFIG
# =========================================================

APP_TITLE = "DermaSense AI"

DB_PATH = Path("data") / "analysis_history.db"

PROCESSING_SECONDS = 5.2


st.set_page_config(
    page_title=f"{APP_TITLE} | Skin Image Analysis",
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
# TECHNICAL UI THEME
# =========================================================

st.markdown(
    """
<style>

/* ======================================================
   MAIN APP BACKGROUND
   ====================================================== */

.stApp {
    background:
        linear-gradient(
            rgba(18, 80, 110, 0.022) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(18, 80, 110, 0.022) 1px,
            transparent 1px
        ),
        radial-gradient(
            circle at 82% 10%,
            rgba(0, 196, 235, 0.050),
            transparent 30%
        ),
        radial-gradient(
            circle at 12% 50%,
            rgba(34, 119, 153, 0.035),
            transparent 35%
        ),
        linear-gradient(
            180deg,
            #020b14 0%,
            #04111d 48%,
            #061522 100%
        );

    background-size:
        38px 38px,
        38px 38px,
        auto,
        auto,
        auto;
}


.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* ======================================================
   REMOVE EXTRA STREAMLIT BRANDING
   DO NOT HIDE TOOLBAR COMPLETELY
   ====================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stStatusWidget"] {
    display: none;
}

button[title="View fullscreen"] {
    display: none !important;
}

.stAppDeployButton {
    display: none !important;
}

[data-testid="stAppDeployButton"] {
    display: none !important;
}


/* ======================================================
   KEEP SIDEBAR OPEN/COLLAPSE CONTROL VISIBLE
   ====================================================== */

[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}


/* ======================================================
   SIDEBAR
   ====================================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #020b14,
            #04121e
        );

    border-right:
        1px solid rgba(66, 190, 220, 0.16);
}


[data-testid="stSidebar"] .block-container {
    padding-top: 1.3rem;
}


/* ======================================================
   SIDEBAR TECH STACK
   ====================================================== */

.tech-stack-card {
    background:
        linear-gradient(
            145deg,
            rgba(4, 22, 36, .96),
            rgba(6, 34, 51, .72)
        );

    border:
        1px solid rgba(64, 184, 213, .17);

    border-radius: 12px;

    padding: 9px 11px;

    margin-top: 5px;
    margin-bottom: 10px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.025);
}


.tech-row {
    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 8px;

    padding: 7px 0;

    border-bottom:
        1px solid rgba(100,160,185,.08);
}


.tech-row:last-child {
    border-bottom: none;
}


.tech-name {
    color: #66899d;

    font-size: 10px;
}


.tech-value {
    color: #d8edf5;

    font-size: 10px;

    font-weight: 700;

    text-align: right;
}


.model-ready {
    display: flex;

    align-items: center;

    gap: 8px;

    padding: 10px 11px;

    border-radius: 10px;

    background:
        rgba(22,135,109,.08);

    border:
        1px solid rgba(62,214,167,.20);

    color: #58e7bc;

    font-size: 10px;

    font-weight: 700;
}


.ready-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #52ebbc;

    box-shadow:
        0 0 7px #52ebbc;
}


.output-class-box {
    padding: 9px 11px;

    border-radius: 10px;

    background:
        rgba(5,25,40,.72);

    border:
        1px solid rgba(65,148,178,.13);

    color: #849dac;

    font-size: 9px;

    line-height: 1.8;
}


/* ======================================================
   HEADINGS
   ====================================================== */

h1,
h2,
h3 {
    color: #e8f4fa !important;

    letter-spacing: -0.02em;
}


/* ======================================================
   BORDERED CONTAINERS
   ====================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        rgba(4,20,33,.50);

    border-color:
        rgba(68,164,196,.16) !important;

    border-radius:
        15px !important;
}


/* ======================================================
   FILE UPLOAD / CAMERA
   ====================================================== */

[data-testid="stFileUploader"] {
    background:
        rgba(4,22,36,.76);

    border:
        1px dashed rgba(62,203,229,.31);

    border-radius: 12px;

    padding: 8px;
}


[data-testid="stCameraInput"] {
    background:
        rgba(4,22,36,.76);

    border:
        1px solid rgba(62,203,229,.25);

    border-radius: 12px;

    padding: 8px;
}


/* ======================================================
   BUTTONS
   ====================================================== */

div.stButton > button {
    background:
        linear-gradient(
            135deg,
            #075f73,
            #114d7c
        );

    border:
        1px solid rgba(56,218,242,.35);

    border-radius: 9px;

    color: #edfaff;

    font-weight: 700;

    min-height: 38px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.04);

    transition:
        all .2s ease;
}


div.stButton > button:hover {
    border-color: #4be6ff;

    box-shadow:
        0 0 18px rgba(55,218,244,.12);

    transform:
        translateY(-1px);

    color: white;
}


/* ======================================================
   METRICS
   ====================================================== */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(5,26,42,.92),
            rgba(5,32,50,.72)
        );

    border:
        1px solid rgba(60,169,199,.15);

    border-radius: 12px;

    padding: 12px 14px;
}


/* ======================================================
   ALERTS
   ====================================================== */

[data-testid="stAlert"] {
    border-radius: 10px;

    border:
        1px solid rgba(80,160,185,.16);
}


/* ======================================================
   FINAL CLASSIFICATION CARD
   ====================================================== */

.final-card {
    border:
        1px solid rgba(63,199,224,.23);

    border-radius: 16px;

    padding: 22px 24px;

    background:
        linear-gradient(
            145deg,
            rgba(5,24,39,.97),
            rgba(7,34,52,.84)
        );

    box-shadow:
        0 16px 45px rgba(0,0,0,.20);
}


.final-label {
    color: #658498;

    font-size: 10px;

    letter-spacing: 1.3px;
}


.final-benign {
    color: #43dfbf;

    font-size: 32px;

    font-weight: 800;

    margin-top: 6px;
}


.final-melanoma {
    color: #ff719e;

    font-size: 32px;

    font-weight: 800;

    margin-top: 6px;
}


.final-other {
    color: #58ddff;

    font-size: 32px;

    font-weight: 800;

    margin-top: 6px;
}


.final-description {
    color: #8ca7b7;

    font-size: 13px;

    line-height: 1.55;

    margin-top: 8px;
}


.other-clean-card {
    padding: 30px;

    text-align: center;

    border-radius: 16px;

    background:
        radial-gradient(
            circle at center,
            rgba(47,165,201,.10),
            transparent 62%
        ),
        rgba(5,25,40,.84);

    border:
        1px solid rgba(71,198,228,.20);
}


.other-circle {
    width: 58px;
    height: 58px;

    margin: 0 auto 12px auto;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    color: #59e4ff;

    font-size: 24px;

    background:
        rgba(45,154,188,.13);

    border:
        1px solid rgba(83,223,248,.26);
}


/* ======================================================
   DERMAGUIDE
   ====================================================== */

.derma-result-chip {
    display: inline-block;

    padding: 7px 12px;

    border:
        1px solid rgba(64,222,255,.30);

    border-radius: 15px;

    color: #5eeaff;

    background:
        rgba(23,92,119,.18);

    font-size: 12px;

    font-weight: 700;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# IMAGE TO BASE64
# =========================================================

def image_to_base64(image):

    preview = image.copy()

    preview.thumbnail(
        (620, 620)
    )

    buffer = io.BytesIO()

    preview.save(
        buffer,
        format="JPEG",
        quality=92,
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

    image_data = image_to_base64(
        image
    )

    safe_filename = html.escape(
        str(filename)
    )


    processing_html = """
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


.root {
    width: 100%;

    height: 355px;

    display: flex;

    align-items: center;

    justify-content: center;
}


.processor {
    position: relative;

    width: 790px;

    max-width: 96%;

    height: 320px;

    overflow: hidden;

    border-radius: 19px;

    background:
        linear-gradient(
            rgba(31,115,145,.025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(31,115,145,.025) 1px,
            transparent 1px
        ),
        radial-gradient(
            circle at 21% 42%,
            rgba(39,202,224,.11),
            transparent 40%
        ),
        linear-gradient(
            145deg,
            #04131f,
            #072337
        );

    background-size:
        25px 25px,
        25px 25px,
        auto,
        auto;

    border:
        1px solid rgba(58,201,228,.25);

    box-shadow:
        0 20px 55px rgba(0,0,0,.28);
}


/* ======================================================
   HEADER
   ====================================================== */

.header {
    height: 54px;

    display: flex;

    align-items: center;

    padding: 0 19px;

    border-bottom:
        1px solid rgba(87,151,181,.12);
}


.live-dot {
    width: 8px;

    height: 8px;

    margin-right: 9px;

    border-radius: 50%;

    background: #4cf0bb;

    box-shadow:
        0 0 12px #4cf0bb;

    animation:
        pulse
        1s
        infinite;
}


.title {
    color: #eaf5fa;

    font-size: 14px;

    font-weight: 750;
}


.status {
    margin-left: auto;

    color: #65869a;

    font-size: 7px;

    letter-spacing: 1.2px;
}


/* ======================================================
   IMAGE ANALYSIS AREA
   ====================================================== */

.image-zone {
    position: absolute;

    left: 29px;

    top: 74px;

    width: 290px;

    height: 210px;

    display: flex;

    align-items: center;

    justify-content: center;

    perspective: 950px;
}


.stack {
    position: relative;

    width: 215px;

    height: 170px;

    transform-style:
        preserve-3d;

    animation:
        modelMove
        5.2s
        ease-in-out
        forwards;
}


.back {
    position: absolute;

    left: 9px;

    top: 8px;

    width: 198px;

    height: 145px;

    border-radius: 14px;

    background:
        rgba(24,69,89,.28);

    border:
        1px solid rgba(66,212,235,.13);
}


.back1 {
    transform:
        translateZ(-29px)
        translate(11px,9px);
}


.back2 {
    transform:
        translateZ(-15px)
        translate(6px,5px);
}


.image-face {
    position: absolute;

    left: 9px;

    top: 8px;

    width: 198px;

    height: 145px;

    overflow: hidden;

    border-radius: 14px;

    transform:
        translateZ(24px);

    border:
        1px solid rgba(64,226,247,.46);

    box-shadow:
        0 14px 30px rgba(0,0,0,.28);
}


.image-face img {
    width: 100%;

    height: 100%;

    object-fit: cover;
}


.grid {
    position: absolute;

    inset: 0;

    opacity: 0;

    background-image:
        linear-gradient(
            rgba(59,222,243,.14) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(59,222,243,.14) 1px,
            transparent 1px
        );

    background-size: 22px 22px;

    animation:
        gridShow
        5.2s
        linear
        forwards;
}


.scan {
    position: absolute;

    left: 0;

    top: 5px;

    width: 100%;

    height: 4px;

    background:
        linear-gradient(
            90deg,
            transparent,
            #44e8ff,
            white,
            #5aa6ff,
            transparent
        );

    box-shadow:
        0 0 16px rgba(62,227,255,.82);

    animation:
        scanMove
        1.18s
        ease-in-out
        infinite
        alternate;
}


.filename {
    position: absolute;

    bottom: -8px;

    width: 100%;

    color: #648195;

    text-align: center;

    font-size: 7px;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;
}


/* ======================================================
   PIPELINE
   ====================================================== */

.pipeline {
    position: absolute;

    left: 347px;

    right: 22px;

    top: 74px;

    height: 210px;

    padding: 13px;

    border-radius: 14px;

    background:
        rgba(4,25,41,.80);

    border:
        1px solid rgba(76,145,175,.13);
}


.pipeline-title {
    color: #dcebf3;

    font-size: 11px;

    font-weight: 700;

    margin-bottom: 9px;
}


.step {
    height: 29px;

    display: flex;

    align-items: center;

    padding: 0 9px;

    margin-bottom: 5px;

    border-radius: 7px;

    background:
        rgba(7,35,55,.58);

    border:
        1px solid rgba(73,136,163,.09);

    color: #668397;

    font-size: 8px;
}


.step-number {
    width: 19px;

    height: 19px;

    margin-right: 8px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    color: #70a8be;

    border:
        1px solid rgba(68,214,237,.22);
}


.check {
    margin-left: auto;

    color: #4beeb7;

    opacity: 0;
}


.s1 {
    animation: stage1 5.2s linear forwards;
}

.s2 {
    animation: stage2 5.2s linear forwards;
}

.s3 {
    animation: stage3 5.2s linear forwards;
}

.s4 {
    animation: stage4 5.2s linear forwards;
}

.s5 {
    animation: stage5 5.2s linear forwards;
}


.s1 .check {
    animation: done1 5.2s linear forwards;
}

.s2 .check {
    animation: done2 5.2s linear forwards;
}

.s3 .check {
    animation: done3 5.2s linear forwards;
}

.s4 .check {
    animation: done4 5.2s linear forwards;
}

.s5 .check {
    animation: done5 5.2s linear forwards;
}


/* ======================================================
   PROGRESS
   ====================================================== */

.progress {
    height: 5px;

    margin-top: 9px;

    overflow: hidden;

    border-radius: 20px;

    background: #0e293c;
}


.progress-value {
    width: 0;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #41ddef,
            #397ee5,
            #765bd7
        );

    animation:
        progress
        5.2s
        linear
        forwards;
}


/* ======================================================
   ANIMATIONS
   ====================================================== */

@keyframes pulse {

    0%,
    100% {
        opacity: .4;
        transform: scale(.8);
    }

    50% {
        opacity: 1;
        transform: scale(1.25);
    }
}


@keyframes modelMove {

    0% {
        transform:
            rotateY(-11deg)
            rotateX(3deg);
    }

    28% {
        transform:
            rotateY(11deg)
            rotateX(-3deg);
    }

    55% {
        transform:
            rotateY(-8deg)
            rotateX(5deg)
            scale(1.03);
    }

    78% {
        transform:
            rotateY(7deg)
            rotateX(-2deg);
    }

    100% {
        transform:
            rotateY(0deg)
            rotateX(0deg);
    }
}


@keyframes scanMove {

    from {
        top: 5px;
    }

    to {
        top: 138px;
    }
}


@keyframes gridShow {

    0%,
    29% {
        opacity: 0;
    }

    38%,
    76% {
        opacity: .72;
    }

    90%,
    100% {
        opacity: .08;
    }
}


@keyframes progress {

    0% { width: 2%; }
    20% { width: 20%; }
    40% { width: 42%; }
    60% { width: 64%; }
    80% { width: 84%; }
    100% { width: 100%; }
}


@keyframes stage1 {

    0%,
    18% {
        color: white;
        border-color: #45e8ff;
    }
}


@keyframes stage2 {

    0%,
    19% {
        opacity: .40;
    }

    21%,
    38% {
        opacity: 1;

        color: white;

        border-color: #469fe7;
    }
}


@keyframes stage3 {

    0%,
    39% {
        opacity: .40;
    }

    41%,
    58% {
        opacity: 1;

        color: white;

        border-color: #557dde;
    }
}


@keyframes stage4 {

    0%,
    59% {
        opacity: .40;
    }

    61%,
    78% {
        opacity: 1;

        color: white;

        border-color: #6c69d9;
    }
}


@keyframes stage5 {

    0%,
    79% {
        opacity: .40;
    }

    81%,
    100% {
        opacity: 1;

        color: white;

        border-color: #49c9dd;
    }
}


@keyframes done1 {

    0%,19% { opacity: 0; }

    20%,100% { opacity: 1; }
}


@keyframes done2 {

    0%,39% { opacity: 0; }

    40%,100% { opacity: 1; }
}


@keyframes done3 {

    0%,59% { opacity: 0; }

    60%,100% { opacity: 1; }
}


@keyframes done4 {

    0%,79% { opacity: 0; }

    80%,100% { opacity: 1; }
}


@keyframes done5 {

    0%,96% { opacity: 0; }

    97%,100% { opacity: 1; }
}

</style>

</head>


<body>


<div class="root">


<div class="processor">


<div class="header">

<div class="live-dot"></div>

<div class="title">
DermaSense Neural Analysis
</div>

<div class="status">
INFERENCE ACTIVE
</div>

</div>


<div class="image-zone">


<div class="stack">

<div class="back back1"></div>

<div class="back back2"></div>


<div class="image-face">

<img
src="data:image/jpeg;base64,__IMAGE__"
/>

<div class="grid"></div>

<div class="scan"></div>

</div>


<div class="filename">
__FILENAME__
</div>

</div>


</div>


<div class="pipeline">


<div class="pipeline-title">
Inference Pipeline
</div>


<div class="step s1">

<div class="step-number">
1
</div>

Input image decoding

<div class="check">
✓
</div>

</div>


<div class="step s2">

<div class="step-number">
2
</div>

224 × 224 RGB preprocessing

<div class="check">
✓
</div>

</div>


<div class="step s3">

<div class="step-number">
3
</div>

MobileNetV2 feature encoding

<div class="check">
✓
</div>

</div>


<div class="step s4">

<div class="step-number">
4
</div>

Class-response computation

<div class="check">
✓
</div>

</div>


<div class="step s5">

<div class="step-number">
5
</div>

Classification output generated

<div class="check">
✓
</div>

</div>


<div class="progress">

<div class="progress-value"></div>

</div>


</div>


</div>


</div>


</body>

</html>
    """


    processing_html = processing_html.replace(
        "__IMAGE__",
        image_data,
    )

    processing_html = processing_html.replace(
        "__FILENAME__",
        safe_filename,
    )


    components.html(
        processing_html,
        height=365,
        scrolling=False,
    )


# =========================================================
# CONNECTED TECHNICAL 3D VISUALIZATION
# =========================================================

def render_connected_3d_skin(
    image,
    prediction,
):

    image_data = image_to_base64(
        image
    )


    if prediction == "Benign-like":
        accent = "#43dfc0"
    else:
        accent = "#ff6d99"


    visual_html = """
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


.root {
    width: 100%;

    height: 500px;

    display: flex;

    justify-content: center;

    align-items: center;
}


.card {
    position: relative;

    width: 960px;

    max-width: 98%;

    height: 470px;

    overflow: hidden;

    border-radius: 20px;

    background:
        linear-gradient(
            rgba(34,110,141,.025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(34,110,141,.025) 1px,
            transparent 1px
        ),
        radial-gradient(
            circle at 38% 47%,
            rgba(35,151,185,.13),
            transparent 48%
        ),
        linear-gradient(
            145deg,
            #041420,
            #071e30
        );

    background-size:
        28px 28px,
        28px 28px,
        auto,
        auto;

    border:
        1px solid rgba(58,199,225,.22);

    box-shadow:
        0 20px 58px rgba(0,0,0,.22);
}


/* ======================================================
   HEADER
   ====================================================== */

.header {
    height: 58px;

    display: flex;

    align-items: center;

    padding: 0 20px;

    border-bottom:
        1px solid rgba(88,153,181,.11);
}


.header-icon {
    width: 30px;

    height: 30px;

    display: flex;

    justify-content: center;

    align-items: center;

    margin-right: 9px;

    border-radius: 8px;

    color: #4ce5ff;

    background:
        rgba(31,112,140,.16);

    border:
        1px solid rgba(75,219,244,.20);
}


.header-title {
    color: #e7f2f8;

    font-size: 13px;

    font-weight: 750;
}


.header-sub {
    color: #5e7e91;

    font-size: 6px;

    margin-top: 2px;
}


.status {
    margin-left: auto;

    display: flex;

    align-items: center;

    gap: 6px;

    padding: 5px 9px;

    border-radius: 13px;

    color: #60e8bd;

    font-size: 6px;

    background:
        rgba(38,157,115,.08);

    border:
        1px solid rgba(72,223,170,.17);
}


.status-dot {
    width: 6px;

    height: 6px;

    border-radius: 50%;

    background: #4cf0b9;
}


/* ======================================================
   LEFT PANEL
   ====================================================== */

.scene-panel {
    position: absolute;

    left: 17px;

    top: 73px;

    bottom: 17px;

    width: 625px;

    border-radius: 15px;

    background:
        radial-gradient(
            circle at 50% 48%,
            rgba(36,142,178,.14),
            transparent 55%
        ),
        rgba(4,24,39,.70);

    border:
        1px solid rgba(73,147,177,.11);
}


.scene-title {
    position: absolute;

    left: 16px;

    top: 14px;

    color: #dceaf2;

    font-size: 10px;

    font-weight: 700;
}


.scene-subtitle {
    position: absolute;

    left: 16px;

    top: 30px;

    color: #5d7b8e;

    font-size: 6px;
}


/* ======================================================
   3D SCENE
   ====================================================== */

.scene {
    position: absolute;

    left: 0;

    right: 0;

    top: 42px;

    bottom: 0;

    display: flex;

    align-items: center;

    justify-content: center;

    perspective: 1150px;
}


.skin-model {
    position: relative;

    width: 480px;

    height: 270px;

    transform-style:
        preserve-3d;

    transform:
        rotateX(52deg)
        rotateY(-7deg)
        rotateZ(-3deg);

    animation:
        wholeModel
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   USER IMAGE SURFACE
   ====================================================== */

.surface {
    position: absolute;

    left: 44px;

    top: 19px;

    width: 390px;

    height: 175px;

    overflow: hidden;

    border-radius:
        16px 16px 3px 3px;

    transform:
        translateZ(55px)
        translateY(0);

    background: #d78f83;

    border:
        1px solid rgba(64,222,244,.40);

    box-shadow:
        0 12px 30px rgba(0,0,0,.24);

    animation:
        surfaceMotion
        5.3s
        ease-in-out
        forwards;
}


.surface img {
    width: 100%;

    height: 100%;

    object-fit: cover;
}


.surface-grid {
    position: absolute;

    inset: 0;

    background-image:
        linear-gradient(
            rgba(56,215,236,.07) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(56,215,236,.07) 1px,
            transparent 1px
        );

    background-size: 27px 27px;
}


.analysis-line {
    position: absolute;

    left: 0;

    top: 6px;

    width: 100%;

    height: 4px;

    opacity: 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            #40e5ff,
            white,
            __ACCENT__,
            transparent
        );

    box-shadow:
        0 0 15px rgba(62,226,255,.76);

    animation:
        surfaceScan
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   EPIDERMIS
   ====================================================== */

.epidermis {
    position: absolute;

    left: 48px;

    top: 188px;

    width: 382px;

    height: 24px;

    background:
        linear-gradient(
            180deg,
            #df968a,
            #c47676
        );

    transform:
        translateZ(40px)
        translateY(0);

    animation:
        epidermisMotion
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   DERMIS
   ====================================================== */

.dermis {
    position: absolute;

    left: 48px;

    top: 211px;

    width: 382px;

    height: 78px;

    overflow: hidden;

    background:
        linear-gradient(
            180deg,
            #a65e6f 0%,
            #79485d 100%
        );

    transform:
        translateZ(28px)
        translateY(0);

    animation:
        dermisMotion
        5.3s
        ease-in-out
        forwards;
}


.dermis-texture {
    position: absolute;

    inset: 0;

    opacity: .32;

    background-image:
        radial-gradient(
            circle,
            rgba(241,170,169,.40) 1px,
            transparent 2px
        );

    background-size: 15px 14px;
}


/* ======================================================
   SUBCUTANEOUS
   ====================================================== */

.subcutaneous {
    position: absolute;

    left: 48px;

    top: 287px;

    width: 382px;

    height: 39px;

    border-radius:
        0 0 12px 12px;

    background:
        radial-gradient(
            circle,
            #d3a24c 0 35%,
            #b57830 38% 60%,
            transparent 62%
        );

    background-size:
        25px 22px;

    background-color: #b98035;

    transform:
        translateZ(17px)
        translateY(0);

    animation:
        subcutaneousMotion
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   NERVES AND BLOOD VESSELS
   ====================================================== */

.anatomy {
    position: absolute;

    left: 0;

    top: 5px;

    width: 100%;

    height: 70px;

    z-index: 5;
}


.nerve {
    fill: none;

    stroke: #ebbe68;

    stroke-width: 2;

    stroke-linecap: round;

    stroke-dasharray: 240;

    stroke-dashoffset: 240;

    animation:
        nerveDraw
        5.3s
        ease-out
        forwards;
}


.branch {
    fill: none;

    stroke: #dda954;

    stroke-width: 1.2;

    stroke-linecap: round;

    stroke-dasharray: 90;

    stroke-dashoffset: 90;

    animation:
        branchDraw
        5.3s
        ease-out
        forwards;
}


.vessel-red {
    fill: none;

    stroke: #d45b60;

    stroke-width: 2.3;

    stroke-linecap: round;

    stroke-dasharray: 400;

    stroke-dashoffset: 400;

    animation:
        vesselDraw
        5.3s
        ease-out
        forwards;
}


.vessel-blue {
    fill: none;

    stroke: #4789bc;

    stroke-width: 2.3;

    stroke-linecap: round;

    stroke-dasharray: 400;

    stroke-dashoffset: 400;

    animation:
        vesselDraw
        5.3s
        ease-out
        forwards;
}


.node {
    fill: #54ddf7;

    opacity: 0;

    animation:
        nodeGlow
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   INFORMATION PANEL
   ====================================================== */

.info-panel {
    position: absolute;

    right: 17px;

    top: 73px;

    bottom: 17px;

    width: 283px;

    padding: 14px;

    border-radius: 15px;

    background:
        rgba(4,24,39,.76);

    border:
        1px solid rgba(73,147,177,.11);
}


.info-title {
    color: #dceaf2;

    font-size: 10px;

    font-weight: 700;

    margin-bottom: 12px;
}


.info-row {
    display: flex;

    align-items: center;

    gap: 8px;

    padding: 8px;

    margin-bottom: 7px;

    border-radius: 8px;

    background:
        rgba(7,35,54,.55);

    border:
        1px solid rgba(73,136,163,.09);

    opacity: .32;

    animation:
        infoReveal
        5.3s
        ease-out
        forwards;
}


.code {
    width: 29px;

    height: 29px;

    display: flex;

    justify-content: center;

    align-items: center;

    border-radius: 7px;

    color: #50dcef;

    font-size: 7px;

    background:
        rgba(30,103,128,.17);
}


.info-name {
    color: #cbdde6;

    font-size: 7px;

    font-weight: 700;
}


.info-desc {
    color: #5e7d90;

    font-size: 5.5px;

    margin-top: 2px;
}


/* ======================================================
   END STATUS
   ====================================================== */

.settled {
    position: absolute;

    left: 50%;

    bottom: 15px;

    transform:
        translateX(-50%);

    opacity: 0;

    padding: 5px 10px;

    border-radius: 13px;

    color: #5fe6bb;

    font-size: 6px;

    background:
        rgba(40,157,117,.09);

    border:
        1px solid rgba(74,223,173,.16);

    animation:
        settledReveal
        5.3s
        ease-in-out
        forwards;
}


/* ======================================================
   ANIMATION
   ====================================================== */

@keyframes wholeModel {

    0% {
        opacity: 0;

        transform:
            rotateX(58deg)
            rotateY(-16deg)
            rotateZ(-5deg)
            scale(.92);
    }

    15% {
        opacity: 1;
    }

    32% {
        transform:
            rotateX(52deg)
            rotateY(6deg)
            rotateZ(-3deg);
    }

    68% {
        transform:
            rotateX(52deg)
            rotateY(-7deg)
            rotateZ(-3deg)
            scale(1.015);
    }

    100% {
        opacity: 1;

        transform:
            rotateX(52deg)
            rotateY(-7deg)
            rotateZ(-3deg)
            scale(1);
    }
}


@keyframes surfaceMotion {

    0%,
    18% {
        transform:
            translateZ(55px)
            translateY(0);
    }

    36%,
    58% {
        transform:
            translateZ(80px)
            translateY(-22px);
    }

    77% {
        transform:
            translateZ(60px)
            translateY(-4px);
    }

    100% {
        transform:
            translateZ(55px)
            translateY(0);
    }
}


@keyframes epidermisMotion {

    0%,
    18% {
        transform:
            translateZ(40px)
            translateY(0);
    }

    36%,
    58% {
        transform:
            translateZ(40px)
            translateY(10px);
    }

    78%,
    100% {
        transform:
            translateZ(40px)
            translateY(0);
    }
}


@keyframes dermisMotion {

    0%,
    18% {
        transform:
            translateZ(28px)
            translateY(0);
    }

    36%,
    58% {
        transform:
            translateZ(28px)
            translateY(23px);
    }

    78%,
    100% {
        transform:
            translateZ(28px)
            translateY(0);
    }
}


@keyframes subcutaneousMotion {

    0%,
    18% {
        transform:
            translateZ(17px)
            translateY(0);
    }

    36%,
    58% {
        transform:
            translateZ(17px)
            translateY(37px);
    }

    78%,
    100% {
        transform:
            translateZ(17px)
            translateY(0);
    }
}


@keyframes surfaceScan {

    0%,
    9% {
        opacity: 0;

        top: 5px;
    }

    15% {
        opacity: 1;
    }

    27% {
        top: 164px;
    }

    34% {
        top: 14px;
    }

    42% {
        top: 164px;

        opacity: 1;
    }

    50%,
    100% {
        opacity: 0;
    }
}


@keyframes nerveDraw {

    0%,
    31% {
        stroke-dashoffset: 240;
    }

    63%,
    100% {
        stroke-dashoffset: 0;
    }
}


@keyframes branchDraw {

    0%,
    36% {
        stroke-dashoffset: 90;
    }

    67%,
    100% {
        stroke-dashoffset: 0;
    }
}


@keyframes vesselDraw {

    0%,
    33% {
        stroke-dashoffset: 400;
    }

    69%,
    100% {
        stroke-dashoffset: 0;
    }
}


@keyframes nodeGlow {

    0%,
    40% {
        opacity: 0;
    }

    54% {
        opacity: 1;

        filter:
            drop-shadow(
                0 0 5px
                #54ddf7
            );
    }

    100% {
        opacity: .75;

        filter: none;
    }
}


@keyframes infoReveal {

    0%,
    28% {
        opacity: .25;

        transform:
            translateX(4px);
    }

    55%,
    100% {
        opacity: 1;

        transform:
            translateX(0);
    }
}


@keyframes settledReveal {

    0%,
    79% {
        opacity: 0;
    }

    88%,
    100% {
        opacity: 1;
    }
}

</style>

</head>


<body>


<div class="root">


<div class="card">


<div class="header">

<div class="header-icon">
⬡
</div>


<div>

<div class="header-title">
DermaSense Structural Analysis
</div>

<div class="header-sub">
POST-INFERENCE TECHNICAL VISUALIZATION
</div>

</div>


<div class="status">

<span class="status-dot"></span>

ANALYSIS COMPLETE

</div>

</div>


<div class="scene-panel">


<div class="scene-title">
Connected Skin Structure
</div>


<div class="scene-subtitle">
Uploaded surface image with illustrative internal layers
</div>


<div class="scene">


<div class="skin-model">


<div class="surface">

<img
src="data:image/jpeg;base64,__IMAGE__"
/>

<div class="surface-grid"></div>

<div class="analysis-line"></div>

</div>


<div class="epidermis"></div>


<div class="dermis">


<div class="dermis-texture"></div>


<svg
class="anatomy"
viewBox="0 0 382 70"
preserveAspectRatio="none"
>


<path
class="nerve"
d="
M12 48
C44 26,
69 57,
98 30
S148 17,
173 44
S218 61,
246 28
S295 18,
368 45
"
/>


<path
class="nerve"
d="
M39 61
C69 43,
88 32,
113 48
S151 60,
171 26
S212 22,
234 52
S281 51,
316 27
"
/>


<path
class="branch"
d="
M98 30
C94 20,
90 13,
86 5
"
/>


<path
class="branch"
d="
M173 44
C175 29,
179 19,
185 7
"
/>


<path
class="branch"
d="
M246 28
C252 19,
260 12,
267 5
"
/>


<path
class="vessel-red"
d="
M5 57
C58 49,
105 64,
154 53
S246 46,
377 58
"
/>


<path
class="vessel-blue"
d="
M2 65
C54 57,
106 70,
160 60
S250 54,
379 66
"
/>


<circle
class="node"
cx="86"
cy="5"
r="2.5"
/>


<circle
class="node"
cx="185"
cy="7"
r="2.5"
/>


<circle
class="node"
cx="267"
cy="5"
r="2.5"
/>


<circle
class="node"
cx="234"
cy="52"
r="2.5"
/>


</svg>


</div>


<div class="subcutaneous"></div>


</div>


</div>


</div>


<div class="info-panel">


<div class="info-title">
STRUCTURAL COMPONENTS
</div>


<div class="info-row">

<div class="code">
S
</div>

<div>

<div class="info-name">
Surface Input
</div>

<div class="info-desc">
Uploaded photograph
</div>

</div>

</div>


<div class="info-row">

<div class="code">
L1
</div>

<div>

<div class="info-name">
Epidermis
</div>

<div class="info-desc">
Illustrative outer layer
</div>

</div>

</div>


<div class="info-row">

<div class="code">
L2
</div>

<div>

<div class="info-name">
Dermis
</div>

<div class="info-desc">
Illustrative deeper layer
</div>

</div>

</div>


<div class="info-row">

<div class="code">
N/V
</div>

<div>

<div class="info-name">
Nerve + Vessel Network
</div>

<div class="info-desc">
Simplified anatomical model
</div>

</div>

</div>


<div class="info-row">

<div class="code">
L3
</div>

<div>

<div class="info-name">
Subcutaneous Layer
</div>

<div class="info-desc">
Illustrative lower tissue
</div>

</div>

</div>


</div>


<div class="settled">
CONNECTED MODEL READY
</div>


</div>


</div>


</body>

</html>
    """


    visual_html = visual_html.replace(
        "__IMAGE__",
        image_data,
    )

    visual_html = visual_html.replace(
        "__ACCENT__",
        accent,
    )


    components.html(
        visual_html,
        height=510,
        scrolling=False,
    )


# =========================================================
# FINAL PREDICTION
# =========================================================

def render_final_prediction(
    prediction,
    probabilities,
):

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


    # =====================================================
    # OTHER
    # =====================================================

    if prediction == "Other":

        st.markdown(
            """<div class="other-clean-card">
<div class="other-circle">—</div>
<div class="final-label">CLASSIFICATION OUTPUT</div>
<div class="final-other">Other</div>
<div class="final-description">
The Other / rejection response was stronger than the supported
Benign-like and Melanoma-suspicious responses.
</div>
</div>""",
            unsafe_allow_html=True,
        )

        return


    # =====================================================
    # BENIGN
    # =====================================================

    if prediction == "Benign-like":

        result_class = "final-benign"

        description = (
            "The Benign-like neural response was stronger "
            "than the melanoma response."
        )


    # =====================================================
    # MELANOMA
    # =====================================================

    else:

        result_class = "final-melanoma"

        description = (
            "The melanoma neural response was stronger than "
            "the Benign-like response. This machine-learning "
            "result does not confirm melanoma."
        )


    safe_prediction = html.escape(
        prediction
    )

    safe_description = html.escape(
        description
    )


    final_html = (
        '<div class="final-card">'
        '<div class="final-label">CLASSIFICATION OUTPUT</div>'
        f'<div class="{result_class}">{safe_prediction}</div>'
        f'<div class="final-description">{safe_description}</div>'
        '</div>'
    )


    st.markdown(
        final_html,
        unsafe_allow_html=True,
    )


    # =====================================================
    # NO MODEL CONFIDENCE
    # ONLY BENIGN + MELANOMA RESPONSES
    # =====================================================

    left_gap, benign_col, melanoma_col, right_gap = st.columns(
        [
            .28,
            1,
            1,
            .28,
        ]
    )


    with benign_col:

        st.metric(
            "Benign-like Response",
            f"{benign * 100:.1f}%",
        )


    with melanoma_col:

        st.metric(
            "Melanoma Response",
            f"{melanoma * 100:.1f}%",
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
                    now
                    -
                    previous_time
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


            if duplicate and seconds <= 10:
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
# GENERAL DERMAGUIDE FALLBACK
# =========================================================

def general_dermaguide_reply(
    question,
):

    q = question.lower()


    if (
        "grad" in q
        or
        "attention" in q
    ):

        return (
            "**Grad-CAM** is an explainability method that highlights "
            "image regions that had greater influence on a neural-network "
            "prediction. It shows model attention, not a confirmed "
            "medical abnormality."
        )


    if (
        "mobilenet" in q
        or
        "model" in q
        or
        "ai work" in q
    ):

        return (
            "DermaSense uses **MobileNetV2 transfer learning**. "
            "The uploaded image is resized to 224 × 224 RGB, processed "
            "through the network, converted into learned visual features, "
            "and compared across the trained output classes."
        )


    if "benign" in q:

        return (
            "**Benign** generally means non-cancerous. In DermaSense, "
            "`Benign-like` means the image produced a stronger response "
            "for visual patterns learned from the benign training examples. "
            "It is not a medical diagnosis."
        )


    if "melanoma" in q:

        return (
            "Melanoma is a serious type of skin cancer involving "
            "pigment-producing cells. A DermaSense "
            "`Melanoma-suspicious` result is only a machine-learning "
            "classification and cannot confirm melanoma."
        )


    if (
        "risk" in q
        or
        "protect" in q
        or
        "prevent" in q
    ):

        return (
            "General skin-risk reduction includes limiting excessive "
            "UV exposure, using suitable sun protection, avoiding tanning "
            "devices, and seeking professional evaluation for skin changes "
            "that are new, changing, unusual, or concerning."
        )


    if (
        "treatment" in q
        or
        "cure" in q
    ):

        return (
            "Treatment depends on the actual medical diagnosis and cannot "
            "be determined from an AI image classification. Healthcare "
            "professionals may use examination, dermoscopy, biopsy, and "
            "other tests before deciding on treatment."
        )


    return (
        "I can explain **MobileNetV2, transfer learning, Grad-CAM, "
        "Benign-like results, Melanoma-suspicious results, general "
        "skin-risk information, and how DermaSense processes images**."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🔬 DermaSense AI"
    )

    st.caption(
        "AI-Powered Skin Image Analysis"
    )

    st.write("")


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
        "### ⚙ Technical Stack"
    )


    tech_html = """<div class="tech-stack-card">
<div class="tech-row">
<span class="tech-name">Architecture</span>
<span class="tech-value">MobileNetV2</span>
</div>
<div class="tech-row">
<span class="tech-name">Learning</span>
<span class="tech-value">Transfer Learning</span>
</div>
<div class="tech-row">
<span class="tech-name">Input</span>
<span class="tech-value">224 × 224 RGB</span>
</div>
<div class="tech-row">
<span class="tech-name">Framework</span>
<span class="tech-value">TensorFlow / Keras</span>
</div>
<div class="tech-row">
<span class="tech-name">Explainability</span>
<span class="tech-value">Grad-CAM</span>
</div>
<div class="tech-row">
<span class="tech-name">Storage</span>
<span class="tech-value">SQLite</span>
</div>
<div class="tech-row">
<span class="tech-name">Interface</span>
<span class="tech-value">Streamlit</span>
</div>
</div>"""


    st.markdown(
        tech_html,
        unsafe_allow_html=True,
    )


    st.markdown(
        "### ◉ System Status"
    )


    if MODEL_PATH.exists():

        st.markdown(
            '<div class="model-ready">'
            '<span class="ready-dot"></span>'
            'Model Loaded & Ready'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.error(
            "Model unavailable"
        )


    st.write("")


    st.caption(
        "OUTPUT CATEGORIES"
    )


    st.markdown(
        '<div class="output-class-box">'
        '<strong>Benign-like</strong><br>'
        '<strong>Melanoma-suspicious</strong><br>'
        '<strong>Other</strong>'
        '</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN HEADER
# =========================================================

with st.container(
    border=True
):

    st.caption(
        "EXPLAINABLE MACHINE LEARNING / SKIN IMAGE CLASSIFICATION"
    )

    st.title(
        "DermaSense AI"
    )

    st.write(
        """
        AI-assisted skin-image classification using
        **MobileNetV2 transfer learning** with
        **Grad-CAM explainability**.

        Output categories:
        **Benign-like**, **Melanoma-suspicious**, and **Other**.
        """
    )


# =========================================================
# ANALYZE PAGE
# =========================================================

if page == "Analyze":

    st.warning(
        """
        **Educational machine-learning prototype — not a medical diagnosis.**
        Concerning or changing skin lesions should be assessed by a
        qualified healthcare professional.
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
    # IMAGE ACQUISITION
    # =====================================================

    st.markdown(
        "## Image Acquisition"
    )


    input_mode = st.radio(
        "Input source",
        [
            "Upload Images",
            "Use Camera",
        ],
        horizontal=True,
    )


    valid_images = []


    # =====================================================
    # UPLOAD
    # =====================================================

    if input_mode == "Upload Images":

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
                f"{len(uploaded_files)} image(s) selected"
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


                    # -----------------------------------------
                    # ONLY ONE PREVIEW
                    # -----------------------------------------

                    with preview_columns[
                        index
                        %
                        len(preview_columns)
                    ]:

                        st.image(
                            image,
                            caption=uploaded.name,
                            width=125,
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
            "Capture image",
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
                    "Image captured"
                )


                st.image(
                    camera_image,
                    caption="Captured image",
                    width=160,
                )


            except (
                UnidentifiedImageError,
                OSError,
            ):

                st.error(
                    "Unable to process captured image."
                )


    # =====================================================
    # SMALL ANALYZE BUTTON
    # =====================================================

    if (
        valid_images
        and
        model is not None
    ):

        if len(valid_images) == 1:

            button_text = "Analyze"

        else:

            button_text = (
                f"Analyze {len(valid_images)}"
            )


        left_button, center_button, right_button = st.columns(
            [
                2.3,
                .8,
                2.3,
            ]
        )


        with center_button:

            analyze_clicked = st.button(
                button_text,
                use_container_width=True,
            )


        # =================================================
        # PROCESS
        # =================================================

        if analyze_clicked:

            results = []


            for filename, image in valid_images:

                # -----------------------------------------
                # ONE PLACEHOLDER CONTAINS:
                # - Neural Inference heading
                # - processing animation
                #
                # BOTH disappear together afterward.
                # -----------------------------------------

                processing_placeholder = st.empty()


                with processing_placeholder.container():

                    st.markdown(
                        "## Neural Inference"
                    )


                    render_processing_animation(
                        image=image,
                        filename=filename,
                    )


                # -----------------------------------------
                # REAL ML PREDICTION
                # -----------------------------------------

                result = predict_lesion(
                    model,
                    image,
                    metadata,
                )


                # Keep animation visible for presentation

                time.sleep(
                    PROCESSING_SECONDS
                )


                # -----------------------------------------
                # REMOVES HEADING + ANIMATION
                # -----------------------------------------

                processing_placeholder.empty()


                results.append(
                    (
                        filename,
                        image,
                        result,
                    )
                )


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            for result_number, (
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
                # STORE LATEST RESULT
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


                add_history(
                    filename,
                    prediction,
                    confidence,
                    melanoma_score,
                )


                if len(results) > 1:

                    st.markdown(
                        f"## Image {result_number}"
                    )


                # =================================================
                # OTHER
                # =================================================

                if prediction == "Other":

                    # No structural visualization
                    # No nerves
                    # No Grad-CAM
                    # No response percentages

                    st.markdown(
                        "## Classification Output"
                    )


                    render_final_prediction(
                        prediction=prediction,
                        probabilities=probabilities,
                    )


                    st.info(
                        """
                        The **Other** response was strongest.
                        Lesion-specific structural visualization and
                        Grad-CAM are skipped for this result.
                        """
                    )


                # =================================================
                # BENIGN / MELANOMA
                # =================================================

                else:

                    # =============================================
                    # STRUCTURAL VISUALIZATION
                    # =============================================

                    st.markdown(
                        "## Structural Visualization"
                    )


                    render_connected_3d_skin(
                        image=image,
                        prediction=prediction,
                    )


                    # No extra text below 3D model


                    # =============================================
                    # MODEL ATTENTION
                    # =============================================

                    st.markdown(
                        "## Model Attention Analysis"
                    )


                    st.caption(
                        """
                        Grad-CAM highlights image regions that
                        influenced the neural-network response.
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


                        gradcam_result = create_gradcam_result(
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


                        (
                            left_gap,
                            original_col,
                            attention_col,
                            right_gap,
                        ) = st.columns(
                            [
                                .42,
                                1,
                                1,
                                .42,
                            ]
                        )


                        with original_col:

                            st.caption(
                                "INPUT IMAGE"
                            )

                            st.image(
                                image,
                                width=255,
                            )


                        with attention_col:

                            st.caption(
                                "GRAD-CAM RESPONSE"
                            )

                            st.image(
                                gradcam_result[
                                    "overlay"
                                ],
                                width=255,
                            )


                        st.caption(
                            """
                            Model-attention visualization only.
                            It does not diagnose or medically
                            localize cancer.
                            """
                        )


                    except Exception:

                        st.caption(
                            """
                            Classification completed successfully.
                            Grad-CAM visualization is unavailable.
                            """
                        )


                    # =============================================
                    # CLASSIFICATION LAST
                    # =============================================

                    st.markdown(
                        "## Classification Output"
                    )


                    render_final_prediction(
                        prediction=prediction,
                        probabilities=probabilities,
                    )


                    if (
                        prediction
                        ==
                        "Melanoma-suspicious"
                    ):

                        st.warning(
                            """
                            **Melanoma-suspicious** is a
                            machine-learning output and does not
                            confirm melanoma.
                            """
                        )


                st.success(
                    """
                    Open **DermaGuide AI** for an educational
                    explanation of the latest model result.
                    """
                )


# =========================================================
# HISTORY PAGE
# =========================================================

elif page == "History":

    st.markdown(
        "## Analysis History"
    )


    st.caption(
        """
        Prediction metadata is stored locally.
        Uploaded images themselves are not stored.
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
                    ==
                    "Benign-like"
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
                    ==
                    "Melanoma-suspicious"
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
                    ==
                    "Other"
                ).sum()
            ),
        )


        st.write("")


        for _, row in history.iterrows():

            with st.container(
                border=True
            ):

                cols = st.columns(
                    [
                        1.5,
                        2.1,
                        1.8,
                        1,
                        .55,
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
                    ==
                    "Other"
                ):

                    cols[3].write(
                        "—"
                    )

                else:

                    cols[3].write(
                        f"{float(row['confidence']) * 100:.1f}%"
                    )


                if cols[4].button(
                    "Delete",
                    key=f"delete_{int(row['id'])}",
                ):

                    delete_history(
                        row[
                            "id"
                        ]
                    )

                    st.rerun()


        if st.button(
            "Clear All History"
        ):

            clear_history()

            st.rerun()


# =========================================================
# DERMAGUIDE AI PAGE
# =========================================================

elif page == "DermaGuide AI":

    st.markdown(
        "## DermaGuide AI"
    )


    st.caption(
        """
        Educational assistant for explaining DermaSense,
        its machine-learning workflow, and the latest result.
        """
    )


    has_prediction = (
        st.session_state.latest_prediction
        is not None
    )


    # =====================================================
    # RESULT CONTEXT
    # =====================================================

    if has_prediction:

        prediction = (
            st.session_state.latest_prediction
        )

        probabilities = (
            st.session_state.latest_probabilities
        )

        filename = (
            st.session_state.latest_filename
        )


        with st.container(
            border=True
        ):

            left, right = st.columns(
                [
                    1.6,
                    .7,
                ]
            )


            with left:

                st.caption(
                    "LATEST CLASSIFICATION"
                )

                st.subheader(
                    prediction
                )

                st.caption(
                    filename
                )


            with right:

                st.markdown(
                    '<div class="derma-result-chip">'
                    'CONTEXT ACTIVE'
                    '</div>',
                    unsafe_allow_html=True,
                )


    else:

        prediction = None

        probabilities = {}


        st.info(
            """
            General assistant mode is active.
            Analyze an image to enable result-aware explanations.
            """
        )


    st.warning(
        """
        DermaGuide provides educational information only.
        It cannot diagnose a medical condition or prescribe
        personalized treatment.
        """
    )


    # =====================================================
    # QUICK QUESTIONS
    # =====================================================

    st.markdown(
        "### Quick Queries"
    )


    quick_question = None


    if has_prediction:

        q1, q2, q3 = st.columns(
            3
        )


        if q1.button(
            "Why this result?",
            use_container_width=True,
        ):

            quick_question = (
                "Why did DermaSense produce this prediction?"
            )


        if q2.button(
            "What does it mean?",
            use_container_width=True,
        ):

            quick_question = (
                "What does this result mean?"
            )


        if q3.button(
            "What next?",
            use_container_width=True,
        ):

            quick_question = (
                "What should someone generally do next?"
            )


        q4, q5, q6 = st.columns(
            3
        )


        if q4.button(
            "Possible concerns",
            use_container_width=True,
        ):

            quick_question = (
                "What general concerns may be associated "
                "with this type of result?"
            )


        if q5.button(
            "Risk reduction",
            use_container_width=True,
        ):

            quick_question = (
                "What are general skin cancer "
                "risk-reduction steps?"
            )


        if q6.button(
            "Explain Grad-CAM",
            use_container_width=True,
        ):

            quick_question = (
                "Explain what the Grad-CAM attention map means."
            )


    else:

        q1, q2, q3 = st.columns(
            3
        )


        if q1.button(
            "MobileNetV2",
            use_container_width=True,
        ):

            quick_question = (
                "How does MobileNetV2 work in DermaSense?"
            )


        if q2.button(
            "Benign-like",
            use_container_width=True,
        ):

            quick_question = (
                "What does Benign-like mean?"
            )


        if q3.button(
            "Melanoma",
            use_container_width=True,
        ):

            quick_question = (
                "What is melanoma?"
            )


        q4, q5, q6 = st.columns(
            3
        )


        if q4.button(
            "Grad-CAM",
            use_container_width=True,
        ):

            quick_question = (
                "What is Grad-CAM?"
            )


        if q5.button(
            "Skin protection",
            use_container_width=True,
        ):

            quick_question = (
                "Give general skin-risk reduction information."
            )


        if q6.button(
            "How DermaSense works",
            use_container_width=True,
        ):

            quick_question = (
                "Explain the DermaSense analysis pipeline."
            )


    # =====================================================
    # HANDLE QUICK QUESTION
    # =====================================================

    if quick_question:

        if has_prediction:

            answer = dermaguide_reply(
                quick_question,
                prediction=prediction,
                probabilities=probabilities,
            )

        else:

            answer = general_dermaguide_reply(
                quick_question
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


    # =====================================================
    # INITIAL BOT MESSAGE
    # =====================================================

    if not st.session_state.dermaguide_messages:

        with st.chat_message(
            "assistant"
        ):

            if has_prediction:

                st.markdown(
                    f"""
                    **DermaGuide ready.**

                    Latest classification:
                    **{prediction}**

                    Ask about the result, Grad-CAM,
                    general risk information, or the
                    DermaSense machine-learning workflow.
                    """
                )

            else:

                st.markdown(
                    """
                    **DermaGuide general mode ready.**

                    Ask about MobileNetV2, transfer learning,
                    Grad-CAM, Benign-like results,
                    Melanoma-suspicious results, or how
                    DermaSense processes an image.
                    """
                )


    # =====================================================
    # DISPLAY CHAT
    # =====================================================

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


    # =====================================================
    # CHAT INPUT
    # =====================================================

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


        if has_prediction:

            answer = dermaguide_reply(
                question,
                prediction=prediction,
                probabilities=probabilities,
            )

        else:

            answer = general_dermaguide_reply(
                question
            )


        st.session_state.dermaguide_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


        st.rerun()


    if st.button(
        "Clear DermaGuide Chat"
    ):

        st.session_state.dermaguide_messages = []

        st.rerun()