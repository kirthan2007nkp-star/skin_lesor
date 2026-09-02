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
# CONFIG
# =========================================================
APP_TITLE = "DermaSense AI"
DB_PATH = Path("data") / "analysis_history.db"
PROCESSING_SECONDS = 9.0

st.set_page_config(
    page_title="DermaSense AI | Skin Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "result_explanations" not in st.session_state:
    st.session_state.result_explanations = {}


# =========================================================
# GLOBAL STYLE
# =========================================================
CUSTOM_CSS = """
<style>
:root{
  --bg:#090B16;
  --bg2:#11152A;
  --panel:#121625;
  --panel2:#191D31;
  --line:rgba(168,124,255,.18);

  --text:#F4F1FA;
  --muted:#A7A4B6;

  --purple:#8B5CF6;
  --purple2:#A78BFA;
  --navy:#16213E;
  --slate:#26324A;
  --success:#60B89B;
}

.stApp{
  background:
    radial-gradient(circle at 88% 8%,rgba(139,92,246,.17),transparent 30%),
    radial-gradient(circle at 12% 90%,rgba(38,50,74,.18),transparent 30%),
    linear-gradient(135deg,#090B16 0%,#101426 50%,#151A31 100%);
  color:var(--text);
}

.block-container{
  max-width:1180px;
  padding-top:1.45rem;
  padding-bottom:4rem;
}

h1,h2,h3,h4,h5,h6{color:var(--text)!important;}
p,li,label{color:#D7D2E0;}

#MainMenu,footer,[data-testid="stDecoration"],.stAppDeployButton,[data-testid="stAppDeployButton"]{
  display:none!important;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0B0D18 0%,#12172A 58%,#181C32 100%);
  border-right:1px solid rgba(167,139,250,.14);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
  color:#F5F1FC!important;
}

[data-testid="stSidebar"] [role="radiogroup"] label{
  color:#D0C8DC!important;
}

/* HERO */
.hero{
  position:relative;
  overflow:hidden;
  border:1px solid rgba(167,139,250,.16);
  border-radius:18px;
  padding:24px 26px;
  background:
    radial-gradient(circle at 92% 8%,rgba(139,92,246,.09),transparent 27%),
    linear-gradient(145deg,rgba(18,22,37,.98),rgba(27,31,50,.98));
  box-shadow:0 16px 40px rgba(0,0,0,.28);
  margin-bottom:16px;
}

.hero-kicker{
  color:#A78BFA;
  font-size:10px;
  font-weight:850;
  letter-spacing:1.2px;
}

.hero-title{
  color:#F8F5FC;
  font-size:35px;
  font-weight:850;
  margin-top:5px;
}

.hero-sub{
  color:#AAA6B7;
  font-size:13px;
  line-height:1.65;
  margin-top:7px;
  max-width:800px;
}

/* MODEL READY */
.model-ready{
  margin-top:12px;
  padding:11px 12px;
  border-radius:10px;
  border:1px solid rgba(96,184,155,.22);
  background:rgba(96,184,155,.06);
  color:#8FD3BC;
  font-size:11px;
  font-weight:800;
  display:flex;
  align-items:center;
  gap:8px;
}

.model-dot{
  width:7px;
  height:7px;
  border-radius:50%;
  background:#60B89B;
  box-shadow:0 0 9px rgba(96,184,155,.65);
}

/* INPUTS */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"]{
  background:rgba(18,22,37,.92)!important;
  border:1px dashed rgba(167,139,250,.24)!important;
  border-radius:13px;
  padding:8px;
}

div[data-baseweb="select"]>div,
.stTextInput input,
.stTextArea textarea{
  background:#121625!important;
  border-color:rgba(167,139,250,.16)!important;
  color:#F4F1FA!important;
}

/* BUTTONS */
div.stButton>button{
  min-height:40px;
  border-radius:10px;
  border:1px solid rgba(167,139,250,.20);
  background:linear-gradient(135deg,#6D3FDB 0%,#8B5CF6 55%,#4D5B8A 100%);
  color:white!important;
  font-weight:800;
  box-shadow:0 8px 22px rgba(139,92,246,.18);
}

div.stButton>button:hover{
  color:white!important;
  border-color:#A78BFA;
  box-shadow:0 10px 26px rgba(139,92,246,.22);
  transform:translateY(-1px);
}

/* METRICS */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,rgba(18,22,37,.98),rgba(27,31,50,.98))!important;
  border:1px solid rgba(167,139,250,.14)!important;
  border-radius:13px;
  padding:13px 15px;
  box-shadow:0 10px 26px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"]{color:#A7A4B6!important;}
[data-testid="stMetricValue"]{color:#F4F1FA!important;}

/* RESULT CARDS */
.final-card{
  border:1px solid rgba(167,139,250,.16);
  background:linear-gradient(145deg,#121625,#1A1E33);
  border-radius:16px;
  padding:22px 23px;
  box-shadow:0 12px 30px rgba(0,0,0,.18);
}

.final-label{
  color:#A7A4B6;
  font-size:9px;
  letter-spacing:1.1px;
  font-weight:850;
}

.final-benign{
  color:#7FCDB1;
  font-size:31px;
  font-weight:850;
  margin-top:5px;
}

.final-melanoma{
  color:#B695FF;
  font-size:31px;
  font-weight:850;
  margin-top:5px;
}

.final-other{
  color:#A78BFA;
  font-size:31px;
  font-weight:850;
  margin-top:5px;
}

.other-clean-card{
  border:1px solid rgba(167,139,250,.16);
  background:linear-gradient(145deg,#121625,#191D31);
  border-radius:16px;
  text-align:center;
  padding:30px;
}

.other-circle{
  width:50px;
  height:50px;
  border-radius:50%;
  margin:0 auto 10px auto;
  display:flex;
  align-items:center;
  justify-content:center;
  border:1px solid rgba(167,139,250,.26);
  background:rgba(139,92,246,.08);
  color:#A78BFA;
  font-size:20px;
}

/* CONTAINERS */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:rgba(167,139,250,.13)!important;
  background:rgba(18,22,37,.78)!important;
  border-radius:14px!important;
}

[data-testid="stExpander"]{
  background:linear-gradient(145deg,#121625,#191D31)!important;
  border:1px solid rgba(167,139,250,.13)!important;
  border-radius:14px!important;
}

[data-testid="stAlert"]{border-radius:12px;}
a{color:#A78BFA!important;}
hr{border-color:rgba(167,139,250,.09)!important;}

@media(max-width:760px){
  .hero-title{font-size:28px;}
  .block-container{padding-left:1rem;padding-right:1rem;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def image_to_base64(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((900, 900))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_processing_animation(image: Image.Image, filename: str):
    image_data = image_to_base64(image)
    safe_name = html.escape(str(filename))

    processing_html = """
<style>
.ds-process {
    min-height:390px;
    border:1px solid rgba(167,139,250,.24);
    border-radius:18px;
    background:
        radial-gradient(circle at 28% 47%, rgba(77,91,138,.13), transparent 38%),
        linear-gradient(145deg,#0B0E19,#151A31);
    padding:22px;
    position:relative;
    overflow:hidden;
    font-family:Arial,sans-serif;
    color:#F4F1FA;
}
.ds-process-grid {
    position:absolute; inset:0; opacity:.10;
    background-image:
        linear-gradient(rgba(167,139,250,.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(167,139,250,.13) 1px, transparent 1px);
    background-size:28px 28px;
}
.ds-process-head { position:relative; z-index:5; }
.ds-process-title { font-size:15px; font-weight:750; }
.ds-process-file { margin-top:4px; color:#AAA6B7; font-size:9px; letter-spacing:.75px; }
.ds-process-body {
    position:relative; z-index:5;
    display:grid; grid-template-columns:minmax(320px,.98fr) minmax(300px,1.02fr);
    gap:26px; align-items:center; margin-top:20px;
}
.scan-stage { height:275px; display:flex; align-items:center; justify-content:center; perspective:1000px; }
.scan-object {
    width:355px; height:220px; position:relative; transform-style:preserve-3d;
    animation:imageEnter 1s cubic-bezier(.2,.75,.25,1) forwards;
}
@keyframes imageEnter {
    0% { opacity:0; transform:rotateX(72deg) rotateZ(-12deg) translateY(35px) scale(.78); }
    100% { opacity:1; transform:rotateX(55deg) rotateZ(-7deg) translateY(0) scale(1); }
}
.scan-shadow {
    position:absolute; left:18px; right:18px; top:22px; bottom:-20px;
    transform:translateZ(-30px); border-radius:17px;
    background:rgba(38,50,74,.24); border:1px solid rgba(167,139,250,.10);
}
.image-plane {
    position:absolute; inset:0; overflow:hidden; border-radius:16px; transform:translateZ(12px);
    border:1px solid rgba(167,139,250,.42); background:#101426;
    box-shadow:0 24px 42px rgba(0,0,0,.34), 0 0 24px rgba(167,139,250,.08);
}
.image-plane img { width:100%; height:100%; object-fit:cover; }
.image-filter {
    position:absolute; inset:0;
    background:linear-gradient(90deg,rgba(167,139,250,.035),transparent 50%,rgba(167,139,250,.04));
}
.scan-line {
    position:absolute; left:-3%; top:5%; width:106%; height:4px;
    background:linear-gradient(90deg,transparent,#A78BFA,white,#A78BFA,transparent);
    box-shadow:0 0 10px #A78BFA,0 0 22px rgba(167,139,250,.65);
    animation:scanMove 1.2s ease-in-out infinite alternate;
}
@keyframes scanMove { from { top:5%; } to { top:94%; } }
.scan-radar {
    position:absolute; width:34px; height:34px; left:54%; top:45%; border-radius:50%;
    border:1px solid rgba(167,139,250,.82); box-shadow:0 0 15px rgba(167,139,250,.30);
    animation:radarPulse 1s infinite;
}
.scan-radar:before,.scan-radar:after { content:""; position:absolute; background:rgba(167,139,250,.65); }
.scan-radar:before { left:50%; top:-8px; width:1px; height:50px; }
.scan-radar:after { left:-8px; top:50%; width:50px; height:1px; }
@keyframes radarPulse { 0%,100% { transform:scale(.85); opacity:.45; } 50% { transform:scale(1.05); opacity:1; } }
.corner { position:absolute; width:28px; height:28px; opacity:.9; }
.c1 { top:10px; left:10px; border-top:2px solid #A78BFA; border-left:2px solid #A78BFA; }
.c2 { top:10px; right:10px; border-top:2px solid #A78BFA; border-right:2px solid #A78BFA; }
.c3 { bottom:10px; left:10px; border-bottom:2px solid #A78BFA; border-left:2px solid #A78BFA; }
.c4 { bottom:10px; right:10px; border-bottom:2px solid #A78BFA; border-right:2px solid #A78BFA; }

.pipeline-heading { color:#B3AFBE; font-size:9px; font-weight:700; letter-spacing:1px; margin-bottom:8px; }
.pipeline-window {
    height:186px; position:relative; overflow:hidden; border-radius:12px;
    border:1px solid rgba(167,139,250,.11); background:rgba(3,16,27,.48);
}
.pipeline-track { padding:9px; animation:pipelineScroll 8.6s cubic-bezier(.42,0,.22,1) forwards; }
@keyframes pipelineScroll {
    0%,14% { transform:translateY(0); }
    22%,34% { transform:translateY(-50px); }
    42%,54% { transform:translateY(-100px); }
    62%,74% { transform:translateY(-150px); }
    82%,100% { transform:translateY(-200px); }
}
.pipeline-step {
    height:43px; box-sizing:border-box; display:flex; align-items:center; gap:10px;
    margin-bottom:7px; padding:0 11px; border-radius:9px;
    border:1px solid rgba(103,118,168,.12); background:rgba(6,31,48,.86);
}
.step-number {
    width:23px; height:23px; flex:0 0 23px; display:flex; align-items:center; justify-content:center;
    border-radius:50%; background:#A78BFA; color:#161A28; font-size:9px; font-weight:900;
    box-shadow:0 0 8px rgba(167,139,250,.28);
}
.step-name { color:#EDE8F4; font-size:9px; font-weight:750; }
.step-desc { margin-top:2px; color:#9894A6; font-size:7.5px; }
.live-state {
    margin-top:11px; display:flex; align-items:center; gap:8px; padding:9px 10px;
    border-radius:8px; border:1px solid rgba(96,184,155,.13);
    background:rgba(96,184,155,.045); color:#8FD3BC; font-size:8px;
}
.live-dot { width:7px;height:7px;border-radius:50%;background:#60B89B;box-shadow:0 0 8px #60B89B;animation:pulse .85s infinite; }
@keyframes pulse { 0%,100% {opacity:.35;} 50% {opacity:1;} }

@media(max-width:760px) {
    .ds-process-body { grid-template-columns:1fr; }
    .scan-stage { height:220px; }
    .scan-object { width:290px; height:178px; }
}
</style>

<div class="ds-process">
  <div class="ds-process-grid"></div>
  <div class="ds-process-head">
    <div class="ds-process-title">DermaSense Neural Inference</div>
    <div class="ds-process-file">__FILENAME__</div>
  </div>
  <div class="ds-process-body">
    <div class="scan-stage">
      <div class="scan-object">
        <div class="scan-shadow"></div>
        <div class="image-plane">
          <img src="data:image/jpeg;base64,__IMAGE__" alt="Uploaded skin image" />
          <div class="image-filter"></div>
          <div class="scan-line"></div>
          <div class="scan-radar"></div>
          <div class="corner c1"></div><div class="corner c2"></div>
          <div class="corner c3"></div><div class="corner c4"></div>
        </div>
      </div>
    </div>

    <div>
      <div class="pipeline-heading">AI ANALYSIS PIPELINE</div>
      <div class="pipeline-window">
        <div class="pipeline-track">
          <div class="pipeline-step"><div class="step-number">1</div><div><div class="step-name">IMAGE DECODING</div><div class="step-desc">Read uploaded RGB image</div></div></div>
          <div class="pipeline-step"><div class="step-number">2</div><div><div class="step-name">PREPROCESSING</div><div class="step-desc">Prepare 224 × 224 model input</div></div></div>
          <div class="pipeline-step"><div class="step-number">3</div><div><div class="step-name">MOBILENETV2</div><div class="step-desc">Extract learned visual features</div></div></div>
          <div class="pipeline-step"><div class="step-number">4</div><div><div class="step-name">CLASS RESPONSE</div><div class="step-desc">Calculate learned category responses</div></div></div>
          <div class="pipeline-step"><div class="step-number">5</div><div><div class="step-name">PREDICTION</div><div class="step-desc">Select strongest output category</div></div></div>
        </div>
      </div>
      <div class="live-state"><span class="live-dot"></span>Analyzing image with trained MobileNetV2 model</div>
    </div>
  </div>
</div>
"""
    processing_html = processing_html.replace("__IMAGE__", image_data)
    processing_html = processing_html.replace("__FILENAME__", safe_name)
    st.html(processing_html)


def scroll_to_processing():
    """Smoothly move the browser to the live inference panel after Analyze is clicked."""
    components.html(
        """
        <script>
        setTimeout(function () {
            const doc = window.parent.document;
            const target = doc.getElementById("dermasense-processing-anchor");
            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        }, 120);
        </script>
        """,
        height=0,
    )


def render_connected_3d_skin(image: Image.Image, prediction: str):
    image_data = image_to_base64(image)
    accent = "#60B89B" if prediction == "Benign-like" else "#A78BFA"
    accent_soft = "rgba(96,184,155,.24)" if prediction == "Benign-like" else "rgba(167,139,250,.24)"

    visual_html = """
<style>
.ds3d-wrap {
    width:100%; min-height:500px; padding:24px 20px; box-sizing:border-box;
    border-radius:18px; border:1px solid rgba(121,131,183,.18);
    background:radial-gradient(circle at 50% 35%,rgba(50,130,170,.10),transparent 45%),linear-gradient(160deg,#0B0E19,#171C34);
    overflow:hidden; position:relative; font-family:Arial,sans-serif;
}
.ds3d-grid {
    position:absolute; inset:0; opacity:.10;
    background-image:linear-gradient(rgba(74,185,215,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(74,185,215,.12) 1px,transparent 1px);
    background-size:28px 28px;
}
.ds3d-title { position:relative;z-index:5;color:#F4F1FA;font-size:14px;font-weight:700; }
.ds3d-sub { position:relative;z-index:5;color:#AAA6B7;font-size:9px;letter-spacing:1px;margin-top:4px; }
.ds3d-stage { position:relative;z-index:3;height:340px;display:flex;align-items:center;justify-content:center;perspective:1100px; }
.skin-model {
    width:540px;height:290px;position:relative;transform-style:preserve-3d;
    transform:rotateX(58deg) rotateZ(-12deg) translateY(10px);
    animation:revealSkin 4.8s cubic-bezier(.2,.75,.25,1) forwards;
}
@keyframes revealSkin {
    0% { transform:rotateX(72deg) rotateZ(-20deg) scale(.78) translateY(35px); opacity:0; }
    20% { opacity:1; }
    70% { transform:rotateX(54deg) rotateZ(-9deg) scale(1.02) translateY(4px); }
    100% { transform:rotateX(58deg) rotateZ(-12deg) scale(1) translateY(10px); }
}
.skin-layer { position:absolute;left:20px;width:500px;border-radius:18px;overflow:hidden;box-shadow:0 18px 36px rgba(0,0,0,.26); }
.surface { top:0;height:165px;z-index:10;border:1px solid __ACCENT_SOFT__;background:#111; }
.surface img { width:100%;height:100%;object-fit:cover; }
.scan-glow {
    position:absolute;left:0;top:0;width:100%;height:4px;
    background:linear-gradient(90deg,transparent,__ACCENT__,white,__ACCENT__,transparent);
    box-shadow:0 0 18px __ACCENT__; animation:scanSurface 2.6s ease-in-out 1 .8s forwards;
}
@keyframes scanSurface { 0%{top:4%;opacity:.3;}15%{opacity:1;}85%{opacity:1;}100%{top:94%;opacity:0;} }
.epidermis { top:164px;height:26px;z-index:8;background:linear-gradient(90deg,#d89d90,#ba797a,#d89d90); }
.dermis { top:188px;height:61px;z-index:7;background:linear-gradient(90deg,#a85f70,#78445b,#a85f70); }
.subcutaneous {
    top:246px;height:45px;z-index:6;
    background:radial-gradient(circle at 15px 14px,#dba84b 0 6px,#b97b31 7px 11px,transparent 12px),#9f6930;
    background-size:32px 28px;
}
.vessel { position:absolute;z-index:15;height:4px;border-radius:10px;opacity:0;transform-origin:left center; }
.vessel-red { width:280px;left:120px;top:215px;background:linear-gradient(90deg,transparent,#ff5878,#ff879b,transparent);animation:redReveal 1.3s ease-out 2.8s forwards; }
.vessel-blue { width:250px;left:180px;top:228px;background:linear-gradient(90deg,transparent,#4b9eff,#73c1ff,transparent);animation:blueReveal 1.3s ease-out 3s forwards; }
.nerve {
    position:absolute;z-index:16;width:190px;height:3px;left:80px;top:237px;border-radius:8px;
    background:linear-gradient(90deg,transparent,#ffe067,#fff0a0,transparent);opacity:0;transform-origin:left center;
    animation:nerveReveal 1.2s ease-out 3.2s forwards;
}
@keyframes redReveal { from{opacity:0;transform:scaleX(.15) rotate(-7deg);} to{opacity:.9;transform:scaleX(1) rotate(-7deg);} }
@keyframes blueReveal { from{opacity:0;transform:scaleX(.15) rotate(5deg);} to{opacity:.9;transform:scaleX(1) rotate(5deg);} }
@keyframes nerveReveal { from{opacity:0;transform:scaleX(.15) rotate(10deg);} to{opacity:.9;transform:scaleX(1) rotate(10deg);} }
.corner3d { position:absolute;width:34px;height:34px;opacity:.78; }
.k1{top:9px;left:9px;border-top:2px solid __ACCENT__;border-left:2px solid __ACCENT__;}
.k2{top:9px;right:9px;border-top:2px solid __ACCENT__;border-right:2px solid __ACCENT__;}
.k3{bottom:9px;left:9px;border-bottom:2px solid __ACCENT__;border-left:2px solid __ACCENT__;}
.k4{bottom:9px;right:9px;border-bottom:2px solid __ACCENT__;border-right:2px solid __ACCENT__;}
.info-grid { position:relative;z-index:4;display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:5px; }
.info-card { border:1px solid rgba(100,180,205,.14);background:rgba(13,16,29,.82);border-radius:9px;padding:9px 10px; }
.info-label { color:#9290A0;font-size:8px;text-transform:uppercase;letter-spacing:.7px; }
.info-value { margin-top:4px;color:#EDE8F4;font-size:9px;font-weight:700; }
.ds3d-note { position:relative;z-index:4;margin-top:10px;color:#9290A0;font-size:8px;line-height:1.5; }
@media(max-width:700px) { .ds3d-stage{height:290px;} .skin-model{transform:rotateX(57deg) rotateZ(-8deg) scale(.68);} .info-grid{grid-template-columns:repeat(2,1fr);} }
</style>

<div class="ds3d-wrap">
  <div class="ds3d-grid"></div>
  <div class="ds3d-title">Technical 3D Skin Visualization</div>
  <div class="ds3d-sub">UPLOADED IMAGE • ILLUSTRATIVE SKIN-LAYER MODEL</div>
  <div class="ds3d-stage">
    <div class="skin-model">
      <div class="skin-layer surface">
        <img src="data:image/jpeg;base64,__IMAGE__" alt="Uploaded image" />
        <div class="scan-glow"></div>
        <div class="corner3d k1"></div><div class="corner3d k2"></div><div class="corner3d k3"></div><div class="corner3d k4"></div>
      </div>
      <div class="skin-layer epidermis"></div>
      <div class="skin-layer dermis"></div>
      <div class="skin-layer subcutaneous"></div>
      <div class="vessel vessel-red"></div>
      <div class="vessel vessel-blue"></div>
      <div class="nerve"></div>
    </div>
  </div>
  <div class="info-grid">
    <div class="info-card"><div class="info-label">Surface Input</div><div class="info-value">Uploaded Image</div></div>
    <div class="info-card"><div class="info-label">Upper Layer</div><div class="info-value">Epidermis</div></div>
    <div class="info-card"><div class="info-label">Support Layer</div><div class="info-value">Dermis</div></div>
    <div class="info-card"><div class="info-label">Illustrative Network</div><div class="info-value">Nerves + Vessels</div></div>
  </div>
  <div class="ds3d-note">The uploaded photograph is used as the visible surface. Deeper skin layers, nerves and vessels are illustrative educational graphics and are not reconstructed from the photograph.</div>
</div>
"""
    visual_html = visual_html.replace("__IMAGE__", image_data)
    visual_html = visual_html.replace("__ACCENT__", accent)
    visual_html = visual_html.replace("__ACCENT_SOFT__", accent_soft)
    st.html(visual_html)


def render_final_prediction(prediction: str, confidence: float, probabilities=None):
    if prediction == "Other":
        st.markdown(
            """<div class="other-clean-card">
<div class="other-circle">?</div>
<div class="final-label">CLASSIFICATION OUTPUT</div>
<div class="final-other">Other</div>
</div>""",
            unsafe_allow_html=True,
        )
        return

    probabilities = probabilities or {}

    css_class = "final-benign" if prediction == "Benign-like" else "final-melanoma"
    description = (
        "The strongest supported model response matched patterns learned from the Benign-like category."
        if prediction == "Benign-like"
        else "The strongest supported model response matched patterns learned from melanoma examples. This does not confirm melanoma."
    )

    st.markdown(
        f'<div class="final-card"><div class="final-label">CLASSIFICATION OUTPUT</div>'
        f'<div class="{css_class}">{html.escape(prediction)}</div>'
        f'<div style="margin-top:7px;color:#A7A4B6;font-size:12px;line-height:1.55">{html.escape(description)}</div></div>',
        unsafe_allow_html=True,
    )

    # Show the two real supported-class model responses.
    # These values come directly from predict_lesion(); they are model outputs,
    # not medical diagnosis probabilities.
    benign_value = probabilities.get("benign", probabilities.get("Benign-like"))
    melanoma_value = probabilities.get("melanoma", probabilities.get("Melanoma-suspicious"))

    st.write("")
    col_benign, col_melanoma = st.columns(2)

    if benign_value is None:
        col_benign.metric("Benign-like Response", "Not available")
    else:
        col_benign.metric("Benign-like Response", f"{float(benign_value) * 100:.1f}%")

    if melanoma_value is None:
        col_melanoma.metric("Melanoma Response", "Not available")
    else:
        col_melanoma.metric("Melanoma Response", f"{float(melanoma_value) * 100:.1f}%")


def render_gradcam(model, metadata, image, prediction):
    st.markdown("## Model Attention Analysis")
    st.caption("Grad-CAM shows image regions that had greater influence on the model response.")

    class_names = metadata.get(
        "class_names",
        ["benign", "melanoma", "non_skin", "other_skin"],
    )
    target = "benign" if prediction == "Benign-like" else "melanoma"

    try:
        class_index = class_names.index(target)
        image_size = int(metadata.get("image_size", 224))
        result = create_gradcam_result(
            model=model,
            image=image,
            class_index=class_index,
            image_size=image_size,
        )

        _, c1, c2, _ = st.columns([0.22, 1, 1, 0.22])
        with c1:
            st.caption("ORIGINAL IMAGE")
            st.image(image, use_container_width=True)
        with c2:
            st.caption("AI ATTENTION / GRAD-CAM")
            st.image(result["overlay"], use_container_width=True)

        st.markdown("### Why did the AI make this prediction?")
        st.markdown(
            "The highlighted regions indicate areas that influenced the neural-network output more strongly. "
            "Grad-CAM visualizes **model attention**; it does not identify the exact location of cancer or provide a diagnosis."
        )
    except Exception:
        st.info("Grad-CAM visualization is unavailable for this analysis.")


def generate_result_explanation(prediction, probabilities):
    prompt = f"""
The latest DermaSense machine-learning result is: {prediction}

Create a concise educational explanation with exactly these sections:

### Why this prediction?
### Possible Causes / Risk Factors
### Possible Effects / Concerns
### General Treatment / Management
### What Should Someone Generally Do Next?

Explain MobileNetV2 and Grad-CAM simply. Never state that the system diagnoses cancer.
Do not prescribe medication or doses. End by saying that DermaSense is an educational decision-support prototype, not a medical diagnosis system.
"""
    try:
        return dermaguide_reply(
            prompt,
            prediction=prediction,
            probabilities=probabilities,
            history=[],
        )
    except Exception:
        return (
            "### Why this prediction?\n\n"
            "The trained MobileNetV2 model extracted visual features from the image and selected the supported category with the strongest learned response.\n\n"
            "### What Should Someone Generally Do Next?\n\n"
            "A concerning or changing skin lesion should be assessed by a qualified healthcare professional.\n\n"
            "DermaSense is an educational decision-support prototype, not a medical diagnosis system."
        )


# =========================================================
# DATABASE
# =========================================================
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


def add_history(filename, prediction, confidence, melanoma_score):
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        previous = conn.execute(
            "SELECT timestamp, filename, prediction, confidence FROM history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if previous:
            try:
                previous_time = datetime.strptime(previous[0], "%Y-%m-%d %H:%M:%S")
                seconds = (now - previous_time).total_seconds()
            except Exception:
                seconds = 999
            duplicate = (
                previous[1] == filename
                and previous[2] == prediction
                and abs(float(previous[3]) - float(confidence)) < 0.0001
            )
            if duplicate and seconds <= 10:
                return

        conn.execute(
            """
            INSERT INTO history
            (timestamp, filename, prediction, confidence, melanoma_score, score_band)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now.strftime("%Y-%m-%d %H:%M:%S"),
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
        return pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 100", conn)


def delete_history(history_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (int(history_id),))
        conn.commit()


def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history")
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
    st.title("🔬 DermaSense AI")
    st.caption("AI-Powered Skin Image Analysis")
    page = st.radio("Navigation", ["Analyze", "History"], label_visibility="collapsed")
    st.divider()

    if MODEL_PATH.exists():
        st.markdown(
            '<div class="model-ready"><span class="model-dot"></span>Model Loaded & Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Model unavailable")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">EXPLAINABLE MACHINE LEARNING</div>
  <div class="hero-title">DermaSense AI</div>
  <div class="hero-sub">Skin-image analysis using MobileNetV2 transfer learning, Grad-CAM explainability, technical 3D visualization, camera/upload input and prediction history.</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================
model = None
metadata = {}
if MODEL_PATH.exists():
    try:
        model = get_model()
        metadata = get_metadata()
    except Exception as exc:
        st.error(f"Unable to load trained model: {exc}")


# =========================================================
# ANALYZE PAGE
# =========================================================
if page == "Analyze":
    st.warning(
        "**Educational prototype.** DermaSense is not a medical diagnostic system and does not replace professional examination."
    )

    st.markdown("## Image Input")
    input_mode = st.radio("Input source", ["Upload Image", "Use Camera"], horizontal=True)
    valid_images = []

    if input_mode == "Upload Image":
        uploaded_files = st.file_uploader(
            "Upload skin image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            cols = st.columns(min(len(uploaded_files), 4))
            for idx, uploaded in enumerate(uploaded_files):
                try:
                    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                    valid_images.append((uploaded.name, image))
                    with cols[idx % len(cols)]:
                        st.image(image, caption=uploaded.name, width=135)
                except (UnidentifiedImageError, OSError):
                    st.error(f"{uploaded.name} is not a valid image.")
    else:
        captured = st.camera_input("Capture skin image", label_visibility="collapsed")
        if captured:
            try:
                image = Image.open(io.BytesIO(captured.getvalue())).convert("RGB")
                filename = "camera_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                valid_images.append((filename, image))
                st.image(image, caption="Captured image", width=150)
            except (UnidentifiedImageError, OSError):
                st.error("Unable to read camera image.")

    if valid_images and model is not None:
        _, center, _ = st.columns([2.4, .8, 2.4])
        with center:
            analyze_clicked = st.button("Analyze", use_container_width=True)

        if analyze_clicked:
            st.session_state.result_explanations = {}
            new_results = []

            for filename, image in valid_images:
                # Move directly to the live prediction process as soon as Analyze is clicked.
                st.markdown(
                    '<div id="dermasense-processing-anchor" style="scroll-margin-top:18px;"></div>',
                    unsafe_allow_html=True,
                )

                processing_placeholder = st.empty()
                with processing_placeholder.container():
                    st.markdown("## Neural Inference")
                    render_processing_animation(image, filename)

                scroll_to_processing()

                inference_start = time.perf_counter()
                try:
                    result = predict_lesion(model, image, metadata)
                except Exception as exc:
                    processing_placeholder.empty()
                    st.error(f"Analysis failed for {filename}: {exc}")
                    continue
                inference_time = time.perf_counter() - inference_start

                # Keep the original video-like processing experience.
                if inference_time < PROCESSING_SECONDS:
                    time.sleep(PROCESSING_SECONDS - inference_time)

                processing_placeholder.empty()

                prediction = result.get("prediction", "Other")
                confidence = float(result.get("confidence", 0.0))
                probabilities = result.get("probabilities", {})
                melanoma_score = float(
                    result.get("melanoma_score", probabilities.get("melanoma", 0.0))
                )

                new_results.append(
                    {
                        "filename": filename,
                        "image": image,
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "melanoma_score": melanoma_score,
                        "inference_time": inference_time,
                    }
                )
                add_history(filename, prediction, confidence, melanoma_score)

            st.session_state.analysis_results = new_results

    if st.session_state.analysis_results:
        for result_index, result in enumerate(st.session_state.analysis_results, start=1):
            filename = result["filename"]
            image = result["image"]
            prediction = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            if len(st.session_state.analysis_results) > 1:
                st.markdown(f"## Result {result_index}")

            # -------------------------------------------------
            # OTHER: stop here exactly as before
            # -------------------------------------------------
            if prediction == "Other":
                st.markdown("## Classification Output")
                render_final_prediction(prediction, confidence, probabilities)
                st.markdown("## Why this prediction?")
                st.info(
                    """
The uploaded image produced stronger patterns for the model's **Other / rejection category**.

It did not match the learned **Benign-like** or **Melanoma-suspicious** patterns strongly enough.

DermaSense therefore returns **Other** instead of forcing the image into one of its supported skin-lesion classifications.
"""
                )
                if result_index < len(st.session_state.analysis_results):
                    st.divider()
                continue

            # -------------------------------------------------
            # ORIGINAL VISUAL FLOW
            # -------------------------------------------------
            st.markdown("## Structural Visualization")
            render_connected_3d_skin(image, prediction)

            # Show the prediction immediately after the 3D structural visualization.
            st.markdown("## Prediction Result")
            render_final_prediction(prediction, confidence, probabilities)

            # Then explain where the model focused.
            render_gradcam(model, metadata, image, prediction)

            st.markdown("## AI Result Explanation")
            explanation_key = f"{filename}_{prediction}_{result_index}"
            if explanation_key not in st.session_state.result_explanations:
                with st.spinner("Generating explanation..."):
                    st.session_state.result_explanations[explanation_key] = generate_result_explanation(
                        prediction,
                        probabilities,
                    )
            st.markdown(st.session_state.result_explanations[explanation_key])

            st.warning(
                "**Medical Safety Notice:** DermaSense is an educational AI/ML decision-support prototype. Its prediction, confidence score, Grad-CAM visualization and 3D illustration must not be interpreted as a medical diagnosis."
            )

            if result_index < len(st.session_state.analysis_results):
                st.divider()


# =========================================================
# HISTORY PAGE
# =========================================================
elif page == "History":
    st.markdown("## Analysis History")
    st.caption("Prediction metadata is stored locally. Uploaded image files themselves are not stored.")

    history = read_history()
    if history.empty:
        st.info("No analysis history yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(history))
        c2.metric("Benign-like", int((history["prediction"] == "Benign-like").sum()))
        c3.metric("Melanoma-suspicious", int((history["prediction"] == "Melanoma-suspicious").sum()))
        c4.metric("Other", int((history["prediction"] == "Other").sum()))

        st.write("")
        for _, row in history.iterrows():
            with st.container(border=True):
                cols = st.columns([1.5, 2, 1.8, 1, .65])
                cols[0].write(row["timestamp"])
                cols[1].write(row["filename"])
                cols[2].write(row["prediction"])
                if row["prediction"] == "Other":
                    cols[3].write("—")
                else:
                    cols[3].write(f"{float(row['confidence']) * 100:.1f}%")
                if cols[4].button("Delete", key=f"delete_{int(row['id'])}"):
                    delete_history(row["id"])
                    st.rerun()

        st.write("")
        if st.button("Clear All History"):
            clear_history()
            st.rerun()
