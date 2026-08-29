from pathlib import Path
import json

import numpy as np
from PIL import Image
import streamlit as st

MODEL_PATH = Path("models") / "skin_lesion_model.keras"
METADATA_PATH = Path("models") / "metadata.json"

DEFAULT_METADATA = {
    "image_size": 224,
    "threshold": 0.5,
    "class_names": ["benign", "melanoma"],
}


@st.cache_resource(show_spinner=False)
def load_trained_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def load_metadata():
    if not METADATA_PATH.exists():
        return DEFAULT_METADATA.copy()

    try:
        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        merged = DEFAULT_METADATA.copy()
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return DEFAULT_METADATA.copy()


def prepare_image(image: Image.Image, image_size: int):
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_lesion(model, image: Image.Image, metadata=None):
    metadata = metadata or DEFAULT_METADATA
    image_size = int(metadata.get("image_size", 224))
    threshold = float(metadata.get("threshold", 0.5))

    batch = prepare_image(image, image_size)
    raw = model.predict(batch, verbose=0)
    melanoma_score = float(np.asarray(raw).reshape(-1)[0])
    melanoma_score = float(np.clip(melanoma_score, 0.0, 1.0))

    if melanoma_score >= threshold:
        prediction = "Melanoma-suspicious"
        confidence = melanoma_score
    else:
        prediction = "Benign-like"
        confidence = 1.0 - melanoma_score

    return {
        "prediction": prediction,
        "confidence": confidence,
        "melanoma_score": melanoma_score,
        "threshold": threshold,
        "image_size": image_size,
    }
