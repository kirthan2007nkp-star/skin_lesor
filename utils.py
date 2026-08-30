from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models") / "skin_lesion_model.keras"
METADATA_PATH = Path("models") / "metadata.json"


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata():

    if not METADATA_PATH.exists():

        return {
            "class_names": [
                "benign",
                "melanoma",
                "non_skin",
                "other_skin",
            ],
            "image_size": 224,
        }

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# LOAD MODEL
# =========================================================

def load_trained_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================================================
# PREPARE IMAGE
# =========================================================

def prepare_image(
    image: Image.Image,
    image_size: int,
):

    image = image.convert("RGB")

    image = image.resize(
        (
            image_size,
            image_size,
        )
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    array = np.expand_dims(
        array,
        axis=0,
    )

    return array


# =========================================================
# PREDICTION
# =========================================================

def predict_lesion(
    model,
    image,
    metadata=None,
):

    if metadata is None:

        metadata = load_metadata()


    image_size = int(
        metadata.get(
            "image_size",
            224,
        )
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


    prepared = prepare_image(
        image,
        image_size,
    )


    predictions = model.predict(
        prepared,
        verbose=0,
    )


    raw_probabilities = np.asarray(
        predictions[0],
        dtype=float,
    )


    probability_map = {
        class_names[index]:
            float(
                raw_probabilities[index]
            )

        for index in range(
            min(
                len(class_names),
                len(raw_probabilities),
            )
        )
    }


    # =====================================================
    # FINAL 3 APP CATEGORIES
    # =====================================================

    benign_score = float(
        probability_map.get(
            "benign",
            0.0,
        )
    )


    melanoma_score = float(
        probability_map.get(
            "melanoma",
            0.0,
        )
    )


    # Both internal categories become "Other"
    other_score = float(

        probability_map.get(
            "non_skin",
            0.0,
        )

        +

        probability_map.get(
            "other_skin",
            0.0,
        )
    )


    final_probabilities = {

        "benign":
            benign_score,

        "melanoma":
            melanoma_score,

        "other":
            other_score,
    }


    # Pick highest final category
    best_class = max(
        final_probabilities,
        key=final_probabilities.get,
    )


    labels = {

        "benign":
            "Benign-like",

        "melanoma":
            "Melanoma-suspicious",

        "other":
            "Other",
    }


    prediction_label = labels[
        best_class
    ]


    confidence = float(
        final_probabilities[
            best_class
        ]
    )


    return {

        "prediction":
            prediction_label,

        "class_key":
            best_class,

        "confidence":
            confidence,

        "melanoma_score":
            melanoma_score,

        "probabilities":
            final_probabilities,

        "raw_probabilities":
            probability_map,

        "image_size":
            image_size,

        "low_confidence":
            False,
    }