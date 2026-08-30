from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models") / "skin_lesion_model.keras"
METADATA_PATH = Path("models") / "metadata.json"


# =========================================================
# SETTINGS
# =========================================================

# The model must be at least this confident
# before we trust its main classification.
MIN_CONFIDENCE = 0.75

# The best prediction should also be clearly
# ahead of the second-best prediction.
MIN_CLASS_MARGIN = 0.25


# =========================================================
# METADATA
# =========================================================

def load_metadata():

    if not METADATA_PATH.exists():

        return {
            "class_names": [
                "benign",
                "melanoma",
                "non_skin",
                "other",
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
# MODEL
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
# IMAGE PREPARATION
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
            "other",
        ],
    )


    # -----------------------------------------------------
    # Prepare image
    # -----------------------------------------------------

    prepared = prepare_image(
        image,
        image_size,
    )


    # -----------------------------------------------------
    # Run CNN
    # -----------------------------------------------------

    predictions = model.predict(
        prepared,
        verbose=0,
    )


    probabilities = np.asarray(
        predictions[0],
        dtype=float,
    )


    # -----------------------------------------------------
    # Probability map
    # -----------------------------------------------------

    probability_map = {
        class_names[index]:
            float(probabilities[index])

        for index in range(
            len(class_names)
        )
    }


    # -----------------------------------------------------
    # Sort predictions
    # -----------------------------------------------------

    sorted_indices = np.argsort(
        probabilities
    )[::-1]


    best_index = int(
        sorted_indices[0]
    )


    second_index = int(
        sorted_indices[1]
    )


    best_probability = float(
        probabilities[
            best_index
        ]
    )


    second_probability = float(
        probabilities[
            second_index
        ]
    )


    class_margin = (
        best_probability
        - second_probability
    )


    class_key = class_names[
        best_index
    ]


    melanoma_score = float(
        probability_map.get(
            "melanoma",
            0.0,
        )
    )


    # -----------------------------------------------------
    # Human-readable labels
    # -----------------------------------------------------

    labels = {

        "benign":
            "Benign-like",

        "melanoma":
            "Melanoma-suspicious",

        "other":
            "Other skin lesion",

        "non_skin":
            "Unsupported image",
    }


    prediction_label = labels.get(
        class_key,
        class_key,
    )


    # =====================================================
    # SAFETY RULES
    # =====================================================

    low_confidence = False

    rejection_reason = None


    # Rule 1:
    # The model itself thinks the image is non-skin.
    if class_key == "non_skin":

        prediction_label = (
            "Unsupported image"
        )

        rejection_reason = (
            "The image appears outside "
            "the supported skin-lesion classes."
        )


    # Rule 2:
    # Model confidence is too low.
    elif best_probability < MIN_CONFIDENCE:

        prediction_label = (
            "Uncertain / Unsupported"
        )

        low_confidence = True

        rejection_reason = (
            "The model confidence is too low "
            "for a reliable classification."
        )


    # Rule 3:
    # Best and second-best classes are too close.
    elif class_margin < MIN_CLASS_MARGIN:

        prediction_label = (
            "Uncertain / Unsupported"
        )

        low_confidence = True

        rejection_reason = (
            "The model cannot clearly distinguish "
            "between its top predictions."
        )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "prediction":
            prediction_label,

        "class_key":
            class_key,

        "confidence":
            best_probability,

        "melanoma_score":
            melanoma_score,

        "probabilities":
            probability_map,

        "image_size":
            image_size,

        "low_confidence":
            low_confidence,

        "class_margin":
            class_margin,

        "second_best_confidence":
            second_probability,

        "rejection_reason":
            rejection_reason,
    }