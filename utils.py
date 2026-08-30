from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models") / "skin_lesion_model.keras"
METADATA_PATH = Path("models") / "metadata.json"


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


def load_trained_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


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


    prepared = prepare_image(
        image,
        image_size,
    )


    predictions = model.predict(
        prepared,
        verbose=0,
    )


    probabilities = np.asarray(
        predictions[0],
        dtype=float,
    )


    predicted_index = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    class_key = class_names[
        predicted_index
    ]


    probability_map = {

        class_names[index]:
            float(probabilities[index])

        for index in range(
            len(class_names)
        )
    }


    melanoma_score = float(
        probability_map.get(
            "melanoma",
            0.0,
        )
    )


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


    return {

        "prediction":
            prediction_label,

        "class_key":
            class_key,

        "confidence":
            confidence,

        "melanoma_score":
            melanoma_score,

        "probabilities":
            probability_map,

        "image_size":
            image_size,

        "low_confidence":
            False,

        "rejection_reason":
            None,
    }