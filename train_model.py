from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = Path("dataset_v2")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "skin_lesion_model.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42

EXPECTED_CLASSES = [
    "benign",
    "melanoma",
    "non_skin",
    "other_skin",
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading training data...")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    print("\nLoading validation data...")

    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    print("\nLoading test data...")

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    print("\nClass order:")
    print(class_names)

    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected classes: {EXPECTED_CLASSES}\n"
            f"Found classes: {class_names}"
        )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        class_names,
    )


# =========================================================
# CLASS WEIGHTS
# =========================================================

def get_class_weights():

    counts = {
        0: 916,   # benign
        1: 903,   # melanoma
        2: 1120,  # non_skin
        3: 684,   # other_skin
    }

    total = sum(
        counts.values()
    )

    number_of_classes = len(
        counts
    )

    weights = {}

    for class_index, count in counts.items():

        weights[class_index] = (
            total
            / (
                number_of_classes
                * count
            )
        )

    print("\nClass weights:")

    for key, value in weights.items():

        print(
            f"{key}: {value:.3f}"
        )

    return weights


# =========================================================
# BUILD MODEL
# =========================================================

def build_model():

    print(
        "\nLoading MobileNetV2..."
    )

    base_model = (
        keras.applications.MobileNetV2(
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            ),
            include_top=False,
            weights="imagenet",
        )
    )

    # Freeze pretrained feature extractor
    base_model.trainable = False


    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.08
            ),

            layers.RandomZoom(
                0.08
            ),

            layers.RandomContrast(
                0.08
            ),
        ],
        name="augmentation",
    )


    inputs = keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )
    )


    x = augmentation(
        inputs
    )


    # Convert 0-255 pixels to MobileNetV2 range
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(x)


    x = base_model(
        x,
        training=False,
    )


    x = layers.GlobalAveragePooling2D()(
        x
    )


    x = layers.Dropout(
        0.35
    )(x)


    x = layers.Dense(
        128,
        activation="relu",
    )(x)


    x = layers.Dropout(
        0.25
    )(x)


    outputs = layers.Dense(
        4,
        activation="softmax",
        name="class_probabilities",
    )(x)


    model = keras.Model(
        inputs,
        outputs,
        name="DermaSense_MobileNetV2",
    )


    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


    return model


# =========================================================
# MAIN
# =========================================================

def main():

    tf.random.set_seed(
        SEED
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        train_ds,
        val_ds,
        test_ds,
        class_names,
    ) = load_data()


    class_weights = (
        get_class_weights()
    )


    model = build_model()


    model.summary()


    callbacks = [

        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=0.000001,
            verbose=1,
        ),
    ]


    print(
        "\n=================================="
    )

    print(
        "Training DermaSense AI"
    )

    print(
        "MobileNetV2 Transfer Learning"
    )

    print(
        "==================================\n"
    )


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    best_model = keras.models.load_model(
        MODEL_PATH
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    validation_results = (
        best_model.evaluate(
            val_ds,
            verbose=0,
            return_dict=True,
        )
    )


    # =====================================================
    # FINAL TEST
    # =====================================================

    test_results = (
        best_model.evaluate(
            test_ds,
            verbose=0,
            return_dict=True,
        )
    )


    # =====================================================
    # SAVE METADATA
    # =====================================================

    metadata = {

        "project":
            "DermaSense AI",

        "architecture":
            "MobileNetV2 Transfer Learning",

        "pretrained":
            True,

        "pretrained_dataset":
            "ImageNet",

        "class_names":
            class_names,

        "display_mapping": {

            "benign":
                "Benign-like",

            "melanoma":
                "Melanoma-suspicious",

            "non_skin":
                "Other",

            "other_skin":
                "Other",
        },

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "dataset":
            "HAM10000 + non-skin examples",

        "split_method":
            (
                "Separate train, validation "
                "and test sets"
            ),

        "educational_use_only":
            True,
    }


    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics = {

        "validation": {
            key: float(value)
            for key, value
            in validation_results.items()
        },

        "test": {
            key: float(value)
            for key, value
            in test_results.items()
        },

        "best_validation_accuracy":
            float(
                max(
                    history.history[
                        "val_accuracy"
                    ]
                )
            ),
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n=================================="
    )

    print(
        "TRAINING COMPLETE ✅"
    )

    print(
        "=================================="
    )


    print(
        "\nValidation Results:"
    )

    for key, value in (
        validation_results.items()
    ):

        print(
            f"{key}: {value:.4f}"
        )


    print(
        "\nFINAL UNSEEN TEST RESULTS:"
    )

    for key, value in (
        test_results.items()
    ):

        print(
            f"{key}: {value:.4f}"
        )


    print(
        f"\nModel saved:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()