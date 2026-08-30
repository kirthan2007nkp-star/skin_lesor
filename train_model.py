from pathlib import Path
import argparse
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SUPPORTED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_images(folder: Path):
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED
    )


def build_datasets(
    dataset_dir,
    image_size,
    batch_size,
    seed,
):
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )

    expected_classes = [
        "benign",
        "melanoma",
        "non_skin",
        "other",
    ]

    if train_ds.class_names != expected_classes:
        raise ValueError(
            "Expected folders:\n"
            f"{expected_classes}\n\n"
            "But found:\n"
            f"{train_ds.class_names}"
        )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_ds,
        val_ds,
        class_names,
    )


def build_model(
    image_size,
    number_of_classes,
):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),
            layers.RandomRotation(
                0.10
            ),
            layers.RandomZoom(
                0.10
            ),
            layers.RandomContrast(
                0.10
            ),
        ],
        name="augmentation",
    )

    model = keras.Sequential(
        [
            keras.Input(
                shape=(
                    image_size,
                    image_size,
                    3,
                )
            ),

            augmentation,

            layers.Rescaling(
                1.0 / 255
            ),

            layers.Conv2D(
                32,
                3,
                padding="same",
                activation="relu",
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(),

            layers.Conv2D(
                64,
                3,
                padding="same",
                activation="relu",
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(),

            layers.Conv2D(
                128,
                3,
                padding="same",
                activation="relu",
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(),

            layers.Conv2D(
                256,
                3,
                padding="same",
                activation="relu",
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(
                256,
                activation="relu",
            ),

            layers.Dropout(
                0.40
            ),

            layers.Dense(
                128,
                activation="relu",
            ),

            layers.Dropout(
                0.30
            ),

            layers.Dense(
                number_of_classes,
                activation="softmax",
                name="class_probabilities",
            ),
        ],
        name="DermaSense_Custom_4_Class_CNN",
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train DermaSense "
            "4-class CNN from scratch."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        default="dataset",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    folders = {
        "benign":
            dataset_dir / "benign",

        "melanoma":
            dataset_dir / "melanoma",

        "non_skin":
            dataset_dir / "non_skin",

        "other":
            dataset_dir / "other",
    }

    for name, folder in folders.items():

        if not folder.exists():

            raise SystemExit(
                f"Missing folder: "
                f"dataset/{name}"
            )

    counts = {}

    print(
        "\nDataset image counts"
    )

    print(
        "--------------------"
    )

    for name, folder in folders.items():

        count = count_images(
            folder
        )

        counts[name] = count

        print(
            f"{name}: {count}"
        )

        if count < 100:

            raise SystemExit(
                f"Not enough images "
                f"in dataset/{name}"
            )

    (
        train_ds,
        val_ds,
        class_names,
    ) = build_datasets(
        dataset_dir,
        args.image_size,
        args.batch_size,
        args.seed,
    )

    print(
        "\nClass order:"
    )

    for index, name in enumerate(
        class_names
    ):

        print(
            f"{index} = {name}"
        )

    model = build_model(
        args.image_size,
        len(class_names),
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy",
        ],
    )

    model.summary()

    model_dir = Path(
        "models"
    )

    model_dir.mkdir(
        exist_ok=True
    )

    model_path = (
        model_dir
        / "skin_lesion_model.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    total_images = sum(
        counts.values()
    )

    class_weight = {}

    for index, class_name in enumerate(
        class_names
    ):

        class_weight[index] = (
            total_images
            / (
                len(class_names)
                * counts[class_name]
            )
        )

    print(
        "\nClass weights:"
    )

    print(
        class_weight
    )

    print(
        "\nTraining custom "
        "4-class CNN from scratch...\n"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    best_model = (
        keras.models.load_model(
            model_path
        )
    )

    results = best_model.evaluate(
        val_ds,
        verbose=0,
        return_dict=True,
    )

    metadata = {
        "project":
            "DermaSense AI",

        "architecture":
            "Custom 4-class CNN trained from scratch",

        "pretrained":
            False,

        "class_names":
            class_names,

        "image_size":
            args.image_size,

        "classes": {
            "benign":
                "Benign-like skin lesion",

            "melanoma":
                "Melanoma-suspicious skin lesion",

            "other":
                "Other skin lesion",

            "non_skin":
                "Unsupported non-skin image",
        },

        "dataset_counts":
            counts,

        "validation_split":
            0.20,

        "seed":
            args.seed,

        "note":
            (
                "Educational AI prototype. "
                "Not a medical diagnostic device."
            ),
    }

    with open(
        model_dir / "metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    with open(
        model_dir / "metrics.json",
        "w",
    ) as file:

        json.dump(
            {
                key: float(value)
                for key, value
                in results.items()
            },
            file,
            indent=2,
        )

    print(
        "\nTraining complete."
    )

    print(
        f"Model saved to: "
        f"{model_path}"
    )

    print(
        "\nValidation results:"
    )

    for key, value in results.items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )


if __name__ == "__main__":
    main()