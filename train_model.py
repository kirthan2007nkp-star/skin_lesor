from pathlib import Path
import argparse
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path):
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED
    )


def build_datasets(dataset_dir, image_size, batch_size, seed):
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
    )

    expected_classes = ["benign", "melanoma"]

    if train_ds.class_names != expected_classes:
        raise ValueError(
            f"Expected folders {expected_classes}, "
            f"but found {train_ds.class_names}"
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, expected_classes


def build_model(image_size):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, 3)),

            augmentation,

            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),

            layers.Dense(
                1,
                activation="sigmoid",
                name="melanoma_probability"
            ),
        ],
        name="DermaSense_Custom_CNN",
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train DermaSense custom CNN from scratch."
    )

    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    benign_dir = dataset_dir / "benign"
    melanoma_dir = dataset_dir / "melanoma"

    if not benign_dir.exists() or not melanoma_dir.exists():
        raise SystemExit(
            "Create dataset/benign and dataset/melanoma folders first."
        )

    benign_count = count_images(benign_dir)
    melanoma_count = count_images(melanoma_dir)

    print(f"Benign images: {benign_count}")
    print(f"Melanoma images: {melanoma_count}")

    if benign_count < 10 or melanoma_count < 10:
        raise SystemExit(
            "Not enough images. Add at least 10 images to EACH folder. "
            "For better results, use hundreds or thousands."
        )

    train_ds, val_ds, class_names = build_datasets(
        dataset_dir,
        args.image_size,
        args.batch_size,
        args.seed,
    )

    model = build_model(args.image_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "skin_lesion_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
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

    total = benign_count + melanoma_count

    class_weight = {
        0: total / (2 * benign_count),
        1: total / (2 * melanoma_count),
    }

    print("\nTraining custom CNN from scratch...\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    best_model = keras.models.load_model(model_path)

    results = best_model.evaluate(
        val_ds,
        verbose=0,
        return_dict=True,
    )

    metadata = {
        "project": "DermaSense AI",
        "architecture": "Custom CNN trained from scratch",
        "pretrained": False,
        "class_names": class_names,
        "image_size": args.image_size,
        "threshold": 0.50,
        "benign_images": benign_count,
        "melanoma_images": melanoma_count,
        "validation_split": 0.20,
        "seed": args.seed,
        "note": "Educational AI prototype. Not a medical diagnostic device.",
    }

    with open(model_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    with open(model_dir / "metrics.json", "w") as file:
        json.dump(
            {key: float(value) for key, value in results.items()},
            file,
            indent=2,
        )

    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")

    print("\nValidation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()