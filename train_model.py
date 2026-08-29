from pathlib import Path
import argparse
import json
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path):
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
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
        shuffle=False,
    )

    expected = ["benign", "melanoma"]
    if train_ds.class_names != expected:
        raise ValueError(
            f"Expected class folders {expected}, but TensorFlow found "
            f"{train_ds.class_names}. Rename the dataset folders exactly."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, expected


def build_model(image_size):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    # MobileNetV2 expects values in approximately [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_rescale")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="melanoma_probability")(x)

    model = keras.Model(inputs, outputs, name="DermaSense_MobileNetV2")
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def combine_histories(*histories):
    combined = {}
    for hist in histories:
        if hist is None:
            continue
        for key, values in hist.history.items():
            combined.setdefault(key, []).extend([float(v) for v in values])
    return combined


def save_training_curves(history, out_path):
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not list(epochs):
        return

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("loss", []), label="Training loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation loss")
    if "accuracy" in history:
        plt.plot(epochs, history["accuracy"], label="Training accuracy")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(model, val_ds, out_path, threshold=0.5):
    y_true = []
    y_score = []

    for images, labels in val_ds:
        scores = model.predict(images, verbose=0).reshape(-1)
        y_score.extend(scores.tolist())
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()

    fig = plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.title("Validation confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks([0, 1], ["Benign", "Melanoma"])
    plt.yticks([0, 1], ["Benign", "Melanoma"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train the DermaSense binary skin-lesion classifier."
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    benign_dir = dataset_dir / "benign"
    melanoma_dir = dataset_dir / "melanoma"

    if not benign_dir.exists() or not melanoma_dir.exists():
        raise SystemExit(
            "Dataset folders missing. Create dataset/benign and dataset/melanoma."
        )

    benign_count = count_images(benign_dir)
    melanoma_count = count_images(melanoma_dir)

    if benign_count < 10 or melanoma_count < 10:
        raise SystemExit(
            "Not enough training images. Add at least 10 images to EACH class "
            "to test the pipeline; for a meaningful project, use a much larger, "
            "properly labeled dataset."
        )

    print(f"Benign images: {benign_count}")
    print(f"Melanoma images: {melanoma_count}")

    train_ds, val_ds, class_names = build_datasets(
        dataset_dir,
        args.image_size,
        args.batch_size,
        args.seed,
    )

    total = benign_count + melanoma_count
    class_weight = {
        0: total / (2.0 * benign_count),
        1: total / (2.0 * melanoma_count),
    }
    print("Class weights:", class_weight)

    model, base_model = build_model(args.image_size)
    compile_model(model, 1e-3)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
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
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nStage 1: training classifier head...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    history2 = None
    if args.fine_tune_epochs > 0:
        print("\nStage 2: fine-tuning top MobileNetV2 layers...")
        base_model.trainable = True

        fine_tune_at = max(0, len(base_model.layers) - 30)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # BatchNormalization layers are kept frozen for more stable fine-tuning.
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        compile_model(model, 1e-5)

        completed = len(history1.history.get("loss", []))
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=completed,
            epochs=completed + args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    # Reload the best checkpoint before final evaluation.
    best_model = keras.models.load_model(model_path)
    eval_values = best_model.evaluate(val_ds, verbose=0, return_dict=True)
    eval_values = {k: float(v) for k, v in eval_values.items()}

    threshold = 0.50
    metadata = {
        "project": "DermaSense AI",
        "architecture": "MobileNetV2 transfer learning",
        "class_names": class_names,
        "image_size": args.image_size,
        "threshold": threshold,
        "benign_images": benign_count,
        "melanoma_images": melanoma_count,
        "validation_split": 0.20,
        "seed": args.seed,
        "note": "Educational prototype; not a medical diagnostic device.",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(eval_values, indent=2),
        encoding="utf-8",
    )

    combined = combine_histories(history1, history2)
    save_training_curves(combined, model_dir / "training_curves.png")
    save_confusion_matrix(
        best_model,
        val_ds,
        model_dir / "confusion_matrix.png",
        threshold=threshold,
    )

    print("\nTraining complete.")
    print(f"Saved model: {model_path}")
    print("Validation metrics:")
    for key, value in eval_values.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
