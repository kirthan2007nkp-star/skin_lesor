from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# =========================================================
# SETTINGS
# =========================================================

TEST_DIR = Path("dataset_v2") / "test"

MODEL_PATH = (
    Path("models")
    / "skin_lesion_model.keras"
)

OUTPUT_DIR = Path("evaluation")

IMAGE_SIZE = 224
BATCH_SIZE = 32

INTERNAL_CLASSES = [
    "benign",
    "melanoma",
    "non_skin",
    "other_skin",
]

DISPLAY_CLASSES = [
    "Benign-like",
    "Melanoma-suspicious",
    "Other",
]


# =========================================================
# LOAD TEST DATA
# =========================================================

def load_test_dataset():

    dataset = (
        keras.utils.image_dataset_from_directory(
            TEST_DIR,
            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=False,
        )
    )

    if (
        dataset.class_names
        != INTERNAL_CLASSES
    ):

        raise ValueError(
            "\nUnexpected class order.\n"
            f"Expected: {INTERNAL_CLASSES}\n"
            f"Found: {dataset.class_names}"
        )

    return dataset


# =========================================================
# MAP TRUE LABELS TO 3 APP CATEGORIES
# =========================================================

def map_true_labels(labels):

    labels = np.asarray(
        labels,
        dtype=int,
    )

    mapped = np.zeros_like(
        labels
    )

    # benign -> Benign-like
    mapped[
        labels == 0
    ] = 0

    # melanoma -> Melanoma-suspicious
    mapped[
        labels == 1
    ] = 1

    # non_skin + other_skin -> Other
    mapped[
        (labels == 2)
        |
        (labels == 3)
    ] = 2

    return mapped


# =========================================================
# COMBINE MODEL OUTPUTS INTO 3 APP CATEGORIES
# =========================================================

def combine_predictions(
    probabilities,
):

    benign = (
        probabilities[:, 0]
    )

    melanoma = (
        probabilities[:, 1]
    )

    other = (
        probabilities[:, 2]
        +
        probabilities[:, 3]
    )

    combined = np.stack(
        [
            benign,
            melanoma,
            other,
        ],
        axis=1,
    )

    predictions = np.argmax(
        combined,
        axis=1,
    )

    return predictions


# =========================================================
# CONFUSION MATRIX
# =========================================================

def make_confusion_matrix(
    true_labels,
    predicted_labels,
):

    matrix = tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=3,
    ).numpy()

    return matrix


# =========================================================
# PRECISION / RECALL / F1
# =========================================================

def calculate_metrics(
    matrix,
):

    rows = []

    for index, class_name in enumerate(
        DISPLAY_CLASSES
    ):

        true_positive = float(
            matrix[index, index]
        )

        false_positive = float(
            matrix[:, index].sum()
            - true_positive
        )

        false_negative = float(
            matrix[index, :].sum()
            - true_positive
        )

        support = int(
            matrix[index, :].sum()
        )

        precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
            if (
                true_positive
                + false_positive
            ) > 0
            else 0.0
        )

        recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        rows.append(
            {
                "Class":
                    class_name,

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1-score":
                    f1,

                "Support":
                    support,
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    matrix,
):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        np.arange(
            len(
                DISPLAY_CLASSES
            )
        )
    )

    ax.set_yticks(
        np.arange(
            len(
                DISPLAY_CLASSES
            )
        )
    )

    ax.set_xticklabels(
        DISPLAY_CLASSES,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        DISPLAY_CLASSES
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "DermaSense AI - Confusion Matrix"
    )

    threshold = (
        matrix.max()
        / 2
        if matrix.max() > 0
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column
            ]

            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
            )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n===================================="
    )

    print(
        "DermaSense AI Model Evaluation"
    )

    print(
        "====================================\n"
    )


    if not MODEL_PATH.exists():

        raise SystemExit(
            f"Model not found:\n{MODEL_PATH}"
        )


    if not TEST_DIR.exists():

        raise SystemExit(
            f"Test dataset not found:\n{TEST_DIR}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print(
        "Loading trained model..."
    )

    model = (
        keras.models.load_model(
            MODEL_PATH
        )
    )


    # -----------------------------------------------------
    # LOAD TEST DATA
    # -----------------------------------------------------

    test_ds = (
        load_test_dataset()
    )


    true_internal = []

    model_probabilities = []


    print(
        "\nEvaluating unseen test images..."
    )


    for images, labels in test_ds:

        probabilities = (
            model.predict(
                images,
                verbose=0,
            )
        )

        true_internal.extend(
            labels.numpy().tolist()
        )

        model_probabilities.extend(
            probabilities.tolist()
        )


    true_internal = np.asarray(
        true_internal,
        dtype=int,
    )


    model_probabilities = np.asarray(
        model_probabilities,
        dtype=float,
    )


    # -----------------------------------------------------
    # CONVERT TO FINAL 3 CATEGORIES
    # -----------------------------------------------------

    true_labels = (
        map_true_labels(
            true_internal
        )
    )


    predicted_labels = (
        combine_predictions(
            model_probabilities
        )
    )


    # -----------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------

    accuracy = float(
        np.mean(
            true_labels
            ==
            predicted_labels
        )
    )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    matrix = (
        make_confusion_matrix(
            true_labels,
            predicted_labels,
        )
    )


    # -----------------------------------------------------
    # PRECISION / RECALL / F1
    # -----------------------------------------------------

    report = (
        calculate_metrics(
            matrix
        )
    )


    macro_precision = float(
        report[
            "Precision"
        ].mean()
    )


    macro_recall = float(
        report[
            "Recall"
        ].mean()
    )


    macro_f1 = float(
        report[
            "F1-score"
        ].mean()
    )


    # -----------------------------------------------------
    # SAVE FILES
    # -----------------------------------------------------

    confusion_path = (
        save_confusion_matrix(
            matrix
        )
    )


    report_path = (
        OUTPUT_DIR
        / "classification_report.csv"
    )


    report.to_csv(
        report_path,
        index=False,
    )


    metrics = {
        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "classes":
            DISPLAY_CLASSES,

        "confusion_matrix":
            matrix.tolist(),
    }


    metrics_path = (
        OUTPUT_DIR
        / "evaluation_metrics.json"
    )


    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


    # -----------------------------------------------------
    # TERMINAL RESULTS
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "FINAL 3-CATEGORY TEST RESULTS"
    )

    print(
        "===================================="
    )


    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%"
    )


    print(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%"
    )


    print(
        f"Macro F1-score: "
        f"{macro_f1 * 100:.2f}%"
    )


    print(
        "\nPer-Class Results:\n"
    )


    for _, row in (
        report.iterrows()
    ):

        print(
            f"{row['Class']}"
        )

        print(
            f"  Precision: "
            f"{row['Precision'] * 100:.2f}%"
        )

        print(
            f"  Recall: "
            f"{row['Recall'] * 100:.2f}%"
        )

        print(
            f"  F1-score: "
            f"{row['F1-score'] * 100:.2f}%"
        )

        print(
            f"  Test images: "
            f"{int(row['Support'])}"
        )

        print()


    print(
        "Confusion Matrix:\n"
    )

    print(
        matrix
    )


    print(
        "\nSaved:"
    )

    print(
        confusion_path
    )

    print(
        report_path
    )

    print(
        metrics_path
    )


if __name__ == "__main__":
    main()