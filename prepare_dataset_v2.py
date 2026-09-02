from pathlib import Path
import csv
import random
import shutil

from PIL import Image, ImageDraw


# =========================================================
# SETTINGS
# =========================================================

SEED = 42
random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "dataset_v2"

DOWNLOADS = Path.home() / "Downloads"

HAM_IMAGES = DOWNLOADS / "ISIC-images"

HAM_METADATA = (
    DOWNLOADS
    / "ham10000_metadata_2026-08-30.csv"
)

# Existing Fashion-MNIST non-skin images
OLD_OTHER_DIR = (
    PROJECT_DIR
    / "dataset"
    / "other"
)

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Keep benign roughly similar to melanoma
BENIGN_TARGET = 1305

# Generate enough non-skin examples to make
# rejection of screenshots/documents stronger
NON_SKIN_TARGET = 1600


# =========================================================
# HELPERS
# =========================================================

def reset_output():

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:

        for class_name in [
            "benign",
            "melanoma",
            "other_skin",
            "non_skin",
        ]:

            (
                OUTPUT_DIR
                / split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


def read_metadata():

    with open(
        HAM_METADATA,
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def image_path(isic_id):

    return (
        HAM_IMAGES
        / f"{isic_id}.jpg"
    )


def group_by_lesion(records):

    groups = {}

    for row in records:

        lesion_id = (
            row.get("lesion_id")
            or row["isic_id"]
        )

        groups.setdefault(
            lesion_id,
            []
        ).append(row)

    return groups


def select_groups_until_target(
    records,
    target=None,
):

    groups = group_by_lesion(
        records
    )

    lesion_ids = list(
        groups.keys()
    )

    random.shuffle(
        lesion_ids
    )

    if target is None:

        return {
            lesion_id:
                groups[lesion_id]

            for lesion_id
            in lesion_ids
        }

    selected = {}
    count = 0

    for lesion_id in lesion_ids:

        selected[
            lesion_id
        ] = groups[
            lesion_id
        ]

        count += len(
            groups[lesion_id]
        )

        if count >= target:
            break

    return selected


def split_groups(
    grouped_records
):

    lesion_ids = list(
        grouped_records.keys()
    )

    random.shuffle(
        lesion_ids
    )

    total = len(
        lesion_ids
    )

    train_end = int(
        total * 0.70
    )

    val_end = train_end + int(
        total * 0.15
    )

    return {
        "train":
            lesion_ids[
                :train_end
            ],

        "val":
            lesion_ids[
                train_end:
                val_end
            ],

        "test":
            lesion_ids[
                val_end:
            ],
    }


def copy_skin_class(
    grouped_records,
    split_ids,
    class_name,
):

    copied = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for split, lesion_ids in (
        split_ids.items()
    ):

        destination = (
            OUTPUT_DIR
            / split
            / class_name
        )

        for lesion_id in lesion_ids:

            for row in grouped_records[
                lesion_id
            ]:

                source = image_path(
                    row["isic_id"]
                )

                if not source.exists():
                    continue

                shutil.copy2(
                    source,
                    destination
                    / source.name,
                )

                copied[
                    split
                ] += 1

    return copied


# =========================================================
# SYNTHETIC SCREENSHOT / DOCUMENT IMAGES
# =========================================================

def random_color(
    low=20,
    high=235,
):

    return (
        random.randint(
            low,
            high
        ),
        random.randint(
            low,
            high
        ),
        random.randint(
            low,
            high
        ),
    )


def make_fake_screen(
    output_path,
):

    # Random landscape or portrait screenshot
    if random.random() < 0.75:

        width = random.choice(
            [320, 360, 400]
        )

        height = random.choice(
            [220, 240, 280]
        )

    else:

        width = random.choice(
            [220, 240, 280]
        )

        height = random.choice(
            [320, 360, 400]
        )


    dark_mode = (
        random.random()
        < 0.5
    )


    if dark_mode:

        background = (
            random.randint(5, 30),
            random.randint(10, 40),
            random.randint(20, 55),
        )

        text_color = (
            210,
            220,
            235,
        )

        panel_color = (
            25,
            40,
            60,
        )

    else:

        background = (
            random.randint(225, 255),
            random.randint(225, 255),
            random.randint(225, 255),
        )

        text_color = (
            35,
            45,
            60,
        )

        panel_color = (
            235,
            238,
            245,
        )


    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        background,
    )

    draw = ImageDraw.Draw(
        image
    )


    # Browser / app top bar
    top_height = random.randint(
        18,
        32,
    )

    draw.rectangle(
        (
            0,
            0,
            width,
            top_height,
        ),
        fill=random_color(
            30,
            180,
        ),
    )


    # Optional sidebar
    if random.random() < 0.6:

        sidebar_width = int(
            width
            * random.uniform(
                0.15,
                0.28,
            )
        )

        draw.rectangle(
            (
                0,
                top_height,
                sidebar_width,
                height,
            ),
            fill=panel_color,
        )


    # Cards / panels
    for _ in range(
        random.randint(
            3,
            8,
        )
    ):

        x1 = random.randint(
            int(width * 0.15),
            max(
                int(width * 0.15),
                width - 90,
            ),
        )

        y1 = random.randint(
            top_height + 8,
            max(
                top_height + 8,
                height - 60,
            ),
        )

        x2 = min(
            width - 5,
            x1 + random.randint(
                50,
                150,
            ),
        )

        y2 = min(
            height - 5,
            y1 + random.randint(
                25,
                75,
            ),
        )

        draw.rounded_rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            radius=5,
            fill=panel_color,
            outline=random_color(
                60,
                160,
            ),
            width=1,
        )


    # Text-like lines
    for _ in range(
        random.randint(
            12,
            30,
        )
    ):

        x = random.randint(
            10,
            max(
                10,
                width - 100,
            ),
        )

        y = random.randint(
            top_height + 5,
            height - 10,
        )

        line_width = random.randint(
            25,
            min(
                140,
                max(
                    30,
                    width - x - 5,
                ),
            ),
        )

        draw.line(
            (
                x,
                y,
                x + line_width,
                y,
            ),
            fill=text_color,
            width=random.choice(
                [1, 2, 3]
            ),
        )


    # Chart-like bars
    if random.random() < 0.5:

        base_y = height - 20

        start_x = random.randint(
            20,
            max(
                20,
                width // 2,
            ),
        )

        for index in range(
            random.randint(
                4,
                10,
            )
        ):

            bar_width = 8

            bar_height = random.randint(
                10,
                min(
                    90,
                    max(
                        15,
                        height // 3,
                    ),
                ),
            )

            x1 = (
                start_x
                + index * 12
            )

            if x1 + bar_width >= width:
                break

            draw.rectangle(
                (
                    x1,
                    base_y
                    - bar_height,
                    x1
                    + bar_width,
                    base_y,
                ),
                fill=random_color(
                    40,
                    220,
                ),
            )


    image.save(
        output_path
    )


# =========================================================
# NON-SKIN DATASET
# =========================================================

def prepare_non_skin():

    all_sources = []

    if OLD_OTHER_DIR.exists():

        all_sources = list(
            OLD_OTHER_DIR.glob(
                "non_skin_*.png"
            )
        )


    random.shuffle(
        all_sources
    )


    # Rough split counts
    train_target = int(
        NON_SKIN_TARGET
        * 0.70
    )

    val_target = int(
        NON_SKIN_TARGET
        * 0.15
    )

    test_target = (
        NON_SKIN_TARGET
        - train_target
        - val_target
    )


    split_targets = {
        "train":
            train_target,

        "val":
            val_target,

        "test":
            test_target,
    }


    source_index = 0


    for split, target in (
        split_targets.items()
    ):

        destination = (
            OUTPUT_DIR
            / split
            / "non_skin"
        )

        existing_count = min(
            target // 2,
            len(
                all_sources
            )
            - source_index,
        )


        # Copy Fashion-MNIST examples
        for _ in range(
            max(
                0,
                existing_count,
            )
        ):

            source = all_sources[
                source_index
            ]

            source_index += 1

            shutil.copy2(
                source,
                destination
                / source.name,
            )


        # Generate screenshot/document examples
        current_count = len(
            list(
                destination.glob("*")
            )
        )


        needed = (
            target
            - current_count
        )


        for number in range(
            needed
        ):

            filename = (
                f"synthetic_screen_"
                f"{split}_"
                f"{number:04d}.png"
            )

            make_fake_screen(
                destination
                / filename
            )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\nPreparing DermaSense dataset v2...\n"
    )


    if not HAM_IMAGES.exists():

        raise SystemExit(
            f"Images folder not found:\n"
            f"{HAM_IMAGES}"
        )


    if not HAM_METADATA.exists():

        raise SystemExit(
            f"Metadata file not found:\n"
            f"{HAM_METADATA}"
        )


    reset_output()


    rows = read_metadata()


    # -----------------------------------------------------
    # BENIGN
    # -----------------------------------------------------

    benign_rows = [
        row
        for row in rows

        if (
            row.get(
                "diagnosis_1"
            )
            == "Benign"
        )
    ]


    benign_groups = (
        select_groups_until_target(
            benign_rows,
            BENIGN_TARGET,
        )
    )


    benign_splits = (
        split_groups(
            benign_groups
        )
    )


    benign_counts = (
        copy_skin_class(
            benign_groups,
            benign_splits,
            "benign",
        )
    )


    # -----------------------------------------------------
    # MELANOMA
    # -----------------------------------------------------

    melanoma_rows = [
        row
        for row in rows

        if (
            row.get(
                "diagnosis_3"
            )
            == "Melanoma, NOS"
        )
    ]


    melanoma_groups = (
        select_groups_until_target(
            melanoma_rows
        )
    )


    melanoma_splits = (
        split_groups(
            melanoma_groups
        )
    )


    melanoma_counts = (
        copy_skin_class(
            melanoma_groups,
            melanoma_splits,
            "melanoma",
        )
    )


    # -----------------------------------------------------
    # OTHER SKIN
    # -----------------------------------------------------

    other_skin_rows = [
        row
        for row in rows

        if (
            (
                row.get(
                    "diagnosis_1"
                )
                == "Malignant"
                and
                row.get(
                    "diagnosis_3"
                )
                != "Melanoma, NOS"
            )
            or
            row.get(
                "diagnosis_1"
            )
            == "Indeterminate"
        )
    ]


    other_skin_groups = (
        select_groups_until_target(
            other_skin_rows
        )
    )


    other_skin_splits = (
        split_groups(
            other_skin_groups
        )
    )


    other_skin_counts = (
        copy_skin_class(
            other_skin_groups,
            other_skin_splits,
            "other_skin",
        )
    )


    # -----------------------------------------------------
    # NON-SKIN
    # -----------------------------------------------------

    prepare_non_skin()


    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print(
        "\nDataset v2 created successfully."
    )


    print(
        "\nFinal counts:"
    )


    for split in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"\n{split.upper()}"
        )

        for class_name in [
            "benign",
            "melanoma",
            "other_skin",
            "non_skin",
        ]:

            count = len(
                list(
                    (
                        OUTPUT_DIR
                        / split
                        / class_name
                    ).glob("*")
                )
            )

            print(
                f"  "
                f"{class_name}: "
                f"{count}"
            )


    print(
        "\nImportant:"
    )

    print(
        "Images from the same HAM10000 lesion ID "
        "stay inside only one split."
    )


if __name__ == "__main__":
    main()