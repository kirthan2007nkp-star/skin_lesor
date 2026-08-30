from pathlib import Path
import random

from PIL import Image
from tensorflow.keras.datasets import fashion_mnist


OUTPUT_DIR = Path("dataset/non_skin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_TO_SAVE = 1000
SEED = 42


def main():
    random.seed(SEED)

    print("Downloading small Fashion-MNIST dataset...")

    (x_train, _), _ = fashion_mnist.load_data()

    indices = list(range(len(x_train)))
    random.shuffle(indices)

    selected = indices[:IMAGES_TO_SAVE]

    for number, index in enumerate(selected, start=1):

        image = Image.fromarray(x_train[index])

        # Convert grayscale image to RGB
        image = image.convert("RGB")

        # Resize so it is closer to normal uploaded images
        image = image.resize((224, 224))

        image.save(
            OUTPUT_DIR / f"non_skin_{number:04d}.png"
        )

        if number % 100 == 0:
            print(f"Saved {number}/{IMAGES_TO_SAVE}")

    print("\nDone.")
    print(
        f"Saved {IMAGES_TO_SAVE} non-skin images "
        f"to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()