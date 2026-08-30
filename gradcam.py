import numpy as np
import tensorflow as tf
from PIL import Image


# =========================================================
# FIND MOBILENET FEATURE EXTRACTOR
# =========================================================

def find_feature_model(model):
    """
    Find the MobileNetV2 model nested inside
    the DermaSense classifier.
    """

    for layer in model.layers:

        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenet" in layer.name.lower()
        ):

            return layer

    # Fallback: find a nested CNN model
    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            try:

                shape = layer.output_shape

                if (
                    isinstance(shape, tuple)
                    and len(shape) == 4
                ):

                    return layer

            except Exception:
                pass

    raise ValueError(
        "MobileNetV2 feature extractor not found."
    )


# =========================================================
# PREPARE IMAGE
# =========================================================

def prepare_gradcam_image(
    image,
    image_size=224,
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
# APPLY LAYER SAFELY
# =========================================================

def apply_layer(
    layer,
    tensor,
):

    try:

        return layer(
            tensor,
            training=False,
        )

    except TypeError:

        return layer(
            tensor
        )


# =========================================================
# GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    image,
    class_index,
    image_size=224,
):

    image_array = prepare_gradcam_image(
        image,
        image_size,
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    feature_model = find_feature_model(
        model
    )

    feature_maps = None


    # -----------------------------------------------------
    # FORWARD PASS
    # -----------------------------------------------------

    with tf.GradientTape() as tape:

        x = image_tensor

        for layer in model.layers:

            # InputLayer does not need to be called
            if isinstance(
                layer,
                tf.keras.layers.InputLayer,
            ):

                continue


            x = apply_layer(
                layer,
                x,
            )


            # Save MobileNet feature maps
            if layer is feature_model:

                feature_maps = x

                tape.watch(
                    feature_maps
                )


        predictions = x


        if feature_maps is None:

            raise ValueError(
                "Feature maps could not be obtained."
            )


        class_score = predictions[
            :,
            class_index,
        ]


    # -----------------------------------------------------
    # GRADIENTS
    # -----------------------------------------------------

    gradients = tape.gradient(
        class_score,
        feature_maps,
    )


    if gradients is None:

        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )


    # -----------------------------------------------------
    # WEIGHT FEATURE MAPS
    # -----------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(
            0,
            1,
            2,
        ),
    )


    feature_maps = feature_maps[
        0
    ]


    weighted_maps = (
        feature_maps
        *
        pooled_gradients
    )


    heatmap = tf.reduce_sum(
        weighted_maps,
        axis=-1,
    )


    # -----------------------------------------------------
    # KEEP POSITIVE CONTRIBUTIONS
    # -----------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0,
    )


    max_value = tf.reduce_max(
        heatmap
    )


    if float(max_value) > 0:

        heatmap = (
            heatmap
            / max_value
        )


    return heatmap.numpy()


# =========================================================
# CREATE COLOR OVERLAY
# =========================================================

def create_overlay(
    image,
    heatmap,
):

    original = image.convert(
        "RGB"
    )


    # Resize heatmap to original uploaded image size
    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        ),
        mode="L",
    )


    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR,
    )


    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0


    # -----------------------------------------------------
    # CREATE RED/YELLOW ATTENTION COLOR
    # -----------------------------------------------------

    red = np.ones_like(
        heatmap_array
    ) * 255


    green = (
        np.sqrt(
            heatmap_array
        )
        * 210
    )


    blue = np.zeros_like(
        heatmap_array
    )


    heatmap_rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )


    original_array = np.asarray(
        original,
        dtype=np.float32,
    )


    # Strong attention gets stronger overlay.
    # Areas with no attention stay close to original.
    alpha = (
        heatmap_array[
            ...,
            np.newaxis
        ]
        * 0.55
    )


    overlay_array = (

        original_array
        * (
            1.0
            - alpha
        )

        +

        heatmap_rgb
        * alpha
    )


    overlay_array = np.clip(
        overlay_array,
        0,
        255,
    ).astype(
        np.uint8
    )


    return Image.fromarray(
        overlay_array
    )


# =========================================================
# MAIN FUNCTION USED BY STREAMLIT
# =========================================================

def create_gradcam_result(
    model,
    image,
    class_index,
    image_size=224,
):

    heatmap = generate_gradcam(
        model=model,
        image=image,
        class_index=class_index,
        image_size=image_size,
    )


    overlay = create_overlay(
        image=image,
        heatmap=heatmap,
    )


    return {
        "heatmap":
            heatmap,

        "overlay":
            overlay,
    }