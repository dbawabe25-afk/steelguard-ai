
import os
import traceback

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

IMG_SIZE = 160

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

st.set_page_config(
    page_title="SteelGuard AI",
    page_icon="🔍",
    layout="wide",
)

# Model file must be in the same folder as app.py.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "steel_defect_model.keras")


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Place steel_defect_model.keras beside app.py."
        )

    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
    )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input,
        },
        compile=False,
    )

    if loaded_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"The model outputs {loaded_model.output_shape[-1]} "
            f"classes, but CLASS_NAMES contains {len(CLASS_NAMES)}."
        )

    return loaded_model


# --------------------------------------------------
# 3. PREDICTION
# --------------------------------------------------

def predict_image(model, image):
    """Resize the image and run model inference."""

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or IMG_SIZE
    width = input_shape[2] or IMG_SIZE

    resized = image.convert("RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # Keep this preprocessing consistent with model training.
    # If preprocessing is already included in the saved model,
    # do not apply preprocess_input a second time.

    predictions = model(
        image_array,
        training=False,
    )

    probabilities = np.asarray(predictions)[0]

    if probabilities.ndim != 1:
        raise ValueError(
            f"Unexpected model output shape: {probabilities.shape}"
        )

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, "
            f"received {len(probabilities)}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid prediction values.")

    return resized, image_array, probabilities


# --------------------------------------------------
# 4. GRAD-CAM
# --------------------------------------------------

def make_gradcam(model, image_array, class_index):
    """
    Generate Grad-CAM for a model containing a nested
    MobileNetV2 backbone and a classification head.
    """

    # Find the nested MobileNetV2 model.
    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ),
        None,
    )

    if backbone is None:
        raise ValueError(
            "Could not find a nested MobileNetV2 backbone. "
            "Grad-CAM requires the model's actual convolutional layer."
        )

    # Find a spatial feature layer in the backbone.
    try:
        conv_layer = backbone.get_layer("out_relu")
    except ValueError:
        conv_layer = next(
            (
                layer
                for layer in reversed(backbone.layers)
                if len(getattr(layer, "output_shape", ())) == 4
            ),
            None,
        )

    if conv_layer is None:
        raise ValueError(
            "Could not find a convolutional feature layer."
        )

    # A feature extractor returning the convolutional activations
    # and the backbone output.
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output],
    )

    # Run any outer layers that appear before the backbone.
    x = tf.convert_to_tensor(image_array, dtype=tf.float32)
    backbone_index = model.layers.index(backbone)

    for layer in model.layers[:backbone_index]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = feature_model(
            x,
            training=False,
        )

        # Apply the classification head after the backbone.
        predictions = backbone_features

        for layer in model.layers[backbone_index + 1:]:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            predictions = layer(predictions, training=False)

        score = predictions[0, class_index]

    gradients = tape.gradient(score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients are unavailable for this model architecture."
        )

    # Average gradients across spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )

    heatmap = tf.reduce_sum(
        weights * conv_outputs,
        axis=-1,
    )[0]

    heatmap = tf.maximum(heatmap, 0)

    heatmap_max = tf.reduce_max(heatmap)
    heatmap = heatmap / (heatmap_max + 1e-8)

    # tf.image.resize requires a (height, width) tuple.
    height = int(image_array.shape[1])
    width = int(image_array.shape[2])

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (height, width),
    )

    return heatmap.numpy().squeeze()


# --------------------------------------------------
# 5. STREAMLIT INTERFACE
# --------------------------------------------------

st.title("🔍 SteelGuard AI")

st.subheader("Steel Surface Defect Classification")

st.write(
    "Upload a steel surface image to classify its defect "
    "and visualize the regions influencing the prediction."
)

with st.expander("Supported defect classes"):
    for class_name in CLASS_NAMES:
        st.write(f"• {class_name.replace('_', ' ').title()}")

# Load the model.
try:
    model = load_model()
    st.success("Steel defect model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a steel image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")

        # Predict independently of Grad-CAM.
        resized, image_array, probabilities = predict_image(
            model,
            image,
        )

        pred_idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probabilities[pred_idx])

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original steel surface image",
                use_container_width=True,
            )

            st.success(
                f"Predicted defect: "
                f"{predicted_class.replace('_', ' ').title()}"
            )

            st.metric(
                "Model output score",
                f"{confidence:.2%}",
            )

        with col2:
            st.subheader("Grad-CAM explanation")

            try:
                heatmap = make_gradcam(
                    model,
                    image_array,
                    pred_idx,
                )

                fig, ax = plt.subplots(figsize=(6, 5))

                ax.imshow(resized)
                ax.imshow(
                    heatmap,
                    cmap="jet",
                    alpha=0.40,
                    vmin=0,
                    vmax=1,
                )

                ax.set_title("Regions influencing the prediction")
                ax.axis("off")

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            except Exception as gradcam_error:
                st.warning(
                    "The prediction succeeded, but Grad-CAM "
                    "could not be generated for this model."
                )
                st.code(str(gradcam_error))

        st.subheader("Class scores")

        chart_data = {
            "Defect class": [
                name.replace("_", " ").title()
                for name in CLASS_NAMES
            ],
            "Score": probabilities.astype(float),
        }

        st.bar_chart(
            chart_data,
            x="Defect class",
            y="Score",
        )

        st.caption(
            "Model scores are not necessarily calibrated probabilities. "
            "Grad-CAM is an explanatory aid, not proof of correctness. "
            "Verify results before industrial quality-control decisions."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.code(traceback.format_exc())

