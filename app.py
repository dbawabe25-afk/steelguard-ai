IMG_SIZE = 160
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


st.set_page_config(
    page_title="SteelGuard AI",
    page_icon="🔍",
    layout="wide"
)


import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "steel_defect_model.keras"
)



@st.cache_resource
def load_model():
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )


def make_gradcam(model, image_array):
    # Find the nested MobileNetV2 backbone
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    )

    conv_layer = backbone.get_layer("out_relu")

    # Build feature extractor inside the backbone's own graph
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output]
    )

    # Apply outer preprocessing layers before the backbone
    x = tf.convert_to_tensor(image_array, dtype=tf.float32)
    for layer in model.layers:
        if layer is backbone:
            break
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = feature_model(
            x, training=False
        )

        # Follow the model's actual classification head
        predictions = backbone_features
        backbone_index = model.layers.index(backbone)
        for layer in model.layers[backbone_index + 1:]:
            predictions = layer(predictions, training=False)

        class_index = tf.argmax(predictions[0], output_type=tf.int32)
        score = predictions[0, class_index]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise ValueError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    return predictions.numpy()[0], heatmap

st.title("🔍 SteelGuard AI")
st.subheader("Steel Surface Defect Classification")
st.write(
    "Upload a steel surface image to classify its defect "
    "and visualize the regions influencing the prediction."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a steel image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        resized = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.asarray(resized, dtype=np.float32)[None, ...]

        probabilities, heatmap = make_gradcam(model, image_array)
        pred_idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probabilities[pred_idx])

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original image", use_container_width=True)
            st.success(f"Prediction: {predicted_class}")
            st.metric("Prediction confidence", f"{confidence:.2%}")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(resized)
            ax.imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.set_title("Grad-CAM explanation")
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Class probabilities")
        st.bar_chart({
            "Defect class": CLASS_NAMES,
            "Probability": probabilities
        }, x="Defect class", y="Probability")

        st.caption(
            "Grad-CAM is an explanatory aid, not proof of correctness. "
            "Review predictions before industrial quality-control decisions."
        )

    except Exception as exc:
        import traceback
        st.error(f"Prediction or Grad-CAM failed: {exc}")
        st.code(traceback.format_exc())
