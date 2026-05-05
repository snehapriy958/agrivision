import os
import io
import sys
import tempfile
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from inference.predictor import predict, load_model
from utils.visualize import overlay_heatmap_on_image
from PIL import Image
import numpy as np
from typing import Optional, List, Dict

logging.basicConfig(level=logging.ERROR)

MODEL_VERSION = "v1"


# ---------------------------------------------------------------------------
# Model preload (IMPORTANT FIX)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_once():
    load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_confidence_status(confidence: float) -> str:
    if confidence > 0.8:
        return "✅ High confidence"
    elif confidence > 0.5:
        return "⚠️ Moderate confidence"
    return "❌ Low confidence"


def get_advice(label: str) -> Optional[str]:
    label_lower = label.lower()
    if "blight" in label_lower:
        return "Use fungicide and avoid overwatering."
    if "rust" in label_lower:
        return "Ensure proper air circulation and apply fungicide."
    if "healthy" in label_lower:
        return "Plant is healthy. Maintain current care."
    return None


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Cached prediction (no Grad-CAM)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict(image_bytes: bytes, model_version: str) -> List[Dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        return predict(tmp_path, return_heatmap=False)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Grad-CAM prediction (no caching)
# ---------------------------------------------------------------------------

def gradcam_predict(image_bytes: bytes) -> Dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        return predict(tmp_path, return_heatmap=True)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AgriVision", page_icon="🌿", layout="centered")

    # ✅ Load model once (FIXED)
    load_model_once()

    st.title("🌿 AgriVision — Crop Disease Detection")
    st.caption("⚡ First prediction may take ~30–60 seconds (model loads once)")
    st.caption("Upload a crop image to detect diseases and get recommendations.")

    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear leaf image (max ~5MB).",
    )

    if uploaded_file and uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large (>5MB). Please upload a smaller image.")
        return

    if not uploaded_file:
        return

    st.info("📸 Tip: Use a clear, close-up image of a single leaf.")

    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    show_gradcam = st.toggle("🔬 Enable Explainability (Grad-CAM)", value=False)

    # ------------------------------------------------------------------
    # Image preview
    # ------------------------------------------------------------------

    if not show_gradcam:
        st.image(pil_image, caption="Uploaded Image", width=320)
        st.write("")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    with st.spinner("Analyzing image..."):
        try:
            if show_gradcam:
                st.caption("⚠️ Grad-CAM may take slightly longer")
                result = gradcam_predict(image_bytes)
                predictions = result.get("predictions", [])
                heatmap = result.get("heatmap", None)
            else:
                predictions = cached_predict(image_bytes, MODEL_VERSION)
                heatmap = None

            if not predictions:
                st.error("No predictions returned. Try another image.")
                return

        except Exception as e:
            import traceback
            st.error(f"Inference failed: {str(e)}")
            st.text(traceback.format_exc())
            return

    # ------------------------------------------------------------------
    # Heatmap display
    # ------------------------------------------------------------------

    if show_gradcam:
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="Original", use_container_width=True)

        with col2:
            if heatmap is not None:
                heatmap = np.clip(heatmap, 0, 1)
                overlay = overlay_heatmap_on_image(pil_image, heatmap)
                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                st.warning("Heatmap unavailable.")
                st.image(pil_image, caption="Original", use_container_width=True)

        st.write("")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    st.subheader("Top Predictions")

    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    top = predictions[0]
    conf = top["confidence"]
    conf_display = max(min(conf, 0.999), 0.001)
    st.success(
        f"Most likely: **{format_label(top['label'])}** "
        f"({conf_display * 100:.2f}%)"
    )
    
    if conf < 0.3:
        st.error("❌ Model is not reliable. Likely due to poor image or domain mismatch.")
        return
    elif conf < 0.6:
        st.warning("⚠️ Low confidence — try a clearer, close-up leaf image.")
    elif conf < 0.8:
        st.info("ℹ️ Moderate confidence — verify the result.")
    else:
        st.success("✅ High confidence in prediction.")

    for rank, pred in enumerate(predictions, start=1):
        label = pred["label"]
        confidence = pred["confidence"]
        confidence_display = max(min(confidence, 0.999), 0.001)
        status = get_confidence_status(confidence)

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.5, 3, 1.5, 2.5])
            c1.markdown(f"**#{rank}**")
            c2.markdown(f"**{format_label(label)}**")
            c3.markdown(f"`{confidence_display*100:.2f}%`")
            c4.markdown(status)
            st.progress(confidence_display)

        if rank == 1:
            advice = get_advice(label)
            if advice:
                st.info(f"💡 Recommendation: {advice}")

    st.divider()
    st.caption("Model: Custom CNN • Input: 224×224 • Classes: 9 • Top-3 predictions")


if __name__ == "__main__":
    main()