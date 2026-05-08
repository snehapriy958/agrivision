"""
frontend/app.py — AgriVision Streamlit Frontend
Run : streamlit run frontend/app.py
"""

import io
import os
import sys
import logging
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import streamlit as st
from PIL import Image

from inference.predictor import load_model, predict
from utils.visualize import overlay_heatmap_on_image

logging.basicConfig(level=logging.ERROR)

MODEL_VERSION = "v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_BYTES  = 5 * 1024 * 1024     # 5 MB
ENTROPY_LOW     = 0.5
ENTROPY_HIGH    = 1.5
CONF_CLAMP_MIN  = 0.001
CONF_CLAMP_MAX  = 0.999
GAP_AMBIGUOUS   = 0.2
CONF_HIGH       = 0.8
CONF_MODERATE   = 0.5


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model_once() -> None:
    load_model()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict(image_bytes: bytes, _model_version: str) -> dict:
    """Standard inference — result is cached per unique image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return predict(tmp_path, return_heatmap=False)
    finally:
        _safe_remove(tmp_path)

def gradcam_predict(image_bytes: bytes) -> dict:
    """Grad-CAM inference — NOT cached (requires live gradients)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return predict(tmp_path, return_heatmap=True)
    finally:
        _safe_remove(tmp_path)
    

def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _clamp(value: float) -> float:
    return max(CONF_CLAMP_MIN, min(value, CONF_CLAMP_MAX))


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def confidence_status(top1: float, top2: float) -> str:
    gap = top1 - top2
    if gap < GAP_AMBIGUOUS:
        return "⚠️ Ambiguous"
    if top1 >= CONF_HIGH:
        return "✅ High"
    if top1 >= CONF_MODERATE:
        return "⚠️ Moderate"
    return "❌ Low"


def entropy_label(entropy: float) -> str:
    if entropy <= ENTROPY_LOW:
        return "🟢 Low — model is confident"
    if entropy <= ENTROPY_HIGH:
        return "🟡 Medium — some uncertainty"
    return "🔴 High — prediction is unreliable"


def get_advice(label: str) -> str | None:
    label_lower = label.lower()
    if "blight" in label_lower:
        return "Use fungicide and avoid overwatering."
    if "rust" in label_lower:
        return "Ensure proper air circulation and apply fungicide."
    if "healthy" in label_lower:
        return "Plant is healthy. Maintain current care."
    return None


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def _section_upload() -> tuple[bytes | None, Image.Image | None]:
    st.header("📤 Upload")
    uploaded = st.file_uploader(
        "Choose a crop leaf image",
        type=["jpg", "jpeg", "png"],
        help="Clear, close-up images produce the best results.",
    )

    if uploaded is None:
        return None, None

    if uploaded.size > MAX_FILE_BYTES:
        st.error("File exceeds 5 MB. Please upload a smaller image.")
        return None, None

    image_bytes = uploaded.getvalue()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("Invalid or corrupted image.")
        return None, None
    return image_bytes, pil_image
    


def _section_preview(
    pil_image   : Image.Image,
    show_gradcam: bool,
    heatmap     : object | None,
) -> None:
    st.header("🖼️ Preview")

    if show_gradcam and heatmap is not None:
        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.image(pil_image, caption="Original", use_container_width=True)
        with col_cam:
            overlay = overlay_heatmap_on_image(pil_image, np.clip(heatmap, 0, 1))
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
    elif show_gradcam and heatmap is None:
        st.image(pil_image, caption="Original", use_container_width=True)
        st.warning("Grad-CAM heatmap is unavailable for this image.")
    else:
        st.image(pil_image, caption="Original", use_container_width=True)


def _section_analysis(
    predictions : list[dict],
    entropy     : float,
) -> None:
    st.header("🔬 Analysis")

    top1_conf = predictions[0]["confidence"]
    top2_conf = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
    gap       = top1_conf - top2_conf
    top_label = format_label(predictions[0]["label"])

    # Diagnosis
    if gap < GAP_AMBIGUOUS and len(predictions) > 1:
        diagnosis = (
            f"Possible **{top_label}** or "
            f"**{format_label(predictions[1]['label'])}**"
        )
    else:
        diagnosis = f"**{top_label}**"

    st.success(f"Diagnosis: {diagnosis} ({_clamp(top1_conf) * 100:.2f}%)")

    # Secondary disease risk
    if len(predictions) > 1:
        second = predictions[1]
        if any(d in second["label"].lower() for d in ("blight", "rust", "spot")):
            st.warning(
                f"⚠️ Secondary risk: {format_label(second['label'])} "
                f"({_clamp(second['confidence']) * 100:.2f}%)"
            )

    # Confidence banner
    if top1_conf < CONF_MODERATE:
        st.error("❌ Very low confidence — result is unreliable.")
    elif gap < GAP_AMBIGUOUS:
        st.warning("⚠️ Ambiguous — two classes have similar probabilities.")
    elif top1_conf < CONF_HIGH:
        st.info("ℹ️ Moderate confidence — consider manual verification.")
    else:
        st.success("✅ High confidence prediction.")

    # Entropy
    col_e1, col_e2 = st.columns([1, 3])
    col_e1.metric("Entropy", f"{entropy:.4f}")
    col_e2.markdown(f"**Uncertainty:** {entropy_label(entropy)}")

    if entropy > ENTROPY_HIGH:
        st.error("❌ Model is highly uncertain — result may be unreliable.")
    elif entropy > ENTROPY_LOW:
        st.warning("⚠️ Moderate uncertainty — verify result.")
    else:
        st.success("✅ Low uncertainty — model is confident in this prediction.")

    # Recommendation
    advice = get_advice(predictions[0]["label"])
    if advice:
        st.info(f"💡 **Recommendation:** {advice}")


def _section_predictions(predictions: list[dict]) -> None:
    st.header("📊 Predictions")

    top1_conf = predictions[0]["confidence"]
    top2_conf = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
    

    for rank, pred in enumerate(predictions, start=1):
        label      = pred["label"]
        confidence = _clamp(pred["confidence"])

        with st.container(border=True):
            c_rank, c_label, c_pct, c_status = st.columns([0.5, 3, 1.5, 2.5])
            c_rank.markdown(f"**#{rank}**")
            c_label.markdown(f"**{format_label(label)}**")
            c_pct.markdown(f"`{confidence * 100:.2f}%`")
            c_status.markdown(
                confidence_status(top1_conf, top2_conf) if rank == 1 else ""
            )
            st.progress(confidence)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AgriVision",
        page_icon="🌿",
        layout="centered",
    )

    load_model_once()

    st.title("🌿 AgriVision — Crop Disease Detection")
    st.caption("Upload a crop leaf image to identify diseases and receive care advice.")
    st.caption("💡 Tip: Use a single leaf with minimal background for best results.")
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────
    image_bytes, pil_image = _section_upload()

    if image_bytes is None or pil_image is None:
        return

    st.divider()

    # ── Grad-CAM toggle ───────────────────────────────────────────────
    show_gradcam = st.toggle("🔬 Enable Explainability (Grad-CAM)", value=False)

    # ── Inference ─────────────────────────────────────────────────────
    with st.spinner("Analysing image…"):
        try:
            if show_gradcam:
                result      = gradcam_predict(image_bytes)
                predictions = result.get("predictions", [])
                entropy     = result.get("entropy", 0.0)
                heatmap     = result.get("heatmap", None)
            else:
                result      = cached_predict(image_bytes, MODEL_VERSION)
                predictions = result.get("predictions", [])
                entropy     = result.get("entropy", 0.0)
                heatmap     = None

        except Exception as exc:
            logging.exception(exc)
            st.error(f"Inference failed: {exc}")
            return

    if not predictions:
        st.error("No predictions returned. Please try a different image.")
        return

    predictions = sorted(predictions, key=lambda p: p["confidence"], reverse=True)

    st.divider()

    # ── Preview ───────────────────────────────────────────────────────
    _section_preview(pil_image, show_gradcam, heatmap)
    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────
    _section_analysis(predictions, entropy)
    st.divider()

    # ── Predictions ───────────────────────────────────────────────────
    _section_predictions(predictions)
    st.divider()

    st.caption(
        "Model: PlantDiseaseResNet (ResNet50) · "
        "Input: 224×224 · Classes: 9 · Output: Top-3 predictions"
    )


if __name__ == "__main__":
    main()