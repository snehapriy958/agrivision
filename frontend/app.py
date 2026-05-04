"""
frontend/app.py — AgriVision Streamlit Frontend
Run : streamlit run frontend/app.py
"""

import os
import sys

# 🔥 FORCE absolute project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from inference.predictor import predict
import tempfile
import logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_VERSION = "v1"   # 🔁 change when model updates


# ---------------------------------------------------------------------------
# Logging (minimal)
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_confidence_status(confidence: float) -> str:
    if confidence > 0.8:
        return "✅ High confidence"
    elif confidence > 0.5:
        return "⚠️ Moderate confidence"
    return "❌ Low confidence"


def get_advice(label: str) -> str | None:
    label_lower = label.lower()
    if "blight" in label_lower:
        return "Use fungicide and avoid overwatering."
    if "rust" in label_lower:
        return "Ensure proper air circulation and apply fungicide."
    if "healthy" in label_lower:
        return "Plant is healthy. Maintain current care."
    return None


def format_label(label: str) -> str:
    return label.replace('_', ' ').title()


# ---------------------------------------------------------------------------
# Cached Prediction
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict(image_bytes: bytes, model_version: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()  # important for Windows

        return predict(tmp.name)

    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AgriVision", page_icon="🌿", layout="centered")

    # Header
    st.title("🌿 AgriVision — Crop Disease Detection")
    st.caption("⚡ First prediction may take ~30–60 seconds (model will download once)")
    st.caption("Upload a crop image to detect potential diseases and receive care advice.")

    # Upload
    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear leaf image (max ~5MB). Center the leaf and avoid blur."
    )

    # File size guard
    if uploaded_file and uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large (>5MB). Please upload a smaller image.")
        return

    if uploaded_file:
        st.info("📸 Tip: Use a clear, close-up image of a single leaf for best accuracy.")

        # Preview
        st.image(uploaded_file, caption="Uploaded Image", width=320)
        st.write("")

        # Convert once
        image_bytes = uploaded_file.getvalue()

        # Predict
        with st.spinner("Analysing image… (first run may take ~30–60 seconds)"):
            try:
                predictions = cached_predict(image_bytes, MODEL_VERSION)

                if not predictions:
                    st.error("No predictions returned. Try another image.")
                    return

            except Exception as exc:
                logging.error(str(exc))
                st.error("⚠️ Inference failed. Please try another image or refresh.")
                return

        # Results
        st.write("")
        st.subheader("Top Predictions")

        top = predictions[0]
        st.success(
            f"Most likely: **{format_label(top['label'])}** "
            f"({top['confidence']*100:.2f}%)"
        )

        # 🔥 Unknown detection logic

        if top["confidence"] < 0.5:
            st.error("❌ Unsupported or unclear image. Please upload a valid crop leaf.")
        if top["confidence"] < 0.7:
            st.warning(
                "⚠️ Low confidence prediction.\n\n"
                "This image may not belong to supported crops (Tomato, Potato, Corn). "
                "Please verify manually." 
            )

        

        # Top-K display
        for rank, pred in enumerate(predictions, start=1):
            label = pred["label"]
            confidence = pred["confidence"]
            status = get_confidence_status(confidence)
            pct = f"{confidence * 100:.2f}%"

            with st.container(border=True):
                col_rank, col_label, col_conf, col_status = st.columns([0.5, 3, 1.5, 2.5])
                col_rank.markdown(f"**#{rank}**")
                col_label.markdown(f"**{format_label(label)}**")
                col_conf.markdown(f"`{pct}`")
                col_status.markdown(status)

                st.progress(max(1, min(int(confidence * 100), 100)))

            # Recommendation
            advice = get_advice(label)
            if rank == 1 and advice:
                st.info(f"💡 Recommendation: {advice}")

    # Footer
    st.divider()
    st.caption("Model: Custom CNN • Input: 224×224 • Classes: 9 • Output: Top-3 predictions")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()