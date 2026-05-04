"""
frontend/app.py — AgriVision Streamlit Frontend
Run : streamlit run frontend/app.py
"""


import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
from inference.predictor import predict
import tempfile



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



def call_api(image_file):
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_file.seek(0)
        tmp.write(image_file.read())
        tmp_path = tmp.name

    return predict(tmp_path)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AgriVision", page_icon="🌿", layout="centered")

    st.title("🌿 AgriVision — Crop Disease Detection")
    st.caption("Upload a crop image to detect potential diseases and receive care advice.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.divider()

        if st.button("🔍 Predict", use_container_width=True):
            with st.spinner("Analysing image…"):
                try:
                    predictions = call_api(uploaded_file)
                    if not predictions:
                        st.error("No predictions returned from API")
                        return

                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    return

            st.subheader("Top Predictions")
            top = predictions[0]
            st.success(
                f"Most likely: **{format_label(top['label'])}** "
                f"({top['confidence']*100:.2f}%)"
            )

            for rank, pred in enumerate(predictions, start=1):
                label      = pred["label"]
                confidence = pred["confidence"]
                status     = get_confidence_status(confidence)
                pct = f"{max(confidence * 100, 0.01):.2f}%"
                
                with st.container(border=True):
                    col_rank, col_label, col_conf, col_status = st.columns([0.5, 3, 1.5, 2.5])
                    col_rank.markdown(f"**#{rank}**")
                    col_label.markdown(f"**{format_label(label)}**")
                    col_conf.markdown(f"`{pct}`")
                    col_status.markdown(status)
                    st.progress(min(int(confidence * 100), 100))

                advice = get_advice(label)
                if rank == 1 and advice:
                    st.info(f"💡 Recommendation: {advice}")


if __name__ == "__main__":
    main()