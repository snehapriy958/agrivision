"""
utils/visualize.py — Grad-CAM heatmap overlay utility.
"""

import cv2
import numpy as np
from PIL import Image


def overlay_heatmap_on_image(
    image   : Image.Image,
    heatmap : np.ndarray | None,
    alpha   : float = 0.4,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto a PIL image.

    Pipeline:
        Normalize heatmap → Resize to image dims → Apply JET colormap
        → Alpha-blend with original image → Return PIL image

    Args:
        image   : Original PIL image (any mode, any size).
        heatmap : 2-D float array of shape (H, W) with values in [0, 1].
                  If None, the original image is returned unchanged.
        alpha   : Heatmap blend weight (0 = original only, 1 = heatmap only).
                  Default 0.4 keeps the leaf texture clearly visible.

    Returns:
        Blended PIL image in RGB mode, same dimensions as the input.

    Raises:
        ValueError: If `heatmap` is not a 2-D array.
    """
    if heatmap is None:
        return image.convert("RGB")
    
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected a 2-D heatmap array, got shape {heatmap.shape}."
        )

    target_w, target_h = image.size

    # ── Normalize to [0, 1] ───────────────────────────────────────────
    heatmap = heatmap.astype(np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        # Uniform map — return original image to avoid a meaningless overlay
        return image.convert("RGB")

    # ── Clamp and convert to uint8 ────────────────────────────────────
    heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

    # ── Resize to match image dimensions ─────────────────────────────
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # ── Apply JET colormap (OpenCV returns BGR) ───────────────────────
    colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)           # → RGB

    # ── Convert original image to uint8 numpy array ───────────────────
    original_np = np.array(image.convert("RGB"), dtype=np.uint8)

    # ── Alpha blend: alpha * heatmap + (1 - alpha) * original ─────────
    blended = cv2.addWeighted(
        colored_rgb, alpha,
        original_np, 1.0 - alpha,
        0,
    )

    return Image.fromarray(blended)