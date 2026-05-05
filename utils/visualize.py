import cv2
import numpy as np
from PIL import Image


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:

    if heatmap is None:
        return image

    heatmap = np.clip(heatmap, 0, 1)

    target_w, target_h = image.size

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    original_np = np.array(image.convert("RGB"), dtype=np.uint8)

    blended = cv2.addWeighted(
        colored, alpha,
        original_np, 1.0 - alpha,
        0,
    )

    return Image.fromarray(blended)