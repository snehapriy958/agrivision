import io
from pathlib import Path
from typing import Union, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from utils.config import (
    IMAGE_SIZE,
    RESIZE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_IMAGE_EXTENSIONS,
)

__all__ = ["preprocess_image", "build_inference_transform"]


ImageInput = Union[str, Path, Image.Image, io.BytesIO, bytes]

def build_inference_transform() -> transforms.Compose:
    
    return transforms.Compose([
        
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE), antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),          # uint8 HWC → float32 CHW, scales to [0, 1]
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])

# Module-level singleton — built once, zero overhead on repeated calls.
_INFERENCE_TRANSFORM: transforms.Compose = build_inference_transform()

def _to_pil_rgb(source: ImageInput) -> Image.Image:
    
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    if isinstance(source, io.BytesIO):
        try:
            source.seek(0)          # guard against partially-consumed streams
            img = Image.open(source)
            return img.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Stream does not contain a decodable image: {exc}") from exc

    # str or Path — file on disk
    path = Path(source)

    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format.")

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: '{path}'")
    if not path.is_file():
        raise ValueError(f"Path exists but is not a file: '{path}'")

    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot decode image at '{path}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"OS error while reading '{path}': {exc}") from exc

def preprocess_image(
    source: ImageInput,
    *,
    return_pil: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Image.Image]]:
    
    pil_image: Image.Image = _to_pil_rgb(source)

    if pil_image.size[0] < 10 or pil_image.size[1] < 10:
        raise ValueError("Image dimensions too small.")
    
    tensor: torch.Tensor = _INFERENCE_TRANSFORM(pil_image).unsqueeze(0)

    if torch.isnan(tensor).any():
        raise RuntimeError("Tensor contains NaN values.")


    expected = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            f"Preprocessing produced unexpected shape {tuple(tensor.shape)}, "
            f"expected {expected}. Check IMAGE_SIZE and RESIZE_SIZE in config.py."
        )
    
    if tensor.dtype != torch.float32:
        raise RuntimeError("Tensor dtype mismatch.")


    if return_pil:
        return tensor, pil_image
    return tensor