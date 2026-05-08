"""
utils/config.py — Single source of truth for the plant disease classification system.
Covers training, inference, preprocessing, GradCAM, FastAPI backend, and Streamlit frontend.
"""

import torch
from pathlib import Path
import os
from typing import Tuple, Dict, Set

# ─────────────────────────────────────────────
# 1. ROOT & DATA PATHS
# ─────────────────────────────────────────────

# Resolves to the project root regardless of where the script is invoked from.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR:       Path = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
MODEL_PATH: Path = ARTIFACTS_DIR / "best_model.pth"
LOG_DIR:        Path = BASE_DIR / "logs"
OUTPUTS_DIR:    Path = BASE_DIR / "outputs"
GRADCAM_OUTPUT_DIR: Path = OUTPUTS_DIR / "gradcam"

# ─────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ─────────────────────────────────────────────

SEED: int = 42  # Used by train.py, preprocess.py, and any data-split logic.
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────
# 3. DEVICE
# ─────────────────────────────────────────────

# Auto-detects the best available device.
DEVICE: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()   # Apple Silicon
    else "cpu"
)

# ─────────────────────────────────────────────
# 4. MODEL / TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────

IMAGE_SIZE:    int   = 224 
RESIZE_SIZE: int = 256         # EfficientNet-B0 canonical input size.
BATCH_SIZE:    int   = 32
LEARNING_RATE: float = 1e-4
EPOCHS:        int   = 25


MODEL_NAME: str = "efficientnet_b0"
MODEL_VERSION: str = "v1"

TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.2

CONFIDENCE_THRESHOLD: float = 0.6
TOP_K_PREDICTIONS: int = 3

NUM_WORKERS: int  = min(4, os.cpu_count() or 1)
PIN_MEMORY: bool = DEVICE == "cuda"
VALID_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────
# 5. IMAGENET NORMALISATION
# ─────────────────────────────────────────────


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────
# 6. CLASS NAMES  ← the single source of truth
# ─────────────────────────────────────────────

# Integer-to-class mapping
CLASS_NAMES = [
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
]

NUM_CLASSES: int = len(CLASS_NAMES)
EXPECTED_NUM_CLASSES: int = 9
assert NUM_CLASSES == EXPECTED_NUM_CLASSES

IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
