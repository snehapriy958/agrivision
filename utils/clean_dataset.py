import os
import shutil

DATA_DIR = "data/raw"

#  Keep only these (TARGET STRUCTURE)
KEEP_CLASSES = {
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
}

#  Mapping from your current names → correct names
RENAME_MAP = {
    "corn_Bligh": "Corn___Cercospora_leaf_spot",
    "corn_Common_Rust": "Corn___Common_rust",
    "corn_Healthy": "Corn___healthy",

    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",

    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_healthy": "Tomato___healthy"
}

for folder in os.listdir(DATA_DIR):
    old_path = os.path.join(DATA_DIR, folder)

    # Skip non-directories
    if not os.path.isdir(old_path):
        continue

    # Rename if needed
    new_name = RENAME_MAP.get(folder, folder)

    if new_name not in KEEP_CLASSES:
        print(f" Deleting: {folder}")
        shutil.rmtree(old_path)
    else:
        new_path = os.path.join(DATA_DIR, new_name)
        if old_path != new_path:
            print(f" Renaming: {folder} → {new_name}")
            os.rename(old_path, new_path)