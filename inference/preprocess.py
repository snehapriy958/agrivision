import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PlantDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Get class names (folder names)
        self.class_names = sorted(os.listdir(root_dir))

        # Load all image paths
        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                self.image_paths.append(img_path)
                self.labels.append(label)

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        img = self.transform(img)

        return img, label