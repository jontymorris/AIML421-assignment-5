import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths, self.labels = self._load_image_paths_and_labels(root_dir)
        print(f'Loaded {len(self.image_paths)} images')

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

    def _load_image_paths_and_labels(self, root_dir):
        image_paths, labels = [], []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.startswith(("cherry_", "strawberry_", "tomato_")):
                    img_path = os.path.join(root, file)
                    label = 0 if file.startswith("cherry_") else 1 if file.startswith("strawberry_") else 2
                    image_paths.append(img_path)
                    labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
