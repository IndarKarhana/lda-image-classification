"""
CUB-200-2011 Dataset Loader
============================
Fine-grained bird classification dataset.
200 classes, ~5,994 train / ~5,794 test images.

Auto-downloads and extracts if not present.

Reference:
  Wah et al., "The Caltech-UCSD Birds-200-2011 Dataset", 2011
"""

import os
import tarfile
import urllib.request
from typing import Optional, Callable, Tuple

from PIL import Image
from torch.utils.data import Dataset


CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_DIR_NAME = "CUB_200_2011"


class CUB200(Dataset):
    """CUB-200-2011 fine-grained bird classification dataset.

    Args:
        root: Root directory (CUB_200_2011/ will be inside this)
        train: If True, use training split; else test split
        transform: Optional image transform
        download: If True, download dataset if not found
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(root, CUB_DIR_NAME)

        if download and not os.path.isdir(self.data_dir):
            self._download()

        if not os.path.isdir(self.data_dir):
            raise RuntimeError(
                f"Dataset not found at {self.data_dir}. Use download=True."
            )

        self.images, self.labels = self._load_split()

    def _download(self) -> None:
        """Download and extract CUB-200-2011."""
        os.makedirs(self.root, exist_ok=True)
        tgz_path = os.path.join(self.root, "CUB_200_2011.tgz")

        if not os.path.exists(tgz_path):
            print(f"Downloading CUB-200-2011 (~1.1 GB)...")
            urllib.request.urlretrieve(CUB_URL, tgz_path)
            print(f"  Downloaded to {tgz_path}")

        print(f"Extracting CUB-200-2011...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        print(f"  Extracted to {self.data_dir}")

    def _load_split(self) -> Tuple[list, list]:
        """Load train/test split from metadata files."""
        # Read image paths
        images_file = os.path.join(self.data_dir, "images.txt")
        with open(images_file, "r") as f:
            id_to_path = {}
            for line in f:
                img_id, path = line.strip().split()
                id_to_path[int(img_id)] = path

        # Read labels (1-indexed → 0-indexed)
        labels_file = os.path.join(self.data_dir, "image_class_labels.txt")
        with open(labels_file, "r") as f:
            id_to_label = {}
            for line in f:
                img_id, label = line.strip().split()
                id_to_label[int(img_id)] = int(label) - 1  # 0-indexed

        # Read train/test split (1 = train, 0 = test)
        split_file = os.path.join(self.data_dir, "train_test_split.txt")
        with open(split_file, "r") as f:
            id_to_split = {}
            for line in f:
                img_id, is_train = line.strip().split()
                id_to_split[int(img_id)] = int(is_train)

        # Filter by split
        target_split = 1 if self.train else 0
        images = []
        labels = []
        for img_id in sorted(id_to_path.keys()):
            if id_to_split[img_id] == target_split:
                full_path = os.path.join(self.data_dir, "images", id_to_path[img_id])
                images.append(full_path)
                labels.append(id_to_label[img_id])

        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    """Quick test of CUB-200 loading."""
    train_ds = CUB200(root="data", train=True, download=True)
    test_ds = CUB200(root="data", train=False, download=True)
    n_classes = len(set(train_ds.labels))
    print(f"CUB-200-2011: {len(train_ds)} train, {len(test_ds)} test, {n_classes} classes")
    print(f"  Max LDA components: {n_classes - 1}")
