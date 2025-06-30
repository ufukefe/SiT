# SiT/datasets/MVTecLOCO.py

import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class MVTecLOCO(VisionDataset):
    """MVTec LOCO Anomaly Detection Dataset.

    Args:
        root (str): Root directory of the dataset.
        category (str): The category of the dataset to load (e.g., 'breakfast_box').
        split (str, optional): The dataset split, supports "train", "validation", or "test".
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = self._verify_split(split)
        self.category = category
        self.data_dir = Path(self.root) / self.category

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Category '{self.category}' not found in '{self.root}'")

        self.image_paths, self.targets, self.mask_paths = self._load_data()

    def _verify_split(self, split: str) -> str:
        """Verify that the split is valid."""
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'validation', 'test'.")
        return split

    def _load_data(self) -> Tuple[list, list, list]:
        """Load image paths, targets, and mask paths for the specified split."""
        image_paths, targets, mask_paths = [], [], []

        if self.split in ["train", "validation"]:
            # Training and validation sets only contain 'good' images
            img_dir = self.data_dir / self.split / "good"
            if not img_dir.is_dir():
                return [], [], []
            
            image_files = sorted(list(img_dir.glob("*.png")))
            image_paths.extend(image_files)
            targets.extend([0] * len(image_files))  # 0 for normal
            mask_paths.extend([None] * len(image_files))

        elif self.split == "test":
            # Test set contains 'good', 'logical_anomalies', and 'structural_anomalies'
            test_subdirs = ["good", "logical_anomalies"]
            gt_base_dir = self.data_dir / "ground_truth"

            for subdir in test_subdirs:
                img_dir = self.data_dir / "test" / subdir
                if not img_dir.is_dir():
                    continue

                image_files = sorted(list(img_dir.glob("*.png")))
                image_paths.extend(image_files)
                
                is_anomalous = subdir != "good"
                targets.extend([1 if is_anomalous else 0] * len(image_files))

                if is_anomalous:
                    for img_path in image_files:
                        # Ground truth can be a single file or a directory of masks
                        gt_path = gt_base_dir / subdir / img_path.stem
                        if gt_path.is_dir():
                            mask_paths.append(list(gt_path.glob("*.png")))
                        else:
                            # This case is less common for LOCO but good to handle
                            mask_paths.append([gt_path.with_suffix(".png")])
                else:
                    mask_paths.extend([None] * len(image_files))

        return image_paths, targets, mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        image_path = self.image_paths[idx]
        target = self.targets[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply image transform
        if self.transform:
            image = self.transform(image)

        if self.split == "train":
            # For training, we only need the transformed image
            return image
        
        # For validation and test, we also provide target and mask
        mask = self._load_mask(idx)
        
        # Apply target transform (for the mask)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, target, mask

    def _load_mask(self, idx: int) -> Image.Image:
        """Loads and combines ground truth masks for a given index."""
        mask_files = self.mask_paths[idx]
        
        # If no mask files, return a black image (no anomalies)
        if mask_files is None or not mask_files:
            # Get image size from the loaded image path to create a correctly sized mask
            with Image.open(self.image_paths[idx]) as img:
                width, height = img.size
            return Image.new('L', (width, height), 0)

        # Combine multiple masks into one
        combined_mask = None
        for mask_path in mask_files:
            mask = Image.open(mask_path).convert('L')
            if combined_mask is None:
                combined_mask = np.array(mask)
            else:
                # A pixel is anomalous if it's non-zero in ANY mask
                combined_mask = np.maximum(combined_mask, np.array(mask))
        
        # Binarize the mask: 0 for normal, 255 for anomaly
        combined_mask[combined_mask > 0] = 255
        
        return Image.fromarray(combined_mask)