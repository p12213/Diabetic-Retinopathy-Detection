"""
PyTorch Dataset & Augmentation Loader for Retinal Fundus Images (APTOS 2019 / EyePACS).
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from preprocess import crop_image_from_gray

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, ben_graham=True):
        """
        df: pandas DataFrame containing 'id_code' (or 'image') and 'diagnosis' columns.
        img_dir: Path to image directory.
        transform: Albumentations image transformation pipeline.
        is_test: If True, diagnosis label is ignored.
        ben_graham: If True, applies circle crop preprocessing.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.ben_graham = ben_graham
        
        # Auto-detect column names
        self.img_col = 'id_code' if 'id_code' in self.df.columns else ('image' if 'image' in self.df.columns else self.df.columns[0])
        if not self.is_test:
            self.target_col = 'diagnosis' if 'diagnosis' in self.df.columns else 'level'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, self.img_col]
        # Append extension if missing
        if not (img_name.endswith('.png') or img_name.endswith('.jpeg') or img_name.endswith('.jpg')):
            img_path = os.path.join(self.img_dir, f"{img_name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        else:
            img_path = os.path.join(self.img_dir, img_name)

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback black image if missing
            img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if self.ben_graham:
                img_rgb = crop_image_from_gray(img_rgb)

        if self.transform:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                augmented = self.transform(image=img_rgb)
                img_tensor = augmented['image']
            else:
                img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255.0

        if self.is_test:
            return img_tensor, img_name

        label = int(self.df.loc[idx, self.target_col])
        return img_tensor, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=512):
    """Returns training and validation albumentations transforms."""
    if not HAS_ALBUMENTATIONS:
        print("Warning: albumentations package not found. Using default normalization.")
        return None, None
        
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform

if __name__ == "__main__":
    print("RetinopathyDataset module loaded successfully.")
