import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class TGSDataset(Dataset):
    """
    TGS Salt Identification dataset loader.
    - Grayscale → 3 channel
    - Resize to 256x256
    - No heavy augmentation
    """

    def __init__(self, image_dir, mask_dir=None, img_size=256):

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        img = np.stack([img, img, img], axis=-1)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32)

    def _load_mask(self, img_path):

        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 127).astype(np.int64)

        return torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        if self.mask_dir is not None:
            mask = self._load_mask(img_path)
            return image, mask

        return image