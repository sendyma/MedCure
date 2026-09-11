import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Labels = {
    "Tuberculosis": [1, 0],
    "Non-Tuberculosis": [0, 1],
}


class TBCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        with open(os.path.join(root_dir, 'lists', gt), "r") as f:
            image_list = f.readlines()
        labels_str = ['Tuberculosis' if f.split('/')[0] == 'tb' else 'Non-Tuberculosis' for f in image_list]

        self.all_imgs = np.asarray([f.strip() for f in image_list])
        self.gr = np.array([Labels[label] for label in labels_str])

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "imgs", self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
