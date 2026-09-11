import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Labels = {
    "PNEUMONIA": [1, 0],
    "NORMAL": [0, 1],
}


class Pneumonia_Xray2017(Dataset):
    def __init__(self, root_dir, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform

        img_list = glob.glob(os.path.join(root_dir, '*', '*.*'))
        gr_str = [path.split('/')[-2] for path in img_list]
        self.all_imgs = np.asarray(img_list)
        self.gr = np.asarray([Labels[g] for g in gr_str])

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = self.all_imgs[index]
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
