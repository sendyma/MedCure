import glob
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
import numpy as np



class SIIMPneumothoraxCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        all_imgs = np.array(glob.glob(os.path.join(root_dir, 'png-images-test-512', '*.png')))
        df_csv = pd.read_csv(os.path.join(root_dir, gt))
        mask_exist = np.array([True if os.path.splitext(os.path.basename(p))[0] in list(df_csv["ImageId"]) else False for p in all_imgs])
        self.all_imgs = all_imgs[mask_exist]
        ground_truth = [df_csv.loc[df_csv['ImageId'] == os.path.splitext(os.path.basename(img_id))[0], ' EncodedPixels'].values[0].strip() for img_id in self.all_imgs]
        temp = set(ground_truth)   # for check
        self.gr = np.array([[0, 1] if g == '-1' else [1, 0] for g in ground_truth])   # {'Pneumothorax': [1, 0], 'NORMAL': [0, 1]}

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
