import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


Labels = ['Bronchitis', 'Brocho-pneumonia', 'Bronchiolitis', 'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia',
          'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease',
          'Mediastinal tumor', 'Lung tumor', 'Other disease', 'No finding']


class VinDrPCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv = pd.read_csv(os.path.join(root_dir, gt))

        self.all_imgs = np.asarray([x for x in df_csv['image_id']])
        self.gr = np.array([df_csv[label] for label in Labels]).transpose()

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "test_png", self.all_imgs[index] + '.png')
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
