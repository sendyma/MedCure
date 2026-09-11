import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


Labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']


class MIMICCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv = pd.read_csv(os.path.join(root_dir, '2.0.0/metadata', gt))

        df_csv = df_csv[df_csv["ViewPosition"].isin(["AP", 'PA'])]  # only consider frontal view

        self.all_imgs = np.asarray([x for x in df_csv['Path']])
        self.gr = np.array([df_csv[label].fillna(0) for label in Labels]).transpose()  # no annotation as 0, uncertainty is -1

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path


class MIMICCXRRetrieveDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv = pd.read_csv(os.path.join(root_dir, '2.0.0/metadata', gt))
        df_csv = df_csv[df_csv["split"] == 'test']

        df_csv = df_csv[df_csv["ViewPosition"].isin(["AP", 'PA'])]  # only consider frontal view

        self.all_imgs = np.asarray([x for x in df_csv['Path']])
        self.df = df_csv

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)

        row = self.df.iloc[index]
        captions = ""
        captions += row["impression"]
        captions += " "
        captions += row["findings"]

        # use space instead of newline
        captions_raw = captions.replace("\n", " ")

        #     captions_raw, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_bert_length
        # )


        return data, captions_raw, img_path
