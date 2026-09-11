import numpy as np
import torch
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path

# Column order must match labels.json["chexpert_test"]: zeroshot_binary_prompt
# builds its prompts from that list and pairs prompt c with self.gr[:, c].
CheXpert_labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                   'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']


class CheXpertTestDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        cxr_dir = Path(os.path.join(root_dir, 'CheXpert/test'))
        cxr_paths = list(cxr_dir.rglob("*.jpg"))
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths))  # filter only first frontal views
        self.cxr_paths = sorted(cxr_paths)  # sort to align with groundtruth

        df_csv = pd.read_csv(os.path.join(root_dir, gt))
        self.gr = np.array([df_csv[label] for label in CheXpert_labels]).transpose()

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.cxr_paths[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path


class CheXpertRetrieveDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv_gt = pd.read_csv(os.path.join(root_dir, "chexpert_5x200.csv"), low_memory=False)
        gt_paths = df_csv_gt['Path'].apply(lambda path: path.split('CheXpert-v1.0/')[1]).tolist()
        df_csv = pd.read_csv(os.path.join(root_dir, 'df_chexpert_plus_240401.csv'))   # 1000
        mask = df_csv["path_to_image"].apply(lambda path: path in gt_paths)
        df_csv_new = df_csv[mask]  # 998

        df_csv_new.dropna(subset=["section_findings", "section_impression"], how="all", inplace=True)
        df_csv_new[["section_findings"]] = df_csv_new[["section_findings"]].fillna(" ")
        df_csv_new[["section_impression"]] = df_csv_new[["section_impression"]].fillna(" ")

        self.df = df_csv_new
        self.all_imgs = np.asarray([x for x in self.df['path_to_image']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)

        row = self.df.iloc[index]
        captions = ""
        captions += row["section_impression"]
        captions += " "
        captions += row["section_findings"]

        # use space instead of newline
        captions_raw = captions.replace("\n", " ")

        #     captions_raw, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_bert_length
        # )


        return data, captions_raw, img_path
