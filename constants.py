"""Filesystem locations of every corpus MedCure trains or evaluates on.

All paths hang off one root, so pointing the code at your own copies is a single
environment variable:

    export MEDCURE_DATA_ROOT=/path/to/your/cxr/corpora

The per-corpus sub-directory and CSV names below still have to match your
layout -- see the table in README.md for the columns each CSV must provide.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("MEDCURE_DATA_ROOT", "data"))

CHEXPERT_DATA_DIR = DATA_ROOT / ('CheXpert-v1.0')
CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR / "df_chexpert_plus_240401.csv"
MIMIC_CXR_DATA_DIR = DATA_ROOT / ("mimic-cxr-jpg-512/2.0.0")
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR / "metadata/master.csv"
PadChest_CXR_DATA_DIR = DATA_ROOT / ("padchest/")
PadChest_CXR_TRAIN_CSV = PadChest_CXR_DATA_DIR / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
Open_I_CXR_DATA_DIR = DATA_ROOT / ("open-i/")
Open_I_CXR_TRAIN_CSV = Open_I_CXR_DATA_DIR / "indiana_reports.csv"
BIMCV_CXR_DATA_DIR = DATA_ROOT / ("bimcv-covid19/")
BIMCV_CXR_TRAIN_CSV = BIMCV_CXR_DATA_DIR / "final_annotations.csv"
CANDID_CXR_DATA_DIR = DATA_ROOT / ("candid/")
CANDID_CXR_TRAIN_CSV = CANDID_CXR_DATA_DIR / "Pneumothorax_reports.csv"
ReXGradient_CXR_DATA_DIR = DATA_ROOT / ("ReXGradient-160K/")
ReXGradient_CXR_TRAIN_CSV = ReXGradient_CXR_DATA_DIR / "train_metadata.csv"
CASIA_CXR_DATA_DIR = DATA_ROOT / ("CASIA-CXR/")
CASIA_CXR_TRAIN_CSV = CASIA_CXR_DATA_DIR / "CASIA-CXR_Combined_Reports.csv"
Brax_CXR_DATA_DIR = DATA_ROOT / ("brax/")
Brax_CXR_TRAIN_CSV = Brax_CXR_DATA_DIR / "master_spreadsheet_update.csv"
ChestDR_CXR_DATA_DIR = DATA_ROOT / ("ChestDR/")
ChestDR_CXR_TRAIN_CSV = ChestDR_CXR_DATA_DIR / "chestdr_published_19d.csv"
NIHChestXray14_CXR_DATA_DIR = DATA_ROOT / ("chestxray14-512/")
NIHChestXray14_CXR_TRAIN_CSV = NIHChestXray14_CXR_DATA_DIR / "Data_Entry_2017.csv"
Vindr_CXR_DATA_DIR = DATA_ROOT / ("vindr-cxr/")
Vindr_CXR_TRAIN_CSV = Vindr_CXR_DATA_DIR / "image_labels_train.csv"
Vindr_PCXR_DATA_DIR = DATA_ROOT / ("vindr-pcxr/")
Vindr_PCXR_TRAIN_CSV = Vindr_PCXR_DATA_DIR / "image_labels_train.csv"
