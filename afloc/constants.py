from pathlib import Path

PRETRAIN_VIEW_COL = "view"
PRETRAIN_PATH_COL = "path"
PRETRAIN_SPLIT_COL = "split"
PRETRAIN_REPORT_COL = "report"
PRETRAIN_IMPRESSION_COL = "impression"
PICKLE_SUFFIX = "report"

# MIMIC constants
MIMIC_DATA_DIR = Path("/mnt/disk2/hwj/MIMIC-DATA-Final/MIMIC-CXR/")
MIMIC_IMG_DIR = MIMIC_DATA_DIR / "MIMIC-224-inter-area/files/"
MIMIC_ORIGINAL_TRAIN_CSV = MIMIC_DATA_DIR / "BASE-MIMIC.csv"

MIMIC_MASTER_CSV = MIMIC_DATA_DIR / "BASE-MIMIC.csv"  # contains patient information, not PHI conplient

MIMIC_VALID_NUM = 5000
MIMIC_VIEW_COL = "view"
MIMIC_PATH_COL = "path"
MIMIC_SPLIT_COL = "split_with_MS"
MIMIC_REPORT_COL = "report"
MIMIC_IMPRESSION_COL = "impression"
MIMIC_FINDINGS_COL = "findings"

MIMIC_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
MIMIC_USED_COLS = [
    MIMIC_PATH_COL, 
    MIMIC_VIEW_COL, 
    MIMIC_SPLIT_COL, 
    MIMIC_REPORT_COL,
    MIMIC_IMPRESSION_COL] + MIMIC_TASKS




