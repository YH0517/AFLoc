from pathlib import Path
import numpy as np

MIMIC_DATA_DIR = Path("/mnt/disk3/hwj/MIMIC-DATA-Final/MIMIC-CXR/")
MS_CXR_JSON = Path("/mnt/disk3/hwj/MIMIC-DATA-Final/ms-cxr/MS_CXR_Local_Alignment_v1.0.0.json")
MIMIC_IMG_DIR = MIMIC_DATA_DIR / "MIMIC-224-inter-area/files/"

RSNA_DATA_DIR = Path("/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/SIIM_ACR_Pneumothorax_and_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/")
RSNA_IMG_DIR = RSNA_DATA_DIR / "png_all"
RSNA_CSV = RSNA_DATA_DIR / "stage_2_train_labels.csv"
RSNA_GROUNDING_CSV = RSNA_DATA_DIR / "grounding_test.csv"

PNEUMOTHORAX_DATA_DIR = Path("/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/SIIM_ACR_Pneumothorax_and_RSNA_Pneumonia/SIIM ACR Pneumothorax Segmentation Data/")
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_ORIGINAL_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_MAP_CSV = PNEUMOTHORAX_DATA_DIR / "map.csv"

COVID_RURAL_DATA_DIR = Path("/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/covid_rural_annot/")
COVID_RURAL_IMG_DIR = COVID_RURAL_DATA_DIR / "jpgs"
COVID_RURAL_MASK_DIR = COVID_RURAL_DATA_DIR / "pngs_masks"

CHEXLOCALIZE_DATA_DIR = Path("/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/chexlocalize/")
CHEXLOCALIZE_TEST_IMG_DIR = CHEXLOCALIZE_DATA_DIR / "CheXpert/test"
CHEXLOCALIZE_TEST_JSON = CHEXLOCALIZE_DATA_DIR / "CheXlocalize/gt_segmentations_test.json"
CHEXLOCALIZE_VAL_IMG_DIR = CHEXLOCALIZE_DATA_DIR / "CheXpert/val"
CHEXLOCALIZE_VAL_JSON = CHEXLOCALIZE_DATA_DIR / "CheXlocalize/gt_segmentations_val.json"

threshold_list_dct = {
    "MS_CXR": list(np.arange(0.1, 0.6, 0.1)),
    "MS_CXR_CLS": list(np.arange(0.1, 0.6, 0.1)),
    "RSNA_GROUNDING": list(np.arange(0.1, 0.6, 0.1)),
    "COVID_RURAL": list(np.arange(0.1, 0.6, 0.1)),
    "CHEXLOCALIZE": list(np.arange(0.1, 0.6, 0.1)),
    "SICAPV2_GROUNDING": list(np.arange(-1, 0, 0.1)),
}
