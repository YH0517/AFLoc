import os
import cv2
import copy
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools
import pydicom
import matplotlib
from pathlib import Path
FONT_MAX = 50
matplotlib.use('Agg')
from localization.constants import (MIMIC_IMG_DIR, MS_CXR_JSON, RSNA_CSV, 
    RSNA_IMG_DIR, RSNA_GROUNDING_CSV, CHEXLOCALIZE_VAL_JSON, CHEXLOCALIZE_VAL_IMG_DIR, 
    CHEXLOCALIZE_TEST_JSON, CHEXLOCALIZE_TEST_IMG_DIR, COVID_RURAL_IMG_DIR,
    COVID_RURAL_MASK_DIR)


def box_mapping(box, w, h, scale):
    """
    Inputs:
        - box (np.ndarray): box in the original image (x, y, w, h)
        - w (int): width of the original image
        - h (int): height of the original image
        - scale (int): the scale of the resized image

    Returns:
        - box (np.ndarray): box in the resized image
    """
    size = (h, w)
    max_dim = max(size)
    max_ind = size.index(max_dim)
    box = np.array(box)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        box = box * wpercent
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        box = box * hpercent

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - desireable_size[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - desireable_size[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)

    box[0] = int(np.floor(box[0] + left))
    box[1] = int(np.floor(box[1] + top))
    box[2] = int(np.floor(box[2]))
    box[3] = int(np.floor(box[3]))

    return box.astype(np.int32)


def box2mask(box, w, h):
    """
    Inputs:
        - box (np.ndarray): box in the resized image (x, y, w, h)
        - w (int): width of the resized image
        - h (int): height of the resized image

    Returns:
        - mask (np.ndarray): mask in the resized image
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    return mask


def norm_heatmap(heatmap_, nan, mode=0):
    # mode: 0 -> "[-1,1]"
    #       1 -> "[0, 1]"
    heatmap = copy.deepcopy(heatmap_)
    heatmap_wo_nan = heatmap[~nan]

    if heatmap_wo_nan.max() - heatmap_wo_nan.min() == 0:
        print(f"heatmap max == min == {heatmap_wo_nan.max()}")
        return heatmap_wo_nan
    
    heatmap_wo_nan = (heatmap_wo_nan - heatmap_wo_nan.min()) / (heatmap_wo_nan.max() - heatmap_wo_nan.min())

    if mode == 0:
        heatmap_wo_nan  = heatmap_wo_nan * 2 - 1 
    heatmap[~nan] = heatmap_wo_nan
    return heatmap


def load_data(dataset, **kwargs):
    if dataset == "MS_CXR":
        return load_ms_cxr(**kwargs)
    if dataset == "MS_CXR_CLS":
        return load_ms_cxr(use_cxr_text=False, **kwargs)
    elif dataset == "RSNA_GROUNDING":
        return load_rsna_grounding(**kwargs)
    elif dataset == "COVID_RURAL":
        return load_covid_rural(**kwargs)
    elif dataset == "CHEXLOCALIZE":
        return load_chexlocalize(**kwargs)
    elif dataset == "SICAPV2_GROUNDING":
        return load_sicapv2_grounding(**kwargs)
    else:
        raise NotImplementedError


def load_ms_cxr(merge=True, use_cxr_text=True, **kwargs):
    print("loading data...")
    if merge == True:
        data = merge_annotation(MS_CXR_JSON, use_cxr_text=use_cxr_text)
    else:
        data = get_annotation(MS_CXR_JSON)
    data["path"] = list(map(lambda x: MIMIC_IMG_DIR/x.replace("files/", ""), data["path"]))

    return data


def load_rsna_grounding(**kwargs):
    print("loading data...")
    size = 224
    path_to_file = RSNA_GROUNDING_CSV
    path_images = RSNA_IMG_DIR
    df = pd.read_csv(path_to_file)
    df = df[df["classes"] == 1]

    label_text_list = ['Findings suggesting pneumonia.'] * len(df)
    category_list = ['Pneumonia'] * len(df)
    path_list = []
    gtmasks_list = []
    boxes_list = []
    save_path = os.path.join("./dataset", f"rsna_grounding.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data

    for idx, row in df.iterrows():
        pid = row["ID"]
        path = path_images / (pid + '.png')
        path_list.append(path)
        boxes = []
        mask = None
        for box_str in row["boxes"].split("|"):
            bbox = list(map(int, map(float, box_str.split(";"))))
            tbox = box_mapping(bbox, 1024, 1024, size)
            boxes.append(tbox)
            if mask is None:
                mask = box2mask(tbox, size, size)
            else:
                mask += box2mask(tbox, size, size)
        gtmasks_list.append(mask)
        boxes_list.append(boxes)

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data


def load_covid_rural(**kwargs):
    label_text_list = []
    category_list = []
    path_list = []
    gtmasks_list = []
    boxes_list = []
    save_path = os.path.join("./dataset", f"covid_rural.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data

    img_list = sorted([i.rstrip(".jpg") for i in os.listdir(COVID_RURAL_IMG_DIR)
                if i.endswith(".jpg")])
    mask_list = [i.rstrip(".png") for i in os.listdir(COVID_RURAL_MASK_DIR)
                 if i.endswith(".png")]
    
    for img in tqdm(img_list):
        if img not in mask_list:
            continue
        path = os.path.join(COVID_RURAL_IMG_DIR, img + ".jpg")
        mask_path = os.path.join(COVID_RURAL_MASK_DIR, img + ".png")
        mask = Image.open(mask_path)
        mask = np.array(mask).astype("float32")
        if mask.sum() == 0:
            continue
        mask = _resize_img(mask, 224)
        mask = (mask >= 1e-5).astype("uint8")

        path_list.append(Path(path))
        gtmasks_list.append(mask)
        boxes_list.append(None)
        category_list.append("COVID-19")
        label_text_list.append("Findings suggesting pneumonia.") # COVID-19

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data


def load_chexlocalize(**kwargs):
    LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                       "Cardiomegaly",
                       "Lung Lesion",
                       "Airspace Opacity",
                       "Edema",
                       "Consolidation",
                       "Atelectasis",
                       "Pneumothorax",
                       "Pleural Effusion",
                       "Support Devices"]
    suffix = "_" + kwargs["split"] if "split" in kwargs else "_test"
    save_path = os.path.join("./dataset", f"chexlocalize{suffix}.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data
    
    label_text_list = []
    category_list = []
    path_list = []
    gtmasks_list = []
    boxes_list = []
    label_list = []
    if "split" in kwargs and kwargs["split"] == "val":
        jsonfile = CHEXLOCALIZE_VAL_JSON
        img_dir = CHEXLOCALIZE_VAL_IMG_DIR
    else:
        jsonfile = CHEXLOCALIZE_TEST_JSON
        img_dir = CHEXLOCALIZE_TEST_IMG_DIR
        
    with open(jsonfile) as f:
        gt_dict = json.load(f)
    cxr_ids = gt_dict.keys()
    cxr_ids = sorted(list(cxr_ids))
    neg = 0
    pos = 0
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                
                cxr_id = "_".join(path.split("/")[-3:]).rstrip(".jpg")
                if cxr_id not in cxr_ids:
                    for category in LOCALIZATION_TASKS:
                        label_text_list.append(f"Findings suggesting {category}.")
                        category_list.append(category)
                        gtmasks_list.append(np.zeros((224, 224)))
                        boxes_list.append(None)
                        path_list.append(Path(path))
                        neg += 1
                        label_list.append(0)
                else:
                    for category, elem in gt_dict[cxr_id].items():
                        gt_mask = pycocotools.mask.decode(elem)
                        if gt_mask.sum() == 0:
                            gt_mask_ = np.zeros((224, 224))
                            neg += 1
                            label_list.append(0)
                        else:
                            gt_mask_ = _resize_img(gt_mask, 224)
                            gt_mask_ = (gt_mask_ >= 1e-5).astype("uint8")
                            pos += 1
                            label_list.append(1)
                        gtmasks_list.append(gt_mask_)
                        boxes_list.append(None)
                        category_list.append(category)
                        label_text_list.append(f"Findings suggesting {category}.")
                        path_list.append(Path(path))
                        
    print(f"neg: {neg}, pos: {pos}", "total: ", neg + pos)
    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    data["label"] = label_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data


def rle2mask(rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T


def _resize_img(img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


def merge_annotation(path_to_json, scale=224, use_cxr_text=True):
    coco = COCO(annotation_file=path_to_json)
    cats = coco.cats
    merged = {}
    merged["path"] = []
    merged["gtmasks"] = []
    merged["label_text"] = []
    merged["boxes"] = []
    merged["category"] = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        path = img["path"]
        mask_dct = {}
        bbox_dct = {}
        cats_dct = {}
        for ann in anns:
            bbox = ann["bbox"]
            w = ann["width"]
            h = ann["height"]
            category_id = ann["category_id"]
            tbox = box_mapping(bbox, w, h, scale)
            mask = box2mask(tbox, scale, scale)
            if use_cxr_text:
                category = cats[category_id]["name"]
                label_text = ann["label_text"].lower()

                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category
            else:
                category = cats[category_id]["name"]
                label_text = f"Findings suggesting {category}."
                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category

        for k, v in mask_dct.items():
            merged["path"].append(path)
            merged["gtmasks"].append(v)
            merged["label_text"].append(k)
            merged["boxes"].append(bbox_dct[k])
            merged["category"].append(cats_dct[k])

    return merged


def get_annotation(path_to_json, scale=224):
    coco = COCO(annotation_file=path_to_json)
    res = {}
    res["path"] = []
    res["gtmasks"] = []
    res["label_text"] = []
    res["boxes"] = []
    res["category"] = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        path = img["path"]
        for ann in anns:
            bbox = ann["bbox"]
            w = ann["width"]
            h = ann["height"]
            label_text = ann["label_text"]
            category_id = ann["category_id"]
            tbox = box_mapping(bbox, w, h, scale)
            mask = box2mask(tbox, scale, scale)
            category = coco.cats[category_id]["name"]
            res["path"].append(path)
            res["gtmasks"].append(mask)
            res["label_text"].append(label_text)
            res["boxes"].append([tbox])
            res["category"].append(category)
            
    return res


def read_from_dicom(img_path):
    dcm = pydicom.read_file(img_path, force=True)
    x = dcm.pixel_array
    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    img = Image.fromarray(x)
    return img


def load_sicapv2_grounding(**kwargs):
    prompts = {
        "NC": ["an example of benign prostate tissue."],
        "G3": ["an example of prostate cancer, gleason pattern 3."],
        "G4": ["an example of gleason grade 4."],
        "G5": ["an example of prostate cancer, gleason pattern 5."]
        }
    data = {}
    data["path"] = []
    data["gtmasks"] = []
    data["label_text"] = []
    data["boxes"] = []
    data["category"] = []
    img_dir = "/mnt/disk3/yh/dataset/SICAP/SICAPv2/images_224/"
    mask_dir = "/mnt/disk3/yh/dataset/SICAP/SICAPv2/masks/"
    df_test = pd.read_excel(os.path.join("/mnt/disk3/yh/dataset/SICAP/SICAPv2/partition/Test/", "Test.xlsx"))

    df = df_test
    for index, row in tqdm(df.iterrows()):
        imagePath = Path(os.path.join(img_dir, row["image_name"]))
        maskPath = os.path.join(mask_dir, row["image_name"])
        label = row[["NC", "G3", "G4", "G5"]].values.astype(int)
        label_name = ["NC", "G3", "G4", "G5"][np.argmax(label)]
        label_text = prompts[label_name][0]
        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        if mask.sum() == 0:
            continue
        mask_resize = mask
        mask_resize[mask_resize > 0] = 1
        data["path"].append(imagePath)
        data["gtmasks"].append(mask_resize)
        data["label_text"].append(label_text)
        data["boxes"].append(None)
        data["category"].append(label_name)
    return data