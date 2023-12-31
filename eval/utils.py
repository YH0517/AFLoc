import os
import io
import cv2
import copy
import json
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Union
from pathlib import Path
from functools import reduce
FONT_MAX = 50
matplotlib.use('Agg')
from eval.box_transfer import box_transfer, box2mask
from eval.constants import (MIMIC_IMG_DIR, MS_CXR_JSON)
TypeArrayImage = Union[np.ndarray, Image.Image]


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
    else:
        raise NotImplementedError


def load_ms_cxr(use_cxr_text=True, **kwargs):
    print("loading data...")
    
    data = get_annotation(MS_CXR_JSON, use_cxr_text=use_cxr_text)
    data["path"] = list(map(lambda x: MIMIC_IMG_DIR/x.replace("files/", ""), data["path"]))

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


def get_annotation(path_to_json, scale=224, use_cxr_text=True):
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
            tbox = box_transfer(bbox, w, h, scale)
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


def read_from_dicom(img_path):

    dcm = pydicom.read_file(img_path, force=True)
    x = dcm.pixel_array
    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    img = Image.fromarray(x)
    return img

