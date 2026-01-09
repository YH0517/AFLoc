import os
import cv2
import numpy as np 
import pandas as pd 
import pydicom as dicom
import torch
import torch.utils.data as data
from PIL import Image
from typing import List, Tuple, Optional
from .constants import (NIH_SPLIT_DIR, NIH_IMAGE_DIR, SIIM_SPLIT_DIR, SIIM_IMAGE_DIR, 
RSNA_SPLIT_DIR, RSNA_IMAGE_DIR, CXR_LT_SPLIT_DIR, CXR_LT_IMAGE_DIR,
DHMCLUAD_DIR, DHMCLUAD_IMAGE_DIR, SICAPV2_SPLIT_DIR, SICAPV2_IMAGE_DIR,
WSSS4LUAD3_IMAGE_DIR)

class NIHChestXray(data.Dataset):
    categories = np.array([
        "atelectasis",
        "cardiomegaly",
        "effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "edema",
        "emphysema",
        "fibrosis",
        "pleural thickening",
        "hernia",
    ])
    
    def __init__(self, transform=None):
        super(NIHChestXray, self)
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        
        test_label_data = "test_list.txt"
        fileDescriptor = open(os.path.join(NIH_SPLIT_DIR, test_label_data), "r")
        
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imagePath = os.path.join(NIH_IMAGE_DIR, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()
        if self.transform != None:
            imageData = self.transform(imageData)

        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class SIIM(data.Dataset):
    categories = ["Pneumothorax",]
    
    def __init__(self, transform=None):
        super(SIIM, self)
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []

        test_label_data = "test_list.txt"
        fileDescriptor = open(os.path.join(SIIM_SPLIT_DIR, test_label_data), "r")
        
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                line = line.strip()
                imagePath = os.path.join(SIIM_IMAGE_DIR, line.split(" ")[0])
                imageLabel = [int(line.split(" ")[1]),]
                assert os.path.isfile(imagePath)
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   

        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None:
            imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, label_list = [], []

        for b in instances:
            image_list.append(b["image"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(image=image_stack, label=label_stack)


class RSNAPneumonia(data.Dataset):
    categories = np.array([
        "Pneumonia"
    ])
    
    def __init__(self, transform=None):
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        test_label_data = "test_list.txt" 
        fileDescriptor = open(os.path.join(RSNA_SPLIT_DIR, test_label_data), "r")
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imagePath = os.path.join(RSNA_IMAGE_DIR, lineItems[0])
                imageLabel = lineItems[1:]
                assert len(imageLabel) == 1
                imageLabel = int(imageLabel[0])
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        image = dicom.dcmread(path).pixel_array
        image = remap_to_uint8(image)
        return Image.fromarray(image).convert("RGB")

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = self.load_image(imagePath)
        imageLabel = torch.tensor(self.listImageLabels[index])
        imageLabel = imageLabel.unsqueeze(0)
        categorie = self.categories[0]
        if self.transform is not None: 
            imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class CXR_LT_T2(data.Dataset):
    categories = np.array([
        'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly',
        'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum',
        'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion',
        'Lung Opacity', 'Mass', 'Normal', 'Nodule', 'Pleural Effusion',
        'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia',
        'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema',
        'Support Devices', 'Tortuous Aorta'
    ])
    
    def __init__(self, transform=None):
        self.transform = transform
        test_label_data = "test_labeled_task2.csv"
        df = pd.read_csv(os.path.join(CXR_LT_SPLIT_DIR, test_label_data))

        def get_labels(row):
            return [int(row[cat]) for cat in self.categories]
        
        df["labels"] = df.apply(get_labels, axis=1)
        self.listImageLabels = df["labels"].tolist()

        df["image_path"] = df["fpath"].apply(lambda x: os.path.join(CXR_LT_IMAGE_DIR, x))
        self.listImagePaths = df["image_path"].tolist()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()
        if self.transform != None: imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(torch.tensor(b["label"]))

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )
    

class CXR_LT_T3(data.Dataset):
    categories = np.array([
        'Bulla', 
        'Cardiomyopathy', 
        'Hilum',
        'Osteopenia',
        'Scoliosis'
    ])
    
    def __init__(self, split="train", transform=None):
        self.transform = transform
        test_label_data = "test_labeled_task3.csv"
        df = pd.read_csv(os.path.join(CXR_LT_SPLIT_DIR, test_label_data))

        def get_labels(row):
            return [int(row[cat]) for cat in self.categories]
        
        df["labels"] = df.apply(get_labels, axis=1)
        self.listImageLabels = df["labels"].tolist()

        df["image_path"] = df["fpath"].apply(lambda x: os.path.join(CXR_LT_IMAGE_DIR, x))
        self.listImagePaths = df["image_path"].tolist()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()

        if self.transform != None: imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(torch.tensor(b["label"]))

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class DHMCLUAD(data.Dataset):
    categories = np.array([
        'solid', 
        'lepidic', 
        'acinar', 
        'micropapillary', 
        'papillary'
    ])

    def __init__(self, transform=None):
        super(DHMCLUAD, self)
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        csv_list = pd.read_csv(os.path.join(DHMCLUAD_DIR, "MetaData_Release_1.0.csv"))

        def make_label(x):
            label = np.zeros(len(self.categories))
            label[self.categories == x] = 1
            return label
        
        self.listImageLabels = csv_list['Class'].apply(make_label).tolist()
        self.listImagePaths = csv_list['File Name'].apply(lambda x: os.path.join(DHMCLUAD_IMAGE_DIR, x)).tolist()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        x = cv2.imread(imagePath)
        imageData = Image.fromarray(x).convert("RGB")
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()
        
        if self.transform != None:
            imageData = self.transform(imageData)

        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class SICAPV2(DHMCLUAD):
    categories = np.array([
        'NC',
        'G3', 
        'G4', 
        'G5']
    )
    def __init__(self, transform=None):
        super(SICAPV2, self)
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        # read xlsx
        df = pd.read_excel(os.path.join(SICAPV2_SPLIT_DIR, "Test.xlsx"))

        for index, row in df.iterrows():
            imagePath = os.path.join(SICAPV2_IMAGE_DIR, row["image_name"])
            label = row[["NC", "G3", "G4", "G5"]].values.astype(int)
            self.listImagePaths.append(imagePath)
            self.listImageLabels.append(label)


class Wsss4luad3(DHMCLUAD):
    categories = np.array([
        'normal',
        'stroma',
        'tumor']
    )
    def __init__(self, transform=None):
        super(Wsss4luad3, self)
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        path_list = [i for i in os.listdir(WSSS4LUAD3_IMAGE_DIR) if i.endswith(".png")]

        path_0 = [os.path.join(WSSS4LUAD3_IMAGE_DIR, i) for i in path_list if "0, 0, 1" in i]
        path_1 = [os.path.join(WSSS4LUAD3_IMAGE_DIR, i) for i in path_list if "0, 1, 0" in i]
        path_2 = [os.path.join(WSSS4LUAD3_IMAGE_DIR, i) for i in path_list if "1, 0, 0" in i]
        
        self.listImagePaths.extend(path_0)
        self.listImagePaths.extend(path_1)
        self.listImagePaths.extend(path_2)

        self.listImageLabels.extend([np.array([1,0,0])]*len(path_0))
        self.listImageLabels.extend([np.array([0,1,0])]*len(path_1))
        self.listImageLabels.extend([np.array([0,0,1])]*len(path_2))


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)
