# AFLoc

This repository provides the official implementation of *[Multi-modal vision-language model for generalizable annotation-free pathology localization and clinical diagnosis](https://arxiv.org/abs/2401.02044)*

## Key Features

- A generalizable vision-language pre-training model for **annotation-free pathology localization**.
- We use a multi-level semantic structure-based contrastive learning to aligns multi-granularity medical concepts across reports and images.
- Demonstrates strong generalizability to mulitiple modalities including chest X-rays and retinal fundus images.

## Details

Annotation-Free pathology Localization (AFLoc). The core strength of AFLoc lies in its extensive multi-level semantic structure-based contrastive learning, which comprehensively aligns multi-granularity medical concepts from reports with abundant image features, to adapt to the diverse expressions of pathologies and unseen pathologies without the reliance on image annotations from experts. We demonstrate the proof of concept on Chest X-ray images, with extensive experimental validation across 6 distinct external datasets, encompassing 13 types of chest pathologies. The results demonstrate that AFLoc surpasses state-of-the-art methods in pathology localization and classification, and even outperforms the human benchmark in locating 5 different pathologies. Additionally, we further verify its generalization ability by applying it to retinal fundus images. Our approach showcases AFLoc's versatilities and underscores its suitability for clinical diagnosis in complex clinical environments.

<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/fig1.png"></a>
</div>

 **Quantatitive pathological lesions localization results**

<img src="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/results_cxr.jpg" width="50%" />

**Visualizations of pathological lesions localization**

<img src="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/viz_cxr.png" width="50%" />

## Links to download datasets

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d)
- [COVID Rural](https://www.cancerimagingarchive.net/collection/covid-19-ar/)
- [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
- [MS-CXR](https://aka.ms/ms-cxr)

## Get started

**Main requirements**

Our experiments are based on a server with two NVIDIA A100 80GB GPU, 512G memory, and Intel(R) Xeon(R) Gold 6326 CPU. Here are some core python packages:

> torch==1.8.0
>
> pytorch-lightning==1.1.4
>
> transformers==4.2.1
>
> torchvision==0.9.0
>
> omegaconf==2.0.5
>
> pydicom
>
> numpy
>
> pycocotools

**Installation**

```shell
# create a new conda environment
conda create -n AFLoc python=3.9
conda activate AFLoc

# install torch and pytorch-lightning
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.1.4

# install other packages
pip install -r requirements.txt
```

**Download pretraind model & preprocessing files**

You can download our pretrained model and preprocessing files from [this link](https://drive.google.com/drive/folders/1RQktI5NN-vd1-xVnt3DDPI9hl3eUxzpq).

**Preprocessing**

```shell
python preprocess/resize.py
python preprocess/preprocess.py
```

**Pretraining**

Update the directory to your own  within `afloc/constants.py`. Then training AFLoc with following command:

```shell
python train.py -c ./afloc/config.yaml --train
```

**Inference**

Update the directory to your own within `eval/constants.py`. Then you can inference AFLoc on MS-CXR with following command:

```shell
python inference.py -ds MS_CXR --gpu 0 
```

## Feedback and Contact

For further questions with the codes, please feel free to contact [Hao Yang](h.yang1@siat.ac.cn)

## License

This project is under the Apache License 2.0 license. See [LICENSE](https://github.com/YH0517/AFLoc/blob/master/LICENSE) for details.

## Acknowledgement

Some codes are reference from [GLoRIA](https://github.com/marshuang80/gloria), [BioViL](https://github.com/microsoft/hi-ml), and [cheXlocalize]https://github.com/rajpurkarlab/cheXlocalize. We thank the authors for making their valuable code & data publicly available.

## Citation

If you find this repository useful, please consider citing this paper:

```
@article{afloc,
    title={Multi-modal vision-language model for generalizable annotation-free pathology localization and clinical diagnosis},
    author={Hao Yang, Hong-Yu Zhou, Zhihuan Li, Yuanxu Gao, Cheng Li, Weijian Huang, Jiarun Liu, Hairong Zheng, Kang Zhang, and Shanshan Wang},
    journal={arXiv preprint arXiv:2401.02044},
    year={2024}
}
```
