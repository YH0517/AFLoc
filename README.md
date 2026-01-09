# AFLoc

This repository provides the official implementation of *[A multimodal vision–language model for generalizable annotation-free pathology localization](https://www.nature.com/articles/s41551-025-01574-7)*

## Key Features

- A generalizable vision-language pre-training model for **annotation-free pathology localization**.
- We use a multi-level semantic structure-based contrastive learning to aligns multi-granularity medical concepts across reports and images.
- Demonstrates strong generalizability to mulitiple modalities including chest X-rays, histopathology and retinal fundus images.

## Details

Existing deep learning models for defining pathology from clinical imaging data rely on expert annotations and lack generalization capabilities in open clinical environments. Here we present a generalizable vision–language model for Annotation-Free pathology Localization (AFLoc). The core strength of AFLoc is extensive multilevel semantic structure-based contrastive learning, which comprehensively aligns multigranularity medical concepts with abundant image features to adapt to the diverse expressions of pathologies without the reliance on expert image annotations. We conducted primary experiments on a dataset of 220,000 pairs of image–report chest X-ray images and performed validation across 8 external datasets encompassing 34 types of chest pathology. The results demonstrate that AFLoc outperforms state-of-the-art methods in both annotation-free localization and classification tasks. In addition, we assessed the generalizability of AFLoc on other modalities, including histopathology and retinal fundus images. We show that AFLoc exhibits robust generalization capabilities, even surpassing human benchmarks in localizing five different types of pathological image. These results highlight the potential of AFLoc in reducing annotation requirements and its applicability in complex clinical environments.

<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/fig1.png"></a>
</div>

 **Quantatitive pathological lesions localization results**

<img src="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/results_cxr.jpg" width="50%" />

**Visualizations of pathological lesions localization**

<img src="https://"><img width="1000px" height="auto" src="https://github.com/YH0517/AFLoc/blob/master/assets/viz_cxr.png" width="50%" />

## Get started

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

Update the directory to your own within `classification/constants.py` and `localization/constants.py`. Then you can inference AFLoc with following command:

```shell
bash run.sh 
```

## Feedback and Contact

For further questions with the codes, please feel free to contact [Hao Yang](h.yang1@siat.ac.cn)

## License

This project is under the Apache License 2.0 license. See [LICENSE](https://github.com/YH0517/AFLoc/blob/master/LICENSE) for details.

## Acknowledgement

Some codes are reference from [GLoRIA](https://github.com/marshuang80/gloria), [BioViL](https://github.com/microsoft/hi-ml), and [cheXlocalize](https://github.com/rajpurkarlab/cheXlocalize). We thank the authors for making their valuable code & data publicly available.

## Citation

If you find this repository useful, please consider citing this paper:

```
@article{yang2026multimodal,
  title={A multimodal vision--language model for generalizable annotation-free pathology localization},
  author={Yang, Hao and Zhou, Hong-Yu and Liu, Jiarun and Huang, Weijian and Li, Cheng and Li, Zhihuan and Gao, Yuanxu and Liu, Qiegen and Liang, Yong and Yang, Qi and others},
  journal={Nature Biomedical Engineering},
  pages={1--15},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```
