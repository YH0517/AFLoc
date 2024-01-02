# AFLoc

Generalizable vision-language pre-training for annotation-free pathology localization

Some code is borrowed from [GLoRIA](https://github.com/marshuang80/gloria) and [BioViL](https://github.com/microsoft/hi-ml).

## Environmental preparation

```
conda create -n AFLoc python=3.9
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda activate AFLoc
pip install -r requirements.txt
```

## Links to download datasets

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/abfb76e5-70d5-4315-badc-c94dd82e3d6d)
- [COVID Rural](https://www.cancerimagingarchive.net/collection/covid-19-ar/)
- [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
- [MS-CXR](https://aka.ms/ms-cxr)

## Pretraining

Adjust the necessary paths and perform the following code:

```
python train.py -c ./afloc/config.yaml --train
```

## Inference

Download the [pre-trained weight](https://drive.google.com/drive/folders/1RQktI5NN-vd1-xVnt3DDPI9hl3eUxzpq) and place it in the ./pretrained folder

We use MS-CXR as an example:

```
python inference.py -ds MS_CXR --gpu 0 
```
