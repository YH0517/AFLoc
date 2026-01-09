
# localization
CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_CXR.ckpt --margin -ds MS_CXR --gpu 3
CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_CXR.ckpt --margin -ds MS_CXR_CLS --gpu 3 
CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_CXR.ckpt --margin -ds RSNA_GROUNDING --gpu 3
CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_CXR.ckpt --margin -ds COVID_RURAL --gpu 3
CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_CXR.ckpt --margin -ds CHEXLOCALIZE --gpu 3 --opt_th --only_pos

CUDA_VISIBLE_DEVICES=0 python localization.py --ckpt weight/Pretrained_Path.ckpt -ds SICAPV2_GROUNDING --gpu 3

# classification
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_CXR.ckpt --dataset RSNA --multi_class --modality "Chest X-ray" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_CXR.ckpt --dataset NIH --multi_class --modality "Chest X-ray" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_CXR.ckpt --dataset SIIM --multi_class --modality "Chest X-ray" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_CXR.ckpt --dataset CXR-LT-T2 --multi_class --modality "Chest X-ray" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_CXR.ckpt --dataset CXR-LT-T3 --multi_class --modality "Chest X-ray" --redo true 

CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_Path.ckpt --dataset SICAPV2 --modality "Pathology" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_Path.ckpt --dataset Wsss4luad3 --modality "Pathology" --redo true 
CUDA_VISIBLE_DEVICES=0 python classification.py --ckpt weight/Pretrained_Path.ckpt --dataset DHMCLUAD --modality "Pathology" --redo true 
