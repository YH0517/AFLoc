# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from typing import Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from afloc.models.afloc_model import AFLoc


class AFLocCLS(AFLoc):
    def __init__(
            self, 
            cfg,
            embed_dim=768,
            num_classes=5,
            multi_class=True,
        ):
        super().__init__(cfg)

        # classification head
        self.cls_head = ZeroShotClassificationHead(
            input_dim=768,
            num_classes=num_classes,
            multi_class=multi_class,
        )
        print()

    def forward(
            self, 
            imgs: torch.Tensor,
            latent_img: torch.Tensor = None,
        ):
        
        assert self.cls_head is not None, "Please set the classification head first."
        if latent_img is None:
            # image encoder
            img_emb_l, img_emb_l2, img_emb_lf, img_emb_g = self.image_encoder_forward(imgs)
            latent_img = img_emb_g
        else:
            # to device
            latent_img = latent_img.to(next(self.parameters()).device)
        
        
        pred, logits_pos, logits_neg = self.cls_head(latent_img)

        return pred, latent_img, logits_pos, logits_neg

    def set_cls_feature(
            self,
            prompts: Dict[str, Dict[str, List[str]]],
        ):
        pos_cls_embed = torch.zeros(len(prompts), 768)
        neg_cls_embed = torch.zeros(len(prompts), 768)
        
        for i, (key, value) in enumerate(prompts.items()):
            pos_embed = []
            for prompt in value["pos"]:
                # get positive cls feature
                emb = self.get_text_emb(prompt)
                pos_embed.append(emb["teg"])

            neg_embed = []
            for prompt in value["neg"]:
                # get negative cls feature
                if prompt is not None:
                    emb = self.get_text_emb(prompt)
                    neg_embed.append(emb["teg"])

            pos_cls_embed[i] = torch.stack(pos_embed).mean(dim=0)
            neg_cls_embed[i] = torch.stack(neg_embed).mean(dim=0)

        # set cls feature
        self.cls_head.set_cls_feature(pos_cls_embed, neg_cls_embed)

    def get_img_emb(self, image_path):
        '''
        return  iel: [h, w, feature_size]
                ieg: [1, feature_size]
        '''
        device = next(self.parameters()).device
        pi = self.process_img([image_path], device, equalize_hist=False)
        with torch.no_grad():
            iel, iel2, ielf, ieg = self.image_encoder_forward(pi.to(device))
        # iel = iel2
        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)
        emb = {"iel": iel, "ieg": ieg}
        return emb
    
    def get_text_emb(self, query_text):
        '''
        return  tel: [word_num, feature_size]
                teg: [1, feature_size]
                sts: [word_num]
        '''
        device = next(self.parameters()).device
        pt = self.process_text(query_text, device)
        with torch.no_grad():
            res = self.text_encoder_forward(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device))
        tel, teg = res["sent_embeddings"], res["report_embeddings"]
        emb = {"tel": tel, "teg": teg,  "text_mask": pt["attention_mask"]}
        return emb
    

class ZeroShotClassificationHead(nn.Module):
    """
    Zero-shot classification head.
    """
    def __init__(self, input_dim, num_classes=5, multi_class=True) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.multi_class = multi_class

        self.fc = nn.Linear(input_dim, num_classes, bias=False)
        if self.multi_class:
            self.neg_fc = nn.Linear(input_dim, num_classes, bias=False)

    def set_cls_feature(
            self, 
            cls_feature: torch.Tensor,
            neg_cls_feature: torch.Tensor = None,
        ):

        print("Setting positive classification head weights...")
        assert cls_feature.shape == self.fc.weight.shape
        self.fc.weight.data.copy_(cls_feature)
        self.fc.weight.requires_grad = False
        
        # if neg_cls_feature is not None:
        if self.multi_class:
            print("Setting negative classification head weights...")
            assert neg_cls_feature.shape == self.neg_fc.weight.shape
            self.neg_fc.weight.data.copy_(neg_cls_feature)
            self.neg_fc.weight.requires_grad = False
        
        print("Weights have been set.")

    def forward(self, x):
        logits = self.fc(x)  # [N, num_classes]
        
        if self.multi_class:
            neg_logits = self.neg_fc(x)  # [N, num_classes]
            logits = torch.cat([logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=2) # [N, num_classes, 2]
            logits = F.softmax(logits, dim=2)  # [N, num_classes, 2]
            
            return logits, self.fc.weight.data, self.neg_fc.weight.data
        else:
            logits = F.softmax(logits, dim=1)  # [N, num_classes]
            return logits, self.fc.weight.data, None
        


def afloc(ckpt_path,
    num_classes=5,
    multi_class=True,
    bert_type="emilyalsentzer/Bio_ClinicalBERT",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["hyper_parameters"]
    cfg.model.text.bert_type = bert_type
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("afloc.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict
    afloc_model = AFLocCLS(num_classes=num_classes, multi_class=multi_class, cfg=cfg).to(device)

    msg = afloc_model.load_state_dict(ckpt_dict, strict=False)
    print(msg)

    return afloc_model



