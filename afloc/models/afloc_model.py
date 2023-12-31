import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics

from PIL import Image
from .. import builder
from . import losses
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
import os
from skimage import exposure
import torch.nn.functional as F


class AFLoc(nn.Module):
    def __init__(self, cfg):
        super(AFLoc, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)

        self.local_loss = losses.local_loss
        self.global_loss = losses.global_loss

        self.temp1 = self.cfg.model.afloc.temp1
        self.temp2 = self.cfg.model.afloc.temp2
        self.temp3 = self.cfg.model.afloc.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        res = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return res

    def image_encoder_forward(self, imgs):
        img_feat_g, img_feat_l, img_feat_l2, img_feat_lf = self.img_encoder(imgs, get_local=True)  # [b, 2048] [b, 1024, 19, 19]
        img_emb_g, img_emb_l, img_emb_l2, img_emb_lf = self.img_encoder.generate_embeddings(
            img_feat_g, img_feat_l, img_feat_l2, img_feat_lf
        )  

        return img_emb_l, img_emb_l2, img_emb_lf, img_emb_g
    

    def _calc_local_loss(self, img_emb_l, text_emb_w, report, weight_matrix=None):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in report
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_w,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            weight_matrix=weight_matrix,
        )
        return l_loss0, l_loss1, attn_maps
    

    def _calc_global_loss(self, img_emb_g, text_emb_r, weight_matrix=None):
        lgr_0, lgr_1, matrix_cosine = self.global_loss(img_emb_g, text_emb_r, temp3=self.temp3, weight_matrix=weight_matrix)
        return lgr_0, lgr_1, matrix_cosine

    def calc_loss(self, output, batch):
        img_emb_l = output["img_emb_l"]
        img_emb_l2 = output["img_emb_l2"]
        img_emb_g = output["img_emb_g"]
        text_emb_w = output["text_emb_w"]
        text_emb_r = output["text_emb_r"]
        report = output["report"]
        text_emb_s = output["text_emb_s"]
        sent_units_report = output["sent_units_report"]

        if self.cfg.model.afloc.use_global_report_loss:
            lgr_0, lgr_1, matrix_cosine = self._calc_global_loss(img_emb_g, text_emb_r)
        else:
            lgr_0, lgr_1, matrix_cosine = 0, 0, None
        

        if self.cfg.model.afloc.use_local_sent_loss:
            lds_0, lds_1, attn_maps_sent = self._calc_local_loss(
                img_emb_l, text_emb_s, sent_units_report
            )
        else:
            lds_0, lds_1, attn_maps_sent = 0, 0, None


        if self.cfg.model.afloc.use_local_word_loss:
            lsw_0, lsw_1, attn_maps2 = self._calc_local_loss(
                img_emb_l2, text_emb_w, report
            )
        else:
            lsw_0, lsw_1, attn_maps2 = 0, 0, None

        # weighted loss
        lgr = (lgr_0 + lgr_1) * self.cfg.model.afloc.global_report_loss_weight
        lds = (lds_0 + lds_1) * self.cfg.model.afloc.local_sent_loss_weight

        lsw = (lsw_0 + lsw_1) * self.cfg.model.afloc.local_word_loss_weight

        tl = lgr + lds + lsw
        res = {
            "loss": tl,
            "global_report_loss": lgr,
            "local_sent_loss": lds,
            "local_word_loss": lsw,
            "attn_maps_sent": attn_maps_sent,
            "report": report,
            "sent_units_report": sent_units_report,
        }
        return res

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_l2, img_emb_lf, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encorder branch
        res_text = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        ) 
        res = {
            "img_emb_l": img_emb_l,
            "img_emb_l2": img_emb_l2,
            "img_emb_lf": img_emb_lf,
            "img_emb_g": img_emb_g,
            "text_emb_w": res_text["word_embeddings"],
            "text_emb_r": res_text["report_embeddings"], 
            "text_emb_s": res_text["sent_embeddings"],
            "report": res_text["report"],
            "sent_units_report": res_text["sent_units_report"]
            }
        return res

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }


    def process_img(self, paths, device, equalize_hist=False):

        transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:

            x = cv2.imread(str(p), 0)

            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            if equalize_hist:
                x = np.array(x) / 255.
                x = exposure.equalize_hist(x)
            
                x = (255 * x).astype(np.uint8)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)
            all_imgs.append(torch.tensor(img))

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

    def _resize_img(self, img, scale):
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