
import os
import sys
import torch
import matplotlib
import argparse
import numpy as np
from pathlib import Path

sys.path.append(os.getcwd())
from localization.common import Pipeline, ImageTextInferenceEngine
FONT_MAX = 50
matplotlib.use('Agg')


class Engine(ImageTextInferenceEngine):
    """Engine for inference."""

    def __init__(self) -> None:
        super().__init__()

    def load_model(self, **kwargs):
        """Load the model."""
        import afloc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = afloc.builder.load_model(ckpt_path=kwargs["ckpt"], device=device)
        self.model.eval()
        self.image_inference_engine = self.model.image_encoder_forward
        self.text_inference_engine = self.model.text_encoder_forward
        self.model.cfg.data.image.imsize = 224
        self.flag = self.model.cfg.data.image.flag
        
    def get_img_emb(self, image_path: Path, device):
        """
        Get image embeddings.
        Inputs:
            - image_path (Path): path to image
            - device (torch.device): device
        Returns:
            - emb (dict): embeddings
            - iel (torch.Tensor): image embeddings local (h, w, feature_size)
            - ieg (torch.Tensor): image embeddings global (1, feature_size)
        """
        pi = self.model.process_img([image_path], device, equalize_hist=False, flag=self.flag)
        with torch.no_grad():
            iel, _, _, ieg = self.image_inference_engine(pi.to(device))
        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)
        emb = {"iel": iel, "ieg": ieg}
        return emb
    
    def get_text_emb(self, query_text: str, device):
        """
        Get text embeddings.
        Inputs:
            - query_text (str): query text
            - device (torch.device): device
        Returns:
            - emb (dict): embeddings
                - tel (torch.Tensor): text embeddings local (word_num, feature_size)
                - teg (torch.Tensor): text embeddings global (1, feature_size)
        """
        pt = self.model.process_text(query_text, device)
        with torch.no_grad():
            res = self.text_inference_engine(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device))
        tel, teg = res["word_embeddings"], res["report_embeddings"]
        emb = {"tel": tel, "teg": teg}
        return emb


def main(**kwargs):
    engine = Engine()
    pipeline = Pipeline(engine, **kwargs)
    pipeline.run(**kwargs)
    return True


if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="weight/Pretrained_Path.ckpt")
    parser.add_argument("--dataset", "-ds", type=str, default="CHEXLOCALIZE",
                        choices=["MS_CXR", "MS_CXR_CLS", "RSNA_GROUNDING", "COVID_RURAL", "CHEXLOCALIZE",
                                 "IRH", "SICAPV2_GROUNDING"])
    parser.add_argument("--redo", "-r", action='store_true', default=True)
    parser.add_argument('--gpu', type=str, default='6', help='gpu')
    parser.add_argument('--opt_th', action='store_true', default=False)
    parser.add_argument('--only_pos', action='store_true', default=False)
    parser.add_argument('--eval_val_or_test', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--use_prob', action='store_true', default=False)
    parser.add_argument('--margin', action='store_true', default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    res = main(**vars(args))



