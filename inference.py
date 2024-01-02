
import os
import sys
import torch
import matplotlib
import argparse
from pathlib import Path

sys.path.append(os.getcwd())
from eval.common import Pipeline, ImageTextInferenceEngine
FONT_MAX = 50
matplotlib.use('Agg')


class Engine(ImageTextInferenceEngine):

    def __init__(self) -> None:
        super().__init__()

    def load_model(self, **kwargs):
        import afloc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = afloc.builder.load_model(ckpt_path=kwargs["ckpt"], device=device)
        self.model.eval()
        self.image_inference_engine = self.model.image_encoder_forward
        self.text_inference_engine = self.model.text_encoder_forward
        self.model.cfg.data.image.imsize = 224
        
    def get_img_emb(self, image_path: Path, device):
        '''
        return  iel: [h, w, feature_size]
                ieg: [1, feature_size]
        '''
        pi = self.model.process_img([image_path], device, equalize_hist=False)
        with torch.no_grad():
            iel, _, _, ieg = self.image_inference_engine(pi.to(device))
        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)
        emb = {"iel": iel, "ieg": ieg}
        return emb
    
    def get_text_emb(self, query_text: str, device):
        '''
        return  tel: [word_num, feature_size]
                teg: [1, feature_size]
        '''
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

    ckpt_dir = os.path.abspath(kwargs["ckpt_dir"])
    if not os.path.exists(ckpt_dir):
        return False
    ckpt_list = sorted([os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir) if i.endswith(".ckpt")])
    
    engine = Engine()
    pipeline = Pipeline(engine, margin=True, **kwargs)
    for ckpt in ckpt_list:
        print(ckpt)
        pipeline.run(ckpt=ckpt, **kwargs)
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", "-d", type=str, default="pretrained")
    parser.add_argument("--dataset", "-ds", type=str, default="MS_CXR")
    parser.add_argument("--redo", "-r", action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='5', help='gpu')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ckpt_dir = args.ckpt_dir
    res = main(**vars(args))



