
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd 
import matplotlib
import torch.nn.functional as F
from math import ceil, floor
from pathlib import Path
from typing import Callable, Optional
from scipy import ndimage
sys.path.append(os.getcwd())
from eval.utils import load_data, norm_heatmap
from eval.metric import compute_iou, compute_cnr
FONT_MAX = 50
matplotlib.use('Agg')
from functools import reduce

class ImageTextInferenceEngine:

    def __init__(self) -> None:
        pass

    def load_model(self, **kwargs):

        raise NotImplementedError

    def get_img_emb(self, image_path: Path, device):
        raise NotImplementedError

    def get_text_emb(self, query_text: str, device):
        raise NotImplementedError


    def get_similarity_map_from_raw_data(
        self, image_path: Path, query_text: str, device, interpolation: str = "nearest",
        ) -> np.ndarray:
        img_emb = self.get_img_emb(image_path, device)
        text_emb = self.get_text_emb(query_text, device)
        iel, teg = img_emb["iel"], text_emb["teg"]
        sim = self._get_similarity_map_from_embeddings(iel, teg)
        resized_sim_map = self.convert_similarity_to_image_size(
            sim,
            width=224,
            height=224,
            resize_size=224,
            crop_size=224,
            interpolation=interpolation,
        )

        return resized_sim_map, img_emb, text_emb
    
    @staticmethod
    def set_margin(similarity_map, width=224, height=224, resize_size=512, crop_size=448):
        smallest_dimension = min(height, width)
        cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
        target_size = cropped_size_orig_space, cropped_size_orig_space
        margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
        margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
        mask = torch.zeros(target_size)
        mask = F.pad(mask, margins_for_pad, value=float("NaN"))
        nan = torch.isnan(mask)
        similarity_map[nan] = float("NaN")
        return similarity_map

    @staticmethod
    def _get_similarity_from_embeddings(
        projected_image_embeddings: torch.Tensor, projected_text_embeddings: torch.Tensor, sigma: float = 1.5
    ) -> torch.Tensor:
        """
        :param projected_image_embeddings: [1, feature_size]
        :param projected_text_embeddings: [cls_num, feature_size]
        :return: similarity: similarity of shape [1, cls_num]
        """
        img_norm = projected_image_embeddings / projected_image_embeddings.norm(dim=-1, keepdim=True)
        text_norm = projected_text_embeddings / projected_text_embeddings.norm(dim=-1, keepdim=True)
        similarity = img_norm @ text_norm.t()

        # similarity = projected_image_embeddings @ projected_text_embeddings.t()

        return similarity

    @staticmethod
    def _get_similarity_map_from_embeddings(
        projected_patch_embeddings: torch.Tensor, projected_text_embeddings: torch.Tensor, sigma: float = 1.5
    ) -> torch.Tensor:
        """Get smoothed similarity map for a given image patch embeddings and text embeddings.

        :param projected_patch_embeddings: [n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [1, feature_size]
        :return: similarity_map: similarity map of shape [n_patches_h, n_patches_w]
        """
        n_patches_h, n_patches_w, feature_size = projected_patch_embeddings.shape
        assert feature_size == projected_text_embeddings.shape[1]
        assert projected_text_embeddings.shape[0] == 1
        assert projected_text_embeddings.dim() == 2
        patch_wise_similarity = projected_patch_embeddings.view(-1, feature_size) @ projected_text_embeddings.t()
        patch_wise_similarity = patch_wise_similarity.reshape(n_patches_h, n_patches_w).cpu().numpy()
        smoothed_similarity_map = torch.tensor(
            ndimage.gaussian_filter(patch_wise_similarity, sigma=(sigma, sigma), order=0)
        )
        return smoothed_similarity_map
    
    @staticmethod
    def convert_similarity_to_image_size(
        similarity_map: torch.Tensor,
        width: int,
        height: int,
        resize_size: Optional[int],
        crop_size: Optional[int],
        interpolation: str = "nearest",
    ) -> np.ndarray:
        """
        Convert similarity map from raw patch grid to original image size,
        taking into account whether the image has been resized and/or cropped prior to entering the network.
        """
        n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
        target_shape = 1, 1, n_patches_h, n_patches_w
        smallest_dimension = min(height, width)

        reshaped_similarity = similarity_map.reshape(target_shape)
        align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
        align_corners = False if interpolation in align_corners_modes else None

        if crop_size is not None:
            if resize_size is not None:
                cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
                target_size = cropped_size_orig_space, cropped_size_orig_space
            else:
                target_size = crop_size, crop_size
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=target_size,
                mode=interpolation,
                align_corners=align_corners,
            )
            margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
            margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
            similarity_map = F.pad(similarity_map[0, 0], margins_for_pad, value=float("NaN"))
        else:
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=(height, width),
                mode=interpolation,
                align_corners=align_corners,
            )[0, 0]
        return similarity_map.numpy()


class Pipeline:

    def __init__(self, inference: Callable, **kwargs):
        self.image_text_inference = inference
        self.save_dir = None
        self.kwargs = kwargs
    
    def run(self, **kwargs):
        self.kwargs.update(kwargs)
        self.createdir(kwargs["ckpt"], kwargs["dataset"])
        data = load_data(**kwargs)
        hmaps = self.get_hmaps(data["path"], data["label_text"], data["category"], **kwargs)
        self.test(path_list=data["path"],
                label_text=data["label_text"],
                gtmasks=data["gtmasks"],
                boxes=data["boxes"],
                category=data["category"],
                hmaps=hmaps,
                **kwargs) 

    def createdir(self, ckpt: str, dataset: str):
        if os.path.exists(ckpt):
            dn = os.path.join(os.path.dirname(ckpt), dataset)
            bn = os.path.splitext(os.path.basename(ckpt))[0]
            self.save_dir = os.path.join(dn, bn)
        else:
            self.save_dir = os.path.join(os.getcwd(), "result", dataset)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_hmaps(self, path_list: list, label_text: list, category, redo=False, **kwargs):
        dataset = kwargs["dataset"]
        suffix = kwargs["suffix"] if "suffix" in kwargs else ""
        save_path = os.path.join(self.save_dir, f"hmaps{suffix}.npy")
        if os.path.exists(save_path) and redo == False:
            hmaps = np.load(save_path, allow_pickle=True).item()
        else:
            hmaps = {}
            self.image_text_inference.load_model(**kwargs)
            for i in tqdm(range(len(path_list[:]))):
                hmap, img_emb, text_emb = self.image_text_inference.get_similarity_map_from_raw_data(
                    image_path=path_list[i],
                    query_text=label_text[i],
                    device="cuda",
                    interpolation="bilinear",
                )
                key = str(path_list[i]) + label_text[i]
                hmaps[key] = {"hmap": hmap}

            np.save(save_path, hmaps)
        return hmaps
    
    @staticmethod
    def set_margin(similarity_map, width=224, height=224, resize_size=512, crop_size=448):
        smallest_dimension = min(height, width)
        cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
        target_size = cropped_size_orig_space, cropped_size_orig_space
        margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
        margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
        mask = torch.zeros(target_size)
        mask = F.pad(mask, margins_for_pad, value=float("NaN"))
        nan = torch.isnan(mask)
        similarity_map[nan] = float("NaN")
        return similarity_map

    def test(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, **kwargs):
        res = pd.DataFrame()
        iou_thre_cat = []
        cnr_thre_cat = []
        sep_cat = {}
        metric_df_thre_iou = []
        metric_df_thre_cnr = []

        threshold_list = list(np.arange(0.1, 0.6, 0.1))
        for threshold in threshold_list:
            ious = []
            cnrs = []
            cat_ious = {}
            cat_cnrs = {}

            for i in tqdm(range(len(path_list[:]))):
                key = str(path_list[i]) + label_text[i]
                hmap = hmaps[key]["hmap"]
                if self.kwargs["margin"]:
                    hmap = self.set_margin(hmap)
                    
                nan = np.isnan(hmap)
                heatmap = norm_heatmap(hmap, nan, mode=0) # [-1, 1]
                gtmask = gtmasks[i]

                mask = np.where(heatmap > threshold, 1, 0)
                iou = compute_iou(gtmask, mask, nan)
                ious.append(iou)
                
                cnr = compute_cnr(gtmask, heatmap, nan)
                cnrs.append(cnr)

                cat = category[i]
                if cat not in cat_ious:
                    cat_ious[cat] = []
                    cat_cnrs[cat] = []
                    
                cat_ious[cat].append(iou)
                cat_cnrs[cat].append(cnr)

            metric_df_iou = to_metric_df(cat_ious)
            metric_df_thre_iou.append(metric_df_iou)
            metric_df_cnr = to_metric_df(cat_cnrs)
            metric_df_thre_cnr.append(metric_df_cnr)

            iou_thre_cat.append(dict_mean(cat_ious))
            cnr_thre_cat.append(dict_mean(cat_cnrs))

            if "iou" not in sep_cat:
                sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
                sep_cat["cnr"] = pd.DataFrame(dict_mean(cat_cnrs, sep=True))
            else:
                sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
                sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)

        total_df_iou = reduce(lambda x, y: x.add(y, fill_value=0), metric_df_thre_iou)
        total_df_cnr = reduce(lambda x, y: x.add(y, fill_value=0), metric_df_thre_cnr)
        average_df_iou = total_df_iou / len(metric_df_thre_iou)
        average_df_cnr = total_df_cnr / len(metric_df_thre_cnr)
        ci_df_iou = bootci(average_df_iou, self.save_dir, "iou")
        ci_df_cnr = bootci(average_df_cnr, self.save_dir, "cnr")

        res["threshold"] = threshold_list + ["mean"]
        res["iou_cat"] = iou_thre_cat + [np.mean(iou_thre_cat)]
        res["cnr_cat"] = cnr_thre_cat + [np.mean(cnr_thre_cat)]

        sep_cat["iou"].rename(columns=prefix("iou_", sep_cat["iou"].columns), inplace=True)
        sep_cat["cnr"].rename(columns=prefix("cnr_", sep_cat["cnr"].columns), inplace=True)

        # sep_cat add a row: mean
        sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(sep_cat["iou"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(sep_cat["cnr"].mean(axis=0)).T], axis=0, ignore_index=True)

        # concat
        for k, v in sep_cat.items():
            res = pd.concat([res, v], axis=1)

        res = res.round(3)
        res.to_csv(f"{self.save_dir}/metric.csv", index=False)
        print(res.loc[:, ["threshold", "iou_cat", "cnr_cat", ]])


def dict_mean(d: dict, sep=False):
    if sep:
        res = {}
        for k, v in d.items():
            res[k] = np.nanmean(v, keepdims=True)
        return res
    res = []
    for k, v in d.items():
        res.append(np.nanmean(v))
        # print(k, len(v))
    return np.nanmean(res)


def prefix(prefix, l):
    return {i: prefix + i for i in l}


def to_metric_df(cat_matric):
    res = pd.DataFrame()
    columns = cat_matric.keys()
    row_n = sum([len(list(i)) for i in cat_matric.values()])
    data = np.zeros((row_n, len(columns)))
    data[data==0] = np.nan
    res = pd.DataFrame(data=data, columns=columns)
    i = 0
    for k, v in cat_matric.items():
        for j in range(len(v)):
            res[k][i] = v[j]
            i += 1
    return res


def bootstrap_metric(df, num_replicates):
    """Create dataframe of bootstrap samples."""
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df.columns:
            performance = df_replicate[task].mean()
            replicate_performances[task] = performance
        return replicate_performances

    all_performances = []
    for _ in range(num_replicates):
        replicate_performances = single_replicate_performances()
        all_performances.append(replicate_performances)

    df_performances = pd.DataFrame.from_records(all_performances)
    return df_performances


def compute_cis(series, confidence_level):
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(),3)
    return lower, mean, upper


def create_ci_record(perfs, task):
    lower, mean, upper = compute_cis(perfs, confidence_level = 0.05)
    record = {"name": task,
              "lower": lower,
              "mean": mean,
              "upper": upper}
    return record

def bootci(df, save_dir, metric="iou"):
    bs_df = bootstrap_metric(df, 1000)
    bs_df.to_csv(f'{save_dir}/{metric}_bootstrap_results.csv', index=False)
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))
    ci_df = pd.DataFrame.from_records(records).sort_values(by='name')
    mean = ci_df.mean(axis=0)
    ci_df = ci_df.append(mean, ignore_index=True)
    ci_df = ci_df.round(3)
    ci_df.to_csv(f'{save_dir}/test_{metric}_summary_results.csv', index=False)
    return ci_df