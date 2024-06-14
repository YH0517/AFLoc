import numpy as np
from eval.utils import norm_heatmap


def compute_iou(gtmask_, premask_, nan, only_pos=True):
    """
    Compute IOU between ground truth mask and predicted mask.

    Inputs:
        - gtmask_ (np.ndarray): shape=(H, W), ground truth mask
        - premask_ (np.ndarray): shape=(H, W), predicted mask
        - nan (np.ndarray): shape=(H, W), True if the pixel is NaN
        - only_pos (bool): if True, compute IOU only for positive pixels

    Returns:
        - iou_score (float): IOU score
    """
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    union = np.logical_or(gtmask, premask)
    if only_pos:
        if np.sum(premask) == 0 or np.sum(gtmask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(union) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    return iou_score


def compute_cnr(gtmask_, heatmap_, nan):
    """
    Compute contrast-to-noise ratio (CNR) between ground truth mask and heatmap.
    For CNR, let A and A_ denote the interior and exterior of the bounding box, respectively.
    CNR = |meanA - meanA_| / pow((varA_ + varA), 0.5)
    
    Inputs:
        - gtmask_ (np.ndarray): shape=(H, W), ground truth mask
        - heatmap_ (np.ndarray): shape=(H, W), heatmap
        - nan (np.ndarray): shape=(H, W), True if the pixel is NaN

    Returns:
        - CNR (float): contrast-to-noise ratio
    """
    heatmap = norm_heatmap(heatmap_, nan)
    heatmap_wo_nan = heatmap[~nan]
    gtmask_wo_nan = gtmask_[~nan]
    # assert (gtmask_wo_nan == 1).sum() > 0, 'gtmask_wo_nan == 1 is empty'
    A = heatmap_wo_nan[gtmask_wo_nan == 1]
    A_ = heatmap_wo_nan[gtmask_wo_nan == 0]
    meanA = A.mean()
    meanA_ = A_.mean()
    varA = A.var()
    varA_ = A_.var()
    if varA + varA_ == 0:
        CNR = 0
    else:
        CNR = (meanA - meanA_) / pow((varA + varA_), 0.5)
    return CNR
