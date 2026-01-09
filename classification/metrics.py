
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from functools import partial
from sklearn.utils import resample 
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def compute_acc(pred: np.ndarray, label: np.ndarray) -> float:
    """ Accuracy for single label classification.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (float) average accuracy 
    
    Code was given by Weijian Huang.
    """
    if pred.shape[-1] == 1:
        return np.mean((pred > 0.5).astype(np.long) == label.astype(np.long))
    return np.mean(pred.argmax(1) == label.argmax(1))


def compute_auc(pred: np.ndarray, label: np.ndarray) -> float:
    """ ROCAUC score for classification task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (float) averaged roc auc score

    Code was given by Weijian Huang.
    """
    try:
        score = roc_auc_score(label, pred)
    except ValueError:
        print("ValueError: ", label.shape, pred.shape)
        score = np.nan
    return score


def compute_mccs_threshold(pred, gt, threshold, n_class=14):
    gt_np = gt 
    pred_np = pred 
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mccs = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mccs)
    return mccs


def compute_overlegnth_mccs(pred, gt, n_class=14):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred.copy()
    select_best_thresholds =[]

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        # generate 1000 possible thresholds 
        prob_i = pred_np.copy()
        prob_i = prob_i[:,i]
        thresholds_range = np.sort(np.unique(prob_i))
        thresholds_range = np.linspace(thresholds_range.min(), thresholds_range.max(), 1000)
        for threshold_idx in range(len(thresholds_range)):
            pred_np_ = pred_np.copy()
            thresholds = thresholds_range[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds
        select_best_thresholds.append(select_best_threshold_i)

    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs, select_best_thresholds, pred_np


def compute_mccs(pred, gt, n_class=14):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred.copy()
    select_best_thresholds =[]

    if pred.shape[0] > 1000:
        print("Computing mccs with fewer thresholds...")
        return compute_overlegnth_mccs(pred, gt, n_class)

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds[i]]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds[i]]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)

    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs, select_best_thresholds, pred_np


def compute_single_classification_metrics(
        pred: np.ndarray, 
        label: np.ndarray,
    ) -> Dict[str,float]:
    """ Evaluation metrics for single classification task.
    Input:
        - pred (np.ndarray): prediction (after softmax) of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (Dict[str,float]): auc
    """
    auc_score = np.mean(compute_auc(pred, label))

    return dict(auc=auc_score)


def compute_multi_classification_metrics(
        pred: np.ndarray, 
        label: np.ndarray,
        categories: List[str],
        compute_avg=False,
        best_thresholds=None,
        softmax=False,
    ) -> Dict[str,Dict[str,float]]:
    """ Evaluation metrics for single classification task.
    Input:
        - pred (np.ndarray): prediction (after sigmoid) of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
        - categories (List[str]): name of each class, to mark 
    Return:
        - (Dict[str,Dict[str,float]]): auc, ba
    """

    if softmax:
        pred_ = np.argmax(pred, axis=1)
        label_ = np.argmax(label, axis=1)
        ba_score = balanced_accuracy_score(label_, pred_)
        return dict(
        auc=0, 
        best_thresholds_dict=0,
        best_thresholds=0,
        ba=ba_score,
        )

    auc_score = {}
    for i in range(len(categories)):
        auc_score[categories[i]] = compute_auc(pred[:,i], label[:,i])

    # compute mcc 
    if best_thresholds is None:
        mccs, best_thresholds, pred_th = compute_mccs(pred, label, n_class=len(categories))
    else:
        assert len(best_thresholds) == len(categories)

    best_thresholds_dict = {}
    ba_score = {}
    for i, cat in enumerate(categories):
        best_thresholds_dict[cat] = best_thresholds[i]
        ba_score[cat] = 0

    if compute_avg:
        auc_score['avg'] = np.mean(list(auc_score.values()))
        ba_score['avg'] = np.mean(list(ba_score.values()))

    return dict(
        auc=auc_score, 
        best_thresholds_dict=best_thresholds_dict,
        best_thresholds=best_thresholds,
        ba=ba_score,
    )


def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df


def bootstrap(y_pred, y_true, eval_func, n_samples=1000): 
    '''
    Args:
        * y_pred - np.array, shape (n_samples, n_total_labels)
        * y_true - np.array, shape (n_samples, n_cxr_labels)
        * eval_func - function, evaluation function to be used
        * n_samples - int, number of samples to use
    '''
    np.random.seed(97)

    idx = np.arange(len(y_true))

    # do one testing to get the best thresholds
    sample_stats = eval_func(y_pred, y_true)
    best_thresholds = sample_stats['best_thresholds']
    
    boot_stats_auc = []
    boot_stats_ba = []
    for i in tqdm(range(n_samples), desc='Bootstrapping'): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = eval_func(y_pred_sample, y_true_sample, best_thresholds=best_thresholds)
        boot_stats_auc.append(sample_stats['auc'])
        boot_stats_ba.append(sample_stats['ba'])
    
    boot_stats_auc = pd.DataFrame(boot_stats_auc)
    boot_stats_ba = pd.DataFrame(boot_stats_ba)

    boot_stats_all = dict(
        auc=boot_stats_auc,
        ba=boot_stats_ba
    )
    ci_all = dict(
        auc=compute_cis(boot_stats_auc),
        ba=compute_cis(boot_stats_ba),
    )
    return boot_stats_all, ci_all


def compute_multi_classification_metrics_bootstrap(
        pred: np.ndarray, 
        label: np.ndarray,
        categories: List[str],
        softmax=False,
    ) -> Dict[str,Dict[str,float]]:
    """ Evaluation metrics for single classification task.
    Input:
        - pred (np.ndarray): prediction (after sigmoid) of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
        - categories (List[str]): name of each class, to mark 
    Return:
        - (Dict[str,Dict[str,float]]): auc, ba
    """

    boot_stats, cis = bootstrap(
        pred, label, 
        partial(compute_multi_classification_metrics, categories=categories, compute_avg=True, softmax=softmax),
    )

    return dict(
        boot_stats=boot_stats,
        cis=cis,
    )
