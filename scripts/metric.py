import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, accuracy_score
import torch
import cv2
warnings.filterwarnings("ignore")

def search_deep_thresholds(eval_list, thrs_list, n_search_workers):
    best_score = 0.
    best_thr = 0.

    progress_bar = tqdm(thrs_list)

    for thr in progress_bar:
        score_list = Parallel(n_jobs=n_search_workers)(delayed(apply_deep_thresholds)(
            probas, labels, thr) for probas, labels in eval_list)
        final_score = np.mean(score_list)
        if final_score > best_score:
            best_score = final_score
            best_thr = thr
        progress_bar.set_description('Best score: {:.4}'.format(best_score))
    return best_score, best_thr


def apply_deep_thresholds(predicted, ground_truth, threshold=0.5):
    classification_mask = predicted > threshold
    mask = torch.where(classification_mask, torch.ones_like(predicted), torch.zeros_like(predicted))
    # return f1_score(ground_truth, mask, average='micro')
    return accuracy_score(ground_truth, mask)