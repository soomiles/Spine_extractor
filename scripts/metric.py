import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, accuracy_score
import torch
import cv2
import pdb
warnings.filterwarnings("ignore")

def search_deep_thresholds(eval_list, thrs_list, n_search_workers):
    best_score = 0.
    best_thr = 0.

    progress_bar = tqdm(thrs_list)
    pdb.set_trace()
    for thr in progress_bar:
        score_list = Parallel(n_jobs=n_search_workers)(delayed(apply_deep_thresholds)(
            probas, labels, thr) for probas, labels in eval_list)
        final_score = np.mean(score_list)
        if final_score > best_score:
            best_score = final_score
            best_thr = thr
        progress_bar.set_description('Best score: {:.4}'.format(best_score))
    return best_score, best_thr


def apply_deep_thresholds(predicted, ground_truth):
    return predicted.eq(ground_truth).sum().item() / len(predicted)