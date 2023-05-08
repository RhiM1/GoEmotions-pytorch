import os
import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds, threshold = None, mode = None):
    assert len(preds) == len(labels)
    results = dict()

    addum = "" if mode is None else "_" + mode
    addum = addum if threshold is None else addum + "_" + str(threshold)

    results["accuracy" + addum] = accuracy_score(labels, preds)
    results["macro_precision" + addum], results["macro_recall" + addum], results[
        "macro_f1" + addum], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision" + addum], results["micro_recall" + addum], results[
        "micro_f1" + addum], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision" + addum], results["weighted_recall" + addum], results[
        "weighted_f1" + addum], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results
