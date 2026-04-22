import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, multilabel_confusion_matrix,
    hamming_loss, jaccard_score,
)
from sklearn.preprocessing import label_binarize
import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Compute classification metrics.
    
    Args:
        y_true : np.ndarray
                 - Multi-class: (N,) with class indices
                 - Multi-label: (N, n_classes) with binary labels
        y_pred : np.ndarray  predictions (same shape as y_true)
        y_proba : np.ndarray probabilities (N, n_classes), optional
    
    Returns:
        dict: Metrics dictionary
    """
    if config.MULTI_LABEL:
        return _compute_metrics_multilabel(y_true, y_pred, y_proba)
    else:
        return _compute_metrics_multiclass(y_true, y_pred, y_proba)


def _compute_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Metrics for single-label multi-class classification (MIT-BIH)."""
    y_true = y_true.astype(int)
    n      = config.N_CLASSES

    acc              = accuracy_score(y_true, y_pred)
    macro_precision  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_recall     = recall_score   (y_true, y_pred, average='macro',    zero_division=0)
    macro_f1         = f1_score       (y_true, y_pred, average='macro',    zero_division=0)
    w_precision      = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    w_recall         = recall_score   (y_true, y_pred, average='weighted', zero_division=0)
    w_f1             = f1_score       (y_true, y_pred, average='weighted', zero_division=0)

    # AUROC / AUPRC (one-vs-rest)
    try:
        y_bin         = label_binarize(y_true, classes=list(range(n)))
        macro_auroc   = roc_auc_score(y_bin, y_proba, average='macro',    multi_class='ovr')
        w_auroc       = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        macro_auprc   = average_precision_score(y_bin, y_proba, average='macro')
        w_auprc       = average_precision_score(y_bin, y_proba, average='weighted')
    except Exception:
        macro_auroc = w_auroc = macro_auprc = w_auprc = 0.0

    # Per-class: TN/FP/FN/TP → Se, Sp, Pr, F1, Acc
    cms = multilabel_confusion_matrix(y_true, y_pred)
    pc_acc, pc_se, pc_sp, pc_pr, pc_f1 = [], [], [], [], []
    for cm in cms:
        TN, FP, FN, TP = cm.ravel()
        total = TN + FP + FN + TP
        se  = TP / (TP + FN) if (TP + FN) else 0.0
        sp  = TN / (TN + FP) if (TN + FP) else 0.0
        pr  = TP / (TP + FP) if (TP + FP) else 0.0
        f1  = 2 * pr * se / (pr + se) if (pr + se) else 0.0
        ac  = (TP + TN) / total        if total    else 0.0
        pc_acc.append(ac); pc_se.append(se); pc_sp.append(sp)
        pc_pr.append(pr);  pc_f1.append(f1)

    counts           = np.bincount(y_true, minlength=n).astype(float)
    macro_sp         = float(np.mean(pc_sp))
    w_sp             = float(np.average(pc_sp, weights=counts))

    return {
        'acc':               acc,
        'macro_precision':   macro_precision,
        'macro_recall':      macro_recall,
        'macro_f1':          macro_f1,
        'macro_specificity': macro_sp,
        'macro_auroc':       macro_auroc,
        'macro_auprc':       macro_auprc,
        'w_precision':       w_precision,
        'w_recall':          w_recall,
        'w_f1':              w_f1,
        'w_specificity':     w_sp,
        'w_auroc':           w_auroc,
        'w_auprc':           w_auprc,
        'pc_acc':            pc_acc,
        'pc_se':             pc_se,
        'pc_sp':             pc_sp,
        'pc_pr':             pc_pr,
        'pc_f1':             pc_f1,
    }


def _compute_metrics_multilabel(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Metrics for multi-label classification (Hicardi)."""
    y_true = (y_true > 0).astype(int)   # guard against cached non-binary labels
    y_pred = (y_pred > 0).astype(int)
    n      = config.N_CLASSES
    
    # Subset accuracy (exact match)
    subset_acc = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Per-label metrics (macro and weighted)
    macro_precision  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_recall     = recall_score   (y_true, y_pred, average='macro',    zero_division=0)
    macro_f1         = f1_score       (y_true, y_pred, average='macro',    zero_division=0)
    
    w_precision      = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    w_recall         = recall_score   (y_true, y_pred, average='weighted', zero_division=0)
    w_f1             = f1_score       (y_true, y_pred, average='weighted', zero_division=0)
    
    # Micro (global TP/FP/FN)
    micro_precision  = precision_score(y_true, y_pred, average='micro',    zero_division=0)
    micro_recall     = recall_score   (y_true, y_pred, average='micro',    zero_division=0)
    micro_f1         = f1_score       (y_true, y_pred, average='micro',    zero_division=0)
    
    # Hamming loss (fraction of labels that are incorrectly predicted)
    hamming = hamming_loss(y_true, y_pred)
    
    # Per-class metrics
    pc_pr, pc_re, pc_f1 = [], [], []
    for i in range(n):
        pr = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        re = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        pc_pr.append(pr)
        pc_re.append(re)
        pc_f1.append(f1)
    
    # AUC per label (if probabilities provided)
    per_label_auc = []
    if y_proba is not None:
        y_proba = y_proba.astype(float)
        for i in range(n):
            try:
                auc = roc_auc_score(y_true[:, i], y_proba[:, i])
                per_label_auc.append(auc)
            except Exception:
                per_label_auc.append(0.0)
    
    return {
        'subset_accuracy': subset_acc,      # Exact match ratio
        'hamming_loss':    hamming,         # Fraction of incorrect labels
        'macro_precision': macro_precision,
        'macro_recall':    macro_recall,
        'macro_f1':        macro_f1,
        'micro_precision': micro_precision,
        'micro_recall':    micro_recall,
        'micro_f1':        micro_f1,
        'w_precision':     w_precision,
        'w_recall':        w_recall,
        'w_f1':            w_f1,
        'pc_precision':    pc_pr,
        'pc_recall':       pc_re,
        'pc_f1':           pc_f1,
        'per_label_auc':   per_label_auc if y_proba is not None else [],
    }
