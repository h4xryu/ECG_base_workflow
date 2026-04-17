import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, multilabel_confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
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
