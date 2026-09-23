"""Classification metrics and validation-based threshold selection."""

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    p,
    threshold,
):
    """Threshold-free and thresholded metrics for predicted probabilities."""

    y_true = np.asarray(y_true)
    p = np.asarray(p)

    y_pred = (
        p >= threshold
    ).astype(int)

    single_class = (
        len(np.unique(y_true)) < 2
    )

    return {
        'roc_auc':
            np.nan
            if single_class
            else roc_auc_score(y_true, p),

        'pr_auc':
            np.nan
            if single_class
            else average_precision_score(
                y_true,
                p,
            ),

        'f1':
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        'precision':
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        'recall':
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        'brier':
            brier_score_loss(
                y_true,
                np.clip(p, 0, 1),
            ),

        'threshold':
            threshold,
    }


def safe_auc(
    y_true,
    p,
):
    """ROC-AUC that degrades to 0.5 on a single-class partition."""

    y_true = np.asarray(y_true)
    p = np.asarray(p)

    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(
        y_true,
        p,
    )


def select_threshold(
    y_true,
    p,
):
    """Return the threshold maximising validation F1."""

    y_true = np.asarray(y_true)
    p = np.asarray(p)

    candidates = np.unique(
        np.round(
            p,
            4,
        )
    )

    if len(candidates) < 2:
        return 0.5

    return max(
        (
            f1_score(
                y_true,
                (p >= threshold).astype(int),
                zero_division=0,
            ),
            threshold,
        )
        for threshold in candidates
    )[1]