import numpy as np
import pandas as pd

from mne_hfo.posthoc import match_detected_annotations
from mne_hfo.sklearn import _convert_y_sklearn_to_annot_df
from mne_hfo.utils import _check_df


def true_positive_rate(y, y_pred):
    """
    Calculate true positive rate as: tpr = tp / (tp + fn).

    Parameters
    ----------
    y : pd.DataFrame
        Event Dataframe with actual labels
    y_pred : pd.DataFrame
        Event Dataframe with predicted labels

    Returns
    -------
    float

    """
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')

    # return actual metric
    return tp / (tp + fn)


def precision(y, y_pred):
    """
    Calculate precision as: precision = tp / (tp + fp).

    Parameters
    ----------
    y : pd.DataFrame
        Event Dataframe with actual labels
    y_pred : pd.DataFrame
        Event Dataframe with predicted labels

    Returns
    -------
    float

    """
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')

    if tp == 0:
        return 0.

    # return actual metric
    return tp / (tp + fp)


def false_negative_rate(y, y_pred):
    """
    Calculate false negative rate as: fnr = fn / (fn + tp).

    Parameters
    ----------
    y : pd.DataFrame
        Event Dataframe with actual labels
    y_pred : pd.DataFrame
        Event Dataframe with predicted labels

    Returns
    -------
    float

    """
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')

    # return actual metric
    return fn / (fn + tp)


def false_discovery_rate(y, y_pred):
    """
    Calculate false positive rate as: fdr = fp / (fp + tp).

    Parameters
    ----------
    y : pd.DataFrame
        Event Dataframe with actual labels
    y_pred : pd.DataFrame
        Event Dataframe with predicted labels

    Returns
    -------
    float

    """
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')

    if fp == 0.:
        return 0.

    # return the actual metric
    return fp / (fp + tp)


def accuracy(y, y_pred):
    """
    Calculate accuracy as: accuracy = tp / (tp + fp + fn).

    Follows usual formula for accuracy:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    but assumes tn = 0.

    Parameters
    ----------
    y : pd.DataFrame
        Annotation Dataframe with actual labels
    y_pred : pd.DataFrame
        Annotation Dataframe with predicted labels

    Returns
    -------
    float

    """
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')
    # return actual metric
    return tp / (tp + fp + fn)


def _compute_score_data(y, y_pred, method):
    """Compute basic HFO scoring metrics."""
    if isinstance(y, pd.DataFrame):
        y = _check_df(y, df_type='annotations')
    else:
        # assume y is now in the form of list of (onset, offset) per channel
        y = _convert_y_sklearn_to_annot_df(y)

    if isinstance(y_pred, pd.DataFrame):
        y_pred = _check_df(y_pred, df_type='annotations')
    else:
        # assume y is now in the form of list of (onset, offset) per channel
        y_pred = _convert_y_sklearn_to_annot_df(y_pred)

    overlap_df = match_detected_annotations(y, y_pred, method=method)

    # get the indices from the match event overlap output
    y_true_series = overlap_df['true_index']
    y_pred_series = overlap_df['pred_index']
    tp, fp, fn = _calculate_match_stats(ytrue_indices=y_true_series,
                                        ypred_indices=y_pred_series)
    return tp, fp, fn


def _calculate_match_stats(ytrue_indices, ypred_indices):
    """
    Calculate true positives, false positives, and false negatives.

    True negatives cannot be calculated for this dataset as of now.

    Parameters
    ----------
    ytrue_indices : pd.Series
        Pandas Series with a number corresponding to an index and a
        ``nan`` if there is no match.
    ypred_indices : pd.Series
        Pandas Series with a number corresponding to an index and a
        ``nan`` if there is no match.

    Returns
    -------
    tp: int
        number of true positives - i.e. prediction and actual are both true
    fp: int
        number of false positives - i.e. prediction is true and actual is false
    fn: int
        number of false negatives - i.e. prediction is false and actual is true
    """
    # Convert the match df structure to two lists of booleans.
    # (True) and negatives (False).
    # True if an index is present, False if Nan
    y_true_bool = ytrue_indices.notna().to_list()
    y_pred_bool = ypred_indices.notna().to_list()

    # compute true positive, false positive and false negative
    label_pairs = tuple(zip(y_true_bool, y_pred_bool))
    tp = np.sum([(t and p) for t, p in label_pairs])
    fp = np.sum([(p and not t) for t, p in label_pairs])
    fn = np.sum([(t and not p) for t, p in label_pairs])
    return tp, fp, fn
