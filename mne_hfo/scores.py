import numpy as np

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
    # y = _check_df(y, df_type='events')
    # y_pred = _check_df(y_pred, df_type='events')
    # overlap_df = match_detections(y, y_pred, method="match-total")
    # tp, fp, fn = _calculate_match_stats(overlap_df)
    # return tp / (tp + fn)


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
    # y = _check_df(y, df_type='events')
    # y_pred = _check_df(y_pred, df_type='events')
    # overlap_df = match_detections(y, y_pred, method="match-total")
    # tp, fp, fn = _calculate_match_stats(overlap_df)
    # return tp / (tp + fp)


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
    # y = _check_df(y, df_type='events')
    # y_pred = _check_df(y_pred, df_type='events')
    # overlap_df = match_detections(y, y_pred, method="match-total")
    # tp, fp, fn = _calculate_match_stats(overlap_df)
    # return fn / (fn + tp)


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
    # y = _check_df(y, df_type='events')
    # y_pred = _check_df(y_pred, df_type='events')
    # overlap_df = match_detections(y, y_pred, method="match-total")
    # tp, fp, fn = _calculate_match_stats(overlap_df)
    # return fp / (fp + tp)


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
    return 1
    # compute score statistics
    tp, fp, fn = _compute_score_data(y, y_pred, method='match-total')

    # return actual metric
    return tp / (tp + fp + fn)


def _compute_score_data(y, y_pred, method):
    """Compute basic HFO scoring metrics."""
    from mne_hfo import match_detections
    # if isinstance(y, pd.DataFrame):
    #     y = _check_df(y, df_type='annotations')
    # else:
    # assume y is now in the form of list of (onset, offset) per channel
    # y = _make_ydf_sklearn(y, ch_names=)

    # convert both list of list of tuples into a DataFrame

    # y predictions from HFO detectors should always be a dataframe
    y_pred = _check_df(y_pred, df_type='annotations')
    overlap_df = match_detections(y, y_pred, method=method)

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
