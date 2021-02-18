import numpy as np


def true_positive_rate(y, y_pred):
    """
    Calculate true positive rate as: tpr = tp / (tp + fn).

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
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-total")
    tp, fp, fn = _calculate_match_stats(overlap_df)
    return tp / (tp + fn)


def precision(y, y_pred):
    """
    Calculate precision as: precision = tp / (tp + fp).

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
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-total")
    tp, fp, fn = _calculate_match_stats(overlap_df)
    return tp / (tp + fp)


def false_negative_rate(y, y_pred):
    """
    Calculate false negative rate as: fnr = fn / (fn + tp).

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
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-total")
    tp, fp, fn = _calculate_match_stats(overlap_df)
    return fn / (fn + tp)


def false_discovery_rate(y, y_pred):
    """
    Calculate false positive rate as: fdr = fp / (fp + tp).

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
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-total")
    tp, fp, fn = _calculate_match_stats(overlap_df)
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
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-total")
    tp, fp, fn = _calculate_match_stats(overlap_df)
    return tp / (tp + fp + fn)


def _convert_matchdf_to_pred_array(match_df):
    """
    Convert the match df structure to two lists of positives
    (True) and negatives (False).

    Parameters
    ----------
    match_df : pd.DataFrame
        pandas dataframe with columns true_index and pred_index

    Returns
    -------
    y_true_bool: list
        Boolean list for true labels. True if an index is present, False if Nan
    y_pred_bool: list
        Boolean list for predicted labels. True if an index is
        present, False if Nan
    """
    y_true_series = match_df['true_index']
    y_true_bool = y_true_series.notna().to_list()
    y_pred_series = match_df['pred_index']
    y_pred_bool = y_pred_series.notna().to_list()
    return y_true_bool, y_pred_bool


def _calculate_match_stats(match_df):
    """
    Calculate true positives, false positives, and false negatives.

    True negatives cannot be calculated for this dataset as of now.

    Parameters
    ----------
    match_df : pd.DataFrame
        pandas dataframe with columns true_index and pred_index

    Returns
    -------
    tp: int
        number of true positives - i.e. prediction and actual are both true
    fp: int
        number of false positives - i.e. prediction is true and actual is false
    fn: int
        number of false negatives - i.e. prediction is false and actual is true

    """
    y_true_bool, y_pred_bool = _convert_matchdf_to_pred_array(match_df)
    label_pairs = tuple(zip(y_true_bool, y_pred_bool))
    tp = np.sum([(t and p) for t, p in label_pairs])
    fp = np.sum([(p and not t) for t, p in label_pairs])
    fn = np.sum([(t and not p) for t, p in label_pairs])
    return tp, fp, fn
