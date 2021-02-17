
def accuracy_true(y, y_pred, sample_weight=None):
    """
    Calculate percentage of true labels accurately predicted.

    Parameters
    ----------
    y : pd.DataFrame
        True labels
    y_pred : pd.DataFrame
        Predicted labels
    sample_weight :

    Returns
    -------
    float
    """
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-true")
    correct_series = overlap_df.count()
    return correct_series.get('pred_index') / correct_series.get('true_index')


def accuracy_pred(y, y_pred, sample_weight=None):
    """
    Calculate percentage of predicted labels that are true.

    Parameters
    ----------
    y : pd.DataFrame
        True labels
    y_pred : pd.DataFrame
        Predicted labels
    sample_weight :

    Returns
    -------
    float
    """
    from mne_hfo import match_detections
    overlap_df = match_detections(y, y_pred, method="match-pred")
    correct_series = overlap_df.count()
    return correct_series.get('true_index') / correct_series.get('pred_index')