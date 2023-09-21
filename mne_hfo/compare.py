import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mutual_info_score

from .posthoc import match_detected_annotations

implemented_comparisons = ["cohen-kappa", "mutual-info"]


def compare_detectors(clf_1, clf_2, method="cohen-kappa"):
    """
    Compare fits for two classifiers per channel.

    Comparisons should be symmetrical.

    Parameters
    ----------
    clf_1: Detector
        Detector that contains detections from calling detector.fit()
    clf_2: Detector
        Detector that contains detections from calling detector.fit()
    method: str
        The method to use for comparison. Either 'cohen-kappa' or 'mutual-info'

    Returns
    -------
    ch_compares: dict
        Map of channel name to metric value.

    """
    if not hasattr(clf_1, "df_"):
        raise RuntimeError("clf_1 must be fit to data before using compare")
    if not hasattr(clf_2, "df_"):
        raise RuntimeError("clf_2 must be fit to data before using compare")
    if method == "cohen-kappa":
        comp = _cohen_kappa
    elif method == "mutual-info":
        comp = _mutual_info
    else:
        raise NotImplementedError(
            f"Method must be one of "
            f"{', '.join(implemented_comparisons)}. You"
            f"passed {method}."
        )
    # Get the detection dataframes
    df1 = clf_1.df_
    df2 = clf_2.df_

    # Calculate the overlap between fits
    overlap_df = match_detected_annotations(df1, df2, method="match-total")

    # Create dataframe groups by channel
    df1_g = df1.groupby("channels")
    df2_g = df2.groupby("channels")

    ch_names = df1_g.groups.keys()
    ch_compares = dict()
    for ch in ch_names:
        # Get the group for the channel
        df1_channel = df1_g.get_group(ch)
        df2_channel = df2_g.get_group(ch)
        # Get the indices of the original dataframe for that channel
        df1_indices = list(df1_channel.index.values)
        df2_indices = list(df2_channel.index.values)

        # Get the matching rows from the overlap dataframe
        # and merge these results
        df1_matches = overlap_df[overlap_df["true_index"].isin(df1_indices)]
        df2_matches = overlap_df[overlap_df["pred_index"].isin(df2_indices)]
        df_match_total = (
            pd.concat([df1_matches, df2_matches])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Get the detections as a list
        df1_predictions = list(df_match_total["true_index"])
        df2_predictions = list(df_match_total["pred_index"])

        # Convert the detections into labels
        df1_labels = [~np.isnan(pred) for pred in df1_predictions]
        df2_labels = [~np.isnan(pred) for pred in df2_predictions]

        ch_compares[ch] = comp(df1_labels, df2_labels)

    return ch_compares


def _mutual_info(lab1, lab2):
    """Call sklearn's mutual info function."""
    return mutual_info_score(lab1, lab2)


def _cohen_kappa(lab1, lab2):
    """Call sklearn's kappa score function."""
    return cohen_kappa_score(lab1, lab2)
