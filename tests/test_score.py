import pytest
import pandas as pd

from mne_hfo import create_annotations_df
from mne_hfo.scores import accuracy_pred, accuracy_true


def test_match_detections():
    # First create two annotation dataframes with expected columns. We will consider df1 to be ground truth and df2 to
    # be the prediction
    onset1 = [0.0, 7.3, 12.6, 22.342, 59.9]
    duration1 = [6.73, 1.2,  2.27, 8.758, 21.3]
    ch_name = ['A1', 'A1', 'A1', 'A1', 'A1']
    sfreq = 1000
    annot_df1 = create_annotations_df(onset1, duration1, ch_name)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    onset2 = [0.2, 12.3, 45.8, 98.3]
    duration2 = [6.73, 2.82, 19.8, 3.15]
    ch_name = ['A1', 'A1', 'A1', 'A1']
    annot_df2 = create_annotations_df(onset2, duration2, ch_name)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # In the above example, we have correctly predicted 3/5 of the hfos from df1, which is ground truth. We expect this
    # accuracy to be 0.6

    score_true = accuracy_true(annot_df1, annot_df2, sample_weight=None)
    assert score_true == 0.6

    # Also, 3/4 of the predicted hfos correlate to actual hfos. So we expect this accuracy to be 0.75

    score_true = accuracy_pred(annot_df1, annot_df2, sample_weight=None)
    assert score_true == 0.75
