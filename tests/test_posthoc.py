import itertools

import numpy as np
import pandas as pd
import pytest

from mne_hfo import (
    create_annotations_df, find_coincident_events,
    compute_chs_hfo_rates, merge_overlapping_events,
    match_detections)
from mne_hfo.config import TIME_SCALE_TO_SECS


def test_find_coincident_events():
    df1 = {
        "01": [(0.0, 6.73), (12.6, 14.87), (22.342, 31.1), (45.9, 67.2)]
    }
    df2 = {
        "01": [(0.2, 6.93), (12.3, 15.12), (45.8, 65.6), (98.3, 101.45)]
    }
    # Expect to return the first three events from df1,
    # which have overlap with df2
    expected_output = {
        "01": [(0.0, 6.73), (12.6, 14.87), (45.9, 67.2)]
    }
    coincident_df = find_coincident_events(df1, df2)
    assert expected_output == coincident_df

    df1["02"] = df1["01"]
    with pytest.raises(RuntimeError, match='The two dictionaries'
                                           ' must have the same keys.'):
        find_coincident_events(df1, df2)


def test_match_detections():
    # First create two annotation dataframes with expected columns.
    # We will consider df1 to be ground truth and df2 to be the prediction
    onset1 = [0.0, 12.6, 22.342, 59.9]
    duration1 = [6.73, 2.27, 8.758, 21.3]
    ch_name = ['A1', 'A1', 'A1', 'A1']
    sfreq = 1000
    annot_df1 = create_annotations_df(onset1, duration1, ch_name)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    onset2 = [0.2, 12.3, 45.8, 98.3]
    duration2 = [6.73, 2.82, 19.8, 3.15]
    annot_df2 = create_annotations_df(onset2, duration2, ch_name)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # We see overlap in the following pairs, listed with index from df1, df2:
    # (0, 0), (1,1), (3,2)

    # We first want to see what true labels are correctly predicted
    expected_dict_true = {
        "true_index": [0, 1, 2, 3],
        "pred_index": [0, 1, None, 2]
    }
    expected_df_true = pd.DataFrame(expected_dict_true)
    expected_df_true = expected_df_true.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_true = match_detections(annot_df1, annot_df2,
                                      method="match-true")
    pd.testing.assert_frame_equal(expected_df_true, output_df_true)

    # Now lets check what predicted labels correspond to true labels
    expected_dict_pred = {
        "pred_index": [0, 1, 2, 3],
        "true_index": [0, 1, 3, None]
    }
    expected_df_pred = pd.DataFrame(expected_dict_pred)
    expected_df_pred = expected_df_pred.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_pred = match_detections(annot_df1, annot_df2,
                                      method="match-pred")
    pd.testing.assert_frame_equal(expected_df_pred, output_df_pred)

    # Now we can check the total output that will make it easier
    # to compute other stats
    expected_dict_total = {
        "true_index": [0, 1, 2, 3, None],
        "pred_index": [0, 1, None, 2, 3]
    }
    expected_df_total = pd.DataFrame(expected_dict_total)
    expected_df_total = expected_df_total.apply(pd.to_numeric, errors="coerce",
                                                downcast="float")
    output_df_total = match_detections(annot_df1, annot_df2,
                                       method="match-total")
    pd.testing.assert_frame_equal(expected_df_total, output_df_total)

    # Error should be thrown for any other passed methods
    with pytest.raises(NotImplementedError, match=''):
        match_detections(annot_df1, annot_df2, method="match-average")


def test_match_detections_empty():
    # First create two annotation dataframes with expected columns. We will
    # consider df1 to be ground truth and df2 to be the prediction
    onset1 = [0.0, 12.6, 22.342, 59.9]
    duration1 = [6.73, 2.27, 8.758, 21.3]
    ch_name = ['A1', 'A1', 'A1', 'A1']
    sfreq = 1000
    annot_df1 = create_annotations_df(onset1, duration1, ch_name)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    onset2 = []
    duration2 = []
    ch_name = []
    annot_df2 = create_annotations_df(onset2, duration2, ch_name)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    expected_dict_true = {
        "true_index": [0, 1, 2, 3],
        "pred_index": [None, None, None, None]
    }
    expected_df_true = pd.DataFrame(expected_dict_true)
    expected_df_true = expected_df_true.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_true = match_detections(annot_df1, annot_df2,
                                      method="match-true")
    pd.testing.assert_frame_equal(expected_df_true, output_df_true)

    # Now lets check what predicted labels correspond to true labels.
    # Should be empty
    output_df_pred = match_detections(annot_df1, annot_df2,
                                      method="match-pred")
    assert output_df_pred.empty

    # Now we can check the total output that will make it easier
    # to compute other stats
    expected_dict_total = {
        "true_index": [0, 1, 2, 3],
        "pred_index": [None, None, None, None]
    }
    expected_df_total = pd.DataFrame(expected_dict_total)
    expected_df_total = expected_df_total.apply(pd.to_numeric, errors="coerce",
                                                downcast="float")
    output_df_total = match_detections(annot_df1, annot_df2,
                                       method="match-total")
    pd.testing.assert_frame_equal(expected_df_total, output_df_total)


def test_merge_overlapping_hfos():
    onset = [1.5, 2.0]
    duration = [0.1, 1.0]
    ch_name = ['A1', 'A1']
    sfreq = 1000

    # create the annotations dataframe
    annot_df = create_annotations_df(onset, duration, ch_name)
    annot_df['sample'] = annot_df['onset'] * sfreq

    # first, test that no merging occurs when it shouldn't
    # merge overlapping HFOs should result in the exact same
    # annotations dataframe
    new_annot_df = merge_overlapping_events(annot_df)
    pd.testing.assert_frame_equal(annot_df, new_annot_df)

    # now
    onset = [1.5, 2.0, 1.55]
    duration = [0.1, 1.0, 0.5]
    ch_name = ['A1', 'A1', 'A1']

    # create the annotations dataframe
    annot_df = create_annotations_df(onset, duration, ch_name)
    annot_df['sample'] = annot_df['onset'] * sfreq

    # nexxt, test that merging occurs when all three events overlap
    # merge overlapping HFOs should result in the exact same
    # annotations dataframe
    new_annot_df = merge_overlapping_events(annot_df)
    assert new_annot_df.shape == (1, 5)
    assert new_annot_df['onset'].values == [1.5]
    assert new_annot_df['duration'].values == [0.55]
    assert new_annot_df['channels'].values == ['A1']


# parameters for testing HFO rates over units of time
end_secs = [10, 20, 10.25, 50.5]
rates = ['s', 'm', 'h', 'd']


@pytest.mark.parametrize('end_sec, rate', itertools.product(end_secs, rates))
def test_metrics_df(end_sec, rate):
    """Test metrics based on reference dataframe and detector ran dataframe."""
    onset = [1.5, 2.0, 3, 5.5]
    duration = [0.1, 1.0, 1.0, 0.1]
    ch_name = ['A1', 'A2', 'A3', 'A1']
    sfreq = 1000

    annot_df = create_annotations_df(onset, duration, ch_name)

    # error will occur without the sample column
    with pytest.raises(RuntimeError, match='Annotations dataframe '
                                           'columns must contain'):
        compute_chs_hfo_rates(annot_df=annot_df, rate=rate)

    # now add sample column
    annot_df['sample'] = annot_df['onset'] * sfreq
    chs_hfo_rates = compute_chs_hfo_rates(annot_df=annot_df,
                                          rate=rate,
                                          end_sec=end_sec)
    assert chs_hfo_rates['A2'] == chs_hfo_rates['A3']
    np.testing.assert_almost_equal(chs_hfo_rates['A2'],
                                   (1. / end_sec) / TIME_SCALE_TO_SECS[rate],
                                   decimal=6)
    np.testing.assert_almost_equal(chs_hfo_rates['A1'],
                                   (2. / end_sec) / TIME_SCALE_TO_SECS[rate],
                                   decimal=6)

    # error if specifying channel names are not inside dataframe
    with pytest.raises(ValueError, match=''):
        compute_chs_hfo_rates(annot_df, rate=rate, ch_names=['A0', 'A1'])
