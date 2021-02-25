import itertools

import mne
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from mne_hfo import (
    create_annotations_df, find_coincident_events,
    compute_chs_hfo_rates, merge_overlapping_events,
    LineLengthDetector)
from mne_hfo.config import TIME_SCALE_TO_SECS
from mne_hfo.posthoc import match_detected_annotations
from mne_hfo.score import (accuracy, false_negative_rate, false_discovery_rate,
                           true_positive_rate, precision)
from mne_hfo.sklearn import make_Xy_sklearn, DisabledCV


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


def test_match_hfo_annotations():
    """Test matching HFO detections encoded in annotations DataFrame."""
    sfreq = 1000
    # create dummy reference annotations
    onset1 = [1.5, 12.6, 22.342, 59.9]
    offset1 = [6.7300, 14.870, 31.1, 81.2]
    duration1 = [offset - onset for onset, offset in zip(onset1, offset1)]
    ch_name = ['A1'] * len(onset1)
    annotation_label = ['hfo'] * len(onset1)
    annot_df1 = create_annotations_df(onset1, duration1, ch_name,
                                      annotation_label)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    # create dummy predicted HFO annotations
    onset2 = [2, 12.3, 60.1, 98.3]
    offset2 = [6.93, 15.12, 65.6, 101.45]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]
    ch_name = ['A1'] * len(onset2)
    annotation_label = ['hfo'] * len(onset2)
    annot_df2 = create_annotations_df(onset2, duration2, ch_name,
                                      annotation_label)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # We first want to see what true labels are correctly predicted
    expected_dict_true = {
        "true_index": [0, 1, 2, 3],
        "pred_index": [0, 1, None, 2]
    }
    expected_df_true = pd.DataFrame(expected_dict_true)
    expected_df_true = expected_df_true.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_true = match_detected_annotations(annot_df1, annot_df2,
                                                method="match-true")
    pd.testing.assert_frame_equal(expected_df_true, output_df_true,
                                  check_dtype=False)

    # Now lets check what predicted labels correspond to true labels
    expected_dict_pred = {
        "pred_index": [0, 1, 2, 3],
        "true_index": [0, 1, 3, None]
    }
    expected_df_pred = pd.DataFrame(expected_dict_pred)
    expected_df_pred = expected_df_pred.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_pred = match_detected_annotations(annot_df1, annot_df2,
                                                method="match-pred")
    pd.testing.assert_frame_equal(expected_df_pred, output_df_pred,
                                  check_dtype=False)

    # Now we can check the total output that will make it easier
    # to compute other stats
    expected_dict_total = {
        "true_index": [0, 1, 2, 3, None],
        "pred_index": [0, 1, None, 2, 3]
    }
    expected_df_total = pd.DataFrame(expected_dict_total)
    expected_df_total = expected_df_total.apply(pd.to_numeric, errors="coerce",
                                                downcast="float")
    output_df_total = match_detected_annotations(annot_df1, annot_df2,
                                                 method="match-total")
    pd.testing.assert_frame_equal(expected_df_total, output_df_total,
                                  check_dtype=False)

    # Error should be thrown for any other passed methods
    with pytest.raises(NotImplementedError, match=''):
        match_detected_annotations(annot_df1, annot_df2,
                                   method="match-average")


def test_match_detections_empty():
    # First create two annotation dataframes with expected columns. We will
    # consider df1 to be ground truth and df2 to be the prediction
    sfreq = 1000
    # create dummy reference annotations
    onset1 = [1.5, 12.6, 22.342, 59.9]
    offset1 = [6.7300, 14.870, 31.1, 81.2]
    duration1 = [offset - onset for onset, offset in zip(onset1, offset1)]
    ch_name = ['A1'] * len(onset1)
    annotation_label = ['hfo'] * len(onset1)
    annot_df1 = create_annotations_df(onset1, duration1, ch_name,
                                      annotation_label)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    # create dummy reference annotations
    onset2 = []
    offset2 = []
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]
    ch_name = ['A1'] * len(onset2)
    annotation_label = ['hfo'] * len(onset2)
    annot_df2 = create_annotations_df(onset2, duration2, ch_name,
                                      annotation_label)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    expected_dict_true = {
        "true_index": [0, 1, 2, 3],
        "pred_index": [None, None, None, None]
    }
    expected_df_true = pd.DataFrame(expected_dict_true)
    expected_df_true = expected_df_true.apply(pd.to_numeric, errors="coerce",
                                              downcast="float")
    output_df_true = match_detected_annotations(
        annot_df1, annot_df2, method="match-true")
    pd.testing.assert_frame_equal(expected_df_true, output_df_true,
                                  check_dtype=False)

    # Now lets check what predicted labels correspond to true labels.
    # Should be empty
    output_df_pred = match_detected_annotations(
        annot_df1, annot_df2, method="match-pred")
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
    output_df_total = match_detected_annotations(
        annot_df1, annot_df2, method="match-total")
    pd.testing.assert_frame_equal(expected_df_total, output_df_total,
                                  check_dtype=False)


@pytest.mark.parametrize("scorer", [
    accuracy,
    precision, true_positive_rate,
    false_negative_rate, false_discovery_rate
])
def test_hyperparameter_search_cv(scorer, create_testing_eeg_data):
    sfreq = 5000
    ch_names = ['0']

    parameters = {'threshold': [1, 2, 3], 'win_size': [50, 100, 250]}
    detector = LineLengthDetector()
    scorer = make_scorer(scorer)
    cv = DisabledCV()
    gs = GridSearchCV(detector, param_grid=parameters, scoring=scorer,
                      cv=cv, refit=False, verbose=True)

    # create dummy EEG data with "true" HFO samples
    data, hfo_samps = create_testing_eeg_data
    data_2d = data[np.newaxis, :]
    data_2d = np.vstack((data_2d, data_2d))
    onset_samp = np.array([samp[0] for samp in hfo_samps])
    offset_samp = np.array([samp[1] for samp in hfo_samps])
    onset = onset_samp / sfreq
    offset = offset_samp / sfreq
    duration = offset - onset
    ch_names = ['A0'] * len(onset)

    # create actual Raw input data
    info = mne.create_info(ch_names=['A0', 'A1'], sfreq=sfreq, ch_types='ecog')
    raw = mne.io.RawArray(data_2d, info=info)

    # create the annotations dataframe
    annot_df = create_annotations_df(onset, duration, ch_names)
    annot_df['sample'] = annot_df['onset'] * sfreq

    # make sklearn compatible
    raw_df, y = make_Xy_sklearn(raw, annot_df)
    # run Gridsearch
    gs.fit(raw_df, y, groups=None)
    # print(pd.concat([pd.DataFrame(gs.cv_results_["params"]),
    #                 pd.DataFrame(gs.cv_results_["mean_test_score"],
    #                              columns=["Accuracy"])],axis=1))

    # uncomment this to see that gridsearch results
    # raise Exception('check out the print statements')


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
