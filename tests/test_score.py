from mne_hfo import create_annotations_df
from mne_hfo.scores import accuracy, true_positive_rate, \
    false_negative_rate, false_discovery_rate, precision


def test_match_detections():
    # First create two annotation dataframes with expected columns. We will
    # consider df1 to be ground truth and df2 to be the prediction
    onset1 = [0.0, 7.3, 12.6, 22.342, 59.9]
    duration1 = [6.73, 1.2, 2.27, 8.758, 21.3]
    ch_name = ['A1', 'A1', 'A1', 'A1', 'A1']
    sfreq = 1000
    annot_df1 = create_annotations_df(onset1, duration1, ch_name)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    onset2 = [0.2, 12.3, 45.8, 98.3]
    duration2 = [6.73, 2.82, 19.8, 3.15]
    ch_name = ['A1', 'A1', 'A1', 'A1']
    annot_df2 = create_annotations_df(onset2, duration2, ch_name)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # In the above example, we have 3 true positives, 2 false negatives,
    # and 1 false positive Therefore, we expect accuracy = 0.5, tpr = 0.6,
    # fnr = 0.4, fdr = 0.25, and precision = 0.75

    acc = accuracy(annot_df1, annot_df2)
    assert acc == 0.5

    tpr = true_positive_rate(annot_df1, annot_df2)
    assert tpr == 0.6

    fnr = false_negative_rate(annot_df1, annot_df2)
    assert fnr == 0.4

    fdr = false_discovery_rate(annot_df1, annot_df2)
    assert fdr == 0.25

    prec = precision(annot_df1, annot_df2)
    assert prec == 0.75
