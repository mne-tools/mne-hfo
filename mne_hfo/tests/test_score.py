from mne_hfo import create_annotations_df
from mne_hfo.score import accuracy, true_positive_rate, \
    false_negative_rate, false_discovery_rate, precision
from mne_hfo.sklearn import _make_ydf_sklearn


def test_match_detection_scoring_df():
    # First create two event dataframes with expected columns. We will
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

    # create dummy predicted HFO annotations
    onset2 = [2, 12.3, 60.1, 98.3]
    offset2 = [6.93, 15.12, 65.6, 101.45]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]
    ch_name = ['A1'] * len(onset2)
    annotation_label = ['hfo'] * len(onset2)
    annot_df2 = create_annotations_df(onset2, duration2, ch_name,
                                      annotation_label)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # In the above example, we have 3 true positives, 1 false negative,
    # and 1 false positive. Therefore, we expect accuracy = 0.6, tpr = 0.75,
    # fnr = 0.25, fdr = 0.25, and precision = 0.75

    acc = accuracy(annot_df1, annot_df2)
    assert acc == 0.6

    tpr = true_positive_rate(annot_df1, annot_df2)
    assert tpr == 0.75

    fnr = false_negative_rate(annot_df1, annot_df2)
    assert fnr == 0.25

    fdr = false_discovery_rate(annot_df1, annot_df2)
    assert fdr == 0.25

    prec = precision(annot_df1, annot_df2)
    assert prec == 0.75


def test_match_detection_scoring_sklearn():
    # First create two event dataframes with expected columns. We will
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

    # create dummy predicted HFO annotations
    onset2 = [2, 12.3, 60.1, 98.3]
    offset2 = [6.93, 15.12, 65.6, 101.45]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]
    ch_name = ['A1'] * len(onset2)
    annotation_label = ['hfo'] * len(onset2)
    annot_df2 = create_annotations_df(onset2, duration2, ch_name,
                                      annotation_label)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # Now convert the annotation dataframes to "sklearn format" of a list
    # of lists. Will use y_true for annot_df1 and y_pred for annot_df2
    ch_names = ['A1']
    y_true = _make_ydf_sklearn(annot_df1, ch_names)
    y_pred = _make_ydf_sklearn(annot_df2, ch_names)

    # Perform scoring tests again

    acc = accuracy(y_true, y_pred)
    assert acc == 0.6

    tpr = true_positive_rate(y_true, y_pred)
    assert tpr == 0.75

    fnr = false_negative_rate(y_true, y_pred)
    assert fnr == 0.25

    fdr = false_discovery_rate(y_true, y_pred)
    assert fdr == 0.25

    prec = precision(y_true, y_pred)
    assert prec == 0.75
