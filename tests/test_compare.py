import pytest

from mne_hfo import create_annotations_df
from mne_hfo.compare import compare_detectors
from mne_hfo.detect import RMSDetector
from numpy.testing import assert_almost_equal


def test_compare_detectors():
    """Test comparison metrics."""

    # Create two dummy RMSDetector objects.
    rms1 = RMSDetector()
    rms2 = RMSDetector()

    # Make sure you can't run compare when Detectors haven't been fit
    with pytest.raises(RuntimeError, match='clf_1 must be fit'
                                           ' to data before using compare'):
        compare_detectors(rms1, rms2, method="mutual-info")

    # Create two event dataframes with expected columns. We will
    # consider df1 to be predictions from rms1 and df2 to be predictions
    # from rms2
    sfreq = 1000
    # create dummy reference annotations
    onset1 = [8, 12.6, 59.9, 99.2, 150.4]
    offset1 = [9.7300, 14.870, 66.1, 101.22, 156.1]
    duration1 = [offset - onset for onset, offset in zip(onset1, offset1)]
    ch_name = ['A1'] * len(onset1)
    annotation_label = ['hfo'] * len(onset1)
    annot_df1 = create_annotations_df(onset1, duration1, ch_name,
                                      annotation_label)
    annot_df1['sample'] = annot_df1['onset'] * sfreq

    # create dummy predicted HFO annotations
    onset2 = [2, 60.1, 98.3, 110.23]
    offset2 = [6.93, 65.6, 101.45, 112.89]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]
    ch_name = ['A1'] * len(onset2)
    annotation_label = ['hfo'] * len(onset2)
    annot_df2 = create_annotations_df(onset2, duration2, ch_name,
                                      annotation_label)
    annot_df2['sample'] = annot_df2['onset'] * sfreq

    # Attach the annotation dataframes to the dummy detectors
    rms1.df_ = annot_df1

    # Make sure you can't run compare when Detectors haven't been fit
    with pytest.raises(RuntimeError, match='clf_2 must be fit'
                                           ' to data before using compare'):
        compare_detectors(rms1, rms2, method="mutual-info")

    rms2.df_ = annot_df2

    # We expect the labels from rms1 to be [False, True, True, True,
    # True, False, True]
    # and the labels from rms2 to be [True, False, False, True, True,
    # True, False]
    # which gives the following mutual info and kappa scores

    expected_mutual_info = 0.20218548540814557
    expected_kappa_score = -0.5217391304347827

    # Calculate mutual info and assert almost equal
    mutual_info = compare_detectors(rms1, rms2, method="mutual-info")
    mi = mutual_info['A1']
    assert_almost_equal(mi, expected_mutual_info, decimal=5)

    # Calculate kappa score and assert almost equal
    kappa = compare_detectors(rms1, rms2, method="cohen-kappa")
    k = kappa['A1']
    assert_almost_equal(k, expected_kappa_score, decimal=5)

    # Make sure you can't run a random method
    with pytest.raises(NotImplementedError):
        compare_detectors(rms1, rms2, method="average")
