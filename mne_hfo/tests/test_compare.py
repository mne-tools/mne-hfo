import pytest

from mne import Annotations

from mne_hfo import create_annotations_df
from mne_hfo.compare import compare_detectors
from mne_hfo.detect import RMSDetector
from numpy.testing import assert_almost_equal

@pytest.fixture(scope='function')
def create_detector1():
    # Create two dummy RMSDetector objects.
    rms1 = RMSDetector()

    # Create two event dataframes with expected columns. We will
    # consider df1 to be predictions from rms1 and df2 to be predictions
    # from rms2
    sfreq = 1000
    # create dummy reference annotations
    onset1 = [8, 12.6, 59.9, 99.2, 150.4]
    offset1 = [9.7300, 14.870, 66.1, 101.22, 156.1]
    duration1 = [offset - onset for onset, offset in zip(onset1, offset1)]

    hfo_annotations = []
    description = ['hfo'] * len(onset1)
    ch_names = [['A1'] for _ in range (len(onset1))]
    ch_hfo_events = Annotations(onset=onset1, duration=duration1,
                                description=description,
                                ch_names=ch_names)
    hfo_annotations.append(ch_hfo_events)
    rms1.hfo_annotations_ = hfo_annotations[0]
    rms1.sfreq = sfreq
    rms1.ch_names = ['A1']

    # Gives dummy detector a length of the data used.
    rms1.n_times = 200*1000

    return rms1

@pytest.fixture(scope='function')
def create_detector2():
    # Create two dummy RMSDetector objects.
    rms2 = RMSDetector()

    # Create two event dataframes with expected columns. We will
    # consider df1 to be predictions from rms1 and df2 to be predictions
    # from rms2
    sfreq = 1000

    # create dummy predicted HFO annotations
    onset2 = [2, 60.1, 98.3, 110.23]
    offset2 = [6.93, 65.6, 101.45, 112.89]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]

    hfo_annotations = []
    description = ['hfo'] * len(onset2)
    ch_names = [['A1'] for _ in range (len(onset2))]
    ch_hfo_events = Annotations(onset=onset2, duration=duration2,
                                description=description,
                                ch_names=ch_names)
    hfo_annotations.append(ch_hfo_events)
    rms2.hfo_annotations_ = hfo_annotations[0]
    rms2.sfreq = sfreq
    rms2.ch_names = ['A1']

    # Gives dummy detector a length of the data used.
    rms2.n_times = 200*1000

    return rms2

def test_compare_detectors():
    """Test comparison metrics."""

    # Create two dummy RMSDetector objects.
    rms1 = RMSDetector()
    rms2 = RMSDetector()

    # Make sure you can't run compare when Detectors haven't been fit
    with pytest.raises(RuntimeError, match='clf_1 must be fit'
                                           ' to data before using compare'):
        compare_detectors(rms1, rms2,
                          comp_method="mutual-info",
                          label_method="overlap-predictions")

    # Create two event dataframes with expected columns. We will
    # consider df1 to be predictions from rms1 and df2 to be predictions
    # from rms2
    sfreq = 1000
    # create dummy reference annotations
    onset1 = [8, 12.6, 59.9, 99.2, 150.4]
    offset1 = [9.7300, 14.870, 66.1, 101.22, 156.1]
    duration1 = [offset - onset for onset, offset in zip(onset1, offset1)]

    # Make sure you can't run compare when Detectors haven't been fit
    with pytest.raises(RuntimeError, match='clf_1 must be fit'
                                           ' to data before using compare'):
        compare_detectors(rms1, rms2,
                          comp_method="mutual-info",
                          label_method="overlap-predictions")

    hfo_annotations = []
    description = ['hfo'] * len(onset1)
    ch_names = [['A1'] for _ in range (len(onset1))]
    ch_hfo_events = Annotations(onset=onset1, duration=duration1,
                                description=description,
                                ch_names=ch_names)
    hfo_annotations.append(ch_hfo_events)
    rms1.hfo_annotations_ = hfo_annotations[0]
    rms1.sfreq = sfreq
    rms1.ch_names = ['A1']

    # Make sure you can't run compare when Detectors haven't been fit
    with pytest.raises(RuntimeError, match='clf_2 must be fit'
                       ' to data before using compare'):
        compare_detectors(rms1, rms2,
                          comp_method="mutual-info",
                          label_method="overlap-predictions")

    # create dummy predicted HFO annotations
    onset2 = [2, 60.1, 98.3, 110.23]
    offset2 = [6.93, 65.6, 101.45, 112.89]
    duration2 = [offset - onset for onset, offset in zip(onset2, offset2)]

    hfo_annotations = []
    description = ['hfo'] * len(onset2)
    ch_names = [['A1'] for _ in range (len(onset2))]
    ch_hfo_events = Annotations(onset=onset2, duration=duration2,
                                description=description,
                                ch_names=ch_names)
    hfo_annotations.append(ch_hfo_events)
    rms2.hfo_annotations_ = hfo_annotations[0]
    rms2.sfreq = sfreq
    rms2.ch_names = ['A1']

    # Gives dummy detector a length of the data used.
    rms1.n_times = 200*1000

    with pytest.raises(RuntimeError, match='clf_2 must be fit'
                       ' to data before using compare'):
        compare_detectors(rms1, rms2,
                          comp_method="mutual-info",
                          label_method="overlap-predictions")
        
    rms2.n_times = 100*1000

    # Make sure the length of the raw data for each classifier are identical
    with pytest.raises(RuntimeError, match='clf_1 and clf_2 must be fit'
                       ' on the same length of data'):
        compare_detectors(rms1, rms2,
                          comp_method="mutual-info",
                          label_method="overlap-predictions")
        
    rms2.n_times = 200*1000

def test_comparison_methods(create_detector1, create_detector2):
    det1 = create_detector1
    det2 = create_detector2

    # We expect the labels from rms1 to be [False, True, True, True,
    # True, False, True]
    # and the labels from rms2 to be [True, False, False, True, True,
    # True, False]
    # which gives the following mutual info and kappa scores

    expected_mutual_info = 0.20218548540814557
    expected_kappa_score = -0.5217391304347827
    expected_similarity = 0.28571429

    # Calculate mutual info and assert almost equal
    mutual_info = compare_detectors(det1, det2, label_method="overlap-predictions", comp_method="mutual-info")
    mi = mutual_info['A1']
    assert_almost_equal(mi, expected_mutual_info, decimal=5)

    # Calculate kappa score and assert almost equal
    kappa = compare_detectors(det1, det2, label_method="overlap-predictions", comp_method="cohen-kappa")
    k = kappa['A1']
    assert_almost_equal(k, expected_kappa_score, decimal=5)

    similarity = compare_detectors(det1, det2, label_method="overlap-predictions", comp_method="similarity-ratio")
    s = similarity['A1']
    assert_almost_equal(s, expected_similarity, decimal=5)

    # Make sure you can't run a random method
    with pytest.raises(NotImplementedError):
        compare_detectors(det1, det2, label_method="overlap=predictions", comp_method="average")



def test_labeling_methods(create_detector1, create_detector2):
    det1 = create_detector1
    det2 = create_detector2

    # We expect the labels from rms1 to be [False, True, True, True,
    # True, False, True]
    # and the labels from rms2 to be [True, False, False, True, True,
    # True, False]
    # which gives the following mutual info and kappa scores

    expected_binning_score = 0.86
    expected_raw_detections = 0.88575

    # Calculate kappa score and assert almost equal
    binning = compare_detectors(det1, det2,
                                label_method="simple-binning",
                                comp_method="similarity-ratio",
                                bin_width=1000)
    b = binning['A1']
    assert_almost_equal(b, expected_binning_score, decimal=2)

    raw_detections = compare_detectors(det1, det2,
                                       label_method="raw-detections",
                                       comp_method="similarity-ratio")
    r = raw_detections['A1']
    assert_almost_equal(r, expected_raw_detections, decimal=5)

    # Make sure you can't run a random method
    with pytest.raises(NotImplementedError):
        compare_detectors(det1, det2, label_method="matching", comp_method="similarity-ratio")