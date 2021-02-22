"""Testing HFO detection algorithms."""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt
from sklearn.utils.estimator_checks import parametrize_with_checks
from mne.io import RawArray
from mne import create_info

from mne_hfo import LineLengthDetector, RMSDetector, \
    HilbertDetector, CSDetector


@parametrize_with_checks([
    RMSDetector(sfreq=2000, win_size=5),
    LineLengthDetector(sfreq=2000, win_size=5),
    # HilbertDetector(sfreq=2000),
])
def test_sklearn_compat(estimator, check):
    """Tests sklearn API compatibility."""
    # in case there are any sklearn checks that need to be ignored
    if check.func.__name__ in [
        # methods should never have only one sample
        # 'check_fit2d_1sample',
        # samples should be ordered wrt
        # 'check_methods_sample_order_invariance',
        # negative window dimension not allowed
        # 'check_methods_subset_invariance',
    ]:
        pytest.skip()

    # skip tests if the number of samples are too low
    try:
        check(estimator)
    except ValueError as e:
        if 'Got data matrix with' in str(e):
            pytest.skip(msg='Skipping sklearn tests with short '
                            'number of features')
        else:
            raise e


def test_detect_hfo_ll(create_testing_eeg_data, benchmark):
    data, hfo_samps = create_testing_eeg_data
    fs = 5000
    b, a = butter(3, [80 / (fs / 2), 600 / (fs / 2)], 'bandpass')
    filt_data = filtfilt(b, a, data)[np.newaxis, :]
    window_size = int((1 / 80) * fs)

    # create input data structure
    info = create_info(sfreq=fs, ch_names=['a'], ch_types='seeg')
    raw = RawArray(filt_data, info=info)

    compute_instance = LineLengthDetector(sfreq=fs, win_size=window_size,
                                          filter_band=None)
    dets = benchmark(compute_instance.fit,
                     raw)

    compute_instance.fit(raw)

    # copied from epycom
    expected_vals = [(5040, 5198),
                     (34992, 35134)]

    # loop over detected events
    for idx, (exp_val) in enumerate(expected_vals):
        assert dets.chs_hfos_list[0][idx][0] == exp_val[0]
        assert dets.chs_hfos_list[0][idx][1] == exp_val[1]


def test_detect_hfo_rms(create_testing_eeg_data, benchmark):
    """Test RMSDetector with simulated HFO.

    Assumes simulated data has already been "bandpass" filtered.
    """
    data, hfo_samps = create_testing_eeg_data
    fs = 5000
    b, a = butter(3, [80 / (fs / 2), 600 / (fs / 2)], 'bandpass')
    filt_data = filtfilt(b, a, data)[np.newaxis, :]
    window_size = int((1 / 80) * fs)

    # create input data structure
    info = create_info(sfreq=fs, ch_names=['a'], ch_types='seeg')
    raw = RawArray(filt_data, info=info)

    compute_instance = RMSDetector(sfreq=fs, win_size=window_size,
                                   filter_band=None)
    dets = benchmark(compute_instance.fit,
                     raw)

    # copied from epycom
    expected_vals = [(5040, 5198),
                     (35008, 35134)]

    # loop over detected events
    for idx, (exp_val) in enumerate(expected_vals):
        assert dets.chs_hfos_list[0][idx][0] == exp_val[0]
        assert dets.chs_hfos_list[0][idx][1] == exp_val[1]


@pytest.mark.skip(reason='need to implement...')
def test_detect_hfo_hilbert(create_testing_eeg_data, benchmark):
    data, hfo_samps = create_testing_eeg_data
    fs = 5000

    compute_instance = HilbertDetector(sfreq=fs)
    compute_instance.params = {'fs': fs,
                               'low_fc': 80,
                               'high_fc': 600,
                               'threshold': 7}
    dets = benchmark(compute_instance.fit,
                     data)

    compute_instance.fit(data)

    expected_vals = [(5056, 5123),
                     (35028, 35063)]

    for exp_val, det in zip(expected_vals, dets):
        assert det[0] == exp_val[0]
        assert det[1] == exp_val[1]


@pytest.mark.skip(reason='need to implement...')
def test_detect_hfo_cs_beta(create_testing_eeg_data, benchmark):
    data, hfo_samps = create_testing_eeg_data
    fs = 5000
    compute_instance = CSDetector()
    compute_instance.params = {'fs': fs,
                               'low_fc': 40,
                               'high_fc': 1000,
                               'threshold': 0.1,
                               'cycs_per_detect': 4.0}

    dets = benchmark(compute_instance.fit,
                     data)

    compute_instance.fit(data)

    # Only the second HFO is caught by CS (due to signal artificiality)
    expected_vals = [(34992, 35090),  # Band detection
                     (34992, 35090)]  # Conglomerate detection
    # [hfo_samps[1], hfo_samps[1]]  #

    for exp_val, det in zip(expected_vals, dets):
        assert det[0] == exp_val[0]
        assert det[1] == exp_val[1]
