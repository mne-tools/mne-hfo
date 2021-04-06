import numpy as np

from mne_hfo.utils import (compute_rms, compute_line_length,
                           threshold_tukey, threshold_quian,
                           _get_threshold_std, _band_zscore_detect)


def test_compute_rms(create_testing_data):
    assert (round(np.sum(compute_rms(create_testing_data)), 5) ==
            round(101737.24636480425, 5))


def test_compute_line_length(create_testing_data):
    assert (round(np.sum(compute_line_length(create_testing_data)), 5) ==
            round(58084.721256107114, 5))  # noqa


# ----- Thresholds -----
def test_threshold_std(create_testing_data):
    assert (round(_get_threshold_std(create_testing_data, 3), 5) ==
            round(6.708203932499344, 5))


def test_threshold_tukey(create_testing_data):
    assert (round(threshold_tukey(create_testing_data, 3), 5) ==
            round(10.659619047273361, 5))


def test_threshold_quian(create_testing_data):
    assert (round(threshold_quian(create_testing_data, 3), 5) ==
            round(6.777704219110832, 5))


def test_band_z_score_detect(create_testing_zscore_data):
    X, hfo_outline = create_testing_zscore_data
    fs = 2000
    band_idx = 0
    l_freq = 81
    h_freq = 82
    dur = 30
    n_times = fs * dur
    cycles_thresh = 1
    gap_thresh = 1
    zscore_thresh = 3
    hfo_detect = _band_zscore_detect(X, fs, band_idx, l_freq, h_freq, n_times,
                                     cycles_thresh, gap_thresh, zscore_thresh)
    assert hfo_detect[0][1] == hfo_outline[0]
    assert hfo_detect[0][2] == hfo_outline[1]
