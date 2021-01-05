import numpy as np

from mne_hfo.utils import (compute_rms, compute_line_length,
                           threshold_std, threshold_tukey,
                           threshold_quian)


def test_compute_rms(create_testing_data):
    assert (round(np.sum(compute_rms(create_testing_data)), 5) ==
            round(101737.24636480425, 5))


def test_compute_line_length(create_testing_data):
    assert (round(np.sum(compute_line_length(create_testing_data)), 5) ==
            round(58084.721256107114, 5))  # noqa


# ----- Thresholds -----
def test_threshold_std(create_testing_data):
    assert (round(threshold_std(create_testing_data, 3), 5) ==
            round(6.708203932499344, 5))


def test_threshold_tukey(create_testing_data):
    assert (round(threshold_tukey(create_testing_data, 3), 5) ==
            round(10.659619047273361, 5))


def test_threshold_quian(create_testing_data):
    assert (round(threshold_quian(create_testing_data, 3), 5) ==
            round(6.777704219110832, 5))
