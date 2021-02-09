import numpy as np
import pytest

from mne_hfo.utils import (compute_rms, compute_line_length,
                           threshold_std, threshold_tukey,
                           threshold_quian, find_coincident_events)


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


def test_find_coincident_events(create_testing_events_dicts):
    df1, df2 = create_testing_events_dicts
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
