import itertools

import numpy as np
import pandas as pd
import pytest

from mne_hfo import (
    create_annotations_df, find_coincident_events,
    compute_chs_hfo_rates, merge_overlapping_events
)
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
