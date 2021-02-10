import pytest

from mne_hfo import (
    create_annotations_df, find_coincident_events,
    compute_chs_hfo_rates
)


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


@pytest.mark.parametrize('end_sec', [10,
                                     # 20, 50
                                     ])
def test_metrics_df(end_sec):
    """Test metrics based on reference dataframe and detector ran dataframe."""
    onset = [1.5, 2.0, 3, 5.5]
    duration = [0.1, 1.0, 1.0, 0.1]
    ch_name = ['A1', 'A2', 'A3', 'A1']
    sfreq = 1000

    annot_df = create_annotations_df(onset, duration, ch_name)

    # error will occur without the sample column
    with pytest.raises(RuntimeError, match='Annotations dataframe '
                                           'columns must contain'):
        compute_chs_hfo_rates(annot_df=annot_df)

    # now add sample column
    annot_df['sample'] = annot_df['onset'] * sfreq
    chs_hfo_rates = compute_chs_hfo_rates(annot_df=annot_df, rate='s',
                                          end_sec=end_sec)
    assert chs_hfo_rates['A2'] == chs_hfo_rates['A3']
    assert chs_hfo_rates['A2'] == 1 / end_sec
    print(chs_hfo_rates)
    raise Exception('hi')
