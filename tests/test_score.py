from mne_hfo import create_events_df
from mne_hfo.scores import accuracy, true_positive_rate, \
    false_negative_rate, false_discovery_rate, precision


def test_match_detections():
    # First create two event dataframes with expected columns. We will
    # consider df1 to be ground truth and df2 to be the prediction
    onset1 = [0, 7300, 12600, 22342, 59900]
    offset1 = [67300, 8500, 14870, 31100, 81200]
    event_list = [(onset, offset) for onset, offset in zip(onset1, offset1)]
    event_dict1 = {'A1':event_list}
    sfreq = 1000
    event_df1 = create_events_df(event_dict1, event_name='hfo', sfreq=sfreq)

    onset2 = [2000, 12300, 45800, 98300]
    offset2 = [6930, 15120, 65600, 101450]
    event_list = [(onset, offset) for onset, offset in zip(onset2, offset2)]
    event_dict2 = {'A1':event_list}
    event_df2 = create_events_df(event_dict2, event_name='hfo', sfreq=sfreq)

    # In the above example, we have 3 true positives, 2 false negatives,
    # and 1 false positive Therefore, we expect accuracy = 0.5, tpr = 0.6,
    # fnr = 0.4, fdr = 0.25, and precision = 0.75

    acc = accuracy(event_df1, event_df2)
    assert acc == 0.5

    tpr = true_positive_rate(event_df1, event_df2)
    assert tpr == 0.6

    fnr = false_negative_rate(event_df1, event_df2)
    assert fnr == 0.4

    fdr = false_discovery_rate(event_df1, event_df2)
    assert fdr == 0.25

    prec = precision(event_df1, event_df2)
    assert prec == 0.75
