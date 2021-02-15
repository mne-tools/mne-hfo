"""Testing reading, creating and writing of files."""
from pathlib import Path

import pandas as pd
import pytest
from mne.utils import _TempDir
from mne_bids import BIDSPath, read_raw_bids

from mne_hfo import (create_annotations_df, create_events_df,
                     read_annotations, events_to_annotations,
                     write_annotations)

data_path = Path('data')
subject = '01'
session = 'interictalsleep'
run = '01'
datatype = 'ieeg'
bids_path = BIDSPath(subject=subject, session=session,
                     run=run, datatype=datatype,
                     suffix='ieeg', extension='.vhdr',
                     root=data_path)
events_path = bids_path.copy().update(
    suffix='events', extension='.tsv')


def test_events_to_annotations():
    """Test converting events to annotations DataFrame."""
    # test conversion from events dataframe to annotations dataframe
    events_df = pd.read_csv(events_path, delimiter='\t', index_col=None)
    annot_df = events_to_annotations(events_df)

    expected_cols = ['onset', 'duration', 'label', 'channels']
    assert all([col in annot_df.columns for col in expected_cols])

    # check errors when input event dataframe is missing a column
    with pytest.raises(RuntimeError, match='Passed in events '
                                           'dataframe is missing'):
        events_df.drop('onset', axis=1, inplace=True)
        events_to_annotations(events_df)


@pytest.mark.usefixtures('test_bids_root')
def test_io_annot_df(test_bids_root):
    # create dummy annotations
    onset = [1.5, 2.0, 3]
    duration = [0.0, 0, 1.5]
    ch_name = ['A1', 'A2', 'A3']
    annotation_label = ['ripple', 'frandr', 'fast-ripple']
    annot_df = create_annotations_df(onset, duration, ch_name,
                                     annotation_label)

    annot_path = bids_path.copy().update(root=None,
                                         suffix='annotations', check=False)
    out_fname = Path(test_bids_root) / 'derivatives' / 'sub-01' / annot_path.basename  # noqa

    # save to temporary directory
    write_annotations(annot_df, fname=out_fname,
                      intended_for=bids_path,
                      root=test_bids_root)

    # read them back
    annot_df = read_annotations(fname=out_fname, root=test_bids_root)

    # if you fail to pass in root, it should be inferred correctly
    new_annot_df = read_annotations(fname=out_fname)
    pd.testing.assert_frame_equal(annot_df, new_annot_df)

    # if derivatives is not in the subdirectory of bids dataset,
    # then an error will raise if root is not passed in
    tempdir = _TempDir()
    out_fname = Path(tempdir) / 'derivatives' / 'sub-01' / annot_path.basename  # noqa
    # save to temporary directory
    write_annotations(annot_df, fname=out_fname,
                      intended_for=bids_path,
                      root=test_bids_root)
    with pytest.raises(RuntimeError, match='No raw dataset found'):
        read_annotations(fname=out_fname)


def test_create_events_df():
    # read in test raw file
    raw = read_raw_bids(bids_path)
    sfreq = raw.info['sfreq']

    # create fake HFO event endpoints detected
    input_dict = {
        'A1': [[1500, 2000], ],
        'A2': [[3000, 5000], ],
    }
    events_df = create_events_df(input_dict, sfreq, event_name='hfo')
    assert len(events_df['trial_type'].unique()) == 2
    exp_series = pd.Series([0.25, 1.0], name='duration')
    pd.testing.assert_series_equal(events_df['duration'], exp_series)


def test_create_annot_df():
    onset = [1.5, 2.0, 3]
    duration = [0.0, 0, 1.5]
    ch_name = ['A1', 'A2', 'A3']
    annotation_label = ['ripple', 'frandr', 'fast-ripple']

    # without using annotation label, everything is labeled HFO
    annot_df = create_annotations_df(onset, duration, ch_name)
    assert len(annot_df['label'].unique()) == 1
    assert annot_df['label'][0] == 'HFO'

    # using annotation label, everything is labeled
    annot_df = create_annotations_df(onset, duration, ch_name,
                                     annotation_label)
    assert len(annot_df['label'].unique()) == 3
    assert annot_df['label'][0] == 'ripple'

    # check errors when lengths mismatch
    with pytest.raises(ValueError, match='Length of "onset", "description", '
                                         '"duration", need to be the same.'):
        _onset = onset + [2]
        create_annotations_df(_onset, duration, ch_name,
                              annotation_label)
    with pytest.raises(ValueError, match='Length of "annotation_label" need '
                                         'to be the same as other arguments'):
        create_annotations_df(onset, duration, ch_name,
                              annotation_label[0])

    # check typing mismatch, should be caught by pandas
    onset[0] = 'blah'
    with pytest.raises(ValueError, match='could not convert string to float:'):
        create_annotations_df(onset, duration, ch_name)
