from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _convert_y_sklearn_to_annot_df(ylist):
    """Convert y sklearn list to Annotations DataFrame."""
    from .io import create_annotations_df

    # store basic data points needed for annotations dataframe
    onset_sec = []
    duration_sec = []
    ch_names = []
    labels = []
    sfreqs = []

    # loop over all channel HFO results
    for idx, ch_results in enumerate(ylist):
        # sklearn is returning a single HFO with onset and duration of 0
        for jdx, res in enumerate(ch_results):
            onset, offset, ch_name, label, sfreq = res

            # if onset/offset is None, then there is
            # on HFO for this channel
            if onset is not None:
                if (sfreq is not None) and \
                        (not np.isnan(np.array([sfreq], dtype=np.float64))):
                    # Sampling frequencies should always be integers
                    # Solves issues with unique check due to float
                    # division
                    sfreq = int(np.round(sfreq))
                    sfreqs.append(sfreq)
                    onset_sec.append(onset)
                    duration_sec.append(offset - onset)

                    ch_names.append(ch_name)
                    labels.append(label)
    # If no hfos detected, return an empty annotation df
    if not sfreqs:
        empty_annotation_df = pd.DataFrame(
            columns=['onset', 'duration', 'channels', 'label',
                     'sample'])
        return empty_annotation_df
    # If hfos are detected, assert they all have the same frq
    assert len(np.unique(sfreqs)) == 1
    sfreq = sfreqs[0]

    # create the output annotations dataframe
    annot_df = create_annotations_df(onset=onset_sec, duration=duration_sec,
                                     ch_name=ch_names, annotation_label=labels)
    annot_df['sample'] = annot_df['onset'].multiply(sfreq)
    return annot_df


def make_Xy_sklearn(raw, df):
    """Make X/y for HFO detector compliant with scikit-learn.

    To render a dataframe "sklearn" compatible, by
    turning it into a list of list of tuples.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data.
    df : pd.DataFrame
        The HFO labeled dataframe, in the form of ``*_annotations.tsv``.
        Should be read in through ``read_annotations`` function.

    Returns
    -------
    raw_df : pd.DataFrame
        The Raw dataframe generated from :meth:`mne.io.Raw.to_data_frame`.
        It should be structured as channels X time.
    ch_results : list[list[tuple]]
        List of channel HFO events, ordered by the channel names from the
        ``raw`` dataset. Each channel corresponds to a list of "onset"
        and "offset" time points (in seconds) that an HFO was detected.
    """
    raw.to_data_frame
    ch_names = raw.ch_names

    ch_results = _make_ydf_sklearn(df, ch_names)

    # set arbitrary measurement date to allow time format as a datetime
    if raw.info['meas_date'] is None:
        raw.set_meas_date(datetime.now(tz=timezone.utc))

    # keep as C x T
    raw_df = raw.to_data_frame(index='time',
                               time_format='datetime').T

    return raw_df, ch_results


def _make_ydf_sklearn(ydf, ch_names):
    """Convert HFO annotations DataFrame into scikit-learn y input.

    Parameters
    ----------
    ydf : pd.Dataframe
        Annotations DataFrame containing HFO events.
    ch_names : list
        A list of channel names in the raw data.

    Returns
    -------
    ch_results : List of list[tuple]
        Ordered dictionary of channel HFO events, ordered by the channel
        names from the ``raw`` dataset. Each channel corresponds to a
        list of "onset" and "offset" time points (in seconds) that an
        HFO was detected. The channel is also appended to the third
        element of each HFO event. For example::

            # ch_results has length of ch_names
            ch_results = [
                [
                    (0, 10, 'A1'),
                    (20, 30, 'A1'),
                    ...
                ],
                [
                    (None, None, 'A2'),
                ],
                [
                    (20, 30, 'A3'),
                    ...
                ],
                ...
            ]
    """
    # create channel results
    ch_results = []

    # make sure offset in column
    if 'offset' not in ydf.columns:
        ydf['offset'] = ydf['onset'] + ydf['duration']

    ch_groups = ydf.groupby(['channels'])
    if any([ch not in ch_names for ch in ch_groups.groups]):  # type: ignore
        raise RuntimeError(f'Channel {ch_groups.groups} contain '
                           f'channels not in '
                           f'actual data channel names: '
                           f'{ch_names}.')

    # group by channels
    for idx, ch in enumerate(ch_names):
        if ch not in ch_groups.groups.keys():
            ch_results.append([(None, None, ch, None, None)])
            continue
        # get channel name
        ch_df = ch_groups.get_group(ch)

        # obtain list of HFO onset, offset for this channel
        ch_name_as_list = [ch] * len(ch_df['onset'])
        sfreqs = ch_df['sample'].divide(ch_df['onset'])
        ch_results.append(list(zip(ch_df['onset'],
                                   ch_df['offset'],
                                   ch_name_as_list,
                                   ch_df['label'],
                                   sfreqs)))

    ch_results = np.asarray(ch_results, dtype='object')
    return ch_results


class DisabledCV:
    """Dummy CV class for SearchCV scikit-learn functions."""

    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        """Disabled split."""
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        """Disabled split."""
        return self.n_splits
