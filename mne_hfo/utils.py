"""Utility and helper functions for MNE-HFO."""
# License: BSD (3-clause)
import json
import os
from datetime import datetime, timezone
from os import path as op

import numpy as np
import pandas as pd

from mne_hfo.config import ANNOT_COLUMNS, EVENT_COLUMNS


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
        if ch not in ch_groups.groups:
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
        for jdx, res in enumerate(ch_results):
            onset, offset, ch_name, label, sfreq = res

            # if onset/offset is None, then there is
            # on HFO for this channel
            if onset is not None:
                onset_sec.append(onset)
                duration_sec.append(offset - onset)
                sfreqs.append(sfreq)
                ch_names.append(ch_name)
                labels.append(label)

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
        The Raw dataframe generated from :func:`mne.io.Raw.to_data_frame`.
        It should be structured as channels X time.
    ch_results : list[list[tuple]]
        List of channel HFO events, ordered by the channel names from the
        ``raw`` dataset. Each channel corresponds to a list of "onset"
        and "offset" time points (in seconds) that an HFO was detected.
    """
    ch_names = raw.ch_names

    ch_results = _make_ydf_sklearn(df, ch_names)

    # set arbitrary measurement date to allow time format as a datetime
    if raw.info['meas_date'] is None:
        raw.set_meas_date(datetime.now(tz=timezone.utc))

    # keep as C x T
    raw_df = raw.to_data_frame(index='time',
                               time_format='datetime').T

    return raw_df, ch_results


def _check_df(df: pd.DataFrame, df_type: str,
              copy: bool = True) -> pd.DataFrame:
    """Check dataframe for correctness."""
    if df_type == 'annotations':
        if any([col not in df.columns
                for col in ANNOT_COLUMNS + ['sample']]):
            raise RuntimeError(f'Annotations dataframe columns must contain '
                               f'{ANNOT_COLUMNS + ["sample"]}.')
    elif df_type == 'events':
        if any([col not in df.columns
                for col in EVENT_COLUMNS + ['sample']]):
            raise RuntimeError(f'Events dataframe columns must contain '
                               f'{EVENT_COLUMNS}.')

    # Only want to do this check if there are multiple rows. Handles edge case
    # of 1 HFO starting at 0. TODO: handle this more elegantly
    if df.shape[0] > 1:
        # first compute sampling rate from sample / onset columns
        sfreq = df['sample'].divide(df['onset']).round(2)

        # onset=0 will cause sfreq to be inf, drop these rows to
        # prevent additional sfreqs
        sfreq = sfreq.replace([np.inf, -np.inf], np.nan).dropna()
        if sfreq.nunique() != 1:
            raise ValueError(f'All rows in the annotations dataframe '
                             f'should have the same sampling rate. '
                             f'Found {sfreq.nunique()} different '
                             f'sampling rates.')

    if copy:
        return df.copy()

    return df


def _ensure_tuple(x):
    """Return a tuple."""
    if x is None:
        return tuple()
    elif isinstance(x, str):
        return (x,)
    else:
        return tuple(x)


def _check_types(variables):
    """Make sure all vars are str or None."""
    for var in variables:
        if not isinstance(var, (str, type(None))):
            raise ValueError(f"You supplied a value ({var}) of type "
                             f"{type(var)}, where a string or None was "
                             f"expected.")


def _write_json(fname, dictionary, overwrite=False, verbose=False):
    """Write JSON to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError(f'"{fname}" already exists. '
                              'Please set overwrite to True.')

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, 'w', encoding='utf-8') as fid:
        fid.write(json_output)
        fid.write('\n')

    if verbose is True:
        print(os.linesep + f"Writing '{fname}'..." + os.linesep)
        print(json_output)


def compute_rms(signal, win_size: int = 6):
    """
    Calculate the Root Mean Square (RMS) energy.

    Parameters
    ----------
    signal: numpy array
        1D signal to be transformed
    win_size: int
        Number of the points of the window (default=6)

    Returns
    -------
    rms: numpy array
        Root mean square transformed signal
    """
    aux = np.power(signal, 2)
    window = np.ones(win_size) / float(win_size)
    return np.sqrt(np.convolve(aux, window, 'same'))


def compute_line_length(signal, win_size=6):
    """Calculate line length.

    Parameters
    ----------
    signal: numpy array
        1D signal to be transformed
    win_size: int
        Number of the points of the window (default=6)

    Returns
    -------
    line_length: numpy array
        Line length transformed signal

    Notes
    -----
    ::

        return np.mean(np.abs(np.diff(data, axis=-1)), axis=-1)

    References
    ----------
    .. [1] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. In Engineering in Medicine and Biology
           Society, 2001. Proceedings of the 23rd Annual International
           Conference of the IEEE (Vol. 2, pp. 1707-1710). IEEE.

    .. [2] Dümpelmann et al, 2012.  Clinical Neurophysiology: 123 (9): 1721-31.
    """
    aux = np.abs(np.subtract(signal[1:], signal[:-1]))
    window = np.ones(win_size) / float(win_size)
    data = np.convolve(aux, window)
    start = int(np.floor(win_size / 2))
    stop = int(np.ceil(win_size / 2))
    return data[start:-stop]


def threshold_std(signal, threshold):
    """
    Calculate threshold by Standard Deviations above the mean.

    Parameters
    ----------
    signal: numpy array
        1D signal for threshold determination
    threshold: float
        Number of SD above the mean

    Returns
    -------
    ths_value: float
        Value of the threshold

    """
    ths_value = np.mean(signal) + threshold * np.std(signal)
    return ths_value


def threshold_tukey(signal, threshold):
    """
    Calculate threshold by Tukey method.

    Parameters
    ----------
    signal: numpy array
        1D signal for threshold determination
    threshold: float
        Number of interquartile interval above the 75th percentile

    Returns
    -------
    ths_value: float
        Value of the threshold

    References
    ----------
    [1] TUKEY JW. Comparing individual means in the analysis of
        variance. Biometrics. 1949 Jun;5(2):99-114. PMID: 18151955.
    """
    ths_value = np.percentile(signal, 75) + threshold * (np.percentile(signal, 75) - np.percentile(signal, 25))  # noqa
    return ths_value


def threshold_quian(signal, threshold):
    """
    Calculate threshold by Quian.

    Parameters
    ----------
    signal: numpy array
        1D signal for threshold determination
    threshold: float
        Number of estimated noise SD above the mean

    Returns
    -------
    ths_value: float
        Value of the threshold

    References
    ----------
    1. Quian Quiroga, R. 2004. Neural Computation 16: 1661–87.
    """
    ths_value = threshold * np.median(np.abs(signal)) / 0.6745
    return ths_value
