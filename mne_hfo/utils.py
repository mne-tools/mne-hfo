"""Utility and helper functions for MNE-HFO."""
# License: BSD (3-clause)
import json
import os
from os import path as op

import numpy as np
import pandas as pd

from mne_hfo.config import ANNOT_COLUMNS, EVENT_COLUMNS


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
