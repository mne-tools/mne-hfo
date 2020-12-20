"""Utility and helper functions for MNE-BIDS."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import json
import os
from os import path as op

import numpy as np


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


def compute_line_length(signal, window_size=6):
    """Calculate line length.

    Parameters
    ----------
    signal: numpy array
        1D signal to be transformed
    window_size: int
        Number of the points of the window (default=6)

    Returns
    -------
    line_length: numpy array
        Line length transformed signal

    Notes
    -----

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
    window = np.ones(window_size) / float(window_size)
    data = np.convolve(aux, window)
    start = int(np.floor(window_size / 2))
    stop = int(np.ceil(window_size / 2))
    return data[start:-stop]


def threshold_std(signal, threshold):
    """
    Calculate threshold by Standard Deviations above the mean

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
    1. TUKEY JW. Comparing individual means in the analysis of variance. Biometrics. 1949 Jun;5(2):99-114. PMID: 18151955.
    """
    ths_value = np.percentile(
        signal, 75) + threshold * (np.percentile(signal, 75)
                                   - np.percentile(signal, 25))
    return ths_value


def threshold_quian(signal, threshold):
    """
    Calcule threshold by Quian

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


def match_detections(gs_df, dd_df, bn, freq_name=None,
                     sec_unit=None, sec_margin=1):
    """
    Matches gold standard detections with detector detections.

    Parameters
    ----------
    gs_df: pandas.DataFrame
        Gold standard detections
    dd_df: pandas.DataFrame
        Detector detections
    bn: list
        Names of event start stop [start_name, stop_name]
    freq_name: str
        Name of frequency column
    sec_unit: int
        Number representing one second of signal - this can
        significantly imporove the speed of this function
    sec_margin: int
        Margin for creating subsets of compared data - should be set according
        to the legnth of compared events (1s for HFO should be enough)

    Returns
    -------
    match_df: pandas.DataFrame
        Dataframe with matched indeces (pandas DataFrame)
    """

    match_df = pd.DataFrame(columns=('gs_index', 'dd_index'))
    match_df_idx = 0
    for row_gs in gs_df.iterrows():
        matched_idcs = []
        gs = [row_gs[1][bn[0]], row_gs[1][bn[1]]]
        if sec_unit:  # We can create subset - significant speed improvement
            for row_dd in dd_df[(dd_df[bn[0]] < gs[0]
                                 + sec_unit * sec_margin) &
                                (dd_df[bn[0]] > gs[0]
                                 - sec_unit * sec_margin)].iterrows():
                dd = [row_dd[1][bn[0]], row_dd[1][bn[1]]]
                if check_detection_overlap(gs, dd):
                    matched_idcs.append(row_dd[0])
        else:
            for row_dd in dd_df.iterrows():
                dd = [row_dd[1][bn[0]], row_dd[1][bn[1]]]
                if check_detection_overlap(gs, dd):
                    matched_idcs.append(row_dd[0])

        if len(matched_idcs) == 0:
            match_df.loc[match_df_idx] = [row_gs[0], None]
        elif len(matched_idcs) == 1:
            match_df.loc[match_df_idx] = [row_gs[0], matched_idcs[0]]
        else:
            # In rare event of multiple overlaps get the closest frequency
            if freq_name:
                dd_idx = (
                    abs(dd_df.loc[matched_idcs, freq_name]
                        - row_gs[1][freq_name])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]
            # Closest event start - less precision than frequency
            else:
                dd_idx = (
                    abs(dd_df.loc[matched_idcs, bn[0]]
                        - row_gs[1][bn[0]])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]

        match_df_idx += 1

    return match_df


def check_detection_overlap(gs, dd):
    """
    Evaluates if two detections overlap

    Paramters
    ---------
    gs: list
        Gold standard detection [start,stop]
    dd: list
        Detector detection [start,stop]

    Returns
    -------
    overlap: bool
        Whether two events overlap.

    """

    overlap = False

    # dd stop in gs + (dd inside gs)
    if (dd[1] >= gs[0]) and (dd[1] <= gs[1]):
        overlap = True
    # dd start in gs + (dd inside gs)
    if (dd[0] >= gs[0]) and (dd[0] <= gs[1]):
        overlap = True
    # gs inside dd
    if (dd[0] <= gs[0]) and (dd[1] >= gs[1]):
        overlap = True

    return overlap
