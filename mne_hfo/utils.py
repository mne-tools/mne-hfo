"""Utility and helper functions for MNE-HFO."""
# License: BSD (3-clause)
import json
import os
from os import path as op

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from tqdm import tqdm

from mne_hfo.config import ANNOT_COLUMNS


def autocorr(x, t=1):
    """Autocorrelation function of a vector."""
    return np.corrcoef(np.array([x[:-t], x[t:]]))


def _check_df(df: pd.DataFrame, df_type: str,
              copy: bool = True) -> pd.DataFrame:
    """Check dataframe for correctness."""
    if df_type == 'annotations':
        if any([col not in df.columns
                for col in ANNOT_COLUMNS + ['sample']]):
            raise RuntimeError(f'Annotations dataframe columns must contain '
                               f'{ANNOT_COLUMNS + ["sample"]}.')

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


def _band_zscore_detect(signal, sfreq, band_idx, l_freq, h_freq, n_times,
                        cycles_threshold, gap_threshold, zscore_threshold):
    """
    Find detections that meet the Hilbert envelope criteria.

    Parameters
    ----------
    signal : np.ndarray
        A single channel's Hilbert transform within a frequency band
    sfreq : float
        Sampling frequency of the data
    band_idx :  int
        The index of the frequency band
    l_freq : float
        The low frequency of the band
    h_freq : float
        The high frequency of the band
    n_times : int
        The number of timepoints used to calculate the signal
    cycles_threshold : float
        The number of cycles to be considered a valid envelope
    gap_threshold : float
        The number of cycles needed to be considered a gap
    zscore_threshold : float
        Value to threshold the signal on

    Returns
    -------
    tdetects : List[Tuple[int, int, int, int]]
        All HFO events that passed the bandpass, zscore. Each tuple contains:
        [0] - The band index
        [1] - The timepoint of the start of a detection
        [2] - The timepoint of the end of the detection
        [3] - Maximum value of the Hilbert envelope in this event window

    """
    # Detections where the envelope has a zscore greater than threshold
    tdetects = []

    # Create boolean mask of signal greater than zscore_threshold
    thresh_sig = np.zeros(n_times, dtype='bool')
    thresh_sig[signal > zscore_threshold] = 1

    idx = 0

    # Find indices where threshold is met
    thresh_idxs = np.where(thresh_sig == 1)[0]

    # Calculate the required samples to be considered a valid gap
    gap_samp = round(gap_threshold * sfreq / l_freq)

    # Iterate over valid indices (significant zscore timepoints)
    while idx < len(thresh_idxs) - 1:
        # Find the start of the envelope, which occurs when back to back
        # time-points meet the threshold
        if (thresh_idxs[idx + 1] - thresh_idxs[idx]) == 1:
            start_idx = thresh_idxs[idx]
            # Find where the envelope ends by iterating over indices
            while idx < len(thresh_idxs) - 1:
                # Check if last index over threshold. If so,
                # consider this index to be the end of the envelope
                if (thresh_idxs[idx + 1] - thresh_idxs[idx]) == 1:
                    idx += 1
                    if idx == len(thresh_idxs) - 1:
                        stop_idx = thresh_idxs[idx]
                        # Check that envelope meets number of cycles criteria
                        dur = (stop_idx - start_idx) / sfreq
                        cycs = l_freq * dur
                        if cycs > cycles_threshold:
                            # Valid, so append to detections
                            tdetects.append([band_idx, start_idx, stop_idx,
                                             max(signal[start_idx:stop_idx]),
                                             [l_freq, h_freq]])
                else:
                    # If there is no gap between this and the next index,
                    # it is still part of this envelope. Increment the
                    # index.
                    if (thresh_idxs[idx + 1] - (thresh_idxs[idx])) < gap_samp:
                        idx += 1
                    # If the next index has a gap, the current index is the
                    # end of the envelope
                    else:
                        stop_idx = thresh_idxs[idx]
                        # Check that envelope meets number of cycles criteria
                        dur = (stop_idx - start_idx) / sfreq
                        cycs = l_freq * dur
                        if cycs > cycles_threshold:
                            # Valid, so append to detections
                            tdetects.append([band_idx, start_idx, stop_idx,
                                             max(signal[start_idx:stop_idx]),
                                             [l_freq, h_freq]])
                        idx += 1
                        break
        else:
            idx += 1
    return tdetects


def compute_rms(signal, win_size=6):
    """
    Calculate the Root Mean Square (RMS) energy.

    Parameters
    ----------
    signal : numpy array
        1D signal to be transformed.
    win_size : int
        Number of the points of the window (default=6).

    Returns
    -------
    rms: np.ndarray
        Root mean square transformed signal.
    """
    aux = np.power(signal, 2)
    window = np.ones(win_size) / float(win_size)
    return np.sqrt(np.convolve(aux, window, 'same'))


def compute_line_length(signal, win_size=6):
    """Calculate line length.

    Parameters
    ----------
    signal : numpy array
        1D signal to be transformed.
    win_size : int
        Number of the points of the window (default=6).

    Returns
    -------
    line_length: numpy array
        Line length transformed signal.

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


def compute_hilbert(signal, freq_cutoffs, freq_span, sfreq):
    """Compute the Hilbert envelope for a single channel.

    Parameters
    ----------
    signal : np.ndarray
        EEG signal for a single channel.
    freq_cutoffs : tuple
        The lower and higher frequency cutoff.
    freq_span : tuple
        The span of how many frequencies there are.
    sfreq : float
        The sampling rate.

    Returns
    -------
    hfx_bands : np.ndarray
        Hilbert transforms per freq band.
    """
    hfx_bands = []
    # Iterate over freq bands
    for ind in range(freq_span):
        l_freq = freq_cutoffs[ind]
        h_freq = freq_cutoffs[ind + 1]

        # Filter the data for this frequency band
        signal = mne.filter.filter_data(signal, sfreq=sfreq,
                                        l_freq=l_freq, h_freq=h_freq,
                                        method='iir', verbose=False)
        # compute z-score of data
        signal = (signal - np.mean(signal)) / np.std(signal)

        # Chunk the signal into 30 second windows and compute the Hilbert
        # to save memory
        hfx = np.empty(signal.shape)
        n_times = len(hfx)
        win_size = int(sfreq * 30)
        n_wins = int(np.ceil(n_times / win_size))
        for win in range(n_wins):
            start_samp = win * win_size
            end_samp = (win + 1) * win_size
            if win == n_wins:
                end_samp = n_times
            sig = signal[start_samp:end_samp]
            hfx[start_samp:end_samp] = np.abs(hilbert(sig))

        # return the absolute value of the Hilbert transform.
        # (i.e. the envelope)
        hfx_bands.append(hfx)
        hfx = None
    return hfx_bands


def apply_hilbert(metric, threshold_dict, kwargs):
    """Apply the Hilbert z-score thresholding scheme.

    Parameters
    ----------
    metric : np.ndarray
        The values to apply the threshold rules to.
    threshold_dict : dict
        Dictionary of  threshold parameters to apply to metric.
        Must have zscore, gap, and cycles keys.
    kwargs : dict
        Additional model parameters needed to apply hilbert threshold.
        Must have n_times, sfreq, filter_band, freq_cutoffs,
        freq_span, and n_jobs.

    Returns
    -------
    tdetects : List of Tuple
        Detected hfo events with the structure [band_idx, start,
        stop, max_amplitude, freq_band].
    """
    # get threshold vals
    zscore_threshold = threshold_dict["zscore"]
    gap_threshold = threshold_dict["gap"]
    cycles_threshold = threshold_dict["cycles"]
    if any(elem is None for elem in [zscore_threshold, gap_threshold,
                                     cycles_threshold]):
        raise RuntimeError(f"threshold_dict must have values for zscore,"
                           f" gap, and cycles. You passed {threshold_dict}")
    n_times = kwargs["n_times"]
    sfreq = kwargs["sfreq"]
    filter_band = kwargs["filter_band"]
    freq_cutoffs = kwargs["freq_cutoffs"]
    freq_span = kwargs["freq_span"]
    n_jobs = kwargs["n_jobs"]
    if any(elem is None for elem in [n_times, sfreq, filter_band,
                                     freq_cutoffs, freq_span, n_jobs]):
        raise RuntimeError(f"kwargs must have values for n_times, sfreq,"
                           f" filter_band, freq_cutoffs, freq_span, n_jobs."
                           f" You passed {kwargs}")

    tdetects = []
    for i in tqdm(range(freq_span), unit="HFO-first-phase"):
        # Find bottom and top of the frequency band
        bot = freq_cutoffs[i]
        top = freq_cutoffs[i + 1]
        # Make sure you only look at Hilbert envelope values
        # for the specific freq band
        tdetects.append(_band_zscore_detect(metric[i], sfreq, i, bot,
                                            top, n_times, cycles_threshold,
                                            gap_threshold,
                                            zscore_threshold))
    return tdetects


def apply_std(metric, threshold_dict, kwargs):
    """Calculate and apply the threshold based on number of standard deviations.

    Parameters
    ----------
    metric : np.ndarray
        Values to apply the threshold to.
    threshold_dict : dict
        Dictionary of threshold values. Should just have thresh,
        which is the number of standard deviations to check against.
    kwargs : dict
        Additional key-word args from the detector needed to
        apply the threshold.
        Step_size, win_size, and n_times are required keys.

    Returns
    -------
    output: List of tuple
        List of detected events that pass the threshold.
    """
    # determine threshold value
    threshold = threshold_dict["thresh"]
    if threshold is None:
        raise RuntimeError(f"threshold_dict must have a value for 'thresh'."
                           f" You passed {threshold_dict}")
    det_th = _get_threshold_std(metric, threshold)

    n_windows = len(metric)
    step_size = kwargs["step_size"]
    win_size = kwargs["win_size"]
    n_times = kwargs["n_times"]
    if any(elem is None for elem in [step_size, win_size, n_times]):
        raise RuntimeError(f"kwargs must have step_size, win_size, "
                           f"and n_times. You passed {kwargs}")

    # store thresholded hfo events as a list
    output = []
    # Detect and now group events if they are within a
    # step size of each other
    win_idx = 0
    while win_idx < n_windows:
        # log events if they pass our threshold criterion
        if metric[win_idx] >= det_th:
            event_start = win_idx * step_size

            # group events together if they occur in
            # contiguous windows
            # TODO: We could factor this out into an independent step,
            #  but that will just add comp time
            while win_idx < n_windows and \
                    metric[win_idx] >= det_th:
                win_idx += 1
            event_stop = (win_idx * step_size) + win_size

            if event_stop > n_times:
                event_stop = n_times

            # TODO: Optional feature calculations

            # Write into output
            output.append((event_start, event_stop))
            win_idx += 1
        else:
            win_idx += 1

    return output


def _get_threshold_std(signal, threshold):
    """
    Calculate threshold by Standard Deviations above the mean.

    Parameters
    ----------
    signal: numpy array
        1D signal for threshold determination
    threshold: int
        Number of standard deviations to consider.

    Returns
    -------
    ths_value: float
        Value of the threshold

    """
    ths_value = np.mean(signal) + threshold * np.std(signal)
    return ths_value


def merge_contiguous_freq_bands(detections):
    """Merge detected events in contiguous freq bands and time windows.

    Parameters
    ----------
    detections : List(tuple)
        List of detections, which have the form [band_idx, start,
         stop, max_amplitude, freq_band]

    Returns
    -------
    hfo_events: List(tuple)
        List of distinct hfo events, which have the form [start, stop]
    max_hilbert: List(int)
        List of max values in each event
    freq_bands: List(tuple)
        List of the freq_band for each event

    """
    from mne_hfo.posthoc import _check_detection_overlap
    outlines = []
    for detection in detections[0]:
        band_idx = detection[0]
        # If first freq band, always unique so append
        if band_idx == 0:
            outlines.append(detection)
        else:
            for ind, outline in enumerate(outlines):
                # only try to merge contiguous freq bands
                if outline[0] == band_idx + 1:
                    # Check if the events overlap in time
                    if _check_detection_overlap([detection[1], detection[2]],
                                                [outline[1], outline[2]]):
                        # merge the overlapping events
                        outlines[ind] = _merge_outline(outlines, detection)
                    else:
                        # Events dont overlap so append it
                        outlines.append(detection)
                else:
                    # Events are contiguous so append it
                    outlines.append(detection)
    # extract start and stop times
    hfo_events = [[o[1], o[2]] for o in outlines]
    max_hilbert = [o[3] for o in outlines]
    freq_bands = [[o[4][0], o[4][1]] for o in outlines]
    return hfo_events, max_hilbert, freq_bands


def _merge_outline(outline, detection):
    band_idx = detection[0]
    start = min(outline[1], detection[1])
    stop = max(outline[2], detection[2])
    max_frq = max(outline[3], detection[3])
    freq_band = [outline[4][0], detection[4][1]]
    return [band_idx, start, stop, max_frq, freq_band]


def threshold_tukey(signal, threshold):
    """
    Calculate threshold by Tukey method.

    Parameters
    ----------
    signal : numpy array
        1D signal for threshold determination.
    threshold : float
        Number of interquartile interval above the 75th percentile.

    Returns
    -------
    ths_value : float
        Value of the threshold.

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
