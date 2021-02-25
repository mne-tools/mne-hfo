import collections
from typing import Tuple, Union

import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count
from mne.utils import warn
from scipy.signal import hilbert
from tqdm import tqdm

from mne_hfo.base import Detector
from mne_hfo.config import ACCEPTED_BAND_METHODS
from mne_hfo.posthoc import _check_detection_overlap


def _band_z_score_detect(x_cond, sfreq, band_idx, l_freq, h_freq,
                         cycle_threshold, gap_threshold, threshold):
    """Iterate bandpass and z-score a channel's signal.

    Creates a bandpass filter (order=3) between ``l_freq`` and ``h_freq``.
    Then it z-scores the data

    Parameters
    ----------
    x_cond :
    sfreq :
    band_idx :
    l_freq :
    h_freq :
    cycle_threshold :
    gap_threshold :
    threshold :

    Returns
    -------
    tdetects : list of tuples
        All HFO events that passed the bandpass, zscore. It will store
        the band index (``band_idx``), timepoint of start, endpoint of start,
        and the maximum value of the Hilbert envelope in this event window.
    """
    tdetects = []

    # create highpass and lowpass filters
    x_cond = mne.filter.filter_data(
        x_cond, sfreq=sfreq,
        l_freq=l_freq, h_freq=h_freq, method='iir')
    # b, a = butter(N=3, Wn=l_freq / (sfreq / 2), btype='highpass')
    # x_cond = filtfilt(b, a, x_cond)
    # b, a = butter(N=3, Wn=h_freq / (sfreq / 2), btype='lowpass')
    # x_cond = filtfilt(b, a, x_cond)

    # Compute the z-scores
    x_cond = (x_cond - np.mean(x_cond)) / np.std(x_cond)

    # compute the absolute value of the Hilbert transform
    # (i.e. the envelope)
    hfx = np.abs(hilbert(x_cond))

    # threshold the Hilbert envelope to create a
    # thresholded mask
    thresh_sig = np.zeros(len(x_cond), dtype='bool')
    thresh_sig[hfx > threshold] = 1

    # Now get the lengths of each detected HFO
    idx = 0

    # indices where the threshold was met
    thresh_idxs = np.where(thresh_sig == 1)[0]
    gap_samp = round(gap_threshold * sfreq / l_freq)

    # loop through all significant zscore time points
    while idx < len(thresh_idxs) - 1:
        if (thresh_idxs[idx + 1] - thresh_idxs[idx]) == 1:
            start_idx = thresh_idxs[idx]
            while idx < len(thresh_idxs) - 1:
                if (thresh_idxs[idx + 1] - thresh_idxs[idx]) == 1:
                    idx += 1  # Move to the end of the detection
                    if idx == len(thresh_idxs) - 1:
                        stop_idx = thresh_idxs[idx]
                        # Check for number of cycles
                        dur = (stop_idx - start_idx) / sfreq
                        cycs = l_freq * dur
                        if cycs > cycle_threshold:
                            # Carry the amplitude and frequency info
                            tdetects.append([band_idx, start_idx, stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                else:  # Check for gap
                    if (thresh_idxs[idx + 1] - thresh_idxs[idx]) < gap_samp:
                        idx += 1
                    else:
                        stop_idx = thresh_idxs[idx]
                        # Check for number of cycles
                        dur = (stop_idx - start_idx) / sfreq
                        cycs = l_freq * dur
                        if cycs > cycle_threshold:
                            tdetects.append([band_idx, start_idx, stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                        idx += 1
                        break
        else:
            idx += 1

    return tdetects


def _run_detect_branch(detects, det_idx, HFO_outline):
    """
    Process detections from hilbert detector.

    HFO_outline structure:
    [0] - bands in which the detection happened
    [1] - starts for each band
    [2] - stop for each band
    [3] -
    """
    HFO_outline.append(np.copy(detects[det_idx, :]))

    # Create a subset for next band
    next_band_idcs = np.where(detects[:, 0] == detects[det_idx, 0] + 1)
    if not len((next_band_idcs)[0]):
        # No detects in band - finish the branch
        detects[det_idx, 0] = 0  # Set the processed detect to zero
        return HFO_outline
    else:
        # Get overllaping detects
        for next_det_idx in next_band_idcs[0]:
            if _check_detection_overlap([detects[det_idx, 1], detects[det_idx,
                                                                      2]],
                                        [detects[next_det_idx, 1],
                                         detects[next_det_idx,
                                                 2]]):
                # Go up the tree
                _run_detect_branch(detects, next_det_idx, HFO_outline)

        detects[det_idx, 0] = 0
        return HFO_outline


class HilbertDetector(Detector):  # noqa
    """2D HFO hilbert detection used in Kucewicz et al. 2014.

    A multi-taper method with: 4 Hz bandwidth, 1 sec sliding window,
    stepsize 100 ms, for the 1-500 Hz range, no padding, 2 tapers.

    Parameters
    ----------
    sfreq: float
        Sampling frequency of the signal
    l_freq: float
        Low cut-off frequency
    h_freq: float
        High cut-off frequency
    threshold: float
        Threshold for detection (default=3)
    band_method: str
        Spacing of hilbert frequency bands - options: 'linear' or 'log'
        (default='linear'). Linear provides better frequency resolution but
        is slower.
    n_bands: int
        Number of bands if band_spacing = log (default=300)
    cycle_threshold: float
        Minimum number of cycles to detect (default=1)
    gap_threshold: float
        Number of cycles for gaps (default=1)
    n_jobs: int
        Number of cores to use (default=1)
    offset: int
        Offset which is added to the final detection. This is used when the
        function is run in separate windows. Default = 0

    References
    ----------
    [1] M. T. Kucewicz, J. Cimbalnik, J. Y. Matsumoto, B. H. Brinkmann,
    M. Bower, V. Vasoli, V. Sulc, F. Meyer, W. R. Marsh, S. M. Stead, and
    G. A. Worrell, “High frequency oscillations are associated with
    cognitive processing in human recognition memory.,” Brain, pp. 1–14,
    Jun. 2014.
    """

    def __init__(self, sfreq: float, l_freq: float, h_freq: float,
                 threshold: float = 3, band_method: str = 'linear',
                 n_bands: int = 300, cycle_threshold: float = 1,
                 gap_threshold: float = 1, n_jobs: int = 1,
                 offset: int = 0, scoring_func: str = 'f1',
                 verbose: bool = False):
        if band_method not in ACCEPTED_BAND_METHODS:
            raise ValueError(f'Band method {band_method} is not '
                             f'an acceptable parameter. Please use '
                             f'one of {ACCEPTED_BAND_METHODS}')

        super(HilbertDetector, self).__init__(
            threshold, win_size=None, overlap=None,
            scoring_func=scoring_func, n_jobs=n_jobs, verbose=verbose)

        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.band_method = band_method
        self.n_bands = n_bands
        self.cycle_threshold = cycle_threshold
        self.gap_threshold = gap_threshold
        self.n_jobs = n_jobs
        self.offset = offset

    def fit(self, X):
        """Override ``Detector.fit`` function."""
        n_chs, n_times = X.shape

        # store all hfo occurrences as an array of channels X windows
        n_windows = int(np.ceil((n_times - self.win_size) /
                                self.step_size)) + 1
        hfo_event_arr = np.empty((n_chs, n_windows))

        # store contiguous hfos as one occurrence, so we store
        # them as a dictionary of lists
        chs_hfos = collections.defaultdict(list)

        # bandpass the signal using FIR filter
        # X = mne.filter.filter_data(X, sfreq=self.sfreq,
        #                            l_freq=self.l_freq,
        #                            h_freq=self.h_freq, picks=picks,
        #                            method='iir', verbose=self.verbose)

        # Construct filter cut offs that are either logarithmically
        # or linearly spaced
        if self.band_method == 'log':
            low_fc = float(self.l_freq)
            high_fc = float(self.h_freq)
            freq_cutoffs = np.logspace(0, np.log10(high_fc), self.n_bands)
            freq_cutoffs = freq_cutoffs[(freq_cutoffs > low_fc) &
                                        (freq_cutoffs < high_fc)]
            freq_span = len(freq_cutoffs) - 1
        elif self.band_method == 'linear':
            freq_cutoffs = np.arange(self.l_freq, self.h_freq)
            freq_span = (self.h_freq - self.l_freq) - 1

        # run detector on all channels
        for idx in range(self.n_chs):
            sig = X[idx, :]

            output = []

            # run detection algorithm possibly in parallel on this channel
            tdetects_concat = []
            if self.n_jobs > 1 or self.n_jobs == -1:
                # for a single job, we don't need joblib
                if self.n_jobs != 1:
                    try:
                        from joblib import Parallel, delayed
                    except ImportError:
                        try:
                            from sklearn.externals.joblib import (
                                Parallel, delayed)
                        except ImportError:
                            warn('joblib not installed. '
                                 'Cannot run in parallel.')
                            self.n_jobs = 1

                # Run the filters in their threads and return the result
                iter_mat = [(sig, self.sfreq, i,
                             freq_cutoffs[i],
                             freq_cutoffs[i + 1],
                             self.cycle_threshold, self.gap_threshold,
                             self.threshold) for i in range(freq_span)]
                tdetects_concat = Parallel(n_jobs=self.n_jobs)(
                    delayed(_band_z_score_detect)(
                        *iter_args
                    )
                    for iter_args in tqdm(iter_mat, unit='HFO-first-phase')
                )
            else:
                # OPTIMIZE - check if there is a better way to do this
                # (S transform?+ spectra zeroing?)
                for i in tqdm(range(freq_span), unit='HFO-first-phase'):
                    bot = freq_cutoffs[i]
                    top = freq_cutoffs[i + 1]

                    args = [sig, self.sfreq, i, bot, top,
                            self.cycle_threshold, self.gap_threshold,
                            self.threshold]

                    tdetects_concat.append(_band_z_score_detect(args))

            # post process detected HFO events by detecting outline of events
            detects = np.array([det for band in tdetects_concat
                                for det in band])
            outlines = []
            if len(detects):
                while sum(detects[:, 0] != 0):
                    det_idx = np.where(detects[:, 0] != 0)[0][0]
                    HFO_outline = []
                    outlines.append(np.array(_run_detect_branch(detects,
                                                                det_idx,
                                                                HFO_outline)))

            # Get the detections
            for outline in outlines:
                start = min(outline[:, 1])
                stop = max(outline[:, 2])
                freq_min = freq_cutoffs[int(outline[0, 0])]
                freq_max = freq_cutoffs[int(outline[-1, 0])]
                frequency_at_max = freq_cutoffs[
                    int(outline[np.argmax(outline[:, 3]), 0])]
                max_amplitude = max(outline[:, 3])

                output.append((start, stop,
                               freq_min, freq_max, frequency_at_max,
                               max_amplitude))
            chs_hfos[idx] = output

        self.hfo_event_arr_ = hfo_event_arr
        self.chs_hfos_ = chs_hfos
        return chs_hfos


class LineLengthDetector(Detector):
    """Line-length detection algorithm.

    Original paper defines HFOS as "(HFOs), which we collectively
    term as all activity >40 Hz (including gamma, high-gamma,
    ripple, and fast ripple oscillations), may have a
    fundamental role in the generation and spread of focal seizures"

    In the paper, data were sampled at 200 Hz and bandpass-filtered (0.1 – 100
    Hz) during acquisition. Data were further digitally bandpass-filtered
    (4th-order Butterworth, forward-backward filtering, ``0.1 – 85 Hz``)
    to minimize potential artifacts due to aliasing. (IIR for forward-backward
    pass).

    Compared to RMS detector, they utilize line-length metric.

    Parameters
    ----------
    filter_band : tuple(float, float) | None
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1.
    threshold: float
        Number of standard deviations to use as a threshold
    win_size: int
        Sliding window size in samples
    overlap: float
        Fraction of the window overlap (0 to 1)
    offset: int
        Offset which is added to the final detection. This is used when the
        function is run in separate windows. Default = 0
    hfo_name: str
        What to name the events detected (i.e. fast ripple if freq_band is
        (250, 500)).

    Notes
    -----
    For processing, a sliding window is used.

    For post-processing, any events that overlap are considered to be the same.

    References
    ----------
    .. [1] A. B. Gardner, G. A. Worrell, E. Marsh, D. Dlugos, and B. Litt,
           “Human and automated detection of high-frequency oscillations in
           clinical intracranial EEG recordings,” Clin. Neurophysiol.,
           vol. 118, no. 5, pp. 1134–1143, May 2007.
    .. [2] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. In Engineering in Medicine and Biology
           Society, 2001. Proceedings of the 23rd Annual International
           Conference of the IEEE (Vol. 2, pp. 1707-1710). IEEE.
    """

    def __init__(self,
                 threshold: Union[int, float] = 3, win_size: int = 100,
                 overlap: float = 0.25, sfreq: int = None,
                 filter_band: Tuple[int, int] = (30, 100),
                 scoring_func: str = 'f1', n_jobs: int = -1,
                 hfo_name: str = "hfo",
                 verbose: bool = False):
        super(LineLengthDetector, self).__init__(
            threshold, win_size=win_size, overlap=overlap,
            scoring_func=scoring_func, n_jobs=n_jobs,
            verbose=verbose)

        self.filter_band = filter_band
        self.sfreq = sfreq
        self.hfo_name = hfo_name

    @property
    def l_freq(self):
        """Lower frequency band for HFO definition."""
        if self.filter_band is None:
            return None
        return self.filter_band[0]

    @property
    def h_freq(self):
        """Higher frequency band for HFO definition."""
        if self.filter_band is None:
            return None
        return self.filter_band[1]

    def _compute_hfo(self, X):
        """Override ``Detector._compute_hfo`` function."""
        # store all hfo occurrences as an array of channels X windows
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)
        hfo_event_arr = np.empty((self.n_chs, n_windows))

        # bandpass the signal using FIR filter
        if self.filter_band is not None:
            X = mne.filter.filter_data(X, sfreq=self.sfreq,
                                       l_freq=self.l_freq,
                                       h_freq=self.h_freq,
                                       method='iir', verbose=self.verbose)

        # run HFO detection on all the channels
        if self.n_jobs == 1:
            for idx in tqdm(range(self.n_chs)):
                sig = X[idx, :]

                # compute sliding window RMS
                hfo_event_arr[idx, :] = \
                    self._compute_sliding_window_detection(
                        sig, method='line_length')
        else:
            if self.n_jobs == -1:
                n_jobs = cpu_count()
            else:
                n_jobs = self.n_jobs

            # run joblib parallelization over channels
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._compute_sliding_window_detection)(
                    X[idx, :], 'line_length'
                ) for idx in tqdm(range(self.n_chs))
            )
            for idx in range(len(results)):
                hfo_event_arr[idx, :] = results[idx]

        return hfo_event_arr


class RMSDetector(Detector):
    """Root mean square (RMS) detection algorithm (Staba Detector).

    The original algorithm described in the reference, takes a sliding
    window of 3 ms, computes the RMS values between 100 and 500 Hz.
    Then events separated by less than 10 ms were combined into one event.
    Then events not having a minimum of 6 peaks (i.e. band-pass signal
    rectified above 0 V) with greater then 3 std above mean baseline
    were removed. A finite impulse response (FIR) filter with a
    Hamming window was used.

    Parameters
    ----------
    filter_band : tuple(float, float) | None
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1.
    threshold: float
        Number of standard deviations to use as a threshold.
        Default = 3.
    win_size: int
        Sliding window size in samples. Default = 100. The
        original paper uses a window size equivalent to 3 ms.
    overlap: float
        Fraction of the window overlap (0 to 1). Default = 0.25.
        The original paper uses an overlap of 0.
    offset: int
        Offset which is added to the final detection. This is used when the
        function is run in separate windows. Default = 0

    References
    ----------
    [1] R. J. Staba, C. L. Wilson, A. Bragin, I. Fried, and J. Engel,
    “Quantitative Analysis of High-Frequency Oscillations (80 − 500 Hz)
    Recorded in Human Epileptic Hippocampus and Entorhinal Cortex,”
    J. Neurophysiol., vol. 88, pp. 1743–1752, 2002.
    """

    def __init__(self, threshold: Union[int, float] = 3, win_size: int = 100,
                 overlap: float = 0.25, sfreq=None,
                 filter_band: Tuple[int, int] = (100, 500),
                 scoring_func='f1', n_jobs: int = -1,
                 hfo_name: str = "hfo",
                 verbose: bool = False):
        super(RMSDetector, self).__init__(
            threshold, win_size, overlap,
            scoring_func,
            n_jobs=n_jobs, verbose=verbose)

        # hyperparameters
        self.filter_band = filter_band
        self.sfreq = sfreq
        self.hfo_name = hfo_name

    @property
    def l_freq(self):
        """Lower frequency band for HFO definition."""
        if self.filter_band is None:
            return None
        return self.filter_band[0]

    @property
    def h_freq(self):
        """Higher frequency band for HFO definition."""
        if self.filter_band is None:
            return None
        return self.filter_band[1]

    def _compute_hfo(self, X):
        """Override ``Detector._compute_hfo`` function.

        Returns
        -------
        hfo_event_arr : np.ndarray (channels x windows)
            The HFO metric value within each window per channel.
        """
        # store all hfo occurrences as an array of channels X windows
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)
        hfo_event_arr = np.empty((self.n_chs, n_windows))

        if self.l_freq is not None or self.h_freq is not None:
            # bandpass the signal using FIR filter
            X = mne.filter.filter_data(X, sfreq=self.sfreq,
                                       l_freq=self.l_freq,
                                       h_freq=self.h_freq,
                                       method='fir', verbose=self.verbose)

        # run HFO detection on all the channels
        if self.n_jobs == 1:
            for idx in tqdm(range(self.n_chs)):
                sig = X[idx, :]

                # compute sliding window RMS
                hfo_event_arr[idx, :] = \
                    self._compute_sliding_window_detection(sig, method='rms')
        else:
            if self.n_jobs == -1:
                n_jobs = cpu_count()
            else:
                n_jobs = self.n_jobs

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._compute_sliding_window_detection)(
                    X[idx, :], 'rms'
                ) for idx in tqdm(range(self.n_chs))
            )
            for idx in range(len(results)):
                hfo_event_arr[idx, :] = results[idx]

        return hfo_event_arr
