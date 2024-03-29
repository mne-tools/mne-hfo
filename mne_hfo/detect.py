from typing import Optional, Tuple, Union

import mne
import numpy as np

from .base import Detector
from .config import ACCEPTED_BAND_METHODS


class HilbertDetector(Detector):  # noqa
    """2D HFO hilbert detection used in Kucewicz et al. 2014.

    A multi-taper method with: 4 Hz bandwidth, 1 sec sliding window,
    stepsize 100 ms, for the 1-500 Hz range, no padding, 2 tapers.
    For full details, see :footcite:`kucewicz2014high`.

    Parameters
    ----------
    threshold : float
        Threshold for detection (default=3).
    filter_band : tuple(float, float)
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1. The default is ``(30, 100)``.
    band_method : str
        Spacing of hilbert frequency bands - options: 'linear' or 'log'
        (default='linear'). Linear provides better frequency resolution but
        is slower.
    n_bands : int
        Number of bands if band_spacing = log (default=300).
    cycle_threshold : float
        Minimum number of cycles to detect (default=1).
    gap_threshold : float
        Number of cycles for gaps (default=1).
    n_jobs : int
        Number of cores to use (default=1).
    offset : int
        Offset which is added to the final detection. This is used when the
        function is run in separate windows. Default = 0.
    scoring_func : str
        The scoring function to apply when trying to match HFOs with
        a different dataset, such as manual annotations.
    hfo_name : str
        What to name the events detected (i.e. fast ripple if freq_band is
        (250, 500)).
    verbose : bool
        Verbosity of the detector.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        threshold: Union[int, float] = 3,
        filter_band: Tuple[int, int] = (30, 100),
        band_method: str = "linear",
        n_bands: int = 300,
        cycle_threshold: float = 1,
        gap_threshold: float = 1,
        n_jobs: int = -1,
        offset: int = 0,
        scoring_func: str = "f1",
        hfo_name: str = "hfo",
        verbose: bool = False,
    ):
        if band_method not in ACCEPTED_BAND_METHODS:
            raise ValueError(
                f"Band method {band_method} is not "
                f"an acceptable parameter. Please use "
                f"one of {ACCEPTED_BAND_METHODS}"
            )

        super(HilbertDetector, self).__init__(
            threshold,
            win_size=1,
            overlap=1,
            scoring_func=scoring_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.band_method = band_method
        self.n_bands = n_bands
        self.filter_band = filter_band
        self.hfo_name = hfo_name
        self.cycle_threshold = cycle_threshold
        self.gap_threshold = gap_threshold
        self.n_jobs = n_jobs
        self.offset = offset

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

    def _create_empty_event_arr(self):
        """Override ``Detector._create_empty_event_arr`` function.

        Also sets the frequency span of the Hilbert detector.
        """
        # Determine the splits for freq bands
        if self.band_method == "log":
            low_fc = float(self.filter_band[0])
            high_fc = float(self.filter_band[1])
            freq_cutoffs = np.logspace(0, np.log10(high_fc), self.n_bands)
            self.freq_cutoffs = freq_cutoffs[
                (freq_cutoffs > low_fc) & (freq_cutoffs < high_fc)
            ]
            self.freq_span = len(self.freq_cutoffs) - 1
        elif self.band_method == "linear":
            self.freq_cutoffs = np.arange(self.filter_band[0], self.filter_band[1])
            self.freq_span = (self.filter_band[1] - self.filter_band[0]) - 1
        n_windows = self.n_times
        n_bands = len(self.freq_cutoffs) - 1
        hfo_event_arr = np.empty((self.n_chs, n_bands, n_windows))
        return hfo_event_arr

    def _compute_hfo_statistic(self, X):
        """Override ``Detector._compute_hfo_statistic`` function."""
        # Override the attribute set by fit so we actually slide on freq
        # bands not time windows
        self.n_windows = self.n_bands
        self.win_size = 1
        self.n_times = len(X)

        hfo_event_arr = self._compute_frq_band_detection(X, method="hilbert")

        return hfo_event_arr

    def _threshold_statistic(self, X):
        """Override ``Detector._threshold_statistic`` function."""
        hfo_threshold_arr = np.transpose(
            np.array(
                self._apply_threshold(X, threshold_method="hilbert"), dtype="object"
            )
        )
        return hfo_threshold_arr

    def _post_process_ch_hfos(self, detections):
        """Override ``Detector._post_process_ch_hfos`` function."""
        hfo_events = self._merge_contiguous_ch_detections(
            detections, method="freq-bands"
        )
        return hfo_events


class LineLengthDetector(Detector):
    """Line-length detection algorithm.

    Original paper defines HFOS as:

    "(HFOs), which we collectively term as all activity >40 Hz
    (including gamma, high-gamma, ripple, and fast ripple oscillations),
    may have a fundamental role in the generation and spread of focal
    seizures." See :footcite:`gardner2007human`.

    In the paper, data were sampled at 200 Hz and bandpass-filtered (0.1 – 100
    Hz) during acquisition. Data were further digitally bandpass-filtered
    (4th-order Butterworth, forward-backward filtering, ``0.1 – 85 Hz``)
    to minimize potential artifacts due to aliasing. (IIR for forward-backward
    pass).

    Compared to RMS detector, they utilize line-length metric
    :footcite:`esteller2001line`.

    Parameters
    ----------
    threshold : float
        Number of standard deviations to use as a threshold.
    win_size : int
        Sliding window size in samples.
    overlap : float
        Fraction of the window overlap (0 to 1).
    sfreq : int | None
        The sampling rate of the data.
    filter_band : tuple(float, float)
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1. The default is ``(30, 100)``.
    scoring_func : str
        The scoring function to apply when trying to match HFOs with
        a different dataset, such as manual annotations.
    n_jobs : int
        The number of jobs for joblib parallelization.
    hfo_name : str
        What to name the events detected (i.e. fast ripple if freq_band is
        (250, 500)).
    verbose : bool
        Verbosity of the detector.

    Notes
    -----
    For processing, a sliding window is used.

    For post-processing, any events that overlap are considered to be the same.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        threshold: Union[int, float] = 3,
        win_size: int = 100,
        overlap: float = 0.25,
        sfreq: Optional[int] = None,
        filter_band: Tuple[int, int] = (30, 100),
        scoring_func: str = "f1",
        n_jobs: int = -1,
        hfo_name: str = "hfo",
        verbose: bool = False,
    ):
        super(LineLengthDetector, self).__init__(
            threshold,
            win_size=win_size,
            overlap=overlap,
            scoring_func=scoring_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )

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

    def _compute_hfo_statistic(self, X):
        """Override ``Detector._compute_hfo_statistic`` function."""
        # store all hfo occurrences as an array of length windows

        # bandpass the signal using FIR filter
        if self.filter_band is not None:
            X = mne.filter.filter_data(
                X,
                sfreq=self.sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method="iir",
                verbose=self.verbose,
            )

        hfo_event_arr = self._compute_sliding_window_detection(X, method="line_length")

        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, self.step_size, self.n_times)
        n_bands = len(self.freq_cutoffs) - 1
        shape = (n_windows, n_bands)
        hfo_event_arr = np.array(hfo_event_arr).reshape(shape)

        return hfo_event_arr

    def _threshold_statistic(self, X):
        """Override ``Detector._threshold_statistic`` function."""
        hfo_threshold_arr = self._apply_threshold(X, threshold_method="std")
        return hfo_threshold_arr

    def _post_process_ch_hfos(self, detections):
        """Override ``Detector._post_process_ch_hfos`` function."""
        return self._merge_contiguous_ch_detections(detections, method="time-windows")


class RMSDetector(Detector):
    """Root mean square (RMS) detection algorithm (Staba Detector).

    The original algorithm described in the reference, takes a sliding
    window of 3 ms, computes the RMS values between 100 and 500 Hz.
    Then events separated by less than 10 ms were combined into one event.
    Then events not having a minimum of 6 peaks (i.e. band-pass signal
    rectified above 0 V) with greater then 3 std above mean baseline
    were removed. A finite impulse response (FIR) filter with a
    Hamming window was used. See :footcite:`staba2002quantitative`.

    Parameters
    ----------
    threshold : float
        Number of standard deviations to use as a threshold.
    win_size : int
        Sliding window size in samples.
    overlap : float
        Fraction of the window overlap (0 to 1).
    sfreq : int | None
        The sampling rate of the data.
    filter_band : tuple(float, float)
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1. The default is ``(100, 500)``.
    scoring_func : str
        The scoring function to apply when trying to match HFOs with
        a different dataset, such as manual annotations.
    n_jobs : int
        The number of jobs for joblib parallelization.
    hfo_name : str
        What to name the events detected (i.e. fast ripple if freq_band is
        (250, 500)).
    verbose : bool
        Verbosity of the detector.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        threshold: Union[int, float] = 3,
        win_size: int = 100,
        overlap: float = 0.25,
        sfreq=None,
        filter_band: Tuple[int, int] = (100, 500),
        scoring_func="f1",
        n_jobs: int = -1,
        hfo_name: str = "hfo",
        verbose: bool = False,
    ):
        super(RMSDetector, self).__init__(
            threshold, win_size, overlap, scoring_func, n_jobs=n_jobs, verbose=verbose
        )

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

    def _compute_hfo_statistic(self, X):
        """Override ``Detector._compute_hfo`` function."""
        # store all hfo occurrences as an array of length windows

        if self.l_freq is not None or self.h_freq is not None:
            # bandpass the signal using FIR filter
            X = mne.filter.filter_data(
                X,
                sfreq=self.sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method="fir",
                verbose=self.verbose,
            )

        hfo_event_arr = self._compute_sliding_window_detection(X, method="rms")

        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, self.step_size, self.n_times)
        n_bands = len(self.freq_cutoffs) - 1
        shape = (n_windows, n_bands)
        hfo_event_arr = np.array(hfo_event_arr).reshape(shape)

        return hfo_event_arr

    def _threshold_statistic(self, X):
        """Override ``Detector._threshold_statistic`` function."""
        hfo_threshold_arr = self._apply_threshold(X, threshold_method="std")
        return hfo_threshold_arr

    def _post_process_ch_hfos(self, detections):
        """Override ``Detector._post_process_ch_hfos`` function."""
        return self._merge_contiguous_ch_detections(detections, method="time-windows")
