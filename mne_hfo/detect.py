from typing import Tuple, Union

import mne
import numpy as np
from scipy import stats
from mne.time_frequency import tfr_array_morlet


from mne_hfo.base import Detector
from mne_hfo.config import ACCEPTED_BAND_METHODS
from mne_hfo.utils import autocorr


class HilbertDetector(Detector):  # noqa
    """2D HFO hilbert detection used in Kucewicz et al. 2014.

    A multi-taper method with: 4 Hz bandwidth, 1 sec sliding window,
    stepsize 100 ms, for the 1-500 Hz range, no padding, 2 tapers.

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
    [1] M. T. Kucewicz, J. Cimbalnik, J. Y. Matsumoto, B. H. Brinkmann,
    M. Bower, V. Vasoli, V. Sulc, F. Meyer, W. R. Marsh, S. M. Stead, and
    G. A. Worrell, “High frequency oscillations are associated with
    cognitive processing in human recognition memory.,” Brain, pp. 1–14,
    Jun. 2014.
    """

    def __init__(self,
                 threshold: Union[int, float] = 3,
                 filter_band: Tuple[int, int] = (30, 100),
                 band_method: str = 'linear', n_bands: int = 300,
                 cycle_threshold: float = 1, gap_threshold: float = 1,
                 n_jobs: int = -1, offset: int = 0,
                 scoring_func: str = 'f1',
                 hfo_name: str = "hilberthfo", verbose: bool = False):
        if band_method not in ACCEPTED_BAND_METHODS:
            raise ValueError(f'Band method {band_method} is not '
                             f'an acceptable parameter. Please use '
                             f'one of {ACCEPTED_BAND_METHODS}')

        super(HilbertDetector, self).__init__(
            threshold, win_size=1, overlap=1,
            scoring_func=scoring_func, name=hfo_name,
            n_jobs=n_jobs, verbose=verbose)

        self.band_method = band_method
        self.n_bands = n_bands
        self.filter_band = filter_band
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
        if self.band_method == 'log':
            low_fc = float(self.filter_band[0])
            high_fc = float(self.filter_band[1])
            freq_cutoffs = np.logspace(0, np.log10(high_fc), self.n_bands)
            self.freq_cutoffs = freq_cutoffs[(freq_cutoffs > low_fc) &
                                             (freq_cutoffs < high_fc)]
            self.freq_span = len(self.freq_cutoffs) - 1
        elif self.band_method == 'linear':
            self.freq_cutoffs = np.arange(self.filter_band[0],
                                          self.filter_band[1])
            self.freq_span = (self.filter_band[1] - self.filter_band[0]) - 1
        n_windows = self.n_times
        n_bands = len(self.freq_cutoffs) - 1
        hfo_event_arr = np.empty((self.n_chs, n_bands, n_windows))
        return hfo_event_arr

    def compute_hfo_statistic(self, X):
        """Override ``Detector.compute_hfo_statistic`` function."""
        # Override the attribute set by fit so we actually slide on freq
        # bands not time windows
        self.n_windows = self.n_bands
        self.win_size = 1
        self.n_times = len(X)

        hfo_event_arr = self._compute_freq_band_detection(X, method='hilbert')

        return hfo_event_arr

    def threshold_hfo_statistic(self, X):
        """Override ``Detector.threshold_hfo_statistic`` function."""
        hfo_threshold_arr = np.transpose(np.array(self._apply_threshold(
            X, threshold_method='hilbert'), dtype='object'))
        return hfo_threshold_arr

    def post_process_ch_hfos(self, detections):
        """Override ``Detector.post_process_ch_hfos`` function."""
        hfo_events = self._merge_contiguous_ch_detections(
            detections, method="freq-bands")
        return hfo_events


class LineLengthDetector(Detector):
    """Line-length detection algorithm.

    Original paper defines HFOS as:

    "(HFOs), which we collectively term as all activity >40 Hz
    (including gamma, high-gamma, ripple, and fast ripple oscillations),
    may have a fundamental role in the generation and spread of focal
    seizures."

    In the paper, data were sampled at 200 Hz and bandpass-filtered (0.1 – 100
    Hz) during acquisition. Data were further digitally bandpass-filtered
    (4th-order Butterworth, forward-backward filtering, ``0.1 – 85 Hz``)
    to minimize potential artifacts due to aliasing. (IIR for forward-backward
    pass).

    Compared to RMS detector, they utilize line-length metric.

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
                 hfo_name: str = "llhfo",
                 verbose: bool = False):
        super(LineLengthDetector, self).__init__(
            threshold, win_size=win_size, overlap=overlap,
            scoring_func=scoring_func, name=hfo_name, n_jobs=n_jobs,
            verbose=verbose)

        self.filter_band = filter_band
        self.sfreq = sfreq

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

    def compute_hfo_statistic(self, X):
        """Override ``Detector.compute_hfo_statistic`` function."""
        # store all hfo occurrences as an array of length windows

        # bandpass the signal using FIR filter
        if self.filter_band is not None:
            X = mne.filter.filter_data(X, sfreq=self.sfreq,
                                       l_freq=self.l_freq,
                                       h_freq=self.h_freq,
                                       method='iir', verbose=self.verbose)

        hfo_event_arr = self._compute_sliding_window_detection(
            X, method='line_length')

        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, self.step_size,
                                         self.n_times)
        n_bands = len(self.freq_cutoffs) - 1
        shape = (n_windows, n_bands)
        hfo_event_arr = np.array(hfo_event_arr).reshape(shape)

        return hfo_event_arr

    def threshold_hfo_statistic(self, X):
        """Override ``Detector.threshold_hfo_statistic`` function."""
        hfo_threshold_arr = self._apply_threshold(
            X, threshold_method='std'
        )
        return hfo_threshold_arr

    def post_process_ch_hfos(self, detections):
        """Override ``Detector.post_process_ch_hfos`` function."""
        return self._merge_contiguous_ch_detections(
            detections, method="time-windows")


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
    [1] R. J. Staba, C. L. Wilson, A. Bragin, I. Fried, and J. Engel,
    “Quantitative Analysis of High-Frequency Oscillations (80 − 500 Hz)
    Recorded in Human Epileptic Hippocampus and Entorhinal Cortex,”
    J. Neurophysiol., vol. 88, pp. 1743–1752, 2002.
    """

    def __init__(self, threshold: Union[int, float] = 3, win_size: int = 100,
                 overlap: float = 0.25, sfreq=None,
                 filter_band: Tuple[int, int] = (100, 500),
                 scoring_func='f1', n_jobs: int = -1,
                 hfo_name: str = "rmshfo",
                 verbose: bool = False):
        super(RMSDetector, self).__init__(
            threshold, win_size, overlap,
            scoring_func, name=hfo_name,
            n_jobs=n_jobs, verbose=verbose)

        # hyperparameters
        self.filter_band = filter_band
        self.sfreq = sfreq

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

    def compute_hfo_statistic(self, X):
        """Override ``Detector._compute_hfo`` function."""
        # store all hfo occurrences as an array of length windows

        if self.l_freq is not None or self.h_freq is not None:
            # bandpass the signal using FIR filter
            X = mne.filter.filter_data(X, sfreq=self.sfreq,
                                       l_freq=self.l_freq,
                                       h_freq=self.h_freq,
                                       method='fir', verbose=self.verbose)

        hfo_event_arr = self._compute_sliding_window_detection(
            X, method='rms')

        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, self.step_size,
                                         self.n_times)
        n_bands = len(self.freq_cutoffs) - 1
        shape = (n_windows, n_bands)
        hfo_event_arr = np.array(hfo_event_arr).reshape(shape)

        return hfo_event_arr

    def threshold_hfo_statistic(self, X):
        """Override ``Detector.threshold_hfo_statistic`` function."""
        hfo_threshold_arr = self._apply_threshold(
            X, threshold_method='std'
        )
        return hfo_threshold_arr

    def post_process_ch_hfos(self, detections):
        """Override ``Detector.post_process_ch_hfos`` function."""
        return self._merge_contiguous_ch_detections(
            detections, method="time-windows")


class MNIDetector(Detector):
    def __init__(self, threshold: Union[int, float],
                 win_size: Union[int, None],
                 overlap: Union[float, None],
                 min_window_size: int = 10e-3,
                 min_gap_size: int = 10e-3,
                 baseline_threshold: float = 0.67,
                 baseline_seg_size: float = 0.125,
                 baseline_step_size: float = 0.5,
                 baseline_min_time: float = 5,
                 baseline_n_bootstrap: int = 100,
                 filter_band: Tuple[int, int] = (80, 450),
                 scoring_func: str = 'f1', sfreq=None,
                 hfo_name: str = 'mnihfo',
                 n_jobs: int = -1, verbose: bool = True):
        super().__init__(threshold, win_size, overlap, scoring_func,
                         hfo_name, n_jobs, verbose)
        # hyperparameters
        self.filter_band = filter_band
        self.sfreq = sfreq

        self.min_window_size = min_window_size
        self.min_gap_size = min_gap_size
        self.baseline_threshold = baseline_threshold
        self.baseline_seg_size = baseline_seg_size
        self.baseline_step_size = baseline_step_size
        self.baseline_min_time = baseline_min_time
        self.baseline_n_bootstrap = baseline_n_bootstrap

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

    def compute_hfo_statistic(self, X):
        """Override ``Detector._compute_hfo`` function."""
        # first, bandpass the signal using FIR filter
        X = mne.filter.filter_data(X, sfreq=self.sfreq,
                                   l_freq=self.l_freq,
                                   h_freq=self.h_freq,
                                   method='fir', verbose=self.verbose)

        # first compute the baseline
        baseline_windows = self.compute_baseline(X)

        # convert the necessary baseline needed into samples
        window_samples = self.baseline_min_time / 60 * self.n_times

        # compute the threshold used, depending on if baseline was detected
        if np.sum(baseline_windows) >= window_samples:
            # baseline was found, so this is the 99.9999 percentil
            # value of a gamma distribution modeled for the empirical
            # CDF of each 10second baseline segment
            energy_threshold = 'posbaseline'
        else:
            energy_threshold = 'negbaseline'

        # second, compute the RMS over a sliding window
        rms_event_arr = self._compute_sliding_window_detection(
            X, method='rms')

        # if the baseline was met
        window_min_secs = window_min_secs * sfreq

        # DO REST

        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, self.step_size,
                                         self.n_times)
        n_bands = len(self.freq_cutoffs) - 1
        shape = (n_windows, n_bands)
        rms_event_arr = np.array(rms_event_arr).reshape(shape)

        # compute the parts of signal with high RMS
        threshold = None
        high_rms_energy = rms_event_arr >= threshold

        # get the window threshold
        window_threshold = np.concatenate((0, high_rms_energy, 0))
        window_jumps = np.diff(window_threshold)

        # find when the windows jump up, or down
        window_jump_up = np.argwhere(window_jumps == 1)
        window_jump_down = np.argwhere(window_jumps == -1)
        window_dist = window_jump_down - window_jump_up

        # window distance selection where RMS energy is exceeded
        # for more then a certain length of time
        window_dist_select = window_dist > window_min_secs

        # find windows that were selected
        window_select = np.nonzero(window_dist_select)

        # get the endpoints of each window
        start_windows = window_jump_up[window_select]
        end_windows = window_jump_down[window_select]

        # loop until no more HFOs

    def compute_baseline(self, X):
        base_threshold = self.baseline_threshold

        # compute parameters in terms of samples
        baseline_step_size = self.baseline_step_size * self.sfreq

        # known as s_EpochSamples
        base_seg_sample = self.baseline_seg_size * self.sfreq
        num_epoch_samples = np.round(base_seg_sample * self.sfreq)

        n_repeats = self.baseline_n_bootstrap

        # compute estimate of baseline using bootstrap of 100 times
        baseline_we_max = np.zeros((n_repeats,))
        for idx in range(n_repeats):
            # create a white-noise segment
            white_noise_seg = np.random.rand(num_epoch_samples, 1)

            # compute auto-correlation of the signal
            ac_x = autocorr(white_noise_seg)

            # compute the wavelet entropy of the auto-correlation
            wavelet_ac_x = tfr_array_morlet(ac_x[np.newaxis, ...]).squeeze()

            # compute the mean normalized energy
            mean_energy = np.mean(np.power(wavelet_ac_x, 2), axis=1)
            mean_energy = mean_energy / np.sum(mean_energy)

            # compute the max theoretical wavelet entropy
            we_max = -np.sum(np.multiply(mean_energy, np.log(mean_energy)))
            baseline_we_max[idx] = we_max

        # get the median as our baseline max Wavelet Entropy
        baseline_we_max = np.median(baseline_we_max)

        # now compute portions of the dataset that are considered baseline
        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(self.win_size, baseline_step_size,
                                         self.n_times)
        baseline_wins = np.zeros(X.shape)

        start_index = np.linspace(0, self.n_times, self.win_size)
        end_index = start_index + base_seg_sample

        for idx in range(n_windows):
            # compute auto-correlation of the signal
            ac_x = np.apply_along_axis(autocorr, axis=1, arr=X)
            ac_x = ac_x / np.sum(np.power(X, 2), axis=1)

            # compute the wavelet entropy of the auto-correlation
            wavelet_ac_x = tfr_array_morlet(ac_x[np.newaxis, ...]).squeeze()

            # compute the mean normalized energy
            mean_energy = np.mean(np.power(wavelet_ac_x, 2), axis=1)
            mean_energy = mean_energy / np.sum(mean_energy)

            # compute the maximum wavelet entropy in this section
            we_max = -np.sum(np.multiply(mean_energy, np.log(mean_energy)))
            if we_max < baseline_we_max * base_threshold:
                baseline_wins[start_index[idx]:end_index[idx]] = 1

        return baseline_wins

    def compute_threshold(self, baseline_wins, X, sfreq):

        return threshold

    def threshold_hfo_statistic(self, X):
        """Override ``Detector.threshold_hfo_statistic`` function."""
        hfo_threshold_arr = self._apply_threshold(
            X, threshold_method='std'
        )
        return hfo_threshold_arr

    def post_process_ch_hfos(self, detections):
        """Override ``Detector.post_process_ch_hfos`` function."""
        return self._merge_contiguous_ch_detections(
            detections, method="time-windows")

    def _compute_energy_threshold(self, X, baseline, baseline_windows):
        if baseline:
            # cycle time
            window_threshold = cycle_time * self.sfreq

            # find how many discrete windows there are
            baseline_jump = np.diff(np.concatenate((0, baseline_windows, 0)))
            baseline_jump_up = np.argwhere(baseline_jump == 1)
            n_windows = len(baseline_jump_up)

            # get a list of all the baseline windows
            for idx in range(n_windows):
                pass

            for idx in range(n_windows):
                # fit gamma function to the window
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)

                # now generate the empirical CDF of the baseline data
                gamma_perc = stats.gamma.cdf(
                    data, fit_alpha, loc=fit_loc, scale=fit_beta)

                # find the index where gamma less then the threshold percentile
                index = np.arghwere(
                    gamma_perc <= energy_threshold_percentile)[::-1][0]

                # now apply threshold for each data point
        else:
            # continuous hfo
            while 1:
                # fit gamma function to the window
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)

                # now generate the empirical CDF of the baseline data
                gamma_perc = stats.gamma.cdf(
                    data, fit_alpha, loc=fit_loc, scale=fit_beta)

                # find the index where gamma less then the threshold percentile
                index = np.arghwere(
                    gamma_perc <= energy_threshold_percentile)[::-1][0]
