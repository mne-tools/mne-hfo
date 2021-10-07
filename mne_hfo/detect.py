from typing import Tuple, Union

import mne
import numpy as np
from scipy import stats
from mne.time_frequency import tfr_array_morlet


from mne_hfo.base import Detector
from mne_hfo.config import ACCEPTED_BAND_METHODS
from mne_hfo.utils import autocorr, rolling_rms
from .docs import fill_doc


@fill_doc
class HilbertDetector(Detector):  # noqa
    """2D HFO hilbert detection used in Kucewicz et al. 2014.

    A multi-taper method with: 4 Hz bandwidth, 1 sec sliding window,
    stepsize 100 ms, for the 1-500 Hz range, no padding, 2 tapers.

    Parameters
    ----------
    threshold : float
        Threshold for detection (default=3).
    %(sfreq)s
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
    %(scoring_func)s
    %(n_jobs)s
    %(hfo_name)s
    %(verbose)s

    References
    ----------
    [1] M. T. Kucewicz, J. Cimbalnik, J. Y. Matsumoto, B. H. Brinkmann,
    M. Bower, V. Vasoli, V. Sulc, F. Meyer, W. R. Marsh, S. M. Stead, and
    G. A. Worrell, “High frequency oscillations are associated with
    cognitive processing in human recognition memory.,” Brain, pp. 1–14,
    Jun. 2014.
    """

    def __init__(self,
                 threshold: Union[int, float] = 3, sfreq=None,
                 filter_band: Tuple[int, int] = (30, 100),
                 band_method: str = 'linear', n_bands: int = 300,
                 cycle_threshold: float = 1, gap_threshold: float = 1,
                 offset: int = 0, scoring_func: str = 'f1',
                 n_jobs: int = -1, hfo_name: str = "hilberthfo",
                 verbose: bool = False):
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


@fill_doc
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
    %(sfreq)s
    filter_band : tuple(float, float)
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1. The default is ``(30, 100)``.
    %(scoring_func)s
    %(n_jobs)s
    %(hfo_name)s
    %(verbose)s

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


@fill_doc
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
    %(sfreq)s
    filter_band : tuple(float, float)
        Low cut-off frequency at index 0 and high cut-off frequency
        at index 1. The default is ``(100, 500)``.
    %(scoring_func)s
    %(n_jobs)s
    %(hfo_name)s
    %(verbose)s

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


@fill_doc
class MNIDetector(Detector):
    def __init__(self, threshold: Union[int, float],
                 win_size: Union[int, None] = 0.002,
                 overlap: Union[float, None] = 0.99,
                 filter_band: Tuple[int, int] = (80, 450),
                 min_window_size: int = 10e-3,
                 min_gap_size: int = 10e-3,
                 baseline_threshold: float = 0.67,
                 baseline_seg_size: float = 0.125,
                 baseline_step_size: float = 0.5,
                 baseline_min_time: float = 5,
                 baseline_n_bootstrap: int = 100,
                 cycle_time: float = 10,
                 energy_baseline_thresh: float = 0.9999,
                 energy_chfo_thresh: float = 0.95,
                 sfreq=None,
                 scoring_func: str = 'f1',
                 n_jobs: int = -1, hfo_name: str = 'mnihfo',
                 verbose: bool = True):
        """[summary]

        Parameters
        ----------
        threshold : Union[int, float]
            [description]
        win_size : Union[int, None]
            [description]
        overlap : Union[float, None]
            [description]
        filter_band : Tuple[int, int], optional
            [description], by default (80, 450)
        min_window_size : int, optional
            [description], by default 10e-3
        min_gap_size : int, optional
            [description], by default 10e-3
        baseline_threshold : float, optional
            [description], by default 0.67
        baseline_seg_size : float, optional
            [description], by default 0.125
        baseline_step_size : float, optional
            [description], by default 0.5
        baseline_min_time : float, optional
            [description], by default 5
        baseline_n_bootstrap : int, optional
            [description], by default 100
        cycle_time : float
            The amount of time in each segment cycle in seconds. By default 10.
            This is the window of data to consider when looking for HFOs.
        energy_baseline_thresh : float
            The energy percentile threshold for EEG channels with a baseline
            segment. By default 0.9999.
        energy_chfo_thresh : float
            The energy percentile threshold for EEG channels without a baseline
            segment, where HFOs are looked for iteratively. By default 0.95.
        %(sfreq)s
        %(scoring_func)s
        %(n_jobs)s
        %(hfo_name)s
        %(verbose)s
        """
        super().__init__(threshold, win_size, overlap, scoring_func,
                         hfo_name, n_jobs, verbose)
        # hyperparameters
        self.filter_band = filter_band
        self.sfreq = sfreq

        # minimum HFO time
        self.min_window_size = min_window_size

        # minimum gap between HFOs
        self.min_gap_size = min_gap_size

        # cycle time for each segment
        self.cycle_time = cycle_time

        self.energy_baseline_thresh = energy_baseline_thresh
        self.energy_chfo_thres = energy_chfo_thresh

        # baseline threshold, window size, step size, and time
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
                                   h_freq=self.h_freq, n_jobs=self.n_jobs,
                                   method='fir', verbose=self.verbose)

        # compute the max wavelet entropy of white noise
        baseline_we_max = self._compute_white_noise_wemax()

        # first compute the baseline
        baseline_windows = self.compute_baseline(X, baseline_we_max)

        # convert the necessary baseline needed into samples
        # minimum size of the baseline
        min_baseline_size = self.baseline_min_time / 60 * self.n_times

        # minimum size of each HFO event
        min_window_size = self.min_window_size * self.sfreq

        # second, compute the RMS over a sliding window
        rms_event_arr = rolling_rms(X, 5)

        # compute the threshold used, depending on if baseline was detected
        baseline_condition = np.sum(baseline_windows) >= min_baseline_size

        # compute the necessary energy threshold
        energy_threshold = self._compute_energy_threshold(
            X, baseline_condition, baseline_windows)

        # energy threshold met by RMS values
        # compute the parts of signal with high RMS
        high_rms_energy = rms_event_arr >= energy_threshold

        # get the window threshold
        window_threshold = np.concatenate((0, high_rms_energy, 0))
        window_jumps = np.diff(window_threshold)

        # find when the windows jump up, or down
        window_jump_up = np.argwhere(window_jumps == 1)
        window_jump_down = np.argwhere(window_jumps == -1)
        window_dist = window_jump_down - window_jump_up

        # window distance selection where RMS energy is exceeded
        # for more then a certain length of time
        window_dist_select = window_dist > min_window_size

        # find windows that were selected
        window_select = np.nonzero(window_dist_select)

        # get the endpoints of each window
        start_windows = window_jump_up[window_select]
        end_windows = window_jump_down[window_select]

        # loop until no more HFOs
        hfo_event_arr = np.zeros(rms_event_arr.shape)
        for idx in range(len(start_windows)):
            hfo_event_arr[start_windows[idx]: end_windows[idx]] = 1
        return hfo_event_arr

    def _compute_white_noise_wemax(self):
        # compute parameters in terms of samples
        baseline_win_size = self.baseline_seg_size * self.sfreq

        # known as s_EpochSamples
        num_epoch_samples = np.round(baseline_win_size * self.sfreq)

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

        return np.median(baseline_we_max)

    def compute_baseline(self, X, baseline_we_max):
        """Compute the baseline for an EEG channel.

        Computes the baseline of an EEG channel by looking
        for segments without oscillatory activity of any kind.
        This function first computes the maximum wavelet entropy
        for the autocorrelation of white noise (i.e. WEmax). Then,
        it computes the wavelet entropy of the autocorrelation of
        the filtered EEG signal. Then taking a sliding window,
        segments are considered baseline if the minimum wavelet
        entropy is larger then the threshold (e.g. 0.67 * WEmax).

        Consecutive or overlapping segments are joined.

        Parameters
        ----------
        X : np.ndarray, shape (n_times,)
            A 1-D vector of an EEG electrode activity over time.

        Returns
        -------
        baseline_windows : np.ndarray, shape (n_times, )
            A 1-D vector of 1's and 0's representing an array mask
            where the baseline was detected.

        Notes
        -----
        This function assumes that the input EEG data is already
        bandpass filtered.
        """
        assert X.ndim == 1
        base_threshold = self.baseline_threshold

        # compute parameters in terms of samples
        baseline_win_size = self.baseline_seg_size * self.sfreq
        baseline_step_size = self.baseline_step_size * self.sfreq

        # now compute portions of the dataset that are considered baseline
        # reshape array to be n_wins x n_bands (i.e. 1)
        n_windows = self._compute_n_wins(
            baseline_win_size, baseline_step_size, self.n_times)
        baseline_wins = np.zeros(X.shape)

        # create array of the start and end indices for the sliding window
        start_index = np.linspace(0, self.n_times, self.win_size)
        end_index = start_index + baseline_win_size

        for idx in range(n_windows):
            # compute auto-correlation of the signal
            windowed_X = X[start_index[idx]:end_index[idx]]
            ac_x = autocorr(windowed_X)
            ac_x = ac_x / np.sum(np.power(windowed_X, 2))

            # compute the wavelet of the auto-correlation
            wavelet_ac_x = tfr_array_morlet(
                ac_x[np.newaxis, np.newaxis, ...]).squeeze()

            # compute the mean normalized energy
            mean_energy = np.mean(np.power(wavelet_ac_x, 2))
            mean_energy = mean_energy / np.sum(mean_energy)

            # compute the maximum wavelet entropy in this section
            we_max = -np.sum(np.multiply(mean_energy, np.log(mean_energy)))
            if we_max < baseline_we_max * base_threshold:
                baseline_wins[start_index[idx]:end_index[idx]] = 1

        return baseline_wins

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

    def _compute_energy_threshold(self, X: np.ndarray, baseline: bool,
                                  baseline_windows: np.ndarray):
        assert X.ndim == 1

        energy_threshold = np.zeros(X.shape)

        minimum_window_size = self.min_window_size * self.sfreq

        if baseline:
            # cycle time - s_WindowThreshold
            window_cycle = np.round(self.cycle_time * self.sfreq)

            # find how many discrete windows there are
            baseline_jump = np.diff(np.concatenate((0, baseline_windows, 0)))
            baseline_jump_up = np.argwhere(baseline_jump == 1)
            baseline_jump_down = np.argwhere(baseline_jump == -1) - 1

            # get starting indices of all the windows
            window_start = []
            for idx in range(len(baseline_jump_up)):
                index = np.arange(
                    baseline_jump_up[idx],
                    baseline_jump_down[idx],
                    window_cycle).tolist()
                window_start.extend(index)

            # get ending indices of all the windows
            window_end = np.array(window_start) + window_cycle
            n_windows = len(window_start)

            # loop through each baseline window of at least "window_cycle" long
            for idx in range(n_windows):
                # get the baseline window data
                data = X[window_start[idx]:window_end[idx]]

                # fit gamma function to the window
                fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)

                # now generate the empirical CDF of the baseline data
                gamma_perc = stats.gamma.cdf(
                    data, fit_alpha, loc=fit_loc, scale=fit_beta)

                # find the index where gamma less then the threshold percentile
                index = np.arghwere(
                    gamma_perc <= self.energy_baseline_thresh)[::-1][0]

                # now apply threshold for each data point
                energy_threshold[window_start[idx]:window_end[idx]] = data[index]
        else:
            # number of seconds to consider for every continuous HFO
            cont_hfo_epoch = 60
            window_chfo = np.round(cont_hfo_epoch * self.sfreq)

            window_start = np.arange(0, len(X), window_chfo)
            window_end = window_start + window_chfo

            for idx in range(len(window_start)):
                data = np.sort(X[window_start[idx]:window_end[idx]])

                curr_energy_threshold = np.max(data)

                # continuous hfo
                while 1:
                    if np.sum(np.abs(data)) == 0:
                        break

                    # fit gamma function to the window
                    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)

                    # now generate the empirical CDF of the baseline data
                    gamma_perc = stats.gamma.cdf(
                        data, fit_alpha, loc=fit_loc, scale=fit_beta)

                    # find the index where gamma less then the threshold percentile
                    index = np.arghwere(
                        gamma_perc <= self.energy_chfo_thres)[::-1][0]

                    # new energy threshold
                    this_energy_threshold = data[index]

                    # find portions of the data segment that are over this energy threshold
                    over_energy_X = data >= this_energy_threshold

                    # find jumps
                    baseline_jump = np.diff(np.concatenate((0, over_energy_X)))
                    baseline_jump_up = np.argwhere(baseline_jump == 1)
                    baseline_jump_down = np.argwhere(baseline_jump == -1) - 1

                    # find distance of the jump
                    jump_dist = baseline_jump_down - baseline_jump_up

                    # select windows where there are events meeting the minimum
                    # size (in time) criterion for an HFO event
                    hfo_events_index = np.argwhere(
                        jump_dist > minimum_window_size)

                    if hfo_events_index.size == 0:
                        break

                    # assign new window start and window ends
                    window_start = baseline_jump_up[hfo_events_index]
                    window_end = baseline_jump_down[hfo_events_index]

                    # remove these windows from the next iteration
                    for idx in range(len(window_start)):
                        data[window_start[idx]:window_end[idx]] = 0

                    # assign new energy threshold
                    curr_energy_threshold = this_energy_threshold

                energy_threshold[window_start[idx]:window_end[idx]] = curr_energy_threshold
        return energy_threshold
