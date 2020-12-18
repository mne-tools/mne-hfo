import numpy as np
from mne.utils import warn
from scipy.signal import butter, hilbert, filtfilt
from sklearn.base import BaseEstimator

from mne_hfo.utils import (threshold_std, compute_rms,
                           check_detection_overlap, compute_line_length)

MINIMUM_SUGGESTED_SFREQ = 2000


def _band_z_score_detect(args):
    x_cond = args[0]
    fs = args[1]
    band_i = args[2]
    bot = args[3]
    top = args[4]
    cyc_th = args[5]
    gap_th = args[6]
    threshold = args[7]

    tdetects = []
    thresh_sig = np.zeros(len(x_cond), dtype='bool')

    b, a = butter(3, bot / (fs / 2), 'highpass')
    fx = filtfilt(b, a, x_cond)

    b, a = butter(3, top / (fs / 2), 'lowpass')
    fx = filtfilt(b, a, fx)

    # Compute the z-scores
    fx = (fx - np.mean(fx)) / np.std(fx)

    hfx = np.abs(hilbert(fx))

    # Create dot product and threshold the signal
    thresh_sig[:] = 0
    thresh_sig[hfx > threshold] = 1

    # Now get the lengths
    idx = 0
    th_idcs = np.where(thresh_sig == 1)[0]
    gap_samp = round(gap_th * fs / bot)
    while idx < len(th_idcs) - 1:
        if (th_idcs[idx + 1] - th_idcs[idx]) == 1:
            start_idx = th_idcs[idx]
            while idx < len(th_idcs) - 1:
                if (th_idcs[idx + 1] - th_idcs[idx]) == 1:
                    idx += 1  # Move to the end of the detection
                    if idx == len(th_idcs) - 1:
                        stop_idx = th_idcs[idx]
                        # Check for number of cycles
                        dur = (stop_idx - start_idx) / fs
                        cycs = bot * dur
                        if cycs > cyc_th:
                            # Carry the amplitude and frequency info
                            tdetects.append([band_i, start_idx, stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                else:  # Check for gap
                    if (th_idcs[idx + 1] - th_idcs[idx]) < gap_samp:
                        idx += 1
                    else:
                        stop_idx = th_idcs[idx]
                        # Check for number of cycles
                        dur = (stop_idx - start_idx) / fs
                        cycs = bot * dur
                        if cycs > cyc_th:
                            tdetects.append([band_i, start_idx, stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                        idx += 1
                        break
        else:
            idx += 1

    return tdetects


def _run_detect_branch(detects, det_idx, HFO_outline):
    """
    Function to process detections from hilbert detector.
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
            if check_detection_overlap([detects[det_idx, 1], detects[det_idx,
                                                                     2]],
                                       [detects[next_det_idx, 1],
                                        detects[next_det_idx,
                                                2]]):
                # Go up the tree
                _run_detect_branch(detects, next_det_idx, HFO_outline)

        detects[det_idx, 0] = 0
        return HFO_outline


class LineLengthDetector(BaseEstimator):
    """Line-length detection algorithm.

    Parameters
    ----------
    sfreq: int
        Sampling frequency
    threshold: float
        Number of standard deviations to use as a threshold
    wsize: int
        Sliding window size in samples
    overlap: float
        Fraction of the window overlap (0 to 1)
    offset: int
        Offset which is added to the final detection. This is used when the
        function is run in separate windows. Default = 0

    References
    ----------
    .. [1] A. B. Gardner, G. A. Worrell, E. Marsh, D. Dlugos, and B. Litt,
           “Human and automated detection of high-frequency oscillations in
           clinical intracranial EEG recordings,” Clin. Neurophysiol., vol. 118,
           no. 5, pp. 1134–1143, May 2007.
    .. [2] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. In Engineering in Medicine and Biology
           Society, 2001. Proceedings of the 23rd Annual International
           Conference of the IEEE (Vol. 2, pp. 1707-1710). IEEE.
    """

    def __init__(self, sfreq: int, threshold: float = 3, wsize: int = 100,
                 overlap: float = 0.25):
        if sfreq < MINIMUM_SUGGESTED_SFREQ:
            warn(f'Sampling frequency of {sfreq} is '
                 f'below the suggested rate of {MINIMUM_SUGGESTED_SFREQ}. '
                 f'Please use with caution.')

        self.sfreq = sfreq
        self.threshold_ = threshold
        self.wsize_ = wsize
        self.overlap_ = overlap

    @property
    def wsize(self):
        return self.wsize_

    @property
    def overlap(self):
        return self.overlap_

    @property
    def threshold(self):
        return self.threshold_

    def fit(self, X):
        n_chs, n_times = X.shape

        # Calculate window values for easier operation
        window_increment = int(np.ceil(self.wsize * self.overlap))

        output = []

        for idx in range(n_chs):
            sig = X[idx, :]

            # Overlapping window
            win_start = 0
            win_stop = self.wsize
            n_windows = int(np.ceil((len(sig) - self.wsize) / window_increment)) + 1
            LL = np.empty(n_windows)
            LL_i = 0
            while win_start < len(sig):
                if win_stop > len(sig):
                    win_stop = len(sig)

                LL[LL_i] = compute_line_length(sig[int(win_start):int(win_stop)],
                                               self.wsize)[0]

                if win_stop == len(sig):
                    break

                win_start += window_increment
                win_stop += window_increment

                LL_i += 1

            # Create threshold
            det_th = threshold_std(LL, self.threshold)

            # Detect
            LL_idx = 0
            while LL_idx < len(LL):
                if LL[LL_idx] >= det_th:
                    event_start = LL_idx * window_increment
                    while LL_idx < len(LL) and LL[LL_idx] >= det_th:
                        LL_idx += 1
                    event_stop = (LL_idx * window_increment) + self.wsize

                    if event_stop > len(sig):
                        event_stop = len(sig)

                    # Optional feature calculations can go here

                    # Write into output
                    output.append((event_start, event_stop))

                    LL_idx += 1
                else:
                    LL_idx += 1

        return output


class RMSDetector(BaseEstimator):
    """Root mean square (RMS) detection algorithm (Staba Detector).

    Parameters
    ----------
    sfreq: float
        Sampling frequency
    threshold: float
        Number of standard deviations to use as a threshold.
        Default = 3.
    wsize: int
        Sliding window size in samples. Default = 100.
    overlap: float
        Fraction of the window overlap (0 to 1). Default = 0.25.
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

    def __init__(self, sfreq: float, threshold: float = 3, wsize: int = 100,
                 overlap: float = 0.25, offset: int = 0):
        if sfreq < MINIMUM_SUGGESTED_SFREQ:
            warn(f'Sampling frequency of {sfreq} is '
                 f'below the suggested rate of {MINIMUM_SUGGESTED_SFREQ}. '
                 f'Please use with caution.')

        self.sfreq = sfreq
        self.threshold_ = threshold
        self.wsize_ = wsize
        self.overlap_ = overlap
        self.offset_ = offset

    @property
    def wsize(self):
        return self.wsize_

    @property
    def overlap(self):
        return self.overlap_

    @property
    def threshold(self):
        return self.threshold_

    def fit(self, X):
        # Calculate window values for easier operation
        window_increment = int(np.ceil(self.wsize * self.overlap))

        output = []

        # extract the number of chs and times for the dataset
        n_chs, n_times = X.shape

        # run HFO detection on all the channels
        for idx in range(n_chs):
            sig = X[idx, :]
            # Overlapping window
            win_start = 0
            win_stop = self.wsize
            n_windows = int(np.ceil((len(sig) - self.wsize) / window_increment)) + 1
            RMS = np.empty(n_windows)
            RMS_i = 0
            while win_start < len(sig):
                if win_stop > len(sig):
                    win_stop = len(sig)

                RMS[RMS_i] = compute_rms(sig[int(win_start):int(win_stop)],
                                         self.wsize)[0]

                if win_stop == len(sig):
                    break

                win_start += window_increment
                win_stop += window_increment

                RMS_i += 1

            # Create threshold
            det_th = threshold_std(RMS, self.threshold)

            # Detect
            RMS_idx = 0
            while RMS_idx < len(RMS):
                if RMS[RMS_idx] >= det_th:
                    event_start = RMS_idx * window_increment
                    while RMS_idx < len(RMS) and RMS[RMS_idx] >= det_th:
                        RMS_idx += 1
                    event_stop = (RMS_idx * window_increment) + self.wsize

                    if event_stop > len(sig):
                        event_stop = len(sig)

                    # Optional feature calculations can go here

                    # Write into output
                    output.append((event_start, event_stop))

                    RMS_idx += 1
                else:
                    RMS_idx += 1

        return output

    def score(self, raw, X):
        pass


class HilbertDetector(BaseEstimator):
    """2D HFO hilbert detection used in Kucewicz et al. 2014.

    Parameters
    ----------
    fs: float
        Sampling frequency of the signal
    low_fc: float
        Low cut-off frequency
    high_fc: float
        High cut-off frequency
    threshold: float
        Threshold for detection (default=3)
    band_spacing: str
        Spacing of hilbert freqeuncy bands - options: 'linear' or 'log'
        (default='linear'). Linear provides better frequency resolution but
        is slower.
    num_bands: int
        Number of bands if band_spacing = log (default=300)
    cyc_th: float
        Minimum number of cycles to detect (deafult=1)
    gap_th: float
        Number of cycles for gaps (default=1)
    mp: int
        Number of cores to use (default=1)
    sample_offset: int
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
                 offset: int = 0):
        self.sfreq = sfreq,
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.threshold_ = threshold
        self.band_method = band_method
        self.n_bands = n_bands
        self.cycle_threshold = cycle_threshold
        self.gap_threshold = gap_threshold
        self.n_jobs = n_jobs
        self.offset = offset

    @property
    def threshold(self):
        return self.threshold_

    def fit(self, X):
        n_chs, n_times = X.shape

        # Create output dataframe
        output = []

        # Construct filter cut offs

        if self.band_method == 'log':
            low_fc = float(self.l_freq)
            high_fc = float(self.h_freq)
            coffs = np.logspace(0, np.log10(high_fc), self.n_bands)
            coffs = coffs[(coffs > low_fc) & (coffs < high_fc)]
            freq_span = len(coffs) - 1
        elif self.band_method == 'linear':
            coffs = np.arange(self.l_freq, self.h_freq)
            freq_span = (self.h_freq - self.l_freq) - 1

        # Start the looping
        tdetects_concat = []
        if self.n_jobs > 1 or self.n_jobs == -1:
            # Run the filters in their threads and return the result
            iter_mat = [(sig, fs, i, coffs[i], coffs[i + 1],
                         cyc_th, gap_th, threshold) for i in range(freq_span)]
            tdetects_concat = work_pool.map(_band_z_score_detect, iter_mat)
        else:
            # OPTIMIZE - check if there is a better way to do this (S transform?+
            # spectra zeroing?)
            for i in range(freq_span):
                bot = coffs[i]
                top = coffs[i + 1]

                args = [sig, fs, i, bot, top, cyc_th, gap_th, threshold]

                tdetects_concat.append(_band_z_score_detect(args))

        # Process detects
        detects = np.array([det for band in tdetects_concat for det in band])

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
            freq_min = coffs[int(outline[0, 0])]
            freq_max = coffs[int(outline[-1, 0])]
            frequency_at_max = coffs[int(outline[np.argmax(outline[:, 3]), 0])]
            max_amplitude = max(outline[:, 3])

            output.append((start, stop,
                           freq_min, freq_max, frequency_at_max,
                           max_amplitude))

            # Plot the image
        # if plot_flag:
        #    f, axarr = plt.subplots(2, sharex = True)
        #    axarr[0].plot(data)
        #    plt.ion()
        #    axarr[1].imshow(hfa, aspect='auto',origin='lower')
        #    labels = [(i*100)+int(low_mat_fc) for i in range(int(np.ceil(
        # (high_mat_fc-low_mat_fc)/100.0)))]
        #    x_pos = [i*100 for i in range(int(np.ceil((high_mat_fc-low_mat_fc)/100)))]
        #    plt.yticks(x_pos,labels)
        #    plt.show(block=False)
        #    plt.waitforbuttonpress(1)

        if mp > 1:
            work_pool.close()

        return output

    def score(self, X, y):
        pass


def detect_hfo_hilbert(sig, fs, low_fc, high_fc, threshold=3,
                       window=10, window_overlap=0,
                       band_spacing='linear', num_bands=300,
                       cyc_th=1, gap_th=1, mp=1):
    # Create output dataframe
    output = []

    # Construct filter cut offs

    if band_spacing == 'log':
        low_fc = float(low_fc)
        high_fc = float(high_fc)
        coffs = np.logspace(0, np.log10(high_fc), num_bands)
        coffs = coffs[(coffs > low_fc) & (coffs < high_fc)]
        freq_span = len(coffs) - 1
    elif band_spacing == 'linear':
        coffs = np.arange(low_fc, high_fc)
        freq_span = (high_fc - low_fc) - 1

    # Create a pool of workers
    if mp > 1:
        work_pool = Pool(mp)

    # Start the looping
    tdetects_concat = []
    if mp > 1:

        # Run the filters in their threads and return the result
        iter_mat = [(sig, fs, i, coffs[i], coffs[i + 1],
                     cyc_th, gap_th, threshold) for i in range(freq_span)]
        tdetects_concat = work_pool.map(_band_z_score_detect, iter_mat)

    else:
        # OPTIMIZE - check if there is a better way to do this (S transform?+
        # spectra zeroing?)
        for i in range(freq_span):
            bot = coffs[i]
            top = coffs[i + 1]

            args = [sig, fs, i, bot, top, cyc_th, gap_th, threshold]

            tdetects_concat.append(_band_z_score_detect(args))

    # Process detects
    detects = np.array([det for band in tdetects_concat for det in band])

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
        freq_min = coffs[int(outline[0, 0])]
        freq_max = coffs[int(outline[-1, 0])]
        frequency_at_max = coffs[int(outline[np.argmax(outline[:, 3]), 0])]
        max_amplitude = max(outline[:, 3])

        output.append((start, stop,
                       freq_min, freq_max, frequency_at_max,
                       max_amplitude))

        # Plot the image
    # if plot_flag:
    #    f, axarr = plt.subplots(2, sharex = True)
    #    axarr[0].plot(data)
    #    plt.ion()
    #    axarr[1].imshow(hfa, aspect='auto',origin='lower')
    #    labels = [(i*100)+int(low_mat_fc) for i in range(int(np.ceil(
    # (high_mat_fc-low_mat_fc)/100.0)))]
    #    x_pos = [i*100 for i in range(int(np.ceil((high_mat_fc-low_mat_fc)/100)))]
    #    plt.yticks(x_pos,labels)
    #    plt.show(block=False)
    #    plt.waitforbuttonpress(1)

    if mp > 1:
        work_pool.close()

    return output


# =============================================================================
# Subfunctions
# =============================================================================


class HilbertDetector(Method):
    algorithm = 'HILBERT_DETECTOR'
    version = '1.0.0'
    dtype = [('event_start', 'int32'),
             ('event_stop', 'int32'),
             ('freq_min', 'float32'),
             ('freq_max', 'float32'),
             ('freq_at_max', 'float32'),
             ('max_amplitude', 'float32')]

    def __init__(self, **kwargs):
        """
        Slightly modified algorithm which uses the 2D HFO hilbert detection
        used in Kucewicz et al. 2014.

        Parameters
        ----------
        fs: float
            Sampling frequency of the signal
        low_fc: float
            Low cut-off frequency
        high_fc: float
            High cut-off frequency
        threshold: float
            Threshold for detection (default=3)
        band_spacing: str
            Spacing of hilbert freqeuncy bands - options: 'linear' or 'log'
            (default='linear'). Linear provides better frequency resolution but
            is slower.
        num_bands: int
            Number of bands if band_spacing = log (default=300)
        cyc_th: float
            Minimum number of cycles to detect (deafult=1)
        gap_th: float
            Number of cycles for gaps (default=1)
        mp: int
            Number of cores to use (default=1)
        sample_offset: int
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

        super().__init__(detect_hfo_hilbert, **kwargs)
        self._event_flag = True
