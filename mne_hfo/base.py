import numpy as np
from sklearn.base import BaseEstimator

from mne_hfo.utils import (threshold_std, compute_rms, compute_line_length)

ACCEPTED_THRESHOLD_METHODS = ['std']
ACCEPTED_HFO_METHODS = ['line_length', 'rms']


class Detector(BaseEstimator):
    def __init__(self, threshold: int, win_size: int, overlap: float,
                 verbose: bool = True):
        """Base class for any HFO detector.

        Parameters
        ----------
        threshold: float
            Number of standard deviations to use as a threshold.
        win_size: int
            Sliding window size in samples.
        overlap: float
            Fraction of the window overlap (0 to 1).
        verbose: bool
        """
        self._win_size = win_size
        self._threshold = threshold
        self._overlap = overlap
        self.verbose = verbose

        # store all HFO events found
        self.hfo_event_arr = None

    @property
    def win_size(self):
        return self._win_size

    @property
    def overlap(self):
        return self._overlap

    @property
    def threshold(self):
        return self._threshold

    @property
    def step_size(self):
        # Calculate window values for easier operation
        return int(np.ceil(self.win_size * self.overlap))

    def _compute_sliding_window_detection(self, sig, method):
        if method not in ACCEPTED_HFO_METHODS:
            raise ValueError(f'Sliding window HFO detection method '
                             f'{method} is not implemented. Please '
                             f'use one of {ACCEPTED_HFO_METHODS}.')

        if method == 'rms':
            hfo_detect_func = compute_rms
        elif method == 'line_length':
            hfo_detect_func = compute_line_length

        # Overlapping window
        win_start = 0
        win_stop = self.win_size
        n_windows = int(np.ceil((len(sig) - self.win_size) / self.step_size)) + 1

        # store the RMS of each window
        signal_win_rms = np.empty(n_windows)
        win_idx = 0
        while win_start < len(sig):
            if win_stop > len(sig):
                win_stop = len(sig)

            # compute the RMS of filtered signal in this window
            signal_win_rms[win_idx] = hfo_detect_func(sig[int(win_start):int(win_stop)],
                                                      win_size=self.win_size)[0]
            if win_stop == len(sig):
                break

            win_start += self.step_size
            win_stop += self.step_size
            win_idx += 1
        return signal_win_rms

    def _post_process_ch_hfos(self, ch_hfo_events, n_times,
                              threshold_method='std'):
        """Post process one channel's HFO events.

        Joins contiguously detected HFOs as one event, and applies
        the threshold based on number of stdev above baseline on the
        RMS of the bandpass filtered signal.
        """
        if threshold_method not in ACCEPTED_THRESHOLD_METHODS:
            raise ValueError(f'Threshold method {threshold_method} '
                             f'is not an implemented threshold method. '
                             f'Please use one of {ACCEPTED_THRESHOLD_METHODS} '
                             f'methods.')
        if threshold_method == 'std':
            threshold_func = threshold_std

        if self.verbose:
            print(f'Using {threshold_method} to perform HFO '
                  f'thresholding.')

        n_windows = len(ch_hfo_events)

        # store post-processed hfo events as a list
        output = []

        # only keep RMS values above a certain number
        # stdevs above baseline (threshold)
        det_th = threshold_func(ch_hfo_events, self.threshold)

        # Detect and now group events if they are within a
        # step size of each other
        win_idx = 0
        while win_idx < n_windows:
            # log events if they pass our threshold criterion
            if ch_hfo_events >= det_th:
                event_start = win_idx * self.step_size

                # group events together if they occur in
                # contiguous windows
                while win_idx < n_windows and \
                        ch_hfo_events >= det_th:
                    win_idx += 1
                event_stop = (win_idx * self.step_size) + self.win_size

                if event_stop > n_times:
                    event_stop = n_times

                # TODO: Optional feature calculations

                # Write into output
                output.append((event_start, event_stop))
                win_idx += 1
            else:
                win_idx += 1
        return output
