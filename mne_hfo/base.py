from typing import Union

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from mne.utils import warn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from mne import Annotations

from mne_hfo.config import MINIMUM_SUGGESTED_SFREQ
from mne_hfo.io import create_annotations_df
from mne_hfo.score import accuracy, false_negative_rate, \
    true_positive_rate, precision, false_discovery_rate
from mne_hfo.sklearn import _make_ydf_sklearn
from mne_hfo.utils import (apply_std, compute_rms,
                           compute_line_length, compute_hilbert, apply_hilbert,
                           merge_contiguous_freq_bands)

ACCEPTED_THRESHOLD_METHODS = ['std', 'hilbert']
ACCEPTED_MERGE_METHODS = ['time-windows', 'freq-bands']
ACCEPTED_HFO_METHODS = ['line_length', 'rms', 'hilbert']


class BaselineMethod:
    def compute_baseline(self):
        raise NotImplementedError(
            'All methods that compute a baseline should have this method implemented.')


class Detector(BaseEstimator):
    """Any sliding-window based HFO detector.

    Note: Detection will occur on all channels present.
    Subset your dataset before detecting.

    Detectors fit follow the following general flow by implementing
    private functions:
        1. Compute a statistic on the raw data in ``compute_hfo_statistic``.
            i.e. the LineLength of a time-window
        2. Apply a threshold to the statistic computed in (1) in
            ``threshold_hfo_statistic``. i.e. std of LineLength
        3. Merge contiguous/overlapping events into unique detections
            in ``post_process_chs_hfo``. i.e. contiguous time windows

    Parameters
    ----------
    threshold: float
        Number of standard deviations to use as a threshold.
    win_size: int
        Sliding window size in samples.
    overlap: float
        Fraction of the window overlap (0 to 1).
    scoring_func : str
        Either ``'f1'``, or ``'r2'``.
    name : str
        The name of the HFO detector.
    n_jobs : int
        The number of jobs used in `joblib` parallelization.
    verbose: bool
    """

    def __init__(self, threshold: Union[int, float],
                 win_size: Union[int, None], overlap: Union[float, None],
                 scoring_func: str, name: str, n_jobs: int,
                 verbose: bool):
        self.win_size = win_size
        self.threshold = threshold
        self.overlap = overlap
        self.scoring_func = scoring_func
        self.name = name
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _create_empty_event_arr(self):
        """Create an empty HFO event array.

        Assumes there is only one set of frequency cutoffs (i.e. one
        frequency band) right now.

        Returns
        -------
        hfo_event_arr : np.ndarray (n_chs, n_windows, n_bands)
            An array that consists of channels X windows X
            frequency bands.
        """
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)
        if self.filter_band is not None:
            self.freq_cutoffs = np.array([self.filter_band[0],
                                          self.filter_band[1]])
        else:
            self.freq_cutoffs = np.array([30, 500])
        n_bands = len(self.freq_cutoffs) - 1
        hfo_event_arr = np.empty((self.n_chs, n_windows, n_bands))
        return hfo_event_arr

    def compute_hfo_statistic(self, X):
        """Compute HFO statistic.

        Takes a sliding window approach and computes the existence
        of an HFO defined by algorithm parameters. If an HFO is
        present, then a ``1`` will be in the array, otherwise
        a ``0`` will be in the array.

        Parameters
        ----------
        X : np.array
            EEG data array for single channel: N = n_times.

        Returns
        -------
        hfo_event_arr : np.ndarray
            HFO event array that is (n_chs, n_windows). It is a boolean mask
            that consists of either 1's and 0's, or True's and False's.
        """
        raise NotImplementedError('Private function that computes the HFOs '
                                  'needs to be implemented.')

    def threshold_hfo_statistic(self, hfo_statistic_arr):
        """Apply threshold(s) to the calculated statistic to generate hfo events.

        Parameters
        ----------
        hfo_statistic_arr: np.ndarray
            The output of compute_hfo_statistic

        Returns
        -------
        hfo_event_array: np.ndarray
            HFO event array that contains (at minimum) a series of start
            and stop times.
        """
        raise NotImplementedError('Private function that computes the HFOs '
                                  'needs to be implemented.')

    def post_process_ch_hfos(self, hfo_event_array):
        """Post process one channel's HFO events generally after thresholding.

        Joins contiguously detected HFOs as one event.

        Parameters
        ----------
        hfo_event_array : np.ndarray
            List of HFO metric values (e.g. Line Length, or RMS) over windows
            for a specific channel.

        Returns
        -------
        output : List[Tuple[int, int]]
            A list of tuples, storing the event start and stop sample index
            for the detected HFO.
        """
        raise NotImplementedError('Private function that computes the HFOs '
                                  'needs to be implemented.')

    def _compute_n_wins(self, win_size, step_size, n_times):
        n_windows = int(np.ceil((n_times - win_size) / step_size)) + 1
        return n_windows

    def _check_input_raw(self, X, y):
        if isinstance(X, mne.io.BaseRaw):
            X.shape = (len(X.ch_names), len(X))
            self.sfreq = X.info['sfreq']
            self.ch_names = X.ch_names
            X = X.get_data()
        elif isinstance(X, pd.DataFrame):
            '''Handle the case of SearchCV'''
            ch_names = X.index

            # Dataframe was transposed
            time_index = X.T.index

            # compute the time indices and get the pairwise
            # differences
            time_indices = time_index.to_series().to_numpy()
            diff = np.diff(time_indices)

            # compute periods and the sampling rate
            periods = [x.total_seconds() for x in diff]
            if not np.all(np.isclose(periods, np.median(periods),
                                     rtol=1e-3, atol=1e-5)):
                raise RuntimeError('Not all sampling periods of the '
                                   'raw data are similar...')
            sfreq = 1. / periods[0]

            self.ch_names = ch_names
            self.sfreq = sfreq
            X = X.to_numpy()
        else:
            if not hasattr(self, 'ch_names'):
                self.ch_names = np.arange(len(X)).astype(str)
            # pass
            # raise ValueError(f'Only dataframe and mne.io.Raw input is '
            #                  f'accepted into HFO detectors.')

        # use sklearn's validation of data
        if y is None:
            X = self._validate_data(X, dtype='float64')
        else:
            X, y = self._validate_data(X, y, accept_sparse=False,
                                       dtype='float64',
                                       multi_output=True,
                                       accept_large_sparse=False)

        self.n_chs, self.n_times = X.shape
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)

        if n_windows < 0:
            raise ValueError(f'Negative dimensions are not allowed. '
                             f'This is probably due to there being '
                             f'n_features=1 (sample point) in the dataset. '
                             f'Current data segment has shape {X.shape}. '
                             f'Pass in a longer data segment.')

        # if the number of time points is smaller then the window size
        # then raise an Error
        if self.n_times < self.win_size:
            raise ValueError(f'Got data matrix with {self.n_times} sample '
                             f'points, which is less then {self.win_size} '
                             f'window size. Please pass in a longer segment.')

        return X, y

    def fit_predict(self, X, y=None):
        """Perform fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
            (n_samples, n_features)
            The input data.
        y : np.ndarray
            Not used, present for API consistency by convention.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X).predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the score of the HFO prediction.

        Parameters
        ----------
        X : np.ndarray
            Channel data to detect HFOs on.
        y : DataFrame
            Event Dataframe of true labels.
        sample_weight : np.ndarray
            Not used.

        Returns
        -------
        score : float
            The score relative to true annotations we want.
        """
        # y_true should be an annotations DataFrame actually
        # fit and predict
        y_pred = self.fit_predict(X, y)

        # match predictions with reference dataframe
        # call `match_detections`. This should
        # return y_true, y_pred, which are just lists of 0s and 1s
        # representing overlap detection or not

        # compute score
        if self.scoring_func == "accuracy":
            score = accuracy(y, y_pred)
        elif self.scoring_func == "fnr":
            score = false_negative_rate(y, y_pred)
        elif self.scoring_func == "tpr":
            score = true_positive_rate(y, y_pred)
        elif self.scoring_func == "precision":
            score = precision(y, y_pred)
        elif self.scoring_func == "fdr":
            score = false_discovery_rate(y, y_pred)
        return score

    @property
    def hfo_annotations(self):
        """HFO Annotations.

        Returns
        -------
        hfo_annotations : instance of Annotations
            `mne.Annotations` object with ``onset``, ``duration``
            and specified ``ch_name`` for each HFO event detected.
        """
        return self.hfo_annotations_

    @property
    def hfo_event_arr(self):
        """HFO event array.

        Returns
        -------
        hfo_event_arr : np.ndarray
            Array that is (n_chs, n_samples), which has a
            value of ``1`` if there is an HFO in that sample.
        """
        return self.hfo_event_arr_

    @property
    def step_size(self):
        """Step size of each window.

        Window increment over the samples of signal.
        """
        # Calculate window values for easier operation
        return int(np.ceil(self.win_size * self.overlap))

    def to_data_frame(self, format=None):
        """Export HFO annotations in tabular structure as a pandas DataFrame.

        Parameters
        ----------
        format : str | None
            If None (default), will return a `mne.Annotations` object as a
            DataFrame. If 'bids', then will return a DataFrame formatted
            with columns names as an 'annotations.tsv' derivatve file.

        Returns
        -------
        result : pandas.DataFrame
            Returns a pandas DataFrame with onset, duration, and description
            columns. A column named ch_names is added if any annotations are
            channel-specific.
        """
        if format is None:
            return self.hfo_annotations.to_data_frame()
        elif format == 'bids':
            # format as an annots.tsv DataFrame
            annots = self.hfo_annotations
            onset = annots.onset
            duration = annots.duration
            label = annots.description

            # each annotation only has one channel associated with it
            ch_names = [ch[0] for ch in annots.ch_names]

            # create an annotations.tsv dataframe
            df = create_annotations_df(
                onset, duration, ch_names,
                annotation_label=label, sfreq=self.sfreq)
            return df

    def fit(self, X, y=None):
        """Fit the model according to the optionally given training data.

        Parameters
        ----------
        X : mne.io.Raw of shape (n_samples, n_features) | pd.DataFrame
            Training vector, where n_samples is the number of samples and
            ``n_features`` is the number of features. In MNE-HFO,
            ``n_features`` are the number of time points in the EEG data,
            and n_samples are the number of channels.

        y : array-like of shape (n_samples, n_output)
            Target vector relative to X.

        Returns
        -------
        self : Detector
            Fitted estimator.

        Notes
        -----
        All detectors use a sliding window to compute HFOs in windows.
        """
        X, y = self._check_input_raw(X, y)

        sfreq = self.sfreq
        if sfreq < MINIMUM_SUGGESTED_SFREQ:
            warn(f'Sampling frequency of {sfreq} is '
                 f'below the suggested rate of {MINIMUM_SUGGESTED_SFREQ}. '
                 f'Please use with caution.')

        # store HFO events as list of mne.annotations
        hfo_description = 'hfo'
        hfo_annotations = []

        self.hfo_event_arr_ = self._create_empty_event_arr()
        if self.n_jobs == 1:
            for idx in tqdm(range(self.n_chs)):
                sig = X[idx, :]
                ch_name = self.ch_names[idx]

                # compute HFOs for this channel
                ch_hfo_events, statistic = self.fit_channel(
                    sig, sfreq, ch_name, hfo_description=hfo_description)

                # create list of annotations
                hfo_annotations.append(ch_hfo_events)
                self.hfo_event_arr[idx, :, :] = statistic

        else:
            if self.n_jobs == -1:
                n_jobs = cpu_count()
            else:
                n_jobs = self.n_jobs

            # run joblib parallelization over channels
            ch_hfos, statistics = zip(*Parallel(n_jobs=n_jobs)(
                delayed(self.fit_channel)(
                    X[idx, :], sfreq, self.ch_names[idx], hfo_description
                ) for idx in tqdm(range(self.n_chs))
            ))
            for idx in range(len(ch_hfos)):
                # chs_hfos[self.ch_names[idx]] = ch_hfos[idx]
                hfo_annotations.append(ch_hfos[idx])
                self.hfo_event_arr[idx, :, :] = statistics[idx]

        # merge HFO annotation list into one Annotations data structure
        all_hfo_annots = hfo_annotations[0]
        for idx in range(1, len(hfo_annotations)):
            all_hfo_annots += hfo_annotations[idx]

        # assign annotations object
        self.hfo_annotations_ = all_hfo_annots
        self.chs_hfos_ = all_hfo_annots
        return self

    def fit_channel(self, sig, sfreq, ch_name, hfo_description='hfo'):
        """Compute a list of HFO events for channel."""
        # compute the metric over the signal used to compute the HFO
        # e.g. RMS, or Line Length over time
        hfo_statistic_arr = self.compute_hfo_statistic(sig)

        # apply the threshold(s) to the statistic to get detections
        # of start and stop samples
        hfo_detection_arr = self.threshold_hfo_statistic(hfo_statistic_arr)

        # (optionally) post process HFOs
        ch_hfo_list = self.post_process_ch_hfos(hfo_detection_arr)

        # extract onset, and durations of each HFO detected to form Annotations
        onset, duration = [], []
        for (start_sample, stop_sample) in ch_hfo_list:
            onset.append(start_sample / sfreq)
            duration.append((stop_sample - start_sample) / sfreq)

        # create Annotations object
        description = [hfo_description] * len(onset)
        ch_names = [[ch_name] for _ in range(len(onset))]
        ch_hfo_events = Annotations(onset=onset, duration=duration,
                                    description=description,
                                    ch_names=ch_names)
        return ch_hfo_events, hfo_statistic_arr

    def _apply_threshold(self, metric, threshold_method):
        """Apply the threshold(s) to the calculated metric for a single channel.

        Parameters
        ----------
        metric : np.ndarray
            The single channel values to check against a threshold
        threshold_method : str
            The type of threshold to use
        Returns
        -------
        thresholded_metric: np.ndarray
            Metric values that pass the given threshold

        """
        if threshold_method not in ACCEPTED_THRESHOLD_METHODS:
            raise ValueError(f'Threshold method {threshold_method} '
                             f'is not an implemented threshold method. '
                             f'Please use one of {ACCEPTED_THRESHOLD_METHODS} '
                             f'methods.')
        if threshold_method == 'std':
            threshold_func = apply_std
            threshold_dict = dict(thresh=self.threshold)
            kwargs = dict(step_size=self.step_size,
                          win_size=self.win_size,
                          n_times=self.n_times)
        elif threshold_method == 'hilbert':
            threshold_func = apply_hilbert
            threshold_dict = dict(zscore=self.threshold,
                                  cycles=self.cycle_threshold,
                                  gap=self.gap_threshold)
            kwargs = dict(n_times=self.n_times,
                          sfreq=self.sfreq,
                          filter_band=self.filter_band,
                          freq_cutoffs=self.freq_cutoffs,
                          freq_span=self.freq_span,
                          n_jobs=self.n_jobs)

        if self.verbose:
            print(f'Using {threshold_method} to perform HFO '
                  f'thresholding.')

        thresholded_metric = threshold_func(metric, threshold_dict, kwargs)
        return thresholded_metric

    def predict(self, X):
        """Scikit-learn override predict function.

        Just directly computes HFOs using ``fit`` function.

        Parameters
        ----------
        X : mne.io.Raw | pd.DataFrame
            Input data.

        Returns
        -------
        ypred : list[list[tuple]]
            List of HFO events per channel in order of ``ch_names`` of
            input data. HFO events are stored as list of tuples: onset
            and offset of the HFO event.
        """
        check_is_fitted(self)
        self.fit(X, None)
        ypred = _make_ydf_sklearn(
            self.to_data_frame(format='bids'), ch_names=self.ch_names)
        return ypred

    def _compute_sliding_window_detection(self, sig, method):
        """Compute detections on an individual channel data using a sliding window.

        If the method does not use sliding windows, make win_size
        equal to the length of the dataset.

        Parameters
        ----------
        sig: np.ndarray
            Data (1D array) from a single channel
        method: str
            Method used to compute the detection. Can be one of
            ``'line_length', 'rms', 'hilbert'``.

        Returns
        -------
        signal_win_stat: np.ndarray, shape (n_chs, n_windows)
            Statistic calculated per window, where the number of
            windows is equal to ``(n_samples - win_size) / step_size``.
        """
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
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)

        # store the statistic of each window
        signal_win_stat = np.empty(n_windows)
        win_idx = 0
        while win_start < self.n_times:
            if win_stop > self.n_times:
                win_stop = self.n_times

            # compute the statistic based on 'method' on filtered signal
            # in this window
            stat = hfo_detect_func(
                sig[int(win_start):int(win_stop)], win_size=self.win_size)[0]
            signal_win_stat[win_idx] = stat

            if win_stop == self.n_times:
                break

            win_start += self.step_size
            win_stop += self.step_size
            win_idx += 1
        return signal_win_stat

    def _compute_freq_band_detection(self, sig, method):
        if method not in ACCEPTED_HFO_METHODS:
            raise ValueError(f'Sliding window HFO detection method '
                             f'{method} is not implemented. Please '
                             f'use one of {ACCEPTED_HFO_METHODS}.')
        if method == 'hilbert':
            hfo_detect_func = compute_hilbert
        signal_stat = hfo_detect_func(sig, self.freq_cutoffs,
                                      self.freq_span, self.sfreq)
        return signal_stat

    def _merge_contiguous_ch_detections(self, detections, method):
        """Merge contiguous hfo detections into distinct events.

        Parameters
        ----------
        detections : List(tuples)
            List of raw hfo detected events
        method : str
            Method to use to merge the detections. Can be one of
            ``'time-windows', 'freq-bands'``.

        Returns
        -------
        events: List(tuples)
            List of start and stop times of the distinct HFO events.

        """
        if method not in ACCEPTED_MERGE_METHODS:
            raise ValueError(f'Merging method {method} '
                             f'is not an implemented merging method. '
                             f'Please use one of {ACCEPTED_MERGE_METHODS} '
                             f'methods.')
        if method == "time-windows":
            return detections
        elif method == "freq-bands":
            merge_func = merge_contiguous_freq_bands

        events = merge_func(detections)
        if method == "freq-bands":
            events, max_amplitude, freq_bands = events
            self.hfo_max_amplitudes_ = max_amplitude
            self.hfo_freq_bands_ = freq_bands

        return events
