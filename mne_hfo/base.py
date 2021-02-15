from typing import Union

import mne
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, f1_score
from sklearn.utils.validation import check_is_fitted

from mne_hfo.io import create_events_df
from mne_hfo.utils import (threshold_std, compute_rms,
                           compute_line_length)

ACCEPTED_THRESHOLD_METHODS = ['std']
ACCEPTED_HFO_METHODS = ['line_length', 'rms']


class Detector(BaseEstimator):
    """Any sliding-window based HFO detector.

    Note: Detection will occur on all channels present.
    Subset your dataset before detecting.

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
    verbose: bool
    """

    def __init__(self, threshold: Union[int, float],
                 win_size: Union[int, None], overlap: Union[float, None],
                 scoring_func: str, n_jobs: int,
                 verbose: bool):
        self.win_size = win_size
        self.threshold = threshold
        self.overlap = overlap
        self.scoring_func = scoring_func
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _compute_hfo(self, X, picks):
        """Compute HFO event array.

        Takes a sliding window approach and computes the existence
        of an HFO defined by algorithm parameters. If an HFO is
        present, then a ``1`` will be in the array, otherwise
        a ``0`` will be in the array.

        Parameters
        ----------
        X : np.ndarray
            EEG data matrix (n_chs, n_times).
        picks : np.ndarray | list | None
            Corresponds to ``picks`` in mne-python.

        Returns
        -------
        hfo_event_arr : np.ndarray
            HFO event array that is (n_chs, n_windows).
        """
        raise NotImplementedError('Private function that computes the HFOs '
                                  'needs to be implemented.')

    def _compute_n_wins(self, win_size, step_size, n_times):
        n_windows = int(np.ceil((n_times - win_size) / step_size)) + 1
        return n_windows

    def fit_predict(self, X, y=None):
        """Perform fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
            (n_samples, n_features)

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X).predict(X)

    def score(self, X, y, sample_weight=None):
        r"""Return the score of the HFO prediction.

        The coefficient :math:`R^2` is defined as :math:`(1 - \\frac{u}{v})`,
        where :math:`u` is the residual sum of squares ``((y_true - y_pred)
        ** 2).sum()`` and :math:`v` is the total sum of squares ``((y_true -
        y_true.mean()) ** 2).sum()``. The best possible score is 1.0 and it
        can be negative (because the model can be arbitrarily worse). A
        constant model that always predicts the expected value of `y`,
        disregarding the input features, would get a :math:`R^2` score of
        0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """
        # fit and predict
        y_pred = self.fit_predict(X, y)

        # compute score
        if self.scoring_func == 'f1':
            score = f1_score(y, y_pred, sample_weight=sample_weight)
        elif self.scoring_func == 'r2':
            score = r2_score(y, y_pred, sample_weight=sample_weight)
        return score

    def _check_input_raw(self, X, y):
        if isinstance(X, mne.io.BaseRaw):
            self.sfreq = X.info['sfreq']
            self.ch_names = X.ch_names
            X = X.get_data()
        elif self.sfreq is not None:
            # Just name the channels their index
            self.ch_names = [str(i) for i in range(X.shape[0])]
            pass
        else:
            raise RuntimeError('If "X" passed in is not a `mne.io.BaseRaw` '
                               'object, then "sfreq" must be set on '
                               'instantation of detector.')

        # use sklearn's validation of data
        if y is None:
            X = self._validate_data(X, dtype='float64')
        else:
            X, y = self._validate_data(X, y, accept_sparse=False,
                                       dtype='float64',
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

    @property
    def hfo_event_arr(self):
        """HFO event array.

        Returns
        -------
        hfo_event_arr : np.ndarray
            Array that is (n_chs, n_samples), which has a
            value of ``1`` if
        """
        return self.hfo_event_arr_

    @property
    def chs_hfos_dict(self):
        """Return dictionary of HFO start/end points."""
        return self.chs_hfos_

    @property
    def chs_hfos_list(self):
        """Return list of HFO start/end points for each channel."""
        return [vals for vals in self.chs_hfos_dict.values()]

    @property
    def hfo_df(self):
        """Return HFO detections as a dataframe."""
        return self.df_

    def predict(self, X):
        """Scikit-learn override predict function.

        Just directly computes HFOs using ``fit`` function.
        """
        check_is_fitted(self)
        X, y = self._check_input_raw(X, None)
        return self.fit(X, None)

    def _create_event_df(self, chs_hfos_list, hfo_name):
        event_df = create_events_df(chs_hfos_list, sfreq=self.sfreq,
                                    event_name=hfo_name)
        self.df_ = event_df

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. In MNE-HFO, n_features
            are the number of time points in the EEG data, and n_samples
            are the number of channels.

        y : array-like of shape (n_samples, n_output)
            Target vector relative to X.

        picks : array-like of shape (n_samples,) default=None
            Corresponds to ``mne-python`` picks.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        All detectors use a sliding window to compute HFOs in windows.
        """
        raise NotImplementedError('')

    @property
    def step_size(self):
        """Step size of each window.

        Window increment over the samples of signal.
        """
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
        n_windows = self._compute_n_wins(self.win_size,
                                         self.step_size,
                                         self.n_times)

        # store the RMS of each window
        signal_win_rms = np.empty(n_windows)
        win_idx = 0
        while win_start < self.n_times:
            if win_stop > self.n_times:
                win_stop = self.n_times

            # compute the RMS of filtered signal in this window
            signal_win_rms[win_idx] = hfo_detect_func(
                sig[int(win_start):int(win_stop)], win_size=self.win_size)[0]

            if win_stop == self.n_times:
                break

            win_start += self.step_size
            win_stop += self.step_size
            win_idx += 1
        return signal_win_rms

    def _post_process_ch_hfos(self, metric_vals_list, n_times,
                              threshold_method='std'):
        """Post process one channel's HFO events.

        Joins contiguously detected HFOs as one event, and applies
        the threshold based on number of stdev above baseline on the
        RMS of the bandpass filtered signal.

        Parameters
        ----------
        metric_vals_list : list
            List of HFO metric values (e.g. Line Length, or RMS) over windows.
        n_times : int
            The number of time points in the original data matrix fed in.
        threshold_method : str
            The threshold method to use.

        Returns
        -------
        output : List[Tuple[int, int]]
            A list of tuples, storing the event start and stop sample index
            for the detected HFO.
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

        n_windows = len(metric_vals_list)

        # store post-processed hfo events as a list
        output = []

        # only keep RMS values above a certain number
        # stdevs above baseline (threshold)
        det_th = threshold_func(metric_vals_list, self.threshold)

        # Detect and now group events if they are within a
        # step size of each other
        win_idx = 0
        while win_idx < n_windows:
            # log events if they pass our threshold criterion
            if metric_vals_list[win_idx] >= det_th:
                event_start = win_idx * self.step_size

                # group events together if they occur in
                # contiguous windows
                while win_idx < n_windows and \
                        metric_vals_list[win_idx] >= det_th:
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
