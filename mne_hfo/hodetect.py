import mne
import numpy as np
from scipy.signal import hilbert

from mne_hfo.base import Detector


class CSDetector(Detector):
    """CS Detector.

    Not implemented.
    """

    def __init__(self):
        pass


class MorphologyDetector(Detector):  # noqa
    """Morphology detector.

    Detection phase looks for ripples (80-250 Hz),
    and fast ripples (250-500 Hz). Using two FIR
    bandpass filters.

    The baseline was defined as the wavelet entropy of
    the Stockwell-transform. Then a segment is considered
    baseline when the Stockwell entropy was larger then 90%
    of the Stockwell entropy maximum theoretical limit.

    The next step detects HFOs.

    """

    def __init__(self, sfreq: int, l_freq: int, h_freq: int, entropy_threshold: float = 0.9):
        self.sfreq = sfreq
        self.ripple_l_freq = 80
        self.ripple_h_freq = 250
        self.fr_l_freq = 250
        self.fr_h_freq = 500

        self.entropy_threshold = entropy_threshold

        # threshold in ms for ripple and fast ripple
        self.ripple_threshold = 20
        self.fr_threshold = 10

        """
        HFOobj.BLborder  = 0.02; % sec, ignore borders of 1 sec interval because of ST transform
        HFOobj.BLmindist = 10*p.fs/1e3; % pt, min disance interval for baseline in po
        HFOobj.dur       = input.dur; % number os seconds to take for baseline detection
        """

    def _detect_baseline(self, X, fmin, fmax, entropy_threshold):
        # compute stockwell transform: C x F x T
        st_power, freqs = mne.time_frequency.tfr_stockwell(X, fmin=fmin, fmax=fmax)

        # get the total power: F x T
        st_power_total = np.sum(st_power, axis=0)

        # compute relative energy to convert into "probability" distribution for each frequency
        # should result in F x T array
        rel_st_energy = np.divide(st_power, st_power_total)

        # compute total entropy per frequency
        freq_entropy = -np.sum(rel_st_energy * np.log2(rel_st_energy), axis=1)

        # get the entropy threshold per freq
        entropy_freq_threshold = freq_entropy * entropy_threshold

        # apply threshold and baseline
        freq_ent_thresh = freq_entropy > entropy_freq_threshold

        # determine length of each epoch that passes the threshold

    def fit(self, X, y=None):
        """Override ``Detector.fit`` function."""
        # create a copy of the ripple data
        ripple_data = mne.filter.filter_data(X, sfreq=self.sfreq,
                                             l_freq=self.ripple_l_freq,
                                             h_freq=self.ripple_h_freq,
                                             method='fir',
                                             copy=True)

        # create a copy of the fast ripple data
        fr_data = mne.filter.filter_data(X, sfreq=self.sfreq,
                                         l_freq=self.fr_l_freq,
                                         h_freq=self.fr_h_freq,
                                         method='fir',
                                         copy=True)

        # compute the Hilbert transform envelope
        # (i.e. the envelope)
        hfx = np.abs(hilbert(ripple_data))

        pass
