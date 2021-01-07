"""MNE software for computing HFOs from iEEG data."""

__version__ = '0.1.dev0'
from mne_hfo import commands

from .detect import RMSDetector, LineLengthDetector, HilbertDetector
from .hodetect import MorphologyDetector, CSDetector