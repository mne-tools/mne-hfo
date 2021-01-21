"""MNE software for computing HFOs from iEEG data."""

__version__ = '0.1.dev0'

from mne_hfo.detect import RMSDetector, LineLengthDetector, HilbertDetector
from mne_hfo.hodetect import MorphologyDetector, CSDetector
from mne_hfo.io import create_events_df, read_events_tsv
