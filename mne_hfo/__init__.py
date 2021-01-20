"""MNE software for computing HFOs from iEEG data."""

__version__ = '0.1.dev0'

from .detect import RMSDetector, LineLengthDetector, HilbertDetector
from .hodetect import MorphologyDetector, CSDetector
from .io import create_events_df, read_events_tsv