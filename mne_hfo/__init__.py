"""MNE software for computing HFOs from iEEG data."""
from ._version import __version__
from .base import Detector
from .compare import compare_detectors
from .detect import HilbertDetector, LineLengthDetector, RMSDetector
from .hodetect import CSDetector, MorphologyDetector
from .io import create_annotations_df, read_annotations, write_annotations
from .posthoc import (
    compute_chs_hfo_rates,
    find_coincident_events,
    match_detected_annotations,
    merge_overlapping_events,
    merge_channel_events
)
