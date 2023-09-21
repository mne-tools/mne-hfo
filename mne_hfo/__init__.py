"""MNE software for computing HFOs from iEEG data."""

from mne_hfo.base import Detector
from mne_hfo.detect import RMSDetector, LineLengthDetector, HilbertDetector
from mne_hfo.hodetect import MorphologyDetector, CSDetector
from mne_hfo.io import read_annotations, create_annotations_df, write_annotations
from mne_hfo.posthoc import (
    find_coincident_events,
    compute_chs_hfo_rates,
    match_detected_annotations,
    merge_overlapping_events,
)
from mne_hfo.compare import compare_detectors
