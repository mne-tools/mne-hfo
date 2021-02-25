"""MNE software for computing HFOs from iEEG data."""

__version__ = '0.1'

from mne_hfo.detect import RMSDetector, LineLengthDetector, HilbertDetector
from mne_hfo.hodetect import MorphologyDetector, CSDetector
from mne_hfo.io import (create_events_df, read_annotations,
                        create_annotations_df, events_to_annotations,
                        write_annotations)
from mne_hfo.posthoc import (
    find_coincident_events, compute_chs_hfo_rates,
    match_detected_annotations, merge_overlapping_events)
