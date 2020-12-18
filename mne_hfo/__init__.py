"""MNE software for easily interacting with BIDS compatible datasets."""

__version__ = '0.7.dev0'
from mne_hfo import commands
from mne_hfo.report import make_report
from mne_hfo.path import (BIDSPath, get_datatypes, get_entity_vals,
                          print_dir_tree, get_entities_from_fname)
from mne_hfo.read import get_head_mri_trans, read_raw_bids
from mne_hfo.utils import (get_anonymization_daysback)
from mne_hfo.write import (make_dataset_description, write_anat,
                           write_raw_bids, mark_bad_channels,
                           write_meg_calibration, write_meg_crosstalk)
from mne_hfo.sidecar_updates import update_sidecar_json
from mne_hfo.inspect import inspect_dataset
