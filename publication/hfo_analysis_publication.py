from pathlib import Path
from mne.filter import _overlap_add_filter

import numpy as np
import pandas as pd
from mne_bids import read_raw_bids, get_entity_vals

from mne_hfo import (RMSDetector, compute_chs_hfo_rates,
                     events_to_annotations)
from mne_hfo.simulate import simulate_hfo


def analyze_zurich():
    root = Path('/Users/adam2392/OneDrive - Johns Hopkins/ds003498')

    # get all subjects
    subjects = get_entity_vals(root, 'subject')

    # get outcomes
    part_df = pd.read_csv(root / 'participants.tsv', sep='\t')

    for subject in subjects:
        subj_dir = root / f'sub-{subject}'

        outcome = part_df[part_df['participant_id'] == f'sub-{subject}']['outcome']

        # get all file paths
        fpaths = []

        # loop through each file path and compute HFOs
        for fpath in fpaths:

            # 