from pathlib import Path
import json
import numpy as np
import pandas as pd
from natsort import natsorted

from mne_bids.path import BIDSPath, get_entities_from_fname
from mne_bids import read_raw_bids, get_entity_vals, mark_channels, update_sidecar_json

from mne_hfo import (RMSDetector, HilbertDetector,
                     LineLengthDetector, compute_chs_hfo_rates)
from mne_hfo.simulate import simulate_hfo


def _map_soz_chs_to_bids():
    """Map SOZ/RZ channels to the BIDS dataset.

    "We excluded all electrode contacts where electrical stimulation evoked motor or language responses".

    Reference: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-13064-1/MediaObjects/41598_2017_13064_MOESM1_ESM.pdf  # noqa
    """
    root = Path('/Users/adam2392/OneDrive - Johns Hopkins/ds003498')

    # get all subjects
    ignore_subjects = ['01',
                       #  '02',
                       # '03', '04', '05', '06'
                       ]
    subjects = get_entity_vals(
        root, 'subject', ignore_subjects=ignore_subjects)

    exclusion_contacts = {
        '11': ['IAR3', 'IAR4', 'IAR5', 'IAR6', 'PLR5', 'PLR6', 'PLR7', 'PLR8', 'PLR13', 'PLR14', 'PLR15', 'PLR16', 'PMR6', 'PMR7', 'PMR8', 'PMR13', 'PMR14', 'PMR15', 'PMR16'],
        '13': ['GR4', 'GR5', 'GR6', 'GR7', 'GR8', 'GR12', 'GR13', 'GR14', 'GR15', 'GR16', 'GR20', 'GR21', 'GR22', 'GR23', 'GR24'],
        '15': ['TLL6', 'TLL7', 'TLL14', 'TLL15', 'TLL16', 'TLL22', 'TLL23', 'TLL24', 'TLL31', 'TLL32'],
        '18': ['PM1', 'PM2', 'PMT3'],
        '19': ['OTL1', 'OTL2']
    }
    rz_chs_dict = {
        '01': [
            'AHR1', 'AHR2', 'AHR3', 'AHR4', 'AR1', 'AR2', 'AR3', 'AR4', 'PHR1', 'PHR2', 'PHR3', 'PHR4'],
        '02': ['AR1', 'AR2', 'AR3', 'AR4', 'ECR1', 'ECR2', 'ECR3', 'ECR4', 'PHR1', 'PHR2', 'PHR3', 'PHR4'],
        '03': ['AHL1', 'AHL2', 'AHL3', 'AHL4', 'AL1', 'AL2', 'AL3', 'AL4', 'ECL1', 'ECL2', 'ECL3', 'ECL4', 'PHL1', 'PHL2', 'PHL3', 'PHL4'],
        '04': ['AR1', 'AR2', 'AR3', 'AR4', 'ER1', 'ER2', 'ER3', 'ER4', 'HR1', 'HR2', 'HR3', 'HR4', 'PR1', 'PR2', 'PR3', 'PR4'],
        '05': ['AR1', 'AR2', 'AR3', 'AR4', 'ECR1', 'ECR2', 'ECR3', 'ECR4', 'PHR1', 'PHR2', 'PHR3', 'PHR4', 'AHR1', 'AHR2', 'AHR3', 'AHR4'],
        '06': ['AR1', 'AR2', 'AR3', 'AR4', 'ECR1', 'ECR2', 'ECR3', 'ECR4', 'AHR1', 'AHR2', 'AHR3', 'AHR4', 'PHR1', 'PHR2', 'PHR3', 'PHR4'],
        '07': ['AL1', 'AL2', 'AL3', 'AL4', 'AHL1', 'AHL2', 'AHL3', 'AHL4', 'ECL1', 'ECL2', 'ECL3', 'ECL4', 'PHL1', 'PHL2', 'PHL3', 'PHL4'],
        '08': ['AL1', 'AL2', 'AL3', 'AL4', 'ECL1', 'ECL2', 'ECL3', 'ECL4', 'AHL1', 'AHL2', 'AHL3', 'AHL4', 'PHL1', 'PHL2', 'PHL3', 'PHL4'],
        '09': ['AL1', 'AL2', 'AL3', 'AL4', 'EL1', 'EL2', 'EL3', 'EL4', 'HL1', 'HL2', 'HL3', 'HL4', 'PL1', 'PL2', 'PL3', 'PL4'],
        '10': ['IPR1', 'IPR2', 'IPR3', 'IPR4'],
        '11': ['TR1', 'TR2', 'TR3', 'TR4'],
        '12': ['GL1', 'GL2', 'GL3', 'GL9', 'GL10', 'GL11', 'GL12', 'GL13', 'GL17', 'GL18', 'GL19', 'GL20', 'GL21', 'GL22', 'GL25', 'GL26', 'GL27', 'GL28', 'GL29', 'GL30', 'GL31', 'GL32', 'TL1', 'TL2', 'TL3', 'TL4'],
        '13': ['TR1', 'TR2', 'TR3', 'TR4'],
        '14': ['IPR3', 'IPR4'],
        '15': ['TBAL2', 'TBAL3', 'TBAL4', 'TLL1', 'TLL2', 'TLL9', 'TLL10'],
        '16': ['TL1', 'TL2', 'TL3', 'TL4'],
        '17': ['FAR1', 'FAR2', 'FAR3', 'FAR4', 'FAR5', 'FAR6', 'FAR7', 'FAR8', 'FAR9', 'FAR10', 'FAR11', 'FAR12', 'FAR13', 'FAR14', 'FAR15', 'FAR16', 'FPR3', 'FPR4', 'FPR5', 'FPR6', 'FPR7', 'FPR8', 'FPR11', 'FPR12', 'FPR13', 'FPR14', 'FPR15', 'FPR16', 'TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'TR6'],
        '18': ['TL1', 'TL2', 'TL3', 'TL4', 'TL5'],
        '19': ['TL1', 'TL2', 'TL3', 'TL4', 'TL9', 'TL10', 'TL11', 'TL12'],
        '20': ['OTL4', 'OTL5', 'OTL6', 'OTL7', 'OTL13', 'OTL14', 'OTL15']
    }

    for subject in subjects:
        bids_path = BIDSPath(
            subject=subject, suffix='channels', extension='.tsv', root=root)
        channels_fpaths = bids_path.match()
        for channel_fpath in channels_fpaths:
            print(f'Modifying {subject} {channel_fpath}')
            # add a new column
            ch_names = rz_chs_dict[subject]
            mark_channels(ch_names, descriptions=['resected'] * len(ch_names),
                          bids_path=channel_fpath, status='good', verbose=False)


def analyze_zurich():
    """Use for analyzing Zurich dataset."""
    root = Path('/Users/adam2392/OneDrive - Johns Hopkins/ds003498')

    # get all subjects
    ignore_subjects = ['01', '02',
                       '03']
    subjects = get_entity_vals(
        root, 'subject', ignore_subjects=ignore_subjects)
    datatype = 'ieeg'
    overwrite = False

    # get outcomes
    part_df = pd.read_csv(root / 'participants.tsv', sep='\t')

    for subject in subjects:
        subj_dir = root / f'sub-{subject}'

        outcome = part_df[part_df['participant_id'] ==
                          f'sub-{subject}']['outcome']

        # get all file paths
        fpaths = natsorted(list(subj_dir.rglob('*.vhdr')))

        # loop through each file path and compute HFOs
        for bids_fpath in fpaths:
            bids_path = BIDSPath(root=root, datatype=datatype,
                                 **get_entities_from_fname(bids_fpath))
            raw = read_raw_bids(bids_path)

            # filter line noise
            raw.load_data()
            line_freq = raw.info['line_freq']
            sfreq = raw.info['sfreq']
            freqs = np.arange(line_freq, sfreq // 2, line_freq)
            raw = raw.notch_filter(freqs=freqs)

            # run filtering
            for name, detect_func in zip(['rms', 'linelength',
                                          #   'hilbert'
                                          ],
                                         [RMSDetector, LineLengthDetector,
                                          #  HilbertDetector
                                          ]):
                fname = bids_path.copy().update(
                    suffix=f'desc-{name}_ieeg', extension='.tsv', check=False)
                if fname.fpath.exists() and not overwrite:
                    continue

                detector = detect_func()
                detector.fit(raw)

                # get the annotations
                annot_df = detector.to_data_frame(format='bids')

                # save it to derivatives
                annot_df.to_csv(fname, sep='\t', index=False)


if __name__ == '__main__':
    # _map_soz_chs_to_bids()
    analyze_zurich()
