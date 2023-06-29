import json
import os
import platform
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas
import pandas as pd
from mne.utils import run_subprocess
from mne_bids import read_raw_bids, get_bids_path_from_fname, BIDSPath

from mne_hfo.config import ANNOT_COLUMNS


def _bids_validate(bids_root):
    """Run BIDS validator."""
    vadlidator_args = ['--config.error=41']
    exe = os.getenv('VALIDATOR_EXECUTABLE', 'bids-validator')

    if platform.system() == 'Windows':
        shell = True
    else:
        shell = False

    bids_validator_exe = [exe, *vadlidator_args]

    cmd = [*bids_validator_exe, bids_root]
    run_subprocess(cmd, shell=shell)


def create_annotations_df(onset: List[float], duration: List[float],
                          ch_name: List[str], sfreq: Union[float, List[float]],
                          annotation_label: Optional[List[str]] = None) \
        -> pd.DataFrame:
    """Create a BIDS-derivative annotations dataframe for HFO events.

    Parameters
    ----------
    onset : list of float
        Onset time in seconds.
    duration : list of float
        Duration in seconds.
    ch_name : list of str
        The name of the event to add to the "trial_type" column.
    sfreq : list of float | float
        The sample rate for each channel.
    annotation_label : list of str | None
        List of annotation labels to label the annotation. If None (default)
        then will be ``"HFO"``.

    Returns
    -------
    annot_df : DataFrame
        The annotations dataframe according to BIDS-Derivatives [1].

    Notes
    -----
    For many post-hoc operations, it will be required to know the sampling
    rate of the data. In order to compute that, it is recommended to take
    the sampling rate and multiply it with the ``'onset'`` column to get
    a new ``'sample'`` column denoting the sample point each HFO occurs at.

    References
    ----------
    .. [1] https://bids-specification.readthedocs.io/en/stable/
    """
    if duration is None:
        duration = [0] * len(onset)
    if len(onset) != len(ch_name) or \
            len(onset) != len(duration):
        msg = (f'Length of "onset", "description", '
               f'"duration", need to be the same. '
               f'The passed in arguments have length '
               f'{len(onset)}, {len(ch_name)}, '
               f'{len(duration)}.')
        raise ValueError(msg)

    if not isinstance(sfreq, list):
        sfreq = [sfreq] * len(onset)

    sample = np.multiply(onset, sfreq).astype(int)

    # set annotation labels
    if annotation_label is None:
        # all labels are just "HFO"
        label = ['HFO'] * len(onset)
    else:
        if len(annotation_label) != len(onset):
            msg = (f'Length of "annotation_label" need to '
                   f'be the same as other arguments if it '
                   f'is not None. The passed in arguments '
                   f'have length {len(onset)}, {len(ch_name)}, '
                   f'{len(duration)}, {len(annotation_label)}.')
            raise ValueError(msg)
        label = annotation_label

    # all description has the HFO events in there
    channels = [[desc] for desc in ch_name]

    # create the event dataframe according to BIDS events
    annot_df = pd.DataFrame(data=np.column_stack([onset, duration, label,
                                                  channels, sample, sfreq]),
                            index=None,
                            columns=ANNOT_COLUMNS + ['sfreq'])
    print(annot_df.head())
    annot_df = annot_df.astype({
        'onset': 'float64',
        'duration': 'float64',
        'label': 'str',
        'channels': 'object',
        'sample': 'int',
        'sfreq': 'float64',
    })
    return annot_df


def read_annotations(fname: Union[str, Path], root: Path = None) \
        -> pandas.core.frame.DataFrame:
    """Read annotations.tsv Derivative file.

    Annotations are part of the BIDS-Derivatives for Common
    Electrophysiological derivatives [1].

    Parameters
    ----------
    fname : str | pathlib.Path
        The BIDS file path for the ``*annotations.tsv|json`` files.
    root : str | pathlib.Path | None
        The root of the BIDS dataset. If None (default), will try
        to infer the BIDS root from the ``fname`` argument.

    Returns
    -------
    annot_tsv : pd.DataFrame
        The DataFrame for the annotations.tsv with extra columns appended
        to make sense of the sample data.

    References
    ----------
    .. [1] https://docs.google.com/document/d/1PmcVs7vg7Th-cGC-UrX8rAhKUHIzOI-uIOh69_mvdlw/edit#  # noqa
    """
    fname, ext = os.path.splitext(str(fname))
    fname = Path(fname)
    tsv_fname = fname.with_suffix('.tsv')
    json_fname = fname.with_suffix('.json')

    if root is None:
        fpath = fname

        while fpath != fpath.root:
            if fpath.name == 'derivatives':
                break
            fpath = fpath.parent

        # once derivatives is found, then
        # BIDS root is its parent
        root = fpath.parent

    # read the annotations.tsv file
    annot_tsv = pd.read_csv(tsv_fname, delimiter='\t')

    # read the annotations.json file
    with open(json_fname, 'r') as fin:
        annot_json = json.load(fin)

    # extract the sample freq
    raw_rel_fpath = annot_json['IntendedFor']
    raw_fpath = get_bids_path_from_fname(raw_rel_fpath)
    raw_fpath.update(root=root)
    if not raw_fpath.fpath.exists():
        raise RuntimeError(
            f'No raw dataset found for {fpath}. '
            f'Please set "root" kwarg.'
        )

    # read data
    raw = read_raw_bids(raw_fpath)
    sfreq = raw.info['sfreq']

    # create sample column
    annot_tsv['sample'] = annot_tsv['onset'] * sfreq
    return annot_tsv


def write_annotations(annot_df: pd.DataFrame, fname: Union[str, Path],
                      intended_for: str, root: Path,
                      description: str = None):
    """Write annotations dataframe to disc.

    Parameters
    ----------
    annot_df : pd.DataFrame
        The annotations DataFrame.
    fname : str | pathlib.Path
        The BIDS filename to write annotations to.
    intended_for : str | pathlib.Path | BIDSPath
        The ``IntendedFor`` BIDS keyword corresponding to the
        ``Raw`` file that the Annotations were created from.
    root : str | pathlib.Path
        The root of the BIDS dataset.
    description : str | None
        The description of the Annotations file. If None (default),
        will describe it as HFO events detected using mne-hfo.
    """
    fname, ext = os.path.splitext(str(fname))
    fname = Path(fname)
    tsv_fname = fname.with_suffix('.tsv')
    json_fname = fname.with_suffix('.json')

    if description is None:
        description = 'HFO annotated events detected using ' \
                      'mne-hfo algorithms.'

    # error check that intendeFor exists
    intended_for_path = get_bids_path_from_fname(intended_for)
    # write the correct extension for BrainVision
    if intended_for_path.extension == '.eeg':
        intended_for_path.update(extension = '.vhdr')

    if not intended_for_path.fpath.exists():
        raise RuntimeError(f'The intended for raw dataset '
                           f'does not exist at {intended_for_path}. '
                           f'Please make sure it does.')

    # make sure parent directories exist
    tsv_fname.parent.mkdir(parents=True, exist_ok=True)

    # write the dataframe itself as a tsv file
    annot_df.to_csv(tsv_fname, sep='\t', index=False)

    # create annotations json
    annot_json = {
        'Description': description,
        'IntendedFor': intended_for_path.basename,
        'Author': 'mne-hfo',
        'LabelDescription': {
            'hfo_<ch_name>': 'Generic HFO detected at channel name.',
            'ripple_<ch_name>': 'Ripple HFO detected at channel.',
            'fastripple_<ch_name>': 'Fast ripple HFO detected at channel',
            'frandr_<ch_name>': 'Fast ripple and ripple HFOs co-occurrence '
                                'at channel'
        },
    }
    with open(json_fname, 'w', encoding='utf-8') as fout:
        json.dump(annot_json, fout, ensure_ascii=False, indent=4)
