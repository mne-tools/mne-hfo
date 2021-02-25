import json
import os
import platform
from pathlib import Path
from typing import List, Dict, Union, Optional

import mne
import numpy as np
import pandas
import pandas as pd
from mne.utils import run_subprocess
from mne_bids import read_raw_bids, get_entities_from_fname, BIDSPath

from mne_hfo.config import EVENT_COLUMNS, ANNOT_COLUMNS


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


def _create_events_df(onset: List[float], duration: List[float],
                      description: List[str], sample: List[int]
                      ) -> pandas.core.frame.DataFrame:
    """Create ``events.tsv`` file.

    Parameters
    ----------
    onset :
    description :
    sample : list
        Samples are the onset (in secs) multiplied by sample rate.
    duration :

    Returns
    -------
    event_df : pd.DataFrame
    """
    if duration is None:
        duration = [0] * len(onset)
    if len(onset) != len(description) or \
            len(onset) != len(duration) or len(sample) != len(onset):
        raise RuntimeError(f'Length of "onset", "description", '
                           f'"duration", "sample" need to be the same. '
                           f'The passed in arguments have length '
                           f'{len(onset)}, {len(description)}, '
                           f'{len(duration)}.')

    # create the event dataframe according to BIDS events
    event_df = pd.DataFrame(data=np.column_stack([onset, duration,
                                                  sample, description]),
                            index=None,
                            columns=EVENT_COLUMNS)

    event_df = event_df.astype({
        'onset': 'float64',
        'duration': 'float64',
        'sample': 'int',
        'trial_type': 'str'
    })
    return event_df


def _read_annotations_json(fname: Union[str, Path]) -> Dict:
    with open(fname, 'r', encoding='utf-8') as fin:
        annot_dict = json.load(fin)

    return annot_dict


def events_to_annotations(events_df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible function to convert events to annotations.

    HFO events could be stored as ``*events.tsv`` files, but since
    they are computed on the raw data, they should be ``*annotations``
    BIDS-Derivative files. This function will take in an event DataFrame
    and convert to annotations DataFrame.

    It assumes that the event DataFrame ``description`` column has
    the structure: ``<hfo_event_name>_<ch_name>``. For example,
    it could be ``ripple_A1`` for a "ripple" detected in
    channel A1, or ``fastripple_A2-A1`` for a "fast ripple" detected
    in bipolar channel A2-A1.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe.

    Returns
    -------
    annot_df : pd.DataFrame
        Annotations dataframe structured using
        :func:`create_annotations_df`.
    """
    required_cols = ['onset', 'duration', 'sample', 'trial_type']
    if any([col not in events_df.columns for col in required_cols]):
        raise RuntimeError(f'Passed in events dataframe is missing '
                           f'one of required columns {required_cols}. '
                           f'Please double check that the events dataframe '
                           f'is BIDS-compliant.')
    if events_df.empty:
        annot_df = pd.DataFrame(columns=ANNOT_COLUMNS)
        return annot_df

    # extract data points from events dataframe
    onset = events_df['onset'].tolist()
    duration = events_df['duration'].tolist()
    description = events_df['trial_type'].tolist()

    sfreqs = events_df['sample'].divide(events_df['onset'])
    sfreq = sfreqs.values[0]

    # extract channels for each HFO event
    annotation_label = []
    ch_names = []

    # split events dataframe description
    for desc_str in description:
        annot, ch = desc_str.split('_')
        annotation_label.append(annot)
        ch_names.append(ch)

    # create the annotations dataframe
    annot_df = create_annotations_df(onset, duration, ch_names,
                                     annotation_label)
    annot_df['sample'] = annot_df['onset'] * sfreq
    return annot_df


def create_events_df(input: Union[Dict[str, List], mne.io.Raw],
                     sfreq: float = None, event_name: str = "hfo") \
        -> pd.DataFrame:
    """Create a BIDS events dataframe for HFO events.

    Parameters
    ----------
    input : dictionary(list(tuple(int, int))) | mne.io.Raw
        The input data structure that is either a mne ``Raw``
        object with ``Annotations`` set that correspond to the
        HFO events, or a dictionary of lists of HFO
        start/end sample points.
    sfreq : float | None
        The sampling frequency. Only required if the input is
        not a ``mne.io.BaseRaw`` object.
    event_name: str
        The prefix of the channel event to add to the "trial_type" column (e.g. ``hfo`` for channel ``A1``
        would result in ``hfo_A1``).

    Returns
    -------
    events_df : DataFrame
        The event dataframe according to BIDS [1].

    Notes
    -----
    Output of the dataframe will have the following column structures:
    ..

        'onset': 'float64',
        'duration': 'float64',
        'sample': 'int',
        'trial_type': 'str'

    References
    ----------
    [1] https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html  # noqa
    """
    # handle error checks and extract
    if isinstance(input, mne.io.BaseRaw):
        if input.annotations is None:
            raise ValueError('Trying to create events DataFrame using '
                             'a mne Raw object without Annotations. '
                             'Please use `raw.set_annotations` to '
                             'add the HFO events.')
        annotations = input.annotations

        onset = annotations.onset
        duration = annotations.duration
        sample = onset * input.info['sfreq']
        description = annotations.description
    elif isinstance(input, dict):
        if sfreq is None:
            raise RuntimeError('If input is a dictionary of a list of HFO '
                               'events, then the sampling frequency must '
                               'be also passed in.')
        onset, duration = [], []
        sample, description = [], []
        for ch_name, endpoints_list in input.items():
            for endpoints in endpoints_list:
                if len(endpoints) != 2:
                    raise ValueError(f'All HFO events must be stored as a '
                                     f'tuple of 2 numbers: the start and end '
                                     f'sample point of the event. For '
                                     f'{ch_name}, there is an event '
                                     f'stored as {endpoints} that does '
                                     f'not follow this.')
                onset_sec = endpoints[0] / sfreq
                offset_sec = endpoints[1] / sfreq
                onset.append(onset_sec)

                duration.append(offset_sec - onset_sec)
                sample.append(endpoints[0])
                description.append(f'{event_name}_{ch_name}')
    else:
        raise ValueError('Unaccepted data structure for input.')
    sample = [int(s) for s in sample]
    # now create the dataframe
    event_df = _create_events_df(onset=onset, duration=duration, sample=sample,
                                 description=description)
    return event_df


def create_annotations_df(onset: List[float], duration: List[float],
                          ch_name: List[str],
                          annotation_label: Optional[List[str]] = None) \
        -> pd.DataFrame:
    """Create a BIDS-derivative annotations dataframe for HFO events.

    Parameters
    ----------
    onset : float
        Onset time in seconds.
    duration : float
        Duration in seconds.
    ch_name : str
        The name of the event to add to the "trial_type" column
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
                                                  channels]),
                            index=None,
                            columns=ANNOT_COLUMNS)
    annot_df = annot_df.astype({
        'onset': 'float64',
        'duration': 'float64',
        'label': 'str',
        'channels': 'object',
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
    entities = get_entities_from_fname(raw_rel_fpath)
    raw_fpath = BIDSPath(**entities,
                         datatype='ieeg',
                         extension=Path(raw_rel_fpath).suffix,
                         root=root)
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
                      description: str = None) -> None:
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
    entities = get_entities_from_fname(intended_for)
    _, ext = os.path.splitext(intended_for)
    # write the correct extension for BrainVision
    if ext == '.eeg':
        ext = '.vhdr'
    intended_for_path = BIDSPath(**entities, extension=ext, root=root)
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
