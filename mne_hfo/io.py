from pathlib import Path
from typing import List, Dict, Union

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

EVENT_COLUMNS = ['onset', 'duration', 'sample', 'trial_type']


def read_events_tsv(bids_path: Union[Path, BIDSPath]) -> pd.DataFrame:
    """Read an *events.tsv file to a DataFrame.

    Parameters
    ----------
    bids_path : BIDSPath | pathlib.Path

    Returns
    -------
    events_df : pd.DataFrame
    """
    pass


def create_events_df(input: Union[Dict[str, List], mne.io.BaseRaw],
                     sfreq: float = None) -> pd.DataFrame:
    """Create a BIDS events dataframe for HFO events.

    Parameters
    ----------
    input : dictionary(list(tuple(int, int))) | mne.io.BaseRaw
        The input data structure that is either a mne ``Raw``
        object with ``Annotations`` set that correspond to the
        HFO events, or a dictionary of lists of HFO
        start/end points.
    sfreq : float | None
        The sampling frequency. Only required if the input is
        not a mne.io.BaseRaw object.

    Returns
    -------
    events_df : pd.DataFrame
        The event dataframe according to BIDS [1].

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
                description.append(f'hfo_{ch_name}')
    else:
        raise ValueError('Unaccepted data structure for input.')

    # now create the dataframe
    event_df = _create_events_df(onset=onset, duration=duration, sample=sample,
                                 description=description)
    return event_df


def _create_events_df(onset: List[float], duration: List[float],
                      description: List[str], sample: List[int],
                      ) -> pd.DataFrame:
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
    event_df.astype({
        'onset': 'float64',
        'duration': 'float64',
        'sample': 'int',
        'trial_type': 'str'
    })
    return event_df


def _compute_hfo_rate(events_df, ch_names):
    pass


def compute_ch_rates(events_df: pd.DataFrame, ch_names: List[str],
                     picks: List[str] = None) -> Dict[str, float]:  # noqa
    if any([col not in events_df.columns for col in EVENT_COLUMNS]):
        raise RuntimeError(f'Event dataframe columns must contain '
                           f'{EVENT_COLUMNS} in order to compute '
                           f'HFO rate.')

    # merge overlapping HFOs
    # events_df = _merge_overlapping_hfos(events_df)

    # compute the rate of each channel found
    ch_hfo_rate = _compute_hfo_rate(events_df, ch_names)

    return ch_hfo_rate
