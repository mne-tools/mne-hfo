from typing import List, Dict

import numpy as np
import pandas as pd

EVENT_COLUMNS = ['onset', 'duration', 'sample', 'trial_type']


def create_events_tsv(onset: List[float], description: List[str],
                      sfreq: float,
                      duration: List[float] = None) -> pd.DataFrame:
    """Create ``events.tsv`` file.

    Parameters
    ----------
    onset :
    description :
    sfreq :
    duration :

    Returns
    -------
    event_df : pd.DataFrame
    """
    if duration is None:
        duration = [0] * len(onset)
    if len(onset) != len(description) or \
            len(onset) != len(duration):
        raise RuntimeError(f'Length of "onset", "description", '
                           f'"duration" need to be the same. '
                           f'The passed in arguments have length '
                           f'{len(onset)}, {len(description)}, '
                           f'{len(duration)}.')

    # samples are the onset (in secs) multiplied by sample rate
    sample = np.multiply(onset, sfreq)

    # create the event dataframe according to BIDS events
    event_df = pd.DataFrame(data=[onset, duration, sample, description],
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
