from typing import List, Dict

import pandas as pd

from mne_hfo.io import EVENT_COLUMNS


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
