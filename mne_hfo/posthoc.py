import collections
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd

from mne_hfo.io import ANNOT_COLUMNS
from mne_hfo.utils import _find_overlapping_events


def _compute_hfo_rate(df, rate_rule, origin):
    resampled_df = df.resample(rate_rule, origin=origin).apply(
        {'onset': 'count'}
    )
    print(f'REsampled dataframe: {rate_rule}, {origin}')
    print(resampled_df)
    print('heres the mean')
    print(resampled_df.mean().values)
    return resampled_df.mean().values[0]


def compute_chs_hfo_rates(annot_df: pd.DataFrame,
                          ch_names: List[str] = None,
                          rate: str = 'h', over_time: bool = False,
                          end_sec: float = None, verbose: bool = True) -> Dict[str, float]:  # noqa
    """Compute channel HFO rates from annotations DataFrame.

    This function will assume that each row is another
    HFO event. If you want to pre-process the HFOs that
    in some way overlap, do so beforehand.

    Parameters
    ----------
    annot_df : pd.DataFrame
        The DataFrame corresponding to the ``annotations.tsv`` file.
    ch_names : list of str | None
    rate : str
        The frequency at which to compute the HFO rate. Default='h', for
        every hour.
    over_time :

    Returns
    -------

    See Also
    --------
    mne_hfo.io.read_annotations
    """
    if any([col not in annot_df.columns for col in ANNOT_COLUMNS + ['sample']]):
        raise RuntimeError(f'Annotations dataframe columns must contain '
                           f'{ANNOT_COLUMNS + ["sample"]} in order to compute '
                           f'HFO rate.')

    # ensure certain columns are numeric

    # first compute sampling rate from sample / onset columns
    sfreq = annot_df['sample'] / annot_df['onset']
    if sfreq.nunique() != 1:
        raise ValueError(f'All rows in the annotations dataframe '
                         f'should have the same sampling rate. '
                         f'Found {sfreq.nunique()} different '
                         f'sampling rates.')
    sfreq = sfreq.values[0]

    # store channel rates over sliding window
    ch_hfo_rates = collections.defaultdict(list)

    # start timestamp with current time
    ref_timestamp = datetime.now(tz=timezone.utc)
    print(annot_df)
    annot_df['timestamp'] = \
        ref_timestamp + pd.to_timedelta(annot_df['onset'], unit='s')
    if end_sec is None:
        end_timestamp = annot_df['timestamp'].max()
    else:
        end_timestamp = ref_timestamp + timedelta(seconds=end_sec - 1)

    if verbose:
        print(f'Beginning timestamp: {ref_timestamp}')
        print(f'Got end timestamp of: {end_timestamp}')

    # set timestamp as the datetime index to allow resampling
    annot_df.set_index('timestamp', inplace=True)

    # get all unique channels
    if ch_names is None:
        ch_names = annot_df['channels'].unique()
        print('Found ... ', ch_names)
    else:
        # search for channel names not inside pandas dataframe
        print('not yet...')

    for idx, group in annot_df.groupby(['channels']):
        # get channel name
        ch_name = group['channels'].values[0]

        print(ch_name)
        print(group)

        # resample datetime indices over a certain frequency
        # so we can now count the number of HFO occurrences in a
        # set time frame
        dt_idx = pd.date_range(ref_timestamp, end_timestamp, freq=rate)
        group = group.reindex(dt_idx, fill_value=np.nan)

        print('Resampled index...')
        print(group.shape)
        print(group)

        # now compute the rate in this group
        ch_hfo_rates[ch_name] = _compute_hfo_rate(group,
                                                  rate_rule=rate,
                                                  origin=ref_timestamp)

        print('inside here...')
        print(ch_hfo_rates[ch_name])
        if ch_name == 'A1':
            raise Exception('hi')
        # print(group.groupby(['timestamp']).size())#.unstack(fill_value=0))

        # if not over_time:
        # ch_hfo_rates[ch_name] = ch_hfo_rates[ch_name].count()

    return ch_hfo_rates


def find_coincident_events(hfo_dict1, hfo_dict2):
    """
    Get a dictionary of hfo events that overlap between two sets.

    Note: Both input dictionaries should come from the same original
    dataset and therefore contain the same keys.

    Parameters
    ----------
    hfo_dict1 : dict
        keys are channel names and values are list of tuples of start
        and end times.
    hfo_dict2 : dict
        keys are channel names and values are list of tuples of start
        and end times.

    Returns
    -------
    coincident_hfo_dict : Dict
        Subset of hfo_dict1 containing just the entries that overlap
        with hfo_dict2.
    """
    if set(hfo_dict1.keys()) != set(hfo_dict2.keys()):
        raise RuntimeError("The two dictionaries must have the same keys.")
    coincident_hfo_dict = {}
    for ch_name, hfo_list1 in hfo_dict1.items():
        hfo_list2 = hfo_dict2.get(ch_name)
        coincident_hfo_list = _find_overlapping_events(hfo_list1, hfo_list2)
        coincident_hfo_dict.update({ch_name: coincident_hfo_list})
    return coincident_hfo_dict
