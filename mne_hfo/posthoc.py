import collections
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import pandas as pd

from mne_hfo.config import TIME_SCALE_TO_SECS
from mne_hfo.io import ANNOT_COLUMNS
from mne_hfo.utils import _find_overlapping_events


def _to_freq(x, rate='s'):
    f = x.count() / x.mean()
    print('here...')
    print(x)
    print(f)
    print(x.count(), x.mean())
    return f / TIME_SCALE_TO_SECS[rate]


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
                          ch_names: Optional[List[str]] = None,
                          rate: str = 'h',
                          end_sec: float = None,
                          verbose: bool = True) -> Dict[str, float]:  # noqa
    """Compute channel HFO rates from annotations DataFrame.

    This function will assume that each row is another
    HFO event. If you want to pre-process the HFOs that
    in some way overlap, do so beforehand.

    Parameters
    ----------
    annot_df : pd.DataFrame
        The DataFrame corresponding to the ``annotations.tsv`` file.
    ch_names : list of str | None
        A list of channel names to constrain the rate computation to.
        Default = None will compute rate for all channels present in the
        ``annot_df``.
    rate : str
        The frequency at which to compute the HFO rate. Default='h', for
        every hour.
    end_sec : float | None
        The end time (in seconds) of the dataset that HFOs were computed on.
        If None (default), then will take the last detected HFO as the end
        time point.
    verbose : bool
        Verbosity.

    Returns
    -------
    ch_hfo_rates : dict
        The HFO rates per channel with any HFOs.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/66143839/computing-rate-of-occurrences-per-unit-of-time-in-a-pandas-dataframe  # noqa

    See Also
    --------
    mne_hfo.io.read_annotations
    """
    if any([col not in annot_df.columns
            for col in ANNOT_COLUMNS + ['sample']]):
        raise RuntimeError(f'Annotations dataframe columns must contain '
                           f'{ANNOT_COLUMNS + ["sample"]} in order to compute '
                           f'HFO rate.')

    # first compute sampling rate from sample / onset columns
    sfreq = annot_df['sample'] / annot_df['onset']
    if sfreq.nunique() != 1:
        raise ValueError(f'All rows in the annotations dataframe '
                         f'should have the same sampling rate. '
                         f'Found {sfreq.nunique()} different '
                         f'sampling rates.')

    # store channel rates over sliding window
    ch_hfo_rates = collections.defaultdict(list)

    # start timestamp with current time
    ref_timestamp = datetime.now(tz=timezone.utc)
    onset_tdelta = pd.to_timedelta(annot_df['onset'], unit='s')  # type: ignore
    annot_df['timestamp'] = ref_timestamp + onset_tdelta

    # get the end point in seconds
    if end_sec is None:
        end_timestamp = annot_df['timestamp'].max()
    else:
        end_timestamp = ref_timestamp + timedelta(seconds=end_sec)

    # get end time in seconds
    annot_df['end_time'] = \
        (end_timestamp - ref_timestamp).total_seconds()  # type: ignore

    if verbose:
        print(f'Beginning timestamp: {ref_timestamp}')
        print(f'Got end timestamp of: {end_timestamp}')

    # set timestamp as the datetime index to allow resampling
    annot_df.set_index('timestamp', inplace=True)  # type: ignore

    # get all unique channels
    if ch_names is None:
        ch_names = annot_df['channels'].unique()  # type: ignore
    else:
        # search for channel names not inside pandas dataframe
        if not all([name in annot_df['channels'] for name in ch_names]):
            raise ValueError('Not all channels are inside the '
                             'annotation DataFrame.')

    for idx, group in annot_df.groupby(['channels']):
        # get channel name
        ch_name = group['channels'].values[0]

        if ch_name not in ch_names:  # type: ignore
            continue

        # resample datetime indices over a certain frequency
        # so we can now count the number of HFO occurrences in a
        # set time frame
        # dt_idx = pd.date_range(ref_timestamp, end_timestamp, freq=rate)
        # group = group.reindex(dt_idx, fill_value=np.nan)

        # see Reference [1] where we compute rate of occurrence
        result = group.end_time.agg(lambda x: _to_freq(x, rate=rate))
        if verbose:
            print(f'Found HFO rate per {rate} for {ch_name} as {result}')

        # now compute the rate in this group
        ch_hfo_rates[ch_name] = result

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
