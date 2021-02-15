import collections
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from mne_hfo.config import TIME_SCALE_TO_SECS
from mne_hfo.utils import _append_offset_to_df, _check_df


def _to_freq(x, rate: str = 's'):
    """Convert a groupby DataFrame to rate.

    Parameters
    ----------
    x : pd.Series
        The series of the group to compute frequency of occurrence.
    rate : str
        One of ``s`` (second), ``m`` (minute), ``h`` (hour),
        ``d`` (day) to compute rate of the dataframe.

    Returns
    -------
    rate : float
        The rate of the events per unit of time, selected
        by ``rate`` input.
    """
    f = x.count() / x.mean()
    return f / TIME_SCALE_TO_SECS[rate]


def compute_chs_hfo_rates(annot_df: pd.DataFrame,
                          rate: str,
                          ch_names: Optional[List[str]] = None,
                          end_sec: float = None,
                          verbose: bool = True):
    """Compute channel HFO rates from annotations DataFrame.

    This function will assume that each row is another
    HFO event. If you want to pre-process the HFOs that
    in some way overlap, do so beforehand.

    Parameters
    ----------
    annot_df : pandas.core.DataFrame
        The DataFrame corresponding to the ``annotations.tsv`` file.
    rate : str
        The frequency at which to compute the HFO rate.
        One of ``s`` (second), ``m`` (minute), ``h`` (hour),
        ``d`` (day) to compute rate of the dataframe.
    ch_names : list of str | None
        A list of channel names to constrain the rate computation to.
        Default = None will compute rate for all channels present in the
        ``annot_df``.
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
    mne_hfo.io.read_annotations : Reading in annotations.tsv file as DataFrame.
    """
    annot_df = _check_df(annot_df, df_type='annotations')

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


def _join_times(df: pd.DataFrame) -> pd.DataFrame:
    """Join together start and end times sorted in order.

    Creates a second column ``what`` that marks +1/-1 for start/end times
    to keep track of how many intervals are overlapping. Then a ``newwin``
    column is added to identify the beginning of a new non-overlapping time interval
    and a ``group`` column is added to mark the rows that belong to the same
    overlapping time interval.

    This ``group`` column is added to the original dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    res : pd.DataFrame

    References
    ----------
    .. [1] https://stackoverflow.com/questions/57804145/combining-rows-with-overlapping-time-periods-in-a-pandas-dataframe  # noqa
    """
    startdf = pd.DataFrame({  # type: ignore
        'time': df['start_timestamp'],  # type: ignore
        'what': 1})  # type: ignore
    enddf = pd.DataFrame({  # type: ignore
        'time': df['end_timestamp'],  # type: ignore
        'what': -1})  # type: ignore

    # create merged dataframe of start and end times that are
    # sorted by timestamp
    mergdf = pd.concat([startdf, enddf]).sort_values('time')

    # get a running cumulative sum
    mergdf['running'] = mergdf['what'].cumsum()  # type: ignore

    # assign groups to overlapping intervals
    mergdf['newwin'] = (mergdf['running'].eq(1) &  # type: ignore
                        mergdf['what'].eq(1))  # type: ignore
    mergdf['group'] = mergdf['newwin'].cumsum()  # type: ignore

    # add the group assignments to the original dataframe
    df['group'] = mergdf['group'].loc[mergdf['what'].eq(1)]  # type: ignore

    # now group all overlapping intervals in the original dataframe
    # agg_func_dict = {col: lambda x: set(x) for col in df.columns}
    res = df.groupby('group').agg({'start_timestamp': 'first',
                                   'end_timestamp': 'last',
                                   'label': 'unique',
                                   'ref_timestamp': 'first'})

    return res


def merge_overlapping_events(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping events detected.

    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe generated from HFO events detected.

    Returns
    -------
    merged_df : pd.DataFrame
        New events dataframe with merged HFOs depending on
        overlap criterion.

    See Also
    --------
    mne_hfo.io.create_events_df : Create events DataFrame from HFO detections.
    """
    orig_cols = df.columns

    # check dataframe
    df = _check_df(df, df_type='annotations')

    # compute sfreq
    sfreq = np.unique(df['sample'] / df['onset'])

    # start/end timestamp with current time for every row
    ref_timestamp = datetime.now(tz=timezone.utc)
    onset_tdelta = pd.to_timedelta(df['onset'], unit='s')  # type: ignore
    df['start_timestamp'] = ref_timestamp + onset_tdelta

    duration_secs = pd.to_timedelta(df['duration'], unit='s')  # type: ignore
    df['end_timestamp'] = df['start_timestamp'] + duration_secs
    df['ref_timestamp'] = ref_timestamp

    # first group by channels
    # now join rows that are overlapping
    merged_df = df.groupby(['channels']).apply(  # type: ignore
        _join_times
    ).reset_index().drop('group', axis=1)

    # get the old columns back and drop the intermediate computation columns
    merged_df['duration'] = (merged_df['end_timestamp'] -
                             merged_df['start_timestamp']).dt.total_seconds()
    merged_df['onset'] = (merged_df['start_timestamp'] -
                          merged_df['ref_timestamp']).dt.total_seconds()
    merged_df['sample'] = merged_df['onset'] * sfreq
    merged_df.drop(['start_timestamp', 'end_timestamp', 'ref_timestamp'],
                   axis=1, inplace=True)
    merged_df = merged_df[orig_cols]

    return merged_df


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
    coincident_hfo_dict : dict
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


def check_detection_overlap(y_true: List[float], y_predict: List[float]):
    """
    Evaluate if two detections overlap.

    Parameters
    ----------
    y_true: list
        Gold standard detection [start,stop]
    y_predict: list
        Detector detection [start,stop]

    Returns
    -------
    overlap: bool
        Whether two events overlap.
    """
    overlap = False

    # dd stop in gs + (dd inside gs)
    if (y_predict[1] >= y_true[0]) and (y_predict[1] <= y_true[1]):
        overlap = True
    # dd start in gs + (dd inside gs)
    if (y_predict[0] >= y_true[0]) and (y_predict[0] <= y_true[1]):
        overlap = True
    # gs inside dd
    if (y_predict[0] <= y_true[0]) and (y_predict[1] >= y_true[1]):
        overlap = True

    return overlap


def _find_overlapping_events(list1, list2):
    """
    Get subset of list1 that overlaps with list2.

    Parameters
    ----------
    list1 : list
        list of tuples (start_time, end_time)
    list2 : list
        list of tuples (start_time, end_time)

    Returns
    -------
    overlapping_events : list
        list of tuples (start_time, end_time) that overlap between
        list1 and list2.
    """
    # Sort events by start times to speed up calculation
    list1 = sorted(list1, key=lambda x: x[0])
    list2 = sorted(list2, key=lambda x: x[0])
    overlapping_events = []
    for event_time1 in list1:
        for event_time2 in list2:
            if event_time2[0] > event_time1[1]:
                break
            if check_detection_overlap(event_time1, event_time2):
                overlapping_events.append(event_time1)
    return overlapping_events


def match_detections(ytrue_df, ypredict_df, bn, freq_name=None,
                     sec_unit=None, sec_margin=1):  # noqa
    """
    Match gold standard detections with detector detections.

    Parameters
    ----------
    ytrue_df: pandas.DataFrame
        Gold standard detections
    ypredict_df: pandas.DataFrame
        Detector detections
    bn: list
        Names of event start stop [start_name, stop_name], e.g
        ['onset', 'offset'].
    freq_name: str
        Name of frequency column
    sec_unit: int
        Number representing one second of signal - this can
        significantly improve the speed of this function
    sec_margin: int
        Margin for creating subsets of compared data - should be set according
        to the length of compared events (1s for HFO should be enough)

    Returns
    -------
    match_df: pandas.DataFrame
        Dataframe with matched indices (pandas DataFrame)
    """
    # Ensure the desired columns are numeric
    ytrue_df[bn] = ytrue_df[bn].apply(pd.to_numeric())
    ypredict_df[bn] = ypredict_df[bn].apply(pd.to_numeric())

    # If df has duration instead of end time, add a new column
    if bn[1].lower() == "duration":
        ytrue_df = _append_offset_to_df(ytrue_df, bn)
        ypredict_df = _append_offset_to_df(ypredict_df, bn)
        bn[1] = "offset"
    match_df = pd.DataFrame(columns=('gs_index', 'dd_index'))
    match_df_idx = 0
    for row_gs in ytrue_df.iterrows():
        matched_idcs = []
        gs = [row_gs[1][bn[0]], row_gs[1][bn[1]]]
        if sec_unit:  # We can create subset - significant speed improvement
            # Only look at detector detection rows that have onsets
            # within the window of time provided
            for row_dd in ypredict_df[(ypredict_df[bn[0]] < gs[0] +
                                       sec_unit * sec_margin) &
                                      (ypredict_df[bn[0]] > gs[0] -
                                       sec_unit * sec_margin)].iterrows():
                dd = [row_dd[1][bn[0]], row_dd[1][bn[1]]]
                # Check if the events overlap
                if check_detection_overlap(gs, dd):
                    matched_idcs.append(row_dd[0])
        else:
            # Look at all detector detection rows
            for row_dd in ypredict_df.iterrows():
                dd = [row_dd[1][bn[0]], row_dd[1][bn[1]]]
                # Check if the events overlap
                if check_detection_overlap(gs, dd):
                    matched_idcs.append(row_dd[0])

        # No overlap found for this gold standard row
        if len(matched_idcs) == 0:
            match_df.loc[match_df_idx] = [row_gs[0], None]
        # One overlap found for this gold standard row
        elif len(matched_idcs) == 1:
            match_df.loc[match_df_idx] = [row_gs[0], matched_idcs[0]]
        else:
            # In rare event of multiple overlaps get the closest frequency
            if freq_name:
                dd_idx = (
                    abs(ypredict_df.loc[matched_idcs, freq_name] -
                        row_gs[1][freq_name])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]
            # Closest event start - less precision than frequency
            else:
                dd_idx = (
                    abs(ypredict_df.loc[matched_idcs, bn[0]] -
                        row_gs[1][bn[0]])).idxmin()
                match_df.loc[match_df_idx] = [row_gs[0], dd_idx]

        match_df_idx += 1

    return match_df
