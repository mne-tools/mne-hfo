import collections
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from mne_hfo.config import TIME_SCALE_TO_SECS
from mne_hfo.utils import _check_df


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


def merge_overlapping_events(df: pd.DataFrame):
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


def _check_detection_overlap(y_true: List[float], y_predict: List[float]):
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
            if _check_detection_overlap(event_time1, event_time2):
                overlapping_events.append(event_time1)
    return overlapping_events


def match_detected_annotations(
        ytrue_annot_df: pd.DataFrame, ypred_annot_df: pd.DataFrame,
        ch_names: Union[List[str], str] = None,
        label: str = None, sec_margin: float = 1., method='match-true'):
    """Given two annotations.tsv DataFrames, match HFO detection overlaps.

    Parameters
    ----------
    ytrue_annot_df : pd.DataFrame
        The reference annotations DataFrame containing the HFO events that are
        considered "ground-truth" in this comparison.
    ypred_annot_df : pd.DataFrame
        The estimated annotations DataFrame containing the HFO events that
        are estimated using a ``Detector``.
    ch_names : list | str | None
        Which channels to match. If None (default), then will match all
        available channels in both dataframes. If str, then must be a single
        channel name available in the ``ytrue_annot_df``. If list of
        strings, then must be a list of channel names available in the
        ``ytrue_annot_df``.
    label : str | None
        The HFO label to use. If None (default) will consider all rows in
        both input DataFrames as an HFO event. If a string, then it must
        match to an element of ``label`` column in the dataframes.
    sec_margin : float
        Number of seconds to consider a valid checking window.
        Default = 1.
    method : str
        Type of strategy for matching HFO events. Must be one of
        ``match-true``, ``match-pred``, or ``match-total``.
        If "match-true", will return a dataframe of all true indices
        and matching predicted indices if they exist. If "match-pred",
        will return a dataframe of all predicted indices and matching
        true indices if they exist. If "match-total", will return the
        concatenation of the two. See Notes for more information.

    Returns
    -------
    matched_df : pd.DataFrame
        A DataFrame with the columns ``pred_index`` and ``true_index``,
        which corresponds to indices,
    """
    # check adherence of the annotations dataframe structure
    ytrue_annot_df = _check_df(ytrue_annot_df, df_type='annotations')
    ypred_annot_df = _check_df(ypred_annot_df, df_type='annotations')

    # select only certain labels
    if label is not None:
        if label not in ytrue_annot_df['label'] or \
                label not in ypred_annot_df['label']:
            raise ValueError(f'Label {label} is not inside the input '
                             f'DataFrames.')

        ytrue_annot_df = ytrue_annot_df.loc[ytrue_annot_df['label'] == label]
        ypred_annot_df = ypred_annot_df.loc[ypred_annot_df['label'] == label]

    # select only certain channels
    if ch_names is not None:
        if isinstance(ch_names, str):
            ch_names = [ch_names]
        if any([ch not in ytrue_annot_df['channels'] for ch in ch_names]):
            raise ValueError(f'Channels {ch_names} are not all inside '
                             f'ground-truth HFO DataFrame.')
        if any([ch not in ypred_annot_df['channels'] for ch in ch_names]):
            raise ValueError(f'Channels {ch_names} are not all inside '
                             f'predicted HFO DataFrame.')

        ytrue_annot_df = ytrue_annot_df.loc[
            ytrue_annot_df['channels'].isin(ch_names)
        ]
        ypred_annot_df = ypred_annot_df.loc[
            ypred_annot_df['channels'].isin(ch_names)
        ]

    # if prediction yields no events and method is match-pred,
    # return empty structured dataframe
    if ypred_annot_df.empty and method == "match-pred":
        return pd.DataFrame(columns=('true_index', 'pred_index'))
    # else if prediction yields no events, return structured dataframe
    # containing just true indices
    elif ypred_annot_df.empty:
        match_df = pd.DataFrame(columns=('true_index', 'pred_index'))
        for ind, row in ytrue_annot_df.iterrows():
            match_df.loc[ind] = [ind, None]
        match_df.apply(pd.to_numeric, errors="coerce",
                       downcast="float")
        return match_df

    # make sure columns match what is needed
    ytrue_annot_df['offset'] = \
        ytrue_annot_df['onset'] + ytrue_annot_df['duration']
    ypred_annot_df['offset'] = \
        ypred_annot_df['onset'] + ypred_annot_df['duration']

    if method.lower() == "match-true":
        return _match_detections_overlap(
            ytrue_annot_df, ypred_annot_df, sec_margin,
            ('true_index', 'pred_index'))
    elif method.lower() == "match-pred":
        return _match_detections_overlap(
            ypred_annot_df, ytrue_annot_df, sec_margin,
            ('pred_index', 'true_index'))
    elif method.lower() == "match-total":
        true_match = _match_detections_overlap(
            ytrue_annot_df, ypred_annot_df, sec_margin,
            ('true_index', 'pred_index'))
        pred_match = _match_detections_overlap(
            ypred_annot_df, ytrue_annot_df, sec_margin,
            ('pred_index', 'true_index'))
        return pd.concat([true_match, pred_match]).drop_duplicates(). \
            reset_index(drop=True)
    else:
        raise NotImplementedError("Method must be one of match-true,"
                                  " match-pred, or match-total")
        # Iterate over true labels (gold standard)


def _match_detections_overlap(gs_df, check_df, margin, cols):
    """
    Find the overlapping detections in the two passed dataframes.

    gs_df and check_df need to be the same type (i.e. both annotation
    dataframes or event dataframes). If they are annotation dataframes,
    margin should be in seconds, and if they are event dataframes,
    margin should be in samples.

    Parameters
    ----------
    gs_df : pd.DataFrame
        The reference DataFrame containing the HFO events that are
        considered "ground-truth" in this comparison.
    check_df : pd.DataFrame
        The estimated DataFrame containing the HFO events that
        are estimated using a ``Detector``.
    margin : int
        Margin to check. Should be in the same unit as the data
        in the desired columns
    cols : list[str]
        Name of the columns corresponding to gs indices and check indices

    Returns
    -------
    match_df: pd.DataFrame
        A DataFrame with the columns from cols input,
        which corresponds to indices

    """
    if not all([col in gs_df for col in ['onset', 'offset']]):
        raise ValueError(f'Gold standard reference Annotations '
                         f'DataFrame must have both "onset" and '
                         f'"offset" columns (in seconds). It '
                         f'has columns: {gs_df.columns}')
    if not all([col in check_df for col in ['onset', 'offset']]):
        raise ValueError(f'Estimated Annotations '
                         f'DataFrame must have both "onset" and '
                         f'"offset" columns (in seconds).It '
                         f'has columns: {check_df.columns}')

    # List of tuples to populate the output DataFrame
    match_indices = []

    # Convert the DataFrames that are expensive to manipulate into a list
    # of tuples (index, onset, offset, ch_name)
    # Pandas does not care about column order, but since we are changing
    # the DataFrames to numpy, we need to track the column order
    gs_cols = gs_df.columns
    check_cols = check_df.columns
    gs_keep_inds = (gs_cols.get_loc("onset"),
                    gs_cols.get_loc("offset"),
                    gs_cols.get_loc("channels"))
    check_keep_inds = (check_cols.get_loc("onset"),
                       check_cols.get_loc("offset"),
                       check_cols.get_loc("channels"))
    gs_numpy = gs_df.to_numpy()[:, gs_keep_inds]
    gs_numpy = [[i, onset, offset, ch_name] for
                i, (onset, offset, ch_name)
                in enumerate(gs_numpy)]
    check_numpy = check_df.to_numpy()[:, check_keep_inds]
    check_numpy = [[i, onset, offset, ch_name] for
                   i, (onset, offset, ch_name)
                   in enumerate(check_numpy)]

    # TODO: If there is a way to subset by channel, we can speed
    #  up the loop

    # Now we can iterate
    for gs_hfo in gs_numpy:
        gs_ind, gs_onset, gs_offset, gs_ch_name = gs_hfo
        check_window = (gs_onset - margin, gs_onset + margin)
        # Subset to the same channel and has onset within the expected window
        check_numpy_channel = [x for x in check_numpy
                               if (x[3] == gs_ch_name and
                                   (x[1] > check_window[0] or
                                    x[1] < check_window[1]))]
        # check if nothing meets this criteria
        if not check_numpy_channel:
            match_indices.append((gs_ind, None))
            continue
        potential_matches = []
        # else, see if there is overlap
        for check_hfo in check_numpy_channel:
            check_ind, check_onset, check_offset, check_ch_name = check_hfo
            gs_win = (gs_onset, gs_offset)
            check_win = (check_onset, check_offset)
            if _check_detection_overlap(gs_win, check_win):
                potential_matches.append(check_hfo)
        if not potential_matches:
            match_indices.append((gs_ind, None))
        elif len(potential_matches) == 1:
            match_indices.append((gs_ind, potential_matches[0][0]))
        else:
            # more than one match, find closest
            match_indices.append(_find_best_overlap(gs_hfo, potential_matches))

    if not match_indices:
        match_df = pd.DataFrame(columns=cols)
    else:
        match_df = pd.DataFrame(match_indices, columns=cols).apply(
            pd.to_numeric, errors="coerce", downcast="float")
    return match_df


def _find_best_overlap(gs, check_list):
    """Find best overlap from an ideal (gs) and a possible list."""
    gs_ind, gs_onset, gs_offset, _ = gs
    gs_point = np.array([gs_onset, gs_offset])
    dist = np.inf
    best_inds = (gs_ind, None)
    for check_hfo in check_list:
        check_ind, check_onset, check_offset, _ = check_hfo
        check_point = np.array([check_onset, check_offset])
        # Using distance of the points as the metric
        new_dist = np.linalg.norm(gs_point - check_point)
        if new_dist < dist:
            dist = new_dist
            best_inds = (gs_ind, check_ind)
    return best_inds
