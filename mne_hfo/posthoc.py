import collections
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Union

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

    # desired columns (dc) for the annotations dataframe are
    # onset and duration
    dc = ['onset', 'duration']

    if method.lower() == "match-true":
        return _match_detections_overlap(ytrue_annot_df, ypred_annot_df, dc,
                                         sec_margin,
                                         ('true_index', 'pred_index'))
    elif method.lower() == "match-pred":
        return _match_detections_overlap(ypred_annot_df, ytrue_annot_df, dc,
                                         sec_margin,
                                         ('pred_index', 'true_index'))
    elif method.lower() == "match-total":
        true_match = _match_detections_overlap(ytrue_annot_df, ypred_annot_df,
                                               dc, sec_margin,
                                               ('true_index', 'pred_index'))
        pred_match = _match_detections_overlap(ypred_annot_df, ytrue_annot_df,
                                               dc, sec_margin,
                                               ('pred_index', 'true_index'))
        return pd.concat([true_match, pred_match]).drop_duplicates(). \
            reset_index(drop=True)
    else:
        raise NotImplementedError("Method must be one of match-true,"
                                  " match-pred, or match-total")
        # Iterate over true labels (gold standard)

    # TODO: determine matching HFO events depending on method
    # note if it's simple, you could transform these dataframes into the ones
    # accepted by `match_detections` down below.
    # or you could just refactor the functionality of `match_detections`.
    # TODO:


def match_detections(ytrue_df, ypredict_df, label: str = None,
                     sec_margin: float = 1., method="match-true"):
    """
    Match overlapping detections from ground truth and predicted datasets.

    Parameters
    ----------
    ytrue_df : pd.DataFrame
        Event Dataframe of true labels
    ypredict_df : pd.DataFrame
        Event Dataframe of predicted labels
    label : str
        Specific label to search for. i.e. only match on fast_ripple events.
    sec_margin : float
        Number of seconds to consider a valid checking window
    method : str
        One of ["match-true", "match-pred", or "match-total"]. If "match-true",
        will return a dataframe of all true indices and matching predicted
        indices if they exist. If "match-pred", will return a dataframe of
        all predicted indices and matching true indices if they exist. If
        "match-total", will return the concatenation of the two

    Returns
    -------
    pd.DataFrame
        dataframe with columns "true_index" and "pred_index"


    Examples
    --------
    >>> # Assume annot_df1 is ground truth and annot_df2 is prediction
    >>> onset1 = [0, 7300, 12600, 22342, 59900]
    >>> offset1 = [67300, 8500, 14870, 31100, 81200]
    >>> event_list = [(onset, offset) for onset, offset in\
    >>> zip(onset1, offset1)]
    >>> event_dict1 = {'A1':event_list}
    >>> sfreq = 1000
    >>> event_df1 = create_events_df(event_dict1, sfreq=sfreq,
    >>> event_name="hfo")
    >>> onset2 = [2000, 12300, 45800, 98300]
    >>> offset2 = [6930, 15120, 65600, 101450]
    >>> event_list = [(onset, offset) for onset, offset in\
    >>> zip(onset2, offset2)]
    >>> event_dict2 = {'A1':event_list}
    >>> event_df2 = create_events_df(event_dict2, event_name='hfo',
    >>> sfreq=sfreq)
    >>> match_true_df = match_detections(event_df1, event_df2,
    >>> method="match-true")
    >>> # match_true_df is a dataFrame with the following data:
    >>> # {"true_index" : [0 1 2 3 4 5], "pred_index": [0 None 1 None 2] }
    >>> match_pred_df = match_detections(event_df1, event_df2,
    >>> method="match-pred")
    >>> # match_pred_df is a dataFrame with the following data:
    >>> # {"true_index" : [0 2 4 None], "pred_index": [0 1 2 3] }
    >>> match_total_df = match_detections(event_df1, event_df2,
    >>> method="match-total")
    >>> # match_total_df is a dataFrame with the following data:
    >>> # {"true_index" : [0 1 2 3 4 None], "pred_index": [0 None 1 None 2 3] }

    """
    # if prediction yields no events, return dataframe with just true indices
    if ypredict_df.empty and method == "match-pred":
        return pd.DataFrame(columns=('true_index', 'pred_index'))
    elif ypredict_df.empty:
        match_df = pd.DataFrame(columns=('true_index', 'pred_index'))
        for ind, row in ytrue_df.iterrows():
            match_df.loc[ind] = [ind, None]
        match_df.apply(pd.to_numeric, errors="coerce",
                       downcast="float")
        return match_df
    # Check passed in event dataframes
    ytrue_df = _check_df(ytrue_df, df_type="event")
    ypredict_df = _check_df(ypredict_df, df_type="event")
    # Check that passed dataframes have same sfreq
    sfreq_true = ytrue_df['sample'].divide(ytrue_df['onset']).round(2)
    sfreq_pred = ypredict_df['sample'].divide(ypredict_df['onset']).round(2)
    sfreq = pd.concat([sfreq_true, sfreq_pred], ignore_index=True)
    # onset=0 will cause sfreq to be inf, drop these rows to prevent
    # additional sfreqs
    sfreq = sfreq.replace([np.inf, -np.inf], np.nan).dropna()
    if sfreq.nunique() != 1:
        raise ValueError(f'Passed dataframes must have the same sfreq.'
                         f'There are {sfreq.nunique()} frequencies.')
    frq = sfreq.iloc[0]
    samp_margin = frq * sec_margin

    # Ensure the desired columns are numeric
    dc = ["onset", "duration"]
    ytrue_df[dc] = ytrue_df[dc].apply(pd.to_numeric)
    ypredict_df[dc] = ypredict_df[dc].apply(pd.to_numeric)

    # Subset the dataframes to certain event types if passed
    # Subsets with partial matches accepted, so passing label=hfo
    # subsets all channels
    if label:
        ytrue_df = ytrue_df[ytrue_df['trial_type'].str.contains(label)]
        ypredict_df = ypredict_df[
            ypredict_df['trial_type'].str.contains(label)
        ]

    if method.lower() == "match-true":
        return _match_detections_overlap(ytrue_df, ypredict_df, dc,
                                         samp_margin,
                                         ('true_index', 'pred_index'))
    elif method.lower() == "match-pred":
        return _match_detections_overlap(ypredict_df, ytrue_df, dc,
                                         samp_margin,
                                         ('pred_index', 'true_index'))
    elif method.lower() == "match-total":
        true_match = _match_detections_overlap(ytrue_df, ypredict_df,
                                               dc, samp_margin,
                                               ('true_index', 'pred_index'))
        pred_match = _match_detections_overlap(ypredict_df, ytrue_df,
                                               dc, samp_margin,
                                               ('pred_index', 'true_index'))
        return pd.concat([true_match, pred_match]).drop_duplicates(). \
            reset_index(drop=True)
    else:
        raise NotImplementedError("Method must be one of match-true,"
                                  " match-pred, or match-total")
        # Iterate over true labels (gold standard)


def _match_detections_overlap(gs_df, check_df, dc, margin, cols):
    """
    Find the overlapping detections in the two passed dataframes.

    gs_df and check_df need to be the same type (i.e. both annotation
    dataframes or event dataframes). If they are annotation dataframes,
    margin should be in seconds, and if they are event dataframes,
    margin should be in samples. dc can be either ['onset', duration']
    or ['onset', 'offset'].

    Parameters
    ----------
    gs_df : pd.DataFrame
        The reference DataFrame containing the HFO events that are
        considered "ground-truth" in this comparison.
    check_df : pd.DataFrame
        The estimated DataFrame containing the HFO events that
        are estimated using a ``Detector``.
    dc : list[str]
        List of columns that define an event. Should be either
        ['onset', 'duration'] or ['onset', 'offset'].
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
    # Check column types
    if dc[1] not in ['duration', 'offset']:
        raise ValueError(f'dc must have either an offset or '
                         f'duration column. Passed {dc[1]}.')
    if dc[1] == "duration":
        gs_df = _append_offset_to_df(gs_df, dc)
        check_df = _append_offset_to_df(check_df, dc)
        dc[1] = "offset"
    # We want to create another dataframe of matched gold
    # standard indices and checked indices
    match_df = pd.DataFrame(columns=cols)
    match_df_idx = 0
    for row_gs in gs_df.iterrows():
        matched_idcs = []
        # [onset, offset]
        gs = [row_gs[1][dc[0]], row_gs[1][dc[1]]]
        for row_pred in check_df[(check_df[dc[0]] < gs[0] +
                                  margin) &
                                 (check_df[dc[0]] > gs[0] -
                                  margin)].iterrows():
            # [onset, offset]
            pred = [row_pred[1][dc[0]], row_pred[1][dc[1]]]
            # Check if the events overlap, and append the index
            # of the prediction df
            if check_detection_overlap(gs, pred):
                matched_idcs.append(row_pred[0])
                # No overlap found for this gold standard row
        if len(matched_idcs) == 0:
            match_df.loc[match_df_idx] = [row_gs[0], None]
        # One overlap found for this gold standard row
        elif len(matched_idcs) == 1:
            match_df.loc[match_df_idx] = [row_gs[0], matched_idcs[0]]
        else:
            dd_idx = (
                abs(check_df.loc[matched_idcs, dc[0]] -
                    row_gs[1][dc[0]])).idxmin()
            match_df.loc[match_df_idx] = [row_gs[0], dd_idx]

        match_df_idx += 1
    if not match_df.empty:
        match_df = match_df.apply(pd.to_numeric, errors="coerce",
                                  downcast="float")
    return match_df
