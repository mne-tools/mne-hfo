import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score, cohen_kappa_score
from sklearn.preprocessing import Normalizer, MinMaxScaler

from .posthoc import match_detected_annotations

implemented_labeling = ["raw-detections", "simple-binning", "overlap-predictions"]
implemented_comparisons = ["cohen-kappa", "mutual-info", "similarity-ratio"]

def compare_chart(det_list: list,
                     out_file = None,
                     normalize = True,
                     **comp_kw):
    """
    Compares similarity between detector results.
    Creates a plot of the comparison values in a len(det_list) x len(det_list) plot.


    The detectors should be fit to the same data.

    Parameters
    ----------
    det_list : List
        A list containing Detector instances. Detectors should already been fit
        to the data.
    out_file : String (Default: None)
        The file to write the chart to. If none, plot but do not save.
    normalize : Bool (Default: True)
        The method to use for comparison. Either 'cohen-kappa' or 'mutual-info'
    **comp_kw
        All other keywords are passed to compare_detectors().

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure containing detector comparison values.
    """

    chart_size = (len(det_list), len(det_list))

    comparison_values = []
    for i in det_list:
        for j in det_list:
            if i==j:
                comparison_values.append(1.0)
            else:
                ch_vals = compare_detectors(i, j, **comp_kw)
                comparison_values.append(np.mean([val for val in list(ch_vals.values()) if not math.isnan(val)]))

    comparison_values = np.reshape(comparison_values, chart_size)

    if normalize:
        transformer = MinMaxScaler().fit(comparison_values)
        minMaxVals = transformer.fit_transform(comparison_values)
        transformer = Normalizer().fit(minMaxVals)
        norm_vals = transformer.fit_transform(minMaxVals)
        comparison_values = norm_vals.copy()


    print("Plotting......make take a while")
    fig, ax = plt.subplots()
    im = ax.imshow(comparison_values, cmap='inferno')
    ax.set_xticks(np.arange(len(det_list)), labels=[det.__class__() for det in det_list])
    ax.set_yticks(np.arange(len(det_list)), labels=[det.__class__() for det in det_list])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(det_list)):
        for j in range(len(det_list)):
            if round(float(comparison_values[i, j]),3) > 0.5:
                color = 'k'
            else:
                color = 'w'
            text = ax.text(j, i, round(float(comparison_values[i, j]),3),
                        ha="center", va="center", color=color)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity (normalized)", rotation=-90, va="bottom")


    ax.set_title("Detector Comparison")
    fig.tight_layout()
    if out_file == None:
        plt.show()
    else:
        plt.savefig(out_file)

    return fig



def compare_detectors(clf_1, clf_2, 
                      **kwargs):
    """
    Compare fits for two classifiers per channel.

    Comparisons should be symmetrical.

    Parameters
    ----------
    clf_1 : Detector
        Detector that contains detections from calling detector.fit()
    clf_2 : Detector
        Detector that contains detections from calling detector.fit()
    **kwargs
        label_method : String, default : 'overlap-predictions'
            Implemented labeling method
        comp_method : String, default : 'mutual-info'
            Implemented comparison method
        bin_width : Int, default : 1
            Bin width if labeling requires a bin



    Returns
    -------
    ch_compares : dict
        Map of channel name to metric value.

    """

    if 'label_method' in kwargs:
        label_method = kwargs['label_method']
    else:
        label_method = 'overlap-predictions'

    if 'comp_method' in kwargs:
        comp_method = kwargs['comp_method']
    else:
        comp_method = 'mutual-info'
    
    if 'bin_width' in kwargs:
        bin_width = kwargs['bin_width']
    else:
        bin_width = 1

    if not hasattr(clf_1, 'hfo_annotations_'):
        raise RuntimeError("clf_1 must be fit to data before using compare")
    if not hasattr(clf_2, 'hfo_annotations_'):
        raise RuntimeError("clf_2 must be fit to data before using compare")
    if not hasattr(clf_1, 'n_times'):
        raise RuntimeError("clf_1 must be fit to data before using compare")
    if not hasattr(clf_2, 'n_times'):
        raise RuntimeError("clf_2 must be fit to data before using compare")
    
    #Ensure the classifiers are using data with the same length
    if clf_1.n_times == clf_2.n_times:
        data_len = clf_1.n_times
    else:
        raise RuntimeError("clf_1 and clf_2 must be fit on the same length of data")
    
    if label_method == 'raw-detections':
        label = _raw_detections
        label_kwargs = {'data_len' : data_len}
    elif label_method == 'simple-binning':
        label = _simple_binning
        label_kwargs = {'data_len' : data_len,
                        'bin_width' : bin_width}
    elif label_method == 'overlap-predictions':
        label = _overlap_predictions
        label_kwargs = {}
    else:
        raise NotImplementedError(f"Label Method must be one of "
                                  f"{', '.join(implemented_labeling)}. You"
                                  f"passed {label_method}.")
    if comp_method == 'cohen-kappa':
        comp = _cohen_kappa
    elif comp_method == "mutual-info":
        comp = _mutual_info
    elif comp_method == 'similarity-ratio':
        comp = _similarity_ratio
    else:
        raise NotImplementedError(f"Comp Method must be one of "
                                  f"{', '.join(implemented_comparisons)}. You"
                                  f"passed {comp_method}.")
        
    # Get the detection dataframes
    df1 = clf_1.to_data_frame(format="bids")
    df2 = clf_2.to_data_frame(format="bids")

    ch_names = clf_1.ch_names

    df1_labels, df2_labels = label(df1, df2, ch_names, **label_kwargs)
    
    ch_compares = comp(df1_labels, df2_labels, ch_names)

    return ch_compares
    

def _raw_detections(df1, df2, ch_names, data_len):
    """
    Returns labels for channel comparison.

    Labels each sample True if there is a detection. Labels False
    if there is no detection at the sample.

    Parameters
    ----------
    df1: Pd Dataframe 
        Dataframe that contains detection information for each channel.
    df2: Pd Dataframe
        Dataframe that contains detection information for each channel.
    ch_names: List[Str]
        The method to use for comparison. Either 'cohen-kappa' or 'mutual-info'

    Returns
    -------
    df1_labels: Dict
        Map of channel name to list of labeled values for dataframe 1.
    df2_labels: Dict
        Map of channel name to list of labeled values for dataframe 2.
    """
    label_kwargs = {'data_len' : data_len,
                        'bin_width' : 1}
    return _simple_binning(df1, df2, ch_names, **label_kwargs)

def _simple_binning(df1, df2, ch_names, data_len, bin_width):
    """
    Returns labels for channel comparison.

    Bins sample of data according to bin_width
    Labels each bin True if there is a detection. Labels False
    if there is no detection within the bin.

    Parameters
    ----------
    df1: Pd Dataframe 
        Dataframe that contains detection information for each channel.
    df2: Pd Dataframe
        Dataframe that contains detection information for each channel.
    ch_names: List[Str]
        The method to use for comparison. Either 'cohen-kappa' or 'mutual-info'
    data_len: Int
        The amount of samples in the raw data.
    bin_width: Int
        The length of a bin to be labeled.

    Returns
    -------
    df1_labels: Dict
        Map of channel name to list of labeled values for dataframe 1.
    df2_labels: Dict
        Map of channel name to list of labeled values for dataframe 2.
    """
    # Create dataframe groups by channel
    df1_g = df1.groupby('channels')
    df2_g = df2.groupby('channels')

    df1_labels = {}
    df2_labels = {}
    for ch in ch_names:
        if ch in df1_g.indices.keys():
            df1_channel = df1_g.get_group(ch)
            # Get the indices of the original dataframe for that channel
            df1_indices = list(df1_channel.index.values)
        else:
            df1_indices = []
        if ch in df2_g.indices.keys():
            df2_channel = df2_g.get_group(ch)
            # Get the indices of the original dataframe for that channel
            df2_indices = list(df2_channel.index.values)
        else:
            df2_indices = []

        ## Creates list of sets showing detection start and stop sample times {start_time, stop_time}
        df1_det_intervals = [{df1_channel.at[i, 'sample'], int(df1_channel.at[i, 'sample'] + df1_channel.at[i, 'duration'] * df1_channel.at[i, 'sfreq'])} for i in df1_indices]
        df2_det_intervals = [{df2_channel.at[i, 'sample'], int(df2_channel.at[i, 'sample'] + df2_channel.at[i, 'duration'] * df2_channel.at[i, 'sfreq'])} for i in df2_indices]

        df1_label = np.zeros(data_len)
        df2_label = np.zeros(data_len)

        ## Creates label array where detections are labeled as 1. Non times are labeled 0.
        for start, stop in df1_det_intervals:
            temp_interval = np.zeros_like(df1_label)
            temp_interval[start:stop] = 1
            df1_label = np.add(df1_label, temp_interval)

        for start, stop in df2_det_intervals:
            temp_interval = np.zeros_like(df2_label)
            temp_interval[start:stop] = 1
            df2_label = np.add(df2_label, temp_interval)

        ## If array cannot be properly split into bins, add nondetections to adjust size
        rem = data_len % bin_width
        num_bins = data_len // bin_width
        if rem != 0:
            df1_label = np.append(df1_label, ([0] * (bin_width - rem)))
            df2_label = np.append(df2_label, ([0] * (bin_width - rem)))
            num_bins += 1
        
        ## If any detection is in a bin, the bin becomes 'true'
        df1_label = np.reshape(df1_label, (num_bins, bin_width)).any(axis=1)
        df2_label = np.reshape(df2_label, (num_bins, bin_width)).any(axis=1)

        df1_labels[ch] = df1_label
        df2_labels[ch] = df2_label

    return df1_labels, df2_labels

def _overlap_predictions(df1, df2, ch_names, **kwargs):
    """
    Returns labels for channel comparison.

    Each detection span is a single label. If the dataframes
    contain a detection at the same moment, both will be labeled true.
    Detections found by only one classifier, one label set will have a
    true and the other a false.

    Parameters
    ----------
    df1: Pd Dataframe 
        Dataframe that contains detection information for each channel.
    df2: Pd Dataframe
        Dataframe that contains detection information for each channel.
    ch_names: List[Str]
        The method to use for comparison. Either 'cohen-kappa' or 'mutual-info'

    Returns
    -------
    df1_labels: Dict
        Map of channel name to list of labeled values for dataframe 1.
    df2_labels: Dict
        Map of channel name to list of labeled values for dataframe 2.
    """
    # Calculate the overlap between fits
    overlap_df = match_detected_annotations(df1, df2, method="match-total")

    # Create dataframe groups by channel
    df1_g = df1.groupby("channels")
    df2_g = df2.groupby("channels")

    df1_labels = {}
    df2_labels = {}
    for ch in ch_names:
        if ch in df1_g.indices.keys():
            df1_channel = df1_g.get_group(ch)
            # Get the indices of the original dataframe for that channel
            df1_indices = list(df1_channel.index.values)
        else:
            df1_indices = []
        if ch in df2_g.indices.keys():
            df2_channel = df2_g.get_group(ch)
            # Get the indices of the original dataframe for that channel
            df2_indices = list(df2_channel.index.values)
        else:
            df2_indices = []

        # Get the matching rows from the overlap dataframe
        # and merge these results
        df1_matches = overlap_df[overlap_df["true_index"].isin(df1_indices)]
        df2_matches = overlap_df[overlap_df["pred_index"].isin(df2_indices)]
        df_match_total = (
            pd.concat([df1_matches, df2_matches])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Get the detections as a list
        df1_predictions = list(df_match_total["true_index"])
        df2_predictions = list(df_match_total["pred_index"])

        # Convert the detections into labels
        df1_ch_labels = [~np.isnan(pred) for pred in df1_predictions]
        df2_ch_labels = [~np.isnan(pred) for pred in df2_predictions]

        df1_labels[ch] = np.array(df1_ch_labels)
        df2_labels[ch] = np.array(df2_ch_labels)

    return df1_labels, df2_labels

def _mutual_info(lab1, lab2, ch_names):
    """Call sklearn's mutual info function."""

    ch_compares = {}
    for ch in ch_names:
        try:
            res = mutual_info_score(lab1[ch], lab2[ch])
        except(ValueError):
            res = np.NAN

        ch_compares[ch] = res

    return ch_compares


def _cohen_kappa(lab1, lab2, ch_names):
    """Call sklearn's kappa score function."""

    ch_compares = {}
    for ch in ch_names:
        try:
            res = cohen_kappa_score(lab1[ch], lab2[ch])
        except(ValueError):
            res = np.NAN

        ch_compares[ch] = res

    return ch_compares

def _similarity_ratio(lab1, lab2, ch_names):
    """"""
    ch_compares = {}
    for ch in ch_names:
        matching = lab1[ch] == lab2[ch]
        ratio = np.sum(matching)/matching.shape[0]
        ch_compares[ch] = ratio

    return ch_compares

def compare_chs_plot(det, raw, channels=[], save_file=False):
    if channels == []:
        channels = det.ch_names

    df = det.to_data_frame(format="bids")
    df_g = df.groupby('channels')

    fig, axs = plt.subplots(len(channels), 1)
    sig_len = list(raw.shape)[1]
    data = raw.get_data()

    for n, ch in enumerate(channels):
        if ch not in det.ch_names: ## Not a valid channel
            raise ValueError(f"Channel '{ch}' is not"
                             f"found in raw data.")
        
        ch_idx = det.ch_names.index(ch)

        axs[n].plot(range(0, sig_len), data[ch_idx])
        axs[n].set_ylabel(f"{ch}")
        axs[n].set_yticklabels([])

        if ch in df.channels.values: ## Detections found in channel
            df_channel = df_g.get_group(ch)
            df_indices = list(df_channel.index.values)
            for i in df_indices:
                sfreq = df.at[i, 'sfreq']
                start_sample = df.at[i, 'onset'] * sfreq
                duration = df.at[i, 'duration'] * sfreq
                end_sample = start_sample + duration

                axs[n].axvspan(start_sample, end_sample, color='red', alpha=0.5)

    if save_file:
        fig.set_figheight(2 * len(channels))
        det1_name = det.__class__.__name__.replace("Detector", "")
        plt.savefig(f"{det1_name}_ChPlot.pdf")

    plt.show()

def compare_detectors_ch_plot(det1, det2, raw, channels=[], save_file=False):
    if channels == []:
        channels = det1.ch_names
    
    df1 = det1.to_data_frame(format="bids")
    df2 = det2.to_data_frame(format="bids")
    df1_g = df1.groupby('channels')
    df2_g = df2.groupby('channels')

    fig, axs = plt.subplots(len(channels), 1)
    sig_len = list(raw.shape)[1]
    data = raw.get_data()

    for n, ch in enumerate(channels):
        if ch not in det1.ch_names or ch not in det2.ch_names:
            raise ValueError(f"Channel '{ch}' is not"
                             f"found in raw data.")
        
        ch_idx = det1.ch_names.index(ch)

        axs[n].plot(range(0, sig_len), data[ch_idx])
        axs[n].plot(range(0, sig_len), data[ch_idx])
        axs[n].set_ylabel(f"{ch}")
        axs[n].set_yticklabels([])
        axs[n].set_yticklabels([])

        if ch in df1.channels.values: ## Detections found in detector 1 channel
            df_channel = df1_g.get_group(ch)
            df_indices = list(df_channel.index.values)
            for i in df_indices:
                sfreq = df1.at[i, 'sfreq']
                start_sample = df1.at[i, 'onset'] * sfreq
                duration = df1.at[i, 'duration'] * sfreq
                end_sample = start_sample + duration

                axs[n].axvspan(start_sample, end_sample, color='red', alpha=0.5)

        if ch in df2.channels.values: ## Detections found in detector 2 channel
            df_channel = df2_g.get_group(ch)
            df_indices = list(df_channel.index.values)
            for i in df_indices:
                sfreq = df2.at[i, 'sfreq']
                start_sample = df2.at[i, 'onset'] * sfreq
                duration = df2.at[i, 'duration'] * sfreq
                end_sample = start_sample + duration

                axs[n].axvspan(start_sample, end_sample, color='blue', alpha=0.5)
    
    if save_file:
        fig.set_figheight(2 * len(channels))
        det1_name = det1.__class__.__name__.replace("Detector", "")
        det2_name = det2.__class__.__name__.replace("Detector", "")
        plt.savefig(f"{det1_name}_{det2_name}_ChComparison.pdf")
    else:
        plt.show()