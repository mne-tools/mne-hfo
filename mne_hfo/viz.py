import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne_hfo import merge_channel_events, merge_overlapping_events
from mne_hfo.io import create_annotations_df

def plot_hfos(raw, annotations):
    mne.viz.set_browser_backend("qt")
    raw.set_annotations(annotations)
    raw.plot(block=True)

def plot_hfo_event(raw, annotations, eventId):
    mne.viz.set_browser_backend("qt")

    # convert annotations to annotations dataframe
    onset = annotations.onset
    duration = annotations.duration
    label = annotations.description
    sfreq = raw.info["sfreq"]

    # each annotation only has one channel associated with it
    ch_names = [ch[0] for ch in annotations.ch_names]

    # create an annotations dataframe
    annotations_df = create_annotations_df(
        onset, duration, ch_names, annotation_label=label, sfreq=sfreq
    )
    merged = merge_channel_events(annotations_df)
    print(f"Number of events: {merged.shape[0]}")

    event = merged.iloc[eventId]
    onset = event["onset"]
    duration = event["duration"]
    channels = event["channels"]
    orig_indices = event["orig_indices"]
    t=np.arange(onset, onset+duration, 1/sfreq)
    
    annotations.description = [f"hfo_{ch}" for ch in ch_names]

    raw.set_annotations(annotations)
    ch_indices = [ch_names.index(i) for i in channels]
    subset = raw.get_data(tmin=onset, tmax=onset+duration, picks=ch_indices)

    ch_mins = np.min(subset, axis=1)
    ch_maxs = np.max(subset, axis=1)

    fig, axs = plt.subplots(subset.shape[0],1, sharex='col', gridspec_kw={'hspace': 0})
    # if type(axs) != list:
    #     axs = [axs]
    for i, ax in enumerate(axs):
        ch = ch_names[ch_indices[i]]
        ax.plot(t,subset[i])
        ax.set_ylabel(ch, rotation=20, labelpad=20)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")

        for j in orig_indices:
            orig_event = annotations_df.iloc[j]
            if orig_event.channels == ch:
                orig_onset = orig_event.onset
                orig_duration = orig_event.duration
                t_event = np.arange(orig_onset, orig_onset+orig_duration, 1/sfreq)
                ax.fill_between(t_event, ch_mins[i], ch_maxs[i], facecolor='red', alpha=0.5)
    fig.suptitle(f"Detections comprising Event {eventId}")
    plt.show()
    ...


def get_merged_annotations():
    pass