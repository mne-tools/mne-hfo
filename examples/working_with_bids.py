"""
===============================
02. Detect HFOs on BIDS Dataset
===============================

MNE-HFO requires strict adherence to the BIDS specification for EEG/iEEG data.
It currently depends on the data structures defined by ``MNE-Python`` and
``MNE-BIDS``.

In this example, we use MNE-BIDS to load real raw data and then use
MNE-HFO to detect HFOs. Specifically, we will follow these steps:

1. Load data via :func:`mne_bids.read_raw_bids`. We will load a sample subject
from the Fedele dataset [1].

2. Run a few ``mne_hfo.base.Detector`` instances to detect HFOs

3. Format the detected HFOs as a :class:`pandas.DataFrame`

4. Write to disk according to BEP-021_ and read it in again.

.. _BEP-021: https://docs.google.com/document/d/1PmcVs7vg7Th-cGC-UrX8rAhKUHIzOI-uIOh69_mvdlw/edit#

References
----------
[1] Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T, Sarnthein J.
    Resection of high frequency oscillations predicts seizure outcome in the individual
    patient. Scientific Reports. 2017;7(1):13836.
    https://www.nature.com/articles/s41598-017-13064-1. doi:10.1038/s41598-017-13064-1
"""  # noqa

# Authors: Adam Li <adam2392@gmail.com>
#

###############################################################################
# We are importing everything we need for this example:
from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids, make_report, print_dir_tree

from mne_hfo import (RMSDetector, events_to_annotations, write_annotations,
                     read_annotations)

###############################################################################
# Load the data
# -------------
#
# First, we need some data to work with. We will use the test dataset
# available with the repository under ``data/`` directory.
#

# root of BIDs dataset
root = Path('../data/')

# BIDS entities
subject = '01'
session = 'interictalsleep'
run = '01'
datatype = 'ieeg'

###############################################################################
# show the contents of the BIDS dataset
print_dir_tree(root)

# Let's summarize the dataset.
print(make_report(root, verbose=False))

###############################################################################
# Load the dataset.
bids_path = BIDSPath(subject=subject, session=session,
                     run=run, datatype=datatype, root=root,
                     suffix='ieeg', extension='.vhdr')
raw = read_raw_bids(bids_path)

###############################################################################
# Let's plot the data and see what it looks like
# raw.plot()

###############################################################################
# Detect HFOs
# -----------
# All detectors inherit from the base class ``mne_hfo.base.Detector``,
# which inherits from the :class:`sklearn.base.BaseEstimator` class.
# To run any estimator, one instantiates it along with the hyper-parameters,
# and then calls the ``fit`` function. Afterwards, detected HFOs are available
# in the various data structures. The recommended usage is the DataFrame, which
# is accessible via the ``mne_hfo.base.Detector.hfo_df`` property.

kwargs = {
    'threshold': 3,  # threshold for "significance"
    'win_size': 100,  # window size in samples
    'overlap': 0.25  # overlap in percentage relative to the window size
}
detector = RMSDetector(**kwargs)

# run detector
detector.fit(X=raw)

# get the HFO results as an events.tsv DataFrame
event_df = detector.hfo_event_df
print(event_df.head())

###############################################################################
# Convert HFO events to annotations
# ---------------------------------
# Detectors output HFO events detected as a DataFrame fashioned after the
# ``*_events.tsv`` files in BIDS-iEEG. Instead, HFO events are indeed
# Derivatives of the Raw data, that are estimated/detected using mne-hfo.
# The correct way to store them is in terms of an ``*_annotations.tsv``,
# according to the BIDS-Derivatives specification.

# convert event df -> annotation df
annot_df = events_to_annotations(event_df)

# alternatively save annotation dataframe to disc
annot_path = bids_path.copy().update(suffix='annotations',
                                     root=root / 'derivatives',
                                     extension='.tsv',
                                     check=False)

intended_for = raw.filenames[0]
write_annotations(annot_df, fname=annot_path,
                  intended_for=intended_for, root=root)
print(annot_df.head())

###############################################################################
# Read data back in
# -----------------
# The data will match what was written.
# In addition, you can check for overlapping HFOs.
annot_df = read_annotations(annot_path)

print(annot_df.head())
