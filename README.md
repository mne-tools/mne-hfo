[![Codecov](https://codecov.io/gh/adam2392/mne-hfo/branch/master/graph/badge.svg)](https://codecov.io/gh/adam2392/mne-hfo)
![.github/workflows/main.yml](https://github.com/adam2392/mne-hfo/workflows/.github/workflows/main.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/adam2392/mne-hfo.svg?style=svg)](https://circleci.com/gh/adam2392/mne-hfo)
![License](https://img.shields.io/pypi/l/mne-bids)
[![Code Maintainability](https://api.codeclimate.com/v1/badges/3afe97439ec5133ce267/maintainability)](https://codeclimate.com/github/adam2392/mne-hfo/maintainability)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mne-hfo)
[![Documentation Status](https://readthedocs.org/projects/mne-hfo/badge/?version=latest)](https://mne-hfo.readthedocs.io/en/latest/?badge=latest)
[![PyPI Download count](https://pepy.tech/badge/mne-hfo)](https://pepy.tech/project/mne-hfo)

MNE-HFO
=======

MNE-HFO is a Python package that computes estimates of high-frequency oscillations in iEEG data stored in
the [BIDS](https://bids.neuroimaging.io/)-compatible datasets with the help of
[MNE-Python](https://mne.tools/stable/index.html).

The initial code was adapted and taken from: https://gitlab.com/icrc-bme/epycom
to turn into a sklearn-compatible API that works with ``mne-python``. Additional algorithms and functionality are added.

NOTE: This is currently in ALPHA stage, and we are looking for contributors. Please get in touch via Issues tab if you
would like to contribute.

High frequency oscillations in epilepsy
---------------------------------------
A few notes that are worthy of reading. The initial papers on HFOs (Staba et al.)
actually only observed HFOs on Hippocampus. In addition, the papers cited that are implemented all selected data before
developing their algorithm (i.e. selected channels with HFOs).

It is also noted that the Hilbert detector was used to show HFOs exist in normal brain function, possibly unassociated
with the epileptogenic zone.

Why?
----
Currently HFO detection and algorithms are segmented in Matlab files, which are sometimes not open-source, or possibly
difficult to use. In addition, validation of HFO algorithms depend on i) sharing the algorithms ii) sharing the results
with others in a readable format and iii) comparing algorithms against each other on the same dataset.

MNE-HFO links BIDS, MNE-Python and iEEG HFO event detection with the goal to make HFO detection more transparent, more
robust, and facilitate data and code sharing with co-workers and collaborators.

Installation
------------
Installation is RECOMMENDED via a python virtual environment, using ``pipenv``. The package is hosted on ``pypi``, which
can be installed via pip, or pipenv.

    pip install mne-hfo

or

    pipenv install mne-hfo

Note: Installation has been tested on MacOSX and Ubuntu, but should 
probably work on Windows too.

Documentation and Usage
-----------------------

The documentation can be found under the following links:

- for the [stable release](https://mne-hfo.readthedocs.io/en/stable/index.html)
- for the [latest (development) version](https://mne-hfo.readthedocs.io/en/latest/index.html)

Note: Functionality has been tested on MacOSX and Ubuntu.

Basic Working Example
---------------------

A basic working example is listed here, assuming one has loaded in a mne-Python ``Raw`` object already.
    
    from mne_hfo import RMSDetector
    detector = RMSDetector()

    # assume user has loaded in raw iEEG data using mne-python
    detector.fit(raw)

    # get the HFO events as an *events.tsv style dataframe
    hfo_event_df = detector.hfo_event_df

    # get the HFO events as an *events.tsv style dataframe
    hfo_annot_df = detector.hfo_df


All output to ``*events.tsv`` BIDS-compliant files will look like the following:

| onset      | duration | sample | trial_type |
| ---------- | -------- | ------ | ---------- |
| 1     | 3    | 1000   | hfo_A2-A1  |

which will imply that there is an HFO detected using a bipolar referencing at channel ``A2-A1``
at 1 second with duration of 3 seconds. The onset sample occurs at sample 1000 (thus ``sfreq`` is 1000 Hz). If a
monopolar referencing is used, then the ``trial_type`` might be ``hfo_A2`` to imply that an HFO was detected at
channel ``A2``.

Alternatively, one can output the data in the form of a derivatives ``Annotations`` 
DataFrame, which is the RECOMMENDED way. Outputting data according to BIDS Extension Proposal 21, instead would result in 
an ``*annotations.tsv`` file. 

| onset      | duration | label | channels |
| ---------- | -------- | ------ | ---------- |
| 1     |  3  | hfo  | A2-A1 | 

with a corresponding ``*annotations.json`` file.

    {
        'IntendedFor': sub-01/ses-01/eeg/sub-01_ses-01_task-01_eeg.<ext>,
        'Description': 'Automatic annotations of HFO events using mne-hfo.',
    }

Optimizing Hyperparameters
--------------------------

In all ``MNE-HFO`` HFO detectors, we assume that there are hyper-parameters 
specified by the proposed algorithm. These hyper-parameters can be tuned automatically 
using the ``scikit-learn`` API for [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html#grid-search).


    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from mne_hfo.score import accuracy
    from mne_hfo.sklearn import make_Xy_sklearn, DisabledCV
    
    # define hyperparameter grid to search over
    parameters = {'threshold': [1, 2, 3], 'win_size': [50, 100, 250]}
    
    # define HFO detector
    detector = LineLengthDetector()

    # define a scoring function 
    scorer = make_scorer(accuracy)

    # we don't use cross-validation since the
    # HFO algorithm is deterministic
    cv = DisabledCV()

    # instantiate the GridSearch object
    gs = GridSearchCV(detector, param_grid=parameters, scoring=scorer,
                      cv=cv, refit=False, verbose=True)

    # load in raw data
    # raw = <load_in_raw_data>

    # load in HFO annotations
    # annot_df = <load_in_annotations>

    # make sklearn compatible
    raw_df, y = make_Xy_sklearn(raw, annot_df)

    # run hyperparameter tuning based on accuracy score
    gs.fit(raw_df, y, groups=None)

    # show the results
    print(gs.cv_results_["mean_test_score"])

In the above example, to load in raw data, one can use [``mne-bids``](https://github.com/mne-tools/mne-bids)
and to load in the annotations dataframe, one can check out our API 
for different ways of doing so.

Citing
------
For testing and demo purposes, we use the dataset in [1]. If you use the demo/testing dataset, please cite that paper.
If you use ``mne-hfo`` itself in your research, please cite the paper (TBD).

    Adam Li. (2021, February 1). MNE-HFO: An open-source Python implementation of HFO detection algorithms (Version 0.0.1). Zenodo. http://doi.org/10.5281/zenodo.4485036

References
----------
[1] Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T, Sarnthein J. Resection of high frequency
oscillations predicts seizure outcome in the individual patient. Scientific Reports. 2017;7(1):13836.
https://www.nature.com/articles/s41598-017-13064-1
doi:10.1038/s41598-017-13064-1
