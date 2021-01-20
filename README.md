
[![Codecov](https://codecov.io/gh/adam2392/mne-hfo/branch/master/graph/badge.svg)](https://codecov.io/gh/adam2392/mne-hfo)
[![GitHub Actions](https://github.com/adam2392/mne-hfo/workflows/test_suite/badge.svg)](https://github.com/adam2392/mne-hfo/actions)
[![CircleCI](https://circleci.com/gh/adam2392/mne-hfo.svg?style=svg)](https://circleci.com/gh/adam2392/mne-hfo)
![License](https://img.shields.io/pypi/l/mne-bids)
[![Code Maintainability](https://api.codeclimate.com/v1/badges/3afe97439ec5133ce267/maintainability)](https://codeclimate.com/github/adam2392/mne-hfo/maintainability)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

[comment]: <> ([![PyPI Download count]&#40;https://pepy.tech/badge/mne-bids&#41;]&#40;https://pepy.tech/project/mne-bids&#41;)

MNE-HFO
=======

MNE-HFO is a Python package that computes estimates of high-frequency oscillations 
in iEEG data stored in the [BIDS](https://bids.neuroimaging.io/)-compatible datasets with the help of
[MNE-Python](https://mne.tools/stable/index.html).

The initial code was adapted and taken from: https://gitlab.com/icrc-bme/epycom 
to turn into a sklearn-compatible API that works with ``mne-python``. Additional algorithms and functionality are 
added.

NOTE: This is currently in ALPHA stage, and we are looking for 
contributors. Please get in touch via Issues tab if you would like to
contribute.

High frequency oscillations in epilepsy
---------------------------------------
A few notes that are worthy of reading. The initial papers on HFOs (Staba et al.) 
actually only observed HFOs on Hippocampus. In addition, the papers cited that 
are implemented all selected data before developing their algorithm (i.e. selected 
channels with HFOs). 

It is also noted that the Hilbert detector was used to show HFOs
exist in normal brain function, possibly unassociated with 
the epileptogenic zone.

Why?
----
Currently HFO detection and algorithms are segmented in Matlab files,
which are sometimes not open-source, or possibly difficult to use. In 
addition, validation of HFO algorithms depend on i) sharing the algorithms 
ii) sharing the results with others in a readable format and iii) comparing 
algorithms against each other on the same dataset.

MNE-HFO links BIDS, MNE-Python and iEEG HFO event detection with the goal to make HFO 
detection more transparent, more robust, and facilitate data and code sharing with 
co-workers and collaborators.

How?
----

The documentation can be found under the following links:

- for the [stable release](https://mne.tools/mne-bids/)
- for the [latest (development) version](https://mne.tools/mne-bids/dev/index.html)

A basic working example is listed here, assuming one has loaded in a 
mne-Python ``Raw`` object already.

    from mne_hfo import RMSDetector
    detector = RMSDetector()
    detector.fit(raw)

Citing
------
For testing and demo purposes, we use the dataset in [1]. 
If you use the demo/testing dataset, please cite that paper. 
If you use ``mne-hfo`` itself in your research, please cite 
the paper (TBD).

References
----------
[1] Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T, Sarnthein J. Resection of high frequency oscillations predicts seizure outcome in the individual patient.
Scientific Reports. 2017;7(1):13836.
https://www.nature.com/articles/s41598-017-13064-1
doi:10.1038/s41598-017-13064-1
