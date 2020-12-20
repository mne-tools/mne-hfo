
[![Codecov](https://codecov.io/gh/mne-tools/mne-bids/branch/master/graph/badge.svg)](https://codecov.io/gh/mne-tools/mne-bids)
[![GitHub Actions](https://github.com/mne-tools/mne-bids/workflows/build/badge.svg)](https://github.com/mne-tools/mne-bids/actions)
[![CircleCI](https://circleci.com/gh/mne-tools/mne-bids.svg?style=svg)](https://circleci.com/gh/mne-tools/mne-bids)
[![PyPI Download count](https://pepy.tech/badge/mne-bids)](https://pepy.tech/project/mne-bids)
[![Latest PyPI release](https://img.shields.io/pypi/v/mne-bids.svg)](https://pypi.org/project/mne-bids/)
[![Latest conda-forge release](https://img.shields.io/conda/vn/conda-forge/mne-bids.svg)](https://anaconda.org/conda-forge/mne-bids/)
[![JOSS publication](https://joss.theoj.org/papers/5b9024503f7bea324d5e738a12b0a108/status.svg)](https://joss.theoj.org/papers/5b9024503f7bea324d5e738a12b0a108)

MNE-HFO
=======

MNE-HFO is a Python package that computes estimates of high-frequency oscillations 
in iEEG data stored in the [BIDS](https://bids.neuroimaging.io/)-compatible datasets with the help of
[MNE-Python](https://mne.tools/stable/index.html).

The initial code was adapted and taken from: https://gitlab.com/icrc-bme/epycom 
to turn into a sklearn-compatible API. Additional algorithms and functionality are 
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

Citing
------
TBD
