:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_hfo

What was new in previous releases?
==================================

.. currentmodule:: mne_hfo
.. _changes_0_2:

Version 0.2
-----------

Notable changes
~~~~~~~~~~~~~~~

- Added the ``HilbertDetector`` and optimized its performance on long recordings

Authors
~~~~~~~

* `Adam Li`_
* `Patrick Myers`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

This release is the first one under the ``mne.tools`` umbrella. We introduced three core detectors that 
we found were used and cited in the literature: Line Length, RMS and Hilbert detectors. We have organized 
a roadmap for what future improvements would entail. In addition, we have added tutorials that are rendered 
as jupyter notebooks, which walk through usage of the package to: i) load data from BIDS, ii) run a detection and 
then iii) evaluate the efficacy using tools from ``scikit-learn``.

Enhancements
^^^^^^^^^^^^

- Added :func:`mne_hfo.compute_chs_hfo_rates` to compute HFO rates per unit of time for every channel, by `Adam Li`) (:gh:`13`)
- Added :func:`mne_hfo.io.create_events_df` to generate a DataFrame of HFO events from ``Raw`` object, or dictionary of lists of HFO endpoints, by `Adam Li`_ (:gh:`7`)
- Added :func:`mne_hfo.find_coincident_events` to compare two dicts that contain event information by `Patrick Myers`_ (:gh:`10`)
- Added notebook to demo use of detection algorithms by `Patrick Myers`_ (:gh:`10`)
- Vectorized detection overlap check to enhance scoring speed by `Patrick Myers`_ (:gh:`15`)
- Added notebook to demo use of GridSearchCV to optimize detector performance by `Patrick Myers`_ (:gh:`15`)
- Added module to compare detections and notebook to demo usage by `Patrick Myers`_ (:gh:`22`)
- Added initial implementation of HilbertDetector by `Patrick Myers`_ (:gh:`23`)
- Improve memory utilization by allowing parallelization of the entire workflow per channel by `Patrick Myers`_ (:gh:`38`)


API changes
^^^^^^^^^^^

- Added :func:`mne_hfo.io.events_to_annotations` to go from ``*events.tsv`` to ``*annotations.tsv`` files, by `Adam Li`_ (:gh:`10`)
- Added :func:`mne_hfo.sklearn.make_Xy_sklearn` to format data into scikit-learn compatible data structures for the sake of running hyper-parameter searches with ``SearchCV`` functions, by `Adam Li`_ (:gh:`15`)
- Separated postprocessing step into two discrete steps _threshold_statistic and _post_process_ch_hfos by `Patrick Myers`_ (:gh:`23`)

Requirements
^^^^^^^^^^^^

- Updated requirement version for ``mne`` to ``v0.23+``, by `Adam Li`_ (:gh:``)
- Added ``tqdm``, ``joblib`` and ``pandas`` to requirements, by `Adam Li`_ (:gh:`7`)

Bug fixes
^^^^^^^^^

- Fixed channel name issue introduced by redundant type checks when using `fit_and_predict` by `Patrick Myers`_ (:gh:`15`)

.. include:: authors.rst
