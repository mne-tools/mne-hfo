:orphan:

.. _whats_new:

.. currentmodule:: mne_hfo

What's new?
===========

.. currentmodule:: mne_hfo
.. _changes_0_1:

Version 0.1 (unreleased)
------------------------

xxx

Notable changes
~~~~~~~~~~~~~~~

- xxx

Authors
~~~~~~~

* `Adam Li`_
* `Patrick Myers`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Added :func:`mne_hfo.compute_chs_hfo_rates` to compute HFO rates per unit of time for every channel, by `Adam Li`) (:gh:`13`)
- Added :func:`mne_hfo.io.create_events_df` to generate a DataFrame of HFO events from ``Raw`` object, or dictionary of lists of HFO endpoints, by `Adam Li`_ (:gh:`7`)
- Added :func:`mne_hfo.find_coincident_events` to compare two dicts that contain event information by `Patrick Myers`_ (:gh:`10`)
- Added notebook to demo use of detection algorithms by `Patrick Myers`_ (:gh:`10`)

API changes
^^^^^^^^^^^

- Added :func:`mne_hfo.io.events_to_annotations` to go from ``*events.tsv`` to ``*annotations.tsv`` files, by `Adam Li`_ (:gh:`10`)
- Added :func:`mne_hfo.sklearn.make_Xy_sklearn` to format data into scikit-learn compatible data structures for the sake of running hyper-parameter searches with ``SearchCV`` functions, by `Adam Li`_ (:gh:`15`)

Requirements
^^^^^^^^^^^^

- Added ``tqdm``, ``joblib`` and ``pandas`` to requirements, by `Adam Li`_ (:gh:`7`)

Bug fixes
^^^^^^^^^

- xxx

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
