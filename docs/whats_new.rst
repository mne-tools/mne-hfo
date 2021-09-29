:orphan:

.. _whats_new:

What's new?
===========

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_hfo
.. _changes_0_3:

Version 0.3 (unreleased)
------------------------

xxx

Notable changes
~~~~~~~~~~~~~~~

- xxx

Authors
~~~~~~~
- `Adam Li`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- All detectors now use ``mne.Annotations`` under the hood to create dataframe ``sfreq``, by `Adam Li`_ (:gh:`49`)

API changes
^^^^^^^^^^^

- ``mne_hfo.create_annotations_df`` now requires the sampling rate to be passed into the argument ``sfreq``, by `Adam Li`_ (:gh:`49`)
- All functions used for creating events and handling ``events.tsv`` like dataframes were removed, by `Adam Li`_ (:gh:`49`)

Requirements
^^^^^^^^^^^^
- Now requires ``mne`` v0.23.4+ `Adam Li`_ (:gh:`49`)

Bug fixes
^^^^^^^^^

- xxx

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
