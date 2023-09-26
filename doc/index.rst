..  -*- coding: utf-8 -*-

.. _contents:

Overview of mne-hfo_
====================

.. _mne-hfo: https://readthedocs.org/projects/mne-hfo/

mne-hfo is a Python package for analysis of iEEG data for HFO events.

Motivation
----------

High-frequency oscillations are events that clinicians hypothesize to be related
to the epileptogenic zone. They have also been observed in other physiological
processes. They are loosely defined as oscillations in a "high-frequency band"
that are greater then some baseline according to a metric. For example,
the Line Length HFO detector, uses the ``line length`` metric of the time-series signal
to determine if a certain channel epoch is an HFO or not. In this package, we provide
utilities and algorithms for detecting HFOs that have been proposed in the literature.
In addition, we formulate the design of the package to be closely tied with ``scikit-learn``,
``mne-python``, and the ``BIDS`` data specification. These design choices make the
algorithms easy to tune, easy to use, and the results easy to share.

Python
------

Python is a powerful programming language that allows concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
mne-hfo uses to provide more features such as numerical linear algebra and
plotting.  In order to make the most out of mne-hfo you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

mne-hfo is free software; you can redistribute it and/or modify it under the
terms of the ``BSD`` license.  We welcome contributions.
Join us on `GitHub <https://github.com/adam2392/mne-hfo>`_.

Documentation
=============

mne-hfo is a HFO-detection package in python.

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   mne-hfo @ PyPI <https://pypi.org/project/mne-hfo/>
   Issue Tracker <https://github.com/mne-tools/mne-hfo/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. toctree::
   :hidden:

   News<whats_new>
   Install<install>
   Use<use>
   API<api>
   Tutorial<tutorial>
   Contribute<contribute>
