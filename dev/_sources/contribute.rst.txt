:orphan:

Contributing to MNE-HFO
========================

MNE-HFO is an open-source software project developed through community effort.
You are very welcome to participate in the development by reporting bugs,
writing documentation, or contributing code.

Development takes place on the collaborative platform GitHub at
`github.com/adam2392/mne-hfo <https://github.com/adam2392/mne-hfo>`_.

.. image:: https://mne.tools/mne-bids/assets/GitHub.png
   :width: 400
   :alt: GitHub Logo
   :target: https://github.com/adam2392/mne-hfo


Bug reports
-----------

Use the `GitHub issue tracker <https://github.com/adam2392/mne-hfo/issues>`_
to report bugs.

Contributing code or documentation
----------------------------------

Please see our `contributing guide <https://github.com/adam2392/mne-hfo/blob/master/CONTRIBUTING.md>`_
to find out how to get started.

Contributing your own sliding window-based detector
---------------------------------------------------

1. First step is opening up a `Github issue <https://github.com/adam2392/mne-hfo/issues>`_ to describe
the new HFO detector.

2. Secondly, one should follow our API for all Detectors, namely the ``mne_hfo.base.Detector`` class.
The basic hyperparameters of any Detector are:

    - threshold: The threshold point used in computation of HFOs. This could be a threshold on a certain metric, like line
        length, if the ``LineLengthDetector`` is used.
    - window size: The size of the windows in terms of samples
    - overlap: The amount of overlap between consecutive windows

One must also implement their own `_compute_hfo` function, which returns an HFO event array, consisting
of an array of ``(n_chs, n_windows)`` that is a binary existence of HFO detected or not. It is within
this function that all HFO detection algorithm should be implemented.

3. Finally, if your detector has extra hyperparameters then what may be inherited from ``Detector``, then
one can just add these to your own ``__init__()`` function.

Once you have implemented your detector, then it is always good to provide unit tests for determining
correct functionality of the HFO algorithm. This can be done using simulated HFOs using ``mne_hfo.simulation``
module, or using real data that you provide.