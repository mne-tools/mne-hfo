:orphan:

.. _api_documentation:

=================
API Documentation
=================

Detectors
---------
:py:mod:`mne_hfo`:

.. automodule:: mne_hfo
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_hfo

.. autosummary::
   :toctree: generated/

    LineLengthDetector
    RMSDetector
    HilbertDetector

BIDS-IO functions
-----------------

:py:mod:`mne_hfo.io`:

.. automodule:: mne_hfo.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_hfo.io

.. autosummary::
   :toctree: generated/

    create_annotations_df
    read_annotations
    write_annotations

Post-processing HFO Detections
------------------------------

.. currentmodule:: mne_hfo

.. autosummary::
   :toctree: generated/

    match_detected_annotations
    find_coincident_events
    compute_chs_hfo_rates
    merge_overlapping_events

Help transform data to be scikit-learn compatible (for SearchCV)
----------------------------------------------------------------

.. currentmodule:: mne_hfo.sklearn

.. autosummary::
   :toctree: generated/

    make_Xy_sklearn
    DisabledCV

Metrics
-------

:py:mod:`mne_hfo.utils`:

.. automodule:: mne_hfo.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_hfo.utils

.. autosummary::
   :toctree: generated/

    compute_rms
    compute_line_length
    compute_hilbert
    apply_std
    apply_hilbert
    threshold_tukey

Simulation
----------

:py:mod:`mne_hfo.simulate`:

.. automodule:: mne_hfo.simulate
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_hfo.simulate

.. autosummary::
   :toctree: generated/

    simulate_pink_noise
    simulate_brown_noise
    simulate_line_noise
    simulate_delta
    simulate_artifact_spike
    simulate_spike
    simulate_hfo
