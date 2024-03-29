:orphan:

Installation
============

Dependencies
------------

* ``mne``
* ``mne-bids``
* ``numpy``
* ``scipy`` (for certain operations with EEGLAB data)
* ``joblib``
* ``pandas`` (optional, for generating event statistics)
* ``matplotlib`` (optional, for using the interactive data inspector)

We require that you use Python 3.8 or higher.
You may choose to install ``mne-hfo`` via pip.

Installation via Pip
--------------------

To install MNE-HFO including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -e .

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-hfo/zipball/master

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import mne_hfo'

MNE-HFO works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, run:

.. code-block:: bash

   pip install --user -U mne
