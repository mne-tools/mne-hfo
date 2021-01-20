:orphan:

Using MNE-BIDS
==============

Quickstart
----------

Python
~~~~~~

.. code:: python

    >>> import mne
        >>> from mne_hfo import BIDSPath, write_raw_bids
        >>> raw = mne.io.read_raw_fif('my_old_file.fif')
        >>> bids_path = BIDSPath(subject='01', session='01, run='05',
                                 datatype='meg', bids_root='./bids_dataset')
        >>> write_raw_bids(raw, bids_path=bids_path)
    >>> from mne_bids import BIDSPath, write_raw_bids
    >>> raw = mne.io.read_raw_fif('my_old_file.fif')
    >>> bids_path = BIDSPath(subject='01', session='01, run='05',
                             datatype='meg', bids_root='./bids_dataset')
    >>> write_raw_bids(raw, bids_path=bids_path)


.. _bidspath-intro:

Mastering BIDSPath
------------------
To be able to effectively use MNE-BIDS, you need to understand how to work with
the ``BIDSPath`` object. Follow
:ref:`our basic example <read_bids_datasets-example>` on how to read a
BIDS dataset, and have a look at
:ref:`this introduction <bidspath-example>`
to learn everything you need!


.. include:: auto_examples/index.rst
   :start-after: :orphan:
