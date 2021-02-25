import os
import platform

import numpy as np
import pytest
import shutil as sh
from pathlib import Path
from mne.utils import run_subprocess

from mne_hfo.simulate import simulate_hfo, simulate_spike


# WINDOWS issues:
# the bids-validator development version does not work properly on Windows as
# of 2019-06-25 --> https://github.com/bids-standard/bids-validator/issues/790
# As a workaround, we try to get the path to the executable from an environment
# variable VALIDATOR_EXECUTABLE ... if this is not possible we assume to be
# using the stable bids-validator and make a direct call of bids-validator
# also: for windows, shell = True is needed to call npm, bids-validator etc.
# see: https://stackoverflow.com/q/28891053/5201771


@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    vadlidator_args = ['--config.error=41']
    exe = os.getenv('VALIDATOR_EXECUTABLE', 'bids-validator')

    if platform.system() == 'Windows':
        shell = True
    else:
        shell = False

    bids_validator_exe = [exe, *vadlidator_args]

    def _validate(bids_root):
        cmd = [*bids_validator_exe, bids_root]
        run_subprocess(cmd, shell=shell)

    return _validate


@pytest.fixture(scope='function')
def test_bids_root(tmpdir):
    """Temporary BIDS dataset.

    Copies over dataset in ``data/`` to temporary directory.
    """
    data_path = Path('data')
    sh.copytree(data_path, tmpdir, dirs_exist_ok=True)
    return tmpdir


@pytest.fixture(scope="module")
def create_testing_eeg_data():
    """Create testing data with HFO and a spike."""
    freqs = [2.5, 6.0, 10.0, 16.0, 32.5, 67.5, 165.0,
             250.0, 425.0, 500.0, 800.0, 1500.0]

    fs = 5000
    n = fs * 10
    data = np.zeros(n)
    hfo_samps = []
    # basic_amp = 10

    x = np.arange(n)
    for freq in freqs:
        # freq_amp = basic_amp / freq
        y = np.sin(2 * np.pi * freq * x / fs)
        data += y

    # We have dummy data now inject 2 HFOs and a spike
    fs = 5000
    freq = 250
    numcycles = 9
    sim = simulate_hfo(fs, freq, numcycles)[0]
    ev_start = 5000
    data[ev_start: ev_start + len(sim)] += sim * 10
    hfo_samps.append((ev_start, ev_start + len(sim)))

    fs = 5000
    dur = 0.1
    sim = simulate_spike(fs, dur)
    ev_start = 4 * 5000
    data[ev_start: ev_start + len(sim)] += sim * 30

    fs = 5000
    freq = 500
    numcycles = 9
    sim = simulate_hfo(fs, freq, numcycles)[0]
    ev_start = 7 * 5000
    data[ev_start: ev_start + len(sim)] += sim * 10
    hfo_samps.append((ev_start, ev_start + len(sim)))

    return data, hfo_samps


@pytest.fixture(scope="module")
def create_testing_data():
    """Create testing data with certain frequencies."""
    freqs = [2.5, 6.0, 10.0, 16.0, 32.5, 67.5, 165.0, 425.0, 800.0, 1500.0]

    fs = 5000
    n = fs * 10
    data = np.zeros(n)
    # basic_amp = 10

    x = np.arange(n)
    for freq in freqs:
        # freq_amp = basic_amp / freq
        y = np.sin(2 * np.pi * freq * x / fs)
        data += y

    return data
