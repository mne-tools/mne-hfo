# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


import numpy as np

from mne_hfo.simulate import (simulate_pink_noise,
                              simulate_brown_noise,
                              simulate_delta,
                              simulate_line_noise,
                              simulate_artifact_spike,
                              simulate_hfo,
                              simulate_spike)


# ----- Noise -----
def test_simulate_pinknoise():
    sim = simulate_pink_noise(5000)
    assert round(np.std(sim), 5) == 1.0


def test_simulate_brownnoise():
    sim = simulate_brown_noise(5000)
    assert round(np.std(sim), 5) == 1.0


# ----- Artifacts -----
def test_simulate_delta():
    fs = 5000
    decay_dur = 0.1
    sim = simulate_delta(fs, decay_dur)
    assert len(sim) == int(decay_dur * fs) + 1


def test_simulate_line_noise():
    fs = 5000
    freq = 50
    numcycles = 10
    sim = simulate_line_noise(fs, freq, numcycles)
    assert len(sim) == int((numcycles / freq) * fs)


def test_simulate_artifact_spike():
    fs = 5000
    dur = 0.1
    sim = simulate_artifact_spike(5000, dur)
    assert len(sim) == int(fs * dur) - 1


# ----- HFO -----
def test_simulate_hfo():
    fs = 5000
    freq = 250
    numcycles = 9
    sim = simulate_hfo(fs, freq, numcycles)[0]
    assert len(sim) == int((numcycles / freq) * fs)


# ----- Spikes -----
def test_simulate_spike():
    fs = 5000
    dur = 0.1
    sim = simulate_spike(fs, dur)
    assert len(sim) == int(fs * dur)
