"""
Some parts of code are recoded from package Anderson Brito da Silva's pyhfo.

Reference: (https://github.com/britodasilva/pyhfo)
"""
import numpy as np
from scipy.stats import norm


# ----- Noise types -----
def simulate_pink_noise(N):
    """
    Create a pink noise (1/f) with N points.

    Parameters
    ----------
    N: int
        Number of samples to be returned

    Returns
    -------
    pink_noise_array: numpy array
        1D array of pink noise
    """
    M = N
    # ensure that the N is even
    if N % 2:
        N += 1

    x = np.random.randn(N)  # generate a white noise
    X = np.fft.fft(x)  # FFT

    # prepare a vector for 1/f multiplication
    nPts = int(N / 2 + 1)
    n = range(1, nPts + 1)
    n = np.sqrt(n)

    # multiplicate the left half of the spectrum
    X[range(nPts)] = X[range(nPts)] / n

    # prepare a right half of the spectrum - a copy of the left one
    X[range(nPts, N)] = np.real(X[range(int(N / 2 - 1), 0, -1)])
    X[range(nPts, N)] -= 1j * np.imag(X[range(int(N / 2 - 1), 0, -1)])

    y = np.fft.ifft(X)  # IFFT

    y = np.real(y)
    # normalising
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y ** 2))
    # returning size of N
    if M % 2 == 1:
        y = y[:-1]
    return y


def simulate_brown_noise(N):
    """
    Create a brown noise (1/f²) with N points.

    Parameters
    ----------
    N: int
        Number of samples to be returned

    Returns
    -------
    brown_noise_array: numpy array
        1D array of brown noise
    """
    M = N
    # ensure that the N is even
    if N % 2:
        N += 1

    x = np.random.randn(N)  # generate a white noise

    X = np.fft.fft(x)  # FFT

    # prepare a vector for 1/f² multiplication
    nPts = int(N / 2 + 1)
    n = range(1, nPts + 1)

    # multiplicate the left half of the spectrum
    X[range(nPts)] = X[range(nPts)] / n
    # prepare a right half of the spectrum - a copy of the left one
    X[range(nPts, N)] = np.real(X[range(int(N / 2 - 1), 0, -1)])
    X[range(nPts, N)] -= 1j * np.imag(X[range(int(N / 2 - 1), 0, -1)])

    y = np.fft.ifft(X)  # IFFT

    y = np.real(y)
    # normalising
    y -= np.mean(y)
    y /= np.sqrt(np.mean(y ** 2))
    # returning size of N
    if M % 2 == 1:
        y = y[:-1]
    return y


# ----- Artifacts -----
def simulate_delta(fs=5000, decay_dur=None):
    """
    Delta function with exponential decay.

    Parameters
    ----------
    fs: int
        Sampling frequency of the signal (default=5000)
    decay_dur: float
        Decay duration before returning to 0 in seconds

    Returns
    -------
    delta: numpy array
        1D numpy array with delta function
    """
    if decay_dur is None:
        decay_dur = np.random.random()

    decay_N = int(fs * decay_dur)
    return_value = 0.001  # This is the value where decay finishes
    decay_factor = np.log(return_value) / -decay_N
    t = np.linspace(0, decay_N, decay_N, endpoint=False)
    decay = np.exp(-t * decay_factor)

    delta = np.concatenate([[0], decay])

    return delta


def simulate_line_noise(fs=5000, freq=50, numcycles=None):
    """
    Line noise artifact.

    Parameters
    ----------
    fs: int
        Sampling frequency of the signal (default=5000)
    freq: float
        Line noise frequency (default=50)
    ncycles: float
        Number of cycles to create

    Returns
    -------
    line_noise: numpy array
        1D numpy array with line noise
    """
    if numcycles is None:
        numcycles = np.random.randint(3, 50)

    dur_samps = int((numcycles / freq) * fs)
    x = np.arange(dur_samps)
    y = np.sin(2 * np.pi * freq * x / fs)

    return y


def simulate_artifact_spike(fs=5000, dur=None):
    """
    Artifact like spike (sharp, not gaussian).

    Parameters
    ----------
    fs: int
        Sampling frequency of the signal (default=5000)
    dur: float
        Duration of the event in seconds

    Returns
    -------
    artifact_spike: numpy array
        1D numpy array with artifact spike
    """
    if dur is None:
        dur = round(np.random.random() / 10, 3)

    N = int(fs * dur)
    if not N % 2:  # Check if the number is odd - we want to have proper spike
        N -= 1
    y = np.zeros(N)
    y[:int(N / 2) + 1] = np.linspace(0, 1, int(N / 2) + 1)
    y[-int(N / 2):] = np.linspace(1, 0, int(N / 2) + 1)[1:]

    return y


# ----- HFO -----
def _wavelet(numcycles, f, fs):
    """
    Create a wavelet.

    Parameters
    ----------
    numcycles: int
        Number of cycles (gaussian window)
    f: float
        Central frequency
    fs: float
        Signal sampling rate (default=5000)

    Returns
    -------
    wave: numpy array
        1D numpy array with waveform
    time: numpy array
        1D numpy array with the time vector
    """
    N = int((fs * numcycles) / f)
    time = np.linspace((-numcycles / 2) / float(f),
                       (numcycles / 2) / float(f), N)  # time vector
    std = numcycles / (2 * np.pi * f)  # standard deviation
    wave = np.exp(2 * 1j * np.pi * f * time)
    wave *= np.exp(-(time ** 2) / (2 * (std ** 2)))  # waveform
    return wave, time


def simulate_hfo(fs=5000, freq=None, numcycles=None):
    """
    Create a simulated HFO signal.

    Parameters
    ----------
    fs: float
        Sampling rate of the signal (default=5000)
    freq: float
        Frequency of the artificial HFO (default=None - random frequency
        between 80 nad 600 Hz)
    numcycles: int
        Number of HFO cycles (default=None - cycles between 9 - 15)

    Returns
    -------
    wave: numpy array
        1D numpy array with waveform
    time: numpy array
        1D numpy array with the time vector
    """
    if numcycles is None:
        numcycles = np.random.randint(9, 15)
    if freq is None:
        freq = np.random.randint(80, 600)
    wave, time = _wavelet(numcycles, freq, fs)
    return np.real(wave), time


# ----- Spike -----
def simulate_spike(fs=5000, dur=None):
    """
    Create a simple gaussian spike.

    Parameters
    ----------
    fs: float
        Sampling rate (default=5000)
    dur: float
        Spike duration in seconds

    Returns
    -------
    spike numpy array
        1D numpy array with a sipke
    """
    if dur is None:
        dur = round(np.random.random() * 0.5, 2)

    x = np.linspace(-4, 4, int(fs * dur))  # 4 stds
    spike_dist = norm.pdf(x, loc=0, scale=1)  # Create gaussian shape
    # Normalize so that the peak is at 1
    spike = spike_dist * 1 / max(spike_dist)

    return spike
