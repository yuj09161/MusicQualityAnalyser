import numpy as np

from fft_audio import fft_audio


def diff_lowfreq_maxfreq(freqs: np.ndarray, values: np.ndarray):
    low_max_limit = (freqs < 14500).argmin()
    high_min_limit = (freqs > 15500).argmax()
    high_max_limit = (freqs > 19500).argmax()
    return np.log10(np.min(values[:low_max_limit])) / np.log10(np.min(values[high_min_limit:high_max_limit]))


def calculate_diff(file1: str, file2: str):
    return diff_lowfreq_maxfreq(*fft_audio(file1)) / diff_lowfreq_maxfreq(*fft_audio(file2))
