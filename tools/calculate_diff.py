import numpy as np

from fft_audio import fft_audio


def diff_lowfreq_maxfreq(values: np.ndarray):
    return np.log10(np.min(values[:14500])) / np.log10(np.min(values[15500:19500]))

def calculate_diff(file1: str, file2: str):
    return diff_lowfreq_maxfreq(fft_audio(file1)) / diff_lowfreq_maxfreq(fft_audio(file2))
