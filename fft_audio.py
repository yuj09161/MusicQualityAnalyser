from typing import Optional

import numpy as np
from pydub import AudioSegment


def fft_audio(
    path: str,
    length: Optional[int] = None,
    start_time: Optional[int] = 0,
    *,
    channel_to_do_fft: Optional[int] = None
) -> np.ndarray:
    print('load file')
    audio = AudioSegment.from_file(path)
    frame_rate = audio.frame_rate
    channel_cnt = audio.channel

    print('make array')
    sig_arr = np.array(audio.get_array_of_samples()).reshape(-1, channel_cnt)
    if channel_to_do_fft is None:
        if channel_cnt > 2:
            to_do_fft = np.mean(sig_arr, 1)
    else:
        if channel_to_do_fft >= channel_cnt:
            raise ValueError('Invalid channel.')
        to_do_fft = sig_arr[:,channel_cnt]

    if length:
        print('cut audio')
        end_time = start_time + length
        if end_time * frame_rate > len(to_do_fft):
            raise ValueError('The audio is too short.')
        to_do_fft = to_do_fft[frame_rate * start_time:frame_rate * end_time]

    print('do fft')
    fft_res = abs(np.fft.fft(to_do_fft)) / len(to_do_fft)
    freq = np.fft.fftfreq(len(fft_res), 1 / audio.frame_rate)

    print('calculate misc')
    max_freq = int(freq.max() + 0.5)
    sample_per_group, remove_sample_cnt = divmod(len(fft_res) // 2, max_freq)
    remove_sample_cnt += len(fft_res) // 2

    print('finish...')
    return np.array([
        np.mean(freq[:-remove_sample_cnt].reshape(-1, sample_per_group), 1),
        np.mean(fft_res[:-remove_sample_cnt].reshape(-1, sample_per_group), 1),
    ])


if __name__ == '__main__':
    import os
    import sys
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('-s','--start-time', default=0, type=int)
    parser.add_argument('-l', '--length', default=None, type=int)
    args = parser.parse_args()

    plt.figure(args.file_path)
    result = fft_audio(args.file_path, args.length, args.start_time)
    plt.stem(result[0], np.log10(result[1]))
    plt.show()
