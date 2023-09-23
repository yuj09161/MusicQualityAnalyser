from typing import Optional

import numpy as np
from pydub import AudioSegment


def fft_audio(
    path: str,
    length: int = 0,
    start_time: int = 0,
    *,
    channel_to_do_fft: Optional[int] = None
) -> np.ndarray:
    print('load file')
    audio = AudioSegment.from_file(path)
    frame_rate = audio.frame_rate
    channel_cnt = audio.channels

    print('make array')
    sig_arr = np.array(audio.get_array_of_samples()).reshape(-1, channel_cnt)
    if channel_to_do_fft is None:
        to_do_fft = np.mean(sig_arr, 1)
    elif channel_to_do_fft >= channel_cnt:
        raise ValueError('Invalid channel.')
    else:
        to_do_fft = sig_arr[:,channel_cnt]

    print('cut audio')
    if length:
        end_time = start_time + length
        if length == 0:
            raise ValueError('Zero length.')
        if end_time * frame_rate > len(to_do_fft):
            raise ValueError('The audio is too short.')
        to_do_fft = to_do_fft[frame_rate * start_time:frame_rate * end_time]
    else:
        length = len(to_do_fft) // frame_rate
        to_do_fft = to_do_fft[:len(to_do_fft) // frame_rate * frame_rate]

    print('do fft')
    fft_res = abs(np.fft.fft(to_do_fft)) / len(to_do_fft)

    print('finish...')
    return np.mean(fft_res[1:len(fft_res) // 2 + 1].reshape(-1, length), 1)


if __name__ == '__main__':
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('-s','--start-time', default=0, type=int)
    parser.add_argument('-l', '--length', default=None, type=int)
    args = parser.parse_args()

    plt.figure(args.file_path)
    result = fft_audio(args.file_path, args.length, args.start_time)
    plt.stem(np.arange(1, len(result) + 1), np.log10(result))
    plt.show()
