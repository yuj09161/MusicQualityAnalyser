from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from io import BytesIO
from multiprocessing import Queue, JoinableQueue, cpu_count
from typing import NamedTuple, Optional, Tuple
import os
import sys

from pydub import AudioSegment
import numpy as np


PLATFORM = sys.platform
if PLATFORM == 'win32':
    DATADIR = os.environ['localappdata'] + '/'
elif PLATFORM == 'linux':
    DATADIR = os.path.expanduser('~') + '/.local/share/'
elif PLATFORM == 'darwin':
    DATADIR = os.path.expanduser('~') + '/Library/Application Support/'
else:
    DATADIR = os.path.expanduser('~') + '/'


DEFAULT_SETTINGS = {
    'size_cut_enabled': True,
    'cut_size': 1536,

    'enable_cut': True,
    'start_time': '00:40',
    'duration': '00:20',

    'thread_cnt': max((cpu_count() - 2, 1)),

    'HDivM': 80,
    'NdivM': 30,
    'dB': -12.0,
    'dBDiff': 2,

    'freq_LH': 5000,
    'freq_ML': 10000,
    'freq_MH': 14500,
    'freq_HL': 16500,
    'freq_HH': 19500,
    'freq_NL': 20500,

    'include_fft_raw': False,

    'last_audio_dir': os.path.expanduser('~'),
    'last_results_dir': os.path.expanduser('~'),
}


class FFTConfig(NamedTuple):
    read_size: int
    start_time: int
    duration: int
    freq_LH: int
    freq_ML: int
    freq_MH: int
    freq_HL: int
    freq_HH: int
    freq_NH: int


class FFTResult(NamedTuple):
    succeed: bool
    id_: int
    dbFS: float
    intensity_L: Decimal
    intensity_M: Decimal
    intensity_H: Decimal
    intensity_N: Decimal
    raw_result: np.ndarray


def fft_audiosegment(
    audio: AudioSegment,
    start_time: int = 0,
    length: int = 0,
    *,
    channel_to_do_fft: Optional[int] = None
) -> np.ndarray:
    frame_rate = audio.frame_rate
    channel_cnt = audio.channels

    sig_arr = np.array(audio.get_array_of_samples()).reshape(-1, channel_cnt)
    if channel_to_do_fft is None:
        to_do_fft = np.mean(sig_arr, 1)
    elif channel_to_do_fft >= channel_cnt:
        raise ValueError('Invalid channel.')
    else:
        to_do_fft = sig_arr[:,channel_cnt]

    if length:
        end_time = start_time + length
        if end_time * frame_rate > len(to_do_fft):
            raise ValueError('The audio is too short.')
        to_do_fft = to_do_fft[frame_rate * start_time:frame_rate * end_time]
    else:
        length = len(to_do_fft) // frame_rate
        to_do_fft = to_do_fft[:len(to_do_fft) // frame_rate * frame_rate]

    fft_res = abs(np.fft.fft(to_do_fft)) / len(to_do_fft)
    return np.mean(fft_res[1:len(fft_res) // 2 + 1].reshape(-1, length), 1)


def do_fft(
    work: Tuple[int, str],
    config: FFTConfig
):
    id_, file_to_open = work
    print(file_to_open)
    print(os.getppid(), '|', os.getpid())
    try:
        if config.read_size:
            with open(file_to_open, 'rb') as file:
                data = file.read(config.read_size)
            file_to_open = BytesIO(data)
        audio = AudioSegment.from_file(file_to_open)

        fft_result = fft_audiosegment(audio, config.start_time, config.duration)

        return FFTResult(
            True,
            id_,
            audio.dBFS,
            Decimal(str(np.mean(fft_result[:config.freq_LH]))),
            Decimal(str(np.mean(fft_result[config.freq_ML:config.freq_MH]))),
            Decimal(str(np.mean(fft_result[config.freq_HL:config.freq_HH]))),
            Decimal(str(np.mean(fft_result[config.freq_NH:]))),
            fft_result
        )
    except Exception:
        return FFTResult(
            False,
            id_,
            0,
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            np.array([])
        )


if __name__ == '__main__':
    import time
    from multiprocessing import Process
    from multiprocessing.pool import Pool

    print(os.getppid(), '|', os.getpid())

    PATH_TO_ANALYSE = (
        'c:/Users/CKIRUser/Downloads/Better Things.opus',
        'c:/Users/CKIRUser/Downloads/Love Lee.opus',
        'c:/Users/CKIRUser/Downloads/미친 사랑의 노래.opus',
        # 'c:/Users/CKIRUser/Downloads/여섯 번째 여름.opus',
        # 'c:/Users/CKIRUser/Downloads/후라이의 꿈.opus',
        # 'c:/Users/CKIRUser/Downloads/tmp/Better Things.opus',
        # 'c:/Users/CKIRUser/Downloads/tmp/Love Lee.opus'
    )
    work_size = len(PATH_TO_ANALYSE)
    process_cnt = 2
    remain_cnt = work_size

    # config = FFTConfig(0, 0, 0, 5000, 10000, 14500, 16500, 19500, 20500)
    config = FFTConfig(1536 * 1024, 40, 20, 5000, 10000, 14500, 16500, 19500, 20500)

    works = [((k, path), config) for k, path in enumerate(PATH_TO_ANALYSE)]
    print(works)

    with Pool(process_cnt) as pool:
        result = pool.starmap_async(do_fft, works)
        for _ in range(work_size):
            print(result.get())
