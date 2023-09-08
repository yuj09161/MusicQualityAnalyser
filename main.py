from PySide6.QtCore import QRunnable, QThreadPool, QTimer, Signal, SignalInstance
from PySide6.QtGui import QCloseEvent, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QWidget,
)
from __feature__ import snake_case, true_property

from decimal import Decimal
from io import BytesIO
from itertools import chain, repeat
from multiprocessing import cpu_count
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from queue import Queue
import json
import os
import sys

from pydub import AudioSegment
import numpy as np

from Ui import Ui_Main


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


class FileResult(NamedTuple):
    path: str
    dBFS: float
    intensity_L: Decimal
    intensity_M: Decimal
    intensity_H: Decimal
    intensity_N: Decimal
    H_div_M: Decimal
    N_div_M: Decimal
    raw_result: Optional[np.ndarray]


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
    results: Queue,
    config: FFTConfig
):
    id_, file_to_open = work
    try:
        if config.read_size:
            with open(file_to_open, 'rb') as file:
                data = file.read(config.read_size)
            file_to_open = BytesIO(data)
        audio = AudioSegment.from_file(file_to_open)

        fft_result = fft_audiosegment(audio, config.start_time, config.duration)

        results.put_nowait(FFTResult(
            True,
            id_,
            audio.dBFS,
            Decimal(str(np.mean(fft_result[:config.freq_LH]))),
            Decimal(str(np.mean(fft_result[config.freq_ML:config.freq_MH]))),
            Decimal(str(np.mean(fft_result[config.freq_HL:config.freq_HH]))),
            Decimal(str(np.mean(fft_result[config.freq_NH:]))),
            fft_result
        ))
    except Exception:
        results.put_nowait(FFTResult(
            False,
            id_,
            0,
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            np.array([])
        ))


class Analyser(QRunnable):
    def __init__(
        self,
        work_queue: Queue,
        result_queue: Queue,
        config: FFTConfig
    ):
        super().__init__()

        self.__works = work_queue
        self.__results = result_queue
        self.__config = config

    def run(self):
        while not self.__works.empty():
            id_, path = self.__works.get_nowait()
            do_fft((id_, path), self.__results, self.__config)
            self.__works.task_done()


class AudioInfo(QStandardItemModel):
    HEADER_TEXTS = (
        '파일 경로',
        '상태',
        'dBFS',
        'L', 'M', 'H', 'N',
        'H/M', 'N/M' 
    )

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.set_horizontal_header_labels(self.HEADER_TEXTS)

        self.analyse_results: Dict[int, FileResult] = {}
        self.file_to_analyse: List[Tuple[int, str]] = []

    def clear(self):
        super().clear()
        self.analyse_results: Dict[int, FileResult] = {}
        self.file_to_analyse: List[Tuple[int, str]] = []

    def clear_file_to_analyse(self) -> None:
        self.file_to_analyse.clear()

    def clear_and_get_file_to_analyse(self) -> List[Tuple[int, str]]:
        file_to_analyse = self.file_to_analyse.copy()
        self.file_to_analyse.clear()
        return file_to_analyse

    def add_file(self, file_path: str) -> None:
        self.file_to_analyse.append((len(self.file_to_analyse), file_path))
        self.append_row(self.__make_items(chain(
            (file_path,), repeat('-', len(self.HEADER_TEXTS) - 1)
        )))

    def add_files(self, file_paths: Iterable[str]) -> None:
        for path in file_paths:
            self.add_file(path)

    def set_result(self, index: int, result: FFTResult) -> None:
        intensity_M = result.intensity_M.log10()
        intensity_H = result.intensity_H.log10()
        intensity_N = result.intensity_N.log10()

        if not result.succeed:
            self.item(index, 1).set_text('실패')

        file_result = FileResult(
            self.item(index, 0).text(),
            result.dbFS,
            result.intensity_L.log10(),
            intensity_M,
            intensity_H,
            intensity_N,
            intensity_H / intensity_M,
            intensity_N / intensity_M,
            result.raw_result
        )
        self.analyse_results[index] = file_result

        self.item(index, 1).set_text('완료')
        self.item(index, 2).set_text(f'{file_result.dBFS:.3f}')
        for k in range(2, 6):
            self.item(index, k + 1).set_text(f'{file_result[k]:.2f}')
        for k in range(6, 8):
            self.item(index, k + 1).set_text(f'{file_result[k] * 100:.2f}%')

    def import_(self, data: Iterable) -> None:
        self.clear()
        self.set_horizontal_header_labels(self.HEADER_TEXTS)

        for k, (
            path,
            dBFS,
            intensity_L,
            intensity_M,
            intensity_H,
            intensity_N,
            H_div_M,
            N_div_M,
            raw_result
        ) in enumerate(data):
            if raw_result is not None:
                raw_result = np.array(list(map(np.double, raw_result)))
            file_result = FileResult(
                path,
                dBFS,
                Decimal(intensity_L),
                Decimal(intensity_M),
                Decimal(intensity_H),
                Decimal(intensity_N),
                Decimal(H_div_M),
                Decimal(N_div_M),
                raw_result
            )
            self.analyse_results[k] = file_result

            texts = [path, '완료', f'{file_result.dBFS:.3f}']
            for intensity in (
                intensity_L, intensity_M, intensity_H, intensity_N
            ):
                texts.append(f'{Decimal(intensity):.2f}')
            for intensity_div in (
                H_div_M, N_div_M
            ):
                texts.append(f'{Decimal(intensity_div) * 100:.2f}%')
            self.append_row(self.__make_items(texts))

    def export(self, export_raw_result: bool) -> List:
        return [
            (
                (result := self.analyse_results[k]).path,
                result.dBFS,
                str(result.intensity_L),
                str(result.intensity_M),
                str(result.intensity_H),
                str(result.intensity_N),
                str(result.H_div_M),
                str(result.N_div_M),
                list(map(str, result.raw_result))
                if export_raw_result and result.raw_result is not None else None
            ) for k in sorted(self.analyse_results.keys())
        ]
    
    def __make_items(self, texts):
        items = []
        for d in texts:
            item = QStandardItem(d)
            item.set_editable(False)
            items.append(item)
        return items


class Main(QMainWindow, Ui_Main):
    __CFG_PATH = DATADIR + 'hys.musicqualityanalyser/config.json'

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)

        self.__DISABLE_ON_START_WORK = (
            self.chkEnableSizeCut,
            self.spSize,
            self.chkEnableCut,
            self.teStart,
            self.teDuration,
            self.spThread,
            self.spHDivM,
            self.spNdivM,
            self.spdB,
            self.spdBDiff,
            self.btnApplyThreshold,
            self.spFreqLH,
            self.spFreqML,
            self.spFreqMH,
            self.spFreqHL,
            self.spFreqHH,
            self.spFreqNL,
            self.btnLoadRes,
            self.chkIncludeFFTRes,
            self.btnSaveRes,
            self.btnAddFiles
        )

        self.__remain_works = 0
        self.__results = {}
        self.__result_queue: Queue[FFTResult] = Queue()

        self.__result_check_timer = QTimer(self)
        self.__result_check_timer.interval = 250
        self.__result_check_timer.single_shot_ = False
        self.__result_check_timer.timeout.connect(self.__get_results)

        self.__audio_infos = AudioInfo(self.tvResult)
        self.tvResult.set_model(self.__audio_infos)

        self.__last_audio_dir = ''
        self.__last_results_dir = ''

        self.chkEnableSizeCut.clicked.connect(self.__set_size_cut_enable)
        self.chkEnableCut.clicked.connect(self.__set_time_edit_enable)
        self.btnAddFiles.clicked.connect(self.__add_files)
        self.btnApplyThreshold.clicked.connect(self.__apply_threshold_all)
        self.btnStartAnalyse.clicked.connect(self.__start_analyse)
        self.btnLoadRes.clicked.connect(self.__load_result)
        self.btnSaveRes.clicked.connect(self.__save_result)

        self.__resize_rows()

        if os.path.isfile(self.__CFG_PATH):
            try:
                with open(self.__CFG_PATH, 'r', encoding='utf-8') as file:
                    settings = json.load(file)
                self.__import_settings(settings)
            except Exception:
                self.__import_settings(DEFAULT_SETTINGS)
        else:
            self.__import_settings(DEFAULT_SETTINGS)

    def close_event(self, event: QCloseEvent) -> None:
        if not os.path.isdir(os.path.dirname(self.__CFG_PATH)):
            os.makedirs(os.path.dirname(self.__CFG_PATH))
        try:
            with open(self.__CFG_PATH, 'w', encoding='utf-8') as file:
                json.dump(self.__export_settings(), file, indent=2, ensure_ascii=False)
        except Exception:
            QMessageBox.warning(
                self, '경고', '설정 파일 저장 불가'
            )
        return super().close_event(event)

    def __get_results(self):
        while not self.__result_queue.empty():
            result = self.__result_queue.get_nowait()
            self.__results[result.id_] = result
            self.__remain_works -= 1
        for id_ in self.__results:
            self.__done_analyse(id_)

    def __done_analyse(self, id_: int):
        self.__audio_infos.set_result(id_, self.__results[id_])
        self.__apply_threshold(id_)
        self.__resize_rows()
        if self.__remain_works == 0:
            self.__analyse_end()

    def __resize_rows(self):
        for k in range(0, self.__audio_infos.column_count()):
            self.tvResult.resize_column_to_contents(k)

    def __set_size_cut_enable(self):
        self.spSize.enabled = self.chkEnableSizeCut.checked

    def __set_time_edit_enable(self):
        state = self.chkEnableCut.checked
        self.teStart.enabled = state
        self.teDuration.enabled = state

    def __add_files(self):
        files, _ = QFileDialog.get_open_file_names(
            self, "불러올 음악 파일(들) 선택",
            self.__last_audio_dir,
            'Audio Files (*.mp3 *.m4a *.opus *.webm)'
        )
        if not files:
            return
        self.__audio_infos.add_files(files)
        self.__resize_rows()
        self.__last_audio_dir = os.path.dirname(files[0])

    def __load_result(self):
        save_file, _ = QFileDialog.get_open_file_name(
            self, '결과 파일 선택', self.__last_results_dir, 'JSON File (*.json)'
        )
        if not save_file:
            return
        with open(save_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.__audio_infos.import_(data['result'])
        self.__import_settings(data['settings'])
        self.__resize_rows()
        self.__last_results_dir = os.path.dirname(save_file)

    def __save_result(self):
        save_file, _ = QFileDialog.get_save_file_name(
            self, '결과 파일 선택', self.__last_results_dir, 'JSON File (*.json)'
        )
        if not save_file:
            return
        data = self.__audio_infos.export(self.chkIncludeFFTRes.checked)
        settings = self.__export_settings()
        with open(save_file, 'w', encoding='utf-8') as file:
            json.dump(
                {'result': data, 'settings': settings},
                file, indent=2, ensure_ascii=False
            )
        self.__last_results_dir = os.path.dirname(save_file)

    def __apply_threshold_all(self):
        for row in range(self.__audio_infos.row_count()):
            self.__apply_threshold(row)

    def __apply_threshold(self, row: int):
        threshold_h_div_m = self.spHDivM.value
        threshold_n_div_m = self.spNdivM.value
        idial_dB = self.spdB.value
        threshold_dB_diff = self.spdBDiff.value

        analyse_result = self.__audio_infos.analyse_results[row]
        if analyse_result.H_div_M * 100 < threshold_h_div_m:
            self.__audio_infos.item(row, 7)\
                .set_foreground(QColor(255, 0, 0))
        if analyse_result.N_div_M * 100 > threshold_n_div_m:
            self.__audio_infos.item(row, 8)\
                .set_foreground(QColor(255, 0, 0))
        if (
            analyse_result.dBFS > idial_dB + threshold_dB_diff
            or analyse_result.dBFS < idial_dB - threshold_dB_diff
        ):
            self.__audio_infos.item(row, 2)\
                .set_foreground(QColor(255, 0, 0))

    def __start_analyse(self):
        works = Queue()
        id_ = -1
        for id_, work in self.__audio_infos.clear_and_get_file_to_analyse():
            works.put_nowait((id_, work))
        self.__remain_works = id_ + 1
        if not self.__remain_works:
            return

        for widget in self.__DISABLE_ON_START_WORK:
            widget.enabled = False
        self.btnStartAnalyse.text = '분석 중지'
        self.btnStartAnalyse.clicked.disconnect()
        self.btnStartAnalyse.clicked.connect(self.__abort_analyse)

        if self.chkEnableCut.checked:
            start_time_obj = self.teStart.time
            start_time = start_time_obj.minute() * 60 + start_time_obj.second()
            duration_obj = self.teDuration.time
            duration = duration_obj.minute() * 60 + duration_obj.second()
        else:
            start_time = 0
            duration = 0

        pool = QThreadPool.global_instance()
        pool.max_thread_count = self.spThread.value

        config = FFTConfig(
            self.spSize.value * 1024 if self.chkEnableSizeCut.checked else 0,
            start_time, duration,
            self.spFreqLH.value,
            self.spFreqML.value, self.spFreqMH.value,
            self.spFreqHL.value, self.spFreqHH.value,
            self.spFreqNL.value
        )

        workers = []
        for _ in range(self.spThread.value):
            worker = Analyser(works, self.__result_queue, config)
            pool.start(worker)
            workers.append(worker)

        self.__result_check_timer.start()

    def __abort_analyse(self):
        QMessageBox.critical(self, "미구현 기능", "함수 abort_analyse 은(는) 구현되지 않음")

    def __analyse_end(self):
        self.__result_check_timer.stop()
        self.btnStartAnalyse.clicked.disconnect()
        self.btnStartAnalyse.clicked.connect(self.__start_analyse)
        self.btnStartAnalyse.text = '분석 시작'
        for widget in self.__DISABLE_ON_START_WORK:
            widget.enabled = True
        self.__set_size_cut_enable()
        self.__set_time_edit_enable()

    def __import_settings(self, settings: Dict[str, Any]):
        self.chkEnableSizeCut.checked = settings['size_cut_enabled']
        self.__set_size_cut_enable()
        self.spSize.value = settings['cut_size']

        self.chkEnableCut.checked = settings['enable_cut']
        self.__set_time_edit_enable()
        self.teStart.time.from_string(settings['start_time'], 'mm:ss')
        self.teDuration.time.from_string(settings['duration'], 'mm:ss')

        self.spThread.value = settings['thread_cnt']

        self.spHDivM.value = settings['HDivM']
        self.spNdivM.value = settings['NdivM']
        self.spdB.value = settings['dB']
        self.spdBDiff.value = settings['dBDiff']

        self.spFreqLH.value = settings['freq_LH']
        self.spFreqML.value = settings['freq_ML']
        self.spFreqMH.value = settings['freq_MH']
        self.spFreqHL.value = settings['freq_HL']
        self.spFreqHH.value = settings['freq_HH']
        self.spFreqNL.value = settings['freq_NL']

        self.chkIncludeFFTRes.checked = settings['include_fft_raw']

        self.__last_audio_dir = settings['last_audio_dir']
        self.__last_results_dir = settings['last_results_dir']

    def __export_settings(self) -> Dict[str, Any]:
        return {
            'size_cut_enabled': self.chkEnableSizeCut.checked,
            'cut_size': self.spSize.value,

            'enable_cut': self.chkEnableCut.checked,
            'start_time': self.teStart.time.to_string('mm:ss'),
            'duration': self.teDuration.time.to_string('mm:ss'),

            'thread_cnt': self.spThread.value,

            'HDivM': self.spHDivM.value,
            'NdivM': self.spNdivM.value,
            'dB': self.spdB.value,
            'dBDiff': self.spdBDiff.value,

            'freq_LH': self.spFreqLH.value,
            'freq_ML': self.spFreqML.value,
            'freq_MH': self.spFreqMH.value,
            'freq_HL': self.spFreqHL.value,
            'freq_HH': self.spFreqHH.value,
            'freq_NL': self.spFreqNL.value,

            'include_fft_raw': self.chkIncludeFFTRes.checked,

            'last_audio_dir': self.__last_audio_dir,
            'last_results_dir': self.__last_results_dir,
        }


if __name__ == '__main__':
    app = QApplication()

    main = Main()
    main.show()

    app.exec()
