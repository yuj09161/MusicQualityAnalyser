from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from decimal import Decimal
from functools import partialmethod
from io import BytesIO
from itertools import chain, repeat
from math import log10
from multiprocessing import cpu_count  # , Queue
from queue import Queue
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Union
import os
import time

from PySide6.QtCore import QRunnable, QThreadPool, Signal, QTimer, SignalInstance
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QWidget
from __feature__ import snake_case, true_property

from pydub import AudioSegment
import numpy as np

from Ui import Ui_Main


class Result(NamedTuple):
    id_: int
    dbFS: float
    max_freq: int
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
    works: Queue,
    results: Queue,
    read_size_in_kib: int,
    start_time: int,
    duration: int,
    freq_LH: int,
    freq_ML: int,
    freq_MH: int,
    freq_HL: int,
    freq_HH: int,
    freq_NH: int,
):
    print('do_fft', works, results, works.empty())
    while not works.empty():
        id_, file_to_open = works.get_nowait()

        if read_size_in_kib:
            with open(file_to_open, 'rb') as file:
                data = file.read(read_size_in_kib)
            file_to_open = BytesIO(data)
        audio = AudioSegment.from_file(file_to_open)

        fft_result = fft_audiosegment(audio, start_time, duration)

        results.put_nowait(Result(
            id_,
            audio.dBFS,
            len(fft_result),
            Decimal(str(np.mean(fft_result[:freq_LH]))),
            Decimal(str(np.mean(fft_result[freq_ML:freq_MH]))),
            Decimal(str(np.mean(fft_result[freq_HL:freq_HH]))),
            Decimal(str(np.mean(fft_result[freq_NH:]))),
            fft_result
        ))


class Analyser(QRunnable):
    def __init__(
        self,
        work_queue: Queue,
        result_queue: Queue,
        done_signal: SignalInstance,
        read_size_in_kib: int,
        start_time: int,
        duration: int,
        freq_LH: int,
        freq_ML: int,
        freq_MH: int,
        freq_HL: int,
        freq_HH: int,
        freq_NH: int,
    ):
        super().__init__()

        self.__works = work_queue
        self.__results = result_queue
        self.__work_done_signal = done_signal

        self.__read_size_in_kib = read_size_in_kib
        self.__start_time = start_time
        self.__duration = duration
        self.__freq_LH = freq_LH
        self.__freq_ML = freq_ML
        self.__freq_MH = freq_MH
        self.__freq_HL = freq_HL
        self.__freq_HH = freq_HH
        self.__freq_NH = freq_NH

    def run(self):
        while not self.__works.empty():
            id_, file_to_open = self.__works.get_nowait()

            if self.__read_size_in_kib:
                with open(file_to_open, 'rb') as file:
                    data = file.read(self.__read_size_in_kib)
                file_to_open = BytesIO(data)
            audio = AudioSegment.from_file(file_to_open)

            fft_result = fft_audiosegment(audio, self.__start_time, self.__duration)

            self.__works.task_done()
            self.__results.put_nowait(Result(
                id_,
                audio.dBFS,
                len(fft_result),
                Decimal(str(np.mean(fft_result[:self.__freq_LH]))),
                Decimal(str(np.mean(fft_result[self.__freq_ML:self.__freq_MH]))),
                Decimal(str(np.mean(fft_result[self.__freq_HL:self.__freq_HH]))),
                Decimal(str(np.mean(fft_result[self.__freq_NH:]))),
                fft_result
            ))
            self.__work_done_signal.emit(id_)


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

        self.__files = []
        self.__raw_fft_results: Dict[int, Union[np.ndarray, str]] = {}

    @property
    def files(self) -> List[str]:
        return self.__files.copy()

    def add_file(self, file_path: str):
        def make_items(texts):
            items = []
            for d in texts:
                item = QStandardItem(d)
                item.set_editable(False)
                items.append(item)
            return items

        self.__files.append(file_path)
        self.append_row(make_items(chain(
            (file_path,), repeat('-', len(self.HEADER_TEXTS) - 1)
        )))

    def add_files(self, file_paths: Iterable[str]):
        for path in file_paths:
            self.add_file(path)

    def set_result(self, index: int, result: Result):
        intensity_L = log10(result.intensity_L)
        intensity_M = log10(result.intensity_M)
        intensity_H = log10(result.intensity_H)
        intensity_N = log10(result.intensity_N)
        self.item(index, 1).set_text('완료')
        self.item(index, 2).set_text(str(f'{result.dbFS:.3f}'))
        self.item(index, 3).set_text(f'{intensity_L:.2f}')
        self.item(index, 4).set_text(f'{intensity_M:.2f}')
        self.item(index, 5).set_text(f'{intensity_H:.2f}')
        self.item(index, 6).set_text(f'{intensity_N:.2f}')
        self.item(index, 7).set_text(f'{intensity_H / intensity_M * 100:.2f}%')
        self.item(index, 8).set_text(f'{intensity_N / intensity_M * 100:.2f}%')
        self.__raw_fft_results[index] = result.raw_result

    def load(self, file_name: str):
        raise NotImplementedError

    def save(self, file_name: str):
        raise NotImplementedError


class Main(QMainWindow, Ui_Main):
    analyse_done = Signal(int)

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
        self.__result_queue: Queue[Result] = Queue()

        self.__result_check_timer = QTimer(self)
        self.__result_check_timer.interval = 250
        self.__result_check_timer.single_shot_ = False
        self.__result_check_timer.timeout.connect(self.__get_results)

        self.__audio_infos = AudioInfo(self.tvResult)
        self.tvResult.set_model(self.__audio_infos)

        self.chkEnableSizeCut.clicked.connect(self.__set_size_cut_enable)
        self.chkEnableCut.clicked.connect(self.__set_time_edit_enable)
        self.btnAddFiles.clicked.connect(self.__add_files)
        self.btnApplyThreshold.clicked.connect(self.__apply_threshold_all)
        self.btnStartAnalyse.clicked.connect(self.__start_analyse)
        self.btnLoadRes.clicked.connect(self.__load_result)
        self.btnSaveRes.clicked.connect(self.__save_result)

        # self.analyse_done.connect(self.__done_analyse)

        self.__set_default_values()
        self.__resize_rows()

    def __set_default_values(self):
        self.chkEnableSizeCut.checked = True
        self.chkEnableCut.checked = True

        self.spSize.value = 1536
        self.spThread.value = max((cpu_count() - 2, 1))

        self.spFreqLH.value = 5000
        self.spFreqML.value = 10000
        self.spFreqMH.value = 14500
        self.spFreqHL.value = 16500
        self.spFreqHH.value = 19500
        self.spFreqNL.value = 20500

        self.spHDivM.value = 80
        self.spNdivM.value = 30
        self.spdB.value = -12.0
        self.spdBDiff.value = 2

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
            os.path.expanduser('~')
        )
        self.__audio_infos.add_files(files)
        self.__resize_rows()

    def __apply_threshold_all(self):
        for row in range(self.__audio_infos.row_count()):
            self.__apply_threshold(row)

    def __apply_threshold(self, row: int):
        threshold_h_div_m = self.spHDivM.value
        threshold_n_div_m = self.spNdivM.value
        idial_dB = -Decimal(self.spdB.text[1:-5])
        threshold_dB_diff = Decimal(self.spdBDiff.text[1:-5])

        item_h_div_m = self.__audio_infos.item(row, 7)
        item_n_div_m = self.__audio_infos.item(row, 8)
        item_dBFS = self.__audio_infos.item(row, 2)

        h_div_m = Decimal(item_h_div_m.text().replace('%', ''))
        n_div_m = Decimal(item_n_div_m.text().replace('%', ''))
        dBFS = Decimal(item_dBFS.text())

        if h_div_m < threshold_h_div_m:
            item_h_div_m.set_foreground(QColor(255, 0, 0))
        if n_div_m > threshold_n_div_m:
            item_n_div_m.set_foreground(QColor(255, 0, 0))
        if dBFS > idial_dB + threshold_dB_diff or dBFS < idial_dB - threshold_dB_diff:
            item_dBFS.set_foreground(QColor(255, 0, 0))

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

    def __start_analyse(self):
        for widget in self.__DISABLE_ON_START_WORK:
            widget.enabled = False
        self.btnStartAnalyse.text = '분석 중지'
        self.btnStartAnalyse.clicked.disconnect()
        self.btnStartAnalyse.clicked.connect(self.__abort_analyse)

        works = Queue()
        id_ = -1
        for id_, work in enumerate(self.__audio_infos.files):
            works.put_nowait((id_, work))
        self.__remain_works = id_ + 1

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

        workers = []
        for _ in range(self.spThread.value):
            worker = Analyser(
                works, self.__result_queue, self.analyse_done,
                self.spSize.value if self.chkEnableSizeCut.checked else 0,
                start_time, duration,
                self.spFreqLH.value,
                self.spFreqML.value, self.spFreqMH.value,
                self.spFreqHL.value, self.spFreqHH.value,
                self.spFreqNL.value,
            )
            pool.start(worker)
            workers.append(worker)
            '''
            self.__future = pool.submit(
                do_fft,
                works, self.__result_queue,
                self.spSize.value if self.chkEnableSizeCut.checked else 0,
                start_time, duration,
                self.spFreqLH.value,
                self.spFreqML.value, self.spFreqMH.value,
                self.spFreqHL.value, self.spFreqHH.value,
                self.spFreqNL.value
            )
            '''

        self.__result_check_timer.start()

    def __analyse_end(self):
        self.__result_check_timer.stop()
        self.btnStartAnalyse.clicked.disconnect()
        self.btnStartAnalyse.clicked.connect(self.__start_analyse)
        self.btnStartAnalyse.text = '분석 시작'
        for widget in self.__DISABLE_ON_START_WORK:
            widget.enabled = True
        self.__set_size_cut_enable()
        self.__set_time_edit_enable()

    def __abort_analyse(self):
        QMessageBox.critical(self, "미구현 기능", "함수 abort_analyse 은(는) 구현되지 않음")

    def __load_result(self):
        QMessageBox.critical(self, "미구현 기능", "함수 load_result 은(는) 구현되지 않음")

    def __save_result(self):
        QMessageBox.critical(self, "미구현 기능", "함수 save_result 은(는) 구현되지 않음")


def test():
    for _ in range(10):
        print('running...')
        time.sleep(1)


if __name__ == '__main__':
    app = QApplication()

    main = Main()
    main.show()

    app.exec()
