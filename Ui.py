from PySide6.QtCore import QCoreApplication, QMetaObject, QTime, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton, QSizePolicy,
    QSpinBox, QStatusBar, QTimeEdit, QTreeView, QWidget
)


sizePolicy_EF = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
sizePolicy_EF.setHorizontalStretch(0)
sizePolicy_EF.setVerticalStretch(0)

sizePolicy_FF = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
sizePolicy_FF.setHorizontalStretch(0)
sizePolicy_FF.setVerticalStretch(0)

sizePolicy_FP = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
sizePolicy_FP.setHorizontalStretch(0)
sizePolicy_FP.setVerticalStretch(0)


class Ui_Main:
    def setupUi(self, Main):
        if not Main.objectName():
            Main.setObjectName(u"MainWindow")

        Main.resize(840, 600)

        self.centralwidget = QWidget(Main)
        self.centralwidget.setObjectName(u"centralwidget")
        self.glCent = QGridLayout(self.centralwidget)
        self.glCent.setObjectName(u"glCent")

        # top GroupBoxes
        # gbSetGeneral
        self.gbSetGeneral = QGroupBox(self.centralwidget)
        self.gbSetGeneral.setObjectName(u"gbSetGeneral")
        self.glSetGeneral = QGridLayout(self.gbSetGeneral)
        self.glSetGeneral.setObjectName(u"glSetGeneral")

        self.chkEnableSizeCut = QCheckBox(self.gbSetGeneral)
        self.chkEnableSizeCut.setObjectName(u"lbSize")
        self.glSetGeneral.addWidget(self.chkEnableSizeCut, 0, 0, 1, 1)

        self.spSize = QSpinBox(self.gbSetGeneral)
        self.spSize.setObjectName(u"spSize")
        self.spSize.setMaximum(524288)
        self.glSetGeneral.addWidget(self.spSize, 1, 0, 1, 1)

        self.chkEnableCut = QCheckBox(self.gbSetGeneral)
        self.chkEnableCut.setObjectName(u"chkEnableCut")
        self.glSetGeneral.addWidget(self.chkEnableCut, 0, 1, 1, 2)
        self.glCent.addWidget(self.gbSetGeneral, 0, 0, 1, 1)

        self.teStart = QTimeEdit(self.gbSetGeneral)
        self.teStart.setObjectName(u"teStart")
        self.teStart.setMaximumTime(QTime(0, 59, 58))
        self.glSetGeneral.addWidget(self.teStart, 1, 1, 1, 1)

        self.teDuration = QTimeEdit(self.gbSetGeneral)
        self.teDuration.setObjectName(u"teDuration")
        self.teDuration.setMaximumTime(QTime(0, 59, 59))
        self.glSetGeneral.addWidget(self.teDuration, 1, 2, 1, 1)

        self.lbThread = QLabel(self.gbSetGeneral)
        self.lbThread.setObjectName(u"lbThread")
        self.glSetGeneral.addWidget(self.lbThread, 0, 3, 1, 1)

        self.spThread = QSpinBox(self.gbSetGeneral)
        self.spThread.setObjectName(u"spThread")
        self.spThread.setMinimum(1)
        self.glSetGeneral.addWidget(self.spThread, 1, 3, 1, 1)
        # gbSetGeneral

        # gbSetThreshold
        self.gbSetThreshold = QGroupBox(self.centralwidget)
        self.gbSetThreshold.setObjectName(u"gbSetThreshold")
        self.glSetThreashold = QGridLayout(self.gbSetThreshold)
        self.glSetThreashold.setObjectName(u"glSetThreashold")

        self.lbHDivM = QLabel(self.gbSetThreshold)
        self.lbHDivM.setObjectName(u"lbHDivM")
        self.lbHDivM.setAlignment(Qt.AlignCenter)
        self.glSetThreashold.addWidget(self.lbHDivM, 0, 0, 1, 1)

        self.lbNDivM = QLabel(self.gbSetThreshold)
        self.lbNDivM.setObjectName(u"lbNDivM")
        self.lbNDivM.setAlignment(Qt.AlignCenter)
        self.glSetThreashold.addWidget(self.lbNDivM, 0, 1, 1, 1)

        self.lbdB = QLabel(self.gbSetThreshold)
        self.lbdB.setObjectName(u"lbdB")
        self.lbdB.setAlignment(Qt.AlignCenter)
        self.glSetThreashold.addWidget(self.lbdB, 0, 2, 1, 1)

        self.lbdBDiff = QLabel(self.gbSetThreshold)
        self.lbdBDiff.setObjectName(u"lbdBDiff")
        self.glSetThreashold.addWidget(self.lbdBDiff, 0, 3, 1, 1)

        self.spHDivM = QSpinBox(self.gbSetThreshold)
        self.spHDivM.setObjectName(u"spHDivM")
        self.glSetThreashold.addWidget(self.spHDivM, 1, 0, 1, 1)

        self.spNdivM = QSpinBox(self.gbSetThreshold)
        self.spNdivM.setObjectName(u"spNdivM")
        self.glSetThreashold.addWidget(self.spNdivM, 1, 1, 1, 1)

        self.spdB = QDoubleSpinBox(self.gbSetThreshold)
        self.spdB.setObjectName(u"spdB")
        self.spdB.setDecimals(1)
        self.spdB.setRange(-20, 0)
        self.glSetThreashold.addWidget(self.spdB, 1, 2, 1, 1)

        self.spdBDiff = QDoubleSpinBox(self.gbSetThreshold)
        self.spdBDiff.setObjectName(u"spdBDiff")
        self.spdBDiff.setDecimals(1)
        self.glSetThreashold.addWidget(self.spdBDiff, 1, 3, 1, 1)

        self.btnApplyThreshold = QPushButton(self.gbSetThreshold)
        self.btnApplyThreshold.setObjectName(u"btnApplyThreshold")
        self.glSetThreashold.addWidget(self.btnApplyThreshold, 0, 4, 2, 1)

        self.glCent.addWidget(self.gbSetThreshold, 0, 1, 1, 1)
        # gbSetThreshold

        # gbSetFreq
        self.gbSetFreq = QGroupBox(self.centralwidget)
        self.gbSetFreq.setObjectName(u"gbSetFreq")
        self.glSetFreq = QGridLayout(self.gbSetFreq)
        self.glSetFreq.setObjectName(u"glSetFreq")

        # Frequency - Low
        self.lbFreqL = QLabel(self.gbSetFreq)
        self.lbFreqL.setObjectName(u"lbFreqL")
        self.glSetFreq.addWidget(self.lbFreqL, 0, 0, 2, 1)

        self.lbFreqLH = QLabel(self.gbSetFreq)
        self.lbFreqLH.setObjectName(u"lbFreqLH")
        self.glSetFreq.addWidget(self.lbFreqLH, 1, 1, 1, 1)

        self.spFreqLH = QSpinBox(self.gbSetFreq)
        self.spFreqLH.setObjectName(u"spFreqLH")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqLH.sizePolicy().hasHeightForWidth()
        )
        self.spFreqLH.setSizePolicy(sizePolicy_EF)
        self.spFreqLH.setMinimum(1)
        self.spFreqLH.setMaximum(24000)
        self.glSetFreq.addWidget(self.spFreqLH, 1, 2, 1, 1)
        # Frequency - Low

        # Frequency - Mid
        self.lbFreqM = QLabel(self.gbSetFreq)
        self.lbFreqM.setObjectName(u"lbFreqM")
        self.glSetFreq.addWidget(self.lbFreqM, 0, 3, 2, 1)

        self.lbFreqML = QLabel(self.gbSetFreq)
        self.lbFreqML.setObjectName(u"lbFreqML")
        self.glSetFreq.addWidget(self.lbFreqML, 0, 4, 1, 1)

        self.lbFreqMH = QLabel(self.gbSetFreq)
        self.lbFreqMH.setObjectName(u"lbFreqMH")
        self.glSetFreq.addWidget(self.lbFreqMH, 1, 4, 1, 1)

        self.spFreqML = QSpinBox(self.gbSetFreq)
        self.spFreqML.setObjectName(u"spFreqML")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqML.sizePolicy().hasHeightForWidth()
        )
        self.spFreqML.setSizePolicy(sizePolicy_EF)
        self.spFreqML.setMinimum(2)
        self.spFreqML.setMaximum(24000)
        self.glSetFreq.addWidget(self.spFreqML, 0, 5, 1, 1)

        self.spFreqMH = QSpinBox(self.gbSetFreq)
        self.spFreqMH.setObjectName(u"spFreqMH")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqMH.sizePolicy().hasHeightForWidth()
        )
        self.spFreqMH.setSizePolicy(sizePolicy_EF)
        self.spFreqMH.setMinimum(1)
        self.spFreqMH.setMaximum(23998)
        self.glSetFreq.addWidget(self.spFreqMH, 1, 5, 1, 1)
        # Frequency - Mid

        # Frequency - High
        self.lbFreqH = QLabel(self.gbSetFreq)
        self.lbFreqH.setObjectName(u"lbFreqH")
        self.glSetFreq.addWidget(self.lbFreqH, 0, 6, 2, 1)

        self.lbFreqHL = QLabel(self.gbSetFreq)
        self.lbFreqHL.setObjectName(u"lbFreqHL")
        self.glSetFreq.addWidget(self.lbFreqHL, 0, 7, 1, 1)

        self.lbFreqHH = QLabel(self.gbSetFreq)
        self.lbFreqHH.setObjectName(u"lbFreqHH")
        self.glSetFreq.addWidget(self.lbFreqHH, 1, 7, 1, 1)

        self.spFreqHL = QSpinBox(self.gbSetFreq)
        self.spFreqHL.setObjectName(u"spFreqHL")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqHL.sizePolicy().hasHeightForWidth()
        )
        self.spFreqHL.setSizePolicy(sizePolicy_EF)
        self.spFreqHL.setMinimum(3)
        self.spFreqHL.setMaximum(24000)
        self.glSetFreq.addWidget(self.spFreqHL, 0, 8, 1, 1)

        self.spFreqHH = QSpinBox(self.gbSetFreq)
        self.spFreqHH.setObjectName(u"spFreqHH")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqHH.sizePolicy().hasHeightForWidth()
        )
        self.spFreqHH.setSizePolicy(sizePolicy_EF)
        self.spFreqHH.setMinimum(1)
        self.spFreqHH.setMaximum(23999)
        self.glSetFreq.addWidget(self.spFreqHH, 1, 8, 1, 1)
        # Frequency - High

        # Frequency - Noise (Very High)
        self.lbFreqN = QLabel(self.gbSetFreq)
        self.lbFreqN.setObjectName(u"lbFreqN")
        self.glSetFreq.addWidget(self.lbFreqN, 0, 9, 2, 1)

        self.lbFreqNL = QLabel(self.gbSetFreq)
        self.lbFreqNL.setObjectName(u"lbFreqNL")
        self.glSetFreq.addWidget(self.lbFreqNL, 0, 10, 1, 1)

        self.spFreqNL = QSpinBox(self.gbSetFreq)
        self.spFreqNL.setObjectName(u"spFreqNL")
        sizePolicy_EF.setHeightForWidth(
            self.spFreqNL.sizePolicy().hasHeightForWidth()
        )
        self.spFreqNL.setSizePolicy(sizePolicy_EF)
        self.spFreqNL.setMinimum(1)
        self.spFreqNL.setMaximum(24000)
        self.glSetFreq.addWidget(self.spFreqNL, 0, 11, 1, 1)
        # Frequency - Noise

        self.glCent.addWidget(self.gbSetFreq, 1, 0, 3, 2)
        # gbSetFreq
        # top GroupBoxes

        # gbResult
        self.gbResult = QGroupBox(self.centralwidget)
        self.gbResult.setObjectName(u"gbResult")
        sizePolicy_FP.setHeightForWidth(
            self.gbResult.sizePolicy().hasHeightForWidth()
        )
        self.gbResult.setSizePolicy(sizePolicy_FP)
        self.glResult = QGridLayout(self.gbResult)
        self.glResult.setObjectName(u"glResult")

        self.btnLoadRes = QPushButton(self.gbResult)
        self.btnLoadRes.setObjectName(u"chkLoadRes")
        self.glResult.addWidget(self.btnLoadRes, 0, 0, 1, 1)

        self.chkIncludeFFTRes = QCheckBox(self.gbResult)
        self.chkIncludeFFTRes.setObjectName(u"chkIncludeFFTRes")
        self.glResult.addWidget(self.chkIncludeFFTRes, 1, 0, 1, 1)

        self.btnSaveRes = QPushButton(self.gbResult)
        self.btnSaveRes.setObjectName(u"chkSaveRes")
        self.glResult.addWidget(self.btnSaveRes, 2, 0, 1, 1)

        self.glCent.addWidget(self.gbResult, 0, 2, 2, 1)
        # gbResult

        # btnAddFiles
        self.btnAddFiles = QPushButton(self.centralwidget)
        self.btnAddFiles.setObjectName(u"btnAddFiles")
        sizePolicy_FF.setHeightForWidth(
            self.btnAddFiles.sizePolicy().hasHeightForWidth()
        )
        self.btnAddFiles.setSizePolicy(sizePolicy_FF)
        self.glCent.addWidget(self.btnAddFiles, 2, 2, 1, 1, Qt.AlignHCenter)
        # btnAddFiles

        # btnStartAnalyse
        self.btnStartAnalyse = QPushButton(self.centralwidget)
        self.btnStartAnalyse.setObjectName(u"btnStartAnalyse")
        sizePolicy_FF.setHeightForWidth(
            self.btnStartAnalyse.sizePolicy().hasHeightForWidth()
        )
        self.btnStartAnalyse.setSizePolicy(sizePolicy_FF)
        self.glCent.addWidget(self.btnStartAnalyse, 3, 2, 1, 1, Qt.AlignHCenter)
        # btnStartAnalyse

        # tvResult
        self.tvResult = QTreeView(self.centralwidget)
        self.tvResult.setObjectName(u"tvResult")
        self.glCent.addWidget(self.tvResult, 4, 0, 1, 3)
        # tvResult

        Main.setCentralWidget(self.centralwidget)

        # Status Bar
        self.statusbar = QStatusBar(Main)
        self.statusbar.setObjectName(u"statusbar")
        Main.setStatusBar(self.statusbar)
        # Status Bar

        self.retranslateUi(Main)

        QMetaObject.connectSlotsByName(Main)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("\uc74c\uc6d0 \uc74c\uc9c8 \ubd84\uc11d")

        self.gbSetGeneral.setTitle(
            "FFT / dBFS \ubd84\uc11d \uc124\uc815"
        )
        self.chkEnableSizeCut.setText(
            "\uc77d\uc744 \ud06c\uae30 \uc9c0\uc815"
        )
        self.lbThread.setText("\uc4f0\ub808\ub4dc \uc218")
        self.spSize.setSpecialValueText("Auto")
        self.spSize.setSuffix(" KiB")
        self.teStart.setDisplayFormat("mm:ss")
        self.teDuration.setDisplayFormat("mm:ss")
        self.spThread.setSuffix(" \uac1c")
        self.chkEnableCut.setText(
            "\uc2dc\uc791 \uc2dc\uac04/\ubd84\uc11d \uae38\uc774 \uc9c0\uc815"
        )

        self.gbSetThreshold.setTitle("\uc784\uacc4\uac12")
        self.lbHDivM.setText("\uace0/\uc911")
        self.lbNDivM.setText("\ub178\uc774\uc988/\uc911")
        self.lbdB.setText("dBFS \ubaa9\ud45c")
        self.lbdBDiff.setText("dBFS \uc624\ucc28")
        self.spHDivM.setSuffix(" %")
        self.spNdivM.setSuffix(" %")
        self.spdB.setSuffix(" dBFS")
        self.spdBDiff.setPrefix("\u00b1")
        self.spdBDiff.setSuffix(" dBFS")
        self.btnApplyThreshold.setText("\uc801\uc6a9")

        # gbSetFreq
        self.gbSetFreq.setTitle(
            "\uc8fc\ud30c\uc218 \ub300\uc5ed \uc124\uc815"
        )

        self.lbFreqL.setText("\uc800")
        self.lbFreqLH.setText("H")
        self.spFreqLH.setSuffix(" Hz")

        self.lbFreqM.setText("\uc911")
        self.lbFreqML.setText("L")
        self.lbFreqMH.setText("H")
        self.spFreqML.setSuffix(" Hz")
        self.spFreqMH.setSuffix(" Hz")

        self.lbFreqH.setText("\uace0")
        self.lbFreqHL.setText("L")
        self.lbFreqHH.setText("H")
        self.spFreqHL.setSuffix(" Hz")
        self.spFreqHH.setSuffix(" Hz")

        self.lbFreqN.setText("\ub178\uc774\uc988")
        self.lbFreqNL.setText("L")
        self.spFreqNL.setSuffix(" Hz")

        self.gbResult.setTitle("\uacb0\uacfc \uc800\uc7a5/\ubd88\ub7ec\uc624\uae30")
        self.btnLoadRes.setText("\uacb0\uacfc \ubd88\ub7ec\uc624\uae30")
        self.chkIncludeFFTRes.setText("FFT \uacb0\uacfc \ud3ec\ud568")
        self.btnSaveRes.setText("\uacb0\uacfc \uc800\uc7a5")
        # gbSetFreq

        self.btnAddFiles.setText("\ud30c\uc77c \ucd94\uac00")
        self.btnStartAnalyse.setText("\ubd84\uc11d \uc2dc\uc791")
    # retranslateUi

