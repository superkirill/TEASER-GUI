import subprocess
import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, QRunnable, pyqtSlot, QThreadPool, pyqtSignal
from PyQt5 import QtCore


class WorkerSignals(QtCore.QObject):
    updated_console = pyqtSignal(str)


class RunExecutable(QRunnable):
    def __init__(self, args):
        super(RunExecutable, self).__init__()
        self.signals = WorkerSignals()
        self.args = args

    @pyqtSlot(str)
    def run(self):
        process = subprocess.Popen(
            self.args,
            stdout=subprocess.PIPE
        )
        for line in iter(process.stdout.readline, b''):
            self.signals.updated_console.emit(line.decode("utf-8"))


class SmoothNetWindow(QtWidgets.QMainWindow):
    """A window of 3DSmoothNet settings for the point correspondences search"""
    do_work = pyqtSignal(str)

    def __init__(self, parent=None):
        super(SmoothNetWindow, self).__init__(parent)
        uic.loadUi('design_3dsmoothnet.ui', self)
        self.threadpool = QThreadPool()
        # Connect signals and slots
        self.qt_sample_keypoints_button.clicked.connect(self.__sample_keypoints)
        self.qt_compute_descriptors_button.clicked.connect(self.__compute_and_match)
        # Define constants
        self.SRC_KEYPOINTS_FNAME = "temp_src_keypoints.txt"
        self.TGT_KEYPOINTS_FNAME = "temp_tgt_keypoints.txt"
        self.SDV_DIR = "sdv"
        self.LOG_FILE = "3dsmoothnet_logs.txt"

    @pyqtSlot(str)
    def __update_console(self, text):
        self.qt_console.setPlainText(self.qt_console.toPlainText() + text)

    def __sample_keypoints(self):
        """
            Uniformly sample N keypoints from the source and target point clouds
            :return: None
        """
        num_keypoints = int(self.qt_number_of_keypoints_edit.text())
        # Sample keypoints for the source point cloud

        if num_keypoints > self.parent().source_cloud_size:
            QtWidgets.QMessageBox.warning(self, "Keypoints sampling",
                                          f"Asked to sample {num_keypoints} keypoints when the" +
                                          f" cloud contains only {self.parent().source_cloud_size}." +
                                          f" Sampling {self.parent().source_cloud_size} keypoints instead."
                                          )
            src_keypoints = np.random.randint(0, self.parent().source_cloud_size, size=self.parent().source_cloud_size)
        else:
            src_keypoints = np.random.randint(0, self.parent().source_cloud_size, size=num_keypoints)
        self.qt_src_keypoints_edit.setText("".join([f"{index}\n" for index in src_keypoints]))

        # Sample keypoints for the target point cloud
        if num_keypoints > self.parent().target_cloud_size:
            QtWidgets.QMessageBox.warning(self, "Keypoints sampling",
                                          f"Asked to sample {num_keypoints} keypoints when the" +
                                          f" cloud contains only {self.parent().target_cloud_size}." +
                                          f" Sampling {self.parent().target_cloud_size} keypoints instead."
                                          )
            tgt_keypoints = np.random.randint(0, self.parent().target_cloud_size, size=self.parent().target_cloud_size)
        else:
            tgt_keypoints = np.random.randint(0, self.parent().target_cloud_size, size=num_keypoints)
        self.qt_tgt_keypoints_edit.setText("".join([f"{index}\n" for index in tgt_keypoints]))

    def __save_keypoints(self):
        """
            Save lists of keypoints form the textEdit widgets to files.
            :return: None
        """
        src_keypoints = self.qt_src_keypoints_edit.toPlainText()
        tgt_keypoints = self.qt_src_keypoints_edit.toPlainText()
        with open(self.SRC_KEYPOINTS_FNAME, "w") as out:
            for index in src_keypoints.split("\n")[:-1]:
                for char in index:
                    if char not in list("0123456789"):
                        QtWidgets.QMessageBox.warning(self, "Wrong keypoint index",
                                                      f"Wrong keypoint index in the list of the source cloud" +
                                                      f" keypoints: {index}. Fix the issue and try again."
                                                      )
                        return
                out.write(index + "\n")
        with open(self.TGT_KEYPOINTS_FNAME, "w") as out:
            for index in tgt_keypoints.split("\n")[:-1]:
                for char in index:
                    if char not in list("0123456789"):
                        QtWidgets.QMessageBox.warning(self, "Wrong keypoint index",
                                                      f"Wrong keypoint index in the list of the target cloud" +
                                                      f" keypoints: {index}. Fix the issue and try again."
                                                      )
                        return
                out.write(index + "\n")

    def __compute_and_match(self):
        """
            Compute the 3DSmoothNet descriptors and match them to calculate correspondences
            :return: None
        """
        self.__save_keypoints()
        cloud_a = self.parent().source_cloud
        cloud_b = self.parent().target_cloud

        # Compute input parametrization
        args = f"3DSmoothNet.exe -f {cloud_a} -k {os.path.join(os.getcwd(), self.SRC_KEYPOINTS_FNAME)} -o {self.SDV_DIR}"
        worker1 = RunExecutable(args)
        worker1.signals.updated_console.connect(self.__update_console)
        self.threadpool.start(worker1)

        args = f"3DSmoothNet.exe -f {cloud_b} -k {os.path.join(os.getcwd(), self.TGT_KEYPOINTS_FNAME)} -o {self.SDV_DIR}"
        worker2 = RunExecutable(args)
        worker2.signals.updated_console.connect(self.__update_console)
        self.threadpool.start(worker2)



    def show_window(self):
        """
            Display the SmoothNet window
            :return: None
        """
        self.show()
