import subprocess
import os
import numpy as np
import open3d as o3d
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import QRunnable, pyqtSlot, QThreadPool, pyqtSignal
from PyQt5 import QtCore
from sklearn.neighbors import KDTree

PATH_TO_CORRESPONDENCES = "smooth_net_correspondences.txt"


class WorkerSignals(QtCore.QObject):
    updated_console = pyqtSignal(str)
    parametrization_finished = pyqtSignal()
    inference_finished = pyqtSignal()
    update_button = pyqtSignal()


class RunMatching(QRunnable):
    def __init__(self, args):
        super(RunMatching, self).__init__()
        self.signals = WorkerSignals()
        self.args = args

    @pyqtSlot()
    def run(self):
        print("Running matching")
        # load descriptors
        desc_dir = self.args["desc_dir"]
        src_cloud = self.args["src_cloud"]
        tgt_cloud = self.args["tgt_cloud"]
        src_desc = self.args["src_desc"]
        tgt_desc = self.args["tgt_desc"]
        src_kpoints = self.args["src_kpoints"]
        tgt_kpoints = self.args["tgt_kpoints"]
        frag1_desc_file = os.path.join(
            desc_dir, '32_dim', src_desc
        )
        frag1_desc = np.load(frag1_desc_file)
        frag1_desc = frag1_desc["data"]

        frag2_desc_file = os.path.join(
            desc_dir, '32_dim', tgt_desc
        )
        frag2_desc = np.load(frag2_desc_file)
        frag2_desc = frag2_desc["data"]

        # save as o3d feature
        frag1 = o3d.registration.Feature()
        frag1.data = frag1_desc.T

        frag2 = o3d.registration.Feature()
        frag2.data = frag2_desc.T

        # load point clouds
        frag1_pc = o3d.io.read_point_cloud(
            os.path.join(src_cloud)
        )
        frag2_pc = o3d.io.read_point_cloud(
            os.path.join(tgt_cloud)
        )

        # load keypoints
        frag1_indices = np.genfromtxt(
            os.path.join(src_kpoints)
        )
        frag2_indices = np.genfromtxt(
            os.path.join(tgt_kpoints)
        )

        frag1_pc_keypoints = np.asarray(frag1_pc.points)[frag1_indices.astype(int), :]
        frag2_pc_keypoints = np.asarray(frag2_pc.points)[frag2_indices.astype(int), :]

        # Save as open3d point clouds
        frag1_key = o3d.geometry.PointCloud()
        frag1_key.points = o3d.utility.Vector3dVector(frag1_pc_keypoints)

        frag2_key = o3d.geometry.PointCloud()
        frag2_key.points = o3d.utility.Vector3dVector(frag2_pc_keypoints)

        ref_matched_key, test_matched_key, idx = find_mutually_nn_keypoints(
            frag2_key, frag1_key, frag2, frag1
        )
        correspondences = np.vstack((frag1_indices[idx], frag2_indices[idx]))
        with open(PATH_TO_CORRESPONDENCES, 'w') as corr_file:
            for i in range(correspondences.shape[1]):
                x, y = correspondences[:, i]
                corr_file.write(f"{int(x)} {int(y)}\n")
        self.signals.update_button.emit()


class RunExecutable(QRunnable):
    def __init__(self, args):
        super(RunExecutable, self).__init__()
        self.signals = WorkerSignals()
        self.args = args

    @pyqtSlot()
    def run(self):
        process = subprocess.Popen(
            self.args,
            stdout=subprocess.PIPE
        )
        for line in iter(process.stdout.readline, b''):
            self.signals.updated_console.emit(line.decode("utf-8"))
        self.signals.parametrization_finished.emit()


class RunCNN(QRunnable):
    def __init__(self, args):
        super(RunCNN, self).__init__()
        self.signals = WorkerSignals()
        self.args = args

    @pyqtSlot(str)
    def run(self):
        print("Running CNN")
        process = subprocess.Popen(
            self.args,
            stdout=subprocess.PIPE
        )
        for line in iter(process.stdout.readline, b''):
            self.signals.updated_console.emit(line.decode("utf-8"))
        self.signals.updated_console.emit('\nInference completed\n')
        self.signals.inference_finished.emit()


def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
        Use kdtree to find mutually closest keypoints

        ref_key: reference keypoints (source)
        test_key: test keypoints (target)
        ref: reference feature (source feature)
        test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints), ref_match_idx


class SmoothNetWindow(QtWidgets.QMainWindow):
    """A window of 3DSmoothNet settings for the point correspondences search"""
    def __init__(self, parent=None):
        super(SmoothNetWindow, self).__init__(parent)
        uic.loadUi('design_3dsmoothnet.ui', self)
        self.threadpool = QThreadPool()
        self.parametrized_clouds = 0
        self.paths_to_parametrization = []
        # Connect signals and slots
        self.qt_sample_keypoints_button.clicked.connect(self.__sample_keypoints)
        self.qt_compute_descriptors_button.clicked.connect(self.__compute_and_match)
        # Define constants
        self.SRC_KEYPOINTS_FNAME = "temp_src_keypoints.txt"
        self.TGT_KEYPOINTS_FNAME = "temp_tgt_keypoints.txt"
        self.SDV_DIR = "sdv/"
        self.DESCRIPTORS_DIR = "smooth_net_descriptors/"
        self.LOG_FILE = "3dsmoothnet_logs.txt"

    @pyqtSlot(str)
    def __update_console(self, text):
        self.qt_console.setPlainText(self.qt_console.toPlainText() + text)
        self.qt_console.moveCursor(QtGui.QTextCursor.End)
        if text.split() != [] and text.split()[0] == "save_path_for_parametrization":
            self.paths_to_parametrization.append(text.split()[1])

    def __update_button(self):
        self.qt_compute_descriptors_button.setEnabled(True)

    @pyqtSlot()
    def __match_descriptors(self):
        args = {"desc_dir": self.DESCRIPTORS_DIR,
                "src_cloud": self.parent().source_cloud,
                "tgt_cloud": self.parent().target_cloud,
                "src_kpoints": self.SRC_KEYPOINTS_FNAME,
                "tgt_kpoints": self.TGT_KEYPOINTS_FNAME,
                "src_desc" : self.paths_to_parametrization[0].split('/')[-1][:-4] + ".npz",
                "tgt_desc" : self.paths_to_parametrization[1].split('/')[-1][:-4] + ".npz"
                }
        worker = RunMatching(args)
        worker.signals.update_button.connect(self.__update_button)
        self.threadpool.start(worker)

    @pyqtSlot()
    def __run_inference(self):
        """
            Slot that receives a signal to start inference after the parametrization is done
            :return: None
        """
        self.parametrized_clouds += 1
        print(f"Parametrization computed for {self.parametrized_clouds} cloud(s)")
        if self.parametrized_clouds == 2:
            self.__print_to_console('Starting inference\n')
            args = "python main_cnn.py --run_mode=test" + \
                f" --parametrization1={self.paths_to_parametrization[0]}" + \
                f" --parametrization2={self.paths_to_parametrization[1]}" + \
                f" --evaluate_output_folder={self.DESCRIPTORS_DIR}"

            print(args)
            worker = RunCNN(args)
            worker.signals.updated_console.connect(self.__update_console)
            worker.signals.inference_finished.connect(self.__match_descriptors)
            self.threadpool.start(worker)

    def __sample_keypoints(self):
        """
            Uniformly sample N keypoints from the source and target point clouds
            :return: None
        """
        num_keypoints = int(self.qt_number_of_keypoints_edit.text())
        # Sample keypoints for the source point cloud
        min_size = min(self.parent().source_cloud_size, self.parent().target_cloud_size)
        if num_keypoints > min_size:
            QtWidgets.QMessageBox.warning(self, "Keypoints sampling",
                                          f"Asked to sample {num_keypoints} keypoints when the" +
                                          f" cloud contains only {self.parent().source_cloud_size}." +
                                          f" Sampling {self.parent().source_cloud_size} keypoints for both clouds."
                                          )
            src_keypoints = np.random.randint(0, self.parent().source_cloud_size, size=min_size)
        else:
            src_keypoints = np.random.randint(0, self.parent().source_cloud_size, size=num_keypoints)
        self.qt_src_keypoints_edit.setText("".join([f"{index}\n" for index in src_keypoints]))

        # Sample keypoints for the target point cloud
        if num_keypoints > min_size:
            QtWidgets.QMessageBox.warning(self, "Keypoints sampling",
                                          f"Asked to sample {num_keypoints} keypoints when the" +
                                          f" cloud contains only {self.parent().target_cloud_size}." +
                                          f" Sampling {self.parent().target_cloud_size} keypoints for both clouds."
                                          )
            tgt_keypoints = np.random.randint(0, self.parent().target_cloud_size, size=min_size)
        else:
            tgt_keypoints = np.random.randint(0, self.parent().target_cloud_size, size=num_keypoints)
        self.qt_tgt_keypoints_edit.setText("".join([f"{index}\n" for index in tgt_keypoints]))

    def __save_keypoints(self):
        """
            Save lists of keypoints form the textEdit widgets to files.
            :return: None
        """
        src_keypoints = self.qt_src_keypoints_edit.toPlainText()
        tgt_keypoints = self.qt_tgt_keypoints_edit.toPlainText()
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

    def __print_to_console(self, text):
        """
            Print a message in the window's console
            :param text: str - a message to print
            :return: None
        """
        self.qt_console.setPlainText(self.qt_console.toPlainText() + text + "\n")

    def __compute_and_match(self):
        """
            Compute the 3DSmoothNet descriptors and match them to calculate correspondences
            :return: None
        """
        self.qt_compute_descriptors_button.setEnabled(False)
        # Check that the number of keypoints is the same for both clouds
        if len(self.qt_src_keypoints_edit.toPlainText().split("\n")) != \
                len(self.qt_tgt_keypoints_edit.toPlainText().split("\n")):
            QtWidgets.QMessageBox.warning(self,
                                             'Error',
                                             "The numbers of keypoints for two clouds are different."
                                          )
            self.qt_compute_descriptors_button.setEnabled(True)
            return
        self.__save_keypoints()
        cloud_a = self.parent().source_cloud
        cloud_b = self.parent().target_cloud
        r = float(self.qt_voxel_size_edit.text()) / 2
        n = int(self.qt_number_of_voxels_edit.text())
        h = float(self.qt_gaussian_width_edit.text())

        # Compute input parametrization
        self.__print_to_console("Starting input parametrization\n")
        self.parametrized_clouds = 0

        args = f"3DSmoothNet.exe -f {cloud_a} -r {r} -n {n} -h {h}"
        args += f" -k {os.path.join(os.getcwd(), self.SRC_KEYPOINTS_FNAME)} -o {self.SDV_DIR}"
        worker1 = RunExecutable(args)
        worker1.signals.updated_console.connect(self.__update_console)
        worker1.signals.parametrization_finished.connect(self.__run_inference)
        self.threadpool.start(worker1)

        args = f"3DSmoothNet.exe -f {cloud_b} -r {r} -n {n} -h {h}"
        args += f" -k {os.path.join(os.getcwd(), self.TGT_KEYPOINTS_FNAME)} -o {self.SDV_DIR}"
        worker2 = RunExecutable(args)
        worker2.signals.updated_console.connect(self.__update_console)
        worker2.signals.parametrization_finished.connect(self.__run_inference)
        self.threadpool.start(worker2)

    def show_window(self):
        """
            Display the SmoothNet window
            :return: None
        """
        self.show()
