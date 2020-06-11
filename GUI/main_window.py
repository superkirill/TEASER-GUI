import os
import open3d as o3d
import point_cloud_window, smooth_net_window
import pickle as pkl
import subprocess
import copy
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow):
    """Main window of the GUI for TEASER"""

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('design_main_window.ui', self)
        # Window data
        self.source_cloud = ""
        self.target_cloud = ""
        self.source_cloud_size = 0
        self.target_cloud_size = 0
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.scale = 1
        # Load previous context
        self.__load_context()
        # Create view windows
        self.source_point_cloud_view = point_cloud_window.PointCloudWindow()
        self.target_point_cloud_view = point_cloud_window.PointCloudWindow()
        self.alignment_point_cloud_view = point_cloud_window.PointCloudWindow()
        # Initialize buttons
        self.qt_show_source_button.clicked.connect(self.__show_source)
        self.qt_show_target_button.clicked.connect(self.__show_target)
        self.qt_show_alignment_button.clicked.connect(self.__show_alignment)
        self.qt_run_teaser_button.clicked.connect(self.__run_teaser)
        self.qt_load_source_cloud.triggered.connect(self.__load_source_cloud)
        self.qt_load_target_cloud.triggered.connect(self.__load_target_cloud)
        self.qt_compute_with_3DSmoothNet.triggered.connect(self.__show_smoothnet)
        # Update enabled/disabled status of the buttons
        self.__update_clickability()

    def __show_smoothnet(self):
        """
            Show the window with 3DSmoothNet descriptors and correspondences search parameters
            :return: None
        """
        self.smooth_net_window = smooth_net_window.SmoothNetWindow(self)
        self.smooth_net_window.show_window()

    def __update_transformation_view(self):
        """
            Output the computed transformation in the qt_transformation_textbox
            :return: None
        """
        text = "\tComputed transformation\n\n"
        text += "Rotation:\n"
        for i in range(3):
            text += str(self.rotation[i, 0]) + " " + str(self.rotation[i, 1]) + " " + str(self.rotation[i, 2]) + "\n"
        text += "\nTranslation:\n"
        text += str(self.translation[0]) + " " + str(self.translation[1]) + " " + str(self.translation[2])
        text += "\n\nScale:\n"
        text += str(self.scale)
        text += "\n\n4x4 Matrix:\n"
        mat = self.__compose_transformation()
        for i in range(4):
            text += f"{str(mat[i, 0])}, {str(mat[i, 1])}, {str(mat[i, 2])}, {str(mat[i, 3])}\n"
        self.qt_transformation_textbox.setText(text)

    def __compose_transformation(self):
        """
            Compose a 4-by-4 matrix from teaserpp solution
            :return: numpy 4x4 2D-array
        """
        s = self.scale
        rotR = self.rotation
        t = self.translation
        T = np.eye(4)
        T[0:3, 3] = t
        R = np.eye(4)
        R[0:3, 0:3] = rotR
        M = T.dot(R)
        if s == 1:
            M = T.dot(R)
        else:
            S = np.eye(4)
            S[0:3, 0:3] = np.diag([s, s, s])
            M = T.dot(R).dot(S)
        return M

    def __run_teaser(self):
        """
            Run TEASER++ registration algorithm passing the paths to the point clouds and
            TEASER's parameters as command line arguments
            :return: None
        """
        self.noise_bound = float(self.qt_noise_bound_edit.text())
        self.cbar2 = float(self.qt_cbar2_edit.text())
        self.estimate_scaling = int(self.qt_estimate_scaling.isChecked())
        self.rotation_max_iterations = float(self.qt_max_iter_edit.text())
        self.rotation_gnc_factor = float(self.qt_gnc_factor_edit.text())
        self.rotation_cost_threshold = float(self.qt_cost_threshold_edit.text())
        params = f"{self.noise_bound} {self.cbar2} {self.estimate_scaling} "
        params += f"{self.rotation_max_iterations} {self.rotation_gnc_factor} {self.rotation_cost_threshold}"
        reading_translation = False
        reading_rotation = False
        reading_scale = False
        rot_line = 0
        trans_line = 0

        with open('teaser_output.log', 'wb') as f:
            process = subprocess.Popen(
                                       f"teaser.exe {self.source_cloud} {self.target_cloud} {params}",
                                       stdout=subprocess.PIPE
                                       )
            for line in iter(process.stdout.readline, b''):
                if "Estimated rotation: " in line.decode("utf-8") :
                    reading_rotation = True
                    continue
                elif "Estimated translation: " in line.decode("utf-8") :
                    reading_translation = True
                    continue
                elif "Estimated scale:" in line.decode("utf-8") :
                    reading_scale = True
                    continue
                if reading_rotation and rot_line < 3:
                    values = [float(value) for value in line.decode("utf-8").split()]
                    self.rotation[rot_line, :] = values
                    rot_line += 1
                if reading_translation and trans_line < 3:
                    value = float(line.decode("utf-8"))
                    self.translation[trans_line] = value
                    trans_line += 1
                if reading_scale:
                    self.scale = float(line.decode("utf-8"))
                    reading_scale = False
                f.write(line)
        self.__update_transformation_view()

    def __update_clickability(self):
        """
            Check if all buttons that logically should be clickable are, in fact, clickable.
            :return: None
        """
        if self.source_cloud != "":
            self.qt_show_source_button.setEnabled(True)
        else:
            self.qt_show_source_button.setEnabled(False)
        if self.target_cloud != "":
            self.qt_show_target_button.setEnabled(True)
        else:
            self.qt_show_target_button.setEnabled(False)
        if self.source_cloud != "" and self.target_cloud != "":
            self.qt_show_alignment_button.setEnabled(True)
            self.qt_run_teaser_button.setEnabled(True)
        else:
            self.qt_show_alignment_button.setEnabled(False)
            self.qt_run_teaser_button.setEnabled(False)

    def __load_context(self):
        """
            Load runtime variables
            :return: None
        """
        try:
            with open("context.pkl", "rb") as context:
                fields = pkl.load(context)
            self.source_cloud = fields["src"]
            self.target_cloud = fields["dest"]
            self.source_cloud_size = fields["src_size"]
            self.target_cloud_size = fields["tgt_size"]
            self.rotation = fields["rotation"]
            self.translation = fields["translation"]
            self.scale = fields["scale"]
            self.statusbar.showMessage("Last used data loaded")
        except FileNotFoundError:
            self.statusbar.showMessage("Context not loaded")

    def __save_context(self):
        """
            Save runtime variables
            :return: None
        """
        try:
            fields = dict()
            fields["src"] = self.source_cloud
            fields["dest"] = self.target_cloud
            fields["src_size"] = self.source_cloud_size
            fields["tgt_size"] = self.target_cloud_size
            fields["rotation"] = self.rotation
            fields["translation"] = self.translation
            fields["scale"] = self.scale
            with open("context.pkl", "wb") as out:
                pkl.dump(fields, out)
            self.statusbar.showMessage("Runtime context saved")
        except (FileNotFoundError, RuntimeError):
            self.statusbar.showMessage("Could not save runtime context")

    def __show_source(self):
        """
            Show the source point cloud view window
            :return: None
        """
        pcd = o3d.io.read_point_cloud(
            self.source_cloud
        )
        self.source_point_cloud_view.load_cloud(pcd)
        try:
            self.source_point_cloud_view.show_window()
        except RuntimeError:
            pass

    def __show_target(self):
        """
            Show the target point cloud view window
            :return: None
        """
        pcd = o3d.io.read_point_cloud(
            self.target_cloud
        )
        self.target_point_cloud_view.load_cloud(pcd)
        try:
            self.target_point_cloud_view.show_window()
        except RuntimeError:
            pass

    def __show_alignment(self):
        """
            Show the alignment of the source and target point clouds view window
            :return: None
        """
        pcd = o3d.io.read_point_cloud(
            self.source_cloud
        )
        pcd.paint_uniform_color([1, 1, 0])
        pcd.transform(self.__compose_transformation())
        pcd.estimate_normals()
        self.alignment_point_cloud_view.load_cloud(pcd)
        pcd = o3d.io.read_point_cloud(
            self.target_cloud
        )
        self.alignment_point_cloud_view.load_cloud(pcd)
        try:
            self.alignment_point_cloud_view.show_window()
        except RuntimeError:
            pass

    def __load_source_cloud(self):
        """
            Load the source point cloud
            :return:
        """
        try:
            self.source_cloud = QFileDialog.getOpenFileName(self,
                                                            'Open file',
                                                            os.getcwd(),
                                                            "Point cloud files (*.ply)"
                                                            )[0]
            pcd = o3d.io.read_point_cloud(
                self.source_cloud
            )
            self.source_cloud_size = np.asarray(pcd.points).shape[0]
            self.statusbar.showMessage("Loaded source cloud")
            self.__save_context()
            self.__update_clickability()
        except (FileNotFoundError, RuntimeError):
            self.statusbar.showMessage("Could not load source cloud")

    def __load_target_cloud(self):
        """
            Load the target point cloud
            :return:
        """
        try:
            self.target_cloud = QFileDialog.getOpenFileName(self,
                                                                       'Open file',
                                                                       os.getcwd(),
                                                                       "Point cloud files (*.ply)"
                                                                       )[0]
            self.statusbar.showMessage("Loaded target cloud")
            pcd = o3d.io.read_point_cloud(
                self.target_cloud
            )
            self.target_cloud_size = np.asarray(pcd.points).shape[0]
            self.__save_context()
            self.__update_clickability()
        except (FileNotFoundError, RuntimeError):
            self.statusbar.showMessage("Could not load source cloud")

    def show_window(self):
        """Display the GUI window"""
        self.show()

