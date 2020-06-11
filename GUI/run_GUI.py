import main_window
from PyQt5 import QtWidgets

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    GUI = main_window.MainWindow()
    GUI.show_window()
    app.exec_()