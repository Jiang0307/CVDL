import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from PyQt5 import QtWidgets
from tkinter import filedialog
from pathlib import Path
from source.Code.UI import Ui_MainWindow
from source.Code.YOLO_V3 import *

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.connect_ui(self)

    def connect_ui(self, parent=None):
        self.pushButton.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.select_video)
        
    def select_image(self, parent=None):
        root_path = os.path.join(os.path.dirname(__file__) , "source" , "Image")
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(initialdir=root_path)
        path = str(Path(path))       
        image_detect(path)

    def select_video(self, parent=None):
        root_path = os.path.join(os.path.dirname(__file__) , "source" , "Video")
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(initialdir=root_path)
        path = str(Path(path))
        start_video(path)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())