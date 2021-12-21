from math import sqrt
from typing import Pattern
from PyQt5 import QtWidgets, QtGui, QtCore
from UI_Q1_Q4 import UI
import sys
import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import math

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.UI_SET(self)
 # =================================================GroupBox1=================================================
        self.Button1_1.clicked.connect(self.Load_Image)
        self.Button1_2.clicked.connect(self.Color_Separetion)
        self.Button1_3.clicked.connect(self.Color_Transformation)
        self.Button1_4.clicked.connect(self.Blending)
 # =================================================GroupBox2=================================================
        self.Button2_1.clicked.connect(self.Gaussian_Filter)
        self.Button2_2.clicked.connect(self.Bilateral_Filter)
        self.Button2_3.clicked.connect(self.Median_Filter)
 # =================================================GroupBox3=================================================
        self.Button3_1.clicked.connect(self.Gaussian_Blur)
        self.Button3_2.clicked.connect(self.Sobel_X)
        self.Button3_3.clicked.connect(self.Sobel_Y)
        self.Button3_4.clicked.connect(self.Magnitude)
 # =================================================GroupBox4=================================================
        self.Button4_1.clicked.connect(self.Resize)
        self.Button4_2.clicked.connect(self.Translation)
        self.Button4_3.clicked.connect(self.Rotation_Scaling)
        self.Button4_4.clicked.connect(self.Shearing)

 # =================================GroupBox1=================================================




if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())