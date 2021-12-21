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
    def calculate(self,offset):
        offset = offset/255
        cv2.addWeighted(img_1, 1-offset, img_2, offset, 0, blend_result)
        cv2.imshow("Blend", blend_result)

    def Load_Image(self):
        path = os.path.join(os.path.dirname(__file__),"Dataset_OpenCvDl_Hw1", "Q1_Image", "Sun.jpg")
        img = cv2_imread(path)
        height,width,channel = img.shape
        print("height : ",height)
        print("width : ",width)
        cv2.imshow("Hw1-1", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Color_Separetion(self):
        path = os.path.join(os.path.dirname(__file__),"Dataset_OpenCvDl_Hw1", "Q1_Image", "Sun.jpg")
        img = cv2_imread(path)
        (B,G,R) = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.imshow("B",cv2.merge([B, zeros, zeros]))
        cv2.imshow("G",cv2.merge([zeros, G, zeros]))
        cv2.imshow("R",cv2.merge([zeros, zeros, R]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Color_Transformation(self):
        path = os.path.join(os.path.dirname(__file__),"Dataset_OpenCvDl_Hw1", "Q1_Image", "Sun.jpg")
        img = cv2_imread(path)
        I1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("I1", I1)
        # new image
        I2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                I2[i,j] = (int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2])) / 3

        cv2.imshow("I2", I2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Blending(self):
        global img_1,img_2,blend_result
        path_1 = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q1_Image", "Dog_Strong.jpg")
        img_1 = cv2_imread(path_1)
        path_2 = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q1_Image", "Dog_Weak.jpg")
        img_2 = cv2_imread(path_2)
        blend_result = np.zeros((img_1.shape[0], img_1.shape[1], 3), np.uint8)
        cv2.imshow("Blend", img_1)
        cv2.createTrackbar("bar" , "Blend" , 0 , 255 , self.calculate )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())