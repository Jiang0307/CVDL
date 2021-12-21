from math import sqrt
from typing import Pattern
from PyQt5 import QtWidgets, QtGui, QtCore
from UI_Q1_Q4 import UI
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

PATH = os.path.join(os.path.dirname(__file__) , "Dataset_OpenCvDl_Hw2" , "Q2_Image")
NX = 11
NY = 8
IMAGES = []
IMAGE_COUNT = 0

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8) , -1)
    return img

def count_images(type):
    count = 0
    for file in os.listdir(PATH):       
        if file.endswith(type):
            count += 1
    return count

def read_images(image_count):
    for i in range(image_count):
        path = os.path.join(os.path.dirname(__file__) , "Dataset_OpenCvDl_Hw2" , "Q2_Image" , str(i+1)+".bmp")
        img = cv2_imread(path)
        IMAGES.append(img)
        
    
def draw_corner():
    temp = np.asarray(IMAGES)
    result = []
    for img in temp:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX,NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            img = cv2.drawChessboardCorners(img , (NX,NY) , corners , ret)
            result.append(img)
    return result

def undistort():
    temp = np.asarray(IMAGES)
    objp = np.zeros((1, NX*NY, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    result = []
    for img in temp:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
            imgpoints.append(corners2)
    ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
    temp = np.asarray(IMAGES)
    for img in temp:
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coefficients, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]
        result.append(img)
    return result

def plot_to_image (fig):
    fig.canvas.draw()
    width,height = fig.canvas.get_width_height()
    buffer = np.frombuffer( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buffer.shape = (width,height,4)
    buffer = np.roll (buffer,3,axis=2)
    return buffer

class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.UI_SET(self)
        self.Button2_1.clicked.connect(self.HW2_2_1)
        self.Button2_2.clicked.connect(self.HW2_2_2)
        self.Button2_3.clicked.connect(self.HW2_2_3)
        self.Button2_4.clicked.connect(self.HW2_2_4)
        self.Button2_5.clicked.connect(self.HW2_2_5)
        self.image_count = count_images(".bmp")
        self.images = IMAGES
        self.result_2_1 = draw_corner()
        self.result_2_5 = undistort()
        
    def HW2_2_1(self):
        title = "2-1"
        for img in self.result_2_1:
            cv2.namedWindow(title , cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title , 1600 , 1600)
            cv2.imshow(title , img)
            cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()  
        return

    def HW2_2_2(self):
        self.images = np.asarray(IMAGES)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)       
        ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
        print("intrinsic matrix: ")
        print(intrinsic_matrix)
        print("")
        return

    def HW2_2_3(self):
        try:
            self.images = np.asarray(IMAGES)
            index = int( self.LineEdit2_3.text() )
            if index<0 or index>14:
                return
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
            objp = np.zeros((1 , NX*NY , 3) , np.float32)
            objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
            objpoints = []
            imgpoints = []
            for img in self.images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
                    imgpoints.append(corners2)       
            ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
            r = np.zeros((3,3))
            cv2.Rodrigues(rvecs[index] , r , jacobian=0)
            extrinstic_matrix = np.concatenate((r , tvecs[index]) , axis=1)
            print("extrinsic matrix: ")
            print(extrinstic_matrix)
            print("")
        except ValueError:
            pass
        except Exception:
            pass
        return
    
    def HW2_2_4(self):
        self.images = np.asarray(IMAGES)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)       
        ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
        print("distortion coefficients: ")
        print(distortion_coefficients)
        print("")
        return
        
    def HW2_2_5(self):
        self.images = np.asarray(IMAGES)
        title_1 = "2-5 Distorted"
        title_2 = "2-5 Undistorted"
        for i in range(self.image_count):
            distorted_img = self.images[i]
            distorted_img = cv2.resize(distorted_img , (800,800) )
            undistorted_img = self.result_2_5[i]
            undistorted_img = cv2.resize(undistorted_img , (800,800) )

            cv2.imshow(title_1 , distorted_img)
            cv2.imshow(title_2 , undistorted_img)
            cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    IMAGE_COUNT = count_images(".bmp")
    read_images(IMAGE_COUNT)
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())