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

_3_1 = cv2_imread(os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q3_Image", "House.jpg"))
_3_2 = cv2_imread(os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q3_Image", "House.jpg"))
_3_3 = cv2_imread(os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q3_Image", "House.jpg"))

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
 # =================================================GroupBox2=================================================
    def Gaussian_Filter(self):
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q2_Image", "Lenna_whiteNoise.jpg")
        img = cv2_imread(path)
        Gaussian_Filter_result = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("original" , img)
        cv2.imshow("Gaussian Blur", Gaussian_Filter_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Bilateral_Filter(self):
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q2_Image", "Lenna_whiteNoise.jpg")
        img = cv2_imread(path)
        Bilateral_Filter_result = cv2.bilateralFilter(img , 9 , sigmaColor=90 , sigmaSpace=90)
        cv2.imshow("original", img)
        cv2.imshow("Bilateral Filter", Bilateral_Filter_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Median_Filter(self):
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q2_Image", "Lenna_pepperSalt.jpg")
        img = cv2_imread(path)
        Median_Filter_3x3_result = cv2.medianBlur(img , ksize=3)
        Median_Filter_5x5_result = cv2.medianBlur(img , ksize=5)
        cv2.imshow("original", img)
        cv2.imshow("Median Filter 3x3", Median_Filter_3x3_result)
        cv2.imshow("Median Filter 5x5", Median_Filter_5x5_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 # =================================================GroupBox3=================================================
    def mul(self,x, y, sigma):
        result = float( 1 / (2 * math.pi * (sigma**2) )) * float(math.exp( -((x**2) + (y**2)) / (2 * (sigma**2) )) )
        return result

    def get_gray_scale(self,original_img):
        gray_scale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        return gray_scale

    def get_zero_padding(self,original_img,gray_scale):
        row , col = original_img.shape[0] , original_img.shape[1]
        zero_padding = np.zeros((row+2, col+2), dtype = int)
        for i in range(1,row+1):
            for j in range(1, col+1):
                zero_padding[i][j] = gray_scale[i-1][j-1]
        return zero_padding

    def Gaussian_Blur(self):
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q3_Image", "House.jpg")
        img = cv2_imread(path)
        row , col = img.shape[0] , img.shape[1]
        gray_scale = self.get_gray_scale(img)
        zero_padding = self.get_zero_padding(img,gray_scale)

        sigma = 2
        kernel = [
                    [[-2, -2],[0, -2],[2, -2]] ,
                    [[-2, 0],[0, 0],[2, 0]] , 
                    [[-2, 2],[0, 2],[2, 2]]
                 ]
        filter = np.zeros((3,3), dtype = float)
        for i in range(3):
            for j in range(3):
                filter[i][j] = self.mul(kernel[i][j][0],kernel[i][j][1], sigma)
        filter = filter/filter.sum()
        result = np.zeros((row,col), dtype = int)
        for i in range(row):
            for j in range(col):
                temp = np.array([
                                    [zero_padding[i][j]   , zero_padding[i][j+1]   , zero_padding[i][j+2]] ,
                                    [zero_padding[i+1][j] , zero_padding[i+1][j+1] , zero_padding[i+1][j+2]] ,
                                    [zero_padding[i+2][j] , zero_padding[i+2][j+1] , zero_padding[i+2][j+2]]
                                ])
                result[i][j] = (np.multiply(temp, filter)).sum()

        gray_scale = gray_scale.astype(np.uint8)
        zero_padding = zero_padding.astype(np.uint8)
        result = result.astype(np.uint8)
        _3_1 = result
        cv2.imshow("Gaussian Blur",_3_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Sobel_X(self):
        img = _3_1
        row , col = img.shape[0] , img.shape[1]
        gray_scale = self.get_gray_scale(img)
        zero_padding = self.get_zero_padding(img,gray_scale)
        SOBEL_X_FILTER = [
                            [-1, 0, 1] ,
                            [-2, 0, 2] ,
                            [-1, 0, 1]
                         ]
        result = np.zeros((row,col), dtype = int)
        for i in range(row):
            for j in range(col):
                threshold = 150
                temp = np.array([
                                    [zero_padding[i][j]   , zero_padding[i][j+1]   , zero_padding[i][j+2]] ,
                                    [zero_padding[i+1][j] , zero_padding[i+1][j+1] , zero_padding[i+1][j+2]] ,
                                    [zero_padding[i+2][j] , zero_padding[i+2][j+1] , zero_padding[i+2][j+2]]
                                ])
                result[i][j] = (np.multiply(temp , SOBEL_X_FILTER)).sum()
                result[i][j] = abs(result[i][j])
                #if result[i][j] > threshold and result[i][j]<255:
                 #   result[i][j] = 0
                if result[i][j] < 0:
                    result[i][j] = 0
                elif result[i][j]>=255:
                    result[i][j] = 255
                
        result = result.astype(np.uint8)
        _3_2 = result
        cv2.imshow("Sobel X",_3_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Sobel_Y(self):
        img = _3_1
        row , col = img.shape[0] , img.shape[1]
        gray_scale = self.get_gray_scale(img)
        zero_padding = self.get_zero_padding(img,gray_scale)

        SOBEL_Y_FILTER = [
                            [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]
                         ]

        result = np.zeros((row,col), dtype = int)
        for i in range(row):
            for j in range(col):
                threshold = 150
                t = np.array([
                                [zero_padding[i][j], zero_padding[i][j+1], zero_padding[i][j+2]],
                                [zero_padding[i+1][j], zero_padding[i+1][j+1], zero_padding[i+1][j+2]],
                                [zero_padding[i+2][j], zero_padding[i+2][j+1], zero_padding[i+2][j+2]]
                            ])
                result[i][j] = (np.multiply(t, SOBEL_Y_FILTER)).sum()
                result[i][j] = abs(result[i][j])
                #if result[i][j]>threshold and result[i][j]<255:
                 #       result[i][j] = 0
                if result[i][j] < 0:
                    result[i][j] = 0
                elif result[i][j] > 255:
                    result[i][j] = 255
                
        result = result.astype(np.uint8)
        _3_3 = result
        cv2.imshow("Sobel Y", _3_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Magnitude(self):
        SOBEL_X_FILTER = [
                            [-1, 0, 1] ,
                            [-2, 0, 2] ,
                            [-1, 0, 1]
                         ]
        SOBEL_Y_FILTER = [
                            [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]
                         ]
        img = _3_1
        row , col = img.shape[0] , img.shape[1]
        gray_scale = self.get_gray_scale(img)
        zero_padding = self.get_zero_padding(img,gray_scale)
        result = np.zeros((row,col), dtype = int)
        for i in range(row):
            for j in range(col):
                threshold = 150
                temp = np.array([
                                    [zero_padding[i][j]   , zero_padding[i][j+1]   , zero_padding[i][j+2]] ,
                                    [zero_padding[i+1][j] , zero_padding[i+1][j+1] , zero_padding[i+1][j+2]] ,
                                    [zero_padding[i+2][j] , zero_padding[i+2][j+1] , zero_padding[i+2][j+2]]
                                ])
                result[i][j] = (np.multiply(temp , SOBEL_X_FILTER)).sum()
                result[i][j] = abs(result[i][j])
                #if result[i][j] > threshold and result[i][j]<255:
                 #   result[i][j] = 0
                if result[i][j] < 0:
                    result[i][j] = 0
                elif result[i][j]>=255:
                    result[i][j] = 255
        SX = result.astype(np.uint8)
        img = _3_1
        row , col = img.shape[0] , img.shape[1]
        gray_scale = self.get_gray_scale(img)
        zero_padding = self.get_zero_padding(img,gray_scale)

        result = np.zeros((row,col), dtype = int)
        for i in range(row):
            for j in range(col):
                threshold = 150
                t = np.array([
                                [zero_padding[i][j], zero_padding[i][j+1], zero_padding[i][j+2]],
                                [zero_padding[i+1][j], zero_padding[i+1][j+1], zero_padding[i+1][j+2]],
                                [zero_padding[i+2][j], zero_padding[i+2][j+1], zero_padding[i+2][j+2]]
                            ])
                result[i][j] = (np.multiply(t, SOBEL_Y_FILTER)).sum()
                result[i][j] = abs(result[i][j])
                #if result[i][j]>threshold and result[i][j]<255:
                 #       result[i][j] = 0
                if result[i][j] < 0:
                    result[i][j] = 0
                elif result[i][j] > 255:
                    result[i][j] = 255
        SY = result.astype(np.uint8)

        row , col = _3_1.shape[0] , _3_1.shape[1]
        result = np.zeros((row,col), dtype = np.float32)
        for i in range(row):
            for j in range(col):
                tempx = SX[i][j]**2
                tempy = SY[i][j]**2
                result[i][j] = math.sqrt( tempx + tempy )
                if result[i][j] > 255 and result[i][j]<255:
                    result[i][j] = 255

        result = result.astype(np.uint8)
        cv2.imshow("Magnitude", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# =================================================GroupBox4=================================================
    def translate(self,image , x , y):
        translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
        translation_result = cv2.warpAffine(image , translation_matrix , (image.shape[0]+128 , image.shape[1]+60 ))
        x,y,width,height = 0,0,400,300
        temp = translation_result[y:y+height , x:x+width]
        result = cv2.copyMakeBorder(temp,0,0,0,16, cv2.BORDER_CONSTANT, value=[0,0,0])
        #print(translation_result.shape,result.shape)
        return result

    def Resize(self):
        path = os.path.join(os.path.dirname(__file__),"Dataset_OpenCvDl_Hw1", "Q4_Image", "SQUARE-01.png")
        img = cv2_imread(path)
        Resize_result = cv2.resize(img, (256, 256))
        cv2.namedWindow("Resize",0)
        cv2.imshow("Resize", Resize_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Translation(self):   
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q4_Image", "SQUARE-01.png")
        img = cv2_imread(path)
        Resize_result = cv2.resize(img, (256, 256))
        result = self.translate(Resize_result,0,60)
        #print(result.shape)
        cv2.namedWindow("Translation",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Translation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Rotation_Scaling(self):   
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q4_Image", "SQUARE-01.png")
        img = cv2_imread(path)
        Resize_result = cv2.resize(img, (256, 256))
        translation_result = self.translate(Resize_result,0,60)

        matrix = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        Rotation_Scaling_result = cv2.warpAffine(translation_result, matrix, (img.shape[1], img.shape[0]))
        #crop image
        x,y,width,height = 0,0,400,300
        result = Rotation_Scaling_result[y:y+height , x:x+width]
        #print(result.shape)
        cv2.imshow("Rotation Scaling", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Shearing(self):
        path = os.path.join(os.path.dirname(__file__), "Dataset_OpenCvDl_Hw1", "Q4_Image", "SQUARE-01.png")
        img = cv2_imread(path)
        Resize_result = cv2.resize(img, (256, 256))
        translation_result = self.translate(Resize_result,0,60)
        matrix = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        Rotation_Scaling_result = cv2.warpAffine(translation_result, matrix, (img.shape[1], img.shape[0]))
        old_position = np.float32([[50,50], [200,50], [50,200]] )
        new_position = np.float32([[10,100], [200,50], [100,250]] )
        transform_matrix = cv2.getAffineTransform(old_position,new_position)
        shearing_result = cv2.warpAffine(Rotation_Scaling_result, transform_matrix, (Rotation_Scaling_result.shape[0],Rotation_Scaling_result.shape[1]))
        #crop image
        x,y,width,height = 0,0,400,300
        result = shearing_result[y:y+height , x:x+width]
        #print(result.shape)
        cv2.imshow("Shearing", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())