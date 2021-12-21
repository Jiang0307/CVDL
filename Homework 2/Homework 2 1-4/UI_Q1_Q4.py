import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *

class UI(object):
    def UI_SET(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1850,550)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
 # =================================GroupBox1=================================================
        self.GroupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_1.setGeometry(QtCore.QRect(50, 30, 400, 500))
        self.GroupBox_1.setObjectName("GroupBox_1")

        self.Button1_1 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_1.setGeometry(QtCore.QRect(20, 60, 360, 40))
        self.Button1_1.setObjectName("button1_1")

        self.Label1_1 = QtWidgets.QLabel(self.GroupBox_1)
        self.Label1_1.setGeometry(QtCore.QRect(50, 170, 300, 50))
        self.Label1_1.setObjectName("label1_1")

        self.Button1_2 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_2.setGeometry(QtCore.QRect(20, 280, 360, 40))
        self.Button1_2.setObjectName("button1_2")

        self.Label1_2 = QtWidgets.QLabel(self.GroupBox_1)
        self.Label1_2.setGeometry(QtCore.QRect(50, 390, 300, 50))
        self.Label1_2.setObjectName("label1_2")
# =================================GroupBox2=================================================
        self.GroupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_2.setGeometry(QtCore.QRect(500, 30, 400, 500))
        self.GroupBox_2.setObjectName("GroupBox_2")

        self.Button2_1 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_1.setGeometry(QtCore.QRect(20, 50, 360, 40))
        self.Button2_1.setObjectName("button2_1")

        self.Button2_2 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_2.setGeometry(QtCore.QRect(20, 130, 360, 40))
        self.Button2_2.setObjectName("button2_2")

        self.GroupBox_2_3 = QtWidgets.QGroupBox(self.GroupBox_2)
        self.GroupBox_2_3.setGeometry(QtCore.QRect(20, 210, 360, 120))
        self.GroupBox_2_3.setObjectName("GroupBox_2_3")
        
        self.Button2_3 = QtWidgets.QPushButton(self.GroupBox_2_3)
        self.Button2_3.setGeometry(QtCore.QRect(15, 35, 330, 30))
        self.Button2_3.setObjectName("button2_3")
        
        self.Label2_3 = QtWidgets.QLabel(self.GroupBox_2_3)
        self.Label2_3.setGeometry(15 , 75 , 200 , 30)
        self.Label2_3.setObjectName("label2_3")
        
        self.LineEdit2_3 = QtWidgets.QLineEdit(self.GroupBox_2_3)
        self.LineEdit2_3.setGeometry(165 , 75 , 180 , 30)
        self.LineEdit2_3.setObjectName("text2_3")
        
        self.Button2_4 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_4.setGeometry(QtCore.QRect(20, 370, 360, 40))
        self.Button2_4.setObjectName("button2_4")
        
        self.Button2_5 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_5.setGeometry(QtCore.QRect(20, 450, 360, 40))
        self.Button2_5.setObjectName("button2_5")
# =================================GroupBox3=================================================
        self.GroupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_3.setGeometry(QtCore.QRect(950, 30, 400, 500))
        self.GroupBox_3.setObjectName("GroupBox_3")

        self.Button3_1 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_1.setGeometry(QtCore.QRect(20, 150, 360, 40))
        self.Button3_1.setObjectName("button3_1")

        self.Button3_2 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_2.setGeometry(QtCore.QRect(20, 300, 360, 40))
        self.Button3_2.setObjectName("button3_2")
# =================================GroupBox4=================================================
        self.GroupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_4.setGeometry(QtCore.QRect(1400, 30, 400, 500))
        self.GroupBox_4.setObjectName("GroupBox_4")

        self.Button4_1 = QtWidgets.QPushButton(self.GroupBox_4)
        self.Button4_1.setGeometry(QtCore.QRect(20, 200, 360, 40))
        self.Button4_1.setObjectName("button4_1")
# =================================ELSE=================================================
        MainWindow.setCentralWidget(self.centralwidget)
        self.rename(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def rename(self, MainWindow):
        translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(translate("MainWindow", "MainWindow"))

        self.GroupBox_1.setTitle(translate("MainWindow", "1. Find Contour"))
        self.Button1_1.setText(translate("MainWindow", "1.1 Draw Contour"))
        self.Button1_2.setText(translate("MainWindow", "1.2 Count Rings"))
        self.Label1_1.setText(translate("MainWindow", "There are _ rings in img1.jpg"))
        self.Label1_2.setText(translate("MainWindow", "There are _ rings in img2.jpg"))

        self.GroupBox_2.setTitle(translate("MainWindow", "2. Corner Detection"))        
        self.GroupBox_2_3.setTitle(translate("MainWindow", "2.3 Find Extrinsic"))
        self.Label2_3.setText(translate("MainWindow", "Select image : "))
        self.Button2_1.setText(translate("MainWindow", "2.1 Find Corners"))
        self.Button2_2.setText(translate("MainWindow", "2.2 Find Intrinsic"))
        self.Button2_3.setText(translate("MainWindow", "2.3 Find Extrinsic"))
        self.Button2_4.setText(translate("MainWindow", "2.4 Find Distortion"))
        self.Button2_5.setText(translate("MainWindow", "2.5 Show result"))

        self.GroupBox_3.setTitle(translate("MainWindow", "3. Augmented Reality"))
        self.Button3_1.setText(translate("MainWindow", "3.1 Show Words on Board"))
        self.Button3_2.setText(translate("MainWindow", "3.2 Show Words Vertically"))

        self.GroupBox_4.setTitle(translate("MainWindow", "4. Stereo Disparity Map"))
        self.Button4_1.setText(translate("MainWindow", "4.1 Stereo Disparity Map"))

if __name__ == "__main__":
    start = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    UI = UI()
    UI.UI_SET(MainWindow)
    MainWindow.show()
    sys.exit(start.exec_())
