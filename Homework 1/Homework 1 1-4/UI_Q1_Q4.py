import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

class UI(object):
    def UI_SET(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1350,550)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
 # =================================GroupBox1=================================================
        self.GroupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_1.setGeometry(QtCore.QRect(30, 30, 300, 500))
        self.GroupBox_1.setObjectName("GroupBox_1")

        self.Button1_1 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_1.setGeometry(QtCore.QRect(10, 60, 275, 40))
        self.Button1_1.setObjectName("button1_1")

        self.Button1_2 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_2.setGeometry(QtCore.QRect(10, 170, 275, 40))
        self.Button1_2.setObjectName("button1_2")

        self.Button1_3 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_3.setGeometry(QtCore.QRect(10, 280, 275, 40))
        self.Button1_3.setObjectName("button1_3")

        self.Button1_4 = QtWidgets.QPushButton(self.GroupBox_1)
        self.Button1_4.setGeometry(QtCore.QRect(10, 390, 275, 40))
        self.Button1_4.setObjectName("button1_4")
# =================================GroupBox2=================================================
        self.GroupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_2.setGeometry(QtCore.QRect(360, 30, 300, 500))
        self.GroupBox_2.setObjectName("GroupBox_2")

        self.Button2_1 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_1.setGeometry(QtCore.QRect(25, 120, 250, 40))
        self.Button2_1.setObjectName("button2_1")

        self.Button2_2 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_2.setGeometry(QtCore.QRect(25, 230, 250, 40))
        self.Button2_2.setObjectName("button2_2")

        self.Button2_3 = QtWidgets.QPushButton(self.GroupBox_2)
        self.Button2_3.setGeometry(QtCore.QRect(25, 340, 250, 40))
        self.Button2_3.setObjectName("button2_3")
# =================================GroupBox3=================================================
        self.GroupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_3.setGeometry(QtCore.QRect(690, 30, 300, 500))
        self.GroupBox_3.setObjectName("GroupBox_3")

        self.Button3_1 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_1.setGeometry(QtCore.QRect(25, 60, 250, 40))
        self.Button3_1.setObjectName("button3_1")

        self.Button3_2 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_2.setGeometry(QtCore.QRect(25, 170, 250, 40))
        self.Button3_2.setObjectName("button3_2")

        self.Button3_3 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_3.setGeometry(QtCore.QRect(25, 280, 250, 40))
        self.Button3_3.setObjectName("button3_3")

        self.Button3_4 = QtWidgets.QPushButton(self.GroupBox_3)
        self.Button3_4.setGeometry(QtCore.QRect(25, 390, 250, 40))
        self.Button3_4.setObjectName("button3_4")
# =================================GroupBox4=================================================
        self.GroupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_4.setGeometry(QtCore.QRect(1020, 30, 300, 500))
        self.GroupBox_4.setObjectName("GroupBox_4")

        self.Button4_1 = QtWidgets.QPushButton(self.GroupBox_4)
        self.Button4_1.setGeometry(QtCore.QRect(25, 60, 250, 40))
        self.Button4_1.setObjectName("button4_1")

        self.Button4_2 = QtWidgets.QPushButton(self.GroupBox_4)
        self.Button4_2.setGeometry(QtCore.QRect(25, 170, 250, 40))
        self.Button4_2.setObjectName("button4_2")

        self.Button4_3 = QtWidgets.QPushButton(self.GroupBox_4)
        self.Button4_3.setGeometry(QtCore.QRect(25, 280, 250, 40))
        self.Button4_3.setObjectName("button4_3")

        self.Button4_4 = QtWidgets.QPushButton(self.GroupBox_4)
        self.Button4_4.setGeometry(QtCore.QRect(25, 390, 250, 40))
        self.Button4_4.setObjectName("button4_4")
# =================================ELSE=================================================
        MainWindow.setCentralWidget(self.centralwidget)
        self.rename(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def rename(self, MainWindow):
        translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(translate("MainWindow", "MainWindow"))

        self.GroupBox_1.setTitle(translate("MainWindow", "1. Image Processing"))
        self.Button1_1.setText(translate("MainWindow", "1.1 Load Image"))
        self.Button1_2.setText(translate("MainWindow", "1.2 Color Separation"))
        self.Button1_3.setText(translate("MainWindow", "1.3 Color Transformations"))
        self.Button1_4.setText(translate("MainWindow", "1.4 Blending"))

        self.GroupBox_2.setTitle(translate("MainWindow", "2. Image Smoothing"))
        self.Button2_1.setText(translate("MainWindow", "2.1 Gaussian Blur"))
        self.Button2_2.setText(translate("MainWindow", "2.2 Bilateral Filter"))
        self.Button2_3.setText(translate("MainWindow", "2.3 Median Filter"))

        self.GroupBox_3.setTitle(translate("MainWindow", "3. Edge Detection"))
        self.Button3_1.setText(translate("MainWindow", "3.1 Gaussian Blur"))
        self.Button3_2.setText(translate("MainWindow", "3.2 Sobel X"))
        self.Button3_3.setText(translate("MainWindow", "3.3 Sobel Y"))
        self.Button3_4.setText(translate("MainWindow", "3.4 Magnitude"))

        self.GroupBox_4.setTitle(translate("MainWindow", "4. Transformation"))
        self.Button4_1.setText(translate("MainWindow", "4.1 Resize"))
        self.Button4_2.setText(translate("MainWindow", "4.2 Translation"))
        self.Button4_3.setText(translate("MainWindow", "4.3 Rotation , Scaling"))
        self.Button4_4.setText(translate("MainWindow", "4.4 Shearing"))


if __name__ == "__main__":
    start = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    UI = UI()
    UI.UI_SET(MainWindow)
    MainWindow.show()
    sys.exit(start.exec_())
