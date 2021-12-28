from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(458, 436)
        MainWindow.setStyleSheet(u"background-color: qradialgradient(spread:pad, cx:0.485, cy:0.5, radius:0.796, fx:0.05, fy:0.074, stop:0.272277 rgba(252, 255, 112, 224), stop:1 rgba(255, 255, 255, 255));")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(-10, 80, 481, 101))
        font = QFont()
        font.setFamily(u"Bahnschrift SemiBold Condensed")
        font.setPointSize(26)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(7)
        font.setKerning(True)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(u"background-color: rgb(0, 255, 127);")
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(0, 260, 461, 101))
        font1 = QFont()
        font1.setFamily(u"Bahnschrift SemiBold Condensed")
        font1.setPointSize(26)
        font1.setBold(True)
        font1.setWeight(75)
        self.pushButton_2.setFont(font1)
        self.pushButton_2.setStyleSheet(u"background-color: rgb(85, 255, 127);")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Gun and Fire Detection", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Select Test Image", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Select Test Video", None))