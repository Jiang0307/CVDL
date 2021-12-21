import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets

class UI(object):
    def UI_SET(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(550,600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
 # =================================GroupBox1=================================================
        self.GroupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        self.GroupBox_1.setGeometry(QtCore.QRect(30, 30, 500, 550))
        self.GroupBox_1.setObjectName("GroupBox_1")

        self.button_1 = QtWidgets.QPushButton(self.GroupBox_1)
        self.button_1.setGeometry(QtCore.QRect(25, 50, 450, 40))
        self.button_1.setObjectName("button_2")

        self.button_2 = QtWidgets.QPushButton(self.GroupBox_1)
        self.button_2.setGeometry(QtCore.QRect(25, 150, 450, 40))
        self.button_2.setObjectName("button_3")

        self.button_3 = QtWidgets.QPushButton(self.GroupBox_1)
        self.button_3.setGeometry(QtCore.QRect(25, 250, 450, 40))
        self.button_3.setObjectName("button_4")

        self.textEdit = QtWidgets.QLineEdit(self.GroupBox_1)
        self.textEdit.setGeometry(QtCore.QRect(25, 350, 450, 40))
        self.textEdit.setObjectName("textEdit")

        self.button_4 = QtWidgets.QPushButton(self.GroupBox_1)
        self.button_4.setGeometry(QtCore.QRect(25, 450, 450, 40))
        self.button_4.setObjectName("button_5")
# =================================ELSE=================================================
        MainWindow.setCentralWidget(self.centralwidget)
        self.rename(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def rename(self, MainWindow):
        translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(translate("MainWindow", "MainWindow"))
        self.GroupBox_1.setTitle(translate("MainWindow", "ResNet50"))
        self.button_1.setText(translate("MainWindow", "1. Show Model Structure"))
        self.button_2.setText(translate("MainWindow", "2. Show Tensorboard"))
        self.button_3.setText(translate("MainWindow", "3. Test"))
        self.button_4.setText(translate("MainWindow", "4. Data Augmentation"))


if __name__ == "__main__":
    start = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    UI = UI()
    UI.UI_SET(MainWindow)
    MainWindow.show()
    sys.exit(start.exec_())
