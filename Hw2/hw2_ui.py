# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_hw2(object):
    def setupUi(self, hw2):
        hw2.setObjectName("hw2")
        hw2.resize(1126, 812)
        self.centralwidget = QtWidgets.QWidget(hw2)
        self.centralwidget.setObjectName("centralwidget")
        self.Image = QtWidgets.QPushButton(self.centralwidget)
        self.Image.setGeometry(QtCore.QRect(30, 130, 75, 23))
        self.Image.setObjectName("Image")
        self.Video = QtWidgets.QPushButton(self.centralwidget)
        self.Video.setGeometry(QtCore.QRect(30, 170, 75, 23))
        self.Video.setObjectName("Video")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(140, 30, 211, 91))
        self.groupBox.setObjectName("groupBox")
        self.Subtraction = QtWidgets.QPushButton(self.groupBox)
        self.Subtraction.setGeometry(QtCore.QRect(30, 30, 151, 31))
        self.Subtraction.setObjectName("Subtraction")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(140, 150, 211, 121))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Preprocessing = QtWidgets.QPushButton(self.groupBox_2)
        self.Preprocessing.setGeometry(QtCore.QRect(30, 30, 151, 31))
        self.Preprocessing.setObjectName("Preprocessing")
        self.Tracking = QtWidgets.QPushButton(self.groupBox_2)
        self.Tracking.setGeometry(QtCore.QRect(30, 70, 151, 31))
        self.Tracking.setObjectName("Tracking")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(140, 300, 211, 81))
        self.groupBox_3.setObjectName("groupBox_3")
        self.Reduction = QtWidgets.QPushButton(self.groupBox_3)
        self.Reduction.setGeometry(QtCore.QRect(30, 30, 151, 31))
        self.Reduction.setObjectName("Reduction")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(380, 30, 721, 351))
        self.groupBox_4.setObjectName("groupBox_4")
        self.Struction = QtWidgets.QPushButton(self.groupBox_4)
        self.Struction.setGeometry(QtCore.QRect(30, 30, 151, 31))
        self.Struction.setObjectName("Struction")
        self.Acc_loss = QtWidgets.QPushButton(self.groupBox_4)
        self.Acc_loss.setGeometry(QtCore.QRect(30, 80, 151, 31))
        self.Acc_loss.setObjectName("Acc_loss")
        self.Predict = QtWidgets.QPushButton(self.groupBox_4)
        self.Predict.setGeometry(QtCore.QRect(30, 130, 151, 31))
        self.Predict.setObjectName("Predict")
        self.Reset = QtWidgets.QPushButton(self.groupBox_4)
        self.Reset.setGeometry(QtCore.QRect(30, 180, 151, 31))
        self.Reset.setObjectName("Reset")
        self.paint = QtWidgets.QLabel(self.groupBox_4)
        self.paint.setGeometry(QtCore.QRect(200, 20, 501, 311))
        self.paint.setText("")
        self.paint.setObjectName("paint")
        self.result = QtWidgets.QLabel(self.groupBox_4)
        self.result.setGeometry(QtCore.QRect(90, 260, 31, 41))
        self.result.setText("")
        self.result.setObjectName("result")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(380, 400, 621, 391))
        self.groupBox_5.setObjectName("groupBox_5")
        self.Load = QtWidgets.QPushButton(self.groupBox_5)
        self.Load.setGeometry(QtCore.QRect(30, 30, 151, 31))
        self.Load.setObjectName("Load")
        self.Show_Image = QtWidgets.QPushButton(self.groupBox_5)
        self.Show_Image.setGeometry(QtCore.QRect(30, 80, 151, 31))
        self.Show_Image.setObjectName("Show_Image")
        self.Struction_ResNet = QtWidgets.QPushButton(self.groupBox_5)
        self.Struction_ResNet.setGeometry(QtCore.QRect(30, 130, 151, 31))
        self.Struction_ResNet.setObjectName("Struction_ResNet")
        self.Comprasion = QtWidgets.QPushButton(self.groupBox_5)
        self.Comprasion.setGeometry(QtCore.QRect(30, 180, 151, 31))
        self.Comprasion.setObjectName("Comprasion")
        self.Inference = QtWidgets.QPushButton(self.groupBox_5)
        self.Inference.setGeometry(QtCore.QRect(30, 230, 151, 31))
        self.Inference.setObjectName("Inference")
        self.label_q5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_q5.setGeometry(QtCore.QRect(350, 350, 91, 16))
        self.label_q5.setObjectName("label_q5")
        self.label_img = QtWidgets.QLabel(self.groupBox_5)
        self.label_img.setGeometry(QtCore.QRect(250, 20, 300, 300))
        self.label_img.setText("")
        self.label_img.setScaledContents(True)
        self.label_img.setObjectName("label_img")
        hw2.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(hw2)
        self.statusbar.setObjectName("statusbar")
        hw2.setStatusBar(self.statusbar)

        self.retranslateUi(hw2)
        QtCore.QMetaObject.connectSlotsByName(hw2)

    def retranslateUi(self, hw2):
        _translate = QtCore.QCoreApplication.translate
        hw2.setWindowTitle(_translate("hw2", "MainWindow"))
        self.Image.setText(_translate("hw2", "Load Image"))
        self.Video.setText(_translate("hw2", "Load Video"))
        self.groupBox.setTitle(_translate("hw2", "1.Background Subtraction"))
        self.Subtraction.setText(_translate("hw2", "1.Background Subtraction"))
        self.groupBox_2.setTitle(_translate("hw2", "2.Optical Flow"))
        self.Preprocessing.setText(_translate("hw2", "2.1 Preprocessing"))
        self.Tracking.setText(_translate("hw2", "2.2 Video tracking"))
        self.groupBox_3.setTitle(_translate("hw2", "3. PCA"))
        self.Reduction.setText(_translate("hw2", "3. Dimension Reduction"))
        self.groupBox_4.setTitle(_translate("hw2", "4. MNIST Classifier Using VGG19"))
        self.Struction.setText(_translate("hw2", "1. Show Model Structure"))
        self.Acc_loss.setText(_translate("hw2", "2. Show Accuracy and Loss"))
        self.Predict.setText(_translate("hw2", "3. Predict"))
        self.Reset.setText(_translate("hw2", "4. Reset"))
        self.groupBox_5.setTitle(_translate("hw2", "4. MNIST Classifier Using VGG19"))
        self.Load.setText(_translate("hw2", "Load Image"))
        self.Show_Image.setText(_translate("hw2", "5.1. Show Images"))
        self.Struction_ResNet.setText(_translate("hw2", "5.2. Show Model Structure"))
        self.Comprasion.setText(_translate("hw2", "5.3. Show Comprasion"))
        self.Inference.setText(_translate("hw2", "5.4. Inference"))
        self.label_q5.setText(_translate("hw2", "prediction = "))
