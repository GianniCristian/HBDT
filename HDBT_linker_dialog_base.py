# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HDBT_linker_dialog_base.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(397, 505)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(31, 448, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 73, 121, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.comboBox = QtGui.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(30, 103, 341, 21))
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 123, 121, 31))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.comboBox_2 = QtGui.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(30, 153, 341, 21))
        self.comboBox_2.setObjectName(_fromUtf8("comboBox_2"))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(30, 175, 121, 31))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.lineEdit = QtGui.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(30, 203, 251, 20))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.pushButton = QtGui.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(298, 201, 75, 23))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(41, 281, 61, 31))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(31, 294, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.lineEdit_2 = QtGui.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(131, 291, 31, 20))
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_3 = QtGui.QLineEdit(Dialog)
        self.lineEdit_3.setGeometry(QtCore.QRect(171, 291, 31, 20))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.lineEdit_4 = QtGui.QLineEdit(Dialog)
        self.lineEdit_4.setGeometry(QtCore.QRect(211, 291, 31, 20))
        self.lineEdit_4.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.lineEdit_5 = QtGui.QLineEdit(Dialog)
        self.lineEdit_5.setGeometry(QtCore.QRect(251, 291, 31, 20))
        self.lineEdit_5.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_5.setObjectName(_fromUtf8("lineEdit_5"))
        self.lineEdit_7 = QtGui.QLineEdit(Dialog)
        self.lineEdit_7.setGeometry(QtCore.QRect(171, 331, 31, 20))
        self.lineEdit_7.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_7.setObjectName(_fromUtf8("lineEdit_7"))
        self.label_6 = QtGui.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(31, 334, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.lineEdit_9 = QtGui.QLineEdit(Dialog)
        self.lineEdit_9.setGeometry(QtCore.QRect(251, 331, 31, 20))
        self.lineEdit_9.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_9.setObjectName(_fromUtf8("lineEdit_9"))
        self.label_7 = QtGui.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(31, 321, 81, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.lineEdit_6 = QtGui.QLineEdit(Dialog)
        self.lineEdit_6.setGeometry(QtCore.QRect(131, 331, 31, 20))
        self.lineEdit_6.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_6.setObjectName(_fromUtf8("lineEdit_6"))
        self.lineEdit_8 = QtGui.QLineEdit(Dialog)
        self.lineEdit_8.setGeometry(QtCore.QRect(211, 331, 31, 20))
        self.lineEdit_8.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_8.setObjectName(_fromUtf8("lineEdit_8"))
        self.label_8 = QtGui.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(31, 391, 121, 31))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.comboBox_3 = QtGui.QComboBox(Dialog)
        self.comboBox_3.setGeometry(QtCore.QRect(131, 398, 151, 21))
        self.comboBox_3.setObjectName(_fromUtf8("comboBox_3"))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.lineEdit_10 = QtGui.QLineEdit(Dialog)
        self.lineEdit_10.setGeometry(QtCore.QRect(131, 365, 31, 20))
        self.lineEdit_10.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_10.setObjectName(_fromUtf8("lineEdit_10"))
        self.label_9 = QtGui.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(21, 359, 101, 31))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.label_10 = QtGui.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(165, 360, 41, 31))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.label_11 = QtGui.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(30, 12, 261, 61))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_11.setFont(font)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.lineEdit_1 = QtGui.QLineEdit(Dialog)
        self.lineEdit_1.setGeometry(QtCore.QRect(30, 252, 251, 20))
        self.lineEdit_1.setObjectName(_fromUtf8("lineEdit_1"))
        self.label_12 = QtGui.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(30, 224, 121, 31))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.pushButton_2 = QtGui.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(298, 250, 75, 23))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.logo_unipv = QtGui.QLabel(Dialog)
        self.logo_unipv.setGeometry(QtCore.QRect(294, 6, 91, 81))
        self.logo_unipv.setText(_fromUtf8(""))
        self.logo_unipv.setPixmap(QtGui.QPixmap(_fromUtf8("HDBT/unipv.png")))
        self.logo_unipv.setObjectName(_fromUtf8("logo_unipv"))

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "HDBT", None))
        self.label.setText(_translate("Dialog", "Input Image:", None))
        self.comboBox.setItemText(0, _translate("Dialog", "[Choose from a file...]", None))
        self.label_2.setText(_translate("Dialog", "Training/Validation set:", None))
        self.comboBox_2.setItemText(0, _translate("Dialog", "[Choose from a file...]", None))
        self.label_3.setText(_translate("Dialog", "Output result:", None))
        self.pushButton.setText(_translate("Dialog", "...", None))
        self.label_4.setText(_translate("Dialog", "Morph SEs", None))
        self.label_5.setText(_translate("Dialog", "(lower to higher)", None))
        self.lineEdit_2.setText(_translate("Dialog", "3", None))
        self.lineEdit_3.setText(_translate("Dialog", "5", None))
        self.lineEdit_4.setText(_translate("Dialog", "7", None))
        self.lineEdit_5.setText(_translate("Dialog", "9", None))
        self.lineEdit_7.setText(_translate("Dialog", "5", None))
        self.label_6.setText(_translate("Dialog", "(lower to higher)", None))
        self.lineEdit_9.setText(_translate("Dialog", "9", None))
        self.label_7.setText(_translate("Dialog", "GLCM windows", None))
        self.lineEdit_6.setText(_translate("Dialog", "3", None))
        self.lineEdit_8.setText(_translate("Dialog", "7", None))
        self.label_8.setText(_translate("Dialog", "Optimization:", None))
        self.comboBox_3.setItemText(0, _translate("Dialog", "No Optimization [faster]", None))
        self.comboBox_3.setItemText(1, _translate("Dialog", "Partially Optimized", None))
        self.comboBox_3.setItemText(2, _translate("Dialog", "Full Optimized [slower]", None))
        self.lineEdit_10.setText(_translate("Dialog", "0.8", None))
        self.label_9.setText(_translate("Dialog", "Training set size:", None))
        self.label_10.setText(_translate("Dialog", "[0-1]", None))
        self.label_11.setText(_translate("Dialog", "Hierarchical Binary Decision Tree", None))
        self.label_12.setText(_translate("Dialog", "Output txt:", None))
        self.pushButton_2.setText(_translate("Dialog", "...", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

