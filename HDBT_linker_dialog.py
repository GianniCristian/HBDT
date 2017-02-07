# -*- coding: utf-8 -*-
"""
/***************************************************************************
 HDBTDialog
                                 A QGIS plugin
 Compute the HDBT
                             -------------------
        begin                : 2015-09-30
        git sha              : $Format:%H$
        copyright            : (C) 2015 by UniPV
        email                : giannicristian.iannelli@unipv.it
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

from PyQt4 import QtGui, uic
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui_progress import Ui_Progress 

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'HDBT_linker_dialog_base.ui'))

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class ProgressDialog(QtGui.QDialog, Ui_Progress):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.ui = Ui_Progress()
        self.ui.setupUi(self)
        self.ui.progressBar.setValue( 0 )
        
class HDBTDialog(QtGui.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(HDBTDialog, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        QObject.connect(self.comboBox, SIGNAL("activated(const QString&)"), self.setPath_input)
        QObject.connect(self.comboBox_2, SIGNAL("activated(const QString&)"), self.setPath_training)
        QObject.connect(self.pushButton, SIGNAL("clicked()"), self.setPath_output)
        QObject.connect(self.pushButton_2, SIGNAL("clicked()"), self.setPath_output_2)

    def setPath_input(self):
        if self.comboBox.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/");
            if fileName !="":
                self.comboBox.setItemText(0, _translate("Input Image", "[Choose from a file..] "+fileName, None))
            else:
                self.comboBox.setItemText(0, _translate("Input Image", "[Choose from a file..]", None))
    def setPath_training(self):
        if self.comboBox_2.currentIndex() == 0:
            fileName = QFileDialog.getOpenFileName(self,"Open Image", "~/");
            if fileName !="":
                self.comboBox_2.setItemText(0, _translate("Training Image", "[Choose from a file..] "+fileName, None))
            else:
                self.comboBox_2.setItemText(0, _translate("Training Image", "[Choose from a file..]", None))
    def setPath_output(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Image", "~/","Image Files (*.tiff *.tif)");
        if fileName !="":
            self.lineEdit.setText(fileName)
    def setPath_output_2(self):
        fileName = QFileDialog.getSaveFileName(self,"Output Text File", "~/");
        if fileName !="":
            self.lineEdit_1.setText(fileName)
