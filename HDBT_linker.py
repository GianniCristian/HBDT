# -*- coding: utf-8 -*-
"""
/***************************************************************************
 HDBT
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
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from qgis.core import *
from PyQt4 import QtCore, QtGui
# Initialize Qt resources from file resources.py
import resources_rc
# Import the code for the dialog
from HDBT_linker_dialog import *
import os.path
import subprocess

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

def executeScript(command, progress=None,noerror=True):
    if os.name != "posix" and noerror:
        # Found at http://stackoverflow.com/questions/5069224/handling-subprocess-crash-in-windows
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen's response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
        subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
    else:
        subprocess_flags = 0

    command = (os.path.dirname(os.path.abspath(__file__))+command if os.name == "posix" else 'C:/Python27/python.exe "'+os.path.dirname(os.path.abspath(__file__))+command)
    QMessageBox.information(None, "Info", command)

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess_flags,
        universal_newlines=True,
        ).stdout
    
    for line in iter(proc.readline, ''):
        if '[*' in line:
            idx = line.find('[*')
            perc = line[idx:(idx+102)].count("*")
            status = line[line.find('STATUS: ')+8:idx]
            print status
            if perc != 0 and progress:
                progress.progressBar.setValue(perc)
                progress.label_title.setText(status)
        QtGui.qApp.processEvents()
    

def parse_input(string):
    return string.replace("[Choose from a file..] ","")

class HDBT:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'HDBT_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = HDBTDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&HBDTree')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'HBDT')
        self.toolbar.setObjectName(u'HBDT')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('HBDT', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/HDBT/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'HBDT'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&HBDTree'),
                action)
            self.iface.removeToolBarIcon(action)
    
    def changeActive(self,comboBox):
        comboBox.clear()
        comboBox.addItem(_fromUtf8(""))
        comboBox.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
        current_layer = self.iface.mapCanvas().currentLayer()
        layers = self.iface.mapCanvas().layers()
        for i,layer in enumerate(layers):
            #if layer.type() == "RasterLayer":
            path = str(layer.dataProvider().dataSourceUri()).replace("|layerid=0","")
            comboBox.addItem(_fromUtf8(""))
            comboBox.setItemText(i+1, _translate("Pansharp", path, None))
    
    def run(self):
        """Run method that performs all the real work"""
        # Create the dialog (after translation) and keep reference
        self.dlg_hdbt = HDBTDialog()
        self.changeActive(self.dlg_hdbt.comboBox)
        self.changeActive(self.dlg_hdbt.comboBox_2)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_hdbt.comboBox))
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_hdbt.comboBox_2))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_hdbt.show()
        # Run the dialog event loop
        result = self.dlg_hdbt.exec_()
        # See if OK was pressed
        if result:
            ui = self.dlg_hdbt
            dlgProgress.show()
            input_image = parse_input(str(ui.comboBox.currentText()))
            input_training = parse_input(str(ui.comboBox_2.currentText()))
            output_class = str(ui.lineEdit.text())
            output_txt = str(ui.lineEdit_1.text())
            test_size = float(ui.lineEdit_10.text())
            #Morph
            morph_se_1 = int(ui.lineEdit_2.text())
            morph_se_2 = int(ui.lineEdit_3.text())
            morph_se_3 = int(ui.lineEdit_4.text())
            morph_se_4 = int(ui.lineEdit_5.text())
            #morph = [morph_se_1,morph_se_2,morph_se_3,morph_se_4]
            #GLCM
            glcm_wk_1 = int(ui.lineEdit_6.text())
            glcm_wk_2 = int(ui.lineEdit_7.text())
            glcm_wk_3 = int(ui.lineEdit_8.text())
            glcm_wk_4 = int(ui.lineEdit_9.text())
            #glcm = [glcm_wk_1,glcm_wk_2,glcm_wk_3,glcm_wk_4]
            optimization_str = parse_input(str(ui.comboBox_3.currentText()))
            if optimization_str == 'No Optimization [faster]':
                optimization = 0
            elif optimization_str == 'Partially Optimized':
                optimization = 1
            else:
                optimization = 2
            executeScript('/HDBT_Code/HybridDecisionTree_2509.py" --morph_win \"{}\" \"{}\" \"{}\" \"{}\" --glcm_win \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"'.format(morph_se_1,morph_se_2,morph_se_3,morph_se_4,
                                                                                                                                                     glcm_wk_1,glcm_wk_2,glcm_wk_3,glcm_wk_4,
                                                                                                                                                     input_image,input_training,output_class,output_txt,test_size,optimization),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsRasterLayer(output_class, QFileInfo(output_class).baseName()))
            QMessageBox.information(None, "Info", 'Done!')
            pass
