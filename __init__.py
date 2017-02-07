# -*- coding: utf-8 -*-
"""
/***************************************************************************
 HDBT
                                 A QGIS plugin
 Compute the HDBT
                             -------------------
        begin                : 2015-09-30
        copyright            : (C) 2015 by UniPV
        email                : giannicristian.iannelli@unipv.it
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load HDBT class from file HDBT.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .HDBT_linker import HDBT
    return HDBT(iface)
