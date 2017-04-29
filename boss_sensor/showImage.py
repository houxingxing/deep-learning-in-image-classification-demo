__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import sys

from PyQt5.QtWidgets import QApplication,QLabel
from PyQt5 import QtGui


def show_picture(image_path='1.png'):
    app = QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QLabel()
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())


