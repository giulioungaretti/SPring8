from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

def plot_window(title="data", res=(1000, 1000)):
    """
    create a qt window window with a plot widget  and a customizable 
    button for some action (a global variable called update_on_off_win2 )
    """
    global update_on_off_win2
    app = pg.mkQApp()
    win2 = QtGui.QMainWindow()
    win2.resize(res[0], res[1])
    win2.setWindowTitle(title)
    cw = QtGui.QWidget()
    win2.setCentralWidget(cw)
    l = QtGui.QGridLayout()
    cw.setLayout(l)
    btn = pg.QtGui.QPushButton("update")
    l.addWidget(btn,  2, 0, 1, 1)
    return win2, l, btn
