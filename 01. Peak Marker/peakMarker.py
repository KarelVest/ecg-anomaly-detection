import os
import sys
from PyQt6 import QtWidgets, QtCore
import viewerDesign  # Это наш конвертированный файл дизайна
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from datetime import datetime
import pandas as pd
import fastparquet
import numpy as np

class CustomPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            delta = event.angleDelta().y()
            if delta < 0:
                self.setXRange(*[x + 50 for x in self.viewRange()[0]], padding=0)
            elif delta > 0:
                self.setXRange(*[x - 50 for x in self.viewRange()[0]], padding=0)
        else:
            super().wheelEvent(event)

class ExampleApp(QtWidgets.QMainWindow, viewerDesign.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.buttonBrowse.clicked.connect(self.VisualizeDataFile)
        self.graphWidget = CustomPlotWidget(parent=self.centralwidget)
        self.graphWidget.setObjectName("graphWidget")
        self.verticalLayout.insertWidget(0, self.graphWidget)
        self.graphWidget.setBackground('#ffffff')

        self.mark_lines = []
        self.graphWidget.scene().sigMouseClicked.connect(self.mouseClicked)
        self.graphWidget.installEventFilter(self)
    
    def VisualizeDataFile(self):
        self.df = self.ReadDataFile()
        self.series = self.graphWidget.plot(self.df['time'], self.df['value'])
        self.series.setPen(color='#000000', width=2)
        pen = pg.mkPen(color='r', width=2)

        for time in self.df[self.df['edge'] == 1]['time']:
            line = pg.InfiniteLine(time, pen=pen)
            self.graphWidget.addItem(line)
            self.mark_lines.append(line)
    
    def ReadDataFile(self):
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open File","","Parquet Files (*.parquet);;Text Files (*.txt);;All Files (*)", options=options)
        if filePath:
            self.fileName, fileExtension = os.path.splitext(filePath)
            if fileExtension == '.txt':
                df = self.ReadTXT(filePath)
            elif fileExtension == '.parquet':
                df = pd.read_parquet(filePath)

            return df
    
    # def ReadTXT(self, filePath):
    #     with open(filePath, 'r') as file:
    #         lines = file.readlines()

    #     j = -1
    #     values = []
    #     times = []
        
    #     for line in lines:
    #         j += 1
    #         if j == 0:
    #             continue
    #         # if j == 6000:
    #         #     break
    #         value = line
    #         times.append(float(j))
    #         values.append(float(value))

    #     df1 = pd.DataFrame(data=values, columns=['value'])
    #     df2 = pd.DataFrame(data=times, columns=['time'])
    #     df = pd.concat([df1, df2], axis=1)
    #     df['edge'] = 0
    #     return df
        
    def ReadTXT(self, filePath):
        with open(filePath, 'r') as file:
            lines = file.readlines()
        
        j = -1
        values = []
        times = []
        
        for line in lines:
            j += 1
            if j == 0:
                continue
            # if j == 6000:
            #     break
            value = line.split(" ", 1)[0]
            times.append(float(j))
            values.append(float(value))
        
        df1 = pd.DataFrame(data=values, columns=['value'])
        df2 = pd.DataFrame(data=times, columns=['time'])
        df = pd.concat([df1, df2], axis=1)
        df['edge'] = 0
        return df

    def mouseClicked(self, event):
        pos = event.scenePos()
        if self.graphWidget.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.graphWidget.plotItem.vb.mapSceneToView(pos)
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier and event.button() == QtCore.Qt.MouseButton.LeftButton:
                for i, line in enumerate(self.mark_lines):
                    linePos = line.value()
                    if abs(linePos - mousePoint.x()) < 0.3:
                        self.mark_lines.pop(i)
                        self.graphWidget.removeItem(line)
                        clickValue = linePos
                        rowIndex = self.df[self.df['time'] == clickValue].index[0]
                        self.df.loc[rowIndex, 'edge'] = 0
                        break
            else:
                pen = pg.mkPen(color='r', width=5)
                line = pg.InfiniteLine(round(mousePoint.x()), pen=pen)
                self.graphWidget.addItem(line)
                self.mark_lines.append(line)
                clickValue = round(mousePoint.x())
                rowIndex = self.df[self.df['time'] == clickValue].index[0]
                self.df.loc[rowIndex, 'edge'] = 1

    def eventFilter(self, source, event):
        if (source is self.graphWidget):
            if (event.type() == QtCore.QEvent.Type.KeyPress):
                if (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                    if (event.key() == QtCore.Qt.Key.Key_Z):
                        if self.mark_lines:
                            line = self.mark_lines.pop()
                            self.graphWidget.removeItem(line)
                            clickValue = line.value()
                            rowIndex = self.df[self.df['time'] == clickValue].index[0]
                            self.df.loc[rowIndex, 'edge'] = 0
                    
                    elif (event.key() == QtCore.Qt.Key.Key_S):
                        self.df.to_parquet(f'{self.fileName}.parquet')
                        print('Файл сохранён')
                    
                elif (event.key() == QtCore.Qt.Key.Key_A):
                    self.graphWidget.setXRange(*[x - 10 for x in self.graphWidget.viewRange()[0]], padding=0)
                
                elif (event.key() == QtCore.Qt.Key.Key_D):
                    self.graphWidget.setXRange(*[x + 10 for x in self.graphWidget.viewRange()[0]], padding=0)

        return super(ExampleApp, self).eventFilter(source, event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
