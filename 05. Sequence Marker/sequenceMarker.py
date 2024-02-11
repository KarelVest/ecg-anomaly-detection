import os
import sys
from PyQt6 import QtWidgets, QtCore
import viewerDesign  # Это наш конвертированный файл дизайна
from pyqtgraph import PlotWidget, plot
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.graphicsItems.FillBetweenItem import FillBetweenItem
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

        self.redFills = []
        self.redValues = []

        self.graphWidget.scene().sigMouseClicked.connect(self.mouseClicked)
        self.graphWidget.installEventFilter(self)
    
    def VisualizeDataFile(self):
        self.df = self.ReadDataFile()
        self.series = self.graphWidget.plot(self.df['time'], self.df['value'])
        _, yRange = self.graphWidget.viewRange()
        self.down = yRange[1] * (-0.1)
        self.up = yRange[1] * (1.1)
        self.series.setPen(color='#000000', width=2)

        self.edgeIndices = self.df[self.df['edge'] > 0]['time'].to_list()
        self.redIndices = self.df[self.df['edge'] == 3]['time'].to_list()

        for i in range(len(self.edgeIndices) - 1):
            # Если длина отрезка больше 200, красим в оранжевый
            if self.edgeIndices[i + 1] - self.edgeIndices[i] > 200:
                fill_x = np.linspace(int(self.edgeIndices[i]), int(self.edgeIndices[i + 1]), int((self.edgeIndices[i + 1] - self.edgeIndices[i])*1))
                fill_y1 = np.full_like(fill_x, self.up)
                fill_y2 = np.full_like(fill_x, self.down)

                # Заполняем пространство между двумя точками
                fill = pg.FillBetweenItem(pg.PlotDataItem(fill_x, fill_y1), pg.PlotDataItem(fill_x, fill_y2), brush=(255,165,0,50))
                self.graphWidget.addItem(fill)
                self.df.at[self.edgeIndices[i], 'edge'] = 2
        
        for i in range(len(self.redIndices)):
            idx = np.searchsorted(self.edgeIndices, self.redIndices[i])
            if self.edgeIndices[idx+1]:
                fill_x = np.linspace(int(self.redIndices[i]), int(self.edgeIndices[idx + 1]), int((self.edgeIndices[idx + 1] - self.redIndices[i])*1))
                fill_y1 = np.full_like(fill_x, self.up)
                fill_y2 = np.full_like(fill_x, self.down)

                # Заполняем пространство между двумя точками
                fill = pg.FillBetweenItem(pg.PlotDataItem(fill_x, fill_y1), pg.PlotDataItem(fill_x, fill_y2), brush=(255,0,0,50))
                self.graphWidget.addItem(fill)
                self.redValues.append(self.redIndices[i])
                self.redFills.append(fill)
    
    def ReadDataFile(self):
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        #fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open File","","Parquet Files (*.parquet);;All Files (*)", options=options)
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open File","","Parquet Files (*.parquet)", options=options)
        if filePath:
            self.fileName, fileExtension = os.path.splitext(filePath)
            if fileExtension == '.parquet':
                df = pd.read_parquet(filePath)

            return df

    def mouseClicked(self, event):
        pos = event.scenePos()
        if self.graphWidget.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.graphWidget.plotItem.vb.mapSceneToView(pos)
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier and event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Находим индекс в массиве, где можно вставить num, чтобы сохранить порядок сортировки
                idx = np.searchsorted(self.edgeIndices, round(mousePoint.x()))

                # Проверяем, не является ли idx первым или последним элементом массива
                if idx == 0:
                    print("Число меньше всех чисел в массиве")
                elif idx == len(self.edgeIndices):
                    print("Число больше всех чисел в массиве")
                else:
                    rowIndex = self.df[self.df['time'] == self.edgeIndices[idx-1]].index[0]
                    if (self.df.loc[rowIndex, 'edge'] == 3):
                        ind = np.where(np.isclose(self.redValues, self.edgeIndices[idx-1], atol=1e-3))[0][0]
                        self.graphWidget.removeItem(self.redFills[ind])
                        self.redValues.pop(ind)
                        self.redFills.pop(ind)
                        self.df.loc[rowIndex, 'edge'] = 1
            
            else:
                # Находим индекс в массиве, где можно вставить num, чтобы сохранить порядок сортировки
                idx = np.searchsorted(self.edgeIndices, round(mousePoint.x()))

                # Проверяем, не является ли idx первым или последним элементом массива
                if idx == 0:
                    print("Число меньше всех чисел в массиве")
                elif idx == len(self.edgeIndices):
                    print("Число больше всех чисел в массиве")
                else:
                    rowIndex = self.df[self.df['time'] == self.edgeIndices[idx-1]].index[0]
                    if (self.df.at[rowIndex, 'edge'] < 2):
                        self.df.at[rowIndex, 'edge'] = 3

                        fill_x = np.linspace(int(self.edgeIndices[idx-1]), int(self.edgeIndices[idx]), int((self.edgeIndices[idx] - self.edgeIndices[idx-1])*1))
                        fill_y1 = np.full_like(fill_x, self.up)
                        fill_y2 = np.full_like(fill_x, self.down)

                        # Заполняем пространство между двумя точками
                        fill = pg.FillBetweenItem(pg.PlotDataItem(fill_x, fill_y1), pg.PlotDataItem(fill_x, fill_y2), brush=(255,0,0,50))
                        self.graphWidget.addItem(fill)
                        self.redFills.append(fill)
                        self.redValues.append(self.edgeIndices[idx-1])

    def eventFilter(self, source, event):
        if (source is self.graphWidget):
            if (event.type() == QtCore.QEvent.Type.KeyPress):
                if (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                    if (event.key() == QtCore.Qt.Key.Key_Z):
                        # Тут будет удаление последнего размеченного отрезка
                        if self.redFills:
                            fill = self.redFills.pop()
                            self.graphWidget.removeItem(fill)
                            rowIndex = self.df[self.df['time'] == self.redValues[-1]].index[0]
                            self.redValues.pop()
                            self.df.loc[rowIndex, 'edge'] = 1
                    
                    elif (event.key() == QtCore.Qt.Key.Key_S):
                        if ".V" in self.fileName:
                            self.df.to_parquet(f'{self.fileName}.parquet')
                        else:
                            self.df.to_parquet(f'{self.fileName}.V.parquet')
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
