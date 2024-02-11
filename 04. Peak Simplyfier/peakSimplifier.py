import os
import sys
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from datetime import datetime
import pandas as pd
import fastparquet
import numpy as np
import tkinter as tk
from tkinter import filedialog

def SimplifyData(df, fileName):
    # Выделение строк, где 'edge' равно 1
    eO = df[df['edge'] == 1]
    eO['timeDiff'] = eO['time'].diff()
    eO['group'] = (eO['timeDiff'] > 5).astype(int).cumsum()

    # Создание столбца 'group' в df и обновление его значений
    df.loc[df['edge'] == 1, 'group'] = eO['group']
    for name, group in eO.groupby('group'):
        # Находим индекс первого максимального значения
        max_index = group['value'].idxmax()
        # Сбрасываем edge для всех, кроме самой левой
        df.loc[(df['group'] == name) & (df.index != max_index), 'edge'] = 0

    df.to_parquet(f'{fileName}.S.parquet')
    print('Файл сохранён')


def main():
    root = tk.Tk()
    root.withdraw()

    #file_path = filedialog.askopenfilename(filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")])
    filePath = filedialog.askopenfilename(filetypes=[("Parquet Files", "*.parquet")])

    if filePath:
        print("Выбранный файл:", filePath)
        fileName, fileExtension = os.path.splitext(filePath)
        if fileExtension == '.parquet':
            dfOriginal = pd.read_parquet(filePath)
            df = dfOriginal.copy()
            SimplifyData(df, fileName)
        else:
            print("Выбран неверный файл")
            exit()
    else:
        print("Выбор файла отменен.")
        exit()

if __name__ == '__main__':
    main()
