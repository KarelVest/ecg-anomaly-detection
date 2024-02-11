# Импорт библиотек
import pandas as pd
import os
import random
from math import pi
import torch
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import fastparquet
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog

class SecondEcgDataset(Dataset):
    def __init__(self, df, estimatedIndexes):
        self.estimatedIndexes = estimatedIndexes
        self.df = df
        self.segmentStarts = df.index[df['edge'] > 0].tolist()
        self.startsCount = len(self.segmentStarts)

        self.cachedTensors = []
        self._get_segment_tensor()
        self.segmentsCount = len(self.cachedTensors)
    
    def _get_segment_tensor(self):
        for index in range(self.startsCount):
            result = 0
            if index + 1 < len(self.segmentStarts):
                if self.df.at[self.segmentStarts[index], 'edge'] < 2:
                    segment = self.df.iloc[self.segmentStarts[index]:self.segmentStarts[index+1]]
                    result = 1

            if result:
                self.estimatedIndexes.append(self.segmentStarts[index])

                segmentValues = segment['value']
                segmentTensor = torch.tensor(segmentValues.to_numpy(), dtype=torch.float).unsqueeze(1)
                padding = 200 - len(segmentTensor)
                if padding > 0:
                    # print(padding)
                    segmentTensor = torch.cat((segmentTensor, torch.zeros(padding, 1)), dim=0)
                elif padding < 0:
                    print('В данные зашёл сегмент неверной длины: ', self.segmentStarts[index], ' = ', self.df.at[self.segmentStarts[index], 'edge'])
                self.cachedTensors.append(segmentTensor)

    def __len__(self):
        return self.segmentsCount

    def __getitem__(self, index):
        segmentTensor = self.cachedTensors[index]
        return segmentTensor
    
#* Модель нейронной сети
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=128,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        output, _ = self.encoder(x)

        # Decoder
        output, _ = self.decoder(output)

        # Sigmoid
        output= self.sigmoid(output)

        return output

# Параметры для создания модели
input_size = 1
hidden_size = 64
num_layers = 1

def main():
    device = ("cpu")

    root = tk.Tk()
    root.withdraw()

    filePath = filedialog.askopenfilename(filetypes=[("Parquet Files", "*.parquet")])
    if filePath:
        print("Выбранный файл:", filePath)
        fileName, fileExtension = os.path.splitext(filePath)
        if fileExtension == '.parquet':
            dfOriginal = pd.read_parquet(filePath)
            df = dfOriginal.copy()
        else:
            print("Выбран неверный файл")
            exit()
    else:
        print("Выбор файла отменен.")
        exit()


    df['value'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())       # Нормализация данных

    indexes = []
    testDataset = SecondEcgDataset(df, indexes)
    print(len(testDataset))
    testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=10, shuffle=False)

    model = LSTMAutoencoder()
    criterion = nn.MSELoss()

    # Загрузка весов модели
    model.load_state_dict(torch.load('../00. Resources/Models/SecondStageModel.pt', map_location=device))
    model.to(device)
    model.eval()  # Перевод модели в режим оценки
    sequences = []
    predictions = []
    answers = []

    with torch.no_grad():  # Отключаем вычисление градиентов
        for inputs in testDataloader:
            inputs = inputs.to(device)  # DataLoader возвращает кортеж, берем только данные
            outputs = model(inputs)
            print(outputs.shape)
            
            # loss = criterion(outputs, inputs)
            losses = torch.mean(outputs - inputs, dim=(1, 2))
            
            for i in range(len(losses)):
                sequences.append(inputs[i].squeeze().numpy())
                predictions.append(outputs[i].squeeze().numpy())
                answers.append(losses[i].item())

            #print(outputs.squeeze().cpu().shape)

    #print(predictions)

    fig, ax = plt.subplots()
    coord = np.arange(200)

    # Функция для отрисовки сегмента
    def draw_segment(i):
        ax.clear()
        ax.plot(coord, sequences[i])
        ax.plot(coord, predictions[i])

    # Создание анимации
    ani = animation.FuncAnimation(fig, draw_segment, frames=len(sequences), interval=2000)

    plt.show()


    # dfResult.to_parquet(f'{fileName}.C.parquet')
    # print('Файл сохранён')

if __name__ == '__main__':
    main()

#! Не забыть про необработку последнего участка, доделать
