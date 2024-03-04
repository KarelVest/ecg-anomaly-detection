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
import torch.nn.functional as F

class SecondEcgDataset(Dataset):
    def __init__(self, df, estimatedIndexes):
        self.estimatedIndexes = estimatedIndexes
        self.df = df
        self.segmentStarts = df.index[df['edge'] > 0].tolist()
        self.startsCount = len(self.segmentStarts)

        self.cachedTensors = []
        self.cachedLabels = []
        self._get_segment_tensor()
        self.segmentsCount = len(self.cachedTensors)
    
    def _get_segment_tensor(self):
        for index in range(self.startsCount):
            result = 0
            if index + 1 < len(self.segmentStarts):
                if self.df.at[self.segmentStarts[index], 'edge'] != 2:      # По сути условие, что длина отрезка меньше 200
                    segment = self.df.iloc[self.segmentStarts[index]:self.segmentStarts[index+1]]
                    result = 1
            elif self.df.at[self.segmentStarts[index], 'edge'] != 2:        #! Возможно это нужно будет убрать, т.к. тут в выборку может попасть отрезок длиной больше 200
                segment = self.df.iloc[self.segmentStarts[index]:]
                result = 1

            if result:
                self.estimatedIndexes.append(self.segmentStarts[index])

                segmentValues = segment['value']
                labelValue = (1 if segment.iloc[0]['edge'] > 2.0 else 0)
                segmentTensor = torch.tensor(segmentValues.to_numpy(), dtype=torch.float).unsqueeze(1)
                # print(segmentTensor.shape)
                labelTensor = torch.tensor(labelValue, dtype=torch.float).unsqueeze(0)
                # print(labelTensor.shape)
                padding = 200 - len(segmentTensor)
                if padding > 0:
                    segmentTensor = torch.cat((segmentTensor, torch.zeros(padding, 1)), dim=0)
                elif padding < 0:
                    print('В данные зашёл сегмент неверной длины')
                    exit()
                self.cachedTensors.append(segmentTensor)
                self.cachedLabels.append(labelTensor)

    def __len__(self):
        return self.segmentsCount

    def __getitem__(self, index):
        #segmentTensor = self.cachedTensors[index]
        #return segmentTensor
        return self.cachedTensors[index], self.cachedLabels[index]
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, seq_len=200, bidirectional=True):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        add_size = 2*hidden_size if bidirectional else hidden_size

        # LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            #dropout=0.25
        )

        self.lstm2 = nn.LSTM(
            input_size=add_size,
            hidden_size=hidden_size//2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            #dropout=0.25
        )

        self.lstm3 = nn.LSTM(
            input_size=add_size//2,
            hidden_size=hidden_size//4,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            #dropout=0.25
        )

        #self.resulter = nn.Linear(hidden_size//2 if bidirectional else hidden_size//4, 1)
        self.resulter = nn.Linear(seq_len * add_size//4, 1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # h1 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        # c1 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        # h2 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size//2).to(x.device)
        # c2 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size//2).to(x.device)

        # h3 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size//4).to(x.device)
        # c3 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size//4).to(x.device)

        # Encoder
        output, _ = self.lstm1(x)#, (h1, c1))       #Output shape: [1, 200, 128]   [batch_size, seq_len, num_directions * hidden_size]
        #output = F.dropout(output, p=0.25, training=self.training)
        output, _ = self.lstm2(output)#, (h2, c2))
        #output = F.dropout(output, p=0.25, training=self.training)
        # _, (hidden, _) = self.lstm3(output)#, (h0, c0))
        output, _ = self.lstm3(output)#, (h3, c3))
        #output = F.dropout(output, p=0.25, training=self.training)
        # print(torch.cat([hidden[-2], hidden[-1]], dim=1).shape)

        # Resulter
        # output = self.resulter(torch.cat([hidden[-2], hidden[-1]], dim=1))      #! Надо попробовать использовать не Hidden, а Output
        batch_size, seq_len, _ = output.size()
        output = output.reshape(batch_size, -1)
        output = self.resulter(output)      #! Надо попробовать использовать не Hidden, а Output
        # Надо, чтобы он принимал 32*200 значений
        # print(output.shape)

        # Sigmoid
        output = self.sigmoid(output) 

        return output

# Параметры для создания модели
input_size = 1 # Так как ЭКГ - одномерный сигнал
hidden_size = 1024 # Можно изменить в зависимости от сложности задачи и размера данных
num_layers = 1 # Можно изменить в зависимости от сложности задачи

# Создание экземпляра модели
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

df.loc[df['edge'] > 2.0, 'edge'] = 1.0
df['value'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())       # Нормализация данных

indexes = []
testDataset = SecondEcgDataset(df, indexes)
#trainDataset, valDataset, testDataset = random_split(dataset, [int(0.6 * len(dataset)), int(0.2 * len(dataset)), len(dataset) - int(0.6 * len(dataset)) - int(0.2 * len(dataset))])
#print(len(testDataset))
#print(indexes)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=10, shuffle=False)

model = LSTMAutoencoder(input_size, hidden_size, num_layers, 200, True)
criterion = nn.BCELoss()

# Загрузка весов модели
model.load_state_dict(torch.load('.workspace/Models/SecondStageModel.53.pt', map_location=device))
model.to(device)
model.eval()  # Перевод модели в режим оценки
answers = []

with torch.no_grad():  # Отключаем вычисление градиентов
    for input, _ in testDataloader:
        input = input.to(device)  # DataLoader возвращает кортеж, берем только данные
        output = model(input)
        answers.append(output.squeeze().cpu())

print(len(testDataset))
print(len(answers))
predictions_tensor = torch.cat(answers).flatten()
print(len(predictions_tensor))

# print(len(indexes))

# def AverageBetweenMaximals(arr, percent):
#     # Шаг 1: Сортировка массива
#     sorted_arr = sorted(arr, reverse=True)
    
#     # Шаг 2: Выбор первых 80% минимальных элементов
#     eighty_percent_index = int(percent * len(sorted_arr))
#     max_values = sorted_arr[:eighty_percent_index]
    
#     # Шаг 3: Рассчет среднего значения
#     average = sum(max_values) / len(max_values)
    
#     return average

# print(AverageBetweenMaximals(answers, 0.1))
# print(max(answers))

#!-----------------------------------------------------
dfOriginal.loc[dfOriginal['edge'] > 2.0, 'edge'] = 1.0

for i in range(len(testDataset)):
    if (predictions_tensor[i] > 0.5):
        df.at[indexes[i], 'edge'] = 3.0

df.to_parquet(f'{fileName}.R.parquet')
print(torch.max(predictions_tensor))
print('Файл сохранён')
#!-----------------------------------------------------

# fig, ax = plt.subplots()
# coord = np.arange(200)

# # Функция для отрисовки сегмента
# def draw_segment(i):
#     ax.clear()
#     ax.plot(coord, sequences[i])
#     ax.plot(coord, predictions[i])

# # Создание анимации
# ani = animation.FuncAnimation(fig, draw_segment, frames=len(sequences), interval=1000)

# plt.show()
