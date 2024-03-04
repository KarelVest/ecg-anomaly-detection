# Импорт библиотек
import pandas as pd
import os
from math import pi
import torch
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch import nn
import numpy as np
import fastparquet
import tkinter as tk
from tkinter import filedialog

class EcgLstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(EcgLstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer
        # Если LSTM двунаправленный, то умножаем hidden_size на 2 для конкатенации скрытых состояний из обоих направлений
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        
        # Sigmoid layer для получения вероятности
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Инициализация скрытых состояний и состояний ячейки с нулями
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Преобразование выхода LSTM для предсказания вероятности
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

# Параметры для создания модели
input_size = 1
hidden_size = 64
num_layers = 1

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    root = tk.Tk()
    root.withdraw()

    #filePath = filedialog.askopenfilename(filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")])
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

    values_tensor = torch.tensor(df['value'].values.astype('float32')).unsqueeze(1)      # Размерность: torch.Size([5999, 1])
    print(values_tensor.shape)
    num_segments = len(values_tensor) // 200
    new_size = (num_segments + 1) * 200
    padding = new_size - len(values_tensor)
    if padding > 0:
        values_tensor = torch.cat((values_tensor, torch.zeros(padding, 1)), dim=0)
    print(values_tensor.shape)

    # Теперь values_tensor имеет размер кратный 200 и можно безопасно использовать view
    segments_tensor = values_tensor.view(-1, 200, 1)  # Разбиваем наш одномерный массив точек на массив сегментов по 200 точек

    # Создадим DataLoader без меток, так как нам нужны только предсказания
    test_dataset = TensorDataset(segments_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Подаем данные на вход нейросети и получаем предсказания

    # Загрузка структуры модели (необходимо знать класс модели)
    model = EcgLstmNet(input_size, hidden_size, num_layers)     # Кол-во входных элементов, кол-во нейронов lstm-в слое и количество слоёв
    # Загрузка весов модели
    model.load_state_dict(torch.load('.workspace/Models/FirstStageModel.2.pt', map_location=device))      # Загружаем обученную модель
    model.to(device)
    model.eval()  # Перевод модели в режим оценки
    predictions = []
    with torch.no_grad():  # Отключаем вычисление градиентов
        for inputs in test_dataloader:
            inputs = inputs[0].to(device)  # DataLoader возвращает кортеж, берем только данные
            outputs = model(inputs)
            predictions.append(outputs.squeeze().cpu())  # Удаляем измерения batch и input_size

    # Объединяем предсказания из всех батчей и округляем их до 0 или 1
    def configurableRound(tensor, threshold=0.5):
        return torch.where(tensor % 1 >= threshold, torch.ceil(tensor), torch.floor(tensor))      

    predictions_tensor = configurableRound(torch.cat(predictions), 0.3).flatten()

    # Создаем новый DataFrame с двумя столбцами: значения и метки
    dfResult = pd.DataFrame({
        'value': values_tensor.squeeze(),   #! Вместо этого по-хорошему надо вставить столбец dfOriginal['value'], чтобы оставить исходные значения, а не нормализованные
        'edge': predictions_tensor
    })

    # Добавляем массив временных меток от 0 до количества точек в DataFrame
    dfResult['time'] = np.arange(len(dfResult)).astype('float')

    # Выведем результат
    print(dfResult)


    dfResult.to_parquet(f'{fileName}.C.parquet')
    print('Файл сохранён')

if __name__ == '__main__':
    main()

#* 24.02.24 Были внесены минорные косметические изменения кода, а также проверена работоспособность (проблем нет) и совместимость с
#* планируемой softmax-разметкой данных для второй нейросети (совместимость полная)
