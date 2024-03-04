import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Чтение файла с метками из формата Parquet в DataFrame
    df = pd.read_parquet('../00. Resources/markedData.parquet')

    # Нахождение индексов точек, у которых метка равна 1
    mark_indices = df[df['edge'] == 1].index

    segments = []
    segmentLength = 200
    k = 0
    prevIndex = -1

    for i in range(1, len(mark_indices)):
        prevIndex = mark_indices[i-1]
        index = mark_indices[i]

        for start in range(prevIndex + 1, index, 5):
            if start >= 0 and start + segmentLength <= len(df):
                segment = df.iloc[start : start + segmentLength]

                # Проверяем, есть ли в сегменте метка edge == 1
                if segment['edge'].eq(1).any():
                    segments.append(segment)
                else:
                    # Если метки нет, прекращаем добавление сегментов для этой метки
                    break
                #! Можно улучшить. Например, сделать так, чтобы алгоритм пропускал все пустые отрезки до метки, так первая метка не будет потеряна (тут есть и плюсы, и минусы).
                #! Также можно добавить обработку отрезков без меток, но тогда придётся немного поменять концепцию работы нейросети

    print(len(segments))

    # Объединение всех DataFrame в один
    all_segments_df = pd.concat(segments, ignore_index=True)

    print(len(all_segments_df))

    # Сохранение объединенного DataFrame в файл .parquet
    all_segments_df.to_parquet('../00. Resources/dataset.parquet')

    fig, ax = plt.subplots()

    # Функция для отрисовки сегмента
    def draw_segment(i):
        ax.clear()
        segment = segments[i]
        ax.plot(segment['time'], segment['value'])
        for time in segment[segment['edge'] == 1]['time']:
            ax.axvline(x=time, color='r')

    # Создание анимации
    ani = animation.FuncAnimation(fig, draw_segment, frames=len(segments), interval=100)

    plt.show()

if __name__ == '__main__':
    main()
#* Скорее всего уже не понадобится, так как нейросеть достаточно обучена и может сама размечать для себя реальные обучающие данные, которые
#* нужно будет лишь немного поправить
