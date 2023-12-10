from sklearn.decomposition import PCA
import pandas as pd

# Загрузка данных из файла
data = pd.read_csv('14_16.csv', header=None)

# Создание объекта PCA с параметром svd_solver='full'
pca = PCA(svd_solver='full')

# Обучение модели PCA на данных
pca.fit(data)

# Преобразование данных в новое пространство главных компонент
transformed_data = pca.transform(data)

# Нахождение координаты первого объекта относительно первой главной компоненты
rounded_first_object_coordinates = round(transformed_data[0, 0], 3)  # Координата первого объекта по первой главной компоненте

# Нахождение координаты первого объекта относительно второй главной компоненты
second_component_index = 1  # Индексация начинается с нуля, поэтому вторая компонента имеет индекс 1
first_object_coordinates_second_component = transformed_data[0, second_component_index]  # Координата первого объекта по второй главной компоненте

# Округление до тысячных
rounded_coordinates = round(first_object_coordinates_second_component, 3)

# Получение доли объясненной дисперсии для первых двух главных компонент
explained_variance_ratio = pca.explained_variance_ratio_

# Суммирование долей объясненной дисперсии для первых двух компонент
explained_variance_ratio_2_components = sum(explained_variance_ratio[:2])

# Округление до тысячных
rounded_explained_variance_ratio = round(explained_variance_ratio_2_components, 3)

# Получение долей объясненной дисперсии для каждой главной компоненты
explained_variance_ratio = pca.explained_variance_ratio_

# Вычисление минимального количества главных компонент, чтобы доля объясненной дисперсии превышала 0.85
cumulative_variance = 0.0
num_components = 0

for i in range(len(explained_variance_ratio)):
    cumulative_variance += explained_variance_ratio[i]
    num_components += 1
    if cumulative_variance > 0.85:
        break

print(f"Минимальное количество главных компонент: {num_components}")
print(f"Доля объясненной дисперсии при использовании первых двух главных компонент: {rounded_explained_variance_ratio}")
print(f"Координата первого объекта относительно второй главной компоненты: {rounded_coordinates}")
print(f"Координата первого объекта относительно первой главной компоненты: {rounded_first_object_coordinates}")

