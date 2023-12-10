from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = pd.read_csv('14_16.csv', header=None)

# Создание объекта PCA с параметром svd_solver='full' и двумя компонентами
pca = PCA(n_components=2, svd_solver='full')

# Преобразование данных в новое пространство с двумя главными компонентами
transformed_data = pca.fit_transform(data)

# Выполнение кластеризации для разного числа кластеров и оценка внутригрупповой дисперсии
wcss = []  # within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(transformed_data)
    wcss.append(kmeans.inertia_)

# Построение графика метода локтя
plt.plot(range(1, 11), wcss)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Within-Cluster Sum of Squares')
plt.show()
