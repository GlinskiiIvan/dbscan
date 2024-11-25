import os
import math
from PIL import Image
import numpy as np
from collections import defaultdict

# Функция для вычисления евклидова расстояния
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Функция для поиска соседей в пределах eps
def get_neighbors(data, point_idx, eps):
    neighbors = []
    for idx, point in enumerate(data):
        if euclidean_distance(data[point_idx], point) <= eps:
            neighbors.append(idx)
    return neighbors

# Алгоритм DBSCAN
def dbscan(data, eps, min_samples):
    labels = [0] * len(data)  # -1: шум, 0: не посещено, >0: кластер
    cluster_id = 0

    for i in range(len(data)):
        if labels[i] != 0:
            continue
        neighbors = get_neighbors(data, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)
    return labels

# Расширение кластера
def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = get_neighbors(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
        i += 1

# Функция для вычисления яркости, контраста и шумов
def compute_image_features(image_path):
    img = Image.open(image_path).convert("L")  # Преобразуем в градации серого
    img_array = np.array(img)

    # Яркость: среднее значение интенсивности пикселей
    brightness = np.mean(img_array)

    # Контраст: стандартное отклонение интенсивности пикселей
    contrast = np.std(img_array)

    # Шумы: среднее значение абсолютной разницы между соседними пикселями
    noise = np.mean(np.abs(np.diff(img_array, axis=0))) + np.mean(np.abs(np.diff(img_array, axis=1)))

    return brightness, contrast, noise

# Сканирование директории и выполнение кластеризации
def cluster_images(directory, eps, min_samples):
    data_points = []
    image_paths = []

    # Сканируем директорию
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                image_path = os.path.join(root, file)
                try:
                    features = compute_image_features(image_path)
                    data_points.append(features)
                    image_paths.append(image_path)
                except Exception as e:
                    print(f"Ошибка при обработке {image_path}: {e}")

    # Выполняем кластеризацию
    labels = dbscan(data_points, eps, min_samples)

    # Группируем изображения по кластерам
    clusters = defaultdict(list)
    for label, path in zip(labels, image_paths):
        clusters[label].append(path)

    return clusters

# Сохранение изображений в директории кластеров
def save_clusters(clusters, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for cluster_id, paths in clusters.items():
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        cluster_dir = os.path.join(output_directory, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        for path in paths:
            file_name = os.path.basename(path)
            new_path = os.path.join(cluster_dir, file_name)
            try:
                Image.open(path).save(new_path)
            except Exception as e:
                print(f"Не удалось сохранить {path} в {cluster_dir}: {e}")

# Основная программа
directory = input("Введите путь к директории с PNG изображениями: ").strip().strip("'\"")
if not os.path.isdir(directory):
    print("Указанная директория не существует.")
    exit()

eps = float(input("Введите значение eps (радиус для кластеризации): "))
min_samples = int(input("Введите минимальное количество точек для кластера (min_samples): "))

clusters = cluster_images(directory, eps, min_samples)

output_directory = os.path.join(directory, "clustered_images")
save_clusters(clusters, output_directory)

print(f"Изображения сохранены в директории: {output_directory}")

# for cluster_id, paths in clusters.items():
#     if cluster_id == -1:
#         print("\nШумовые изображения:")
#     else:
#         print(f"\nКластер {cluster_id}:")
#     for path in paths:
#         print(f"  {path}")
