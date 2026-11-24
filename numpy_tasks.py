"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


# Задача 1. Сумма произведений матриц на векторы
def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = np.asarray(data.matrices), np.asarray(data.vectors)
    # Оператор @ умножает каждую матрицу на свой вектор, а sum(..., axis=0) складывает результаты
    return np.sum(matrices @ vectors, axis=0)


# Задача 2. Бинаризация матрицы (превращаем числа в 0 и 1 по порогу)
def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = np.asarray(data.matrix), data.threshold
    # Сравниваем каждый элемент с порогом (получаем True/False) и переводим в 1 и 0 через astype(int)
    return (matrix > threshold).astype(int)


# Задача 3 (часть 1). Уникальные элементы для каждой строки
def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    # Проходим по строкам и оставляем только неповторяющиеся значения
    return list(np.unique(row).tolist() for row in matrix)


# Задача 3 (часть 2). Уникальные элементы для каждого столбца
def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    # matrix.T переворачивает матрицу (столбцы становятся строками), дальше ищем уникальные элементы
    return list(np.unique(col).tolist() for col in matrix.T)


# Задача 4. Генерация случайной матрицы и расчёт её статистик
def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    # Создаём генератор с фиксированным seed и заполняем матрицу нормальным распределением
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    # Считаем среднее (mean) и дисперсию (var): axis=1 по строкам, axis=0 по столбцам
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


# Задача 5. Матрица в шахматном порядке
def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    # Складываем номера строки и столбца: где сумма чётная — ставим first, где нечётная — second
    grid = np.indices((rows, columns)).sum(axis=0)
    return np.where(grid % 2 == 0, first, second)


# Задача 6 (часть 1). Рисуем прямоугольник по центру картинки
def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    # Заливаем весь холст цветом фона, вычисляем отступы от краёв и закрашиваем центр
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    img[top : top + height, left : left + width] = shape_color
    return img


# Задача 6 (часть 2). Рисуем эллипс по центру картинки
def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    # Заливаем фон, находим координаты центра (x0, y0) и по уравнению эллипса закрашиваем нужные пиксели
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    y, x = np.ogrid[:image_height, :image_width]
    x0 = (image_width - 1) / 2.0
    y0 = (image_height - 1) / 2.0

    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    img[mask] = shape_color
    return img


# Задача 7. Анализ временного ряда
def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window

    # Общие статистики: среднее, дисперсия и стандартное отклонение
    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    # Локальные максимумы (больше обоих соседей) и минимумы (меньше обоих соседей)
    max_idx = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    min_idx = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) + 1

    # Скользящее среднее через свёртку (np.convolve)
    kernel = np.ones(window) / window
    moving_avg = np.convolve(values, kernel, mode="valid")

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=max_idx,
        local_minima_indices=min_idx,
        moving_average=moving_avg,
    )


# Задача 8. One-hot кодирование меток классов
def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels, dtype=int), data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
    # Создаём единичную матрицу np.eye и берём из неё нужные строки по номерам классов
    return np.eye(class_count, dtype=int)[labels]

