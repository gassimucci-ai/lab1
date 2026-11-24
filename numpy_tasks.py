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


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = np.asarray(data.matrices), np.asarray(data.vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = np.asarray(data.matrix), data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    grid = np.indices((rows, columns)).sum(axis=0)
    return np.where(grid % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    img[top : top + height, left : left + width] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    y, x = np.ogrid[:image_height, :image_width]
    x0 = (image_width - 1) / 2.0
    y0 = (image_height - 1) / 2.0

    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window

    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    max_idx = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1
    min_idx = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1

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


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels, dtype=int), data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
    return np.eye(class_count, dtype=int)[labels]
