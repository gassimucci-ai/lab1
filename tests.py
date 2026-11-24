"""Тесты для проверки функций и построение гистограмм."""

import matplotlib.pyplot as plt
import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


def test_python_basics():
    # 1. Подсчёт гласных
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("bcdfg")) == 0

    # 2. Уникальные символы
    assert has_unique_characters(TextInput("abcdef")) is True
    assert has_unique_characters(TextInput("hello")) is False
    assert has_unique_characters(TextInput("")) is True

    # 3. Единичные биты
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(8)) == 1

    # 4. Мультипликативная устойчивость
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

    # 5. Средняя квадратичная ошибка
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2], [2, 4])) == 2.5

    # 6. Разложение на простые множители
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"

    # 7. Пирамида из кубиков
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"

    # 8. Сбалансированное число
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(5)) is True


def test_numpy_tasks():
    # 1. Сумма произведений матриц на векторы
    matrices = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    vectors = np.array([[[1], [2]], [[1], [1]]])
    res = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert np.array_equal(res, np.array([[16], [26]]))

    # 2. Бинаризация
    b_res = binarize(BinarizeInput([[0.2, 0.8], [0.5, 0.9]], threshold=0.5))
    assert np.array_equal(b_res, np.array([[0, 1], [0, 1]]))

    # 3. Уникальные элементы строк и столбцов
    mat = [[1.0, 2.0, 1.0], [3.0, 2.0, 3.0]]
    assert unique_rows(MatrixInput(mat)) == [[1.0, 2.0], [2.0, 3.0]]
    assert unique_columns(MatrixInput(mat)) == [[1.0, 3.0], [2.0], [1.0, 3.0]]

    # 4. Статистики случайной матрицы
    stats = matrix_statistics(RandomMatrixInput(rows=10, columns=20, seed=42))
    assert stats.matrix.shape == (10, 20)
    assert len(stats.row_means) == 10
    assert len(stats.column_means) == 20

    # 5. Шахматная матрица
    c_res = chess(ChessInput(rows=2, columns=3, first=0, second=1))
    assert np.array_equal(c_res, np.array([[0, 1, 0], [1, 0, 1]]))

    # 6. Прямоугольник и эллипс
    rect = draw_rectangle(RectangleInput(4, 4, 10, 10, (255, 0, 0), (0, 0, 0)))
    ell = draw_ellipse(EllipseInput(3, 2, 10, 10, (0, 255, 0), (0, 0, 0)))
    assert rect.shape == (10, 10, 3)
    assert ell.shape == (10, 10, 3)

    # 7. Временной ряд
    ts = analyze_time_series(TimeSeriesInput([1, 5, 2, 0, 4], window=3))
    assert list(ts.local_maxima_indices) == [1]
    assert list(ts.local_minima_indices) == [3]
    assert len(ts.moving_average) == 3

    # 8. One-hot encoding
    oh = one_hot(OneHotInput([0, 2, 1, 3]))
    assert oh.shape == (4, 4)
    assert oh[1, 2] == 1


def plot_matrix_histograms():
    """Построение гистограмм для Задачи 4 из части NumPy."""
    stats = matrix_statistics(RandomMatrixInput(rows=50, columns=50, seed=42))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(stats.row_means, bins=10, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Средние по строкам")

    axes[0, 1].hist(stats.column_means, bins=10, color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Средние по столбцам")

    axes[1, 0].hist(stats.row_variances, bins=10, color="salmon", edgecolor="black")
    axes[1, 0].set_title("Дисперсии по строкам")

    axes[1, 1].hist(stats.column_variances, bins=10, color="plum", edgecolor="black")
    axes[1, 1].set_title("Дисперсии по столбцам")

    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.close()


if __name__ == "__main__":
    test_python_basics()
    test_numpy_tasks()
    plot_matrix_histograms()
    print("Все тесты пройдены успешно, гистограммы сохранены!")
