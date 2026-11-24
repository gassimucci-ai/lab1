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
    # 1. Подсчёт гласных (обычный случай и пустая строка / без гласных)
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("bcdfg")) == 0

    # 2. Уникальные символы
    assert has_unique_characters(TextInput("abcdef")) is True
    assert has_unique_characters(TextInput("hello")) is False
    assert has_unique_characters(TextInput("")) is True

    # 3. Единичные биты (7 = 111 в двоичной, 8 = 1000)
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(8)) == 1

    # 4. Мультипликативная устойчивость (примеры из условия)
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

    # 5. Средняя квадратичная ошибка
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 3], [2, 5])) == 2.5

    # 6. Разложение на простые множители
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"

    # 7. Пирамида из кубиков (1^2 + 2^2 + 3^2 = 14)
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"

    # 8. Сбалансированное число
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(5)) is True
