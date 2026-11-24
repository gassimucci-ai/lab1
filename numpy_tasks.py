def test_numpy_tasks():
    # 1. Сумма произведений матриц на векторы
    matrices = np.array([[[1, 2], [3, 4]], [[2, 0], [0, 2]]])
    vectors = np.array([[[1], [1]], [[2], [3]]])
    res = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert np.array_equal(res, np.array([[7], [13]]))

    # 2. Бинаризация
    b_res = binarize(BinarizeInput([[0.2, 0.8], [0.5, 0.9]], threshold=0.5))
    assert np.array_equal(b_res, np.array([[0, 1], [0, 1]]))

    # 3. Уникальные элементы строк и столбцов
    mat = [[1, 2, 1], [3, 3, 4]]
    assert unique_rows(MatrixInput(mat)) == [[1, 2], [3, 4]]
    assert unique_columns(MatrixInput(mat)) == [[1, 3], [2, 3], [1, 4]]

    # 4. Статистики случайной матрицы
    stats = matrix_statistics(RandomMatrixInput(rows=10, columns=20, seed=42))
    assert stats.matrix.shape == (10, 20)
    assert len(stats.row_means) == 10
    assert len(stats.column_means) == 20

    # 5. Шахматная матрица
    c_res = chess(ChessInput(rows=2, columns=3, first=0, second=1))
    assert np.array_equal(c_res, np.array([[0, 1, 0], [1, 0, 1]]))

    # 6. Прямоугольник и эллипс (проверяем размеры картинки)
    rect = draw_rectangle(RectangleInput(4, 4, 10, 10, (255, 0, 0), (0, 0, 0)))
    ell = draw_ellipse(EllipseInput(3, 2, 10, 10, (0, 255, 0), (0, 0, 0)))
    assert rect.shape == (10, 10, 3)
    assert ell.shape == (10, 10, 3)

    # 7. Временной ряд
    ts = analyze_time_series(TimeSeriesInput([1, 5, 2, 6, 1], window=3))
    assert list(ts.local_maxima_indices) == [1, 3]
    assert list(ts.local_minima_indices) == [2]
    assert len(ts.moving_average) == 3

    # 8. One-hot encoding
    oh = one_hot(OneHotInput([0, 2, 3, 0]))
    assert oh.shape == (4, 4)
    assert oh[1, 2] == 1
