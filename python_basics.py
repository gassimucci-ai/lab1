"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


# Задача 1. Считаем гласные буквы (a, e, i, o, u)
def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    # Переводим текст в нижний регистр и считаем буквы, которые есть в списке гласных
    return sum(1 for char in text.lower() if char in vowels)


# Задача 2. Проверяем, все ли символы в строке уникальные (нет ли повторов)
def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    # set() автоматически удаляет дубликаты. Если длина не изменилась — повторов не было
    return len(text) == len(set(text))


# Задача 3. Считаем количество единиц в двоичной записи числа
def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    # bin() переводит число в двоичный вид (строку), а count("1") считает единицы
    return bin(number).count("1")


# Задача 4. Мультипликативная устойчивость (сколько раз нужно перемножить цифры, чтобы осталась одна)
def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    # Пока в числе больше одной цифры (число >= 10), перемножаем его цифры
    while number >= 10:
        prod = 1
        for digit in str(number):
            prod *= int(digit)
        number = prod
        steps += 1
    return steps


# Задача 5. Средняя квадратичная ошибка (MSE) между двумя списками
def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    # Берём пары чисел, возводим их разницу в квадрат, складываем и делим на количество
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


# Задача 6. Разложение числа на простые множители
def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = list()
    d = 2
    # Делим число на 2, 3 и далее, пока делится без остатка, считая степень (count)
    while d * d <= number:
        count = 0
        while number % d == 0:
            count += 1
            number //= d
        if count > 0:
            factors.append(f"({d}**{count})" if count > 1 else f"({d})")
        d += 1
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


# Задача 7. Пирамида из кубиков (сколько слоёв можно сложить из n кубиков: 1^2 + 2^2 + 3^2...)
def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    # Прибавляем квадраты слоёв, пока не наберём нужное число кубиков
    while total < cube_count:
        k += 1
        total += k ** 2
    # Если сошлось ровно — возвращаем число слоёв, если перескочили — пишем, что невозможно
    return k if total == cube_count else "It is impossible"


# Задача 8. Сбалансированное число (равна ли сумма цифр слева и справа от середины)
def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    mid = length // 2
    # Выкидываем 1 цифру по центру (если длина нечётная) или 2 (если чётная)
    if length % 2 == 1:
        left = s[:mid]
        right = s[mid + 1:]
    else:
        left = s[:mid - 1]
        right = s[mid + 1:]
    # Сравниваем сумму цифр левой и правой частей
    return sum(map(int, left)) == sum(map(int, right))
