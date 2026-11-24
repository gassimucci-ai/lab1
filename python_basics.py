"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for char in text.lower() if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        prod = 1
        for digit in str(number):
            prod *= int(digit)
        number = prod
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    d = 2
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


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k ** 2
    return k if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    mid = length // 2
    if length % 2 == 1:
        left = s[:mid]
        right = s[mid + 1:]
    else:
        left = s[:mid - 1]
        right = s[mid + 1:]
    return sum(map(int, left)) == sum(map(int, right))
