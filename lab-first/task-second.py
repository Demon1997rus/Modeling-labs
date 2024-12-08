import numpy as np
import matplotlib.pyplot as plt
import os


def mid_square_method(seed, num_values, n_digits):
    """
    Реализация метода серединных квадратов.
    seed: начальное значение (зёрно генератора)
    num_values: количество чисел, которые нужно сгенерировать
    n_digits: количество цифр в числе (определяет размер средней части)
    """
    random_numbers = []
    current_value = seed

    for _ in range(num_values):
        # 1. Возвести текущее значение в квадрат
        squared = str(current_value * current_value).zfill(
            2 * n_digits
        )  # Оставить достаточно цифр
        # 2. Взять из квадрата средние n_digits цифр
        mid_start = len(squared) // 2 - n_digits // 2
        mid_value = int(squared[mid_start : mid_start + n_digits])
        # 3. Нормализовать результат в диапазон [0, 1]
        random_numbers.append(mid_value / (10**n_digits))
        # 4. Обновить текущее значение
        current_value = mid_value

    return random_numbers


def calculate_statistics(numbers):
    """
    Расчёт статистических характеристик:
    - математического ожидания,
    - дисперсии,
    - среднеквадратичного отклонения.
    """
    mean = np.mean(numbers)  # Математическое ожидание
    variance = np.var(numbers)  # Дисперсия
    std_dev = np.std(numbers)  # Среднеквадратичное отклонение
    return mean, variance, std_dev


def plot_histogram(numbers, N, filename):
    """
    Построение и сохранение гистограммы частот для анализа случайных чисел.
    """
    plt.hist(
        numbers, bins=10, density=True, alpha=0.75, color="skyblue", edgecolor="black"
    )
    plt.title(f"Гистограмма частотности методом серединных квадратов (N={N})")
    plt.xlabel("Интервалы значений X")
    plt.ylabel("Частота (P(X))")
    plt.grid()
    plt.savefig(filename)
    plt.close()


# === Основная часть ===
if __name__ == "__main__":
    # Параметры
    seed = 5678  # Начальное значение (выберем произвольно)
    n_digits = 4  # Количество цифр числа (обычно 4)
    sequence_sizes = [100, 1000, 10000]  # Размеры выборок
    output_folder = "results_lab2"  # Результаты будут сохранены тут

    # Создаем папку для выходных данных (если её нет)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Теоретические значения для равномерного распределения
    theoretical_mean = 0.5
    theoretical_variance = 1 / 12
    theoretical_std_dev = (1 / 12) ** 0.5

    print("=== РЕЗУЛЬТАТЫ ПО МЕТОДУ СЕРЕДИННЫХ КВАДРАТОВ ===\n")
    print(f"Теоретические значения:")
    print(f"- Математическое ожидание (M): {theoretical_mean}")
    print(f"- Дисперсия (D): {theoretical_variance}")
    print(f"- Среднеквадратичное отклонение (σ): {theoretical_std_dev}\n")

    # Обработка каждой выборки
    for N in sequence_sizes:
        print(f"=== Анализ для N = {N} ===")
        # Генерируем случайные числа методом серединных квадратов
        random_numbers = mid_square_method(seed, N, n_digits)

        # Рассчитываем статистические характеристики
        mean, variance, std_dev = calculate_statistics(random_numbers)

        # Выводим результаты
        print(f"Количество чисел: {N}")
        print(f"- Математическое ожидание: {mean}")
        print(f"- Дисперсия: {variance}")
        print(f"- Среднеквадратичное отклонение: {std_dev}")
        print(
            f"- Отклонение мат. ожидания от теоретического значения: {abs(theoretical_mean - mean)}\n"
        )

        # Сохраняем гистограмму
        histogram_file = f"{output_folder}/histogram_N{N}.png"
        plot_histogram(random_numbers, N, histogram_file)
        print(f"- Гистограмма сохранена: {histogram_file}\n")

    print("=== Завершено ===")
