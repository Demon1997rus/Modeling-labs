import random
import numpy as np
import matplotlib.pyplot as plt

# === Функции ===


def generate_random_numbers(N):
    """Генерация N случайных чисел с использованием стандартного генератора Python."""
    return [random.random() for _ in range(N)]


def calculate_statistics(numbers):
    """Расчёт математического ожидания, дисперсии и среднеквадратичного отклонения."""
    mean = np.mean(numbers)  # Математическое ожидание
    variance = np.var(numbers)  # Дисперсия
    std_dev = np.std(numbers)  # Среднеквадратичное отклонение
    return mean, variance, std_dev


def plot_histogram(numbers, N, filename):
    """Построение и сохранение гистограммы файла в папку results."""
    plt.hist(
        numbers, bins=10, density=True, alpha=0.75, color="skyblue", edgecolor="black"
    )
    plt.title(f"Гистограмма частотности (N={N})")
    plt.xlabel("Интервалы значений X")
    plt.ylabel("Вероятность (P(X))")
    plt.grid()
    plt.savefig(filename)  # Сохраняем в файл
    plt.close()


# === Основная часть программы ===
if __name__ == "__main__":
    # Задаём размеры выборок
    sequence_sizes = [100, 1000, 10000]
    output_folder = "results_lab1"

    # Создаем папку для хранения выходных данных (графиков)
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Результаты
    print("=== РЕЗУЛЬТАТЫ ПО ПЕРВОМУ ЗАДАНИЮ ===\n")
    theoretical_mean = 0.5
    theoretical_variance = 1 / 12
    theoretical_std_dev = (1 / 12) ** 0.5
    print(f"Теоретические значения:")
    print(f"- Математическое ожидаение (M): {theoretical_mean}")
    print(f"- Дисперсия (D): {theoretical_variance}")
    print(f"- Среднеквадратичное отклонение (σ): {theoretical_std_dev}\n")

    for N in sequence_sizes:
        print(f"=== Анализ для N = {N} ===")
        # 1. Генерируем числа
        numbers = generate_random_numbers(N)

        # 2. Рассчитываем характеристики
        mean, variance, std_dev = calculate_statistics(numbers)

        # 3. Выводим результаты
        print(f"Количество чисел: N = {N}")
        print(f"- Математическое ожидание: {mean}")
        print(f"- Дисперсия: {variance}")
        print(f"- Среднеквадратичное отклонение: {std_dev}")
        print(
            f"- Отклонение математического ожидания от теоретического: {abs(theoretical_mean - mean)}\n"
        )

        # 4. Сохраняем гистограммы
        histogram_file = f"{output_folder}/histogram_N{N}.png"
        plot_histogram(numbers, N, histogram_file)
        print(f"- Гистограмма сохранена: {histogram_file}\n")

    print("=== Завершено ===")
