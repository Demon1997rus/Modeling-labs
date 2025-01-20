import os
import numpy as np
import matplotlib.pyplot as plt


# Общая функция для расчета характеристик выборки
def calculate_characteristics(numbers):
    """
    Вычисляет математическое ожидание,
    дисперсию и стандартное отклонение.
    """
    M = np.mean(numbers)
    D = np.var(numbers)
    sigma = np.std(numbers)
    return M, D, sigma


# Функция для расчета вероятности попадания чисел в интервалы
def calculate_probability(numbers, bins=10):
    """Рассчитывает вероятность попадания чисел в интервалы."""
    counts, edges = np.histogram(numbers, bins=bins, range=(0, 1))
    probabilities = counts / len(numbers)
    return edges, probabilities


# Построение графика функции распределения вероятностей P(X)
def plot_probability_distribution(
    edges, probabilities, N, method_name, output_folder
):
    plt.bar(
        edges[:-1],
        probabilities,
        width=np.diff(edges),
        edgecolor="black",
        align="edge",
    )
    plt.title(f"Функция P(X) (N={N}) [{method_name}]")
    plt.xlabel("Интервалы")
    plt.ylabel("Вероятность P(X)")
    plt.savefig(
        os.path.join(output_folder, f"probability_N={N}_{method_name}.png")
    )
    plt.close()


# Генерация случайных чисел с использованием линейного конгруэнтного метода
def linear_congruential_method(
    seed, num_values, a=1664525, c=1013904223, m=2**32
):
    """Линейный конгруэнтный метод генерации псевдослучайных чисел."""
    values = []
    X = seed
    for _ in range(num_values):
        X = (a * X + c) % m
        values.append(X / m)  # Нормализация в диапазоне [0, 1)
    return values


# Генерация случайных чисел
# с использованием метода numpy (стандартный генератор)
def numpy_random_numbers(num_values):
    """Генерация псевдослучайных чисел с использованием numpy."""
    return np.random.uniform(0, 1, num_values)


# Проверка равномерности для нескольких последовательностей фиксированной длины
def check_fixed_length_uniformity(
    generate_numbers,
    output_folder,
    method_name,
    sequence_length=1000,
    sequence_count=10,
):
    """Анализ равномерности для фиксированной длины."""
    theoretical_mean = 0.5
    deviations = []
    for i in range(sequence_count):
        numbers = generate_numbers(sequence_length)
        M_i, _, _ = calculate_characteristics(numbers)
        deviations.append(abs(theoretical_mean - M_i))

    # Построение графика отклонений |M - M_i|
    plt.plot(
        range(1, sequence_count + 1), deviations, marker="o", color="blue"
    )
    plt.axhline(
        y=np.mean(deviations),
        color="red",
        linestyle="--",
        label="Среднее отклонение",
    )
    plt.title(f"Отклонение |M - M_i| для фиксированной длины ({method_name})")
    plt.xlabel("Номер последовательности")
    plt.ylabel("|M - M_i|")
    plt.legend()
    plt.savefig(
        os.path.join(
            output_folder, f"fixed_length_deviation_{method_name}.png"
        )
    )
    plt.close()

    return deviations


# Проверка равномерности для последовательностей переменной длины
def check_variable_length_uniformity(
    generate_numbers, output_folder, method_name, max_length=10000, step=1000
):
    """Анализ равномерности для переменной длины."""
    theoretical_mean = 0.5
    deviations = []
    lengths = range(step, max_length + 1, step)

    for N in lengths:
        numbers = generate_numbers(N)
        M_i, _, _ = calculate_characteristics(numbers)
        deviations.append(abs(theoretical_mean - M_i))

    # Построение графика отклонений |M - M_i|
    plt.plot(lengths, deviations, marker="o", color="green")
    plt.axhline(
        y=np.mean(deviations),
        color="red",
        linestyle="--",
        label="Среднее отклонение",
    )
    plt.title(
        f"Отклонение |M - M_i| от длины последовательности ({method_name})"
    )
    plt.xlabel("Длина последовательности N")
    plt.ylabel("|M - M_i|")
    plt.legend()
    plt.savefig(
        os.path.join(
            output_folder, f"variable_length_deviation_{method_name}.png"
        )
    )
    plt.close()

    return deviations


# Расчёт вероятности P(|M - M_i| < σ)
def calculate_within_sigma(deviations, sigma):
    count_within_sigma = sum(
        1 for deviation in deviations if deviation < sigma
    )
    probability = count_within_sigma / len(deviations)
    return count_within_sigma, probability


def run_experiment(generate_numbers, output_folder, method_name):
    # Создаем папку для результатов
    os.makedirs(output_folder, exist_ok=True)

    # Основные последовательности
    for N in [100, 1000, 10000]:
        numbers = generate_numbers(N)
        M, D, sigma = calculate_characteristics(numbers)
        print(
            f"[{method_name}] Для N = {N}: M = {M:.3f}, D = {D:.3f}, "
            f"σ = {sigma:.3f}"
        )

        # Построение гистограммы
        plt.hist(numbers, bins=10, color="blue", edgecolor="black", alpha=0.7)
        plt.title(f"Гистограмма (N={N}) [{method_name}]")
        plt.savefig(
            os.path.join(output_folder, f"histogram_N={N}_{method_name}.png")
        )
        plt.close()

        # Построение функции P(X)
        edges, probabilities = calculate_probability(numbers)
        plot_probability_distribution(
            edges, probabilities, N, method_name, output_folder
        )

    # Проверка равномерности на фиксированных длинах
    deviations_fixed = check_fixed_length_uniformity(
        generate_numbers, output_folder, method_name
    )

    # Проверка равномерности на переменных длинах
    deviations_variable = check_variable_length_uniformity(
        generate_numbers, output_folder, method_name
    )

    # Расчёт вероятности P(|M - M_i| < σ)
    sigma = 1 / np.sqrt(12)  # Теоретическое σ для равномерного распределения
    count_fixed, prob_fixed = calculate_within_sigma(deviations_fixed, sigma)
    count_variable, prob_variable = calculate_within_sigma(
        deviations_variable, sigma
    )
    print(
        f"[{method_name}] Фиксированные длины: {count_fixed}/10"
        f" -> Вероятность: {prob_fixed:.3f}"
    )
    print(
        f"[{method_name}] Переменные длины: {count_variable}/10"
        f" -> Вероятность: {prob_variable:.3f}"
    )


def main():
    np.random.seed(42)

    # Задание 1: Стандартный генератор numpy
    run_experiment(
        numpy_random_numbers,
        output_folder="task_1_results",
        method_name="NumPy",
    )

    # Задание 2: Линейный конгруэнтный метод
    run_experiment(
        lambda n: linear_congruential_method(seed=42, num_values=n),
        output_folder="task_2_results",
        method_name="Linear Congruential Method",
    )


if __name__ == "__main__":
    main()
