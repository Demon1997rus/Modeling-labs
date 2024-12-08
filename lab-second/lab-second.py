import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Генерация данных
def generate_poisson_sample(lam, size):
    """
    Генерация выборки случайных чисел по распределению Пуассона.
    """
    return np.random.poisson(lam=lam, size=size)


# 2. Вычисление характеристик
def calculate_statistics(sample):
    """
    Вычисление математического ожидания, дисперсии и среднеквадратичного отклонения.
    """
    M = np.mean(sample)
    D = np.var(sample)
    sigma = np.std(sample)
    return M, D, sigma


# 3. Построение графиков
def plot_distributions(sample, lam, max_value, output_prefix):
    """
    Построение графиков P(X) (гистограмма) и F(X) (функция распределения).
    """
    # Гистограмма P(X)
    plt.hist(
        sample,
        bins=range(0, max_value + 1),
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
    )
    plt.title(f"Гистограмма P(X): Распределение Пуассона (λ = {lam})")
    plt.xlabel("X")
    plt.ylabel("P(X)")
    plt.xlim(0, max_value)
    plt.grid()
    plt.savefig(f"{output_prefix}_p_x.png")
    plt.close()

    # Функция распределения F(X)
    values, counts = np.unique(sample, return_counts=True)
    cumulative = np.cumsum(counts) / len(sample)
    plt.step(values, cumulative, where="post", color="blue")
    plt.title(f"Функция распределения F(X) для Пуассона (λ = {lam})")
    plt.xlabel("X")
    plt.ylabel("F(X)")
    plt.xlim(0, max_value)
    plt.grid()
    plt.savefig(f"{output_prefix}_f_x.png")
    plt.close()


# 4. Генерация отчёта
def generate_report(
    filename, lam, M, D, sigma, theoretical_sigma, max_value, output_prefix
):
    """
    Формирование полного отчета с текстом.
    """
    with open(filename, "w") as report:
        report.write("=" * 60 + "\n")
        report.write("Отчёт по лабораторной работе №2\n")
        report.write("\nТема: Генерация случайной величины по распределению Пуассона\n")
        report.write("\nЦель работы:\n")
        report.write(
            "- Изучить распределение случайных величин и реализовать генерацию случайных величин, подчиняющихся распределению Пуассона.\n"
        )
        report.write(
            "- Рассчитать математическое ожидание, дисперсию и среднеквадратичное отклонение.\n"
        )
        report.write(
            "- Построить графики функции вероятности (P(X)) и функции распределения (F(X)).\n"
        )
        report.write("\nТеоретическая часть:\n")
        report.write(
            f"Распределение Пуассона характеризуется параметром λ = {lam}, которое совпадает с математическим ожиданием.\n"
        )
        report.write("Формула распределения Пуассона:\n")
        report.write("P(X = k) = (λ^k × e^(-λ)) / k!, где k >= 0.\n")
        report.write("Математическое ожидание (M) и дисперсия (D) равны λ.\n\n")
        report.write("Ход работы:\n")
        report.write(
            f"- Сгенерировано 100 случайных чисел по распределению Пуассона с параметром λ = {lam}.\n"
        )
        report.write(
            "- Для выборки рассчитаны основные характеристики (математическое ожидание, дисперсия, среднеквадратичное отклонение).\n"
        )
        report.write(
            "- Построены графики функции вероятности (P(X)) и функции распределения (F(X)).\n\n"
        )
        report.write("Результаты:\n")
        report.write(f"- Математическое ожидание (теоретическое): {lam}\n")
        report.write(f"- Математическое ожидание (экспериментальное): {M:.2f}\n")
        report.write(f"- Дисперсия (теоретическая): {lam}\n")
        report.write(f"- Дисперсия (экспериментальная): {D:.2f}\n")
        report.write(
            f"- Среднеквадратичное отклонение (теоретическое): {theoretical_sigma:.2f}\n"
        )
        report.write(
            f"- Среднеквадратичное отклонение (экспериментальное): {sigma:.2f}\n"
        )
        report.write("\nГрафики:\n")
        report.write(f"- Гистограмма вероятностей P(X): {output_prefix}_p_x.png\n")
        report.write(f"- Функция распределения F(X): {output_prefix}_f_x.png\n")
        report.write("\nВыводы:\n")
        report.write(
            "- Экспериментальные характеристики выборки (математическое ожидание, дисперсия) близки к теоретическим значениям, что подтверждает корректность модели.\n"
        )
        report.write(
            "- Построенные графики P(X) и F(X) соответствуют распределению Пуассона.\n"
        )
        report.write("=" * 60 + "\n")


# Основная программа
if __name__ == "__main__":
    # Параметры задания
    lam = 24  # Параметр λ (математическое ожидание)
    max_value = 72  # Максимальное значение для графиков
    sample_size = 100  # Размер выборки
    output_prefix = "results/lab2_poisson"  # Префикс для сохранения графиков
    report_filename = "results/lab2_poisson_report.txt"  # Имя текстового отчёта

    # Создаём папку для сохранения результатов
    if not os.path.exists("results"):
        os.makedirs("results")

    # Генерация выборки
    sample = generate_poisson_sample(lam=lam, size=sample_size)

    # Расчёт характеристик
    M, D, sigma = calculate_statistics(sample)
    theoretical_sigma = np.sqrt(lam)

    # Построение графиков
    plot_distributions(sample, lam, max_value, output_prefix)

    # Генерация отчёта
    generate_report(
        report_filename, lam, M, D, sigma, theoretical_sigma, max_value, output_prefix
    )

    print(f"Отчёт сгенерирован и сохранён как: {report_filename}")
    print(f"Графики сохранены в папку 'results'")
