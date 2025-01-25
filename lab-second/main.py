import json
import math
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import chi2  # Для теста хи-квадрат


class Config(BaseModel):
    math_wait: int = Field(..., ge=0, description="Математическое ожидание (λ ≥ 0)")
    max_val: int = Field(..., ge=1, description="Максимальное значение (≥ 1)")
    sample_size: int = Field(..., ge=1, description="Размер выборки (≥ 1)")

    @classmethod
    def from_json(cls, config_path: str):
        with open(config_path, "r") as file:
            config_data = json.load(file)
        return cls(**config_data)


class RandomPoisson:
    def __init__(self, config: Config):
        self.config: Config = config

    def generate(self) -> np.ndarray:
        return np.random.poisson(lam=self.config.math_wait, size=self.config.sample_size)


class Specifications:
    def __init__(self, numbers: np.ndarray):
        self.math_wait: float = np.mean(numbers)
        self.variance: float = np.var(numbers)
        self.std_dev: float = np.std(numbers)

    def print_info(self):
        print(f"Эмпирическое мат. ожидание (M): {self.math_wait:.2f}")
        print(f"Эмпирическая дисперсия (D): {self.variance:.2f}")
        print(f"Среднеквадратичное отклонение (σ): {self.std_dev:.2f}")

    def confidence_interval(self, confidence: float, n: int) -> tuple:
        """
        Вычисление доверительного интервала для математического ожидания.
        """
        z = 1.96
        margin_of_error = z * self.std_dev / math.sqrt(n)
        return self.math_wait - margin_of_error, self.math_wait + margin_of_error


def chi_square_test(data: np.ndarray, lam: float, max_val: int):
    """
    Тест хи-квадрат для проверки того, соответствует ли выборка закону Пуассона.
    """
    observed_freq, bin_edges = np.histogram(data, bins=range(0, max_val + 2), density=False)
    expected_freq = [(lam**k * math.exp(-lam)) / math.factorial(k) * len(data) for k in range(max_val + 1)]

    # Ограничиваем оставшиеся данные для теста хи-квадрат
    observed_freq = observed_freq[: max_val + 1]
    expected_freq = expected_freq[: max_val + 1]

    # Дополняем последний интервал для редких событий
    observed_freq[-1] += len(data) - sum(observed_freq)
    expected_freq[-1] += sum(expected_freq) - sum(expected_freq[:-1])

    chi_square_stat = sum(((obs - exp) ** 2) / exp for obs, exp in zip(observed_freq, expected_freq) if exp > 0)
    p_value = 1 - chi2.cdf(chi_square_stat, df=max_val - 1)

    print("\nРезультаты теста хи-квадрат:")
    print(f"Статистика χ²: {chi_square_stat:.2f}")
    print(f"P-значение: {p_value:.4f}")

    if p_value > 0.05:
        print("Выборка соответствует закону Пуассона (не отвергаем гипотезу H0).")
    else:
        print("Выборка не соответствует закону Пуассона (отвергаем гипотезу H0).")


def plot_graphs(data: np.ndarray, max_val: int, lam: float, save_dir: str):
    """
    Построить и сохранить графики функции плотности вероятностей P(X) и F(X).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Полигон вероятностей (гистограмма)
    plt.figure(figsize=(12, 6))
    counts, bins, _ = plt.hist(
        data,
        bins=range(0, max_val + 1),
        density=True,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label="Гистограмма",
    )
    plt.title("Функция плотности вероятностей (P(X)) — Гистограмма")
    plt.xlabel("Значения случайной величины")
    plt.ylabel("Вероятность")
    plt.legend()
    plt.grid()

    # Теоретическая плотность Пуассона
    x = np.arange(0, max_val + 1)
    poisson_probs = [(lam**k * np.exp(-lam)) / math.factorial(k) for k in x]
    plt.plot(x, poisson_probs, "r-", marker="o", label="Теоретическая P(X)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "probability_density_function.png"))
    plt.close()

    # Эмпирическая функция распределения (F(X))
    sorted_data = np.sort(data)
    F_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(12, 6))
    plt.step(
        sorted_data,
        F_values,
        where="post",
        label="Эмпирическая F(X)",
        color="orange",
    )
    plt.title("Эмпирическая функция распределения (F(X))")
    plt.xlabel("Значения случайной величины")
    plt.ylabel("F(X)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "cumulative_distribution_function.png"))
    plt.close()


def main():
    # 1. Загружаем конфигурацию из файла
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    try:
        config: Config = Config.from_json(config_file)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return

    # 2. Генерируем выборку
    random_poisson: RandomPoisson = RandomPoisson(config=config)
    poisson_sample: np.ndarray = random_poisson.generate()

    # Создаем папку для сохранения результатов
    results_dir = "./results"

    # 3. Рассчитываем выборочные характеристики
    specifications: Specifications = Specifications(numbers=poisson_sample)
    specifications.print_info()

    # 4. Строим и сохраняем графики
    plot_graphs(data=poisson_sample, max_val=config.max_val, lam=config.math_wait, save_dir=results_dir)

    # 5. Теоретические характеристики
    print("\nТеоретические параметры распределения Пуассона:")
    print(f"Математическое ожидание (теор): {config.math_wait}")
    print(f"Дисперсия (теор): {config.math_wait}")
    print(f"Среднеквадратичное отклонение (теор): {np.sqrt(config.math_wait):.2f}")

    # 6. Оценка качества
    print("\nСравнение теоретических и эмпирических характеристик:")
    print(f"Математическое ожидание: теор = {config.math_wait}, эмп = {specifications.math_wait:.2f}")
    print(f"Дисперсия: теор = {config.math_wait}, эмп = {specifications.variance:.2f}")
    print(f"Среднеквадратичное отклонение: теор = {np.sqrt(config.math_wait):.2f}, эмп = {specifications.std_dev:.2f}")

    # 7. Тест хи-квадрат
    chi_square_test(data=poisson_sample, lam=config.math_wait, max_val=config.max_val)

    # 8. Доверительный интервал
    confidence_interval = specifications.confidence_interval(confidence=0.95, n=config.sample_size)
    print(f"\nДоверительный интервал для M (95%): {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}")


if __name__ == "__main__":
    main()
