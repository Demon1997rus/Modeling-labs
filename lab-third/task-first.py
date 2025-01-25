import csv
import os
from collections import Counter

import numpy as np

from config import Task1Config, load_config


def main():
    # --- Загрузка конфигурации ---
    config: Task1Config = load_config("config.json").task1
    types = config.types
    probabilities = config.probabilities
    num_messages = config.num_messages

    # --- Генерация типов сообщений ---
    generated_types = np.random.choice(types, size=num_messages, p=probabilities)

    # --- Подсчёт частоты сообщений каждого типа ---
    frequency = Counter(generated_types)

    # --- Папка для сохранения результатов ---
    results_dir = "task-first-results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Сохранение распределения в CSV + Печать в консоль ---
    distribution_file = os.path.join(results_dir, "distribution.csv")
    print("\nЧастота появления сообщений (распределение):")
    with open(distribution_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Тип сообщения", "Кол-во (эксперим.)", "Вероятность (эксперим.)", "Вероятность (теор.)"])

        for t in types:
            experimental_count = frequency[t]  # Количество сообщений этого типа
            experimental_probability = experimental_count / num_messages * 100  # Экспериментальная вероятность
            theoretical_probability = probabilities[types.index(t)] * 100  # Теоретическая вероятность
            writer.writerow(
                [t, experimental_count, f"{experimental_probability:.2f}%", f"{theoretical_probability:.2f}%"]
            )
            print(
                f"Тип {t}: {experimental_count} сообщений "
                f"(экспериментально = {experimental_probability:.2f}%, "
                f"теоретически = {theoretical_probability:.2f}%)"
            )

    print(f"Распределение сообщений сохранено в: {distribution_file}")


if __name__ == "__main__":
    main()
