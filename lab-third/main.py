import numpy as np


def main():
    # Инициализация параметров
    types = [1, 2, 3]  # Типы сообщений
    probabilities = [0.77, 0.06, 0.17]  # Вероятности появления каждого типа
    N = 100  # Количество сообщений
    data = np.array(
        [
            [0.72, 0.09, 0.00, 0.08, 0.11],  # Для типа 1
            [0.74, 0.09, 0.01, 0.09, 0.07],  # Для типа 2
            [0.27, 0.04, 0.38, 0.29, 0.02],  # Для типа 3
        ]
    )
    A, B = 1, 116  # Границы равномерного распределения для чётных типов
    avg_time_gap = 8.1  # Среднее время между сообщениями
    lambda_erlang = 1 / avg_time_gap  # Параметр λ для экспоненциального распределения (чётные типы)

    # === Задача 1: Генерация типов сообщений ===
    generated_types = np.random.choice(types, size=N, p=probabilities)
    print("Сгенерированные типы сообщений:")
    print(generated_types)

    # === Задача 2: Генерация номеров абонентов ===
    num_abonents = data.shape[1]
    abonents = np.arange(1, num_abonents + 1)

    generated_abonents = []
    for message_type in generated_types:
        type_index = message_type - 1
        probabilities_for_type = data[type_index]
        abonent = np.random.choice(abonents, p=probabilities_for_type)
        generated_abonents.append(abonent)

    generated_abonents = np.array(generated_abonents)
    print("Сгенерированные номера абонентов:")
    print(generated_abonents)

    # === Задача 3: Длины сообщений ===
    lengths = []
    for message_type in generated_types:
        if message_type % 2 == 1:  # Нечётный тип сообщения
            type_index = message_type - 1
            probabilities_for_type = data[type_index]
            abonent_numbers = np.arange(1, len(probabilities_for_type) + 1)
            length = np.random.choice(abonent_numbers, p=probabilities_for_type)
        else:  # Чётный тип сообщения
            length = np.random.randint(A, B + 1)
        lengths.append(length)

    lengths = np.array(lengths)
    print("Сгенерированные длины сообщений:")
    print(lengths)

    # === Задача 4: Моделирование времени поступления сообщений ===
    time_intervals = []  # Промежутки времени между сообщениями

    for message_type in generated_types:
        if message_type % 2 == 1:  # Нечётный тип сообщения
            type_index = message_type - 1
            probabilities_for_type = data[type_index]
            abonent_numbers = np.arange(1, len(probabilities_for_type) + 1)
            interval = np.random.choice(abonent_numbers, p=probabilities_for_type)
        else:  # Чётный тип сообщения
            interval = np.random.exponential(scale=1 / lambda_erlang)
        time_intervals.append(interval)

    time_intervals = np.array(time_intervals)
    time_moments = np.cumsum(time_intervals)  # Моменты поступления сообщений
    print("\nПромежутки времени между сообщениями:")
    print(time_intervals)
    print("\nМоменты поступления сообщений в систему:")
    print(time_moments)

    # === Формирование основной таблицы ===
    main_table = list(zip(range(1, N + 1), generated_types, generated_abonents, lengths, time_moments))
    print("\nТаблица сообщений (Тип, Абонент, Длина, Время):")
    print("№   Тип  Абонент  Длина  Время")
    for row in main_table:
        print(f"{row[0]:<4} {row[1]:<4} {row[2]:<7} {row[3]:<6} {row[4]:.2f}")

    # === Анализ потока сообщений ===

    # Количество сообщений каждого типа и их вероятности
    unique_types, type_counts = np.unique(generated_types, return_counts=True)
    total_messages = len(generated_types)
    type_probabilities = type_counts / total_messages
    print("\nКоличество сообщений каждого типа и их вероятности:")
    for t, count, prob in zip(unique_types, type_counts, type_probabilities):
        print(f"Тип {t}: {count} сообщений, вероятность {prob:.4f}")

    # Средняя и максимальная длина сообщений каждого типа
    print("\nСредняя и максимальная длина сообщений каждого типа:")
    for t in unique_types:
        lengths_for_type = lengths[generated_types == t]
        print(
            f"Тип {t}: Средняя длина = {np.mean(lengths_for_type):.2f}, "
            f"Максимальная длина = {np.max(lengths_for_type)}"
        )

    # Частота поступления сообщений каждого типа
    print("\nЧастота поступления сообщений каждого типа:")
    total_time = time_moments[-1]
    for t in unique_types:
        count = np.sum(generated_types == t)
        print(f"Тип {t}: Средняя частота = {count / total_time:.4f} сообщений/ед. времени")

    # Вероятности и частота поступления сообщений к каждому абоненту
    print("\nПоступление сообщений к абонентам:")
    for t in unique_types:
        abonents_for_type = generated_abonents[generated_types == t]
        unique_abonents, abonent_counts = np.unique(abonents_for_type, return_counts=True)
        total_type_count = len(abonents_for_type)
        print(f"Тип {t}:")
        for abonent, count in zip(unique_abonents, abonent_counts):
            prob = count / total_type_count
            print(f"  Абонент {abonent}: {count} сообщений, вероятность = {prob:.4f}")

    # Математическое ожидание, дисперсия и стандартное отклонение
    mean_length = np.mean(lengths)
    var_length = np.var(lengths)
    std_length = np.std(lengths)
    print("\nСтатистические характеристики длин сообщений:")
    print(f"Математическое ожидание длины: {mean_length:.4f}")
    print(f"Дисперсия длины: {var_length:.4f}")
    print(f"Среднеквадратичное отклонение длины: {std_length:.4f}")

    # Интенсивность потока
    intensity = total_messages / total_time
    print(f"\nИнтенсивность потока сообщений: {intensity:.4f} сообщений/ед. времени")


if __name__ == "__main__":
    main()
