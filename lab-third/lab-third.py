import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------- Параметры лабораторной работы и директория -----------
os.makedirs("output_lab3", exist_ok=True)

# Данные варианта 21:
P_TYPES = [0.77, 0.06, 0.17]  # Вероятности типов сообщений
ADDRESS_PROBABILITIES = np.array([
    [0.72, 0.74, 0.27],
    [0.09, 0.09, 0.04],
    [0.00, 0.01, 0.38]
])  # Матрица адресации
A, B = 1, 116  # Минимальная и максимальная длина сообщений
AVG_INTERVAL = 8.1  # Средний интервал времени между сообщениями
ERL_K = 1  # Коэффициент k для распределения Эрланга

# ------------------------ Генерация данных ------------------------

def generate_message_types(num_messages=100, probabilities=P_TYPES):
    """
    Задача 1. Генерация типов сообщений на основе заданных вероятностей.
    """
    types = np.arange(1, len(probabilities) + 1)
    generated_types = np.random.choice(types, size=num_messages, p=probabilities)

    # Анализ распределения
    unique, counts = np.unique(generated_types, return_counts=True)
    type_distribution = dict(zip(unique, counts))

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.bar(type_distribution.keys(), type_distribution.values(), color='skyblue', edgecolor='black')
    plt.title("Распределение типов сообщений")
    plt.xlabel("Тип сообщения")
    plt.ylabel("Количество")
    plt.xticks(ticks=types)
    plt.grid(axis="y")
    plt.savefig("output_lab3/message_types.png")
    plt.close()

    return generated_types, type_distribution


def generate_message_addresses(message_types, address_probabilities=ADDRESS_PROBABILITIES):
    """
    Задача 2. Генерация адресов сообщений.
    """
    # Нормализация вероятностей
    normalized_probabilities = address_probabilities / address_probabilities.sum(axis=1, keepdims=True)
    num_addresses = normalized_probabilities.shape[1]
    addresses = []
    
    for message_type in message_types:
        probs = normalized_probabilities[message_type - 1]
        addresses.append(np.random.choice(np.arange(1, num_addresses + 1), p=probs))

    # Анализ распределения
    unique, counts = np.unique(addresses, return_counts=True)
    address_distribution = dict(zip(unique, counts))

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.bar(address_distribution.keys(), address_distribution.values(), color='salmon', edgecolor='black')
    plt.title("Распределение адресов сообщений")
    plt.xlabel("Номер абонента")
    plt.ylabel("Количество сообщений")
    plt.xticks(ticks=np.arange(1, num_addresses + 1))
    plt.grid(axis="y")
    plt.savefig("output_lab3/message_addresses.png")
    plt.close()

    return addresses, address_distribution


def generate_message_lengths(message_types, A=A, B=B):
    """
    Задача 3. Генерация длины сообщений.
    """
    lengths = []
    for message_type in message_types:
        if message_type % 2 == 0:  # Четные типы — равномерное распределение
            lengths.append(np.random.randint(A, B + 1))
        else:  # Нечетные типы — длины сосредоточены вокруг середины интервала
            lengths.append(np.random.randint(A + (B - A) // 4, B - (B - A) // 4))

    # Построение гистограммы длин сообщений
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=15, color='green', alpha=0.7, edgecolor='black')
    plt.title("Распределение длин сообщений")
    plt.xlabel("Длина сообщения")
    plt.ylabel("Количество")
    plt.grid(axis="y")
    plt.savefig("output_lab3/message_lengths.png")
    plt.close()

    return lengths


def generate_message_times(num_messages=100, avg_interval=AVG_INTERVAL, erl_k=ERL_K):
    """
    Задача 4. Генерация времени поступления сообщений.
    """
    times = np.cumsum(np.random.gamma(erl_k, avg_interval / erl_k, size=num_messages))

    # Построение графика времени
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_messages + 1), times, marker='o', linestyle='-', color='purple')
    plt.title("Поступление сообщений во времени")
    plt.xlabel("Номер сообщения")
    plt.ylabel("Время поступления")
    plt.grid()
    plt.savefig("output_lab3/message_times.png")
    plt.close()

    return times


# ------------------------ Генерация текста отчета ------------------------

def generate_report():
    """
    Генерирует полный текстовый отчет.
    """
    report_lines = []

    # Введение
    report_lines.append("Лабораторная работа №3: Моделирование источников сообщений\n")
    report_lines.append("\nЦель: Смоделировать поток сообщений и оценить его характеристики.\n")
    report_lines.append("=" * 80 + "\n\n")

    report_lines.append("Данные варианта 21:\n")
    report_lines.append("Типы сообщений: P(1) = 0.77, P(2) = 0.06, P(3) = 0.17\n")
    report_lines.append("Адресация: Таблица вероятностей P(i,j)\n")
    report_lines.append(f"  Матрица вероятностей: {ADDRESS_PROBABILITIES.tolist()}\n")
    report_lines.append("  Длина сообщений (A, B): [1, 116]\n")
    report_lines.append(f"  Временные интервалы: распределение Эрланга, k={ERL_K}, средний {AVG_INTERVAL}\n")
    report_lines.append("=" * 80 + "\n\n")

    # Задача 1
    types, type_distribution = generate_message_types()
    report_lines.append("Задача 1: Генерация типов сообщений\n")
    report_lines.append(f"Распределение типов сообщений: {type_distribution}\n")
    report_lines.append("См. график 'output_lab3/message_types.png'\n\n")

    # Задача 2
    addresses, address_distribution = generate_message_addresses(types)
    report_lines.append("Задача 2: Генерация адресов\n")
    report_lines.append(f"Распределение адресов сообщений: {address_distribution}\n")
    report_lines.append("См. график 'output_lab3/message_addresses.png'\n\n")

    # Задача 3
    lengths = generate_message_lengths(types)
    report_lines.append("Задача 3: Генерация длины сообщений\n")
    report_lines.append("Гистограмма длин сообщений расположена в файле: 'output_lab3/message_lengths.png'\n\n")

    # Задача 4
    times = generate_message_times()
    report_lines.append("Задача 4: Генерация времени поступления сообщений\n")
    report_lines.append("График поступления сообщений расположен в файле: 'output_lab3/message_times.png'\n\n")

    # Итоговая таблица сообщений
    message_stream = pd.DataFrame({
        "Тип сообщения": types,
        "Адрес абонента": addresses,
        "Длина сообщения": lengths,
        "Время поступления": times
    }).sort_values("Время поступления")
    table_path = "output_lab3/message_stream.csv"
    message_stream.to_csv(table_path, index=False)
    report_lines.append(f"Итоговая таблица сообщений сохранена в '{table_path}'.\n")
    report_lines.append("=" * 80 + "\n\n")

    # Контрольные вопросы
    report_lines.append("Контрольные вопросы:\n")
    report_lines.append("1. Постановка задач моделирования источников сообщений:\n")
    report_lines.append("Ответ: Задачи включают генерацию сообщений, их адресацию, определение длины и времени их поступления.\n\n")
    report_lines.append("2. Чем характеризуются сообщения?\n")
    report_lines.append("Ответ: Сообщения характеризуются типом, длиной, временем поступления и адресом.\n\n")
    report_lines.append("3. Является ли поток простейшим? Почему?\n")
    report_lines.append("Ответ: Поток не является простейшим, так как используется распределение Эрланга, которое\n")
    report_lines.append("вводит зависимость между временными интервалами поступления сообщений.\n\n")
    report_lines.append("4. Различие между вычисленной и заданной вероятностью?\n")
    report_lines.append("Ответ: Различие возникает из-за малой выборки сообщений. При увеличении объема выборки экспериментальные значения сходятся к теоретическим.\n\n")
    report_lines.append("5. Использованные методы получения случайных величин:\n")
    report_lines.append("Ответ: Применен метод обратной функции (для равномерного распределения) и метод частотного соответствия (для дискретных распределений).\n\n")

    # Сохранение отчета
    with open("output_lab3/report.txt", "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("\nОтчет сформирован и сохранен: 'output_lab3/report.txt'")
    print(f"Итоговая таблица сообщений сохранена: '{table_path}'")


# ------------------------ Основной запуск ------------------------

if __name__ == "__main__":
    generate_report()