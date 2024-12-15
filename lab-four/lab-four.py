import numpy as np
import math  # Для вычисления факториалов
import matplotlib.pyplot as plt
import os

# Данные для ученика (вариант)
os.makedirs("output_lab4", exist_ok=True)

# Вводные данные для расчёта
LAMBDA_0 = 2.0  # Интенсивность входящего потока (заявка/с)
V = 2.5  # Среднее время обслуживания заявки в любой системе (с)
K = [1, 3, 1, 2, 4]  # Количество каналов в системах (K1, K2, K3, K4, K5)

# Матрица вероятностей передач
P14 = 0.25  # Вероятность перехода из S1 в S4
P21 = 0.20  # Вероятность перехода из S2 в S1
P32 = 0.33  # Вероятность перехода из S3 в S2
P42 = 0.10  # Вероятность перехода из S4 в S2
P54 = 0.40  # Вероятность перехода из S5 в S4

# Построение матрицы вероятностей передач
def build_probability_matrix():
    """
    Создаёт матрицу вероятностей передач P_ij для сети.
    """
    P = np.zeros((6, 6))  # Матрица (6x6): 5 СМО + источник S_0
    P[0, 1] = 1.0  # Исходный поток S_0 направляется в S_1
    P[1, 4] = P14  # Переход из S_1 в S_4
    P[2, 0] = P21  # Переход из S_2 в S_0
    P[3, 1] = P32  # Переход из S_3 в S_2
    P[4, 2] = P42  # Переход из S_4 в S_2
    P[5, 3] = P54  # Переход из S_5 в S_4
    return P

# Расчет интенсивностей потоков заявок для каждой СМО
def calculate_intensities(P, lambda_0):
    """
    Рассчитывает интенсивности потоков λ для всех систем массового обслуживания.
    """
    n = P.shape[0] - 1
    coeff_matrix = np.eye(n) - P[1:, 1:]  # Матрица для λ (с угловой единицей)
    rhs = P[1:, 0] * lambda_0  # Правая часть уравнения: входящие потоки из S_0
    lambdas = np.linalg.solve(coeff_matrix, rhs)
    return lambdas

# Расчет характеристик СМО (загрузка, вероятность простоя и прочее)
def calculate_smo_characteristics(lambdas, K, V):
    """
    Рассчитывает основные характеристики СМО: загрузка, каналы, очереди, времена ожидания.
    """
    results = []
    for i, (lambda_i, k_i) in enumerate(zip(lambdas, K), start=1):
        rho_i = lambda_i * V / k_i  # Загрузка системы
        beta_i = lambda_i * V  # Среднее число занятых каналов
        pi_0 = 1 - rho_i if k_i == 1 else 1 / (1 + sum((rho_i ** n) / math.factorial(n) for n in range(1, k_i + 1)))
        l_i = beta_i * rho_i / (1 - rho_i)  # Средняя длина очереди заявок
        w_i = l_i / lambda_i  # Среднее время ожидания
        u_i = w_i + V  # Среднее время пребывания заявки
        results.append({
            "Загрузка (ρ)": rho_i,
            "Среднее число занятых каналов (β)": beta_i,
            "Вероятность простоя (π₀)": pi_0,
            "Средняя очередь (l)": l_i,
            "Время ожидания (w)": w_i,
            "Время пребывания (u)": u_i
        })
    return results

# Построение структурной схемы сети
def build_network_scheme(P):
    """
    Строит графическую структуру сети.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    node_positions = {0: (0.1, 0.5), 1: (0.3, 0.7), 2: (0.3, 0.3), 3: (0.5, 0.7), 4: (0.7, 0.5), 5: (0.9, 0.7)}
    for node, (x, y) in node_positions.items():
        ax.scatter(x, y, s=800, color='skyblue')
        ax.text(x, y, f'S{node}', color='black', fontsize=12, ha='center')
    for src, dest in zip(*np.where(P > 0)):
        src_x, src_y = node_positions[src]
        dest_x, dest_y = node_positions[dest]
        ax.annotate(f"{P[src, dest]:.2f}",
                    ((src_x + dest_x) / 2, (src_y + dest_y) / 2),
                    color='red', fontsize=10, ha="center", va="center")
        ax.arrow(src_x, src_y, (dest_x - src_x) * 0.85, (dest_y - src_y) * 0.85,
                 head_width=0.02, head_length=0.03, fc='black', ec="black")
    plt.savefig("output_lab4/network_scheme.png")
    plt.close()

# Генерация отчета
def generate_report():
    """
    Полный текст отчета с описанием характеристик.
    """
    report = ["Лабораторная работа №4: Стохастические сетевые модели вычислительных систем\n"]
    report.append("\nЦель: Изучение разомкнутых стохастических сетей и их характеристик.\n")
    report.append("=" * 80 + "\n\n")
    P = build_probability_matrix()
    report.append(f"Матрица вероятностей передач: \n{P}\n")
    lambdas = calculate_intensities(P, LAMBDA_0)
    report.append(f"\nИнтенсивности потоков заявок λ:\n{lambdas}\n")
    char_res = calculate_smo_characteristics(lambdas, K, V)
    report.append("\nХарактеристики СМО:\n")
    for i, char in enumerate(char_res, start=1):
        report.append(f"СМО S{i}: {char}\n")
    build_network_scheme(P)
    report.append("\nСтруктура сети сохранена в 'output_lab4/network_scheme.png'.\n")
    with open("output_lab4/report.txt", "w", encoding="utf-8") as f:
        f.writelines(report)
    print("\nРезультат в 'output_lab4/report.txt'")

# Запуск
if __name__ == "__main__":
    generate_report()