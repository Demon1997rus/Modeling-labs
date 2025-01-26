import numpy as np
from scipy.special import factorial

K = [1, 3, 1, 2, 4]  # Число каналов в каждой СМО: K1, K2, K3, K4, K5
lc_inv = 2
lambda_0 = 1 / lc_inv
V = 2.5
nu = 1 / V

P = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.8],
        [0.0, 0.3, 0.0, 0.0, 0.7],
        [0.0, 0.6, 0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.9, 0.1],
    ]
)

for i in range(len(P)):
    row_sum = np.sum(P[i])
    if row_sum != 1.0:
        print(f"Ошибка: строка {i + 1} матрицы не нормирована (сумма = {row_sum})!")

n = len(P)
lambda_ = np.zeros(n)
lambda_[0] = lambda_0

for _ in range(1000):
    lambda_new = np.dot(lambda_, P)
    if np.allclose(lambda_, lambda_new, atol=1e-6):
        break
    lambda_ = lambda_new

if not np.allclose(lambda_, np.dot(lambda_, P), atol=1e-6):
    print("Внимание: система потоков не сошлась!")

rho = []
beta = []
pi_0 = []
L_q = []
m = []
w = []
u = []

for i in range(n):
    rho_i = lambda_[i] / (nu * K[i])
    rho.append(rho_i)

    sum_term = sum((K[i] * rho_i) ** k / factorial(k, exact=True) for k in range(K[i] + 1))
    pi_0_i = 1 / sum_term
    pi_0.append(pi_0_i)

    if rho_i < 1:
        L_q_i = (pi_0_i * (K[i] * rho_i) ** K[i] * rho_i) / (factorial(K[i], exact=True) * (1 - rho_i) ** 2)
    else:
        L_q_i = 0
    L_q.append(L_q_i)

    m_i = L_q_i + K[i] * rho_i
    m.append(m_i)

    if lambda_[i] > 0:
        w_i = L_q_i / lambda_[i]
    else:
        w_i = 0
    w.append(w_i)

    u_i = w_i + (1 / nu)
    u.append(u_i)

    beta_i = K[i] * rho_i
    beta.append(beta_i)

L_q_total = sum(L_q)
beta_total = sum(beta)
m_total = sum(m)
w_total = L_q_total / lambda_0
u_total = w_total + (1 / nu)

print("Результаты расчётов для сети:")
for i in range(n):
    print(f"\nСистема {i + 1}:")
    print(f"  Загрузка (ρ): {rho[i]:.4f}")
    print(f"  Среднее число занятых каналов (β): {beta[i]:.4f}")
    print(f"  Вероятность простоя (π0): {pi_0[i]:.4f}")
    print(f"  Средняя длина очереди (Lq): {L_q[i]:.4f}")
    print(f"  Среднее число заявок в системе (m): {m[i]:.4f}")
    print(f"  Среднее время ожидания заявки (w): {w[i]:.4f}")
    print(f"  Среднее время пребывания заявки в системе (u): {u[i]:.4f}")

print("\nХарактеристики сети в целом:")
print(f"  Среднее число заявок в очередях (Lq_total): {L_q_total:.4f}")
print(f"  Среднее число занятых каналов (β_total): {beta_total:.4f}")
print(f"  Среднее число заявок в системе (m_total): {m_total:.4f}")
print(f"  Среднее время ожидания в сети (w_total): {w_total:.4f}")
print(f"  Среднее время пребывания в сети (u_total): {u_total:.4f}")
