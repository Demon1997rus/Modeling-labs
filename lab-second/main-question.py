import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, uniform, expon

# Создаем директорию для сохранения графиков
os.makedirs("output_lab2", exist_ok=True)


# Вопрос 1: Что такое распределение случайной величины и какими функциями оно характеризуется?
def question_1():
    """
    Ответ на первый контрольный вопрос текстом.
    """
    answer = """
    Ответ на вопрос 1:
    Распределение случайной величины - это описание того, с какой вероятностью случайная величина X принимает разные значения.
    Оно характеризуется:
    
    1. Для дискретной случайной величины (например, количество выпадений орла при подбрасывании монеты):
       - Вероятностная функция p(x): вероятность того, что X примет конкретное значение x.
       - Функция распределения F(x): вероятность того, что X <= x.
    
    2. Для непрерывной случайной величины (например, рост людей):
       - Функция плотности вероятности f(x): производная функции распределения F(x).
       - Функция распределения F(x): вероятность того, что X <= x.
    """
    return answer


# Вопрос 2: Как вычисляются функции f(x) по F(x) и наоборот F(x) по f(x)?
def question_2():
    """
    Ответ на второй контрольный вопрос текстом.
    """
    answer = """
    Ответ на вопрос 2:
    Связь между функциями f(x) и F(x):
    
    1. Для непрерывной случайной величины:
       - Если известна F(x), то f(x) = dF(x)/dx (f(x) — это производная от F(x)).
       - Если известна f(x), то F(x) = ∫ f(t) dt (интегрирование плотности вероятности от минус бесконечности до x).
    
    2. Для дискретной случайной величины:
       - Если известна вероятность P(x), F(x) вычисляется как сумма вероятностей: F(x) = Σ P(x), где сумма берется для всех значений меньше или равных x.
    """
    return answer


# Вопрос 3: Качественно изобразить графики функций f(x) и F(x) для основных видов распределений случайных величин.
def question_3():
    """
    Строятся графики функций f(x) и F(x) для нормального, равномерного и экспоненциального распределений.
    """
    x = np.linspace(-5, 10, 1000)

    # Нормальное распределение
    norm_pdf = norm.pdf(x)
    norm_cdf = norm.cdf(x)

    # Равномерное распределение
    uniform_pdf = uniform.pdf(x, loc=0, scale=5)
    uniform_cdf = uniform.cdf(x, loc=0, scale=5)

    # Экспоненциальное распределение
    exp_pdf = expon.pdf(x, scale=2)
    exp_cdf = expon.cdf(x, scale=2)

    # Построение графиков
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title("Функции плотности вероятности f(x)")
    plt.plot(x, norm_pdf, label="Нормальное", color="blue")
    plt.plot(x, uniform_pdf, label="Равномерное", color="orange")
    plt.plot(x, exp_pdf, label="Экспоненциальное", color="green")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("Функции распределения вероятности F(x)")
    plt.plot(x, norm_cdf, label="Нормальное", color="blue")
    plt.plot(x, uniform_cdf, label="Равномерное", color="orange")
    plt.plot(x, exp_cdf, label="Экспоненциальное", color="green")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    filepath = "output_lab2/distributions_functions.png"
    plt.savefig(filepath)
    plt.close()

    return f"""
    Ответ на вопрос 3:
    Графики функций плотности вероятности f(x) и функции распределения вероятности F(x) для нормального,
    равномерного и экспоненциального распределений сохранены в файле: '{filepath}'.
    Вставьте их в отчет.
    """


# Вопрос 4: Сформулируйте основные положения метода обратной функции.
def question_4():
    """
    Ответ на четвёртый контрольный вопрос текстом.
    """
    answer = """
    Ответ на вопрос 4:
    
    Метод обратной функции предполагает:
    1. Генерация случайного числа U из равномерного распределения на [0, 1].
    2. Вычисление значения X, используя обратную функцию распределения F^-1(U).
    3. Таким образом, случайная величина X будет иметь заданное распределение F(x).
    """
    return answer


# Вопрос 5: Сформулируйте основные положения табличного метода.
def question_5():
    """
    Ответ на пятый контрольный вопрос текстом.
    """
    answer = """
    Ответ на вопрос 5:
    Табличный метод предполагает:
    1. Построение таблицы значений дискретной величины X и соответствующих ей значений кумулятивной функции распределения F(X).
    2. Генерацию случайного числа U из равномерного распределения [0, 1].
    3. Путём сравнения U с F(X) выбирается значение X, удовлетворяющее условию F(X) >= U.
    """
    return answer


# Вопрос 6: Постройте графики P(x) и F(x) для дискретного распределения.
def question_6():
    """
    Постройка графиков для дискретного распределения и решение методом обратной функции.
    """
    values = [0, 1, 2]
    probabilities = [0.5, 0.3, 0.2]
    random_numbers = [0.025, 0.91, 0.37, 0.26, 0.31]

    # Вычисляем F(X)
    cumulative_probabilities = np.cumsum(probabilities)

    # Преобразуем метод обратной функции
    result_sample = []
    for rn in random_numbers:
        for idx, cp in enumerate(cumulative_probabilities):
            if rn <= cp:
                result_sample.append(values[idx])
                break

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(values, probabilities, color="blue", alpha=0.7, label="P(X)")
    plt.title("Гистограмма вероятностей P(X)")
    plt.xlabel("Значения X")
    plt.ylabel("P(X)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.step([0] + values, [0] + list(cumulative_probabilities), where="post", color="red", label="F(X)")
    plt.title("Функция распределения F(X)")
    plt.xlabel("Значения X")
    plt.ylabel("F(X)")
    plt.legend()
    plt.grid()

    filepath = "output_lab2/discrete_distribution.png"
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return f"""
    Ответ на вопрос 6:
    Графики P(X) и F(X) для дискретного распределения сохранены в файле: '{filepath}'.
    Преобразованные случайные числа: {random_numbers}
    Полученная выборка: {result_sample}
    Вставьте эти графики в отчет.
    """


# Главный запуск всех вопросов
if __name__ == "__main__":
    print(question_1())
    print(question_2())
    print(question_3())
    print(question_4())
    print(question_5())
    print(question_6())