import json
import math

from pydantic import BaseModel, model_validator


# Задача 1: Типы сообщений и вероятности
class Task1Config(BaseModel):
    types: list[int]  # Список типов сообщений
    probabilities: list[float]  # Вероятности появления сообщений каждого типа
    num_messages: int  # Количество сообщений
    results_dir: str  # Путь к папке с результатами

    @model_validator(mode="after")
    def validate_lengths_and_non_empty(self):
        if not self.types:
            raise ValueError("Поле 'types' не должно быть пустым.")
        if not self.probabilities:
            raise ValueError("Поле 'probabilities' не должно быть пустым.")
        if len(self.types) != len(self.probabilities):
            raise ValueError("Поля 'types' и 'probabilities' должны быть одинаковой длины.")
        if self.num_messages <= 0:
            raise ValueError("Поле 'num_messages' должно быть положительным числом.")
        total_probability = sum(self.probabilities)
        if not math.isclose(total_probability, 1.0):
            raise ValueError(
                f"Сумма значений в 'probabilities' должна быть равна 1. Текущее значение: {total_probability}"
            )
        return self


class Config(BaseModel):
    task1: Task1Config


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return Config(**json_data)
