# Лабораторные работы по моделированию

## О проекте

Данный проект включает выполнения набора лабораторных работ по моделированию систем.

---

## Установка и запуск проекта

### 1. Клонирование репозитория
Склонируйте проект на своё устройство:
```bash
git clone git@github.com:Demon1997rus/Modeling-labs.git
cd Modeling-labs
```
### 2. Настройка виртуальной среды и установка зависимостей
Создайте виртуальную среду:
```bash
python3 -m venv venv
```

Активируйте виртуальную среду:
Для Linux/macOS:
```bash
source venv/bin/activate
```
Для Windows:
```bash
venv\Scripts\activate
```
### 3. Установите зависимости из файла requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Для работы с вентилятором (RPM) на Linux убедитесь, что установлена утилита lm-sensors:
```bash
sudo apt install lm-sensors
sudo sensors-detect
sensors
```
