# MLOps — Student Performance Prediction

## Описание проекта

Лабораторная работа №1
«Воспроизводимый проект: данные и пайплайн»

Цель проекта — собрать проект, результат которого воспроизводится другим 
человеком на другой машине одной последовательностью команд, без переписки с автором.

Используется датасет Student Performance (UCI).
Целевая переменная — `G3`.

## Используемые технологии

- Python
- Pandas, NumPy
- Scikit-learn
- DVC
- DagsHub Storage
- Git и GitHub
- Ruff и pre-commit
- pytest
- uv

## Этапы ML-пайплайна
1. Prepare — загрузка и разделение исходных данных.
2. Featurize — обработка числовых и категориальных признаков.
3. Train — обучение модели Random Forest Regressor.
4. Evaluate — оценка качества модели с помощью MAE, RMSE и R².

## Установка и запуск
Клонирование репозитория:
```python
git clone https://github.com/Magistr556/MlOps8.git
cd MlOps8
```
## Установка зависимостей:
```python
pip install uv
uv sync --all-groups
```

## Получение данных из DagsHub Storage:
```python
dvc pull
```

## Запуск полного пайплайна:
```python
dvc repro
```

## Просмотр метрик:
```python
dvc metrics show
```

## Запуск тестов:
```python
pytest -v

## Проверка качества кода:
```python
pre-commit run --all-files
```

## Версионирование данных
Для хранения и версионирования исходных данных используется DVC.
Git хранит код, конфигурации и метафайлы DVC.
Сами данные размещаются в удалённом хранилище DagsHub.

Для восстановления конкретной версии проекта используются
Git-теги и команда:
```python
git checkout v1
dvc checkout
```

## Воспроизводимость
Параметры эксперимента задаются в params.yaml.
Зависимости между этапами определены в dvc.yaml.

Повторный запуск dvc repro не пересчитывает неизменившиеся
этапы пайплайна.


## Структура проекта

```text
MlOps8/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── prepare.py
│   ├── featurize.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_pipeline.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
└── README.md
