"""
Извлечение виртуальных точек из одного скана
по набору опорных координат.

Перед запуском укажите пути и параметры в разделе «НАСТРОЙКИ».

Запуск:
    poetry run python extract_scan_points.py

Файл опорных точек поддерживает форматы:

    name,x,y,z,radius
    P1,614532.123,6601452.456,35.120,0.15

или:

    P1 614532.123 6601452.456 35.120 0.15
    P2 614533.045 6601451.998 35.240

Если радиус не указан для конкретной точки, используется
DEFAULT_RADIUS.

В консоль и CSV сохраняются только точки, имеющие статус SUCCESS:
- окрестность содержит достаточное число точек;
- три плоскости успешно выделены и аппроксимированы;
- геометрия пересечения признана устойчивой;
- оценка точности и ковариационная матрица признаны надёжными;
- рассчитанная точка не удалена от опорной далее установленного допуска.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from app.batch.SingleScanPointExtractor import (
    SingleScanPointExtractor,
)


# =====================================================================
# НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ
# =====================================================================

# Исходный скан в формате LAS или TXT.
SCAN_PATH = Path("data") / "t2" / "scan_2335.las"

# Файл опорных точек: CSV, TSV или TXT.
REFERENCE_POINTS_PATH = (
    Path("data") / "t2" / "vse_tochki.txt"
)

# CSV-файл с надёжно определёнными виртуальными точками.
OUTPUT_CSV_PATH = (
    Path("output") / "epoch_01_virtual_points.csv"
)

# Радиус сферической окрестности по умолчанию, м.
#
# Используется для точек, у которых в файле опорных координат
# не указан индивидуальный радиус.
DEFAULT_RADIUS = 0.25

# Минимальное число точек в извлечённой сферической окрестности.
#
# Значение должно быть не меньше:
#     3 * MIN_POINTS_PER_PLANE
MIN_NEIGHBORHOOD_POINTS = 800

# Минимальное число точек для аппроксимации одной плоскости.
MIN_POINTS_PER_PLANE = 150

# Число ближайших соседей для оценки нормалей.
NORMAL_K = 8

# Радиус DBSCAN-кластеризации пространственных компонент, м.
CLUSTER_EPS = 0.08

# Минимальное число точек в пространственной DBSCAN-компоненте.
CLUSTER_MIN_SAMPLES = 3

# Допустимое удаление вычисленной виртуальной точки
# от исходной опорной точки:
#
#     допустимое расстояние =
#         radius * MAX_REFERENCE_DISTANCE_FACTOR
#
MAX_REFERENCE_DISTANCE_FACTOR = 1.25

# Уровень детальности сообщений:
# DEBUG, INFO, WARNING или ERROR.
LOG_LEVEL = "INFO"

# Если True — первая ошибка по отдельной точке немедленно останавливает
# весь расчёт. Если False — обработка продолжается, а проблемные точки
# учитываются только в итоговой статистике журнала.
FAIL_ON_POINT_ERROR = False


# =====================================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# =====================================================================

def configure_logger() -> None:
    """
    Настраивает компактный цветной вывод Loguru в консоль.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )


def validate_input_paths() -> None:
    """
    Проверяет существование файлов скана и опорных координат.
    """
    if not SCAN_PATH.is_file():
        raise FileNotFoundError(
            "Файл скана не найден:\n"
            f"  {SCAN_PATH.resolve()}"
        )

    if not REFERENCE_POINTS_PATH.is_file():
        raise FileNotFoundError(
            "Файл опорных точек не найден:\n"
            f"  {REFERENCE_POINTS_PATH.resolve()}"
        )


def get_reliable_results(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Возвращает только строки с надёжно определёнными точками.

    Надёжной считается только точка со статусом SUCCESS, который
    назначается SingleScanPointExtractor после прохождения контроля:
    - количества точек в окрестности;
    - сегментации трёх плоскостей;
    - геометрической устойчивости пересечения;
    - наличия принятой ковариационной оценки;
    - расстояния от вычисленной точки до опорной координаты.
    """
    if dataframe.empty:
        return dataframe.copy()

    return dataframe.loc[
        dataframe["status"]
        == SingleScanPointExtractor.SUCCESS
    ].copy()


def print_reliable_points(
    reliable_dataframe: pd.DataFrame,
) -> None:
    """
    Выводит в консоль только надёжно определённые виртуальные точки.
    """
    print()

    if reliable_dataframe.empty:
        logger.warning(
            "Надёжно определённые виртуальные точки отсутствуют."
        )
        return

    console_columns = [
        "name",
        "x",
        "y",
        "z",
        "sigma_x",
        "sigma_y",
        "sigma_z",
    ]

    print("Надёжно определённые виртуальные точки")
    print("-" * 96)

    print(
        reliable_dataframe[
            console_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("-" * 96)


def export_reliable_points(
    reliable_dataframe: pd.DataFrame,
) -> Path:
    """
    Сохраняет только надёжные результаты в UTF-8 CSV.

    В файл не попадают строки со статусами:
    - UNRELIABLE;
    - FAILED.
    """
    OUTPUT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reliable_dataframe.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    logger.success(
        "CSV-файл с надёжными точками сохранён: {}",
        OUTPUT_CSV_PATH.resolve(),
    )

    return OUTPUT_CSV_PATH


def log_processing_summary(
    dataframe: pd.DataFrame,
    reliable_dataframe: pd.DataFrame,
) -> None:
    """
    Выводит итоговую статистику процесса обработки.
    """
    total_count = len(dataframe)
    success_count = len(reliable_dataframe)

    unreliable_count = int(
        (
            dataframe["status"]
            == SingleScanPointExtractor.UNRELIABLE
        ).sum()
    )

    failed_count = int(
        (
            dataframe["status"]
            == SingleScanPointExtractor.FAILED
        ).sum()
    )

    logger.success(
        "Расчёт завершён: всего={}, успешно={}, "
        "ненадёжно={}, ошибки={}",
        total_count,
        success_count,
        unreliable_count,
        failed_count,
    )

    logger.success(
        "В итоговый CSV экспортировано надёжных точек: {}",
        success_count,
    )


# =====================================================================
# ОСНОВНОЙ ЗАПУСК
# =====================================================================

def main() -> None:
    """
    Запускает полный цикл обработки одного скана.

    Последовательность:
    1. Проверка входных путей.
    2. Загрузка скана.
    3. Чтение опорных точек.
    4. Выделение локальных сферических окрестностей.
    5. Извлечение виртуальных точек.
    6. Фильтрация только надёжных результатов.
    7. Вывод надёжных точек в консоль.
    8. Экспорт надёжных точек в CSV.
    """
    configure_logger()
    validate_input_paths()

    logger.info("Начало обработки одного скана")
    logger.info("Скан: {}", SCAN_PATH.resolve())

    logger.info(
        "Опорные точки: {}",
        REFERENCE_POINTS_PATH.resolve(),
    )

    logger.info(
        "Итоговый CSV: {}",
        OUTPUT_CSV_PATH.resolve(),
    )

    logger.info(
        "Параметры: radius={} м, min_neighborhood_points={}, "
        "min_points_per_plane={}, normal_k={}, "
        "cluster_eps={} м, cluster_min_samples={}",
        DEFAULT_RADIUS,
        MIN_NEIGHBORHOOD_POINTS,
        MIN_POINTS_PER_PLANE,
        NORMAL_K,
        CLUSTER_EPS,
        CLUSTER_MIN_SAMPLES,
    )

    extractor = SingleScanPointExtractor.from_files(
        scan_path=SCAN_PATH,
        reference_points_path=REFERENCE_POINTS_PATH,
        default_radius=DEFAULT_RADIUS,
        min_neighborhood_points=MIN_NEIGHBORHOOD_POINTS,
        min_points_per_plane=MIN_POINTS_PER_PLANE,
        max_reference_distance_factor=(
            MAX_REFERENCE_DISTANCE_FACTOR
        ),
        normal_k=NORMAL_K,
        cluster_eps=CLUSTER_EPS,
        cluster_min_samples=CLUSTER_MIN_SAMPLES,
    )

    extractor.run(
        fail_on_point_error=FAIL_ON_POINT_ERROR
    )

    all_results_dataframe = extractor.to_dataframe()

    reliable_results_dataframe = get_reliable_results(
        dataframe=all_results_dataframe
    )

    print_reliable_points(
        reliable_dataframe=reliable_results_dataframe
    )

    export_reliable_points(
        reliable_dataframe=reliable_results_dataframe
    )

    log_processing_summary(
        dataframe=all_results_dataframe,
        reliable_dataframe=reliable_results_dataframe,
    )


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        logger.error(
            "Ошибка входных данных: {}",
            error,
        )
        raise

    except ValueError as error:
        logger.error(
            "Ошибка параметров или обработки: {}",
            error,
        )
        raise

    except Exception:
        logger.exception(
            "Непредвиденная ошибка при обработке скана"
        )
        raise