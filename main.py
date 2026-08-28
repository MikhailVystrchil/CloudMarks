import os
import sys
from pathlib import Path

from loguru import logger

from app.cross_points.CrossPointExacter import CrossPointExacter


def configure_logger() -> None:
    """
    Настраивает единый консольный вывод Loguru.

    Уровень логирования задаётся через переменную окружения LOG_LEVEL:
        INFO  — штатный вывод хода обработки;
        DEBUG — расширенная диагностика сегментации и выбора кластеров;
        WARNING / ERROR — только предупреждения и ошибки.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    logger.info("Логирование Loguru настроено. Уровень: {}", log_level)


def extract_virtual_point(
    file_path: str | Path,
    point_name: str,
) -> object:
    """
    Выделяет одну виртуальную точку — пересечение трёх аппроксимирующих плоскостей.

    Parameters
    ----------
    file_path:
        Путь к локальному фрагменту облака точек формата LAS или TXT.
    point_name:
        Устойчивый идентификатор виртуальной точки между эпохами.

    Returns
    -------
    CrossPoint
        Виртуальная точка с координатами, ковариацией и статусом геометрии.
    """
    file_path = Path(file_path)

    logger.info("Начало обработки виртуальной точки '{}'", point_name)
    logger.info("Входной файл: {}", file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Файл облака точек не найден: {file_path}")

    logger.info("Этап 1/4. Загрузка облака и сегментация локальных плоскостей")
    exacter = CrossPointExacter(
        file_path=str(file_path),
        show_scans=False,
    )

    logger.info("Этап 2/4. Робастная очистка и МНК-аппроксимация плоскостей")
    planes = exacter.calculate_planes()

    for index, plane in enumerate(planes, start=1):
        logger.info(
            "Плоскость {}/{}: точек после очистки={}, RMSE={:.6f}, "
            "A={:.6f}, B={:.6f}, C={:.6f}, D={:.6f}",
            index,
            len(planes),
            len(plane.scan),
            plane.mse,
            plane.A,
            plane.B,
            plane.C,
            plane.D,
        )

    logger.info("Этап 3/4. Контроль геометрической устойчивости пересечения")
    diagnostics = exacter.diagnose_geometry()

    logger.info(
        "Геометрия: status={}, det(N)={:.6f}, cond(N)={:.2f}, parallel={}",
        diagnostics.status,
        diagnostics.det,
        diagnostics.cond,
        diagnostics.has_parallel,
    )

    if not diagnostics.is_reliable:
        logger.warning(
            "Геометрия виртуальной точки признана неустойчивой: {}",
            "; ".join(diagnostics.messages),
        )

    logger.info("Этап 4/4. Вычисление точки пересечения и перенос ковариации")
    point = exacter.calculate_intersect_point()
    point.name = point_name

    logger.success(
        "Точка '{}' вычислена: X={:.6f}, Y={:.6f}, Z={:.6f}, status={}",
        point.name,
        point.x,
        point.y,
        point.z,
        point.status,
    )

    if point.reliable_accuracy and point.sigma_xyz is not None:
        sigma_x, sigma_y, sigma_z = point.sigma_xyz
        logger.info(
            "СКП координат: sigma_X={:.6f}, sigma_Y={:.6f}, sigma_Z={:.6f}",
            sigma_x,
            sigma_y,
            sigma_z,
        )

        if point.ellipsoid is not None:
            semi_axes = point.ellipsoid["semi_axes"]
            confidence = point.ellipsoid["confidence"]
            logger.info(
                "Полуоси эллипсоида погрешности (P={:.0%}): "
                "a={:.6f}, b={:.6f}, c={:.6f}",
                confidence,
                semi_axes[0],
                semi_axes[1],
                semi_axes[2],
            )
    else:
        logger.warning(
            "Координаты точки рассчитаны, но ковариация не принята: "
            "геометрия пересечения ненадёжна."
        )

    logger.info("Обработка виртуальной точки '{}' завершена", point_name)
    return point


def main() -> None:
    configure_logger()

    # Замените путь и, если необходимо, DBSCAN-метки на параметры вашего опыта.
    input_file = Path("data") / "1_A_10_2_l.txt"

    try:
        extract_virtual_point(
            file_path=input_file,
            point_name="P1",
        )
        logger.success("Расчёт успешно завершён.")
    except FileNotFoundError as error:
        logger.error("Ошибка входных данных: {}", error)
        raise
    except ValueError as error:
        logger.error("Невозможно вычислить виртуальную точку: {}", error)
        raise
    except Exception:
        logger.exception("Непредвиденная ошибка при обработке облака точек")
        raise


if __name__ == "__main__":
    main()
