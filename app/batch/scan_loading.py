from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.scan.Scan import Scan


def load_scan_from_file(
    file_path: str | Path,
    *,
    scan_name: str | None = None,
) -> Scan:
    """
    Загружает LAS/TXT-скан и проверяет, что он содержит точки.

    Parameters
    ----------
    file_path:
        Путь к входному LAS- или TXT-файлу.
    scan_name:
        Имя создаваемого Scan. Если не задано, используется stem файла.

    Returns
    -------
    Scan
        Загруженное непустое облако точек.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Файл скана не найден: {path}"
        )

    resolved_scan_name = scan_name or path.stem

    logger.info(
        "Загрузка скана '{}': {}",
        resolved_scan_name,
        path,
    )

    scan = Scan(resolved_scan_name)
    scan.import_points_from_file(
        file_path=str(path),
        compute_normals=False,
    )

    if len(scan) == 0:
        raise ValueError(
            f"Скан '{path}' не содержит точек."
        )

    logger.success(
        "Скан '{}' загружен: {} точек",
        scan.name,
        len(scan),
    )

    return scan
