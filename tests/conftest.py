import numpy as np
import pytest

from app.scan.Scan import Scan
from app.scan.ScanPoint import ScanPoint
from app.scan.ScanPlane import ScanPlane


@pytest.fixture
def make_scan():
    """
    Фабрика тестового объекта Scan.

    Возвращает функцию, создающую Scan из массива координат формы (N, 3).
    Файловые парсеры, LAS/TXT-файлы и вычисление нормалей не используются.
    """

    def _make_scan(name: str, coordinates: np.ndarray) -> Scan:
        scan = Scan(name)

        for x_coord, y_coord, z_coord in np.asarray(
            coordinates,
            dtype=float,
        ):
            scan.add_point(
                ScanPoint(
                    x=float(x_coord),
                    y=float(y_coord),
                    z=float(z_coord),
                )
            )

        return scan

    return _make_scan


@pytest.fixture
def make_scan_plane():
    """
    Фабрика синтетических плоскостей ScanPlane.

    Плоскость задаётся единичной нормалью (A, B, C) и свободным членом D:
        A*x + B*y + C*z + D = 0.

    Возвращаемый объект содержит искусственно заданные:
    - mse;
    - sigma0;
    - cov_params — матрицу 4x4 для (A, B, C, D).

    Это позволяет тестировать диагностику геометрии и перенос ковариации
    без загрузки облака точек и выполнения МНК.
    """

    def _make_scan_plane(
        normal: tuple[float, float, float],
        d: float,
        covariance: np.ndarray | None = None,
        mse: float = 0.001,
    ) -> ScanPlane:
        normal_array = np.asarray(normal, dtype=float)

        normal_norm = np.linalg.norm(normal_array)
        if normal_norm <= 1e-15:
            raise ValueError("Нормаль плоскости не может быть нулевой.")

        normal_array = normal_array / normal_norm

        nonzero_index = int(np.argmax(np.abs(normal_array)))
        point_on_plane = np.zeros(3, dtype=float)
        point_on_plane[nonzero_index] = (
            -float(d) / normal_array[nonzero_index]
        )

        plane = ScanPlane(
            normal=normal_array,
            point_on_plane=point_on_plane,
            d=float(d),
        )

        plane.mse = float(mse)
        plane.sigma0 = float(mse)

        if covariance is None:
            covariance = np.eye(4, dtype=float) * mse ** 2

        plane.cov_params = np.asarray(covariance, dtype=float)

        return plane

    return _make_scan_plane


@pytest.fixture
def orthogonal_planes(make_scan_plane) -> list[ScanPlane]:
    """
    Три плоскости:
        x - 1 = 0;
        y - 2 = 0;
        z - 3 = 0.

    Их точка пересечения равна (1, 2, 3).
    """
    covariance = np.diag([1e-8, 1e-8, 1e-8, 1e-8])

    return [
        make_scan_plane((1.0, 0.0, 0.0), -1.0, covariance),
        make_scan_plane((0.0, 1.0, 0.0), -2.0, covariance),
        make_scan_plane((0.0, 0.0, 1.0), -3.0, covariance),
    ]