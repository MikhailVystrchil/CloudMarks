from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from app.base.Point import Point
from app.scan.ScanPoint import ScanPoint
from app.scan.parsers.ScanParserFactory import ScanParserFactory
from app.scan.plotters.ScanPlotterMPL import ScanPlotterMPL
from app.scan.utils.ScanNormalsCalculator import ScanNormalsCalculator


class Scan:
    """
    Облако точек наземного лазерного сканирования.

    Основные публичные операции:
    - добавление точки;
    - загрузка из LAS/TXT;
    - вычисление нормалей;
    - получение массива координат;
    - безопасное выделение независимого подскана по индексам.
    """

    def __init__(
        self,
        scan_name: str,
    ) -> None:
        self.name = str(scan_name)
        self._points: list[ScanPoint] = []
        self.borders = self._empty_borders()

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[ScanPoint]:
        return iter(self._points)

    def __getitem__(
        self,
        index: int | slice,
    ) -> ScanPoint | list[ScanPoint]:
        return self._points[index]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(scan_name={self.name}, "
            f"point_count={len(self)}, "
            f"borders={self.borders})"
        )

    @staticmethod
    def _empty_borders() -> dict[str, float | None]:
        return {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        }

    @staticmethod
    def _update_borders(
        borders: dict[str, float | None],
        point: Point,
    ) -> dict[str, float | None]:
        """
        Возвращает границы, дополненные координатами одной точки.
        """
        if borders["x_min"] is None:
            return {
                "x_min": point.x,
                "x_max": point.x,
                "y_min": point.y,
                "y_max": point.y,
                "z_min": point.z,
                "z_max": point.z,
            }

        borders["x_min"] = min(borders["x_min"], point.x)
        borders["x_max"] = max(borders["x_max"], point.x)
        borders["y_min"] = min(borders["y_min"], point.y)
        borders["y_max"] = max(borders["y_max"], point.y)
        borders["z_min"] = min(borders["z_min"], point.z)
        borders["z_max"] = max(borders["z_max"], point.z)

        return borders

    @classmethod
    def from_points(
        cls,
        scan_name: str,
        points: Iterable[ScanPoint | Point],
        *,
        copy_points: bool = False,
        include_normals: bool = True,
        include_labels: bool = True,
    ) -> "Scan":
        """
        Создаёт Scan из коллекции точек.

        Parameters
        ----------
        scan_name:
            Имя нового скана.
        points:
            Коллекция ``Point`` или ``ScanPoint``.
        copy_points:
            Если True, для ScanPoint создаются независимые копии.
        include_normals:
            Копировать ли нормали при ``copy_points=True``.
        include_labels:
            Копировать ли метки сегментации при ``copy_points=True``.
        """
        scan = cls(scan_name=scan_name)

        for point in points:
            if isinstance(point, ScanPoint):
                scan_point = (
                    point.copy(
                        include_normals=include_normals,
                        include_labels=include_labels,
                    )
                    if copy_points
                    else point
                )
            elif isinstance(point, Point):
                scan_point = ScanPoint(
                    x=point.x,
                    y=point.y,
                    z=point.z,
                )
            else:
                raise TypeError(
                    "points должен содержать объекты Point или ScanPoint."
                )

            scan.add_point(scan_point)

        return scan

    def add_point(
        self,
        point: Point,
        color: Sequence[int] = (0, 0, 0),
    ) -> "Scan":
        """
        Добавляет точку и обновляет границы Scan.
        """
        if isinstance(point, ScanPoint):
            scan_point = point
        elif isinstance(point, Point):
            scan_point = ScanPoint(
                x=point.x,
                y=point.y,
                z=point.z,
                color=color,
            )
        else:
            raise TypeError(
                "point должен быть экземпляром Point или ScanPoint."
            )

        self._points.append(scan_point)
        self.borders = self._update_borders(
            borders=self.borders,
            point=scan_point,
        )

        return self

    def to_numpy(
        self,
        *,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """
        Возвращает координаты всех точек как массив формы ``(N, 3)``.
        """
        if not self._points:
            return np.empty((0, 3), dtype=dtype)

        return np.asarray(
            [
                point.as_array()
                for point in self._points
            ],
            dtype=dtype,
        )

    def subset(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        scan_name: str | None = None,
        copy_points: bool = True,
        include_normals: bool = True,
        include_labels: bool = True,
    ) -> "Scan":
        """
        Создаёт подскан по индексам исходного облака.

        По умолчанию точки копируются независимо. Это предотвращает
        модификацию нормалей, цветов и сегментационных меток большого скана
        при обработке локальных окрестностей.
        """
        index_array = np.asarray(indices, dtype=np.intp)

        if index_array.ndim != 1:
            raise ValueError(
                "indices должен быть одномерным массивом индексов."
            )

        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError(
                "indices содержит индекс вне диапазона Scan."
            )

        selected_points = [
            self._points[int(index)]
            for index in index_array
        ]

        return self.from_points(
            scan_name=scan_name or f"{self.name}_subset",
            points=selected_points,
            copy_points=copy_points,
            include_normals=include_normals,
            include_labels=include_labels,
        )

    def compute_normals(
        self,
        *args,
        normals_calculator: type[ScanNormalsCalculator] = ScanNormalsCalculator,
        **kwargs,
    ) -> np.ndarray:
        """
        Вычисляет и присваивает нормали всем точкам Scan.
        """
        calculator = normals_calculator(scan=self)
        return calculator.compute_normals(*args, **kwargs)

    def import_points_from_file(
        self,
        file_path: str,
        *,
        parser: type[ScanParserFactory] = ScanParserFactory,
        compute_normals: bool = True,
    ) -> "Scan":
        """
        Загружает точки из файла и при необходимости вычисляет нормали.
        """
        parser_instance = parser(file_path)
        parser_instance.parse(scan=self)

        if compute_normals:
            self.compute_normals()

        return self

    def filter_scan(
        self,
        filter_cls: type,
        *args,
        replace_points_in_scan: bool = True,
        **kwargs,
    ) -> "Scan":
        """
        Применяет внешний фильтр, поддерживающий метод ``filter(scan=...)``.
        """
        filter_instance = filter_cls(*args, **kwargs)
        filtered_points = filter_instance.filter(scan=self)

        filtered_scan = self.from_points(
            scan_name=f"{self.name}_filtered",
            points=filtered_points,
            copy_points=False,
        )

        if replace_points_in_scan:
            self._points = list(filtered_scan._points)
            self.borders = filtered_scan.borders
            return self

        return filtered_scan

    def plot(
        self,
        *args,
        plotter: type[ScanPlotterMPL] = ScanPlotterMPL,
        **kwargs,
    ):
        """
        Визуализирует Scan выбранным плоттером.
        """
        plotter_instance = plotter(*args, **kwargs)
        return plotter_instance.plot(scan=self)
