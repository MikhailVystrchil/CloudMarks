from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from app.batch.ReferencePoint import ReferencePoint
from app.scan.Scan import Scan
from app.scan.ScanPoint import ScanPoint


class ScanNeighborhoodExtractor:
    """
    Быстро извлекает локальные сферические окрестности из большого Scan.

    Индекс cKDTree строится один раз при инициализации. Каждый последующий
    запрос окрестности выполняется через tree.query_ball_point(), поэтому
    не требует полного перебора всех точек исходного облака.

    Пример
    -------
    >>> extractor = ScanNeighborhoodExtractor(large_scan)
    >>> neighborhood = extractor.extract_sphere(reference_point, radius=0.25)

    Notes
    -----
    Экземпляр предполагает, что координаты исходного Scan после его создания
    не меняются. Если Scan был изменён, создайте новый extractor или вызовите
    rebuild_index().
    """

    def __init__(
        self,
        scan: Scan,
        leafsize: int = 64,
        compact_nodes: bool = True,
        balanced_tree: bool = True,
    ) -> None:
        if len(scan) == 0:
            raise ValueError(
                "Невозможно создать пространственный индекс: Scan пуст."
            )

        if leafsize < 1:
            raise ValueError(
                f"leafsize должен быть положительным, получено {leafsize}."
            )

        self.scan = scan
        self.leafsize = int(leafsize)
        self.compact_nodes = bool(compact_nodes)
        self.balanced_tree = bool(balanced_tree)

        self._xyz: np.ndarray | None = None
        self._tree: cKDTree | None = None

        self.rebuild_index()

    @property
    def point_count(self) -> int:
        """Количество точек, включённых в текущий пространственный индекс."""
        return int(self._xyz.shape[0])

    @property
    def tree(self) -> cKDTree:
        """Построенное KD-дерево исходного облака."""
        if self._tree is None:
            raise RuntimeError(
                "KD-дерево не построено. Вызовите rebuild_index()."
            )
        return self._tree

    @property
    def xyz(self) -> np.ndarray:
        """
        Массив координат формы (N, 3).

        Порядок строк строго соответствует порядку Scan._points; это нужно,
        чтобы индексы cKDTree можно было использовать для выбора ScanPoint.
        """
        if self._xyz is None:
            raise RuntimeError(
                "Массив координат не подготовлен. Вызовите rebuild_index()."
            )
        return self._xyz

    def rebuild_index(self) -> None:
        """
        Повторно строит индекс координат и cKDTree.

        Вызывайте этот метод только после изменения состава или координат
        точек self.scan. При неизменном Scan его повторный вызов не нужен.
        """
        if len(self.scan) == 0:
            raise ValueError(
                "Невозможно построить KD-дерево для пустого Scan."
            )

        logger.info(
            "Подготовка cKDTree для скана '{}': {} точек",
            self.scan.name,
            len(self.scan),
        )

        self._xyz = np.asarray(
            [
                [point.x, point.y, point.z]
                for point in self.scan
            ],
            dtype=np.float64,
        )

        if self._xyz.ndim != 2 or self._xyz.shape[1] != 3:
            raise ValueError(
                "Координаты Scan должны формировать массив формы (N, 3)."
            )

        if not np.all(np.isfinite(self._xyz)):
            invalid_count = int(
                np.count_nonzero(~np.isfinite(self._xyz).all(axis=1))
            )
            raise ValueError(
                "Невозможно построить KD-дерево: "
                f"обнаружено точек с NaN или Inf: {invalid_count}."
            )

        self._tree = cKDTree(
            self._xyz,
            leafsize=self.leafsize,
            compact_nodes=self.compact_nodes,
            balanced_tree=self.balanced_tree,
        )

        logger.success(
            "cKDTree для скана '{}' готово: {} точек, leafsize={}",
            self.scan.name,
            self.point_count,
            self.leafsize,
        )

    def query_indices(
        self,
        center: Sequence[float],
        radius: float,
        workers: int = -1,
    ) -> np.ndarray:
        """
        Возвращает отсортированные индексы точек, лежащих в сфере.

        Parameters
        ----------
        center:
            Центр сферы (X, Y, Z).
        radius:
            Радиус сферической окрестности в единицах координат Scan.
        workers:
            Число потоков SciPy. Значение -1 использует все доступные ядра
            для батчевых запросов; для одиночного query_ball_point параметр
            не используется, но оставлен для совместимого API.

        Returns
        -------
        np.ndarray
            Отсортированный массив индексов точек исходного Scan.

        Notes
        -----
        Для одного центра cKDTree.query_ball_point не принимает workers
        во всех поддерживаемых версиях SciPy, поэтому workers здесь не
        передаётся намеренно. Для десятков и тысяч центров используйте
        query_indices_many().
        """
        center_array = self._validate_center(center)
        radius = self._validate_radius(radius)

        indices = self.tree.query_ball_point(
            x=center_array,
            r=radius,
            p=2.0,
            eps=0.0,
            return_sorted=True,
        )

        return np.asarray(indices, dtype=np.intp)

    def query_indices_many(
        self,
        centers: Sequence[Sequence[float]] | np.ndarray,
        radius: float | Sequence[float] | np.ndarray,
        workers: int = -1,
    ) -> list[np.ndarray]:
        """
        Выполняет пакетный запрос сферических окрестностей.

        Для большого количества опорных точек этот метод предпочтительнее
        повторного вызова query_indices(), потому что SciPy может использовать
        несколько ядер через workers=-1.

        Parameters
        ----------
        centers:
            Массив центров формы (M, 3).
        radius:
            Один общий радиус или массив M индивидуальных радиусов.
        workers:
            -1 — все доступные процессорные ядра; 1 — один поток.

        Returns
        -------
        list[np.ndarray]
            Список из M массивов индексов исходного Scan.
        """
        centers_array = np.asarray(centers, dtype=np.float64)

        if centers_array.ndim != 2 or centers_array.shape[1] != 3:
            raise ValueError(
                "centers должен иметь форму (M, 3)."
            )

        if len(centers_array) == 0:
            return []

        if not np.all(np.isfinite(centers_array)):
            raise ValueError(
                "Координаты центров не должны содержать NaN или Inf."
            )

        radii = np.asarray(radius, dtype=np.float64)

        if radii.ndim == 0:
            radii = np.full(
                shape=len(centers_array),
                fill_value=float(radii),
                dtype=np.float64,
            )
        elif radii.ndim != 1 or len(radii) != len(centers_array):
            raise ValueError(
                "radius должен быть скаляром либо массивом длины M."
            )

        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError(
                "Все радиусы должны быть конечными положительными числами."
            )

        raw_indices = self.tree.query_ball_point(
            x=centers_array,
            r=radii,
            p=2.0,
            eps=0.0,
            workers=workers,
            return_sorted=True,
        )

        return [
            np.asarray(indices, dtype=np.intp)
            for indices in raw_indices
        ]

    def extract_sphere(
        self,
        reference_point: ReferencePoint,
        radius: float,
    ) -> Scan:
        """
        Создаёт новый Scan с точками в сфере заданного радиуса вокруг
        reference_point.

        Это удобный метод для обработки одной опорной точки. Для массового
        извлечения используйте extract_spheres(), чтобы выполнить поиск
        через один многопоточный запрос KD-дерева.
        """
        indices = self.query_indices(
            center=reference_point.as_array(),
            radius=radius,
        )

        neighborhood = self._make_neighborhood_scan(
            point_indices=indices,
            reference_name=reference_point.name,
            radius=radius,
        )

        logger.debug(
            "Окрестность '{}': скан='{}', radius={}, точек={}",
            reference_point.name,
            self.scan.name,
            radius,
            len(neighborhood),
        )

        return neighborhood

    def extract_spheres(
        self,
        reference_points: Sequence[ReferencePoint],
        default_radius: float,
        workers: int = -1,
    ) -> dict[str, Scan]:
        """
        Массово извлекает окрестности для набора опорных точек.

        Использует query_indices_many() и многопоточный cKDTree-запрос.
        Радиус конкретной ReferencePoint имеет приоритет над default_radius.

        Returns
        -------
        dict[str, Scan]
            Словарь вида {имя_опорной_точки: локальный_Scan}.

        Raises
        ------
        ValueError
            Если имена опорных точек повторяются или радиусы некорректны.
        """
        if default_radius <= 0:
            raise ValueError(
                f"default_radius должен быть положительным, получено {default_radius}."
            )

        points = list(reference_points)

        if not points:
            return {}

        point_names = [point.name for point in points]
        if len(point_names) != len(set(point_names)):
            raise ValueError(
                "Имена опорных точек должны быть уникальными."
            )

        centers = np.asarray(
            [point.as_array() for point in points],
            dtype=np.float64,
        )

        radii = np.asarray(
            [
                point.radius
                if point.radius is not None
                else default_radius
                for point in points
            ],
            dtype=np.float64,
        )

        logger.info(
            "Пакетное извлечение окрестностей: скан='{}', "
            "опорных точек={}, workers={}",
            self.scan.name,
            len(points),
            workers,
        )

        index_sets = self.query_indices_many(
            centers=centers,
            radius=radii,
            workers=workers,
        )

        neighborhoods: dict[str, Scan] = {}

        for point, point_radius, indices in zip(
            points,
            radii,
            index_sets,
        ):
            neighborhoods[point.name] = self._make_neighborhood_scan(
                point_indices=indices,
                reference_name=point.name,
                radius=float(point_radius),
            )

        point_counts = np.asarray(
            [len(neighborhood) for neighborhood in neighborhoods.values()],
            dtype=int,
        )

        logger.success(
            "Окрестности извлечены: всего={}, "
            "min={}, median={}, max={}",
            len(neighborhoods),
            int(point_counts.min()),
            float(np.median(point_counts)),
            int(point_counts.max()),
        )

        return neighborhoods

    def _make_neighborhood_scan(
        self,
        point_indices: np.ndarray,
        reference_name: str,
        radius: float,
    ) -> Scan:
        """
        Собирает локальное облако, не изменяя исходный большой Scan.

        Создаются новые ScanPoint, чтобы вычисление нормалей и назначение
        классов в локальном CrossPointExacter не добавляли атрибуты или
        изменения к точкам исходного облака.
        """
        neighborhood = Scan(
            scan_name=(
                f"{self.scan.name}"
                f"__{reference_name}"
                f"__r_{radius:.4f}"
            )
        )

        for point_index in point_indices:
            source_point = self.scan._points[int(point_index)]

            neighborhood.add_point(
                ScanPoint(
                    x=float(source_point.x),
                    y=float(source_point.y),
                    z=float(source_point.z),
                    color=getattr(source_point, "color", (0, 0, 0)),
                )
            )

        return neighborhood

    @staticmethod
    def _validate_center(
        center: Sequence[float],
    ) -> np.ndarray:
        center_array = np.asarray(center, dtype=np.float64)

        if center_array.shape != (3,):
            raise ValueError(
                "center должен содержать ровно три координаты: (X, Y, Z)."
            )

        if not np.all(np.isfinite(center_array)):
            raise ValueError(
                "Координаты центра не должны содержать NaN или Inf."
            )

        return center_array

    @staticmethod
    def _validate_radius(radius: float) -> float:
        radius = float(radius)

        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"Радиус должен быть конечным положительным числом, получено {radius}."
            )

        return radius

    def extract_by_indices(
        self,
        point_indices: np.ndarray,
        reference_name: str,
        radius: float,
    ) -> Scan:
        """
        Создаёт локальное облако Scan по индексам точек, предварительно
        найденным методом cKDTree.query_ball_point().

        Этот метод используется PointPairComparisonRunner в пакетном режиме:
        индексы всех сферических окрестностей находятся заранее, после чего
        локальные облака создаются последовательно — только для текущей
        опорной точки. Это не требует повторно искать точки в KD-дереве и
        не удерживает в памяти все локальные окрестности сразу.

        Parameters
        ----------
        point_indices:
            Одномерный массив индексов точек из исходного большого Scan.
        reference_name:
            Имя опорной точки; используется при формировании имени нового
            локального Scan.
        radius:
            Радиус исходной сферической окрестности.

        Returns
        -------
        Scan
            Новый независимый локальный Scan с точками данной окрестности.
        """
        indices = np.asarray(
            point_indices,
            dtype=np.intp,
        )

        if indices.ndim != 1:
            raise ValueError(
                "point_indices должен быть одномерным массивом индексов."
            )

        if np.any(indices < 0) or np.any(indices >= self.point_count):
            raise IndexError(
                "point_indices содержит индекс вне диапазона исходного Scan."
            )

        radius = self._validate_radius(radius)

        return self._make_neighborhood_scan(
            point_indices=indices,
            reference_name=reference_name,
            radius=radius,
        )
