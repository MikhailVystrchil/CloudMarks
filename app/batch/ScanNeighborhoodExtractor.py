from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from app.batch.ReferencePoint import ReferencePoint
from app.scan.Scan import Scan


class ScanNeighborhoodExtractor:
    """
    Быстро извлекает локальные сферические окрестности из большого Scan.

    Индекс ``cKDTree`` создаётся один раз. Исходный Scan предполагается
    неизменяемым после построения индекса; при изменении состава/координат
    необходимо вызвать ``rebuild_index()``.
    """

    def __init__(
        self,
        scan: Scan,
        *,
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
                "leafsize должен быть положительным."
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
        return int(self.xyz.shape[0])

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            raise RuntimeError(
                "KD-дерево не построено. Вызовите rebuild_index()."
            )

        return self._tree

    @property
    def xyz(self) -> np.ndarray:
        if self._xyz is None:
            raise RuntimeError(
                "Координаты не подготовлены. "
                "Вызовите rebuild_index()."
            )

        return self._xyz

    def rebuild_index(self) -> None:
        """
        Перестраивает массив координат и пространственный индекс.
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

        coordinates = self.scan.to_numpy()

        if coordinates.shape != (len(self.scan), 3):
            raise ValueError(
                "Координаты Scan должны иметь форму (N, 3)."
            )

        if not np.all(np.isfinite(coordinates)):
            invalid_count = int(
                np.count_nonzero(
                    ~np.isfinite(coordinates).all(axis=1)
                )
            )

            raise ValueError(
                "Невозможно построить KD-дерево: "
                f"обнаружено точек с NaN/Inf: {invalid_count}."
            )

        self._xyz = coordinates

        self._tree = cKDTree(
            data=coordinates,
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
    ) -> np.ndarray:
        """
        Возвращает отсортированные индексы точек внутри сферы.
        """
        center_array = self._validate_center(center)
        radius_value = self._validate_radius(radius)

        indices = self.tree.query_ball_point(
            x=center_array,
            r=radius_value,
            p=2.0,
            eps=0.0,
            return_sorted=True,
        )

        return np.asarray(
            indices,
            dtype=np.intp,
        )

    def query_indices_many(
        self,
        centers: Sequence[Sequence[float]] | np.ndarray,
        radius: float | Sequence[float] | np.ndarray,
        *,
        workers: int = -1,
    ) -> list[np.ndarray]:
        """
        Выполняет пакетный поиск сферических окрестностей.
        """
        centers_array = np.asarray(
            centers,
            dtype=np.float64,
        )

        if (
            centers_array.ndim != 2
            or centers_array.shape[1] != 3
        ):
            raise ValueError(
                "centers должен иметь форму (M, 3)."
            )

        if len(centers_array) == 0:
            return []

        if not np.all(np.isfinite(centers_array)):
            raise ValueError(
                "Координаты центров не должны содержать NaN или Inf."
            )

        radii = np.asarray(
            radius,
            dtype=np.float64,
        )

        if radii.ndim == 0:
            radii = np.full(
                len(centers_array),
                fill_value=float(radii),
                dtype=np.float64,
            )
        elif (
            radii.ndim != 1
            or len(radii) != len(centers_array)
        ):
            raise ValueError(
                "radius должен быть скаляром или массивом длины M."
            )

        if (
            not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "Все радиусы должны быть конечными "
                "положительными числами."
            )

        if workers == 0 or workers < -1:
            raise ValueError(
                "workers должен быть -1 или положительным целым числом."
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
        Извлекает одну сферическую окрестность.
        """
        point_indices = self.query_indices(
            center=reference_point.as_array(),
            radius=radius,
        )

        return self.extract_by_indices(
            point_indices=point_indices,
            reference_name=reference_point.name,
            radius=radius,
        )

    def extract_spheres(
        self,
        reference_points: Sequence[ReferencePoint],
        default_radius: float,
        *,
        workers: int = -1,
    ) -> dict[str, Scan]:
        """
        Массово извлекает окрестности для набора ReferencePoint.
        """
        default_radius = self._validate_radius(default_radius)

        points = list(reference_points)

        if not points:
            return {}

        names = [point.name for point in points]

        if len(names) != len(set(names)):
            raise ValueError(
                "Имена опорных точек должны быть уникальными."
            )

        centers = np.asarray(
            [point.as_array() for point in points],
            dtype=np.float64,
        )

        radii = np.asarray(
            [
                (
                    point.radius
                    if point.radius is not None
                    else default_radius
                )
                for point in points
            ],
            dtype=np.float64,
        )

        index_sets = self.query_indices_many(
            centers=centers,
            radius=radii,
            workers=workers,
        )

        neighborhoods = {
            point.name: self.extract_by_indices(
                point_indices=indices,
                reference_name=point.name,
                radius=float(radius),
            )
            for point, radius, indices in zip(
                points,
                radii,
                index_sets,
            )
        }

        point_counts = np.asarray(
            [
                len(neighborhood)
                for neighborhood in neighborhoods.values()
            ],
            dtype=np.intp,
        )

        logger.success(
            "Окрестности извлечены: всего={}, min={}, median={}, max={}",
            len(neighborhoods),
            int(point_counts.min()),
            float(np.median(point_counts)),
            int(point_counts.max()),
        )

        return neighborhoods

    def extract_by_indices(
        self,
        point_indices: Sequence[int] | np.ndarray,
        reference_name: str,
        radius: float,
    ) -> Scan:
        """
        Формирует независимый локальный Scan по ранее найденным индексам.
        """
        index_array = np.asarray(
            point_indices,
            dtype=np.intp,
        )

        if index_array.ndim != 1:
            raise ValueError(
                "point_indices должен быть одномерным массивом."
            )

        if (
            np.any(index_array < 0)
            or np.any(index_array >= self.point_count)
        ):
            raise IndexError(
                "point_indices содержит индекс вне диапазона Scan."
            )

        radius_value = self._validate_radius(radius)

        return self.scan.subset(
            indices=index_array,
            scan_name=(
                f"{self.scan.name}"
                f"__{reference_name}"
                f"__r_{radius_value:.4f}"
            ),
            copy_points=True,
            include_normals=False,
            include_labels=False,
        )

    @staticmethod
    def _validate_center(
        center: Sequence[float],
    ) -> np.ndarray:
        center_array = np.asarray(
            center,
            dtype=np.float64,
        )

        if center_array.shape != (3,):
            raise ValueError(
                "center должен содержать три координаты: (X, Y, Z)."
            )

        if not np.all(np.isfinite(center_array)):
            raise ValueError(
                "Координаты центра не должны содержать NaN или Inf."
            )

        return center_array

    @staticmethod
    def _validate_radius(
        radius: float,
    ) -> float:
        radius_value = float(radius)

        if (
            not np.isfinite(radius_value)
            or radius_value <= 0.0
        ):
            raise ValueError(
                "Радиус должен быть конечным положительным числом, "
                f"получено {radius_value}."
            )

        return radius_value
