from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """
    Конфигурация извлечения виртуальных точек из облака точек.

    Экземпляр используется совместно SingleScanPointExtractor и
    PointPairComparisonRunner. Централизует валидацию параметров,
    влияющих на выделение окрестности, сегментацию нормалей,
    DBSCAN-кластеризацию и аппроксимацию плоскостей.
    """

    default_radius: float
    min_neighborhood_points: int = 60
    min_points_per_plane: int = 15
    max_reference_distance_factor: float = 1.25
    normal_k: int = 12
    cluster_eps: float = 0.08
    cluster_min_samples: int = 3

    def __post_init__(self) -> None:
        if self.default_radius <= 0.0:
            raise ValueError(
                "default_radius должен быть положительным."
            )

        if self.min_points_per_plane < 6:
            raise ValueError(
                "min_points_per_plane должен быть не менее 6."
            )

        if (
            self.min_neighborhood_points
            < 3 * self.min_points_per_plane
        ):
            raise ValueError(
                "min_neighborhood_points должен быть не меньше "
                "3 * min_points_per_plane."
            )

        if self.max_reference_distance_factor <= 0.0:
            raise ValueError(
                "max_reference_distance_factor должен быть положительным."
            )

        if self.normal_k < 3:
            raise ValueError(
                "normal_k должен быть не менее 3."
            )

        if self.cluster_eps <= 0.0:
            raise ValueError(
                "cluster_eps должен быть положительным."
            )

        if self.cluster_min_samples < 1:
            raise ValueError(
                "cluster_min_samples должен быть положительным."
            )
