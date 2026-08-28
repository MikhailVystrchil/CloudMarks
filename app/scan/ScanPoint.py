from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from CONFIG import DEFAULT_POINTS_COLOR
from app.base.Point import Point


class ScanPoint(Point):
    """
    Точка облака лазерного сканирования.

    Помимо координат содержит исходный цвет, нормаль и, при необходимости,
    назначенную метку сегментации.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        color: Sequence[int] = DEFAULT_POINTS_COLOR,
        normals: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__(x=x, y=y, z=z)

        if len(color) != 3:
            raise ValueError(
                "color должен содержать три компоненты RGB."
            )

        self.color = tuple(int(component) for component in color)
        self.normals = (
            np.asarray(normals, dtype=np.float64)
            if normals is not None
            else None
        )

    def copy(
        self,
        *,
        include_normals: bool = True,
        include_labels: bool = True,
    ) -> "ScanPoint":
        """
        Создаёт независимую копию точки.

        Метод используется при формировании локальных подсканов, чтобы
        назначение нормалей и меток не изменяло исходное большое облако.
        """
        normals = None

        if include_normals and self.normals is not None:
            normals = self.normals.copy()

        copied_point = ScanPoint(
            x=self.x,
            y=self.y,
            z=self.z,
            color=self.color,
            normals=normals,
        )

        if include_labels and hasattr(self, "labels"):
            copied_point.labels = self.labels

        return copied_point

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
            f"normals={self.normals}, color={self.color})"
        )

    def __repr__(self) -> str:
        return (
            f"ScanPoint("
            f"x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}"
            f")"
        )
