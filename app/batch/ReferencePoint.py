from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.base.Point import NamedPoint


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """
    Опорная точка — центр локальной сферической окрестности.

    Координаты должны быть заданы в общей системе координат обеих эпох
    после их взаимной регистрации.
    """

    name: str
    x: float
    y: float
    z: float
    radius: float | None = None

    def as_array(self) -> np.ndarray:
        """
        Координаты центра окрестности как массив формы ``(3,)``.
        """
        return np.asarray(
            [self.x, self.y, self.z],
            dtype=np.float64,
        )
