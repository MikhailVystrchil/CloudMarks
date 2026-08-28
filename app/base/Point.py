from __future__ import annotations

from collections.abc import Sequence
from math import isclose

import numpy as np


class Point:
    """
    Базовая трёхмерная точка.

    Класс является изменяемым, поэтому намеренно не реализует ``__hash__``:
    изменяемые объекты не следует использовать как ключи словарей или элементы
    множеств. Сравнение выполняется с абсолютным допуском.
    """

    EQUALITY_TOLERANCE = 1e-5

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def as_array(self) -> np.ndarray:
        """
        Возвращает координаты точки в виде массива формы ``(3,)``.
        """
        return np.asarray(
            [self.x, self.y, self.z],
            dtype=np.float64,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented

        tolerance = self.EQUALITY_TOLERANCE

        return (
            isclose(self.x, other.x, abs_tol=tolerance, rel_tol=0.0)
            and isclose(self.y, other.y, abs_tol=tolerance, rel_tol=0.0)
            and isclose(self.z, other.z, abs_tol=tolerance, rel_tol=0.0)
        )

    __hash__ = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"
        )

    def __repr__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class NamedPoint(Point):
    """
    Точка с устойчивым текстовым идентификатором.
    """

    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        super().__init__(x=x, y=y, z=z)
        self.name = str(name)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(name={self.name}, x={self.x:.3f}, "
            f"y={self.y:.3f}, z={self.z:.3f})"
        )

    def __repr__(self) -> str:
        return (
            f"({self.name} {self.x:.3f}, "
            f"{self.y:.3f}, {self.z:.3f})"
        )
