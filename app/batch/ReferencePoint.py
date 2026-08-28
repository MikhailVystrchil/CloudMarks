from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """
    Опорная точка, определяющая центр локальной окрестности большого скана.

    Координаты должны быть заданы в общей системе координат двух
    разновременных облаков точек после их взаимного ориентирования.
    """

    name: str
    x: float
    y: float
    z: float
    radius: float | None = None

    def as_array(self) -> tuple[float, float, float]:
        """Координаты центра окрестности в формате, совместимом с cKDTree."""
        return self.x, self.y, self.z
