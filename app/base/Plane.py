from __future__ import annotations

import numpy as np


class Plane:
    """
    Плоскость вида:

    Ax + By + Cz + D = 0.

    Нормаль хранится в нормированном виде. Поэтому модуль невязки точки
    является её ортогональным расстоянием до плоскости.
    """

    def __init__(
        self,
        normal: np.ndarray,
        point_on_plane: np.ndarray,
        d: float,
    ) -> None:
        normal_array = np.asarray(normal, dtype=np.float64)

        if normal_array.shape != (3,):
            raise ValueError(
                "normal должен быть массивом формы (3,)."
            )

        normal_norm = float(np.linalg.norm(normal_array))

        if normal_norm <= 1e-15:
            raise ValueError("Нормаль плоскости не может быть нулевой.")

        self.normal = normal_array / normal_norm
        self.point = np.asarray(
            point_on_plane,
            dtype=np.float64,
        )

        if self.point.shape != (3,):
            raise ValueError(
                "point_on_plane должен быть массивом формы (3,)."
            )

        self.d = float(d) / normal_norm

    @property
    def A(self) -> float:
        return float(self.normal[0])

    @property
    def B(self) -> float:
        return float(self.normal[1])

    @property
    def C(self) -> float:
        return float(self.normal[2])

    @property
    def D(self) -> float:
        return self.d

    @property
    def equation(self) -> tuple[float, float, float, float]:
        """
        Коэффициенты уравнения плоскости: ``(A, B, C, D)``.
        """
        return self.A, self.B, self.C, self.D

    def distance_to_point(
        self,
        xyz: np.ndarray,
    ) -> np.ndarray:
        """
        Возвращает ортогональное расстояние от одной точки или массива точек.

        Parameters
        ----------
        xyz:
            Массив формы ``(3,)`` или ``(N, 3)``.
        """
        coordinates = np.asarray(xyz, dtype=np.float64)

        if coordinates.shape[-1] != 3:
            raise ValueError(
                "Координаты должны иметь последнюю размерность 3."
            )

        residuals = np.dot(coordinates, self.normal) + self.d
        return np.abs(residuals)

    def project_point(
        self,
        xyz: np.ndarray,
    ) -> np.ndarray:
        """
        Проецирует точку или массив точек на плоскость.
        """
        coordinates = np.asarray(xyz, dtype=np.float64)

        if coordinates.shape[-1] != 3:
            raise ValueError(
                "Координаты должны иметь последнюю размерность 3."
            )

        signed_distance = np.dot(
            coordinates - self.point,
            self.normal,
        )

        return coordinates - np.expand_dims(
            signed_distance,
            axis=-1,
        ) * self.normal

    def __repr__(self) -> str:
        a, b, c, d = self.equation

        return (
            f"{self.__class__.__name__} "
            f"(A={a:.6f}, B={b:.6f}, C={c:.6f}, D={d:.6f})"
        )
