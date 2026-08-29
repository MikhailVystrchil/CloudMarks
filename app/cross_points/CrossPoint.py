from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from app.base.Point import NamedPoint


class CrossPoint(NamedPoint):
    """
    Виртуальная точка — пересечение трёх локальных плоскостей.

    Attributes
    ----------
    mse:
        Интегральная оценка точности.
    planes_mse:
        RMSE трёх аппроксимирующих плоскостей.
    sigma_xyz:
        СКП координат X, Y, Z.
    cov_xyz:
        Ковариационная матрица координат.
    ellipsoid:
        Параметры эллипсоида ошибок заданной доверительной вероятности.
    reliable_accuracy:
        Признак применимости ковариационной оценки.
    """

    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        super().__init__(
            name=name,
            x=x,
            y=y,
            z=z,
        )

        self.status: str | None = None
        self.mse: float | None = None
        self.planes_mse: list[float] | None = None

        self.sigma_xyz: np.ndarray | None = None
        self.cov_xyz: np.ndarray | None = None
        self.ellipsoid: dict[str, object] | None = None

        self.reliable_accuracy = True

    def load_mses(
        self,
        plane_mses: list[float],
    ) -> None:
        """
        Сохраняет RMSE плоскостей и их интегральную оценку.
        """
        self.planes_mse = [
            float(value)
            for value in plane_mses
        ]

        self.mse = float(
            np.sqrt(
                np.sum(
                    np.square(self.planes_mse)
                )
            )
        )

    def load_covariance(
        self,
        cov_xyz: np.ndarray,
        *,
        confidence: float = 0.95,
    ) -> None:
        """
        Сохраняет ковариацию и рассчитывает эллипсоид ошибок.
        """
        covariance = np.asarray(
            cov_xyz,
            dtype=np.float64,
        )

        if covariance.shape != (3, 3):
            raise ValueError(
                "cov_xyz должна иметь форму (3, 3)."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "cov_xyz содержит NaN или Inf."
            )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                "cov_xyz должна быть положительно полуопределённой."
            )

        eigenvalues = np.maximum(
            eigenvalues,
            0.0,
        )

        order = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        confidence_scale = chi2.ppf(
            confidence,
            df=3,
        )

        self.cov_xyz = covariance
        self.sigma_xyz = np.sqrt(
            np.maximum(
                np.diag(covariance),
                0.0,
            )
        )
        self.mse = float(
            np.sqrt(
                np.trace(covariance)
            )
        )
        self.ellipsoid = {
            "semi_axes": np.sqrt(
                eigenvalues * confidence_scale
            ),
            "directions": eigenvectors,
            "confidence": confidence,
        }
        self.reliable_accuracy = True

    def mark_unreliable_accuracy(self) -> None:
        """
        Помечает оценку точности как непригодную.
        """
        self.reliable_accuracy = False
        self.sigma_xyz = None
        self.cov_xyz = None
        self.ellipsoid = None

    def as_dict(self) -> dict[str, object]:
        """
        Возвращает параметры виртуальной точки в виде словаря.

        Значения sigma_* и параметры эллипсоида равны None, если
        ковариационная оценка признана ненадёжной.
        """
        result: dict[str, object] = {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "status": self.status,
            "reliable_accuracy": bool(self.reliable_accuracy),
            "mse": self.mse,
        }

        if self.planes_mse is not None:
            for index, plane_mse in enumerate(
                self.planes_mse,
                start=1,
            ):
                result[f"plane_{index}_mse"] = float(
                    plane_mse
                )

        if self.reliable_accuracy and self.sigma_xyz is not None:
            result.update(
                {
                    "sigma_x": float(self.sigma_xyz[0]),
                    "sigma_y": float(self.sigma_xyz[1]),
                    "sigma_z": float(self.sigma_xyz[2]),
                }
            )
        else:
            result.update(
                {
                    "sigma_x": None,
                    "sigma_y": None,
                    "sigma_z": None,
                }
            )

        if self.ellipsoid is not None:
            semi_axes = self.ellipsoid["semi_axes"]

            result.update(
                {
                    "ellipsoid_confidence": float(
                        self.ellipsoid["confidence"]
                    ),
                    "ellipsoid_a": float(semi_axes[0]),
                    "ellipsoid_b": float(semi_axes[1]),
                    "ellipsoid_c": float(semi_axes[2]),
                }
            )
        else:
            result.update(
                {
                    "ellipsoid_confidence": None,
                    "ellipsoid_a": None,
                    "ellipsoid_b": None,
                    "ellipsoid_c": None,
                }
            )

        return result

    def as_flat_fields(
            self,
            *,
            prefix: str = "",
            status_key: str = "status",
    ) -> dict[str, object]:
        """
        Возвращает набор полей виртуальной точки для плоской таблицы.

        Метод используется результатами одиночной и межэпоховой обработки,
        чтобы централизовать преобразование координат, статуса, признака
        надёжности и СКП в CSV/DataFrame-представление.

        Parameters
        ----------
        prefix:
            Необязательный префикс полей. Например, ``epoch1`` создаёт ключи
            ``epoch1_x``, ``epoch1_y`` и т. д.
        status_key:
            Имя поля статуса без префикса. Для одиночного извлечения удобно
            передавать ``geometry_status``.
        """
        normalized_prefix = (
            f"{prefix}_"
            if prefix
            else ""
        )

        sigma = (
            self.sigma_xyz
            if self.reliable_accuracy and self.sigma_xyz is not None
            else (np.nan, np.nan, np.nan)
        )

        return {
            f"{normalized_prefix}x": float(self.x),
            f"{normalized_prefix}y": float(self.y),
            f"{normalized_prefix}z": float(self.z),
            f"{normalized_prefix}{status_key}": self.status,
            f"{normalized_prefix}reliable_accuracy": bool(
                self.reliable_accuracy
            ),
            f"{normalized_prefix}sigma_x": float(sigma[0]),
            f"{normalized_prefix}sigma_y": float(sigma[1]),
            f"{normalized_prefix}sigma_z": float(sigma[2]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Возвращает виртуальную точку как DataFrame из одной строки.

        Пример:
            point.to_dataframe()
        """
        return pd.DataFrame([self.as_dict()])

    def __str__(self) -> str:
        parts = [
            (
                f"{self.__class__.__name__} "
                f"(name={self.name}, status={self.status}"
            ),
            (
                f"x={self.x:.6f}, "
                f"y={self.y:.6f}, "
                f"z={self.z:.6f}"
            ),
        ]

        if self.planes_mse is not None:
            parts.append(
                "plane_mses="
                f"{[round(value, 6) for value in self.planes_mse]}"
            )

        if self.reliable_accuracy:
            if self.mse is not None:
                parts.append(f"mse={self.mse:.6f}")

            if self.sigma_xyz is not None:
                sigma_x, sigma_y, sigma_z = self.sigma_xyz

                parts.append(
                    "sigma_xyz="
                    f"({sigma_x:.6f}, "
                    f"{sigma_y:.6f}, "
                    f"{sigma_z:.6f})"
                )
        else:
            parts.append("accuracy=UNRELIABLE")

        return ", ".join(parts) + ")"

    def __repr__(self) -> str:
        accuracy = (
            "ok"
            if self.reliable_accuracy
            else "unreliable"
        )

        mse_text = (
            f"{self.mse:.5f}"
            if self.mse is not None
            else "None"
        )

        return (
            f"CrossPoint("
            f"name={self.name!r}, "
            f"status={self.status!r}, "
            f"x={self.x:.3f}, "
            f"y={self.y:.3f}, "
            f"z={self.z:.3f}, "
            f"mse={mse_text}, "
            f"accuracy={accuracy}"
            f")"
        )
