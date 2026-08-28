from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from app.base.Point import NamedPoint


@dataclass(frozen=True, slots=True)
class DeformationResult:
    """
    Результат анализа смещения одной виртуальной точки между эпохами.
    """

    name: str
    delta: np.ndarray
    displacement: float
    sigma_displacement: float
    cov_delta: np.ndarray

    sigma_dx: float
    sigma_dy: float
    sigma_dz: float

    t_value: float
    p_value_t: float
    significant_t: bool

    chi2_value: float | None
    p_value_chi2: float | None
    significant_chi2: bool | None

    alpha: float
    reliable: bool

    @property
    def displacement_mm(self) -> float:
        return self.displacement * 1000.0

    @property
    def sigma_displacement_mm(self) -> float:
        return self.sigma_displacement * 1000.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dx": float(self.delta[0]),
            "dy": float(self.delta[1]),
            "dz": float(self.delta[2]),
            "displacement": self.displacement,
            "displacement_mm": self.displacement_mm,
            "sigma_dx": self.sigma_dx,
            "sigma_dy": self.sigma_dy,
            "sigma_dz": self.sigma_dz,
            "sigma_displacement": self.sigma_displacement,
            "sigma_displacement_mm": self.sigma_displacement_mm,
            "t_value": self.t_value,
            "p_value_t": self.p_value_t,
            "significant_t": self.significant_t,
            "chi2_value": self.chi2_value,
            "p_value_chi2": self.p_value_chi2,
            "significant_chi2": self.significant_chi2,
            "alpha": self.alpha,
            "reliable": self.reliable,
        }

    def __str__(self) -> str:
        t_status = (
            "SIGNIFICANT"
            if self.significant_t
            else "not significant"
        )

        chi2_status = (
            "n/a"
            if self.significant_chi2 is None
            else (
                "SIGNIFICANT"
                if self.significant_chi2
                else "not significant"
            )
        )

        result = (
            f"[{self.name}] "
            f"d={self.displacement_mm:.2f} mm "
            f"± {self.sigma_displacement_mm:.2f} mm | "
            f"T={self.t_value:.3f}, "
            f"p={self.p_value_t:.4f} "
            f"({t_status})"
        )

        if self.chi2_value is None:
            return result + " | chi2=n/a"

        return (
            f"{result} | "
            f"chi2={self.chi2_value:.3f}, "
            f"p={self.p_value_chi2:.4f} "
            f"({chi2_status})"
        )


class DeformationAnalyzer:
    """
    Анализирует пространственные смещения точек между двумя эпохами.

    Для точки с координатами ``X1`` и ``X2`` вычисляется:

    ``d = X2 - X1``

    При наличии ковариаций:

    ``Sigma_d = Sigma_1 + Sigma_2 - C_12 - C_12^T``

    Выполняются:
    - одномерный нормальный тест для длины смещения;
    - многомерный критерий ``chi²`` для вектора смещения.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        cross_cov_map: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha должен принадлежать интервалу (0, 1)."
            )

        self.alpha = float(alpha)
        self.cross_cov_map = cross_cov_map or {}
        self._results: list[DeformationResult] = []

    def analyze_point_sets(
        self,
        points_epoch1: list[NamedPoint],
        points_epoch2: list[NamedPoint],
    ) -> "DeformationAnalyzer":
        """
        Анализирует точки с совпадающими именами.
        """
        points_map_epoch1 = {
            point.name: point
            for point in points_epoch1
        }
        points_map_epoch2 = {
            point.name: point
            for point in points_epoch2
        }

        common_names = sorted(
            set(points_map_epoch1)
            & set(points_map_epoch2)
        )

        self._results = [
            self._analyze_single_point(
                point_epoch1=points_map_epoch1[name],
                point_epoch2=points_map_epoch2[name],
            )
            for name in common_names
        ]

        return self

    def _analyze_single_point(
        self,
        point_epoch1: NamedPoint,
        point_epoch2: NamedPoint,
    ) -> DeformationResult:
        name = point_epoch1.name

        xyz_epoch1 = point_epoch1.as_array()
        xyz_epoch2 = point_epoch2.as_array()

        delta = xyz_epoch2 - xyz_epoch1
        displacement = float(
            np.linalg.norm(delta)
        )

        covariance_epoch1 = getattr(
            point_epoch1,
            "cov_xyz",
            None,
        )
        covariance_epoch2 = getattr(
            point_epoch2,
            "cov_xyz",
            None,
        )

        if (
            covariance_epoch1 is None
            or covariance_epoch2 is None
        ):
            return self._make_unreliable_result(
                name=name,
                delta=delta,
                displacement=displacement,
            )

        covariance_epoch1 = self._validate_covariance(
            covariance=covariance_epoch1,
            covariance_name=f"cov_xyz эпохи 1 для {name}",
        )
        covariance_epoch2 = self._validate_covariance(
            covariance=covariance_epoch2,
            covariance_name=f"cov_xyz эпохи 2 для {name}",
        )

        cross_covariance = self._validate_covariance(
            covariance=self.cross_cov_map.get(
                name,
                np.zeros((3, 3), dtype=np.float64),
            ),
            covariance_name=f"cross_covariance для {name}",
        )

        covariance_delta = (
            covariance_epoch1
            + covariance_epoch2
            - cross_covariance
            - cross_covariance.T
        )

        covariance_delta = 0.5 * (
            covariance_delta
            + covariance_delta.T
        )

        return self._compute_tests(
            name=name,
            delta=delta,
            displacement=displacement,
            cov_delta=covariance_delta,
            reliable=True,
        )

    @staticmethod
    def _validate_covariance(
        covariance: np.ndarray,
        covariance_name: str,
    ) -> np.ndarray:
        covariance_array = np.asarray(
            covariance,
            dtype=np.float64,
        )

        if covariance_array.shape != (3, 3):
            raise ValueError(
                f"{covariance_name} должна иметь форму (3, 3)."
            )

        if not np.all(np.isfinite(covariance_array)):
            raise ValueError(
                f"{covariance_name} содержит NaN или Inf."
            )

        covariance_array = 0.5 * (
            covariance_array
            + covariance_array.T
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance_array
        )

        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                f"{covariance_name} не является "
                "положительно полуопределённой."
            )

        return covariance_array

    def _compute_tests(
        self,
        *,
        name: str,
        delta: np.ndarray,
        displacement: float,
        cov_delta: np.ndarray,
        reliable: bool,
    ) -> DeformationResult:
        sigma_xyz = np.sqrt(
            np.maximum(
                np.diag(cov_delta),
                0.0,
            )
        )

        if displacement > 1e-16:
            direction = (
                delta / displacement
            ).reshape(1, 3)

            displacement_variance = float(
                direction
                @ cov_delta
                @ direction.T
            )

            sigma_displacement = float(
                np.sqrt(
                    max(
                        displacement_variance,
                        0.0,
                    )
                )
            )
        else:
            sigma_displacement = float(
                np.sqrt(
                    max(
                        np.trace(cov_delta) / 3.0,
                        0.0,
                    )
                )
            )

        t_value = (
            displacement / sigma_displacement
            if sigma_displacement > 1e-16
            else 0.0
        )

        p_value_t = float(
            2.0
            * (
                1.0
                - stats.norm.cdf(
                    abs(t_value)
                )
            )
        )

        significant_t = p_value_t < self.alpha

        chi2_value: float | None = None
        p_value_chi2: float | None = None
        significant_chi2: bool | None = None

        try:
            covariance_inverse = np.linalg.inv(
                cov_delta
            )

            chi2_value = float(
                delta
                @ covariance_inverse
                @ delta
            )

            p_value_chi2 = float(
                1.0
                - stats.chi2.cdf(
                    chi2_value,
                    df=3,
                )
            )

            significant_chi2 = (
                p_value_chi2 < self.alpha
            )

        except np.linalg.LinAlgError:
            logger.warning(
                "Не удалось инвертировать ковариацию смещения "
                "для точки '{}'; chi²-тест пропущен.",
                name,
            )

        return DeformationResult(
            name=name,
            delta=delta,
            displacement=displacement,
            sigma_displacement=sigma_displacement,
            cov_delta=cov_delta,
            sigma_dx=float(sigma_xyz[0]),
            sigma_dy=float(sigma_xyz[1]),
            sigma_dz=float(sigma_xyz[2]),
            t_value=float(t_value),
            p_value_t=p_value_t,
            significant_t=significant_t,
            chi2_value=chi2_value,
            p_value_chi2=p_value_chi2,
            significant_chi2=significant_chi2,
            alpha=self.alpha,
            reliable=reliable,
        )

    def _make_unreliable_result(
        self,
        *,
        name: str,
        delta: np.ndarray,
        displacement: float,
    ) -> DeformationResult:
        nan = float("nan")

        return DeformationResult(
            name=name,
            delta=delta,
            displacement=float(displacement),
            sigma_displacement=nan,
            cov_delta=np.full(
                (3, 3),
                nan,
                dtype=np.float64,
            ),
            sigma_dx=nan,
            sigma_dy=nan,
            sigma_dz=nan,
            t_value=nan,
            p_value_t=nan,
            significant_t=False,
            chi2_value=None,
            p_value_chi2=None,
            significant_chi2=None,
            alpha=self.alpha,
            reliable=False,
        )

    @property
    def results(self) -> list[DeformationResult]:
        return list(self._results)

    @property
    def significant_results(
        self,
    ) -> list[DeformationResult]:
        return [
            result
            for result in self._results
            if result.significant_t
        ]

    @property
    def n_significant(self) -> int:
        return sum(
            result.significant_t
            for result in self._results
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                result.as_dict()
                for result in self._results
            ]
        )

    def to_csv(
        self,
        file_path: str,
        *,
        index: bool = False,
    ) -> None:
        self.to_dataframe().to_csv(
            file_path,
            index=index,
            encoding="utf-8",
        )

    def log_summary(self) -> None:
        """
        Выводит итоговый протокол через централизованный logger.
        """
        point_count = len(self._results)
        significant_count = self.n_significant

        logger.info(
            "DeformationAnalyzer: точек={}, "
            "значимых смещений={}/{}, alpha={}",
            point_count,
            significant_count,
            point_count,
            self.alpha,
        )

        for result in sorted(
            self._results,
            key=lambda item: item.displacement,
            reverse=True,
        ):
            logger.info("{}", result)
