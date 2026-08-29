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
    Результат статистического анализа смещения одной виртуальной точки.

    delta:
        Вектор смещения:
            delta = point_epoch2 - point_epoch1.

    displacement:
        Модуль пространственного смещения в метрах.

    sigma_displacement:
        СКП модуля смещения в метрах, вычисленная через перенос
        ковариации в направлении вектора delta.
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
        """
        Модуль смещения в миллиметрах.
        """
        return self.displacement * 1000.0

    @property
    def sigma_displacement_mm(self) -> float:
        """
        СКП модуля смещения в миллиметрах.
        """
        return self.sigma_displacement * 1000.0

    def as_dict(self) -> dict[str, object]:
        """
        Преобразует результат в словарь для DataFrame и CSV.
        """
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
            "sigma_displacement_mm": (
                self.sigma_displacement_mm
            ),
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
        """
        Возвращает компактное текстовое представление результата.
        """
        return (
            f"DeformationResult("
            f"name={self.name!r}, "
            f"dx={self.delta[0]:.6f}, "
            f"dy={self.delta[1]:.6f}, "
            f"dz={self.delta[2]:.6f}, "
            f"displacement={self.displacement:.6f} m, "
            f"displacement_mm={self.displacement_mm:.3f} mm, "
            f"sigma_displacement={self.sigma_displacement:.6f} m, "
            f"t={self.t_value:.3f}, "
            f"p_t={self.p_value_t:.6f}, "
            f"significant_t={self.significant_t}, "
            f"chi2={self.chi2_value}, "
            f"p_chi2={self.p_value_chi2}, "
            f"significant_chi2={self.significant_chi2}, "
            f"reliable={self.reliable}"
            f")"
        )


class DeformationAnalyzer:
    """
    Анализирует смещения одноимённых виртуальных точек двух эпох.

    Для каждой точки вычисляются:

    1. Вектор смещения:

       delta = X_epoch2 - X_epoch1

    2. Модуль пространственного смещения:

       S = ||delta||

    3. Ковариация смещения:

       C_delta = C_epoch1 + C_epoch2

    Предполагается, что оценки координат двух эпох независимы.

    4. СКП компонент вектора смещения:

       sigma_dx = sqrt(C_delta[0, 0])
       sigma_dy = sqrt(C_delta[1, 1])
       sigma_dz = sqrt(C_delta[2, 2])

    5. СКП модуля смещения:

       sigma_S^2 = u^T C_delta u

       где:

       u = delta / ||delta||

    6. t-проверка по модулю:

       t = S / sigma_S

    7. chi-square-проверка полного трёхмерного вектора:

       chi2 = delta^T C_delta^(-1) delta
    """

    def __init__(
        self,
        alpha: float = 0.05,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha должен принадлежать интервалу (0, 1)."
            )

        self.alpha = float(alpha)
        self._results: list[DeformationResult] = []

    def analyze_point_sets(
        self,
        points_epoch1: list[NamedPoint],
        points_epoch2: list[NamedPoint],
    ) -> list[DeformationResult]:
        """
        Анализирует набор одноимённых точек двух эпох.

        Точки сопоставляются по полю ``name``. Множества имён должны
        совпадать, а имена в пределах каждой эпохи быть уникальными.
        """
        epoch1_by_name = self._index_points(
            points=points_epoch1,
            epoch_name="epoch1",
        )

        epoch2_by_name = self._index_points(
            points=points_epoch2,
            epoch_name="epoch2",
        )

        names_epoch1 = set(epoch1_by_name)
        names_epoch2 = set(epoch2_by_name)

        if names_epoch1 != names_epoch2:
            missing_in_epoch2 = sorted(
                names_epoch1 - names_epoch2
            )
            missing_in_epoch1 = sorted(
                names_epoch2 - names_epoch1
            )

            raise ValueError(
                "Наборы имён точек двух эпох не совпадают. "
                f"Нет в epoch2: {missing_in_epoch2}; "
                f"нет в epoch1: {missing_in_epoch1}."
            )

        self._results = []

        for name in sorted(names_epoch1):
            result = self._analyze_single_point(
                point_epoch1=epoch1_by_name[name],
                point_epoch2=epoch2_by_name[name],
            )

            self._results.append(result)

        logger.success(
            "Статистический анализ завершён: точек={}, "
            "значимых t={}, alpha={}",
            len(self._results),
            self.n_significant,
            self.alpha,
        )

        return self.results

    @staticmethod
    def _index_points(
        points: list[NamedPoint],
        epoch_name: str,
    ) -> dict[str, NamedPoint]:
        """
        Формирует индекс точек по имени и контролирует дубликаты.
        """
        indexed: dict[str, NamedPoint] = {}

        for point in points:
            name = str(point.name)

            if not name:
                raise ValueError(
                    f"В {epoch_name} найдена точка с пустым именем."
                )

            if name in indexed:
                raise ValueError(
                    f"В {epoch_name} имя точки '{name}' повторяется."
                )

            indexed[name] = point

        return indexed

    def _analyze_single_point(
        self,
        point_epoch1: NamedPoint,
        point_epoch2: NamedPoint,
    ) -> DeformationResult:
        """
        Рассчитывает деформацию одной одноимённой пары точек.
        """
        if point_epoch1.name != point_epoch2.name:
            raise ValueError(
                "Для анализа должны передаваться точки "
                "с одинаковыми именами."
            )

        coordinate_epoch1 = np.asarray(
            [
                point_epoch1.x,
                point_epoch1.y,
                point_epoch1.z,
            ],
            dtype=np.float64,
        )

        coordinate_epoch2 = np.asarray(
            [
                point_epoch2.x,
                point_epoch2.y,
                point_epoch2.z,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(coordinate_epoch1)):
            raise ValueError(
                f"Координаты точки '{point_epoch1.name}' "
                "в epoch1 содержат NaN или Inf."
            )

        if not np.all(np.isfinite(coordinate_epoch2)):
            raise ValueError(
                f"Координаты точки '{point_epoch2.name}' "
                "в epoch2 содержат NaN или Inf."
            )

        covariance_epoch1 = self._validate_covariance(
            point=point_epoch1,
            epoch_name="epoch1",
        )

        covariance_epoch2 = self._validate_covariance(
            point=point_epoch2,
            epoch_name="epoch2",
        )

        delta = coordinate_epoch2 - coordinate_epoch1

        covariance_delta = (
            covariance_epoch1
            + covariance_epoch2
        )

        covariance_delta = 0.5 * (
            covariance_delta
            + covariance_delta.T
        )

        displacement = float(
            np.linalg.norm(delta)
        )

        return self._compute_tests(
            name=point_epoch1.name,
            delta=delta,
            displacement=displacement,
            cov_delta=covariance_delta,
        )

    @staticmethod
    def _validate_covariance(
        point: NamedPoint,
        epoch_name: str,
    ) -> np.ndarray:
        """
        Проверяет и возвращает ковариационную матрицу точки формы (3, 3).
        """
        reliable_accuracy = getattr(
            point,
            "reliable_accuracy",
            True,
        )

        if not reliable_accuracy:
            raise ValueError(
                f"Точка '{point.name}' в {epoch_name} "
                "имеет ненадёжную оценку точности."
            )

        covariance = getattr(
            point,
            "cov_xyz",
            None,
        )

        if covariance is None:
            raise ValueError(
                f"У точки '{point.name}' в {epoch_name} "
                "отсутствует ковариационная матрица cov_xyz."
            )

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        if covariance.shape != (3, 3):
            raise ValueError(
                f"Ковариация точки '{point.name}' в {epoch_name} "
                f"имеет форму {covariance.shape}; ожидается (3, 3)."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                f"Ковариация точки '{point.name}' в {epoch_name} "
                "содержит NaN или Inf."
            )

        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                f"Ковариация точки '{point.name}' в {epoch_name} "
                "не является положительно полуопределённой."
            )

        return covariance

    def _compute_tests(
        self,
        name: str,
        delta: np.ndarray,
        displacement: float,
        cov_delta: np.ndarray,
    ) -> DeformationResult:
        """
        Вычисляет СКП модуля смещения, t- и chi-square-критерии.

        Важная деталь реализации:
        выражение для дисперсии модуля смещения вычисляется через
        np.einsum("i,ij,j->", ...), чтобы гарантированно получить
        скаляр независимо от формы NumPy-массивов.
        """
        delta = np.asarray(
            delta,
            dtype=np.float64,
        ).reshape(3)

        cov_delta = np.asarray(
            cov_delta,
            dtype=np.float64,
        )

        if cov_delta.shape != (3, 3):
            raise ValueError(
                "cov_delta должна иметь форму (3, 3)."
            )

        if not np.all(np.isfinite(delta)):
            raise ValueError(
                f"Вектор смещения точки '{name}' "
                "содержит NaN или Inf."
            )

        if not np.all(np.isfinite(cov_delta)):
            raise ValueError(
                f"Ковариация смещения точки '{name}' "
                "содержит NaN или Inf."
            )

        cov_delta = 0.5 * (
            cov_delta
            + cov_delta.T
        )

        diagonal = np.diag(cov_delta)

        if np.any(diagonal < -1e-12):
            raise ValueError(
                f"Ковариация смещения точки '{name}' "
                "содержит отрицательные дисперсии."
            )

        sigma_dx = float(
            np.sqrt(max(float(diagonal[0]), 0.0))
        )
        sigma_dy = float(
            np.sqrt(max(float(diagonal[1]), 0.0))
        )
        sigma_dz = float(
            np.sqrt(max(float(diagonal[2]), 0.0))
        )

        if displacement <= 1e-15:
            displacement_variance = 0.0
            sigma_displacement = 0.0
            t_value = 0.0
            p_value_t = 1.0
            significant_t = False

        else:
            unit_vector = delta / displacement

            displacement_variance = float(
                np.einsum(
                    "i,ij,j->",
                    unit_vector,
                    cov_delta,
                    unit_vector,
                )
            )

            displacement_variance = max(
                displacement_variance,
                0.0,
            )

            sigma_displacement = float(
                np.sqrt(displacement_variance)
            )

            if sigma_displacement <= 1e-15:
                t_value = float("inf")
                p_value_t = 0.0
                significant_t = True

            else:
                t_value = float(
                    displacement / sigma_displacement
                )

                p_value_t = float(
                    2.0
                    * stats.norm.sf(
                        abs(t_value)
                    )
                )

                significant_t = bool(
                    p_value_t < self.alpha
                )

        chi2_value: float | None = None
        p_value_chi2: float | None = None
        significant_chi2: bool | None = None
        reliable = True

        try:
            chi2_value = float(
                delta
                @ np.linalg.solve(
                    cov_delta,
                    delta,
                )
            )

            if chi2_value < 0.0:
                chi2_value = max(
                    chi2_value,
                    0.0,
                )

            p_value_chi2 = float(
                stats.chi2.sf(
                    chi2_value,
                    df=3,
                )
            )

            significant_chi2 = bool(
                p_value_chi2 < self.alpha
            )

        except np.linalg.LinAlgError:
            reliable = False

            logger.warning(
                "Точка '{}': ковариация смещения вырождена; "
                "chi-square-проверка не выполнена.",
                name,
            )

        return DeformationResult(
            name=name,
            delta=delta,
            displacement=displacement,
            sigma_displacement=sigma_displacement,
            cov_delta=cov_delta,
            sigma_dx=sigma_dx,
            sigma_dy=sigma_dy,
            sigma_dz=sigma_dz,
            t_value=t_value,
            p_value_t=p_value_t,
            significant_t=significant_t,
            chi2_value=chi2_value,
            p_value_chi2=p_value_chi2,
            significant_chi2=significant_chi2,
            alpha=self.alpha,
            reliable=reliable,
        )

    @property
    def results(self) -> list[DeformationResult]:
        """
        Возвращает копию списка результатов.
        """
        return list(self._results)

    @property
    def significant_results(self) -> list[DeformationResult]:
        """
        Возвращает результаты со значимым смещением по t-критерию.
        """
        return [
            result
            for result in self._results
            if result.significant_t
        ]

    @property
    def n_significant(self) -> int:
        """
        Количество точек со значимым смещением по t-критерию.
        """
        return len(
            self.significant_results
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Формирует таблицу результатов анализа деформаций.
        """
        dataframe = pd.DataFrame(
            [
                result.as_dict()
                for result in self._results
            ]
        )

        if dataframe.empty:
            return dataframe

        return dataframe.sort_values(
            by="name",
            kind="stable",
        ).reset_index(drop=True)

    def to_csv(
        self,
        file_path: str,
        *,
        index: bool = False,
    ) -> None:
        """
        Экспортирует результаты анализа в CSV.
        """
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
