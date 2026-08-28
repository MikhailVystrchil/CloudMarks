from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DeformationResult:
    """
    Результат анализа смещения одной виртуальной точки между двумя эпохами.
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
    chi2_value: Optional[float]
    p_value_chi2: Optional[float]
    significant_chi2: Optional[bool]
    alpha: float
    reliable: bool

    @property
    def displacement_mm(self):
        return self.displacement * 1000.0

    @property
    def sigma_displacement_mm(self):
        return self.sigma_displacement * 1000.0

    def as_dict(self):
        return {
            "name": self.name,
            "dx": float(self.delta[0]), "dy": float(self.delta[1]), "dz": float(self.delta[2]),
            "displacement": self.displacement, "displacement_mm": self.displacement_mm,
            "sigma_dx": self.sigma_dx, "sigma_dy": self.sigma_dy, "sigma_dz": self.sigma_dz,
            "sigma_displacement": self.sigma_displacement, "sigma_displacement_mm": self.sigma_displacement_mm,
            "t_value": self.t_value, "p_value_t": self.p_value_t, "significant_t": self.significant_t,
            "chi2_value": self.chi2_value, "p_value_chi2": self.p_value_chi2,
            "significant_chi2": self.significant_chi2, "alpha": self.alpha, "reliable": self.reliable,
        }

    def __str__(self):
        sig_t = "SIGNIFICANT" if self.significant_t else "not significant"
        sig_chi2 = ("n/a" if self.significant_chi2 is None
                    else ("SIGNIFICANT" if self.significant_chi2 else "not significant"))
        base = (f"[{self.name}] d={self.displacement_mm:.2f} mm ± {self.sigma_displacement_mm:.2f} mm | "
                f"T={self.t_value:.3f} p={self.p_value_t:.4f} ({sig_t}) | ")
        if self.chi2_value is not None:
            return base + f"chi2={self.chi2_value:.3f} p={self.p_value_chi2:.4f} ({sig_chi2})"
        return base + "chi2=n/a"


class DeformationAnalyzer:
    """
    Анализирует пространственные смещения виртуальных точек между двумя
    эпохами по методике, описанной в статье:

        1. d = X2 - X1
        2. Sigma_d = Sigma1 + Sigma2 - C12 - C12^T
        3. СКП компонент и длины смещения (первый порядок / Якоби)
        4. T^2 = d^T Sigma_d^{-1} d ~ chi2(3) — тест значимости смещения
        5. Дополнительно: 1D t-тест по длине смещения
    """

    def __init__(self, alpha=0.05, cross_cov_map=None):
        self.alpha = alpha
        self.cross_cov_map = cross_cov_map or {}
        self._results: list[DeformationResult] = []

    def analyze_point_sets(self, points_epoch1, points_epoch2):
        map1 = {p.name: p for p in points_epoch1}
        map2 = {p.name: p for p in points_epoch2}
        common_names = sorted(set(map1.keys()) & set(map2.keys()))

        self._results = []
        for name in common_names:
            result = self._analyze_single_point(map1[name], map2[name])
            self._results.append(result)

        return self

    def _analyze_single_point(self, p1, p2):
        name = p1.name
        x1 = np.array([p1.x, p1.y, p1.z], dtype=float)
        x2 = np.array([p2.x, p2.y, p2.z], dtype=float)
        delta = x2 - x1
        displacement = float(np.linalg.norm(delta))

        cov1 = getattr(p1, "cov_xyz", None)
        cov2 = getattr(p2, "cov_xyz", None)
        cross_cov = self.cross_cov_map.get(name, np.zeros((3, 3), dtype=float))

        reliable = (cov1 is not None) and (cov2 is not None)

        if not reliable:
            return self._make_unreliable_result(name, delta, displacement)

        cov1 = np.asarray(cov1, dtype=float)
        cov2 = np.asarray(cov2, dtype=float)
        cross_cov = np.asarray(cross_cov, dtype=float)

        cov_delta = cov1 + cov2 - cross_cov - cross_cov.T
        cov_delta = 0.5 * (cov_delta + cov_delta.T)

        return self._compute_tests(name, delta, displacement, cov_delta, reliable=True)

    def _compute_tests(self, name, delta, displacement, cov_delta, reliable):
        diag = np.maximum(np.diag(cov_delta), 0.0)
        sigma_xyz = np.sqrt(diag)

        if displacement > 1e-16:
            g = (delta / displacement).reshape(1, 3)
            var_d = (g @ cov_delta @ g.T).item()
            sigma_displacement = float(np.sqrt(max(var_d, 0.0)))
        else:
            sigma_displacement = float(np.sqrt(np.trace(cov_delta) / 3.0))

        if sigma_displacement > 1e-16:
            t_value = displacement / sigma_displacement
        else:
            t_value = 0.0

        p_value_t = float(2.0 * (1.0 - stats.norm.cdf(abs(t_value))))
        significant_t = p_value_t < self.alpha

        chi2_value = None
        p_value_chi2 = None
        significant_chi2 = None

        try:
            cov_inv = np.linalg.inv(cov_delta)
            chi2_value = float(delta @ cov_inv @ delta)
            p_value_chi2 = float(1.0 - stats.chi2.cdf(chi2_value, df=3))
            significant_chi2 = p_value_chi2 < self.alpha
        except np.linalg.LinAlgError:
            pass

        return DeformationResult(
            name=name, delta=delta, displacement=displacement,
            sigma_displacement=sigma_displacement, cov_delta=cov_delta,
            sigma_dx=float(sigma_xyz[0]), sigma_dy=float(sigma_xyz[1]), sigma_dz=float(sigma_xyz[2]),
            t_value=t_value, p_value_t=p_value_t, significant_t=significant_t,
            chi2_value=chi2_value, p_value_chi2=p_value_chi2, significant_chi2=significant_chi2,
            alpha=self.alpha, reliable=reliable,
        )

    @staticmethod
    def _make_unreliable_result(name, delta, displacement):
        return DeformationResult(
            name=name, delta=delta, displacement=float(displacement),
            sigma_displacement=float("nan"), cov_delta=np.full((3, 3), float("nan")),
            sigma_dx=float("nan"), sigma_dy=float("nan"), sigma_dz=float("nan"),
            t_value=float("nan"), p_value_t=float("nan"), significant_t=False,
            chi2_value=None, p_value_chi2=None, significant_chi2=None,
            alpha=0.05, reliable=False,
        )

    @property
    def results(self):
        return list(self._results)

    @property
    def significant_results(self):
        return [r for r in self._results if r.significant_t]

    @property
    def n_significant(self):
        return sum(r.significant_t for r in self._results)

    def to_dataframe(self):
        return pd.DataFrame([r.as_dict() for r in self._results])

    def to_csv(self, file_path, index=False):
        self.to_dataframe().to_csv(file_path, index=index)

    def print_summary(self):
        n = len(self._results)
        n_sig = self.n_significant
        print(f"\n{'='*60}")
        print(f"DeformationAnalyzer: {n} точек, значимых смещений: {n_sig}/{n}")
        print(f"Уровень значимости α = {self.alpha}")
        print(f"{'='*60}")
        for r in sorted(self._results, key=lambda x: x.displacement, reverse=True):
            print(f"  {r}")
        print(f"{'='*60}\n")
