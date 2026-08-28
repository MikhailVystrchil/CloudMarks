from __future__ import annotations

import numpy as np
from loguru import logger

from app.base.Plane import Plane
from app.scan.Scan import Scan
from app.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class ScanPlane(Plane):
    """
    Плоскость, подогнанная к облаку точек.

    Attributes
    ----------
    scan:
        Очищенное облако точек, использованное при финальной аппроксимации.
    mse:
        RMSE ортогональных расстояний точек до плоскости.
    cov_params:
        Ковариационная матрица параметров ``(A, B, C, D)``.
    sigma0:
        Оценка СКП единицы веса.
    """

    def __init__(
        self,
        normal: np.ndarray,
        point_on_plane: np.ndarray,
        d: float,
    ) -> None:
        super().__init__(
            normal=normal,
            point_on_plane=point_on_plane,
            d=d,
        )

        self.scan: Scan | None = None
        self.mse: float | None = None
        self.cov_params: np.ndarray | None = None
        self.sigma0: float | None = None

    @classmethod
    def fit_plane_to_scan(
        cls,
        scan: Scan,
        *args,
        fitter: type[PlaneLSMFitter] = PlaneLSMFitter,
        **kwargs,
    ) -> "ScanPlane":
        """
        Строит ScanPlane по заданному фиттеру.
        """
        fitter_instance = fitter(scan=scan)

        scan_out, normal, point_on_plane, d = (
            fitter_instance.fit_plane(
                *args,
                **kwargs,
            )
        )

        scan_plane = cls(
            normal=normal,
            point_on_plane=point_on_plane,
            d=d,
        )
        scan_plane.scan = scan_out
        scan_plane.mse = scan_plane._compute_mse_for_scan(
            scan=scan_out
        )
        scan_plane.cov_params = getattr(
            fitter_instance,
            "cov_params",
            None,
        )
        scan_plane.sigma0 = getattr(
            fitter_instance,
            "sigma0",
            None,
        )

        logger.info(
            "Аппроксимация плоскости завершена: fitter={}, "
            "points={}, rmse={:.6f}",
            getattr(fitter, "__name__", str(fitter)),
            len(scan_out),
            scan_plane.mse,
        )

        return scan_plane

    def _compute_mse_for_scan(
        self,
        scan: Scan,
    ) -> float:
        """
        Вычисляет RMSE ортогональных расстояний точек от плоскости.
        """
        if len(scan) == 0:
            raise ValueError(
                "Невозможно вычислить RMSE для пустого Scan."
            )

        distances = self.distance_to_point(
            scan.to_numpy()
        )

        mse = float(
            np.sqrt(np.mean(distances**2))
        )

        self.mse = mse
        return mse

    def has_covariance(self) -> bool:
        return self.cov_params is not None

    def __repr__(self) -> str:
        a, b, c, d = self.equation

        covariance_text = (
            np.array2string(
                self.cov_params,
                precision=6,
                suppress_small=True,
            )
            if self.cov_params is not None
            else "None"
        )

        sigma0_text = (
            f"{self.sigma0:.6f}"
            if self.sigma0 is not None
            else "None"
        )

        point_count = (
            len(self.scan)
            if self.scan is not None
            else 0
        )

        mse_text = (
            f"{self.mse:.6f}"
            if self.mse is not None
            else "None"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  mse={mse_text}, "
            f"sigma0={sigma0_text}, "
            f"scan_len={point_count},\n"
            f"  A={a:.6f}, B={b:.6f}, "
            f"C={c:.6f}, D={d:.6f},\n"
            f"  cov_params=\n{covariance_text}\n"
            f")"
        )
