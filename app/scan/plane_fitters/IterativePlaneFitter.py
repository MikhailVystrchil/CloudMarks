from __future__ import annotations

import numpy as np

from app.scan.Scan import Scan
from app.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC
from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class IterativePlaneFitter(PlaneFitterABC):
    """
    Двухэтапная подгонка плоскости:

    1. Робастная L1-оценка через IRLS.
    2. Итеративное исключение выбросов по правилу k-sigma.
    3. Финальная МНК-аппроксимация очищенного набора.
    4. Оценка ковариации параметров финальной плоскости.
    """

    def __init__(
        self,
        scan: Scan,
    ) -> None:
        super().__init__(scan=scan)

        self.cov_params: np.ndarray | None = None
        self.sigma0: float | None = None
        self.mse: float | None = None

        self.final_plane = None
        self.filtered_scan: Scan | None = None

    def fit_plane(
        self,
        *args,
        mse_threshold: float = 0.001,
        max_iteration: int = 20,
        k_sigma: float = 3.0,
        base_fitter: type[PlaneL1Fitter] = PlaneL1Fitter,
        final_fitter: type[PlaneLSMFitter] = PlaneLSMFitter,
        min_points: int = 6,
        **kwargs,
    ):
        from app.scan.ScanPlane import ScanPlane

        if max_iteration < 1:
            raise ValueError(
                "max_iteration должен быть положительным."
            )

        if min_points < 4:
            raise ValueError(
                "min_points должен быть не меньше 4."
            )

        current_scan = self.scan
        robust_plane = None

        for _ in range(max_iteration):
            if len(current_scan) < min_points:
                raise RuntimeError(
                    "После фильтрации осталось недостаточно точек "
                    f"для аппроксимации: {len(current_scan)}."
                )

            robust_plane = ScanPlane.fit_plane_to_scan(
                scan=current_scan,
                fitter=base_fitter,
                *args,
                **kwargs,
            )

            if robust_plane.mse <= mse_threshold:
                break

            next_scan = self._filter_outliers_by_k_sigma(
                current_scan=current_scan,
                current_plane=robust_plane,
                k_sigma=k_sigma,
            )

            if len(next_scan) < min_points:
                break

            if len(next_scan) == len(current_scan):
                current_scan = next_scan
                break

            current_scan = next_scan

        if robust_plane is None:
            raise RuntimeError(
                "Не удалось оценить робастную плоскость."
            )

        if len(current_scan) < min_points:
            raise RuntimeError(
                "После очистки осталось недостаточно точек "
                f"для финального МНК: {len(current_scan)}."
            )

        self.filtered_scan = current_scan

        final_plane = ScanPlane.fit_plane_to_scan(
            scan=current_scan,
            fitter=final_fitter,
            *args,
            **kwargs,
        )

        self.final_plane = final_plane
        self.cov_params = final_plane.cov_params
        self.sigma0 = final_plane.sigma0
        self.mse = final_plane.mse

        return (
            current_scan,
            final_plane.normal,
            final_plane.point,
            final_plane.d,
        )

    @staticmethod
    def _filter_outliers_by_k_sigma(
        current_scan: Scan,
        current_plane,
        k_sigma: float,
    ) -> Scan:
        """
        Исключает точки с абсолютной невязкой выше ``mean + k_sigma * std``.
        """
        if k_sigma <= 0:
            raise ValueError(
                "k_sigma должен быть положительным."
            )

        coordinates = current_scan.to_numpy()
        distances = current_plane.distance_to_point(coordinates)

        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))

        threshold = mean_distance + k_sigma * std_distance

        kept_indices = np.flatnonzero(
            distances <= threshold
        )

        return current_scan.subset(
            kept_indices,
            scan_name=f"{current_scan.name}_filtered",
            copy_points=False,
        )
