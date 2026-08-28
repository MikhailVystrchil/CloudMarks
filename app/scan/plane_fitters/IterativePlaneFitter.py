import numpy as np

from app.scan.Scan import Scan
from app.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC
from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter
from app.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


class IterativePlaneFitter(PlaneFitterABC):
    """
    Схема: robust trimming (PlaneL1Fitter) -> clean inliers ->
    final LSM fit (PlaneLSMFitter) -> надёжная ковариация.
    """

    def __init__(self, scan: Scan):
        super().__init__(scan)
        self.cov_params = None
        self.sigma0 = None
        self.final_plane = None
        self.filtered_scan = None

    def fit_plane(self, *args, mse_threshold=0.001, max_iteration=20, k_sigma=3,
                  base_fitter=PlaneL1Fitter, final_fitter=PlaneLSMFitter,
                  min_points=6, **kwargs):
        from app.scan.ScanPlane import ScanPlane

        current_scan = self.scan
        robust_plane = None

        for _ in range(max_iteration):
            if len(current_scan) < 3:
                raise RuntimeError("После фильтрации осталось меньше 3 точек")

            robust_plane = ScanPlane.fit_plane_to_scan(
                scan=current_scan, fitter=base_fitter, *args, **kwargs)

            if robust_plane.mse <= mse_threshold:
                break

            next_scan = self._filter_outliers_by_k_sigma(current_scan, robust_plane, k_sigma)

            if len(next_scan) < min_points:
                break
            if len(next_scan) == len(current_scan):
                current_scan = next_scan
                break
            current_scan = next_scan

        if robust_plane is None:
            raise RuntimeError("Не удалось оценить плоскость: robust_plane is None")

        self.filtered_scan = current_scan

        final_plane = ScanPlane.fit_plane_to_scan(scan=current_scan, fitter=final_fitter, *args, **kwargs)

        self.final_plane = final_plane
        self.cov_params = getattr(final_plane, "cov_params", None)
        self.sigma0 = getattr(final_plane, "sigma0", None)
        self.mse = getattr(final_plane, "mse", None)

        return current_scan, final_plane.normal, final_plane.point, final_plane.d

    @staticmethod
    def _filter_outliers_by_k_sigma(current_scan, current_plane, k_sigma):
        pts = np.array([[p.x, p.y, p.z] for p in current_scan], dtype=float)
        dists = current_plane.distance_to_point(pts)

        mean = float(np.mean(dists))
        std = float(np.std(dists))
        threshold = mean + k_sigma * std

        filtered_points = [p for p, dist in zip(current_scan, dists) if dist <= threshold]

        f_scan = Scan(scan_name=f"{current_scan.name}_filtered")
        f_scan._points = filtered_points
        f_scan.borders = f_scan._get_borders_dict(f_scan._points)

        return f_scan
