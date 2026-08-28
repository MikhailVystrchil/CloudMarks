import numpy as np

from app.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC


class PlaneLSMFitter(PlaneFitterABC):
    """
    МНК-подгонка плоскости через PCA (SVD центрированных точек).

    Модель:  n^T * x + d = 0,  где ||n|| = 1.

    Ковариация вектора параметров p = (A,B,C,D):
        Sigma_p = sigma0^2 * (X^T X)^{-1}
    где  sigma0^2 = sum(v_i^2) / (N - 4)  — несмещённая оценка дисперсии остатков.
    """

    def fit_plane(self, *args, **kwargs):
        pts = self._scan_to_numpy()
        n_pts = pts.shape[0]
        if n_pts < 4:
            raise ValueError("Для оценки ковариации нужно минимум 4 точки")

        centroid = pts.mean(axis=0)
        centered = pts - centroid

        _, sv, vh = np.linalg.svd(centered, full_matrices=False)

        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)

        point_on_plane = centroid

        ones = np.ones((n_pts, 1), dtype=float)
        X = np.hstack([pts, ones])          # (N, 4)
        XtX = X.T @ X                       # (4, 4)

        residuals = pts @ normal + d        # (N,)

        dof = n_pts - 4
        if dof <= 0:
            self.sigma0 = 0.0
            self.cov_params = np.zeros((4, 4), dtype=float)
        else:
            sigma0_sq = float(np.sum(residuals ** 2) / dof)
            self.sigma0 = float(np.sqrt(sigma0_sq))

            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)

            self.cov_params = sigma0_sq * XtX_inv   # (4, 4)

        scan = self.scan
        return scan, normal, point_on_plane, d
