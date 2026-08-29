import numpy as np

from app.scan.plane_fitters.IterativePlaneFitter import IterativePlaneFitter


def test_iterative_fitter_removes_outliers_and_fits_clean_plane(make_scan):
    rng = np.random.default_rng(3)

    x = rng.uniform(-2.0, 2.0, 300)
    y = rng.uniform(-2.0, 2.0, 300)
    z = np.full_like(x, 1.0) + rng.normal(scale=0.001, size=x.size)

    # 5% выбросов
    outlier_idx = rng.choice(len(x), size=15, replace=False)
    z[outlier_idx] += 3.0

    scan = make_scan("s", np.column_stack([x, y, z]))
    fitter = IterativePlaneFitter(scan)

    filtered_scan, normal, _, d = fitter.fit_plane(
        mse_threshold=0.01,
        max_iteration=15,
        k_sigma=2.5,
        min_points=6,
    )

    normal = normal if normal[2] > 0 else -normal
    assert np.isclose(abs(normal[2]), 1.0, atol=0.02)
    assert len(filtered_scan) < len(scan)  # выбросы должны быть отфильтрованы
    assert fitter.cov_params is not None
    assert fitter.sigma0 is not None


def test_iterative_fitter_raises_when_too_few_points(make_scan):
    import pytest

    scan = make_scan("tiny", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    fitter = IterativePlaneFitter(scan)

    with pytest.raises(ValueError):
        fitter.fit_plane(min_points=4)
