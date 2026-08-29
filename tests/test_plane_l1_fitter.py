import numpy as np

from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter


def test_l1_fitter_is_robust_to_outliers(make_scan):
    rng = np.random.default_rng(1)

    x = rng.uniform(-2.0, 2.0, 200)
    y = rng.uniform(-2.0, 2.0, 200)
    z = np.zeros_like(x)  # плоскость z = 0

    # Добавляем 10% выбросов с большим отклонением по Z
    outlier_count = 20
    z[:outlier_count] += 5.0

    coords = np.column_stack([x, y, z])
    scan = make_scan("outlier_plane", coords)

    fitter = PlaneL1Fitter(scan)
    _, normal, _, d = fitter.fit_plane()

    normal = normal if normal[2] > 0 else -normal

    assert np.isclose(abs(normal[2]), 1.0, atol=0.05)
    assert abs(d) < 0.5  # плоскость должна остаться близкой к z=0 несмотря на выбросы


def test_l1_fitter_requires_at_least_three_points(make_scan):
    import pytest

    scan = make_scan("two_points", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    fitter = PlaneL1Fitter(scan)

    with pytest.raises(ValueError):
        fitter.fit_plane()


def test_l1_fitter_sets_cov_params_shape(make_scan):
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 50)
    y = rng.uniform(-1, 1, 50)
    z = np.full_like(x, 2.0)

    scan = make_scan("s", np.column_stack([x, y, z]))
    fitter = PlaneL1Fitter(scan)
    fitter.fit_plane()

    assert fitter.cov_params.shape == (4, 4)
    assert fitter.sigma0 is not None and fitter.sigma0 >= 0.0
