import numpy as np
import pytest

from app.scan.plane_fitters.PlaneLSMFitter import PlaneLSMFitter


def _absolute_plane_offset(normal: np.ndarray, d: float) -> float:
    """
    Для сравнения плоскостей с возможным изменением знака нормали.
    """
    return abs(float(d))


def test_lsm_fits_exact_horizontal_plane(make_scan):
    """
    Плоскость z = 3 должна быть восстановлена с машинной точностью.
    """
    x_coord, y_coord = np.meshgrid(
        np.linspace(-2.0, 2.0, 7),
        np.linspace(-3.0, 3.0, 9),
    )
    z_coord = np.full_like(x_coord, 3.0)

    coordinates = np.column_stack([
        x_coord.ravel(),
        y_coord.ravel(),
        z_coord.ravel(),
    ])

    scan = make_scan("exact_plane", coordinates)
    fitter = PlaneLSMFitter(scan)

    _, normal, _, d = fitter.fit_plane()

    assert np.isclose(abs(normal[2]), 1.0, atol=1e-12)
    assert np.isclose(normal[0], 0.0, atol=1e-12)
    assert np.isclose(normal[1], 0.0, atol=1e-12)
    assert np.isclose(_absolute_plane_offset(normal, d), 3.0, atol=1e-12)

    assert fitter.cov_params.shape == (4, 4)
    assert np.allclose(fitter.cov_params, fitter.cov_params.T, atol=1e-12)
    assert np.isclose(fitter.sigma0, 0.0, atol=1e-12)


def test_lsm_fits_noisy_plane(make_scan):
    """
    На шумных данных плоскости z = 0.5x - 0.25y + 1.2 нормаль и
    свободный член должны быть восстановлены с разумной точностью.
    """
    random_generator = np.random.default_rng(42)

    x_coord = random_generator.uniform(-2.0, 2.0, 500)
    y_coord = random_generator.uniform(-2.0, 2.0, 500)

    z_without_noise = 0.5 * x_coord - 0.25 * y_coord + 1.2
    z_coord = z_without_noise + random_generator.normal(
        loc=0.0,
        scale=0.002,
        size=x_coord.size,
    )

    coordinates = np.column_stack([x_coord, y_coord, z_coord])

    scan = make_scan("noisy_plane", coordinates)
    fitter = PlaneLSMFitter(scan)

    _, estimated_normal, _, estimated_d = fitter.fit_plane()

    expected_normal = np.array([-0.5, 0.25, 1.0], dtype=float)
    expected_normal /= np.linalg.norm(expected_normal)
    expected_d = -1.2 / np.linalg.norm(np.array([-0.5, 0.25, 1.0]))

    if np.dot(estimated_normal, expected_normal) < 0:
        estimated_normal = -estimated_normal
        estimated_d = -estimated_d

    assert np.allclose(estimated_normal, expected_normal, atol=3e-3)
    assert np.isclose(estimated_d, expected_d, atol=3e-3)

    assert fitter.sigma0 > 0.0
    assert fitter.cov_params.shape == (4, 4)
    assert np.all(np.isfinite(fitter.cov_params))


def test_lsm_requires_at_least_four_points(make_scan):
    """
    Ковариация четырёх параметров плоскости не должна вычисляться
    по выборке из трёх точек.
    """
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    scan = make_scan("three_points", coordinates)
    fitter = PlaneLSMFitter(scan)

    with pytest.raises(ValueError, match="минимум 4 точки"):
        fitter.fit_plane()
