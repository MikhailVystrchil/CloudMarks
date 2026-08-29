import numpy as np
import pytest

from app.scan.plane_fitters.IterativePlaneFitter import (
    IterativePlaneFitter,
)


def test_iterative_fitter_removes_outliers_and_fits_clean_plane(
    make_scan,
):
    """
    Итеративный фиттер должен исключить грубые выбросы и получить
    корректную окончательную плоскость z = 1.
    """
    random_generator = np.random.default_rng(3)

    x_coord = random_generator.uniform(
        -2.0,
        2.0,
        300,
    )
    y_coord = random_generator.uniform(
        -2.0,
        2.0,
        300,
    )
    z_coord = (
        np.full_like(x_coord, 1.0)
        + random_generator.normal(
            scale=0.001,
            size=x_coord.size,
        )
    )

    outlier_indices = random_generator.choice(
        len(x_coord),
        size=15,
        replace=False,
    )
    z_coord[outlier_indices] += 3.0

    scan = make_scan(
        "noisy_plane",
        np.column_stack(
            [
                x_coord,
                y_coord,
                z_coord,
            ]
        ),
    )

    fitter = IterativePlaneFitter(scan)

    filtered_scan, normal, _, d = fitter.fit_plane(
        mse_threshold=0.01,
        max_iteration=15,
        k_sigma=2.5,
        min_points=6,
    )

    if normal[2] < 0.0:
        normal = -normal
        d = -d

    assert len(filtered_scan) < len(scan)
    assert len(filtered_scan) >= 6

    assert np.all(np.isfinite(normal))
    assert np.isfinite(d)

    assert np.isclose(
        np.linalg.norm(normal),
        1.0,
        atol=1e-12,
    )
    assert np.isclose(
        abs(normal[2]),
        1.0,
        atol=0.02,
    )
    assert np.isclose(
        d,
        -1.0,
        atol=0.03,
    )

    assert fitter.cov_params is not None
    assert fitter.cov_params.shape == (4, 4)
    assert np.all(np.isfinite(fitter.cov_params))

    assert fitter.sigma0 is not None
    assert fitter.sigma0 >= 0.0


def test_iterative_fitter_raises_when_scan_has_too_few_points(
    make_scan,
):
    """
    min_points=4 является допустимым параметром, но двух фактических
    точек недостаточно для аппроксимации, поэтому ожидается RuntimeError.
    """
    scan = make_scan(
        "tiny",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )

    fitter = IterativePlaneFitter(scan)

    with pytest.raises(
        RuntimeError,
        match="недостаточно точек",
    ):
        fitter.fit_plane(min_points=4)


def test_iterative_fitter_rejects_invalid_min_points(
    make_scan,
):
    """
    Значение min_points < 4 некорректно независимо от размера Scan.
    """
    scan = make_scan(
        "valid_scan",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )

    fitter = IterativePlaneFitter(scan)

    with pytest.raises(
        ValueError,
        match="min_points должен быть не меньше 4",
    ):
        fitter.fit_plane(min_points=3)
