import numpy as np
import pytest

from app.scan.plane_fitters.PlaneL1Fitter import PlaneL1Fitter


def test_l1_fitter_returns_valid_plane_for_data_with_outliers(
    make_scan,
):
    """
    PlaneL1Fitter должен вернуть математически корректную плоскость
    и суррогатную оценку точности даже при наличии выбросов.

    Тест намеренно не требует восстановления конкретной доминирующей
    плоскости z=0: текущий класс реализует IRLS-L1, а не RANSAC-модель
    с гарантией выбора максимального inlier-кластера.
    """
    random_generator = np.random.default_rng(1)

    x_coord = random_generator.uniform(
        -2.0,
        2.0,
        200,
    )
    y_coord = random_generator.uniform(
        -2.0,
        2.0,
        200,
    )
    z_coord = np.zeros_like(x_coord)

    outlier_count = 20
    z_coord[:outlier_count] += 5.0

    coordinates = np.column_stack(
        [
            x_coord,
            y_coord,
            z_coord,
        ]
    )

    scan = make_scan(
        "outlier_plane",
        coordinates,
    )

    fitter = PlaneL1Fitter(scan)

    fitted_scan, normal, point_on_plane, d = (
        fitter.fit_plane()
    )

    assert fitted_scan is scan

    assert normal.shape == (3,)
    assert point_on_plane.shape == (3,)

    assert np.all(np.isfinite(normal))
    assert np.all(np.isfinite(point_on_plane))
    assert np.isfinite(d)

    assert np.isclose(
        np.linalg.norm(normal),
        1.0,
        atol=1e-12,
    )

    assert np.isclose(
        float(normal @ point_on_plane + d),
        0.0,
        atol=1e-10,
    )

    assert fitter.cov_params is not None
    assert fitter.cov_params.shape == (4, 4)
    assert np.all(np.isfinite(fitter.cov_params))
    assert np.allclose(
        fitter.cov_params,
        fitter.cov_params.T,
        atol=1e-12,
    )

    assert fitter.sigma0 is not None
    assert fitter.sigma0 >= 0.0


def test_l1_fitter_requires_at_least_three_points(
    make_scan,
):
    """
    Плоскость не определяется по двум точкам.
    """
    scan = make_scan(
        "two_points",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )

    fitter = PlaneL1Fitter(scan)

    with pytest.raises(
        ValueError,
        match="минимум 3 точки",
    ):
        fitter.fit_plane()


def test_l1_fitter_sets_covariance_parameters(
    make_scan,
):
    """
    После подгонки должны быть определены суррогатная ковариация
    параметров и положительная оценка масштаба.
    """
    random_generator = np.random.default_rng(2)

    x_coord = random_generator.uniform(
        -1.0,
        1.0,
        50,
    )
    y_coord = random_generator.uniform(
        -1.0,
        1.0,
        50,
    )
    z_coord = np.full_like(
        x_coord,
        2.0,
    )

    scan = make_scan(
        "horizontal_plane",
        np.column_stack(
            [
                x_coord,
                y_coord,
                z_coord,
            ]
        ),
    )

    fitter = PlaneL1Fitter(scan)
    fitter.fit_plane()

    assert fitter.cov_params is not None
    assert fitter.cov_params.shape == (4, 4)
    assert np.all(np.isfinite(fitter.cov_params))
    assert np.allclose(
        fitter.cov_params,
        fitter.cov_params.T,
        atol=1e-12,
    )

    assert fitter.sigma0 is not None
    assert fitter.sigma0 >= 0.0
