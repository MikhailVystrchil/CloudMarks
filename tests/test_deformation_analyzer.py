import numpy as np

from app.cross_points.CrossPoint import CrossPoint
from app.deformation.DeformationAnalyzer import DeformationAnalyzer


def _make_cross_point(
    name: str,
    coordinates: tuple[float, float, float],
    covariance: np.ndarray,
) -> CrossPoint:
    point = CrossPoint(
        name=name,
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
    )
    point.load_covariance(covariance)
    return point


def test_zero_displacement_is_not_significant():
    """
    Для одинаковых координат в двух независимых эпохах смещение равно нулю
    и не должно быть признано статистически значимым.
    """
    covariance = np.diag([1e-6, 1e-6, 1e-6])

    point_epoch_1 = _make_cross_point(
        "P1",
        (1.0, 2.0, 3.0),
        covariance,
    )
    point_epoch_2 = _make_cross_point(
        "P1",
        (1.0, 2.0, 3.0),
        covariance,
    )

    analyzer = DeformationAnalyzer(alpha=0.05)
    analyzer.analyze_point_sets([point_epoch_1], [point_epoch_2])

    result = analyzer.results[0]

    assert result.reliable
    assert np.isclose(result.displacement, 0.0, atol=1e-15)
    assert not result.significant_t
    assert result.chi2_value is not None
    assert np.isclose(result.chi2_value, 0.0, atol=1e-12)
    assert not result.significant_chi2


def test_known_displacement_is_significant():
    """
    Смещение 10 мм при СКП каждой координаты 0.1 мм в каждой эпохе
    должно уверенно обнаруживаться обоими критериями.
    """
    covariance = np.diag([1e-8, 1e-8, 1e-8])

    point_epoch_1 = _make_cross_point(
        "P1",
        (0.0, 0.0, 0.0),
        covariance,
    )
    point_epoch_2 = _make_cross_point(
        "P1",
        (0.010, 0.0, 0.0),
        covariance,
    )

    analyzer = DeformationAnalyzer(alpha=0.05)
    analyzer.analyze_point_sets([point_epoch_1], [point_epoch_2])

    result = analyzer.results[0]

    assert result.reliable
    assert np.isclose(result.displacement, 0.010, atol=1e-15)
    assert result.sigma_dx > 0.0
    assert result.significant_t
    assert result.chi2_value is not None
    assert result.chi2_value > 7.8147
    assert result.significant_chi2
