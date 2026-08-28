import numpy as np
import pytest

from app.cross_points.CrossPointExacter import (
    CrossPointExacter,
    PlaneGeometryDiagnostics,
    PlaneGeometryStatus,
)


def test_intersection_of_three_orthogonal_planes(orthogonal_planes):
    """
    Проверяет решение СЛАУ:
        x = 1,
        y = 2,
        z = 3.
    """
    normal_matrix = np.array(
        [[plane.A, plane.B, plane.C] for plane in orthogonal_planes],
        dtype=float,
    )
    right_side = np.array(
        [-plane.D for plane in orthogonal_planes],
        dtype=float,
    )

    point = np.linalg.solve(normal_matrix, right_side)

    assert np.allclose(point, np.array([1.0, 2.0, 3.0]), atol=1e-12)


def test_geometry_is_reliable_for_orthogonal_planes(orthogonal_planes):
    diagnostics = PlaneGeometryDiagnostics(
        orthogonal_planes,
        cond_threshold=1000.0,
        angle_tol_rad=np.deg2rad(10.0),
    )

    assert diagnostics.status == PlaneGeometryStatus.GOOD
    assert diagnostics.is_reliable
    assert not diagnostics.has_parallel
    assert np.isclose(diagnostics.det, 1.0, atol=1e-12)
    assert np.isclose(diagnostics.cond, 1.0, atol=1e-12)


def test_geometry_rejects_parallel_planes(make_scan_plane):
    """
    Две плоскости x = 1 и x = 2 параллельны; точка пересечения
    трёх плоскостей определена быть не может.
    """
    planes = [
        make_scan_plane((1.0, 0.0, 0.0), -1.0),
        make_scan_plane((1.0, 0.0, 0.0), -2.0),
        make_scan_plane((0.0, 0.0, 1.0), -3.0),
    ]

    diagnostics = PlaneGeometryDiagnostics(
        planes,
        cond_threshold=1000.0,
        angle_tol_rad=np.deg2rad(10.0),
    )

    assert diagnostics.has_parallel
    assert diagnostics.status in {
        PlaneGeometryStatus.PARALLEL,
        PlaneGeometryStatus.SINGULAR,
    }
    assert not diagnostics.is_reliable


def test_geometry_rejects_ill_conditioned_planes(make_scan_plane):
    """
    Нормали первой и второй плоскостей очень близки, но формально
    не параллельны. Такая геометрия должна быть распознана как
    плохо обусловленная при строгом пороге condition number.
    """
    small_angle = np.deg2rad(0.5)

    planes = [
        make_scan_plane((1.0, 0.0, 0.0), -1.0),
        make_scan_plane(
            (np.cos(small_angle), np.sin(small_angle), 0.0),
            -1.0,
        ),
        make_scan_plane((0.0, 0.0, 1.0), -3.0),
    ]

    diagnostics = PlaneGeometryDiagnostics(
        planes,
        cond_threshold=10.0,
        angle_tol_rad=np.deg2rad(0.1),
    )

    assert not diagnostics.has_parallel
    assert diagnostics.status == PlaneGeometryStatus.ILL_CONDITIONED
    assert diagnostics.cond > 10.0
    assert not diagnostics.is_reliable


def test_covariance_propagation_is_symmetric_and_nonnegative(orthogonal_planes):
    """
    Матрица ковариации координат виртуальной точки должна быть
    симметричной и положительно полуопределённой.
    """
    covariance_xyz = CrossPointExacter._propagate_covariance(
        orthogonal_planes
    )

    assert covariance_xyz.shape == (3, 3)
    assert np.allclose(covariance_xyz, covariance_xyz.T, atol=1e-12)
    assert np.all(np.isfinite(covariance_xyz))

    eigenvalues = np.linalg.eigvalsh(covariance_xyz)
    assert np.all(eigenvalues >= -1e-12)
