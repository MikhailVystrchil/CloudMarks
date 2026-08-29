import pytest

from app.batch.ExtractionConfig import ExtractionConfig


def test_extraction_config_accepts_valid_values():
    config = ExtractionConfig(
        default_radius=0.6,
        min_neighborhood_points=60,
        min_points_per_plane=15,
        max_reference_distance_factor=1.25,
        normal_k=12,
        cluster_eps=0.08,
        cluster_min_samples=3,
    )

    assert config.default_radius == 0.6
    assert config.min_neighborhood_points == 60
    assert config.min_points_per_plane == 15
    assert config.max_reference_distance_factor == 1.25
    assert config.normal_k == 12
    assert config.cluster_eps == 0.08
    assert config.cluster_min_samples == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"default_radius": 0.0},
            "default_radius",
        ),
        (
            {
                "default_radius": 0.6,
                "min_points_per_plane": 5,
            },
            "min_points_per_plane",
        ),
        (
            {
                "default_radius": 0.6,
                "min_neighborhood_points": 20,
                "min_points_per_plane": 10,
            },
            "min_neighborhood_points",
        ),
        (
            {
                "default_radius": 0.6,
                "max_reference_distance_factor": 0.0,
            },
            "max_reference_distance_factor",
        ),
        (
            {
                "default_radius": 0.6,
                "normal_k": 2,
            },
            "normal_k",
        ),
        (
            {
                "default_radius": 0.6,
                "cluster_eps": 0.0,
            },
            "cluster_eps",
        ),
        (
            {
                "default_radius": 0.6,
                "cluster_min_samples": 0,
            },
            "cluster_min_samples",
        ),
    ],
)
def test_extraction_config_rejects_invalid_values(
    kwargs,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        ExtractionConfig(**kwargs)


def test_extraction_config_is_immutable():
    config = ExtractionConfig(
        default_radius=0.6,
    )

    with pytest.raises(AttributeError):
        config.default_radius = 1.0
