import numpy as np
import pytest

from spacecraft_sim.navigation import simulate_measurement


def test_measurement_without_noise_matches_true_state():
    true_state = np.array([
        7_071_000.0,
        0.0,
        0.0,
        7_508.0,
    ])

    rng = np.random.default_rng(42)

    measurement = simulate_measurement(
        true_state,
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        rng=rng,
    )

    assert np.array_equal(measurement, true_state)


def test_measurement_has_correct_shape():
    true_state = np.array([
        7_071_000.0,
        0.0,
        0.0,
        7_508.0,
    ])

    rng = np.random.default_rng(42)

    measurement = simulate_measurement(
        true_state,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        rng=rng,
    )

    assert measurement.shape == (4,)


def test_measurement_noise_changes_state():
    true_state = np.array([
        7_071_000.0,
        0.0,
        0.0,
        7_508.0,
    ])

    rng = np.random.default_rng(42)

    measurement = simulate_measurement(
        true_state,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        rng=rng,
    )

    assert not np.array_equal(measurement, true_state)


def test_invalid_state_shape_raises_error():
    true_state = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        simulate_measurement(
            true_state,
            position_noise_std=10.0,
            velocity_noise_std=0.1,
        )


def test_negative_noise_raises_error():
    true_state = np.array([
        7_071_000.0,
        0.0,
        0.0,
        7_508.0,
    ])

    with pytest.raises(ValueError):
        simulate_measurement(
            true_state,
            position_noise_std=-1.0,
            velocity_noise_std=0.1,
        )
def test_measurement_is_reproducible_with_same_rng_seed():
    true_state = np.array([
        7_071_000.0,
        0.0,
        0.0,
        7_508.0,
    ])

    rng_1 = np.random.default_rng(42)
    rng_2 = np.random.default_rng(42)

    measurement_1 = simulate_measurement(
        true_state,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        rng=rng_1,
    )

    measurement_2 = simulate_measurement(
        true_state,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        rng=rng_2,
    )

    assert np.array_equal(measurement_1, measurement_2)