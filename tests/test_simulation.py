import numpy as np
import pytest

from spacecraft_sim.simulation import run_navigation_simulation


def create_simulation_parameters():
    initial_true_state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    initial_estimate = np.array([
        7_000_100.0,
        -50.0,
        5.0,
        7_495.0,
    ])

    initial_covariance = np.diag([
        100.0**2,
        100.0**2,
        10.0**2,
        10.0**2,
    ])

    process_noise = np.diag([
        1.0,
        1.0,
        0.1,
        0.1,
    ])

    measurement_noise = np.diag([
        10.0**2,
        10.0**2,
        0.1**2,
        0.1**2,
    ])

    return (
        initial_true_state,
        initial_estimate,
        initial_covariance,
        process_noise,
        measurement_noise,
    )


def test_navigation_simulation_shapes():
    parameters = create_simulation_parameters()

    (
        times,
        true_history,
        estimated_history,
        covariance_history,
    ) = run_navigation_simulation(
        *parameters,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        dt=1.0,
        num_steps=100,
        rng=np.random.default_rng(42),
    )

    assert times.shape == (101,)
    assert true_history.shape == (101, 4)
    assert estimated_history.shape == (101, 4)
    assert covariance_history.shape == (101, 4, 4)


def test_navigation_simulation_changes_true_state():
    parameters = create_simulation_parameters()

    (
        _,
        true_history,
        _,
        _,
    ) = run_navigation_simulation(
        *parameters,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        dt=1.0,
        num_steps=100,
        rng=np.random.default_rng(42),
    )

    assert not np.array_equal(
        true_history[0],
        true_history[-1],
    )


def test_navigation_simulation_is_reproducible():
    parameters = create_simulation_parameters()

    result_1 = run_navigation_simulation(
        *parameters,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        dt=1.0,
        num_steps=100,
        rng=np.random.default_rng(42),
    )

    result_2 = run_navigation_simulation(
        *parameters,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        dt=1.0,
        num_steps=100,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(result_1[0], result_2[0])
    assert np.array_equal(result_1[1], result_2[1])
    assert np.array_equal(result_1[2], result_2[2])
    assert np.array_equal(result_1[3], result_2[3])


def test_navigation_estimate_moves_towards_truth():
    parameters = create_simulation_parameters()

    (
        _,
        true_history,
        estimated_history,
        _,
    ) = run_navigation_simulation(
        *parameters,
        position_noise_std=10.0,
        velocity_noise_std=0.1,
        dt=1.0,
        num_steps=100,
        rng=np.random.default_rng(42),
    )

    initial_error = np.linalg.norm(
        estimated_history[0] - true_history[0]
    )

    final_error = np.linalg.norm(
        estimated_history[-1] - true_history[-1]
    )

    assert final_error < initial_error


def test_navigation_simulation_rejects_invalid_timestep():
    parameters = create_simulation_parameters()

    with pytest.raises(ValueError):
        run_navigation_simulation(
            *parameters,
            position_noise_std=10.0,
            velocity_noise_std=0.1,
            dt=0.0,
            num_steps=100,
        )
