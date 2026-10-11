
import numpy as np
import pytest

from spacecraft_sim.closed_loop import run_closed_loop_simulation


@pytest.fixture
def simulation_inputs():
    return {
        "initial_true_state": np.array([7.0e6, 0.0, 0.0, 7500.0]),
        "initial_estimate": np.array([7.0e6 + 100.0, -100.0, 1.0, 7499.0]),
        "reference_state": np.array([7.0e6, 500.0, 0.0, 7500.0]),
        "initial_covariance": np.diag([100.0**2, 100.0**2, 1.0, 1.0]),
        "process_noise": np.diag([1.0, 1.0, 0.01, 0.01]),
        "measurement_noise": np.diag([10.0**2, 10.0**2, 0.01, 0.01]),
        "dt": 1.0,
        "num_steps": 20,
        "position_noise_std": 10.0,
        "velocity_noise_std": 0.01,
        "position_gain": 1e-6,
        "velocity_gain": 1e-3,
        "max_acceleration": 0.05,
        "rng": np.random.default_rng(42),
    }


def test_closed_loop_output_shapes(simulation_inputs):
    results = run_closed_loop_simulation(**simulation_inputs)

    times, truth, estimate, reference, covariance, control = results

    assert times.shape == (21,)
    assert truth.shape == (21, 4)
    assert estimate.shape == (21, 4)
    assert reference.shape == (21, 4)
    assert covariance.shape == (21, 4, 4)
    assert control.shape == (20, 2)


def test_closed_loop_outputs_are_finite(simulation_inputs):
    results = run_closed_loop_simulation(**simulation_inputs)

    for result in results:
        assert np.all(np.isfinite(result))


def test_control_respects_acceleration_limit(simulation_inputs):
    max_acceleration = simulation_inputs["max_acceleration"]
    *_, control = run_closed_loop_simulation(**simulation_inputs)

    magnitudes = np.linalg.norm(control, axis=1)

    assert np.all(magnitudes <= max_acceleration + 1e-12)


def test_invalid_timestep_is_rejected(simulation_inputs):
    simulation_inputs["dt"] = 0.0

    with pytest.raises(ValueError, match="dt must be positive"):
        run_closed_loop_simulation(**simulation_inputs)


def test_invalid_step_count_is_rejected(simulation_inputs):
    simulation_inputs["num_steps"] = 0

    with pytest.raises(ValueError, match="num_steps"):
        run_closed_loop_simulation(**simulation_inputs)
