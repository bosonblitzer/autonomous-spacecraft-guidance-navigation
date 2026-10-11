
import numpy as np
import pytest

from spacecraft_sim.closed_loop import run_closed_loop_simulation


@pytest.mark.parametrize(
    "position_noise_std,velocity_noise_std",
    [
        (0.0, 0.0),
        (10.0, 0.01),
        (100.0, 0.1),
    ],
)
def test_simulation_remains_finite_under_measurement_noise(
    position_noise_std,
    velocity_noise_std,
):
    radius = 6.771e6
    speed = np.sqrt(3.986004418e14 / radius)
    reference = np.array([radius, 0.0, 0.0, speed])

    results = run_closed_loop_simulation(
        initial_true_state=reference + np.array([100.0, -100.0, 1.0, -1.0]),
        initial_estimate=reference + np.array([120.0, -120.0, 1.2, -1.2]),
        reference_state=reference,
        initial_covariance=np.diag([10000.0, 10000.0, 1.0, 1.0]),
        process_noise=np.diag([0.1, 0.1, 1e-4, 1e-4]),
        measurement_noise=np.diag(
            [
                max(position_noise_std, 1e-6) ** 2,
                max(position_noise_std, 1e-6) ** 2,
                max(velocity_noise_std, 1e-6) ** 2,
                max(velocity_noise_std, 1e-6) ** 2,
            ]
        ),
        dt=1.0,
        num_steps=100,
        position_noise_std=position_noise_std,
        velocity_noise_std=velocity_noise_std,
        position_gain=1e-6,
        velocity_gain=1e-3,
        max_acceleration=0.05,
        rng=np.random.default_rng(7),
    )

    for result in results:
        assert np.all(np.isfinite(result))

    control = results[-1]
    assert np.all(np.linalg.norm(control, axis=1) <= 0.05 + 1e-12)
