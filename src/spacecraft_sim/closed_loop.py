import numpy as np

from spacecraft_sim.control import calculate_control_acceleration
from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.ekf import ExtendedKalmanFilter
from spacecraft_sim.navigation import simulate_measurement
from spacecraft_sim.propagation import rk4_step


def run_closed_loop_simulation(
    initial_true_state: np.ndarray,
    initial_estimate: np.ndarray,
    initial_reference_state: np.ndarray,
    initial_covariance: np.ndarray,
    process_noise: np.ndarray,
    measurement_noise: np.ndarray,
    position_noise_std: float,
    velocity_noise_std: float,
    position_gain: float,
    velocity_gain: float,
    max_acceleration: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator | None = None,
    mu: float = EARTH_MU,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """Simulate reference tracking with estimated-state feedback."""

    for name, value in (
        ("initial_true_state", initial_true_state),
        ("initial_estimate", initial_estimate),
        ("initial_reference_state", initial_reference_state),
    ):
        if value.shape != (4,):
            raise ValueError(f"{name} must have shape (4,).")

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if max_acceleration <= 0:
        raise ValueError("max_acceleration must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    ekf = ExtendedKalmanFilter(
        initial_state=initial_estimate,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mu=mu,
    )

    true_state = initial_true_state.astype(float).copy()
    reference_state = initial_reference_state.astype(float).copy()

    times = np.arange(num_steps + 1) * dt
    true_history = np.zeros((num_steps + 1, 4))
    estimated_history = np.zeros((num_steps + 1, 4))
    reference_history = np.zeros((num_steps + 1, 4))
    covariance_history = np.zeros((num_steps + 1, 4, 4))
    control_history = np.zeros((num_steps, 2))

    true_history[0] = true_state
    reference_history[0] = reference_state

    for step in range(num_steps + 1):
        measurement = simulate_measurement(
            true_state,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std,
            rng=rng,
        )

        ekf.update(measurement)

        estimated_history[step] = ekf.state
        covariance_history[step] = ekf.covariance

        if step == num_steps:
            break

        control = calculate_control_acceleration(
            ekf.state,
            reference_state,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
        )

        # Represent finite control authority.
        magnitude = np.linalg.norm(control)
        if magnitude > max_acceleration:
            control = control * (max_acceleration / magnitude)

        control_history[step] = control

        true_state = rk4_step(
            true_state, dt, mu,
            control_acceleration=control,
        )

        reference_state = rk4_step(
            reference_state, dt, mu
        )

        ekf.predict(
            dt,
            control_acceleration=control,
        )

        true_history[step + 1] = true_state
        reference_history[step + 1] = reference_state

    return (
        times,
        true_history,
        estimated_history,
        reference_history,
        covariance_history,
        control_history,
    )