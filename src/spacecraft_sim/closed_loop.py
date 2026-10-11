import numpy as np
from spacecraft_sim.control import calculate_control_acceleration
from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.ekf import ExtendedKalmanFilter
from spacecraft_sim.navigation import simulate_measurement
from spacecraft_sim.propagation import rk4_step


def run_closed_loop_simulation(
    initial_true_state: np.ndarray,
    initial_estimate: np.ndarray,
    reference_state: np.ndarray,
    initial_covariance: np.ndarray,
    process_noise: np.ndarray,
    measurement_noise: np.ndarray,
    dt: float,
    num_steps: int,
    position_noise_std: float,
    velocity_noise_std: float,
    position_gain: float,
    velocity_gain: float,
    max_acceleration: float,
    rng: np.random.Generator | None = None,
    mu: float = EARTH_MU,
):
    """Simulate spacecraft dynamics, EKF estimation and feedback control."""

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if position_noise_std < 0 or velocity_noise_std < 0:
        raise ValueError("Measurement noise must be non-negative.")
    if position_gain < 0 or velocity_gain < 0:
        raise ValueError("Controller gains must be non-negative.")
    if max_acceleration <= 0:
        raise ValueError("max_acceleration must be positive.")

    for name, state in (
        ("initial_true_state", initial_true_state),
        ("initial_estimate", initial_estimate),
        ("reference_state", reference_state),
    ):
        if state.shape != (4,):
            raise ValueError(f"{name} must have shape (4,).")

    if rng is None:
        rng = np.random.default_rng()

    true_state = initial_true_state.astype(float).copy()
    reference = reference_state.astype(float).copy()

    ekf = ExtendedKalmanFilter(
        initial_state=initial_estimate,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mu=mu,
    )

    times = np.arange(num_steps + 1) * dt
    true_history = np.zeros((num_steps + 1, 4))
    estimated_history = np.zeros((num_steps + 1, 4))
    reference_history = np.zeros((num_steps + 1, 4))
    covariance_history = np.zeros((num_steps + 1, 4, 4))
    control_history = np.zeros((num_steps, 2))

    true_history[0] = true_state
    estimated_history[0] = ekf.state
    reference_history[0] = reference
    covariance_history[0] = ekf.covariance

    for step in range(num_steps):
        measurement = simulate_measurement(
            true_state,
            position_noise_std,
            velocity_noise_std,
            rng=rng,
        )

        # Correct the state estimate using the latest measurement.
        ekf.update(measurement)

        # Calculate feedback control from the estimated state.
        control = calculate_control_acceleration(
            ekf.state,
            reference,
            position_gain,
            velocity_gain,
        )

        # Apply actuator acceleration saturation.
        control_magnitude = np.linalg.norm(control)
        if control_magnitude > max_acceleration:
            control = control * (max_acceleration / control_magnitude)

        # Propagate the real spacecraft and estimated state consistently.
        true_state = rk4_step(
            true_state,
            dt,
            mu,
            control_acceleration=control,
        )
        ekf.predict(dt, control_acceleration=control)

        # Propagate the unforced reference orbit.
        reference = rk4_step(reference, dt, mu)

        true_history[step + 1] = true_state
        estimated_history[step + 1] = ekf.state
        reference_history[step + 1] = reference
        covariance_history[step + 1] = ekf.covariance
        control_history[step] = control

    return (
        times,
        true_history,
        estimated_history,
        reference_history,
        covariance_history,
        control_history,
    )
