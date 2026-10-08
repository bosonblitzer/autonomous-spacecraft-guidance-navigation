import numpy as np

from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.ekf import ExtendedKalmanFilter
from spacecraft_sim.navigation import simulate_measurement
from spacecraft_sim.propagation import rk4_step


def run_navigation_simulation(
    initial_true_state: np.ndarray,
    initial_estimate: np.ndarray,
    initial_covariance: np.ndarray,
    process_noise: np.ndarray,
    measurement_noise: np.ndarray,
    position_noise_std: float,
    velocity_noise_std: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator | None = None,
    mu: float = EARTH_MU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a closed-loop spacecraft navigation simulation.

    The true spacecraft state is propagated using fourth-order
    Runge-Kutta integration. Noisy measurements are then generated
    from the true state and used by an Extended Kalman Filter.
    """

    if initial_true_state.shape != (4,):
        raise ValueError("initial_true_state must have shape (4,).")

    if initial_estimate.shape != (4,):
        raise ValueError("initial_estimate must have shape (4,).")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

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

    times = np.arange(num_steps + 1) * dt

    true_history = np.zeros((num_steps + 1, 4))
    estimated_history = np.zeros((num_steps + 1, 4))
    covariance_history = np.zeros((num_steps + 1, 4, 4))

    true_history[0] = true_state
    estimated_history[0] = ekf.state
    covariance_history[0] = ekf.covariance

    for step in range(1, num_steps + 1):

        # Propagate the true spacecraft state using RK4.
        true_state = rk4_step(
            true_state,
            dt,
            mu,
        )

        # Generate a noisy measurement of the true state.
        measurement = simulate_measurement(
            true_state,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std,
            rng=rng,
        )

        # Predict the spacecraft state using the EKF.
        ekf.predict(dt)

        # Correct the prediction using the measurement.
        ekf.update(measurement)

        # Store the updated covariance.
        covariance_history[step] = ekf.covariance

        true_history[step] = true_state
        estimated_history[step] = ekf.state

    return (
        times,
        true_history,
        estimated_history,
        covariance_history,
    )