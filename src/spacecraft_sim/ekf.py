import numpy as np

from spacecraft_sim.dynamics import (
    EARTH_MU,
    dynamics_jacobian,
)
from spacecraft_sim.propagation import rk4_step


class ExtendedKalmanFilter:
    """Extended Kalman Filter for spacecraft state estimation."""

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        mu: float = EARTH_MU,
    ):
        if initial_state.shape != (4,):
            raise ValueError("initial_state must have shape (4,).")

        if initial_covariance.shape != (4, 4):
            raise ValueError(
                "initial_covariance must have shape (4, 4)."
            )

        if process_noise.shape != (4, 4):
            raise ValueError("process_noise must have shape (4, 4).")

        if measurement_noise.shape != (4, 4):
            raise ValueError(
                "measurement_noise must have shape (4, 4)."
            )

        self.state = initial_state.astype(float).copy()
        self.covariance = initial_covariance.astype(float).copy()
        self.process_noise = process_noise.astype(float).copy()
        self.measurement_noise = measurement_noise.astype(float).copy()
        self.mu = mu

    def predict(
        self,
        dt: float,
        control_acceleration: np.ndarray | None = None,
    ) -> np.ndarray:
        """Propagate state and covariance forward by dt."""
        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.state = rk4_step(
            self.state,
            dt,
            self.mu,
            control_acceleration=control_acceleration,
        )

        jacobian = dynamics_jacobian(self.state, self.mu)
        state_transition = np.eye(4) + jacobian * dt

        self.covariance = (
            state_transition
            @ self.covariance
            @ state_transition.T
            + self.process_noise
        )

        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Correct the predicted state using a full-state measurement."""
        if measurement.shape != (4,):
            raise ValueError("measurement must have shape (4,).")

        measurement_matrix = np.eye(4)

        innovation = measurement - measurement_matrix @ self.state

        innovation_covariance = (
            measurement_matrix
            @ self.covariance
            @ measurement_matrix.T
            + self.measurement_noise
        )

        kalman_gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = self.state + kalman_gain @ innovation

        identity = np.eye(4)
        residual_matrix = identity - kalman_gain @ measurement_matrix

        # Joseph-form covariance update improves numerical robustness.
        self.covariance = (
            residual_matrix
            @ self.covariance
            @ residual_matrix.T
            + kalman_gain
            @ self.measurement_noise
            @ kalman_gain.T
        )

        return self.state