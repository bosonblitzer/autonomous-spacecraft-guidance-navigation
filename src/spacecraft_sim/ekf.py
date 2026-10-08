import numpy as np

from spacecraft_sim.dynamics import (
    EARTH_MU,
    dynamics_jacobian,
)
from spacecraft_sim.propagation import rk4_step


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for spacecraft state estimation.

    State:
        [x, y, vx, vy]

    The nonlinear spacecraft dynamics are propagated using
    fourth-order Runge-Kutta integration.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        mu: float = EARTH_MU,
    ):
        if initial_state.shape != (4,):
            raise ValueError(
                "initial_state must have shape (4,)."
            )

        if initial_covariance.shape != (4, 4):
            raise ValueError(
                "initial_covariance must have shape (4, 4)."
            )

        if process_noise.shape != (4, 4):
            raise ValueError(
                "process_noise must have shape (4, 4)."
            )

        if measurement_noise.shape != (4, 4):
            raise ValueError(
                "measurement_noise must have shape (4, 4)."
            )

        self.state = initial_state.astype(float).copy()
        self.covariance = initial_covariance.astype(float).copy()
        self.process_noise = process_noise.astype(float).copy()
        self.measurement_noise = measurement_noise.astype(float).copy()
        self.mu = mu

    def predict(self, dt: float) -> np.ndarray:
        """
        Propagate the state and covariance forward by dt.
        """

        if dt <= 0:
            raise ValueError("dt must be positive.")

        # Propagate nonlinear state using RK4.
        self.state = rk4_step(
            self.state,
            dt,
            self.mu,
        )

        # Linearise the dynamics about the predicted state.
        jacobian = dynamics_jacobian(
            self.state,
            self.mu,
        )

        # First-order state-transition approximation.
        state_transition = (
            np.eye(4)
            + jacobian * dt
        )

        # Propagate covariance.
        self.covariance = (
            state_transition
            @ self.covariance
            @ state_transition.T
            + self.process_noise
        )

        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Correct the predicted state using a full-state measurement.
        """

        if measurement.shape != (4,):
            raise ValueError(
                "measurement must have shape (4,)."
            )

        # Full-state measurement model:
        # H = I
        measurement_matrix = np.eye(4)

        # Innovation.
        innovation = (
            measurement
            - measurement_matrix @ self.state
        )

        # Innovation covariance.
        innovation_covariance = (
            measurement_matrix
            @ self.covariance
            @ measurement_matrix.T
            + self.measurement_noise
        )

        # Kalman gain.
        kalman_gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        # State correction.
        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        # Covariance correction.
        identity = np.eye(4)

        self.covariance = (
            identity
            - kalman_gain @ measurement_matrix
        ) @ self.covariance

        return self.state