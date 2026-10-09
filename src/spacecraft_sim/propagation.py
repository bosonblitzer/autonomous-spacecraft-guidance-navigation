import numpy as np

from spacecraft_sim.dynamics import EARTH_MU, state_derivative


def rk4_step(
    state: np.ndarray,
    dt: float,
    mu: float = EARTH_MU,
    control_acceleration: np.ndarray | None = None,
) -> np.ndarray:
    """Advance the spacecraft state using fourth-order Runge-Kutta."""
    if state.shape != (4,):
        raise ValueError("state must have shape (4,).")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    control = None
    if control_acceleration is not None:
        control = np.asarray(control_acceleration, dtype=float)
        if control.shape != (2,):
            raise ValueError(
                "control_acceleration must have shape (2,)."
            )

    k1 = state_derivative(
        0.0, state, mu, control_acceleration=control
    )
    k2 = state_derivative(
        dt / 2, state + dt * k1 / 2,
        mu, control_acceleration=control
    )
    k3 = state_derivative(
        dt / 2, state + dt * k2 / 2,
        mu, control_acceleration=control
    )
    k4 = state_derivative(
        dt, state + dt * k3,
        mu, control_acceleration=control
    )

    return state + (dt / 6) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )