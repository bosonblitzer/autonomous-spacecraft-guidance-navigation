import numpy as np

from spacecraft_sim.dynamics import EARTH_MU, state_derivative


def rk4_step(
    state: np.ndarray,
    dt: float,
    mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Propagate a spacecraft state forward by one timestep
    using fourth-order Runge-Kutta integration.

    Parameters
    ----------
    state : np.ndarray
        Current state [x, y, vx, vy].

    dt : float
        Timestep in seconds.

    mu : float
        Gravitational parameter in m^3/s^2.

    Returns
    -------
    np.ndarray
        State after one timestep.
    """

    if state.shape != (4,):
        raise ValueError("state must have shape (4,).")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    k1 = state_derivative(0.0, state, mu)

    k2 = state_derivative(
        0.0,
        state + 0.5 * dt * k1,
        mu,
    )

    k3 = state_derivative(
        0.0,
        state + 0.5 * dt * k2,
        mu,
    )

    k4 = state_derivative(
        0.0,
        state + dt * k3,
        mu,
    )

    return state + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )