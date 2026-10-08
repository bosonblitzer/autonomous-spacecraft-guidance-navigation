import numpy as np

def calculate_control_acceleration(
    state: np.ndarray,
    target_state: np.ndarray,
    position_gain: float,
    velocity_gain: float,
) -> np.ndarray:
    """
    Calculate a corrective acceleration using PD feedback control.

    State:
        [x, y, vx, vy]

    The controller uses position and velocity errors to calculate
    a two-dimensional control acceleration.

    Parameters
    ----------
    state : np.ndarray
        Current spacecraft state [x, y, vx, vy].
    target_state : np.ndarray
        Desired spacecraft state [x, y, vx, vy].
    position_gain : float
        Proportional gain applied to position error.
    velocity_gain : float
        Derivative gain applied to velocity error.

    Returns
    -------
    np.ndarray
        Control acceleration [ax, ay].
    """

    if state.shape != (4,):
        raise ValueError("state must have shape (4,).")

    if target_state.shape != (4,):
        raise ValueError("target_state must have shape (4,).")

    if position_gain < 0:
        raise ValueError("position_gain must be non-negative.")

    if velocity_gain < 0:
        raise ValueError("velocity_gain must be non-negative.")

    position_error = (
        target_state[:2] - state[:2]
    )

    velocity_error = (
        target_state[2:] - state[2:]
    )

    control_acceleration = (
        position_gain * position_error
        + velocity_gain * velocity_error
    )

    return control_acceleration
