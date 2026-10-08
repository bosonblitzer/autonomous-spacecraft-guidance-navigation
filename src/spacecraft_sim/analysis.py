import numpy as np


def calculate_state_error(
    true_history: np.ndarray,
    estimated_history: np.ndarray,
) -> np.ndarray:
    """
    Calculate the state estimation error over a simulation.

    Parameters
    ----------
    true_history : np.ndarray
        True spacecraft states with shape (N, 4).

    estimated_history : np.ndarray
        Estimated spacecraft states with shape (N, 4).

    Returns
    -------
    np.ndarray
        State estimation errors with shape (N, 4).
    """

    if true_history.shape != estimated_history.shape:
        raise ValueError(
            "true_history and estimated_history must have "
            "the same shape."
        )

    if true_history.ndim != 2 or true_history.shape[1] != 4:
        raise ValueError(
            "Histories must have shape (N, 4)."
        )

    return estimated_history - true_history


def calculate_position_error(
    state_error: np.ndarray,
) -> np.ndarray:
    """
    Calculate the magnitude of position estimation error.

    Parameters
    ----------
    state_error : np.ndarray
        State errors with shape (N, 4).

    Returns
    -------
    np.ndarray
        Position error magnitude at each timestep.
    """

    if state_error.ndim != 2 or state_error.shape[1] != 4:
        raise ValueError(
            "state_error must have shape (N, 4)."
        )

    position_error = state_error[:, :2]

    return np.linalg.norm(position_error, axis=1)


def calculate_velocity_error(
    state_error: np.ndarray,
) -> np.ndarray:
    """
    Calculate the magnitude of velocity estimation error.

    Parameters
    ----------
    state_error : np.ndarray
        State errors with shape (N, 4).

    Returns
    -------
    np.ndarray
        Velocity error magnitude at each timestep.
    """

    if state_error.ndim != 2 or state_error.shape[1] != 4:
        raise ValueError(
            "state_error must have shape (N, 4)."
        )

    velocity_error = state_error[:, 2:]

    return np.linalg.norm(velocity_error, axis=1)


def calculate_rmse(
    errors: np.ndarray,
) -> float:
    """
    Calculate root mean square error.

    Parameters
    ----------
    errors : np.ndarray
        Array of errors.

    Returns
    -------
    float
        Root mean square error.
    """

    return float(np.sqrt(np.mean(errors**2)))
