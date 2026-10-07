
import numpy as np


def simulate_measurement(
    true_state: np.ndarray,
    position_noise_std: float,
    velocity_noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a simulated spacecraft measurement with Gaussian noise.

    Parameters
    ----------
    true_state : np.ndarray
        True spacecraft state [x, y, vx, vy].

    position_noise_std : float
        Standard deviation of position measurement noise in metres.

    velocity_noise_std : float
        Standard deviation of velocity measurement noise in m/s.

    rng : np.random.Generator, optional
        Random number generator used to generate measurement noise.
        If not provided, a default generator is created.

    Returns
    -------
    np.ndarray
        Simulated noisy measurement [x, y, vx, vy].
    """

    if true_state.shape != (4,):
        raise ValueError("true_state must contain [x, y, vx, vy].")

    if position_noise_std < 0:
        raise ValueError("position_noise_std must be non-negative.")

    if velocity_noise_std < 0:
        raise ValueError("velocity_noise_std must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    noise = np.array([
        rng.normal(0.0, position_noise_std),
        rng.normal(0.0, position_noise_std),
        rng.normal(0.0, velocity_noise_std),
        rng.normal(0.0, velocity_noise_std),
    ])

    return true_state + noise

