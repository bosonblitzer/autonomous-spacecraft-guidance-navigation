import numpy as np


# Earth's standard gravitational parameter
# Units: m^3 / s^2
EARTH_MU = 3.986004418e14


def gravitational_acceleration(
    position: np.ndarray,
    mu: float = EARTH_MU,
) -> np.ndarray:
    '''
    Calculate gravitational acceleration acting on the spacecraft.

    Parameters
    ----------
    position : np.ndarray
        Spacecraft position [x, y] in metres.

    mu : float
        Gravitational parameter of the central body in m^3/s^2.

    Returns
    -------
    np.ndarray
        Gravitational acceleration [ax, ay] in m/s^2.
    '''

    radius = np.linalg.norm(position)

    if radius == 0:
        raise ValueError("Spacecraft position cannot be at the centre of Earth.")

    acceleration = -mu * position / radius**3

    return acceleration


def state_derivative(
    time: float,
    state: np.ndarray,
    mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Calculate the time derivative of the spacecraft state.

    Parameters
    ----------
    time : float
        Current simulation time in seconds.

    state : np.ndarray
        State vector [x, y, vx, vy].

    mu : float
        Gravitational parameter in m^3/s^2.

    Returns
    -------
    np.ndarray
        State derivative [vx, vy, ax, ay].
    """

    position = state[:2]
    velocity = state[2:]

    acceleration = gravitational_acceleration(position, mu)

    return np.concatenate((velocity, acceleration))
def dynamics_jacobian(
    state: np.ndarray,
    mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Calculate the Jacobian matrix of the spacecraft dynamics.

    Parameters
    ----------
    state : np.ndarray
        State vector [x, y, vx, vy].

    mu : float
        Gravitational parameter in m^3/s^2.

    Returns
    -------
    np.ndarray
        Continuous-time dynamics Jacobian matrix.
    """

    position = state[:2]
    x, y = position

    radius = np.linalg.norm(position)

    if radius == 0:
        raise ValueError("Spacecraft position cannot be at the central body.")

    radius_squared = radius**2
    radius_cubed = radius**3
    radius_fifth = radius**5

    jacobian = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [
            mu * (3.0 * x**2 / radius_fifth - 1.0 / radius_cubed),
            3.0 * mu * x * y / radius_fifth,
            0.0,
            0.0,
        ],
        [
            3.0 * mu * x * y / radius_fifth,
            mu * (3.0 * y**2 / radius_fifth - 1.0 / radius_cubed),
            0.0,
            0.0,
        ],
    ])
    return jacobian
