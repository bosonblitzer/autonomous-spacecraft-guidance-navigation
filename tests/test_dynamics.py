import numpy as np
import pytest

from spacecraft_sim.dynamics import (
    EARTH_MU,
    gravitational_acceleration,
    state_derivative,
    dynamics_jacobian,
)


def test_gravitational_acceleration_direction():
    position = np.array([7_000_000.0, 0.0])

    acceleration = gravitational_acceleration(position)

    assert acceleration[0] < 0
    assert np.isclose(acceleration[1], 0.0)


def test_gravitational_acceleration_magnitude():
    position = np.array([7_000_000.0, 0.0])

    acceleration = gravitational_acceleration(position)

    expected = EARTH_MU / (7_000_000.0 ** 2)

    assert np.isclose(
        np.linalg.norm(acceleration),
        expected,
        rtol=1e-10,
    )


def test_state_derivative():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    derivative = state_derivative(0.0, state)

    assert np.isclose(derivative[0], 0.0)
    assert np.isclose(derivative[1], 7_500.0)
    assert derivative[2] < 0
    assert np.isclose(derivative[3], 0.0)


def test_zero_position_raises_error():
    with pytest.raises(ValueError):
        gravitational_acceleration(np.array([0.0, 0.0]))


def test_dynamics_jacobian_shape():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    jacobian = dynamics_jacobian(state)

    assert jacobian.shape == (4, 4)


def test_dynamics_jacobian_velocity_terms():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    jacobian = dynamics_jacobian(state)

    assert np.isclose(jacobian[0, 2], 1.0)
    assert np.isclose(jacobian[1, 3], 1.0)


def test_dynamics_jacobian_zero_position_raises_error():
    state = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    with pytest.raises(ValueError):
        dynamics_jacobian(state)
