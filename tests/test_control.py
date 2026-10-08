import numpy as np
import pytest

from spacecraft_sim.control import calculate_control_acceleration


def test_control_acceleration_has_correct_shape():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    target_state = np.array([
        7_001_000.0,
        500.0,
        10.0,
        7_490.0,
    ])

    acceleration = calculate_control_acceleration(
        state,
        target_state,
        position_gain=0.001,
        velocity_gain=0.1,
    )

    assert acceleration.shape == (2,)


def test_control_responds_to_position_error():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    target_state = np.array([
        7_001_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    acceleration = calculate_control_acceleration(
        state,
        target_state,
        position_gain=0.001,
        velocity_gain=0.0,
    )

    expected = np.array([
        1.0,
        0.0,
    ])

    assert np.allclose(acceleration, expected)


def test_control_responds_to_velocity_error():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    target_state = np.array([
        7_000_000.0,
        0.0,
        10.0,
        7_500.0,
    ])

    acceleration = calculate_control_acceleration(
        state,
        target_state,
        position_gain=0.0,
        velocity_gain=0.1,
    )

    expected = np.array([
        1.0,
        0.0,
    ])

    assert np.allclose(acceleration, expected)


def test_zero_error_produces_zero_control():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    acceleration = calculate_control_acceleration(
        state,
        state,
        position_gain=0.001,
        velocity_gain=0.1,
    )

    assert np.allclose(
        acceleration,
        np.zeros(2),
    )


def test_control_rejects_invalid_state_shape():
    state = np.zeros(3)
    target_state = np.zeros(4)

    with pytest.raises(ValueError):
        calculate_control_acceleration(
            state,
            target_state,
            position_gain=0.001,
            velocity_gain=0.1,
        )


def test_control_rejects_invalid_target_shape():
    state = np.zeros(4)
    target_state = np.zeros(3)

    with pytest.raises(ValueError):
        calculate_control_acceleration(
            state,
            target_state,
            position_gain=0.001,
            velocity_gain=0.1,
        )


def test_control_rejects_negative_position_gain():
    state = np.zeros(4)
    target_state = np.zeros(4)

    with pytest.raises(ValueError):
        calculate_control_acceleration(
            state,
            target_state,
            position_gain=-0.001,
            velocity_gain=0.1,
        )


def test_control_rejects_negative_velocity_gain():
    state = np.zeros(4)
    target_state = np.zeros(4)

    with pytest.raises(ValueError):
        calculate_control_acceleration(
            state,
            target_state,
            position_gain=0.001,
            velocity_gain=-0.1,
        )
