import numpy as np
import pytest

from spacecraft_sim.ekf import ExtendedKalmanFilter


def create_test_filter():
    initial_state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    initial_covariance = np.diag([
        100.0**2,
        100.0**2,
        10.0**2,
        10.0**2,
    ])

    process_noise = np.diag([
        1.0,
        1.0,
        0.1,
        0.1,
    ])

    measurement_noise = np.diag([
        10.0**2,
        10.0**2,
        0.1**2,
        0.1**2,
    ])

    return ExtendedKalmanFilter(
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise,
    )


def test_ekf_initialises():
    ekf = create_test_filter()

    assert ekf.state.shape == (4,)
    assert ekf.covariance.shape == (4, 4)


def test_ekf_prediction_changes_state():
    ekf = create_test_filter()

    initial_state = ekf.state.copy()

    predicted_state = ekf.predict(dt=1.0)

    assert predicted_state.shape == (4,)
    assert not np.array_equal(predicted_state, initial_state)


def test_ekf_prediction_updates_covariance():
    ekf = create_test_filter()

    initial_covariance = ekf.covariance.copy()

    ekf.predict(dt=1.0)

    assert not np.array_equal(
        ekf.covariance,
        initial_covariance,
    )


def test_ekf_update_moves_estimate_towards_measurement():
    ekf = create_test_filter()

    measurement = np.array([
        7_000_100.0,
        50.0,
        5.0,
        7_505.0,
    ])

    predicted_state = ekf.state.copy()

    updated_state = ekf.update(measurement)

    # The updated estimate should lie between the prior estimate
    # and the measurement for each component.
    assert np.all(
        np.abs(updated_state - measurement)
        < np.abs(predicted_state - measurement)
    )


def test_ekf_rejects_invalid_measurement_shape():
    ekf = create_test_filter()

    with pytest.raises(ValueError):
        ekf.update(np.array([1.0, 2.0, 3.0]))


def test_ekf_rejects_non_positive_timestep():
    ekf = create_test_filter()

    with pytest.raises(ValueError):
        ekf.predict(dt=0.0)