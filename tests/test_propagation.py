import numpy as np
import pytest

from spacecraft_sim.propagation import rk4_step


def test_rk4_step_preserves_state_shape():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    propagated = rk4_step(state, dt=1.0)

    assert propagated.shape == (4,)


def test_rk4_step_changes_state():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    propagated = rk4_step(state, dt=1.0)

    assert not np.array_equal(propagated, state)


def test_rk4_step_rejects_invalid_state():
    state = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        rk4_step(state, dt=1.0)


def test_rk4_step_rejects_non_positive_timestep():
    state = np.array([
        7_000_000.0,
        0.0,
        0.0,
        7_500.0,
    ])

    with pytest.raises(ValueError):
        rk4_step(state, dt=0.0)