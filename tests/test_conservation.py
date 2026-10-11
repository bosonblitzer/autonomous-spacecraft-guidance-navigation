
import numpy as np

from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.propagation import rk4_step


def specific_energy(state):
    """Specific orbital energy for a planar two-body orbit."""
    radius = np.linalg.norm(state[:2])
    speed_squared = np.dot(state[2:], state[2:])
    return 0.5 * speed_squared - EARTH_MU / radius


def specific_angular_momentum(state):
    """Signed specific angular momentum for a planar orbit."""
    x, y, vx, vy = state
    return x * vy - y * vx


def propagate_orbit(initial_state, dt, num_steps):
    """Propagate an unforced orbit and return all states."""
    states = [initial_state.copy()]
    state = initial_state.copy()

    for _ in range(num_steps):
        state = rk4_step(state, dt)
        states.append(state.copy())

    return np.asarray(states)


def circular_orbit_initial_state():
    """Return a circular-orbit state at the chosen radius."""
    radius = 6.771e6
    speed = np.sqrt(EARTH_MU / radius)

    return np.array([radius, 0.0, 0.0, speed])


def test_specific_orbital_energy_is_conserved():
    initial_state = circular_orbit_initial_state()

    radius = np.linalg.norm(initial_state[:2])
    orbital_period = 2.0 * np.pi * np.sqrt(
        radius**3 / EARTH_MU
    )

    dt = 5.0
    num_steps = int(round(3.0 * orbital_period / dt))

    states = propagate_orbit(initial_state, dt, num_steps)

    initial_energy = specific_energy(initial_state)
    energies = np.array([specific_energy(s) for s in states])

    relative_errors = (
        np.abs(energies - initial_energy) / abs(initial_energy)
    )

    assert np.all(np.isfinite(states))
    assert np.max(relative_errors) < 1e-9


def test_specific_angular_momentum_is_conserved():
    initial_state = circular_orbit_initial_state()

    radius = np.linalg.norm(initial_state[:2])
    orbital_period = 2.0 * np.pi * np.sqrt(
        radius**3 / EARTH_MU
    )

    dt = 5.0
    num_steps = int(round(3.0 * orbital_period / dt))

    states = propagate_orbit(initial_state, dt, num_steps)

    initial_momentum = specific_angular_momentum(initial_state)
    momenta = np.array([
        specific_angular_momentum(s) for s in states
    ])

    relative_errors = (
        np.abs(momenta - initial_momentum) / abs(initial_momentum)
    )

    assert np.all(np.isfinite(states))
    assert np.max(relative_errors) < 1e-9
