import csv
from pathlib import Path

import numpy as np

from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.propagation import rk4_step


def specific_energy(state):
    """Calculate specific orbital energy in J/kg."""
    position = state[:2]
    velocity = state[2:]

    radius = np.linalg.norm(position)
    speed_squared = np.dot(velocity, velocity)

    return 0.5 * speed_squared - EARTH_MU / radius


def specific_angular_momentum(state):
    """Calculate signed specific angular momentum in m^2/s."""
    x, y, vx, vy = state
    return x * vy - y * vx


def run_case(dt, initial_state, duration):
    """Propagate an unforced orbit and measure conservation errors."""
    num_steps = round(duration / dt)
    actual_duration = num_steps * dt

    state = initial_state.copy()

    initial_energy = specific_energy(state)
    initial_momentum = specific_angular_momentum(state)

    energy_history = [initial_energy]
    momentum_history = [initial_momentum]
    radius_history = [np.linalg.norm(state[:2])]

    for _ in range(num_steps):
        # No control acceleration: unforced two-body propagation.
        state = rk4_step(state, dt)

        energy_history.append(specific_energy(state))
        momentum_history.append(specific_angular_momentum(state))
        radius_history.append(np.linalg.norm(state[:2]))

    energy_history = np.asarray(energy_history)
    momentum_history = np.asarray(momentum_history)

    relative_energy_error = (
        np.abs(energy_history - initial_energy) / abs(initial_energy)
    )
    relative_momentum_error = (
        np.abs(momentum_history - initial_momentum)
        / abs(initial_momentum)
    )

    return {
        "timestep_s": dt,
        "steps": num_steps,
        "duration_s": actual_duration,
        "initial_energy_J_kg": initial_energy,
        "final_energy_J_kg": energy_history[-1],
        "max_relative_energy_error": float(
            np.max(relative_energy_error)
        ),
        "final_relative_energy_error": float(
            relative_energy_error[-1]
        ),
        "max_relative_momentum_error": float(
            np.max(relative_momentum_error)
        ),
        "final_relative_momentum_error": float(
            relative_momentum_error[-1]
        ),
        "minimum_radius_m": float(np.min(radius_history)),
        "maximum_radius_m": float(np.max(radius_history)),
        "all_values_finite": bool(
            np.all(np.isfinite(energy_history))
            and np.all(np.isfinite(momentum_history))
            and np.all(np.isfinite(radius_history))
        ),
    }


def main():
    # Ideal circular orbit under Earth's two-body gravity.
    orbital_radius = 6.771e6
    circular_speed = np.sqrt(EARTH_MU / orbital_radius)

    initial_state = np.array([
        orbital_radius,
        0.0,
        0.0,
        circular_speed,
    ])

    orbital_period = (
        2.0 * np.pi
        * np.sqrt(orbital_radius**3 / EARTH_MU)
    )
    duration = 3.0 * orbital_period

    timesteps = [10.0, 5.0, 2.5]
    results = []

    print("=== Orbital Conservation Study ===")
    print(f"Initial orbital radius: {orbital_radius:.1f} m")
    print(f"Circular orbital speed: {circular_speed:.3f} m/s")
    print(f"Theoretical orbital period: {orbital_period:.3f} s")
    print(f"Target duration: {duration:.3f} s")
    print()

    for dt in timesteps:
        result = run_case(dt, initial_state, duration)
        results.append(result)

        print(f"Timestep: {dt:.2f} s")
        print(f"  Integration steps: {result['steps']}")
        print(
            "  Maximum relative energy error: "
            f"{result['max_relative_energy_error']:.6e}"
        )
        print(
            "  Maximum relative angular momentum error: "
            f"{result['max_relative_momentum_error']:.6e}"
        )
        print(
            "  Final relative energy error: "
            f"{result['final_relative_energy_error']:.6e}"
        )
        print(
            "  All values finite: "
            f"{result['all_values_finite']}"
        )
        print()

    output_path = Path("conservation_study.csv")

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()