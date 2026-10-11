
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.propagation import rk4_step


def two_body_dynamics(time, state):
    """Independent two-body equations for the reference solution."""
    x, y, vx, vy = state
    radius = np.hypot(x, y)

    ax = -EARTH_MU * x / radius**3
    ay = -EARTH_MU * y / radius**3

    return [vx, vy, ax, ay]


def calculate_reference(initial_state, times):
    """Generate a high-accuracy reference trajectory."""
    solution = solve_ivp(
        two_body_dynamics,
        (float(times[0]), float(times[-1])),
        initial_state,
        t_eval=times,
        method="DOP853",
        rtol=1e-12,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError("Reference solution contains non-finite values.")

    return solution.y.T


def propagate_rk4(initial_state, dt, num_steps):
    """Propagate the orbit using the project's RK4 implementation."""
    states = np.empty((num_steps + 1, 4))
    states[0] = initial_state

    state = initial_state.copy()

    for index in range(num_steps):
        state = rk4_step(state, dt)
        states[index + 1] = state

    return states


def main():
    orbital_radius = 6.771e6
    circular_speed = np.sqrt(EARTH_MU / orbital_radius)

    initial_state = np.array([
        orbital_radius,
        0.0,
        0.0,
        circular_speed,
    ])

    orbital_period = 2.0 * np.pi * np.sqrt(
        orbital_radius**3 / EARTH_MU
    )
    duration = 3.0 * orbital_period
    timesteps = [20.0, 10.0, 5.0, 2.5]

    results = []

    print("=== RK4 Timestep Convergence Study ===")
    print(f"Orbital period: {orbital_period:.3f} s")
    print(f"Simulation duration: {duration:.3f} s")
    print()

    for dt in timesteps:
        num_steps = int(round(duration / dt))
        times = np.linspace(0.0, duration, num_steps + 1)
        actual_dt = duration / num_steps

        reference = calculate_reference(initial_state, times)
        numerical = propagate_rk4(
            initial_state, actual_dt, num_steps
        )

        position_errors = np.linalg.norm(
            numerical[:, :2] - reference[:, :2],
            axis=1,
        )
        velocity_errors = np.linalg.norm(
            numerical[:, 2:] - reference[:, 2:],
            axis=1,
        )

        result = {
            "timestep_s": dt,
            "actual_timestep_s": actual_dt,
            "steps": num_steps,
            "duration_s": duration,
            "position_rmse_m": float(
                np.sqrt(np.mean(position_errors**2))
            ),
            "maximum_position_error_m": float(
                np.max(position_errors)
            ),
            "final_position_error_m": float(
                position_errors[-1]
            ),
            "maximum_velocity_error_m_s": float(
                np.max(velocity_errors)
            ),
            "all_values_finite": bool(
                np.all(np.isfinite(numerical))
            ),
        }
        results.append(result)

        print(f"Nominal timestep: {dt:.2f} s")
        print(f"  Actual timestep: {actual_dt:.8f} s")
        print(f"  Steps: {num_steps}")
        print(f"  Position RMSE: {result['position_rmse_m']:.6e} m")
        print(
            "  Maximum position error: "
            f"{result['maximum_position_error_m']:.6e} m"
        )
        print(
            "  Final position error: "
            f"{result['final_position_error_m']:.6e} m"
        )
        print(
            "  Maximum velocity error: "
            f"{result['maximum_velocity_error_m_s']:.6e} m/s"
        )
        print(f"  All values finite: {result['all_values_finite']}")
        print()

    output_path = Path("timestep_convergence.csv")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
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
