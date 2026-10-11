
import csv
from pathlib import Path

import numpy as np

from spacecraft_sim.closed_loop import run_closed_loop_simulation
from spacecraft_sim.dynamics import EARTH_MU


def run_case(position_noise_std, velocity_noise_std, seed):
    dt = 1.0
    num_steps = 1800
    max_acceleration = 0.05

    position_gain = 2e-6
    velocity_gain = 2e-3

    radius = 6.771e6
    speed = np.sqrt(EARTH_MU / radius)

    reference = np.array([radius, 0.0, 0.0, speed])
    initial_true = reference + np.array([100.0, -100.0, 1.0, -1.0])
    initial_estimate = initial_true + np.array([20.0, -20.0, 0.2, -0.2])

    initial_covariance = np.diag([10000.0, 10000.0, 1.0, 1.0])
    process_noise = np.diag([0.1, 0.1, 1e-4, 1e-4])

    measurement_noise = np.diag([
        max(position_noise_std, 1e-6) ** 2,
        max(position_noise_std, 1e-6) ** 2,
        max(velocity_noise_std, 1e-6) ** 2,
        max(velocity_noise_std, 1e-6) ** 2,
    ])

    results = run_closed_loop_simulation(
        initial_true_state=initial_true,
        initial_estimate=initial_estimate,
        reference_state=reference,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        dt=dt,
        num_steps=num_steps,
        position_noise_std=position_noise_std,
        velocity_noise_std=velocity_noise_std,
        position_gain=position_gain,
        velocity_gain=velocity_gain,
        max_acceleration=max_acceleration,
        rng=np.random.default_rng(seed),
    )

    times, truth, estimate, reference_history, _, control = results

    position_error = np.linalg.norm(
        truth[:, :2] - reference_history[:, :2], axis=1
    )
    velocity_error = np.linalg.norm(
        truth[:, 2:] - reference_history[:, 2:], axis=1
    )
    estimation_error = np.linalg.norm(
        estimate[:, :2] - truth[:, :2], axis=1
    )
    velocity_estimation_error = np.linalg.norm(
        estimate[:, 2:] - truth[:, 2:], axis=1
    )
    control_magnitude = np.linalg.norm(control, axis=1)

    return {
        "position_noise_std_m": position_noise_std,
        "velocity_noise_std_m_s": velocity_noise_std,
        "seed": seed,
        "position_tracking_rmse_m": float(
            np.sqrt(np.mean(position_error**2))
        ),
        "final_position_tracking_error_m": float(position_error[-1]),
        "position_estimation_rmse_m": float(
            np.sqrt(np.mean(estimation_error**2))
        ),
        "velocity_estimation_rmse_m_s": float(
            np.sqrt(np.mean(velocity_estimation_error**2))
        ),
        "velocity_tracking_rmse_m_s": float(
            np.sqrt(np.mean(velocity_error**2))
        ),
        "rms_acceleration_m_s2": float(
            np.sqrt(np.mean(control_magnitude**2))
        ),
        "peak_acceleration_m_s2": float(np.max(control_magnitude)),
        "all_states_finite": bool(np.all(np.isfinite(truth))),
        "all_estimates_finite": bool(np.all(np.isfinite(estimate))),
        "actuator_limit_respected": bool(
            np.all(control_magnitude <= max_acceleration + 1e-12)
        ),
    }


def main():
    noise_cases = [
        (0.0, 0.0),
        (10.0, 0.01),
        (100.0, 0.1),
    ]
    seeds = [7, 42, 123]
    results = []

    for position_noise, velocity_noise in noise_cases:
        for seed in seeds:
            result = run_case(position_noise, velocity_noise, seed)
            results.append(result)

            print(
                f"Noise=({position_noise:g} m, {velocity_noise:g} m/s), "
                f"seed={seed} | "
                f"tracking RMSE={result['position_tracking_rmse_m']:.2f} m | "
                f"position estimation RMSE="
                f"{result['position_estimation_rmse_m']:.2f} m | "
                f"velocity estimation RMSE="
                f"{result['velocity_estimation_rmse_m_s']:.6f} m/s | "
                f"peak acceleration="
                f"{result['peak_acceleration_m_s2']:.6f} m/s²"
            )

    output_path = Path("robustness_study.csv")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    all_finite = all(
        row["all_states_finite"] and row["all_estimates_finite"]
        for row in results
    )
    all_within_limits = all(
        row["actuator_limit_respected"] for row in results
    )

    print("\n=== Robustness Summary ===")
    print(f"Runs completed: {len(results)}")
    print(f"All state histories finite: {all_finite}")
    print(f"All actuator limits respected: {all_within_limits}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
