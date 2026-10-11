
import csv
from pathlib import Path

import numpy as np

from spacecraft_sim.closed_loop import run_closed_loop_simulation
from spacecraft_sim.dynamics import EARTH_MU


def run_case(position_gain, velocity_gain, seed=42):
    dt = 1.0
    num_steps = 1800
    max_acceleration = 0.05
    position_noise_std = 10.0
    velocity_noise_std = 0.01

    radius = 6.771e6
    speed = np.sqrt(EARTH_MU / radius)

    reference = np.array([radius, 0.0, 0.0, speed])
    true_initial = reference + np.array([100.0, -100.0, 1.0, -1.0])
    estimate_initial = true_initial + np.array([20.0, -20.0, 0.2, -0.2])

    covariance = np.diag([10000.0, 10000.0, 1.0, 1.0])
    process_noise = np.diag([0.1, 0.1, 1e-4, 1e-4])
    measurement_noise = np.diag(
        [
            position_noise_std**2,
            position_noise_std**2,
            velocity_noise_std**2,
            velocity_noise_std**2,
        ]
    )

    results = run_closed_loop_simulation(
        initial_true_state=true_initial,
        initial_estimate=estimate_initial,
        reference_state=reference,
        initial_covariance=covariance,
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
    control_norm = np.linalg.norm(control, axis=1)

    return {
        "position_gain": position_gain,
        "velocity_gain": velocity_gain,
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "final_position_error_m": float(position_error[-1]),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(velocity_error**2))),
        "rms_acceleration_m_s2": float(np.sqrt(np.mean(control_norm**2))),
        "peak_acceleration_m_s2": float(np.max(control_norm)),
        "max_allowed_acceleration_m_s2": max_acceleration,
    }


def main():
    gain_pairs = [
        (5e-7, 5e-4),
        (1e-6, 1e-3),
        (2e-6, 2e-3),
        (5e-6, 5e-3),
    ]

    results = [
        run_case(position_gain, velocity_gain)
        for position_gain, velocity_gain in gain_pairs
    ]

    output_path = Path("gain_trade_study.csv")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Controller Gain Trade Study ===")
    for result in results:
        print(
            f"Kp={result['position_gain']:.1e}, "
            f"Kd={result['velocity_gain']:.1e} | "
            f"Position RMSE={result['position_rmse_m']:.3f} m | "
            f"Final error={result['final_position_error_m']:.3f} m | "
            f"RMS acceleration={result['rms_acceleration_m_s2']:.6f} m/s²"
        )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
