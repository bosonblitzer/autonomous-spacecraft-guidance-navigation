
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_path = Path("closed_loop_results.csv")

    if not results_path.exists():
        raise FileNotFoundError(
            "closed_loop_results.csv not found. "
            "Run scripts/run_closed_loop_experiment.py first."
        )

    data = np.genfromtxt(
        results_path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )

    time_minutes = data["time_s"] / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(
        time_minutes,
        data["position_tracking_error_m"],
        label="Closed-loop tracking",
    )
    axes[0].plot(
        time_minutes,
        data["baseline_position_error_m"],
        linestyle="--",
        label="Uncontrolled baseline",
    )
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title("Spacecraft Position Tracking")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time_minutes,
        data["velocity_tracking_error_m_s"],
    )
    axes[1].set_ylabel("Velocity error (m/s)")
    axes[1].set_title("Velocity Tracking Error")
    axes[1].grid(True)

    axes[2].plot(
        time_minutes,
        data["control_acceleration_m_s2"],
    )
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_title("Control Effort")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("closed_loop_performance.png", dpi=200)
    plt.show()

    print("Saved plot to closed_loop_performance.png")


if __name__ == "__main__":
    main()
