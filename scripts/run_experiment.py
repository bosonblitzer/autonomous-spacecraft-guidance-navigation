import numpy as np

from spacecraft_sim.analysis import (
    calculate_position_error,
    calculate_rmse,
    calculate_state_error,
    calculate_velocity_error,
)
from spacecraft_sim.simulation import run_navigation_simulation


dt = 10.0
num_steps = 1000

position_noise_std = 100.0
velocity_noise_std = 0.1

initial_true_state = np.array([
    7_000_000.0,
    0.0,
    0.0,
    7_500.0,
])

initial_estimate = np.array([
    7_000_500.0,
    -500.0,
    5.0,
    7_495.0,
])

initial_covariance = np.diag([
    500.0**2,
    500.0**2,
    5.0**2,
    5.0**2,
])

process_noise = np.diag([
    1.0,
    1.0,
    0.01,
    0.01,
])

measurement_noise = np.diag([
    position_noise_std**2,
    position_noise_std**2,
    velocity_noise_std**2,
    velocity_noise_std**2,
])

rng = np.random.default_rng(42)


(
    times,
    true_history,
    estimated_history,
    covariance_history,
) = run_navigation_simulation(
    initial_true_state=initial_true_state,
    initial_estimate=initial_estimate,
    initial_covariance=initial_covariance,
    process_noise=process_noise,
    measurement_noise=measurement_noise,
    position_noise_std=position_noise_std,
    velocity_noise_std=velocity_noise_std,
    dt=dt,
    num_steps=num_steps,
    rng=rng,
)


# ------------------------------------------------------------
# Estimation error analysis
# ------------------------------------------------------------

state_error = calculate_state_error(
    true_history,
    estimated_history,
)

position_error = calculate_position_error(
    state_error,
)

velocity_error = calculate_velocity_error(
    state_error,
)

position_rmse = calculate_rmse(position_error)
velocity_rmse = calculate_rmse(velocity_error)


# ------------------------------------------------------------
# EKF uncertainty analysis
# ------------------------------------------------------------

position_uncertainty = np.sqrt(
    covariance_history[:, 0, 0]
    + covariance_history[:, 1, 1]
)

velocity_uncertainty = np.sqrt(
    covariance_history[:, 2, 2]
    + covariance_history[:, 3, 3]
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n=== Spacecraft Navigation Experiment ===")
print(f"Simulation duration: {times[-1]:.1f} s")
print(f"Timestep:            {dt:.1f} s")
print(f"Number of steps:     {num_steps}")

print("\n--- Estimation Performance ---")
print(f"Initial position error: {position_error[0]:.3f} m")
print(f"Final position error:   {position_error[-1]:.3f} m")
print(f"Position RMSE:          {position_rmse:.3f} m")

print(f"Initial velocity error: {velocity_error[0]:.3f} m/s")
print(f"Final velocity error:   {velocity_error[-1]:.3f} m/s")
print(f"Velocity RMSE:          {velocity_rmse:.3f} m/s")

print("\n--- EKF Uncertainty ---")
print(
    f"Final position uncertainty: "
    f"{position_uncertainty[-1]:.3f} m"
)

print(
    f"Final velocity uncertainty: "
    f"{velocity_uncertainty[-1]:.3f} m/s"
)

print("\n--- Final True State ---")
print(true_history[-1])

print("\n--- Final Estimated State ---")
print(estimated_history[-1])