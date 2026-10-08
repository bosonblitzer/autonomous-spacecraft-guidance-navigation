import numpy as np
import matplotlib.pyplot as plt

from spacecraft_sim.analysis import (
    calculate_position_error,
    calculate_state_error,
    calculate_velocity_error,
)
from spacecraft_sim.simulation import run_navigation_simulation


# ------------------------------------------------------------
# Simulation configuration
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------

rng = np.random.default_rng(42)

times, true_history, estimated_history = run_navigation_simulation(
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
# Calculate errors
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


# ------------------------------------------------------------
# Plot spacecraft trajectory
# ------------------------------------------------------------

plt.figure()

plt.plot(
    true_history[:, 0] / 1000,
    true_history[:, 1] / 1000,
    label="True trajectory",
)

plt.plot(
    estimated_history[:, 0] / 1000,
    estimated_history[:, 1] / 1000,
    "--",
    label="EKF estimate",
)

plt.xlabel("x position (km)")
plt.ylabel("y position (km)")
plt.title("Spacecraft Trajectory")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()


# ------------------------------------------------------------
# Plot position estimation error
# ------------------------------------------------------------

plt.figure()

plt.plot(
    times / 60,
    position_error / 1000,
)

plt.xlabel("Time (minutes)")
plt.ylabel("Position error (km)")
plt.title("EKF Position Estimation Error")
plt.grid()

plt.show()


# ------------------------------------------------------------
# Plot velocity estimation error
# ------------------------------------------------------------

plt.figure()

plt.plot(
    times / 60,
    velocity_error,
)

plt.xlabel("Time (minutes)")
plt.ylabel("Velocity error (m/s)")
plt.title("EKF Velocity Estimation Error")
plt.grid()

plt.show()