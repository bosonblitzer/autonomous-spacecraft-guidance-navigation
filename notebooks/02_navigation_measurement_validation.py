import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from spacecraft_sim.dynamics import EARTH_MU, state_derivative
from spacecraft_sim.navigation import simulate_measurement


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

EARTH_RADIUS = 6_371_000.0
ALTITUDE = 700_000.0

INITIAL_RADIUS = EARTH_RADIUS + ALTITUDE
INITIAL_SPEED = np.sqrt(EARTH_MU / INITIAL_RADIUS)

POSITION_NOISE_STD = 100.0   # metres
VELOCITY_NOISE_STD = 0.1     # m/s

MEASUREMENT_INTERVAL = 10.0  # seconds

RANDOM_SEED = 42


# --------------------------------------------------
# Initial spacecraft state
# --------------------------------------------------

initial_state = np.array([
    INITIAL_RADIUS,
    0.0,
    0.0,
    INITIAL_SPEED,
])


# --------------------------------------------------
# Calculate orbital period
# --------------------------------------------------

orbital_period = (
    2 * np.pi
    * np.sqrt(INITIAL_RADIUS**3 / EARTH_MU)
)

print(f"Orbital period: {orbital_period / 60:.2f} minutes")


# --------------------------------------------------
# Propagate true spacecraft trajectory
# --------------------------------------------------

measurement_times = np.arange(
    0.0,
    orbital_period,
    MEASUREMENT_INTERVAL,
)

solution = solve_ivp(
    state_derivative,
    (0.0, orbital_period),
    initial_state,
    t_eval=measurement_times,
    rtol=1e-9,
    atol=1e-9,
    max_step=20.0,
)

true_states = solution.y.T


# --------------------------------------------------
# Generate simulated measurements
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

measured_states = np.array([
    simulate_measurement(
        true_state,
        position_noise_std=POSITION_NOISE_STD,
        velocity_noise_std=VELOCITY_NOISE_STD,
        rng=rng,
    )
    for true_state in true_states
])


# --------------------------------------------------
# Calculate measurement errors
# --------------------------------------------------

measurement_errors = measured_states - true_states

position_errors = measurement_errors[:, :2]
velocity_errors = measurement_errors[:, 2:]

position_error_magnitude = np.linalg.norm(
    position_errors,
    axis=1,
)

velocity_error_magnitude = np.linalg.norm(
    velocity_errors,
    axis=1,
)


# --------------------------------------------------
# Statistical validation
# --------------------------------------------------

print("\n--- Navigation Measurement Validation ---")

print("\nPosition measurement error:")
print(
    f"x mean = {position_errors[:, 0].mean():.3f} m"
)
print(
    f"x std  = {position_errors[:, 0].std():.3f} m"
)
print(
    f"y mean = {position_errors[:, 1].mean():.3f} m"
)
print(
    f"y std  = {position_errors[:, 1].std():.3f} m"
)

print("\nVelocity measurement error:")
print(
    f"vx mean = {velocity_errors[:, 0].mean():.5f} m/s"
)
print(
    f"vx std  = {velocity_errors[:, 0].std():.5f} m/s"
)
print(
    f"vy mean = {velocity_errors[:, 1].mean():.5f} m/s"
)
print(
    f"vy std  = {velocity_errors[:, 1].std():.5f} m/s"
)


# --------------------------------------------------
# Plot true vs measured x-position
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    measurement_times / 60,
    true_states[:, 0] / 1e3,
    label="True position",
)

plt.scatter(
    measurement_times / 60,
    measured_states[:, 0] / 1e3,
    s=15,
    label="Measured position",
)

plt.xlabel("Time (minutes)")
plt.ylabel("x position (km)")
plt.title("True vs Simulated Navigation Measurements")
plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# Plot position measurement error
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    measurement_times / 60,
    position_errors[:, 0],
    s=8,
    label="x-position error",
)

plt.axhline(
    0.0,
    linewidth=1,
)

plt.xlabel("Time (minutes)")
plt.ylabel("Measurement error (m)")
plt.title("Position Measurement Error")
plt.legend()
plt.grid(True)

plt.show()

