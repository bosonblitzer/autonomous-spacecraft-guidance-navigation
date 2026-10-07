import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from spacecraft_sim.dynamics import EARTH_MU, state_derivative


# Earth and spacecraft parameters
EARTH_RADIUS = 6_371_000.0  # m
ALTITUDE = 700_000.0        # m

INITIAL_RADIUS = EARTH_RADIUS + ALTITUDE

# Circular orbital velocity
INITIAL_SPEED = np.sqrt(EARTH_MU / INITIAL_RADIUS)


# Initial state:
# spacecraft starts on x-axis moving in +y direction
initial_state = np.array([
    INITIAL_RADIUS,
    0.0,
    0.0,
    INITIAL_SPEED,
])


# Estimate one orbital period
orbital_period = (
    2 * np.pi
    * np.sqrt(INITIAL_RADIUS**3 / EARTH_MU)
)

print(f"Orbital radius: {INITIAL_RADIUS / 1e3:.1f} km")
print(f"Initial speed: {INITIAL_SPEED:.2f} m/s")
print(f"Orbital period: {orbital_period / 60:.2f} minutes")


# Simulate one orbit
solution = solve_ivp(
    state_derivative,
    (0, orbital_period),
    initial_state,
    rtol=1e-9,
    atol=1e-9,
    max_step=20.0,
)


# Extract state history
x = solution.y[0]
y = solution.y[1]
vx = solution.y[2]
vy = solution.y[3]

# Calculate orbital radius and speed
radius = np.sqrt(x**2 + y**2)
speed = np.sqrt(vx**2 + vy**2)

# Specific orbital mechanical energy
energy = 0.5 * speed**2 - EARTH_MU / radius

print("\n--- Orbit Validation ---")
print(f"Minimum radius: {radius.min() / 1e3:.3f} km")
print(f"Maximum radius: {radius.max() / 1e3:.3f} km")
radius_variation = radius.max() - radius.min()
radius_relative_variation = radius_variation / radius.mean()

print(
    f"Relative radius variation: "
    f"{radius_relative_variation:.3e}"
)

print(f"Minimum speed: {speed.min():.3f} m/s")
print(f"Maximum speed: {speed.max():.3f} m/s")
speed_variation = speed.max() - speed.min()
speed_relative_variation = speed_variation / speed.mean()

print(
    f"Relative speed variation: "
    f"{speed_relative_variation:.3e}"
)

print(f"Energy variation: "
      f"{(energy.max() - energy.min()):.6e} J/kg")

# Numerical convergence study
print("\n--- Numerical Convergence Study ---")

max_steps = [60.0, 20.0, 5.0]

for max_step in max_steps:
    test_solution = solve_ivp(
        state_derivative,
        (0, orbital_period),
        initial_state,
        rtol=1e-9,
        atol=1e-9,
        max_step=max_step,
    )

    final_state = test_solution.y[:, -1]

    # Compare final state with the initial state
    position_error = np.linalg.norm(
        final_state[:2] - initial_state[:2]
    )

    velocity_error = np.linalg.norm(
        final_state[2:] - initial_state[2:]
    )

    print(
        f"max_step = {max_step:5.1f} s | "
        f"position error = {position_error:.6e} m | "
        f"velocity error = {velocity_error:.6e} m/s"
    )
# Plot trajectory
plt.figure(figsize=(7, 7))

plt.plot(x / 1e3, y / 1e3)

# Earth
earth = plt.Circle(
    (0, 0),
    EARTH_RADIUS / 1e3,
    alpha=0.3,
)

plt.gca().add_patch(earth)

plt.xlabel("x position (km)")
plt.ylabel("y position (km)")
plt.title("Simulated Spacecraft Orbit")

plt.axis("equal")
plt.grid(True)

plt.show()