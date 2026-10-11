
import numpy as np

from spacecraft_sim.closed_loop import run_closed_loop_simulation
from spacecraft_sim.dynamics import EARTH_MU
from spacecraft_sim.propagation import rk4_step


def main():
    dt = 1.0
    num_steps = 3600

    position_noise_std = 10.0
    velocity_noise_std = 0.01
    position_gain = 2.0e-6
    velocity_gain = 2.0e-3
    max_acceleration = 0.05

    # Nominal circular reference orbit.
    orbital_radius = 6.771e6
    circular_speed = np.sqrt(EARTH_MU / orbital_radius)

    reference_state = np.array(
        [orbital_radius, 0.0, 0.0, circular_speed]
    )

    # Spacecraft starts with a position and velocity offset.
    initial_true_state = reference_state + np.array(
        [100.0, -100.0, 1.0, -1.0]
    )
    initial_estimate = initial_true_state + np.array(
        [20.0, -20.0, 0.2, -0.2]
    )

    initial_covariance = np.diag(
        [100.0**2, 100.0**2, 1.0, 1.0]
    )
    process_noise = np.diag([0.1, 0.1, 1e-4, 1e-4])
    measurement_noise = np.diag(
        [
            position_noise_std**2,
            position_noise_std**2,
            velocity_noise_std**2,
            velocity_noise_std**2,
        ]
    )

    (
        times,
        true_history,
        estimated_history,
        reference_history,
        covariance_history,
        control_history,
    ) = run_closed_loop_simulation(
        initial_true_state=initial_true_state,
        initial_estimate=initial_estimate,
        reference_state=reference_state,
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
        rng=np.random.default_rng(42),
    )

    # Tracking errors against the reference trajectory.
    position_tracking_error = np.linalg.norm(
        true_history[:, :2] - reference_history[:, :2],
        axis=1,
    )
    velocity_tracking_error = np.linalg.norm(
        true_history[:, 2:] - reference_history[:, 2:],
        axis=1,
    )

    # Estimation errors against the simulated true state.
    position_estimation_error = np.linalg.norm(
        estimated_history[:, :2] - true_history[:, :2],
        axis=1,
    )
    velocity_estimation_error = np.linalg.norm(
        estimated_history[:, 2:] - true_history[:, 2:],
        axis=1,
    )

    control_magnitude = np.linalg.norm(control_history, axis=1)

    # Uncontrolled baseline, starting from the same actual initial state.
    uncontrolled_history = np.zeros_like(true_history)
    uncontrolled_history[0] = initial_true_state

    for step in range(num_steps):
        uncontrolled_history[step + 1] = rk4_step(
            uncontrolled_history[step],
            dt,
            EARTH_MU,
        )

    baseline_position_error = np.linalg.norm(
        uncontrolled_history[:, :2] - reference_history[:, :2],
        axis=1,
    )

    baseline_velocity_error = np.linalg.norm(
        uncontrolled_history[:, 2:] - reference_history[:, 2:],
        axis=1,
    )

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values))))

    print("\n=== Closed-Loop Spacecraft GNC Experiment ===")
    print(f"Duration: {times[-1]:.1f} s ({times[-1] / 60:.1f} min)")
    print(f"Timestep: {dt:.2f} s")
    print(f"Integration steps: {num_steps}")

    print("\n--- Position Tracking ---")
    print(f"Initial error:      {position_tracking_error[0]:.3f} m")
    print(f"Final error:        {position_tracking_error[-1]:.3f} m")
    print(f"Tracking RMSE:      {rmse(position_tracking_error):.3f} m")
    print(f"Baseline final:     {baseline_position_error[-1]:.3f} m")
    print(f"Baseline RMSE:      {rmse(baseline_position_error):.3f} m")

    print("\n--- Velocity Tracking ---")
    print(f"Final error:        {velocity_tracking_error[-1]:.6f} m/s")
    print(f"Tracking RMSE:      {rmse(velocity_tracking_error):.6f} m/s")
    print(f"Baseline final:     {baseline_velocity_error[-1]:.6f} m/s")
    print(f"Baseline RMSE:      {rmse(baseline_velocity_error):.6f} m/s")

    print("\n--- State Estimation ---")
    print(f"Position RMSE:      {rmse(position_estimation_error):.3f} m")
    print(f"Velocity RMSE:      {rmse(velocity_estimation_error):.6f} m/s")

    print("\n--- Control Effort ---")
    print(f"Maximum acceleration: {np.max(control_magnitude):.6f} m/s^2")
    print(f"RMS acceleration:     {rmse(control_magnitude):.6f} m/s^2")
    print(f"Acceleration limit:   {max_acceleration:.6f} m/s^2")

    # Control history has one fewer sample than the state histories.
    padded_control = np.pad(
        control_magnitude,
        (0, 1),
        constant_values=np.nan,
    )

    results = np.column_stack(
        [
            times,
            position_tracking_error,
            velocity_tracking_error,
            position_estimation_error,
            velocity_estimation_error,
            padded_control,
            baseline_position_error,
        ]
    )

    output_path = "closed_loop_results.csv"
    np.savetxt(
        output_path,
        results,
        delimiter=",",
        header=(
            "time_s,position_tracking_error_m,"
            "velocity_tracking_error_m_s,"
            "position_estimation_error_m,"
            "velocity_estimation_error_m_s,"
            "control_acceleration_m_s2,"
            "baseline_position_error_m"
        ),
        comments="",
    )

    print(f"\nSaved results to {output_path}")
    print(f"All state values finite: {np.all(np.isfinite(true_history))}")
    print(f"All estimate values finite: {np.all(np.isfinite(estimated_history))}")
    print(f"All control values finite: {np.all(np.isfinite(control_history))}")


if __name__ == "__main__":
    main()
