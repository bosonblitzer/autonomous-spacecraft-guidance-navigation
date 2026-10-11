# Autonomous Spacecraft Guidance, Navigation & Control Simulation

A Python-based spacecraft simulation project exploring orbital dynamics, state estimation, feedback control and system performance through numerical experiments and automated testing.

## Project Overview

This project models a spacecraft operating under two-body gravitational dynamics and develops an integrated guidance, navigation and control (GNC) simulation. An Extended Kalman Filter (EKF) estimates the spacecraft's state from noisy measurements, while a proportional-derivative (PD) controller generates corrective accelerations to track a reference orbit.

The project investigates numerical integration accuracy, tracking performance, control effort and sensitivity to measurement noise.

## System Architecture

- **Orbital dynamics:** Two-body gravitational model using Earth's gravitational parameter.
- **Numerical propagation:** Fourth-order Runge–Kutta (RK4) integration.
- **Navigation:** Simulated position and velocity measurements with configurable noise.
- **State estimation:** Extended Kalman Filter for estimating spacecraft position and velocity.
- **Feedback control:** PD controller with a configurable acceleration limit.
- **Verification:** Automated unit and integration tests, simulation experiments and parameter trade studies.

The spacecraft state is represented as:

\[
\mathbf{x} = [x,\ y,\ v_x,\ v_y]^T
\]

where \(x,y\) are planar position coordinates and \(v_x,v_y\) are the corresponding velocity components.

## Simulation Results

### State estimation

In a 10,000-second estimation experiment using RK4 state propagation, the simulation produced:

| Metric | Result |
|---|---:|
| Position estimation RMSE | 28.410 m |
| Final position estimation error | 15.059 m |
| Velocity estimation RMSE | 0.244 m/s |
| Final velocity estimation error | 0.106 m/s |
| Final position uncertainty | 16.735 m |

A numerical integration trade study showed that matching the EKF's state propagation to the RK4 dynamics model substantially improved estimation performance compared with the earlier Euler-based implementation.

### Closed-loop tracking

In a separate 3,600-second tracking experiment, the controlled spacecraft was compared with an uncontrolled baseline.

| Metric | Controlled simulation | Uncontrolled baseline |
|---|---:|---:|
| Final position tracking error | 36.854 m | 10,180.094 m |
| Position tracking RMSE | 277.986 m | 4,218.611 m |
| Final velocity tracking error | 0.001417 m/s | 10.327976 m/s |

The controller's peak acceleration was 0.003128 m/s², below the configured limit of 0.05 m/s².

These results apply to the specific simulated initial conditions, parameters and duration; they do not establish performance for all orbital conditions.

### Controller gain trade study

A 30-minute parameter study compared four proportional and derivative gain combinations. Within the tested cases, increasing the gains reduced position tracking RMSE from 808.707 m to 214.181 m, while increasing RMS control acceleration from 0.000631 m/s² to 0.002007 m/s².

This illustrates the trade-off between tracking performance and control effort. The tested combinations are not claimed to be globally optimal.

## Testing

The project currently has **44 passing automated tests** covering the implemented dynamics, propagation, navigation, estimation, control and closed-loop simulation components.

The tests include checks of input validation, numerical outputs, simulation dimensions, acceleration saturation and selected measurement-noise cases.

## Repository Structure

```text
src/spacecraft_sim/   Core simulation modules
tests/                Automated test suite
scripts/              Simulation and analysis scripts
requirements.txt      Python dependencies
pytest.ini            Pytest configuration
README.md             Project documentation
```

## Running the Project

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the automated tests:

```powershell
python -m pytest
```

Run the baseline state-estimation experiment:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_experiment.py
```

Run the closed-loop tracking experiment:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_closed_loop_experiment.py
```

Run the controller gain trade study:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_gain_trade_study.py
```

Run the measurement-noise robustness study:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_robustness_study.py
```

The experiment scripts generate numerical results for analysis. Output CSV files and plots depend on which scripts have been run.

## Modelling Assumptions and Limitations

- Motion is represented in two dimensions.
- The dynamics model uses two-body gravity and does not include perturbations such as atmospheric drag, Earth's non-spherical gravity field or third-body effects.
- Sensor measurements are simulated rather than obtained from real spacecraft hardware.
- The EKF and controller are evaluated under selected simulation conditions; the tests do not prove general stability or flight readiness.
- The controller models commanded acceleration directly and does not simulate detailed thruster dynamics, propellant consumption or actuator delays.

## Future Development

- Expand verification across additional initial conditions and orbital regimes.
- Analyse estimation and tracking performance across repeated noise realisations.
- Investigate numerical integration accuracy and EKF covariance consistency.
- Improve performance visualisations and document design trade-offs.
- Explore more realistic actuator constraints and orbital perturbations.

## Disclaimer

This is an educational engineering simulation developed as a personal technical project. It is not a flight-qualified guidance, navigation or control system.
