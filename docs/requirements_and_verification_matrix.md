# Requirements and Verification Matrix
**Project:** Autonomous Spacecraft Guidance, Navigation & Control Simulation
**Document:** Requirements and Verification Matrix
**Version:** 1.2
**Status:** Verification baseline updated with closed-loop, timestep-convergence and measurement-noise robustness evidence.

## 1. Purpose

This document defines selected functional and numerical requirements for the spacecraft simulation and records the verification methods and available evidence.

Requirements are assessed using automated tests, numerical experiments and output inspection. A passing software test demonstrates the behaviour covered by that test; it does not, by itself, establish flight readiness or general system stability.

## 2. Verification Methods

| Method | Description |
|---|---|
| Test | Automated test of a defined input, behaviour or acceptance criterion |
| Analysis | Mathematical or numerical assessment of system behaviour |
| Demonstration | Execution of a simulation or script and inspection of its outputs |

## 3. Requirements Matrix

| ID | Requirement | Verification method | Acceptance criterion | Evidence / status |
|---|---|---|---|---|
| GNC-001 | The dynamics model shall calculate spacecraft acceleration under two-body gravitational assumptions. | Test | Acceleration has the expected direction and magnitude for defined test states. | `tests/test_dynamics.py` — review test assertions before marking verified. |
| GNC-002 | The propagation module shall advance the spacecraft state using fourth-order Runge–Kutta integration. | Test / Analysis | Propagation returns finite states; relative energy and angular momentum errors remain below 1 × 10⁻⁹ over three simulated orbits at a 5 s timestep. | `tests/test_propagation.py`; `tests/test_conservation.py`; `scripts/run_conservation_study.py`; `conservation_study.csv`. Automated conservation tests pass. Timestep convergence study completed using an independent DOP853 reference solution; see Section 4. |
| GNC-003 | The navigation model shall generate position and velocity measurements with configurable noise. | Test | Measurement output has the expected dimensions and valid numerical values. | `tests/test_navigation.py` — review test assertions before marking verified. |
| GNC-004 | The estimator shall predict the spacecraft state and update its estimate using measurements. | Test / Analysis | EKF state and covariance remain finite for the tested cases; update behaviour satisfies the assertions in the test suite. | tests/test_ekf.py; scripts/run_robustness_study.py; robustness_study.csv. Position and velocity estimation RMSE recorded under three measurement-noise conditions across nine simulation runs. |
| GNC-005 | The controller shall calculate corrective acceleration from position and velocity tracking errors. | Test | Control output has two components and responds to the supplied state and target errors. | `tests/test_control.py` — review test assertions before marking verified. |
| GNC-006 | The closed-loop simulation shall integrate state estimation, reference tracking and feedback control. | Test / Demonstration | The simulation returns histories with consistent dimensions and finite values for the tested configuration. | tests/test_closed_loop.py; scripts/run_closed_loop_experiment.py; scripts/run_robustness_study.py; closed_loop_results.csv; robustness_study.csv. Closed-loop behaviour evaluated across nine measurement-noise and random-seed combinations. |
| GNC-007 | The controller shall respect its configured acceleration-magnitude limit. | Test / Demonstration | Control magnitude does not exceed the configured limit, within numerical tolerance. | tests/test_closed_loop.py; tests/test_closed_loop_robustness.py; scripts/run_robustness_study.py; robustness_study.csv. The configured 0.05 m/s² acceleration limit was respected in all nine robustness-study runs. |
| GNC-008 | The simulation shall support evaluation of controlled tracking against an uncontrolled baseline. | Demonstration / Analysis | Both cases are evaluated using consistent initial conditions, reference trajectory and error metrics. | `closed_loop_results.csv`; `scripts/run_closed_loop_experiment.py`. |
| GNC-009 | The project shall support quantitative comparison of controller gain settings. | Demonstration / Analysis | The gain study records tracking error and control-effort metrics for each tested configuration. | `gain_trade_study.csv`; `scripts/run_gain_trade_study.py`. |
| GNC-010 | The software shall detect selected invalid inputs and handle tested measurement-noise cases without producing non-finite state histories. | Test | The defined tests pass for their specified inputs and scenarios. | `tests/` — 46 tests pass; test coverage is limited to implemented cases. |
| GNC-011 | The project shall record performance results in a form suitable for further analysis. | Demonstration | Experiment outputs are saved in a structured, machine-readable format. | `closed_loop_results.csv`; `gain_trade_study.csv`. |
| GNC-012 | The project shall provide instructions for reproducing the documented experiments. | Demonstration / Inspection | Documented commands run successfully in a configured environment and reproduce the relevant outputs. | `README.md` — confirm all listed scripts execute successfully. |

## 4. Recorded Simulation Evidence

The following results were recorded in the current project experiments.

### State estimation

| Metric | Recorded result |
|---|---:|
| Position estimation RMSE | 28.410 m |
| Final position estimation error | 15.059 m |
| Velocity estimation RMSE | 0.244 m/s |
| Final velocity estimation error | 0.106 m/s |

### Closed-loop tracking

| Metric | Controlled | Uncontrolled baseline |
|---|---:|---:|
| Final position tracking error | 36.854 m | 10,180.094 m |
| Position tracking RMSE | 277.986 m | 4,218.611 m |
| Final velocity tracking error | 0.001417 m/s | 10.327976 m/s |

These values apply to the recorded simulation configurations only. They are not universal performance guarantees.
### RK4 timestep convergence

The RK4 propagation implementation was evaluated over three orbital periods using four nominal timesteps. Each numerical trajectory was compared against an independent high-accuracy reference solution generated using SciPy's DOP853 integrator (`rtol = 1 × 10⁻¹²`, `atol = 1 × 10⁻⁹`).

| Nominal timestep (s) | Actual timestep (s) | Steps | Position RMSE (m) | Maximum position error (m) | Final position error (m) | Maximum velocity error (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 20.0 | 20.01802 | 831 | To be recorded | To be recorded | To be recorded | To be recorded |
| 10.0 | 10.00275 | 1,663 | 0.0324474 | 0.0576876 | 0.0576876 | 6.536555 × 10⁻⁵ |
| 5.0 | 4.99987 | 3,327 | 0.00191884 | 0.00335096 | 0.00334152 | 3.786141 × 10⁻⁶ |
| 2.5 | 2.49993 | 6,654 | 0.000135721 | 0.000234247 | 0.000232784 | 2.635628 × 10⁻⁷ |

The study demonstrates decreasing position and velocity errors as the timestep is refined. The results are consistent with the expected fourth-order convergence behaviour of RK4 over the tested range.

**Evidence:** `scripts/run_timestep_convergence.py`; `timestep_convergence.csv`.

**Verification status:** Numerical convergence demonstrated for the tested orbital configuration. This result does not establish accuracy for every orbit, timestep or operating condition.
### Measurement-noise robustness study

The closed-loop simulation was evaluated under three measurement-noise conditions, with three random seeds per condition, for a total of nine runs. Each run simulated 1,800 time steps at a timestep of 1 s.

| Position noise standard deviation (m) | Velocity noise standard deviation (m/s) | Position estimation RMSE range (m) | Velocity estimation RMSE range (m/s) | Position tracking RMSE range (m) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.67 | 0.006665 | 389.35 |
| 10 | 0.01 | 1.80–2.10 | 0.011526–0.011770 | 389.08–389.45 |
| 100 | 0.1 | 6.81–11.89 | 0.031794–0.034889 | 386.76–390.40 |

All nine runs completed with finite true-state and estimated-state histories. The configured maximum acceleration of 0.05 m/s² was respected in every run. Peak acceleration across the experiments ranged from approximately 0.003111 to 0.003256 m/s².

Both position and velocity estimation errors increased with measurement noise. Position tracking RMSE remained broadly similar across the tested scenarios. These findings apply to the tested simulation configuration and noise levels only.

**Evidence:** `scripts/run_robustness_study.py`; `robustness_study.csv`.

**Verification status:** Numerical behaviour and actuator-limit compliance demonstrated for the nine tested cases. No formal tracking-performance acceptance threshold has yet been established.

## 5. Outstanding Verification Activities

- [ ] Inspect the automated assertions associated with each requirement.
- [x] Perform a timestep-convergence study for RK4 propagation; results recorded in Section 4.
- [ ] Perform a timestep-convergence study for RK4 propagation.
- [ ] Complete and execute the measurement-noise robustness study.
- [ ] Check that all README commands work from a clean environment.
- [ ] Record any failures, limitations and corrective actions.

## 6. Limitations

The simulation uses a two-dimensional, two-body gravitational model with simulated measurements and idealised commanded acceleration. It does not currently represent a flight-qualified spacecraft, detailed actuator dynamics, propellant consumption or all relevant orbital perturbations.

Verification evidence is limited to the implemented tests and recorded experiments. Passing tests do not prove general stability, safety or operational readiness.

## 7. Change History

| Version | Change |
|---|---|
| 1.0 | Initial requirements and verification baseline. |
| 1.1 | Added RK4 timestep-convergence evidence and updated verification status. |
| 1.2 | Added measurement-noise robustness results across nine runs, including velocity estimation RMSE and actuator-limit compliance evidence. Updated verification checklist and test status. |
