# VLEO Mission Analysis Verification & Validation Suite

This test suite provides an **ECSS-style Verification and Validation (V&V)** framework for the Python/Orekit VLEO (Very Low Earth Orbit) preliminary mission analysis codebase.

It is structured to directly support and produce quantitative evidence for the thesis chapters:
* **Section 4.5**: Unit Verification (Electric Propulsion & Electrical Power Subsystem)
* **Section 4.6**: Orbital Analytical Verification (Keplerian mechanics, drag, propellant conservation, eclipse geometry, RAAN drift)
* **Section 4.7**: Integrated Model Verification (Coupled orbit, atmosphere, propulsion control, power generation and battery storage)
* **Section 4.8**: Environmental Model Validation (Atmosphere models, density sensitivity, ballistic drag variations, mission-level impact)

---

## Directory Structure

```text
verification_tests/
├── verification_config.py                    # Shared configuration, Orekit initialization, tolerances & reporting
├── test_45_unit_verification.py               # Section 4.5: Unit Verification
├── test_46_orbital_analytical_verification.py # Section 4.6: Orbital Analytical Verification
├── test_47_integrated_model_verification.py   # Section 4.7: Integrated Model Verification
├── test_48_environmental_model_validation.py  # Section 4.8: Environmental Model Validation
├── run_all_verification_tests.py              # Master test suite runner & report generator
├── README.md                                  # This documentation
└── results/                                   # Generated CSVs, plots, and verification matrix
    ├── unit_verification_results.csv
    ├── orbital_analytical_results.csv
    ├── orbital_analytical_plots/
    ├── integrated_model_results.csv
    ├── integrated_model_plots/
    ├── environmental_model_results.csv
    ├── environmental_model_plots/
    └── verification_summary.md
```

---

## Requirements and Prerequisites

* **Python Environment**: Python 3.9+ with `orekit-jpype`, `numpy`, `matplotlib`, and `scipy`.
* **Orekit Data**: The folder `orekit-data-main/` must be present at the project root.
* **Environment Execution**: Using the provided `orekit_env` virtual environment:
  ```powershell
  .\orekit_env\Scripts\python.exe verification_tests\run_all_verification_tests.py
  ```

---

## How to Run

### Run All Verification Tests
To run the complete verification suite and generate all CSVs, figures, and the final Markdown verification matrix:

```bash
python verification_tests/run_all_verification_tests.py
```
*(or with the project virtual environment)*:
```bash
.\orekit_env\Scripts\python.exe verification_tests\run_all_verification_tests.py
```

### Run Individual Test Groups

* **4.5 Unit Verification**:
  ```bash
  .\orekit_env\Scripts\python.exe verification_tests\test_45_unit_verification.py
  ```
* **4.6 Orbital Analytical Verification**:
  ```bash
  .\orekit_env\Scripts\python.exe verification_tests\test_46_orbital_analytical_verification.py
  ```
* **4.7 Integrated Model Verification**:
  ```bash
  .\orekit_env\Scripts\python.exe verification_tests\test_47_integrated_model_verification.py
  ```
* **4.8 Environmental Model Validation**:
  ```bash
  .\orekit_env\Scripts\python.exe verification_tests\test_48_environmental_model_validation.py
  ```

---

## Verification Logic and ECSS Methods

Verification methods follow European Cooperation for Space Standardization (**ECSS-E-ST-10-02C**) classifications:
1. **Test (T)**: Direct numerical execution under specified initial conditions with quantitative acceptance criteria.
2. **Analysis (A)**: Comparison against closed-form analytical equations (e.g., Keplerian period, rocket mass flow rate, rocket equation, secular $J_2$ precession).
3. **Inspection (I)**: Verification of code logic, sign conventions, parameter routing, and unit consistency.
4. **Review of Design (R)**: Identification of architectural assumptions, interface limits, and documentation of design improvements.

### Standard Verification Tolerances

| Parameter Category | Constant / Symbol | Value | Application |
|:---|:---:|:---:|:---|
| Strict Analytical Identity | `REL_TOL_STRICT` | $10^{-10}$ | Exact physical equations ($\dot{m}$, solar flux) |
| Engineering Step Integration | `REL_TOL_ENGINEERING` | $10^{-3}$ (0.1%) | Propellant balance, numerical battery charge |
| Mission Metric Consistency | `REL_TOL_MISSION` | $10^{-2}$ (1.0%) | Coupled multi-orbit mission metrics |
| Altitude Threshold Boundary | `ALTITUDE_TOL_KM` | 0.5 km | Trigger boundary detection |
| Orbital Period Tolerance | `TIME_TOL_PERCENT` | 1.0% | Keplerian period comparison |
| Eclipse Duration Tolerance | `ECLIPSE_TIME_TOL_S` | 60.0 s | Umbra crossing duration |

---

## Interpreting Statuses

* **`PASS`**: The test executed cleanly and the obtained numerical result satisfies the specified tolerance or exact logical condition.
* **`FAIL`**: A numerical error or logic violation occurred that deviates from the physical requirement.
* **`WARNING`**: Highlights an architectural observation, semantic subtlety, or Review of Design finding without breaking functional execution.

---

## Key Review of Design Findings

1. **`duty_cycle` Property Semantics (`UV-EP-08`)**:
   - In `ElectricPropulsionSystem`, the property `duty_cycle` returns accumulated burn time in seconds rather than a dimensionless fraction ($0 \le \delta \le 1$).
   - *Recommendation*: Rename property to `total_burn_time_s` and add `duty_cycle_fraction = burn_time / mission_time`.

2. **Force Model Toggles (`OAV-ORB-02`)**:
   - `run_simulation()` in `atmospheric_model.py` defaults to activating the full perturbation stack ($10 \times 10$ gravity, Sun/Moon third bodies, SRP, drag).
   - *Recommendation*: Expose a `params['force_models']` dictionary allowing unperturbed two-body and $J_2$-only ablation runs.

3. **Altitude Reference Frames (`OAV-ORB-01` / `OAV-ORB-06`)**:
   - In some paths, logged altitude uses semi-major axis minus mean radius ($a - R_{mean}$), whereas Orekit `AltitudeDetector` uses WGS84 geodetic altitude above the ellipsoid.
   - *Recommendation*: Explicitly output both `sma_altitude_km` and `geodetic_altitude_km`.

4. **Burn Time Check in Propagation Loop (`IMV-INT-05`)**:
   - `run_simulation()` enforces `max_cycles` in its duty-cycle loop, but `max_burn_time` is only evaluated when `ElectricPropulsionSystem.update()` is called directly.
   - *Recommendation*: Add a burn time threshold check inside `run_simulation()`'s internal duty cycle loop.

5. **Universal Power Inhibition (`IMV-INT-07`)**:
   - Power inhibition logic is implemented for duty-cycle and maintenance modes. Ensuring consistent enforcement across all custom maneuver modes is recommended.
