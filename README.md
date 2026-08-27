# VLEO Mission Analysis and Platform Analysis for an EP-driven, EO Satellite

An advanced numerical simulation, trade-off analysis, and validation framework for Very Low Earth Orbit (VLEO) and Low Earth Orbit (LEO) Earth Observation (EO) satellite missions, developed as part of an Individual Research Project at **Cranfield University** in collaboration with **ArianeGroup**.

Powered by **Orekit** (via `orekit_jpype`), this tool integrates high-fidelity atmospheric density models, solar radiation pressure, electric propulsion drag-compensation controllers, solar/eclipse tracking, and power subsystem degradation models.

---

## Key Features

- **High-Fidelity Orbital Propagation**: Numerical orbit propagation driven by Orekit, incorporating Earth gravity harmonics, Solar Radiation Pressure (SRP), and third-body perturbations.
- **Multi-Model Atmospheric Drag**: Supports NRLMSISE-00, JB2008, DTM2000, and Harris-Priester thermospheric density models with historical space weather data (F10.7 solar flux, geomagnetic indices).
- **Electric Propulsion Modeling**: Detailed system models for radio-frequency ion thrusters (e.g., RIT $\mu$X, RIT-10 EVO, RIT-2X) including multi-point operating throttle tables, specific impulse, power draw, propellant consumption, and cycle life.
- **Drag Compensation & Orbit Control**:
  - *Duty-Cycle Drag Compensation*: ON/OFF threshold-based station keeping between configurable altitude bands.
  - *Continuous Altitude Maintenance*: Dynamic real-time drag-cancellation thrust modulation.
  - *Staged Mission Profiles*: Multi-phase operational profiles with dynamic altitude transitions (e.g., GOCE mission reconstruction).
- **Power Subsystem & Eclipse Dynamics**: Umbra/penumbra geometric event detection, solar array generation, battery charge/discharge cycling (SoC), and annual cell degradation.
- **Flight-Data Validation**: Validated against real Two-Line Element (TLE) orbital decay and drag compensation data from historical missions: **GOCE**, **GRACE-1/2**, **CHAMP**, **SLATS**, and **SOAR**.
- **Comprehensive Verification Suite**: Complete test harness comprising 45 unit verification tests, analytical Keplerian/J2 orbital benchmarks, integrated subsystem energy conservation tests, and environmental model cross-comparisons.
- **Interactive GUI & Visualizations**: Tkinter graphical user interface, publication-style plot generators (`orbital_plot_style`), and Plotly 3D orbital trajectory visualizers.

---

## Directory Structure

```text
03_code/
├── .gitignore                      # Git ignore configuration
├── README.md                       # Project overview and documentation
├── requirements.txt                # Python dependencies
│
├── orekit-data-main/               # Orekit physical data (IERS bulletins, leap seconds, geomagnetic data)
│
├── src/                            # Source code modules
│   ├── main.py                     # Main command-line entry point
│   ├── gui.py                      # Interactive Graphical User Interface
│   ├── gui_timestep_sensitivity.py # Headless timestep sensitivity study runner
│   ├── mission_config.py           # Mission presets, constants, and thruster database
│   ├── tradeoff_case_simulations.py# Automated parametric trade-off simulation engine
│   ├── orbital_plot_style.py       # Standardized publication-grade plotting system
│   ├── generate_report.py          # HTML report compilation utilities
│   │
│   ├── orbital_models/             # Core simulation subsystem models
│   │   ├── __init__.py
│   │   ├── atmospheric_model.py    # Orekit propagator setup, drag, SRP, and eclipse detection
│   │   ├── electric_propulsion.py  # Electric propulsion system & thruster dynamics
│   │   └── power_subsystem.py      # Solar array generation, battery storage, and degradation
│   │
│   └── validations/                # Flight data validation scripts & TLE datasets
│       ├── orbital_decay_validation.py      # Multi-spacecraft passive decay validation
│       ├── drag_compensation_validation.py  # Drag compensation validation against TLEs
│       ├── staged_mission_validation.py     # Staged altitude profile tracking validation
│       ├── test_eclipse_detection.py        # Eclipse detector validation against SGP4 truth
│       └── tle_data/                        # Historical TLE archives (GOCE, GRACE, CHAMP, SLATS, SOAR)
│
├── verification_tests/             # Formal model verification suite (45 unit tests + analytical checks)
│   ├── README.md                   # Verification suite documentation
│   ├── run_all_verification_tests.py
│   ├── test_45_unit_verification.py
│   ├── test_46_orbital_analytical_verification.py
│   ├── test_47_integrated_model_verification.py
│   ├── test_48_environmental_model_validation.py
│   ├── verification_config.py
│   └── results/                    # Verification logs, metrics, CSVs, and plots
│
├── scripts/                        # Utility tools
│   └── visualize_orbit.py          # 3D interactive orbit visualizer (Plotly)
│
├── tests/                          # Environment checks & smoke tests
│   ├── test_orekit.py              # Orekit VM initialization and basic propagation check
│   ├── tle_eclipse_detector.py     # Standalone TLE eclipse duration processor
│   ├── download_texture.py         # Earth texture fetcher for 3D visualization
│   └── earth.jpg                   # Earth texture asset
│
└── results/                        # Generated simulation, validation, and sensitivity results
    ├── results_dc_validations/     # Drag compensation validation figures and data
    ├── results_decay_validations/  # Orbital decay validation figures and CSVs
    ├── results_eclipses/           # Eclipse duration validation plots
    └── timestep_sensitivity_results/# Timestep convergence plots and error metrics
```

---

## Prerequisites

1. **Python 3.10+** (64-bit recommended).
2. **Java Runtime Environment (JRE) / Java Development Kit (JDK) 8, 11, or 17+**: Required for `JPype1` and the underlying Java Orekit library. Ensure `JAVA_HOME` is configured in your system environment variables.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd 03_code
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Orekit installation**:
   ```bash
   python tests/test_orekit.py
   ```

---

## Usage Guide

### 1. Interactive Graphical Interface (GUI)
Launch the GUI to configure satellite parameters, select atmospheric models, choose electric thrusters, configure drag compensation strategies, and run simulations interactively:
```bash
python src/gui.py
```

### 2. Command-Line Simulation
Run standard mission simulations and validations directly from the CLI:
```bash
python src/main.py
```

### 3. Flight Data Validations
Execute flight-data validation runs against real satellite TLE archives:
```bash
# Staged altitude mission tracking (GOCE profile)
python src/validations/staged_mission_validation.py

# Drag compensation validation
python src/validations/drag_compensation_validation.py

# Passive orbital decay multi-satellite validation
python src/validations/orbital_decay_validation.py
```

### 4. Verification Test Suite
Execute the full formal verification test suite (Unit tests, Analytical Keplerian/J2, Integrated conservation, and Environmental models):
```bash
python verification_tests/run_all_verification_tests.py
```
Test results and summary reports will be written to `verification_tests/results/`.

### 5. Timestep Sensitivity Analysis
Run headless timestep convergence studies across multiple step sizes (e.g., 1 s, 30 s, 60 s, 300 s):
```bash
python src/gui_timestep_sensitivity.py
```

### 6. 3D Interactive Orbit Visualizer
Render interactive 3D trajectories with Earth coastlines and equatorial planes:
```bash
python scripts/visualize_orbit.py
```

---

## Author & Attribution

- **Project**: VLEO Mission Analysis and Platform Analysis for an EP-driven, EO Satellite
- **Author**: Arnau Pascual
- **Institution**: Cranfield University — Individual Research Project
- **Industrial Collaboration**: ArianeGroup
- **Year**: 2026
#   V L E O - M i s s i o n - A n a l y s i s - a n d - P l a t f o r m - A n a l y s i s - f o r - a n - E P - d r i v e n - E O - S a t e l l i t e  
 