"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          verification_config.py
Description:
    Shared verification framework configuration, tolerances, Orekit initialization, and tabular reporting utilities.
===============================================================================
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# -----------------------------------------------------------------------------
# 1. DIRECTORY PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = SRC_DIR / "orbital_models"
DATA_DIR = PROJECT_ROOT / "orekit-data-main"
VERIF_DIR = PROJECT_ROOT / "verification_tests"
RESULTS_DIR = VERIF_DIR / "results"

PLOTS_DIR_ORBITAL = RESULTS_DIR / "orbital_analytical_plots"
PLOTS_DIR_INTEGRATED = RESULTS_DIR / "integrated_model_plots"
PLOTS_DIR_ENVIRONMENTAL = RESULTS_DIR / "environmental_model_plots"

# Add source directories to Python search path
for d in [str(SRC_DIR), str(MODELS_DIR), str(PROJECT_ROOT)]:
    if d not in sys.path:
        sys.path.insert(0, d)


def ensure_results_directories() -> None:
    """Create results directory and all subfolders if they do not exist."""
    for p in [RESULTS_DIR, PLOTS_DIR_ORBITAL, PLOTS_DIR_INTEGRATED, PLOTS_DIR_ENVIRONMENTAL]:
        p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. OREKIT INITIALIZATION
# -----------------------------------------------------------------------------
_OREKIT_INITIALIZED = False


def init_orekit(data_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Initialize JPype JVM and load Orekit data archive safely.

    Returns
    -------
    bool: True if initialization succeeded, False otherwise.
    """
    global _OREKIT_INITIALIZED
    if _OREKIT_INITIALIZED:
        return True

    target_data = Path(data_path) if data_path else DATA_DIR

    try:
        import orekit_jpype as orekit
        try:
            orekit.initVM()
        except Exception:
            # JVM might already be running
            pass

        from orekit_jpype.pyhelpers import setup_orekit_data

        if not target_data.exists():
            print(f"[ERROR] Orekit data directory not found at: {target_data.resolve()}")
            print("Please ensure 'orekit-data-main' exists in the project root.")
            return False

        setup_orekit_data(filenames=str(target_data), from_pip_library=False)
        _OREKIT_INITIALIZED = True
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize Orekit: {e}")
        return False

# Initialize Orekit upon loading config to ensure seamless imports of models
init_orekit()


# -----------------------------------------------------------------------------
# 3. VERIFICATION TOLERANCES
# -----------------------------------------------------------------------------
REL_TOL_STRICT = 1e-10          # For exact analytical identities (e.g. mdot equation)
REL_TOL_ENGINEERING = 1e-3     # For numerical integration and step balance (0.1%)
REL_TOL_MISSION = 1e-2         # For coupled multi-perturbation mission metrics (1.0%)
ALTITUDE_TOL_KM = 0.5          # Threshold boundary tolerance [km]
TIME_TOL_PERCENT = 1.0         # Timing / period tolerance [%]
ECLIPSE_TIME_TOL_S = 60.0      # Eclipse duration tolerance [s]

# Physical Constants
G0 = 9.80665                   # Standard gravity [m/s^2]
EARTH_RADIUS_WGS84 = 6378137.0 # WGS84 Equatorial radius [m]
EARTH_MU_WGS84 = 3.986004418e14 # Earth gravitational parameter [m^3/s^2]


# -----------------------------------------------------------------------------
# 4. DEFAULT SPACECRAFT & MISSION PARAMETERS
# -----------------------------------------------------------------------------
DEFAULT_SPACECRAFT = {
    "mass": 500.0,             # [kg]
    "cross_section": 1.0,      # [m^2]
    "drag_coeff": 4.0,         # [-]
    "reflectivity_coeff": 1.5, # [-]
    "thrust": 0.015,           # [N] (15 mN)
    "isp": 2500.0,             # [s]
    "propellant_mass": 20.0,   # [kg]
    "solar_panel_area_m2": 2.0,# [m^2]
    "panel_efficiency": 0.30,  # [-]
    "solar_flux_W_m2": 1361.0, # [W/m^2]
    "battery_capacity_Wh": 300.0, # [Wh]
    "battery_initial_Wh": 300.0,  # [Wh]
    "housekeeping_power_W": 50.0, # [W]
    "thruster_power_W": 250.0,    # [W]
}


# -----------------------------------------------------------------------------
# 5. TEST RECORD & REPORTING DATA STRUCTURES
# -----------------------------------------------------------------------------
@dataclass
class TestRecord:
    """Represents a single ECSS verification test result."""
    test_id: str
    test_name: str
    requirement: str
    method: str  # "Analysis", "Test", "Inspection", "Review of Design"
    expected: str
    obtained: str
    error: str = "N/A"
    tolerance: str = "N/A"
    status: str = "PASS"  # "PASS", "FAIL", "WARNING"
    notes: str = ""

    def to_csv_dict(self) -> Dict[str, str]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "requirement": self.requirement,
            "method": self.method,
            "expected": self.expected,
            "obtained": self.obtained,
            "error": self.error,
            "tolerance": self.tolerance,
            "status": self.status,
            "notes": self.notes,
        }


def save_csv(filepath: Union[str, Path], records: Sequence[TestRecord]) -> Path:
    """Save test records to a CSV file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "test_id", "test_name", "requirement", "method",
        "expected", "obtained", "error", "tolerance", "status", "notes"
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_csv_dict())
    return p


def save_plot(fig, filename: str, subfolder: Optional[Union[str, Path]] = None) -> Path:
    """Save a matplotlib figure to PDF and PNG in high resolution."""
    dest_dir = Path(subfolder) if subfolder else RESULTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    png_path = dest_dir / f"{filename}.png"
    pdf_path = dest_dir / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path


def write_markdown_summary(summary_path: Union[str, Path], all_records: Sequence[TestRecord]) -> Path:
    """
    Generate an ECSS-style verification matrix markdown summary.
    """
    p = Path(summary_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    total_tests = len(all_records)
    passes = sum(1 for r in all_records if r.status == "PASS")
    failures = sum(1 for r in all_records if r.status == "FAIL")
    warnings = sum(1 for r in all_records if r.status == "WARNING")

    lines = [
        "# VLEO Mission Analysis Verification & Validation Summary",
        "",
        f"**Date Generated**: Auto-generated during test suite run  ",
        f"**Total Tests**: {total_tests} | **Passed**: {passes} | **Failed**: {failures} | **Warnings / Review Items**: {warnings}",
        "",
        "## 1. Executive Summary & Verification Compliance",
        "",
        "This document presents the formal verification and validation results for the preliminary VLEO orbital mission analysis framework. Verification methods follow standard ECSS nomenclature:",
        "- **Test (T)**: Direct execution of software under controlled conditions and comparison against numerical criteria.",
        "- **Analysis (A)**: Mathematical derivation and comparison against closed-form analytical solutions.",
        "- **Inspection (I)**: Verification of code logic, parameter routing, and unit consistency.",
        "- **Review of Design (R)**: Identification of architectural assumptions, interface limits, and recommended model improvements.",
        "",
        "## 2. Complete Verification Matrix",
        "",
        "| Test ID | Test Name | Req / Target | Method | Expected Result | Obtained Result | Status | Notes |",
        "|:---|:---|:---|:---:|:---|:---|:---:|:---|"
    ]

    for r in all_records:
        status_badge = f"**{r.status}**" if r.status != "PASS" else "PASS"
        escaped_expected = r.expected.replace("|", "\\|")
        escaped_obtained = r.obtained.replace("|", "\\|")
        escaped_notes = r.notes.replace("|", "\\|")
        lines.append(
            f"| `{r.test_id}` | {r.test_name} | {r.requirement} | {r.method} | "
            f"{escaped_expected} | {escaped_obtained} | {status_badge} | {escaped_notes} |"
        )

    lines.extend([
        "",
        "## 3. Key Review of Design Findings & Recommendations",
        "",
        "1. **`duty_cycle` Property Semantics**: In `ElectricPropulsionSystem`, the property `duty_cycle` returns accumulated burn time in seconds rather than a dimensionless percentage. *Recommendation: rename to `total_burn_time_s` and provide `duty_cycle_fraction = burn_time / total_mission_time`.*",
        "2. **Altitude Frame Clarification**: In `run_simulation()`, output altitude in some paths has been referenced to mean Earth radius ($a - R_{mean}$), whereas `AltitudeDetector` uses WGS84 geodetic altitude above the ellipsoid. *Recommendation: explicitly output both `sma_altitude_km` and `geodetic_altitude_km`.*",
        "3. **Force Model Configurability**: In `atmospheric_model.py`, force models (gravity degree/order, third bodies, drag, SRP) are currently hardcoded. Exposing a `params['force_models']` dictionary allows strict two-body and J2-only verification.",
        "4. **Integrator Exposing**: Exposing `params['integrator']` (`min_step`, `max_step`, `position_tolerance`) enables fine-grained convergence studies without modifying source files.",
        "5. **Power Inhibition Across Modes**: Power state checks are active in duty cycle and step-maintenance loops; ensuring universal enforcement across all future custom maneuver modes is recommended.",
        "",
        "---",
        "*Generated by the VLEO Mission Analysis Automated Verification Suite.*"
    ])

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return p
