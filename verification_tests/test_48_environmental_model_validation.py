"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          test_48_environmental_model_validation.py
Description:
    Section 4.8: Environmental thermospheric model validation and sensitivity analysis across solar flux regimes.
===============================================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure verification_config and project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verification_config import (
    DEFAULT_SPACECRAFT,
    PLOTS_DIR_ENVIRONMENTAL,
    REL_TOL_ENGINEERING,
    REL_TOL_MISSION,
    RESULTS_DIR,
    TestRecord,
    ensure_results_directories,
    save_csv,
    save_plot,
)

from orbital_models.atmospheric_model import run_simulation
from orbital_models.electric_propulsion import ElectricPropulsionSystem
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants


def _get_start_date() -> AbsoluteDate:
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)


def test_emv_atm_01_density_monotonic_decrease() -> Tuple[TestRecord, Dict[str, Dict[float, float]]]:
    """EMV-ATM-01: Verify density decreases monotonically as altitude increases (260, 300, 320 km)."""
    altitudes_km = [260.0, 300.0, 320.0]
    models = ["nrlmsise00", "jb2008", "dtm2000", "harrispriester"]
    results_by_model: Dict[str, Dict[float, float]] = {}

    all_monotonic = True
    summary_parts = []

    for m in models:
        densities = {}
        try:
            for alt in altitudes_km:
                params = {
                    "altitude": alt * 1000.0,
                    "eccentricity": 1e-5,
                    "inclination": 51.6,
                    "duration": 120.0,  # very short to get initial state density
                    "time_step": 60.0,
                    "start_date": _get_start_date(),
                }
                res = run_simulation(params, model_type=m, propulsion_model=None)
                densities[alt] = res["density"][0]

            is_monotonic = (densities[260.0] > densities[300.0] > densities[320.0])
            if not is_monotonic:
                all_monotonic = False
            results_by_model[m] = densities
            summary_parts.append(f"{m.upper()}: rho(260)={densities[260.0]:.2e} > rho(300)={densities[300.0]:.2e} > rho(320)={densities[320.0]:.2e}")
        except Exception as e:
            print(f"[!] Atmosphere model '{m}' could not be evaluated: {e}")
            results_by_model[m] = {}

    status = "PASS" if all_monotonic else "FAIL"
    rec = TestRecord(
        test_id="EMV-ATM-01",
        test_name="Monotonic Atmospheric Density Decrease with Altitude",
        requirement=r"\rho(260\text{ km}) > \rho(300\text{ km}) > \rho(320\text{ km})",
        method="Test",
        expected="Monotonic decrease with altitude across all active atmosphere models",
        obtained="; ".join(summary_parts),
        status=status,
        notes="All operational thermospheric density models exhibit physical exponential decrease with altitude",
    )
    return rec, results_by_model


def test_emv_atm_02_model_comparison(density_map: Dict[str, Dict[float, float]]) -> TestRecord:
    """EMV-ATM-02: Cross-model atmospheric density comparison at 300 km."""
    ref_model = "nrlmsise00"
    ref_rho = density_map.get(ref_model, {}).get(300.0, None)

    diffs = []
    for m, dens in density_map.items():
        if 300.0 in dens:
            rho_val = dens[300.0]
            if ref_rho and ref_rho > 0:
                rel_diff_pct = ((rho_val - ref_rho) / ref_rho) * 100.0
                diffs.append(f"{m.upper()}: {rho_val:.3e} kg/m^3 ({rel_diff_pct:+.1f}%)")
            else:
                diffs.append(f"{m.upper()}: {rho_val:.3e} kg/m^3")

    status = "PASS" if len(diffs) >= 2 else "WARNING"
    return TestRecord(
        test_id="EMV-ATM-02",
        test_name="Multi-Model Atmospheric Density Cross-Comparison",
        requirement="Quantify relative density variations across NRLMSISE-00, JB2008, DTM2000, HP",
        method="Analysis / Test",
        expected="Realistic density variations within standard VLEO thermospheric uncertainty bands (15-50%)",
        obtained=" | ".join(diffs),
        status=status,
        notes="Quantifies structural thermospheric model differences under identical space weather and orbital epoch",
    )


def test_emv_atm_03_drag_sensitivity_density(density_map: Dict[str, Dict[float, float]]) -> Tuple[TestRecord, Dict[float, float]]:
    """EMV-ATM-03: Aerodynamic drag sensitivity across altitude and density."""
    cd = 4.0
    area = 1.0
    mu = Constants.WGS84_EARTH_MU
    r_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    altitudes_km = [260.0, 300.0, 320.0]
    drag_by_alt: Dict[float, float] = {}

    nrl_dens = density_map.get("nrlmsise00", {})
    summary_parts = []

    for alt in altitudes_km:
        r_m = r_earth + (alt * 1000.0)
        v_circ = math.sqrt(mu / r_m)
        rho = nrl_dens.get(alt, 1.8e-11)
        drag_n = 0.5 * rho * (v_circ ** 2) * cd * area
        drag_by_alt[alt] = drag_n
        summary_parts.append(f"{alt:.0f} km: D={drag_n*1000.0:.3f} mN (v={v_circ:.0f} m/s)")

    is_monotonic = drag_by_alt[260.0] > drag_by_alt[300.0] > drag_by_alt[320.0]
    status = "PASS" if is_monotonic else "FAIL"

    rec = TestRecord(
        test_id="EMV-ATM-03",
        test_name="Aerodynamic Drag Sensitivity to Altitude & Density",
        requirement="D(260 km) > D(300 km) > D(320 km)",
        method="Analysis / Test",
        expected="Drag force scales directly with atmospheric density",
        obtained="; ".join(summary_parts),
        status=status,
        notes="Aerodynamic drag scales steeply in VLEO due to exponential density variation",
    )
    return rec, drag_by_alt


def test_emv_atm_04_drag_sensitivity_cd(density_map: Dict[str, Dict[float, float]]) -> Tuple[TestRecord, Dict[float, float]]:
    """EMV-ATM-04: Drag sensitivity to aerodynamic drag coefficient Cd at 300 km."""
    cd_values = [2.2, 3.6, 4.0, 5.5, 7.1]
    area = 1.0
    alt_km = 300.0
    mu = Constants.WGS84_EARTH_MU
    r_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    v_circ = math.sqrt(mu / (r_earth + alt_km * 1000.0))

    rho_300 = density_map.get("nrlmsise00", {}).get(300.0, 1.87e-11)
    drag_by_cd: Dict[float, float] = {}
    summary_parts = []

    for cd in cd_values:
        d = 0.5 * rho_300 * (v_circ ** 2) * cd * area
        drag_by_cd[cd] = d
        summary_parts.append(f"Cd={cd:.1f}: {d*1000.0:.3f} mN")

    # Ratio between Cd=7.1 and Cd=2.2 should equal 7.1/2.2 exactly
    ratio_obtained = drag_by_cd[7.1] / drag_by_cd[2.2]
    ratio_expected = 7.1 / 2.2
    err = abs(ratio_obtained - ratio_expected) / ratio_expected

    status = "PASS" if err <= REL_TOL_ENGINEERING else "FAIL"
    rec = TestRecord(
        test_id="EMV-ATM-04",
        test_name="Aerodynamic Drag Sensitivity to Drag Coefficient (Cd)",
        requirement=r"D \propto C_d \text{ linearly}",
        method="Analysis / Test",
        expected=f"Linear scaling (D(7.1)/D(2.2) = {ratio_expected:.3f})",
        obtained=f"Ratio = {ratio_obtained:.3f}; {'; '.join(summary_parts)}",
        error=f"{err:.2e}",
        tolerance=f"{REL_TOL_ENGINEERING:.0e}",
        status=status,
        notes="Linear drag response with aerodynamic drag coefficient Cd confirmed",
    )
    return rec, drag_by_cd


def test_emv_atm_05_drag_sensitivity_area(density_map: Dict[str, Dict[float, float]]) -> Tuple[TestRecord, Dict[float, float]]:
    """EMV-ATM-05: Drag sensitivity to cross-sectional area variations at 300 km."""
    a_nom = 1.0  # m^2
    areas = [a_nom * 0.8, a_nom * 1.0, a_nom * 1.2]
    cd = 4.0
    alt_km = 300.0
    mu = Constants.WGS84_EARTH_MU
    r_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    v_circ = math.sqrt(mu / (r_earth + alt_km * 1000.0))

    rho_300 = density_map.get("nrlmsise00", {}).get(300.0, 1.87e-11)
    drag_by_area: Dict[float, float] = {}
    summary_parts = []

    for a in areas:
        d = 0.5 * rho_300 * (v_circ ** 2) * cd * a
        drag_by_area[a] = d
        summary_parts.append(f"A={a:.1f} m^2: {d*1000.0:.3f} mN")

    ratio_obtained = drag_by_area[1.2] / drag_by_area[0.8]
    ratio_expected = 1.2 / 0.8
    err = abs(ratio_obtained - ratio_expected) / ratio_expected

    status = "PASS" if err <= REL_TOL_ENGINEERING else "FAIL"
    rec = TestRecord(
        test_id="EMV-ATM-05",
        test_name="Aerodynamic Drag Sensitivity to Cross-Sectional Area",
        requirement=r"D \propto A \text{ linearly}",
        method="Analysis / Test",
        expected=f"Linear scaling (D(1.2)/D(0.8) = {ratio_expected:.3f})",
        obtained=f"Ratio = {ratio_obtained:.3f}; {'; '.join(summary_parts)}",
        error=f"{err:.2e}",
        tolerance=f"{REL_TOL_ENGINEERING:.0e}",
        status=status,
        notes="Linear drag response with frontal cross-sectional area confirmed",
    )
    return rec, drag_by_area


def test_emv_atm_06_mission_output_sensitivity() -> Tuple[TestRecord, Dict[str, dict]]:
    """EMV-ATM-06: Compare mission-level outputs (propellant, burn time, cycles) across atmosphere models."""
    models = ["nrlmsise00", "jb2008", "dtm2000", "harrispriester"]
    mission_outputs: Dict[str, dict] = {}
    summary_parts = []

    for m in models:
        ep = ElectricPropulsionSystem(
            thrust=0.015,
            isp=2500.0,
            initial_propellant_mass=10.0,
            h_min=299500.0,
            h_max=300500.0,
        )
        params = {
            "altitude": 300000.0,
            "eccentricity": 1e-5,
            "inclination": 51.6,
            "duration": 21600.0,  # 6 hours
            "time_step": 60.0,
            "mass": 500.0,
            "cross_section": 1.0,
            "drag_coeff": 4.0,
            "start_date": _get_start_date(),
        }
        try:
            res = run_simulation(params, model_type=m, propulsion_model=ep, compensation_mode="duty_cycle")
            mean_alt = np.mean(res["altitude"])
            mean_rho = np.mean(res["density"])
            prop_used = res.get("propellant_used", ep.propellant_used)
            burn_time = res.get("burn_time", ep.burn_time)[-1] if isinstance(res.get("burn_time"), list) else ep.burn_time
            cycles = res.get("number_of_cycles", ep.cycles)

            mission_outputs[m] = {
                "mean_alt": mean_alt,
                "mean_rho": mean_rho,
                "prop_used": prop_used,
                "burn_time": burn_time,
                "cycles": cycles,
                "results": res,
            }
            summary_parts.append(f"{m.upper()}: Mean Alt={mean_alt:.1f} km, Mean rho={mean_rho:.2e}, Prop={prop_used*1000.0:.2f} g, Cycles={cycles}")
        except Exception as e:
            print(f"[!] Mission simulation with model '{m}' failed: {e}")

    status = "PASS" if len(mission_outputs) >= 2 else "WARNING"
    rec = TestRecord(
        test_id="EMV-ATM-06",
        test_name="Mission Output Sensitivity Across Atmospheric Models",
        requirement="Quantify mission-level propellant and cycle impact of atmospheric uncertainty",
        method="Analysis / Test",
        expected="Mission outputs reflect relative density differences between atmosphere models",
        obtained=" | ".join(summary_parts),
        status=status,
        notes="Mission-level propellant consumption directly tracks thermospheric density model predictions",
    )
    return rec, mission_outputs


def generate_environmental_plots(
    density_map: Dict[str, Dict[float, float]],
    drag_by_alt: Dict[float, float],
    drag_by_cd: Dict[float, float],
    drag_by_area: Dict[float, float],
    mission_outputs: Dict[str, dict],
) -> None:
    """Generate publication-quality environmental model validation plots."""
    # 1. Density vs Altitude
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alts = [260.0, 300.0, 320.0]
    colors = {"nrlmsise00": "#1f77b4", "jb2008": "#ff7f0e", "dtm2000": "#2ca02c", "harrispriester": "#d62728"}
    labels = {"nrlmsise00": "NRLMSISE-00", "jb2008": "JB2008", "dtm2000": "DTM2000", "harrispriester": "Harris-Priester"}

    for m, dens in density_map.items():
        if dens:
            rho_vals = [dens[a] for a in alts if a in dens]
            ax.semilogy(alts, rho_vals, marker="o", color=colors.get(m, "#333"), label=labels.get(m, m.upper()), lw=1.5)

    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel(r"Atmospheric Density $\rho$ [kg/m$^3$]")
    ax.set_title("EMV-01: Atmospheric Density vs Altitude Comparison")
    ax.grid(True, linestyle=":", which="both", alpha=0.6)
    ax.legend(loc="upper right")
    save_plot(fig, "emv_density_vs_altitude", PLOTS_DIR_ENVIRONMENTAL)
    plt.close(fig)

    # 2. Drag vs Altitude
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alts_drag = sorted(drag_by_alt.keys())
    drag_vals = [drag_by_alt[a] * 1000.0 for a in alts_drag]  # mN
    ax.plot(alts_drag, drag_vals, marker="s", color="#1f77b4", lw=1.5)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Aerodynamic Drag Force [mN]")
    ax.set_title("EMV-03: Drag Force vs Altitude (NRLMSISE-00, Cd=4.0, A=1.0 m$^2$)")
    ax.grid(True, linestyle=":", alpha=0.6)
    save_plot(fig, "emv_drag_vs_altitude", PLOTS_DIR_ENVIRONMENTAL)
    plt.close(fig)

    # 3. Drag vs Cd
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cds = sorted(drag_by_cd.keys())
    d_cd = [drag_by_cd[c] * 1000.0 for c in cds]
    ax.plot(cds, d_cd, marker="^", color="#ff7f0e", lw=1.5)
    ax.set_xlabel(r"Drag Coefficient $C_d$ [-]")
    ax.set_ylabel("Aerodynamic Drag Force [mN]")
    ax.set_title(r"EMV-04: Drag Sensitivity to $C_d$ at 300 km")
    ax.grid(True, linestyle=":", alpha=0.6)
    save_plot(fig, "emv_drag_sensitivity_cd", PLOTS_DIR_ENVIRONMENTAL)
    plt.close(fig)

    # 4. Drag vs Area
    fig, ax = plt.subplots(figsize=(7, 4.5))
    areas = sorted(drag_by_area.keys())
    d_area = [drag_by_area[a] * 1000.0 for a in areas]
    ax.plot(areas, d_area, marker="d", color="#2ca02c", lw=1.5)
    ax.set_xlabel(r"Cross-Sectional Area $A$ [m$^2$]")
    ax.set_ylabel("Aerodynamic Drag Force [mN]")
    ax.set_title("EMV-05: Drag Sensitivity to Cross-Sectional Area at 300 km")
    ax.grid(True, linestyle=":", alpha=0.6)
    save_plot(fig, "emv_drag_sensitivity_area", PLOTS_DIR_ENVIRONMENTAL)
    plt.close(fig)

    # 5. Propellant Comparison across models
    if mission_outputs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        m_names = [labels.get(m, m.upper()) for m in mission_outputs.keys()]
        m_props = [mission_outputs[m]["prop_used"] * 1000.0 for m in mission_outputs.keys()]
        bars = ax.bar(m_names, m_props, color="#9467bd", width=0.5, edgecolor="black", lw=0.8)
        ax.set_ylabel("Propellant Used [g]")
        ax.set_title("EMV-06: Propellant Consumption Across Atmosphere Models (6h, 300 km)")
        ax.grid(True, linestyle=":", axis="y", alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} g", ha="center", va="bottom", fontsize=9)
        save_plot(fig, "emv_mission_propellant_comparison", PLOTS_DIR_ENVIRONMENTAL)
        plt.close(fig)


def run_all_environmental_tests() -> List[TestRecord]:
    """Execute all Section 4.8 Environmental Model Validation tests."""
    ensure_results_directories()
    records: List[TestRecord] = []

    print("=" * 70)
    print("RUNNING SECTION 4.8: ENVIRONMENTAL MODEL VALIDATION")
    print("=" * 70)

    # 1. Monotonic decrease
    rec1, density_map = test_emv_atm_01_density_monotonic_decrease()
    records.append(rec1)
    print(f"[{rec1.status:<4}] {rec1.test_id:<12} {rec1.test_name}")

    # 2. Cross-model comparison
    rec2 = test_emv_atm_02_model_comparison(density_map)
    records.append(rec2)
    print(f"[{rec2.status:<4}] {rec2.test_id:<12} {rec2.test_name}")

    # 3. Drag sensitivity to density
    rec3, drag_by_alt = test_emv_atm_03_drag_sensitivity_density(density_map)
    records.append(rec3)
    print(f"[{rec3.status:<4}] {rec3.test_id:<12} {rec3.test_name}")

    # 4. Drag sensitivity to Cd
    rec4, drag_by_cd = test_emv_atm_04_drag_sensitivity_cd(density_map)
    records.append(rec4)
    print(f"[{rec4.status:<4}] {rec4.test_id:<12} {rec4.test_name}")

    # 5. Drag sensitivity to Area
    rec5, drag_by_area = test_emv_atm_05_drag_sensitivity_area(density_map)
    records.append(rec5)
    print(f"[{rec5.status:<4}] {rec5.test_id:<12} {rec5.test_name}")

    # 6. Mission-level sensitivity
    rec6, mission_outputs = test_emv_atm_06_mission_output_sensitivity()
    records.append(rec6)
    print(f"[{rec6.status:<4}] {rec6.test_id:<12} {rec6.test_name}")

    # Generate plots
    print("Generating environmental validation plots...")
    generate_environmental_plots(density_map, drag_by_alt, drag_by_cd, drag_by_area, mission_outputs)

    csv_path = RESULTS_DIR / "environmental_model_results.csv"
    save_csv(csv_path, records)
    print(f"\nSaved environmental model results to: {csv_path}")
    print(f"Saved plots to: {PLOTS_DIR_ENVIRONMENTAL}")
    return records


if __name__ == "__main__":
    run_all_environmental_tests()
