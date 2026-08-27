"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          test_45_unit_verification.py
Description:
    Section 4.5: Unit verification test suite for Electric Propulsion and Electrical Power Subsystems.
===============================================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

# Ensure verification_config and project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verification_config import (
    G0,
    REL_TOL_ENGINEERING,
    REL_TOL_STRICT,
    RESULTS_DIR,
    TestRecord,
    ensure_results_directories,
    save_csv,
)

from orbital_models.electric_propulsion import ElectricPropulsionSystem
from orbital_models.power_subsystem import PowerSubsystem


def test_uv_ep_01_mass_flow_rate() -> TestRecord:
    """UV-EP-01: Verify mass flow rate analytical calculation."""
    thrust = 0.015      # [N]
    isp = 2500.0        # [s]
    g0 = 9.80665        # [m/s^2]
    expected_mdot = thrust / (isp * g0)

    ep = ElectricPropulsionSystem(
        thrust=thrust,
        isp=isp,
        initial_propellant_mass=10.0,
        h_min=290000.0,
        h_max=310000.0,
    )
    obtained_mdot = ep.compute_mass_flow_rate()
    rel_error = abs(obtained_mdot - expected_mdot) / expected_mdot

    status = "PASS" if rel_error <= REL_TOL_STRICT else "FAIL"
    assert rel_error <= REL_TOL_STRICT, f"mdot mismatch: {obtained_mdot} vs {expected_mdot}"

    return TestRecord(
        test_id="UV-EP-01",
        test_name="EP Mass Flow Rate Analytical Check",
        requirement=r"\dot{m} = T / (I_{sp} \cdot g_0)",
        method="Analysis / Test",
        expected=f"{expected_mdot:.8e} kg/s",
        obtained=f"{obtained_mdot:.8e} kg/s",
        error=f"{rel_error:.2e}",
        tolerance=f"{REL_TOL_STRICT:.0e}",
        status=status,
        notes="Exact agreement with T/(Isp*g0) mass flow rate formula",
    )


def test_uv_ep_02_propellant_consumption() -> TestRecord:
    """UV-EP-02: Propellant consumption over a fixed burn duration."""
    thrust = 0.015
    isp = 2500.0
    dt_sec = 100.0
    ep = ElectricPropulsionSystem(
        thrust=thrust,
        isp=isp,
        initial_propellant_mass=10.0,
        h_min=295000.0,
        h_max=305000.0,
    )
    # Trigger burn by setting altitude below h_min (e.g. 290 km)
    is_on = ep.update(current_altitude_km=290.0, dt_sec=dt_sec, power_ok=True)
    expected_consumed = (thrust / (isp * G0)) * dt_sec
    obtained_consumed = ep.propellant_used
    rel_error = abs(obtained_consumed - expected_consumed) / expected_consumed

    status = "PASS" if (is_on and rel_error <= REL_TOL_ENGINEERING) else "FAIL"
    assert is_on and rel_error <= REL_TOL_ENGINEERING

    return TestRecord(
        test_id="UV-EP-02",
        test_name="EP Fixed-Burn Propellant Consumption",
        requirement=r"\Delta m = \dot{m} \cdot \Delta t",
        method="Analysis / Test",
        expected=f"{expected_consumed:.8e} kg",
        obtained=f"{obtained_consumed:.8e} kg",
        error=f"{rel_error:.2e}",
        tolerance=f"{REL_TOL_ENGINEERING:.0e}",
        status=status,
        notes="Verified 100s burn propellant consumption and state update",
    )


def test_uv_ep_03_altitude_threshold_logic() -> TestRecord:
    """UV-EP-03: Thruster ON/OFF altitude threshold switching logic."""
    h_min_m = 295000.0
    h_max_m = 305000.0
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=10.0,
        h_min=h_min_m,
        h_max=h_max_m,
    )

    # 1. Start above h_min (e.g. 300 km) -> should be OFF
    s1 = ep.update(current_altitude_km=300.0, dt_sec=10.0)
    # 2. Descend below h_min (e.g. 294 km) -> should turn ON
    s2 = ep.update(current_altitude_km=294.0, dt_sec=10.0)
    # 3. Climb inside hysteresis band (e.g. 300 km) -> should stay ON
    s3 = ep.update(current_altitude_km=300.0, dt_sec=10.0)
    # 4. Climb above h_max (e.g. 306 km) -> should turn OFF
    s4 = ep.update(current_altitude_km=306.0, dt_sec=10.0)

    logic_ok = (not s1) and s2 and s3 and (not s4)
    status = "PASS" if logic_ok else "FAIL"
    assert logic_ok

    return TestRecord(
        test_id="UV-EP-03",
        test_name="EP Altitude Threshold ON/OFF Logic",
        requirement="ON when h < h_min; OFF when h > h_max; latching in-between",
        method="Test",
        expected="s1=OFF, s2=ON, s3=ON, s4=OFF",
        obtained=f"s1={'ON' if s1 else 'OFF'}, s2={'ON' if s2 else 'OFF'}, s3={'ON' if s3 else 'OFF'}, s4={'ON' if s4 else 'OFF'}",
        status=status,
        notes="Hysteresis altitude threshold controller correctly latches ON/OFF",
    )


def test_uv_ep_04_propellant_depletion_shutdown() -> TestRecord:
    """UV-EP-04: Thruster shutdown on propellant depletion."""
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=1e-5,  # very small propellant mass
        h_min=295000.0,
        h_max=305000.0,
    )
    # Force burn with dt=100s
    ep.update(current_altitude_km=290.0, dt_sec=100.0)
    # Next step should be OFF and shutdown reason set
    is_on_next = ep.update(current_altitude_km=290.0, dt_sec=10.0)
    reason = ep.shutdown_reason

    pass_criteria = (not is_on_next) and (ep.propellant_mass <= 0.0) and ("Propellant Depleted" in reason)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-EP-04",
        test_name="EP Propellant Depletion Shutdown",
        requirement="Automatic cutoff when propellant_mass == 0",
        method="Test",
        expected="is_on=False, shutdown_reason='Propellant Depleted'",
        obtained=f"is_on={is_on_next}, shutdown_reason='{reason}'",
        status=status,
        notes="Thruster cuts off and sets shutdown_reason upon fuel exhaustion",
    )


def test_uv_ep_05_max_burn_time_shutdown() -> TestRecord:
    """UV-EP-05: Cutoff when maximum burn time is reached."""
    max_burn = 50.0  # seconds
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=10.0,
        h_min=295000.0,
        h_max=305000.0,
        max_burn_time=max_burn,
    )
    # Burn 40s (within limit)
    ep.update(current_altitude_km=290.0, dt_sec=40.0)
    # Burn another 20s (exceeds 50s total limit)
    ep.update(current_altitude_km=290.0, dt_sec=20.0)
    # Subsequent update must be rejected
    is_on_subseq = ep.update(current_altitude_km=290.0, dt_sec=10.0)

    pass_criteria = (not is_on_subseq) and ("Max Burn Time Reached" in ep.shutdown_reason)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-EP-05",
        test_name="EP Max Burn Time Limit",
        requirement="Cutoff when burn_time >= max_burn_time",
        method="Test",
        expected=f"Cutoff at {max_burn}s, shutdown_reason='Max Burn Time Reached'",
        obtained=f"burn_time={ep.burn_time:.1f}s, shutdown_reason='{ep.shutdown_reason}'",
        status=status,
        notes="Maximum burn time safety limit enforced",
    )


def test_uv_ep_06_max_cycles_limit() -> TestRecord:
    """UV-EP-06: Thruster inhibits new cycle start once max_cycles is reached."""
    max_c = 2
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=10.0,
        h_min=295000.0,
        h_max=305000.0,
        max_cycles=max_c,
    )

    # Cycle 1: turn on at 290 km, turn off at 310 km
    ep.update(current_altitude_km=290.0, dt_sec=10.0)
    ep.update(current_altitude_km=310.0, dt_sec=10.0)

    # Cycle 2: turn on at 290 km, turn off at 310 km
    ep.update(current_altitude_km=290.0, dt_sec=10.0)
    ep.update(current_altitude_km=310.0, dt_sec=10.0)

    # Attempt Cycle 3: turn on at 290 km -> should remain OFF
    s3 = ep.update(current_altitude_km=290.0, dt_sec=10.0)

    # Note: turn_off only sets shutdown_reason if is_on was True. Since is_on was False,
    # shutdown_reason remains 'Reached Target Altitude' from cycle 2, but cycle 3 is prevented.
    pass_criteria = (not s3) and (ep.cycles == max_c)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-EP-06",
        test_name="EP Maximum Cycles Limit",
        requirement=f"Inhibit ignition when cycles >= max_cycles ({max_c})",
        method="Test",
        expected=f"cycles={max_c}, is_on=False",
        obtained=f"cycles={ep.cycles}, is_on={s3}, shutdown_reason='{ep.shutdown_reason}'",
        status=status,
        notes="Cycle limit strictly prevented additional thruster ignitions",
    )


def test_uv_ep_07_power_override() -> TestRecord:
    """UV-EP-07: External power_ok=False override forces thruster OFF."""
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=10.0,
        h_min=295000.0,
        h_max=305000.0,
    )
    # Normal ON request with altitude=290km, but power_ok=False
    is_on = ep.update(current_altitude_km=290.0, dt_sec=10.0, power_ok=False)
    prop_used = ep.propellant_used

    pass_criteria = (not is_on) and (prop_used == 0.0)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-EP-07",
        test_name="EP Power Subsystem Override Check",
        requirement="Force thruster OFF when power_ok=False",
        method="Test",
        expected="is_on=False, propellant_used=0.0 kg",
        obtained=f"is_on={is_on}, propellant_used={prop_used:.1f} kg",
        status=status,
        notes="Power override successfully inhibits thrust and propellant consumption",
    )


def test_uv_ep_08_duty_cycle_property_review() -> TestRecord:
    """UV-EP-08: Review of Design on duty_cycle property semantics."""
    ep = ElectricPropulsionSystem(
        thrust=0.015,
        isp=2500.0,
        initial_propellant_mass=10.0,
        h_min=295000.0,
        h_max=305000.0,
    )
    ep.update(current_altitude_km=290.0, dt_sec=120.0)
    dc_value = ep.duty_cycle

    # Finding: duty_cycle returns burn_time (seconds) rather than percentage fraction (0..1)
    is_seconds = (dc_value == 120.0)

    return TestRecord(
        test_id="UV-EP-08",
        test_name="EP duty_cycle Property Semantics Review",
        requirement="Inspection of duty_cycle property units and semantics",
        method="Review of Design",
        expected="Dimensionless fraction (burn_time / total_time)",
        obtained=f"Returns burn_time in seconds ({dc_value:.1f} s)",
        status="WARNING",
        notes="Finding: ElectricPropulsionSystem.duty_cycle returns accumulated burn time [s]. Recommendation: rename to burn_time_s or divide by mission elapsed time.",
    )


def test_uv_pwr_01_full_sun_generation() -> TestRecord:
    """UV-PWR-01: Solar generation in full sunlight."""
    area = 2.0
    eff = 0.30
    flux = 1361.0
    expected_p_solar = area * eff * flux  # 816.6 W

    pwr = PowerSubsystem(
        solar_panel_area_m2=area,
        panel_efficiency=eff,
        solar_flux_W_m2=flux,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=300.0,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
    )
    allowed, p_gen, p_cons = pwr.update(illumination_fraction=1.0, thruster_requesting=False, dt_s=10.0)
    rel_error = abs(p_gen - expected_p_solar) / expected_p_solar

    status = "PASS" if rel_error <= REL_TOL_STRICT else "FAIL"
    assert rel_error <= REL_TOL_STRICT

    return TestRecord(
        test_id="UV-PWR-01",
        test_name="Power Full-Sun Generation",
        requirement=r"P_{solar} = A \cdot \eta \cdot \Phi_{solar}",
        method="Analysis / Test",
        expected=f"{expected_p_solar:.2f} W",
        obtained=f"{p_gen:.2f} W",
        error=f"{rel_error:.2e}",
        tolerance=f"{REL_TOL_STRICT:.0e}",
        status=status,
        notes="Full sunlight power generation conforms exactly to analytical formula",
    )


def test_uv_pwr_02_penumbra_generation() -> TestRecord:
    """UV-PWR-02: Solar generation in penumbra (50% illumination)."""
    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=300.0,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
    )
    expected_p = 0.5 * (2.0 * 0.30 * 1361.0)  # 408.3 W
    _, p_gen, _ = pwr.update(illumination_fraction=0.5, thruster_requesting=False, dt_s=10.0)
    rel_error = abs(p_gen - expected_p) / expected_p

    status = "PASS" if rel_error <= REL_TOL_STRICT else "FAIL"
    assert rel_error <= REL_TOL_STRICT

    return TestRecord(
        test_id="UV-PWR-02",
        test_name="Power Penumbra Generation",
        requirement="Linear scaling with illumination fraction = 0.5",
        method="Analysis / Test",
        expected=f"{expected_p:.2f} W",
        obtained=f"{p_gen:.2f} W",
        error=f"{rel_error:.2e}",
        tolerance=f"{REL_TOL_STRICT:.0e}",
        status=status,
        notes="Penumbra linear scaling matches 50% solar flux",
    )


def test_uv_pwr_03_umbra_generation() -> TestRecord:
    """UV-PWR-03: Solar generation in full umbra (0.0 illumination)."""
    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=300.0,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
    )
    _, p_gen, _ = pwr.update(illumination_fraction=0.0, thruster_requesting=False, dt_s=10.0)

    status = "PASS" if p_gen == 0.0 else "FAIL"
    assert p_gen == 0.0

    return TestRecord(
        test_id="UV-PWR-03",
        test_name="Power Umbra Generation",
        requirement="P_solar == 0 W in full shadow",
        method="Test",
        expected="0.00 W",
        obtained=f"{p_gen:.2f} W",
        error="0.00",
        tolerance="0.00",
        status=status,
        notes="Zero solar generation in umbra confirmed",
    )


def test_uv_pwr_04_battery_energy_balance() -> TestRecord:
    """UV-PWR-04: Battery energy balance integration."""
    initial_wh = 150.0
    hk_power = 50.0
    dt_s = 3600.0  # 1 hour
    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=initial_wh,
        housekeeping_power_W=hk_power,
        thruster_power_W=250.0,
    )
    # Eclipse discharge (0 solar generation)
    pwr.update(illumination_fraction=0.0, thruster_requesting=False, dt_s=dt_s)
    expected_wh = initial_wh - (hk_power * dt_s / 3600.0)  # 150 - 50 = 100 Wh
    obtained_wh = pwr.battery_Wh
    error = abs(obtained_wh - expected_wh)

    status = "PASS" if error <= REL_TOL_ENGINEERING else "FAIL"
    assert error <= REL_TOL_ENGINEERING

    return TestRecord(
        test_id="UV-PWR-04",
        test_name="Power Battery Energy Balance",
        requirement=r"\Delta E_{Wh} = (P_{gen} - P_{cons}) \cdot \Delta t / 3600",
        method="Analysis / Test",
        expected=f"{expected_wh:.2f} Wh",
        obtained=f"{obtained_wh:.2f} Wh",
        error=f"{error:.2e} Wh",
        tolerance=f"{REL_TOL_ENGINEERING:.0e}",
        status=status,
        notes="Energy balance integration over 1 hour eclipse matches analytical discharge",
    )


def test_uv_pwr_05_battery_lower_bound() -> TestRecord:
    """UV-PWR-05: Battery energy lower bound clamp (E >= 0 Wh)."""
    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=10.0,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
    )
    # Discharge for 10 hours in eclipse (would drain 500 Wh from a 10 Wh battery)
    pwr.update(illumination_fraction=0.0, thruster_requesting=False, dt_s=36000.0)
    obtained_wh = pwr.battery_Wh

    status = "PASS" if obtained_wh == 0.0 else "FAIL"
    assert obtained_wh == 0.0

    return TestRecord(
        test_id="UV-PWR-05",
        test_name="Power Battery Lower Bound Clamp",
        requirement="battery_Wh >= 0.0 Wh under excessive load",
        method="Test",
        expected="0.00 Wh",
        obtained=f"{obtained_wh:.2f} Wh",
        status=status,
        notes="Battery charge clamped strictly to 0 Wh without negative energy anomalies",
    )


def test_uv_pwr_06_battery_upper_bound() -> TestRecord:
    """UV-PWR-06: Battery energy upper bound clamp (E <= Capacity)."""
    cap_wh = 300.0
    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=cap_wh,
        battery_initial_Wh=290.0,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
    )
    # Charge in full sun for 10 hours (net ~766 W surplus = 7660 Wh)
    pwr.update(illumination_fraction=1.0, thruster_requesting=False, dt_s=36000.0)
    obtained_wh = pwr.battery_Wh

    status = "PASS" if obtained_wh == cap_wh else "FAIL"
    assert obtained_wh == cap_wh

    return TestRecord(
        test_id="UV-PWR-06",
        test_name="Power Battery Upper Bound Clamp",
        requirement="battery_Wh <= battery_capacity_Wh under prolonged charging",
        method="Test",
        expected=f"{cap_wh:.2f} Wh",
        obtained=f"{obtained_wh:.2f} Wh",
        status=status,
        notes="Battery charge clamped strictly at maximum usable capacity",
    )


def test_uv_pwr_07_thruster_power_inhibition() -> TestRecord:
    """UV-PWR-07: Thruster inhibition when battery is depleted and power is insufficient."""
    pwr = PowerSubsystem(
        solar_panel_area_m2=0.1,    # very small solar area -> ~40.8 W generation
        panel_efficiency=0.30,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=300.0,
        battery_initial_Wh=0.0,     # empty battery
        housekeeping_power_W=50.0,  # 50 W HK > 40.8 W solar
        thruster_power_W=250.0,     # total demand 300 W >> 40.8 W
    )
    thruster_allowed, p_gen, p_cons = pwr.update(
        illumination_fraction=1.0,
        thruster_requesting=True,
        dt_s=10.0,
    )

    pass_criteria = (not thruster_allowed) and (p_cons == 50.0)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-PWR-07",
        test_name="Power Thruster Inhibition on Deficit",
        requirement="thruster_allowed=False when battery=0 and P_gen < P_hk + P_thr",
        method="Test",
        expected="thruster_allowed=False, P_cons=50.0 W",
        obtained=f"thruster_allowed={thruster_allowed}, P_cons={p_cons:.1f} W",
        status=status,
        notes="Thruster successfully inhibited when platform is in power deficit",
    )


def test_uv_pwr_08_degradation_behaviour() -> TestRecord:
    """UV-PWR-08: Linear degradation of solar efficiency and battery capacity."""
    deg_panel = 0.05   # 5% per year
    deg_bat = 0.04     # 4% per year
    bol_eff = 0.30
    bol_cap = 300.0

    pwr = PowerSubsystem(
        solar_panel_area_m2=2.0,
        panel_efficiency=bol_eff,
        solar_flux_W_m2=1361.0,
        battery_capacity_Wh=bol_cap,
        battery_initial_Wh=bol_cap,
        housekeeping_power_W=50.0,
        thruster_power_W=250.0,
        panel_degradation_yr=deg_panel,
        battery_degradation_yr=deg_bat,
    )

    # Advance by 1 year (365.25 * 86400 s)
    one_year_s = 365.25 * 86400.0
    pwr.update(illumination_fraction=1.0, thruster_requesting=False, dt_s=one_year_s)

    expected_cap_eol = bol_cap * (1.0 - deg_bat)      # 288.0 Wh
    expected_p_solar = 2.0 * (bol_eff * (1.0 - deg_panel)) * 1361.0  # 775.77 W

    obtained_cap = pwr.battery_capacity_Wh
    obtained_p = pwr.P_solar_max

    err_cap = abs(obtained_cap - expected_cap_eol) / expected_cap_eol
    err_p = abs(obtained_p - expected_p_solar) / expected_p_solar
    pass_criteria = (err_cap <= REL_TOL_ENGINEERING) and (err_p <= REL_TOL_ENGINEERING)
    status = "PASS" if pass_criteria else "FAIL"
    assert pass_criteria

    return TestRecord(
        test_id="UV-PWR-08",
        test_name="Power Long-Term Degradation",
        requirement="Linear reduction in solar generation & battery capacity over time",
        method="Analysis / Test",
        expected=f"Cap={expected_cap_eol:.1f} Wh, P_max={expected_p_solar:.2f} W",
        obtained=f"Cap={obtained_cap:.1f} Wh, P_max={obtained_p:.2f} W",
        error=f"cap_err={err_cap:.2e}, p_err={err_p:.2e}",
        tolerance=f"{REL_TOL_ENGINEERING:.0e}",
        status=status,
        notes="Solar and battery multi-year linear degradation models verified",
    )


def run_all_unit_tests() -> List[TestRecord]:
    """Execute all Section 4.5 Unit Verification tests and return records."""
    ensure_results_directories()
    test_functions = [
        test_uv_ep_01_mass_flow_rate,
        test_uv_ep_02_propellant_consumption,
        test_uv_ep_03_altitude_threshold_logic,
        test_uv_ep_04_propellant_depletion_shutdown,
        test_uv_ep_05_max_burn_time_shutdown,
        test_uv_ep_06_max_cycles_limit,
        test_uv_ep_07_power_override,
        test_uv_ep_08_duty_cycle_property_review,
        test_uv_pwr_01_full_sun_generation,
        test_uv_pwr_02_penumbra_generation,
        test_uv_pwr_03_umbra_generation,
        test_uv_pwr_04_battery_energy_balance,
        test_uv_pwr_05_battery_lower_bound,
        test_uv_pwr_06_battery_upper_bound,
        test_uv_pwr_07_thruster_power_inhibition,
        test_uv_pwr_08_degradation_behaviour,
    ]

    records: List[TestRecord] = []
    print("=" * 70)
    print("RUNNING SECTION 4.5: UNIT VERIFICATION (EP & POWER)")
    print("=" * 70)

    for fn in test_functions:
        try:
            rec = fn()
            records.append(rec)
            status_str = f"[{rec.status}]"
            print(f"{status_str:<10} {rec.test_id:<12} {rec.test_name}")
        except Exception as e:
            rec = TestRecord(
                test_id="ERROR",
                test_name=fn.__name__,
                requirement="Unit test execution",
                method="Test",
                expected="Clean execution",
                obtained=f"Exception: {e}",
                status="FAIL",
                notes=str(e),
            )
            records.append(rec)
            print(f"{'[FAIL]':<10} {fn.__name__:<12} Exception: {e}")

    csv_path = RESULTS_DIR / "unit_verification_results.csv"
    save_csv(csv_path, records)
    print(f"\nSaved unit verification results to: {csv_path}")
    return records


if __name__ == "__main__":
    run_all_unit_tests()
