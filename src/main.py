"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          main.py
Description:
    Main CLI entry point for orbital decay and drag compensation simulations.
===============================================================================
"""

import sys
import os

# === OREKIT INITIALIZATION ===
# The JVM must be started before importing any model that uses Orekit.
import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception as e:
    print(f"Warning starting JVM (it might be already running): {e}")

from orekit_jpype.pyhelpers import setup_orekit_data
# Ensure the correct path to 'orekit-data-main'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
setup_orekit_data(filenames=os.path.join(project_root, "orekit-data-main"), from_pip_library=False)
# ===============================

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orbital_models import atmospheric_model
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from time import perf_counter

t0 = perf_counter()

# ============================================================
# HELPERS
# ============================================================

def _print_simulation_summary(params: dict, model_name: str, compensation_mode,
                              thruster_params=None):
    """Print a formatted table of orbital, spacecraft, thruster and power parameters."""
    # Import power config lazily so the function stays self-contained
    try:
        from mission_config import POWER
    except ImportError:
        POWER = {}

    duration_days = params["duration"] / 86400.0
    alt_km        = params["altitude"] / 1000.0
    comp_str      = compensation_mode if compensation_mode else "N/A (orbital decay only)"

    sep = "-" * 55
    print()
    print("=" * 55)
    print("          SIMULATION PARAMETERS SUMMARY")
    print("=" * 55)

    print("  ORBITAL PARAMETERS")
    print(sep)
    print(f"  {'Start date':<30}: {params['start_date'].toString().split('T')[0]}")
    print(f"  {'Initial altitude':<30}: {alt_km:.1f} km")
    print(f"  {'Inclination':<30}: {params['inclination']:.1f} deg")
    print(f"  {'Eccentricity':<30}: {params['eccentricity']:.2e}")
    print(f"  {'Duration':<30}: {duration_days:.1f} days")
    print(f"  {'Time step':<30}: {params['time_step']:.0f} s")

    print()
    print("  SPACECRAFT PARAMETERS")
    print(sep)
    print(f"  {'Dry mass':<30}: {params['mass']:.1f} kg")
    print(f"  {'Cross-section':<30}: {params['cross_section']:.2f} m\u00b2")
    print(f"  {'Drag coefficient (Cd)':<30}: {params['drag_coeff']:.2f}")

    if thruster_params:
        # Fetch thruster identity from mission_config (name + operating-point label)
        try:
            from mission_config import PROPULSION as _PROP
            thruster_name  = _PROP.get("thruster_name",  "Unknown")
            thruster_label = _PROP.get("thruster_label", "")
        except ImportError:
            thruster_name  = "Unknown"
            thruster_label = ""

        print()
        print("  PROPULSION SYSTEM")
        print(sep)
        thrust_mN = float(thruster_params.get("thrust", 0.0)) * 1e3
        isp       = float(thruster_params.get("isp", 0.0))
        power_W   = float(thruster_params.get("power_W", 0.0))
        prop_kg   = float(thruster_params.get("propellant_mass", 0.0))
        h_min     = thruster_params.get("h_min_km", "N/A")
        h_max     = thruster_params.get("h_max_km", "N/A")
        name_str  = f"{thruster_name}  [{thruster_label}]" if thruster_label else thruster_name
        print(f"  {'Thruster':<30}: {name_str}")
        print(f"  {'Thrust':<30}: {thrust_mN:.1f} mN")
        print(f"  {'Specific impulse (Isp)':<30}: {isp:.0f} s")
        if power_W > 0:
            print(f"  {'Thruster power draw':<30}: {power_W:.1f} W")
        print(f"  {'Propellant mass':<30}: {prop_kg:.1f} kg")
        print(f"  {'h_min (thruster ON below)':<30}: {h_min} km")
        print(f"  {'h_max (thruster OFF above)':<30}: {h_max} km")

    if POWER:
        print()
        print("  POWER SUBSYSTEM PARAMETERS")
        print(sep)
        print(f"  {'Solar panel area':<30}: {POWER.get('solar_panel_area_m2', 'N/A')} m\u00b2")
        print(f"  {'Panel efficiency (BOL)':<30}: {POWER.get('panel_efficiency', 'N/A') * 100:.1f} %")
        print(f"  {'Solar flux':<30}: {POWER.get('solar_flux_W_m2', 'N/A'):.0f} W/m\u00b2")
        print(f"  {'Panel degradation':<30}: {POWER.get('solar_panel_degradation_per_year', 0.0) * 100:.1f} %/yr")
        print(f"  {'Battery capacity':<30}: {POWER.get('battery_capacity_Wh', 'N/A'):.1f} Wh")
        print(f"  {'Battery initial SoC':<30}: {POWER.get('battery_initial_Wh', 'N/A'):.1f} Wh")
        print(f"  {'Battery degradation':<30}: {POWER.get('battery_degradation_per_year', 0.0) * 100:.1f} %/yr")
        print(f"  {'Housekeeping power':<30}: {POWER.get('housekeeping_power_W', 'N/A'):.1f} W")

    print()
    print("  SIMULATION SETTINGS")
    print(sep)
    print(f"  {'Atmospheric model':<30}: {model_name.upper()}")
    print(f"  {'Compensation mode':<30}: {comp_str}")
    print("=" * 55)
    print()


def _confirm_launch() -> bool:
    """Ask the user to confirm before starting the simulation."""
    while True:
        answer = input("Proceed with simulation? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        print("Launching GUI...")
        import subprocess
        gui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", gui_path])
        return

    print("=== Orbital Simulation ===")
    print("💡 Tip: You can run the interactive web GUI by executing:")
    print("   .\\orekit_env\\Scripts\\python.exe src\\main.py --gui")
    print("   or directly using Streamlit:")
    print("   .\\orekit_env\\Scripts\\streamlit.exe run src\\gui.py\n")


    # ============================================================
    # LEVEL 1 CHOICE: Simulation type
    # ============================================================
    print("Select simulation type:")
    print("1. Orbital Decay Simulation")
    print("2. Orbital Decay + Drag Compensation")
    sim_type = input("Enter your choice (1/2): ").strip()
    if sim_type not in ("1", "2"):
        print("Invalid choice. Exiting.")
        return

    # ============================================================
    # LEVEL 2 CHOICE: Mode
    # ============================================================
    print("\nSelect mode:")
    print("1. Standalone Simulation (No TLE data)")
    print("2. Validation Simulation (Against TLE data)")
    mode_choice = input("Enter your choice (1/2): ").strip()
    if mode_choice not in ("1", "2"):
        print("Invalid choice. Exiting.")
        return

    # ============================================================
    # LEVEL 3 CHOICE: Atmospheric Model
    # When Drag Compensation + Standalone is chosen the model and
    # compensation mode are fixed automatically (JB2008 / duty_cycle)
    # so the user is not prompted for either.
    # ============================================================
    auto_selected = (sim_type == "2" and mode_choice == "1")

    if auto_selected:
        from mission_config import SIMULATION
        model_name        = SIMULATION.get("atmosphere_model", "jb2008").lower()
        compensation_mode = SIMULATION.get("compensation_mode", "duty_cycle").lower()
        print(f"\n[Auto] Atmospheric model  : {model_name.upper()}")
        print(f"[Auto] Compensation mode  : {compensation_mode.replace('_', ' ').title()}")
    else:
        print("\nWhich atmospheric model do you want to use?")
        print("1. NRLMSISE-00")
        print("2. JB2008")
        print("3. DTM2000")
        print("4. Harris-Priester")
        if mode_choice == "2" and sim_type == "1":
            print("5. All Models (Compare Errors)")
            model_choice_input = input("Enter your choice (1/2/3/4/5): ").strip()
        else:
            model_choice_input = input("Enter your choice (1/2/3/4): ").strip()

        model_map = {
            "1": "nrlmsise00",
            "2": "jb2008",
            "3": "dtm2000",
            "4": "harrispriester",
            "5": "all",
        }
        model_name = model_map.get(model_choice_input, "nrlmsise00")
        if model_choice_input not in model_map:
            print("Invalid choice. Defaulting to NRLMSISE-00.")

        # --------------------------------------------------------
        # LEVEL 4 CHOICE: Drag Compensation Mode
        # --------------------------------------------------------
        compensation_mode = None
        if sim_type == "2":
            print("\nSelect drag compensation mode:")
            print("1. Duty Cycle         - fixed thrust ON below h_min / OFF above h_max")
            print("                        (edit h_min_km / h_max_km in DEFAULT_THRUSTER)")
            print("2. Altitude Maintenance - variable thrust each step to cancel drag")
            print("                          and hold altitude")
            print("3. Altitude Goal      - constant thrust until goal altitude is reached,")
            print("                        then station-keeps there")
            print("                        (edit goal_altitude_km in DEFAULT_THRUSTER)")
            dc_choice = input("Enter your choice (1/2/3): ").strip()
            dc_map = {"1": "duty_cycle", "2": "maintenance", "3": "goal"}
            compensation_mode = dc_map.get(dc_choice, "duty_cycle")
            if dc_choice not in dc_map:
                print("Invalid choice. Defaulting to Duty Cycle.")

    # ============================================================
    # EXECUTION
    # ============================================================
    if mode_choice == "1":
        # ----------------------------------------------------
        # STANDALONE SIMULATION
        # ----------------------------------------------------
        from mission_config import ORBIT, SPACECRAFT, SIMULATION

        utc = TimeScalesFactory.getUTC()
        start_date = AbsoluteDate(
            ORBIT.get("start_year", 2026),
            ORBIT.get("start_month", 1),
            ORBIT.get("start_day", 1),
            ORBIT.get("start_hour", 12),
            ORBIT.get("start_minute", 0),
            float(ORBIT.get("start_second", 0.0)),
            utc
        )
        
        params = {
            "start_date":    start_date,
            "altitude":      float(ORBIT["altitude_km"]) * 1000.0,
            "inclination":   float(ORBIT["inclination_deg"]),
            "eccentricity":  float(ORBIT["eccentricity"]),
            "raan":          float(ORBIT.get("raan_deg", 0.0)),
            "arg_perigee":   float(ORBIT.get("arg_perigee_deg", 0.0)),
            "true_anomaly":  float(ORBIT.get("true_anomaly_deg", 0.0)),
            
            "mass":          float(SPACECRAFT["mass_kg"]),
            "cross_section": float(SPACECRAFT["cross_section_m2"]),
            "drag_coeff":    float(SPACECRAFT["drag_coeff"]),
            
            "time_step":     float(SIMULATION["time_step_s"]),
            "duration":      float(SIMULATION["duration_days"]) * 86400.0,
        }

        # --------------------------------------------------------
        # PRE-LAUNCH: print summary and ask for confirmation
        # --------------------------------------------------------
        # Import thruster config early so it can appear in the summary
        if sim_type == "2":
            from mission_config import PROPULSION
            _thruster_for_summary = {
                "thrust": PROPULSION["thrust_N"],
                "isp": PROPULSION["isp_s"],
                "power_W": PROPULSION.get("power_W", 0.0),
                "propellant_mass": PROPULSION["propellant_mass_kg"],
                "h_min_km": PROPULSION["h_min_km"],
                "h_max_km": PROPULSION["h_max_km"],
            }
        else:
            _thruster_for_summary = None

        _print_simulation_summary(params, model_name, compensation_mode,
                                  thruster_params=_thruster_for_summary)
        if not _confirm_launch():
            print("Simulation cancelled.")
            return

        print(f"\nCalling the {model_name.upper()} model...\n")
        
        if sim_type == "1":
            results = atmospheric_model.run_simulation(params, model_type=model_name)
        else:
            # Standalone Drag Compensation
            from orbital_models.electric_propulsion import ElectricPropulsionSystem
            from mission_config import PROPULSION
            
            h0_km = params["altitude"] / 1000.0
            
            if compensation_mode == "duty_cycle":
                h_min = float(PROPULSION["h_min_km"]) * 1000.0
                h_max = float(PROPULSION["h_max_km"]) * 1000.0
                params["goal_altitude_km"] = None
                params["goal_offset_km"] = None
            elif compensation_mode == "maintenance":
                h_min = 1.0e9
                h_max = 2.0e9
                params["goal_altitude_km"] = None
                params["goal_offset_km"] = None
            else: # goal
                params["goal_altitude_km"] = PROPULSION.get("goal_altitude_km", h0_km)
                params["goal_offset_km"] = PROPULSION.get("goal_offset_km", 1.0)
                h_min = 1.0e9
                h_max = 2.0e9

            ep_system = ElectricPropulsionSystem(
                thrust=PROPULSION["thrust_N"],
                isp=PROPULSION["isp_s"],
                initial_propellant_mass=PROPULSION["propellant_mass_kg"],
                h_min=h_min,
                h_max=h_max
            )

            # Build the power subsystem (all logic lives in power_subsystem.py)
            from orbital_models.power_subsystem import build_power_model
            duration_days = params["duration"] / 86400.0
            power_model, (eff_mean, bat_cap_eom, thr_pwr) = build_power_model(duration_days)
            params["power_model"] = power_model

            print(f"\n  Power model initialised:")
            print(f"    Panel efficiency (mean) : {eff_mean * 100:.2f} %")
            print(f"    Battery capacity (EOL)  : {bat_cap_eom:.1f} Wh")

            results = atmospheric_model.run_simulation(
                params,
                model_type=model_name,
                propulsion_model=ep_system,
                compensation_mode=compensation_mode
            )
            print(f"\n  Propellant used   : {ep_system.propellant_used:.3f} kg")
            print(f"  Total burn time   : {ep_system.burn_time / 3600:.2f} h")
            print(f"  Thruster cycles   : {ep_system.cycles}")


        
        t1 = perf_counter()
        print(f"Time: {t1 - t0:.6f} s")

        if sim_type == "1":
            # Orbital decay only – plain orbital elements figure
            atmospheric_model.plot_results(results)
        else:
            # Drag compensation – orbital elements with threshold lines + extra figures
            if mode_choice == "1":
                h_min_km_plot = float(PROPULSION["h_min_km"])
                h_max_km_plot = float(PROPULSION["h_max_km"])
            else:
                from validations.drag_compensation_validation import DEFAULT_THRUSTER
                h_min_km_plot = float(DEFAULT_THRUSTER["h_min_km"])
                h_max_km_plot = float(DEFAULT_THRUSTER["h_max_km"])
            atmospheric_model.plot_results(results,
                                           h_min_km=h_min_km_plot,
                                           h_max_km=h_max_km_plot)
            atmospheric_model.plot_drag_compensation_figures(results,
                                                             ep_system=ep_system)

    elif mode_choice == "2":
        # ----------------------------------------------------
        # VALIDATION SIMULATION
        # ----------------------------------------------------
        folder_name = "decay" if sim_type == "1" else "drag_compensation"
        tle_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "validations", "tle_data", folder_name
        )

        tle_files = [f for f in os.listdir(tle_dir) if f.endswith(".txt")]
        if not tle_files:
            print(f"No TLE files found in validations/tle_data/{folder_name}/")
            return

        # LEVEL 5 CHOICE: TLE File
        print("\nAvailable TLE files:")
        for i, f in enumerate(tle_files, 1):
            print(f"{i}. {f}")

        try:
            file_choice = int(input(f"Select a TLE file (1-{len(tle_files)}): ").strip())
            selected_file = tle_files[file_choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Defaulting to the first file.")
            selected_file = tle_files[0]

        tle_file = os.path.join(tle_dir, selected_file)

        if sim_type == "1":
            from validations.orbital_decay_validation import run_validation
            run_validation(tle_file=tle_file, model_name=model_name, spacecraft_params=None)
        else:
            from validations.drag_compensation_validation import run_dc_validation
            run_dc_validation(
                tle_file=tle_file,
                model_name=model_name,
                compensation_mode=compensation_mode,
                spacecraft_params=None,
                thruster_params=None,
            )

if __name__ == "__main__":
    main()
