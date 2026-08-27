"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          gui.py
Description:
    Interactive Tkinter GUI for configuring and running orbital and propulsion simulations.
===============================================================================
"""

import sys
import os

# Auto-launch Streamlit if run directly using Python
from streamlit.runtime import Runtime
if __name__ == "__main__" and not Runtime.exists():
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit(0)

import math
from datetime import datetime, timedelta
import contextlib
import io
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# Prevent matplotlib from popping up window blockers and block plt.show()
plt.switch_backend('Agg')
plt.show = lambda *args, **kwargs: None

# Set page config
st.set_page_config(
    page_title="LeoOrbSim - Orbital Simulation GUI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #00bcd4;
        font-weight: 300;
        letter-spacing: 1.5px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00bcd4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Add the src directory to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Initialize JVM if not already running
import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception as e:
    pass

from orekit_jpype.pyhelpers import setup_orekit_data
PROJECT_ROOT = os.path.dirname(SRC_DIR)
setup_orekit_data(filenames=os.path.join(PROJECT_ROOT, "orekit-data-main"), from_pip_library=False)

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from orbital_models import atmospheric_model
from orbital_models.electric_propulsion import ElectricPropulsionSystem
from orbital_models.power_subsystem import build_power_model
from mission_config import RIT_THRUSTERS, get_thruster_config, SPACECRAFT, ORBIT, PROPULSION, POWER, SIMULATION

# Helper to capture logs
class StreamToLogger:
    def __init__(self):
        self.buffer = io.StringIO()
    def write(self, message):
        self.buffer.write(message)
    def flush(self):
        pass
    def get_val(self):
        return self.buffer.getvalue()

# Main GUI Layout
st.markdown("<h1 class='main-title'>🛰️ LeoOrbSim: High-Fidelity LEO Orbital Simulator</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Simulation Dashboard")

sim_mode = st.sidebar.selectbox(
    "Primary Simulation Mode",
    ["Standalone Simulation", "Validation Simulation (Against TLE)", "Staged Mission Profile"]
)

if sim_mode != "Staged Mission Profile":
    sim_type = st.sidebar.selectbox(
        "Simulation Type",
        ["Orbital Decay Only", "Orbital Decay + Drag Compensation"]
    )
else:
    sim_type = "Orbital Decay + Drag Compensation"

run_simulation = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# Main Form logic depending on mode
if sim_mode == "Standalone Simulation":
    st.subheader("Configure Initial Parameters")
    
    tab_orbit, tab_spacecraft, tab_prop, tab_power, tab_sim = st.tabs([
        "📍 Orbit Parameters", 
        "🛠️ Spacecraft Properties", 
        "⚡ Propulsion & Compensation", 
        "🔋 Power Subsystem", 
        "⚙️ Simulation Settings"
    ])
    
    with tab_orbit:
        col1, col2 = st.columns(2)
        with col1:
            orbit_alt = st.number_input("Initial Altitude (km)", min_value=100.0, value=float(ORBIT["altitude_km"]), step=10.0)
            orbit_ecc = st.number_input("Eccentricity (Circular ≈ 1e-8)", min_value=0.0, max_value=0.9999, value=float(ORBIT["eccentricity"]), format="%.2e")
            orbit_inc = st.number_input("Inclination (deg)", min_value=0.0, max_value=180.0, value=float(ORBIT["inclination_deg"]), step=1.0)
        with col2:
            orbit_raan = st.number_input("RAAN (deg)", min_value=0.0, max_value=360.0, value=float(ORBIT["raan_deg"]), step=10.0)
            orbit_arg = st.number_input("Argument of Perigee (deg)", min_value=0.0, max_value=360.0, value=float(ORBIT["arg_perigee_deg"]), step=10.0)
            orbit_anom = st.number_input("True Anomaly (deg)", min_value=0.0, max_value=360.0, value=float(ORBIT["true_anomaly_deg"]), step=10.0)
            
        st.write("**Epoch Start Date**")
        col3, col4, col5 = st.columns(3)
        with col3:
            epoch_year = st.number_input("Year", min_value=2000, max_value=2100, value=ORBIT["start_year"])
            epoch_month = st.number_input("Month", min_value=1, max_value=12, value=ORBIT["start_month"])
        with col4:
            epoch_day = st.number_input("Day", min_value=1, max_value=31, value=ORBIT["start_day"])
            epoch_hour = st.number_input("Hour", min_value=0, max_value=23, value=ORBIT["start_hour"])
        with col5:
            epoch_minute = st.number_input("Minute", min_value=0, max_value=59, value=ORBIT["start_minute"])
            epoch_second = st.number_input("Second", min_value=0.0, max_value=59.0, value=float(ORBIT["start_second"]), step=1.0)

    with tab_spacecraft:
        col1, col2 = st.columns(2)
        with col1:
            sc_mass = st.number_input("Dry Mass (kg)", min_value=0.1, value=float(SPACECRAFT["mass_kg"]), step=5.0)
            sc_area = st.number_input("Cross-sectional Area (m²)", min_value=0.001, value=float(SPACECRAFT["cross_section_m2"]), format="%.4f")
        with col2:
            sc_cd = st.number_input("Drag Coefficient (Cd)", min_value=0.0, value=float(SPACECRAFT["drag_coeff"]), format="%.2f")
            sc_cr = st.number_input("Reflectivity Coefficient (Cr)", min_value=0.0, value=float(SPACECRAFT["reflectivity_coeff"]), format="%.2f")

    with tab_prop:
        if sim_type == "Orbital Decay Only":
            st.info("Drag Compensation is disabled. Propulsion configuration is not required.")
            prop_thrust = 0.0
            prop_isp = 0.0
            prop_power = 0.0
            prop_mass = 0.0
            comp_mode = None
            h_min_km = 0.0
            h_max_km = 0.0
            goal_alt = 0.0
            goal_offset = 0.0
        else:
            col1, col2 = st.columns(2)
            with col1:
                thruster_select = st.selectbox(
                    "Thruster Catalog Model",
                    ["RIT µX", "RIT 10 EVO", "RIT 2X", "Custom Thruster"]
                )
                
                if thruster_select == "RIT µX":
                    key = "RIT_uX"
                    pt_idx = 0
                elif thruster_select == "RIT 10 EVO":
                    key = "RIT_10_EVO"
                    pt_labels = [pt["label"] for pt in RIT_THRUSTERS[key]["operating_points"]]
                    pt_sel = st.selectbox("Operating Point", pt_labels)
                    pt_idx = pt_labels.index(pt_sel)
                elif thruster_select == "RIT 2X":
                    key = "RIT_2X"
                    pt_labels = [pt["label"] for pt in RIT_THRUSTERS[key]["operating_points"]]
                    pt_sel = st.selectbox("Operating Point", pt_labels)
                    pt_idx = pt_labels.index(pt_sel)
                else:
                    key = None
                    pt_idx = None
                
                if key:
                    cfg = get_thruster_config(key, pt_idx)
                    st.write(f"**Selected:** {cfg['name']} ({cfg['label']})")
                    prop_thrust = st.number_input("Thrust (N)", value=cfg["thrust_N"], format="%.6f", disabled=True)
                    prop_isp = st.number_input("Specific Impulse (Isp, s)", value=float(cfg["isp_s"]), format="%.1f", disabled=True)
                    prop_power = st.number_input("Power Draw (W)", value=float(cfg["power_W"]), format="%.1f", disabled=True)
                else:
                    prop_thrust = st.number_input("Custom Thrust (N)", value=0.02, format="%.6f", step=0.001)
                    prop_isp = st.number_input("Custom Specific Impulse (Isp, s)", value=2500.0, format="%.1f", step=100.0)
                    prop_power = st.number_input("Custom Power Draw (W)", value=150.0, format="%.1f", step=10.0)

            with col2:
                prop_mass = st.number_input("Propellant Mass (kg)", min_value=0.0, max_value=1000.0, value=float(PROPULSION["propellant_mass_kg"]), step=1.0)
                comp_mode = st.selectbox(
                    "Drag Compensation Logic",
                    ["Duty Cycle (Threshold-based)", "Altitude Maintenance (Continuous variable)", "Altitude Goal (Seek & Keep)"]
                )
                
                comp_mode_key = "duty_cycle"
                if comp_mode == "Altitude Maintenance (Continuous variable)":
                    comp_mode_key = "maintenance"
                elif comp_mode == "Altitude Goal (Seek & Keep)":
                    comp_mode_key = "goal"

                if comp_mode_key == "duty_cycle":
                    h_min_km = st.number_input("ON Altitude Threshold (km)", value=float(PROPULSION["h_min_km"]), step=5.0)
                    h_max_km = st.number_input("OFF Altitude Threshold (km)", value=float(PROPULSION["h_max_km"]), step=5.0)
                    goal_alt = 0.0
                    goal_offset = 0.0
                elif comp_mode_key == "maintenance":
                    h_min_km = 1.0e9
                    h_max_km = 2.0e9
                    goal_alt = 0.0
                    goal_offset = 0.0
                    st.info("Altitude maintenance applies continuous variable thrust to cancel drag at the starting orbit.")
                else:
                    goal_alt = st.number_input("Goal Target Altitude (km)", value=float(PROPULSION.get("goal_altitude_km", orbit_alt)), step=5.0)
                    goal_offset = st.number_input("Station-keeping Band (± km)", value=float(PROPULSION.get("goal_offset_km", 1.0)), step=0.1)
                    h_min_km = 1.0e9
                    h_max_km = 2.0e9

    with tab_power:
        col1, col2 = st.columns(2)
        with col1:
            pwr_area = st.number_input("Solar Panel Area (m²)", min_value=0.0, value=float(POWER["solar_panel_area_m2"]), step=0.5)
            pwr_eff = st.number_input("Panel Efficiency BOL (%)", min_value=0.0, max_value=100.0, value=float(POWER["panel_efficiency"])*100.0, step=1.0) / 100.0
            pwr_flux = st.number_input("Solar Flux (W/m²)", min_value=0.0, value=float(POWER["solar_flux_W_m2"]), step=10.0)
            pwr_degr = st.number_input("Panel Degradation (%/year)", min_value=0.0, value=float(POWER["solar_panel_degradation_per_year"])*100.0, step=0.1) / 100.0
        with col2:
            pwr_bat_cap = st.number_input("Battery Capacity (Wh)", min_value=0.1, value=float(POWER["battery_capacity_Wh"]), step=10.0)
            pwr_bat_init = st.number_input("Battery Initial Charge (Wh)", min_value=0.0, max_value=pwr_bat_cap, value=float(POWER["battery_initial_Wh"]), step=10.0)
            pwr_bat_degr = st.number_input("Battery Degradation (%/year)", min_value=0.0, value=float(POWER["battery_degradation_per_year"])*100.0, step=0.1) / 100.0
            pwr_hk = st.number_input("Housekeeping Power Draw (W)", min_value=0.0, value=float(POWER["housekeeping_power_W"]), step=5.0)

    with tab_sim:
        col1, col2 = st.columns(2)
        with col1:
            sim_dur = st.number_input("Simulation Duration (days)", min_value=0.1, value=float(SIMULATION["duration_days"]), step=5.0)
            sim_step = st.number_input("Time Step (seconds)", min_value=1.0, value=float(SIMULATION["time_step_s"]), step=10.0)
        with col2:
            sim_atm = st.selectbox(
                "Atmospheric Model",
                ["NRLMSISE-00", "JB2008", "DTM2000", "Harris-Priester"]
            )
            sim_atm_key = sim_atm.replace("-", "").lower()

elif sim_mode == "Validation Simulation (Against TLE)":
    st.subheader("Configure Validation Settings")
    
    # Select folder & TLE directory
    subfolder = "decay" if sim_type == "Orbital Decay Only" else "drag_compensation"
    tle_dir = os.path.join(SRC_DIR, "validations", "tle_data", subfolder)
    
    # Read files
    try:
        tle_files = [f for f in os.listdir(tle_dir) if f.endswith(".txt")]
    except Exception:
        tle_files = []
        
    if not tle_files:
        st.error(f"No TLE files found in validations/tle_data/{subfolder}/")
        st.stop()
        
    col1, col2 = st.columns(2)
    with col1:
        selected_file = st.selectbox("Select TLE File", tle_files)
        tle_path = os.path.join(tle_dir, selected_file)
        
        # Detect spacecraft key from filename
        filename_lower = selected_file.lower()
        if filename_lower.startswith("goce"):
            sc_key = "goce"
        elif filename_lower.startswith("grace"):
            sc_key = "grace"
        elif filename_lower.startswith("champ"):
            sc_key = "champ"
        elif filename_lower.startswith("slats"):
            sc_key = "slats"
        elif filename_lower.startswith("soar"):
            sc_key = "soar"
        else:
            sc_key = "goce"
            
    with col2:
        # User selection of atmospheric model
        if sim_type == "Orbital Decay Only":
            atm_options = ["NRLMSISE-00", "JB2008", "DTM2000", "Harris-Priester", "All Models (Compare Errors)"]
        else:
            atm_options = ["NRLMSISE-00", "JB2008", "DTM2000", "Harris-Priester"]
            
        selected_atm = st.selectbox("Validation Atmospheric Model", atm_options)
        selected_atm_key = selected_atm.replace("-", "").lower()
        if "all" in selected_atm_key:
            selected_atm_key = "all"

    # Display TLE info & Spacecraft parameters
    st.markdown("### 📋 TLE Metadata & Spacecraft Parameters")
    
    # Read TLE entries to show metadata
    from validations.orbital_decay_validation import parse_tle_file, propagate_tle_at_epoch
    tle_entries = parse_tle_file(tle_path)
    num_tles = len(tle_entries)
    
    # Calculate start / end dates
    first_entry = propagate_tle_at_epoch(tle_entries[0][0], tle_entries[0][1], tle_entries[0][2])
    last_entry = propagate_tle_at_epoch(tle_entries[-1][0], tle_entries[-1][1], tle_entries[-1][2])
    
    first_dt = datetime.fromisoformat(str(first_entry["epoch"]).replace("Z", "+00:00"))
    last_dt = datetime.fromisoformat(str(last_entry["epoch"]).replace("Z", "+00:00"))
    span_days = (last_dt - first_dt).total_seconds() / 86400.0

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"""
        <div class="section-card">
            <h4>🛰️ TLE Ground Truth Information</h4>
            <table style="width:100%; border:none;">
                <tr><td><strong>Satellite Name</strong></td><td>{sc_key.upper()} ({tle_entries[0][0]})</td></tr>
                <tr><td><strong>TLE Records Found</strong></td><td>{num_tles} points</td></tr>
                <tr><td><strong>Mission Epoch Span</strong></td><td>{span_days:.2f} days</td></tr>
                <tr><td><strong>Start Epoch Date</strong></td><td>{first_dt.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td><strong>End Epoch Date</strong></td><td>{last_dt.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td><strong>Initial Mean Altitude</strong></td><td>{first_entry['altitude_km']:.2f} km</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Spacecraft Parameters registry editing
    from validations.orbital_decay_validation import SPACECRAFT_REGISTRY as REG_DECAY
    from validations.drag_compensation_validation import SPACECRAFT_REGISTRY as REG_DC
    
    current_reg = REG_DECAY if sim_type == "Orbital Decay Only" else REG_DC
    sc_defaults = current_reg.get(sc_key, list(current_reg.values())[0])

    with col_meta2:
        st.markdown("<div class='section-card'><h4>🛠️ Spacecraft Configuration Parameters</h4>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            val_sc_mass = st.number_input("Dry Mass (kg)", value=float(sc_defaults["mass"]), step=5.0)
            val_sc_area = st.number_input("Cross-section (m²)", value=float(sc_defaults["cross_section"]), format="%.4f")
            val_time_step = st.number_input("Integrator Step size (s)", value=float(sc_defaults.get("time_step", 3600.0)), step=100.0)
        with col_s2:
            if sim_type == "Orbital Decay Only":
                val_sc_cd_min = st.number_input("Drag Coeff Min (Cd)", value=float(sc_defaults.get("drag_coeff_min", 2.2)))
                val_sc_cd_max = st.number_input("Drag Coeff Max (Cd)", value=float(sc_defaults.get("drag_coeff_max", 4.0)))
                val_sc_cd = (val_sc_cd_min + val_sc_cd_max) / 2.0
            else:
                val_sc_cd = st.number_input("Drag Coeff (Cd)", value=float(sc_defaults.get("drag_coeff", 3.8)))
                val_sc_cd_min = val_sc_cd
                val_sc_cd_max = val_sc_cd
        st.markdown("</div>", unsafe_allow_html=True)

    # For Drag Compensation validation
    if sim_type == "Orbital Decay + Drag Compensation":
        st.markdown("### ⚡ Thruster & Compensation Configuration")
        from validations.drag_compensation_validation import DEFAULT_THRUSTER
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            val_thrust = st.number_input("Thrust (N)", value=float(DEFAULT_THRUSTER["thrust"]), format="%.4f")
            val_isp = st.number_input("Specific Impulse (Isp, s)", value=float(DEFAULT_THRUSTER["isp"]))
            val_prop_mass = st.number_input("Propellant Mass (kg)", value=float(DEFAULT_THRUSTER["propellant_mass"]))
        with col_t2:
            val_comp_mode = st.selectbox(
                "Drag Compensation Logic",
                ["Duty Cycle (Threshold-based)", "Altitude Maintenance (Continuous variable)", "Altitude Goal (Seek & Keep)"]
            )
            val_comp_mode_key = "duty_cycle"
            if val_comp_mode == "Altitude Maintenance (Continuous variable)":
                val_comp_mode_key = "maintenance"
            elif val_comp_mode == "Altitude Goal (Seek & Keep)":
                val_comp_mode_key = "goal"

            if val_comp_mode_key == "duty_cycle":
                val_h_min_km = st.number_input("ON Altitude Threshold (km)", value=float(DEFAULT_THRUSTER["h_min_km"]))
                val_h_max_km = st.number_input("OFF Altitude Threshold (km)", value=float(DEFAULT_THRUSTER["h_max_km"]))
                val_goal_alt = 0.0
                val_goal_offset = 0.0
            elif val_comp_mode_key == "maintenance":
                val_h_min_km = 1.0e9
                val_h_max_km = 2.0e9
                val_goal_alt = 0.0
                val_goal_offset = 0.0
            else:
                val_goal_alt = st.number_input("Goal Target Altitude (km)", value=float(DEFAULT_THRUSTER.get("goal_altitude_km", first_entry["altitude_km"])))
                val_goal_offset = st.number_input("Station-keeping Band (± km)", value=float(DEFAULT_THRUSTER.get("goal_offset_km", 1.0)))
                val_h_min_km = 1.0e9
                val_h_max_km = 2.0e9

elif sim_mode == "Staged Mission Profile":
    st.subheader("Configure Staged Mission Parameters")
    
    # Load staged profile definitions
    from validations.staged_mission_validation import STAGES, SPACECRAFT as GOCE_SC, THRUST_N, ISP_S, PROPELLANT_MASS_KG
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="section-card">
            <h4>🛰️ Mission Settings (GOCE Staged Mission)</h4>
            <table style="width:100%;">
                <tr><td><strong>Satellite</strong></td><td>GOCE</td></tr>
                <tr><td><strong>Atmospheric Model</strong></td><td>JB2008 (Certified)</td></tr>
                <tr><td><strong>Compensation Mode</strong></td><td>Staged Tracking</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='section-card'><h4>⚡ Thruster Settings</h4>", unsafe_allow_html=True)
        st_thrust = st.number_input("Thrust (N)", value=float(THRUST_N), format="%.4f")
        st_isp = st.number_input("Specific Impulse (Isp, s)", value=float(ISP_S))
        st_prop = st.number_input("Propellant Mass (kg)", value=float(PROPELLANT_MASS_KG))
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("### 📅 Mission Stages Profile")
    st.table([{"Stage #": idx+1, "End Date": date if date else "End of Simulation", "Target Altitude": f"{alt} km" if isinstance(alt, (int, float)) else alt} for idx, (date, alt) in enumerate(STAGES)])

# RUN SIMULATION TRIGGER
if run_simulation:
    st.write("---")
    st.markdown("### 📝 Execution Logs")
    log_area = st.empty()
    
    # Capture print statements
    logger = StreamToLogger()
    
    with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
        try:
            if sim_mode == "Standalone Simulation":
                utc = TimeScalesFactory.getUTC()
                start_date = AbsoluteDate(
                    int(epoch_year), int(epoch_month), int(epoch_day),
                    int(epoch_hour), int(epoch_minute), float(epoch_second),
                    utc
                )
                
                params = {
                    "start_date":    start_date,
                    "altitude":      float(orbit_alt) * 1000.0,
                    "inclination":   float(orbit_inc),
                    "eccentricity":  float(orbit_ecc),
                    "raan":          float(orbit_raan),
                    "arg_perigee":   float(orbit_arg),
                    "true_anomaly":  float(orbit_anom),
                    "mass":          float(sc_mass),
                    "cross_section": float(sc_area),
                    "drag_coeff":    float(sc_cd),
                    "reflectivity_coeff": float(sc_cr),
                    "time_step":     float(sim_step),
                    "duration":      float(sim_dur) * 86400.0,
                }
                
                print("Running standalone simulation...")
                if sim_type == "Orbital Decay Only":
                    results = atmospheric_model.run_simulation(params, model_type=sim_atm_key)
                    # Generate plots
                    atmospheric_model.plot_results(results)
                else:
                    # Setup propulsion
                    ep_system = ElectricPropulsionSystem(
                        thrust=prop_thrust,
                        isp=prop_isp,
                        initial_propellant_mass=prop_mass,
                        h_min=h_min_km * 1000.0,
                        h_max=h_max_km * 1000.0
                    )
                    
                    if comp_mode_key == "goal":
                        params["goal_altitude_km"] = goal_alt
                        params["goal_offset_km"] = goal_offset
                        
                    # Setup power model by updating in-memory mission_config variables
                    from mission_config import POWER as MC_POWER, PROPULSION as MC_PROPULSION
                    MC_POWER["solar_panel_area_m2"] = pwr_area
                    MC_POWER["panel_efficiency"] = pwr_eff
                    MC_POWER["solar_flux_W_m2"] = pwr_flux
                    MC_POWER["solar_panel_degradation_per_year"] = pwr_degr
                    MC_POWER["battery_capacity_Wh"] = pwr_bat_cap
                    MC_POWER["battery_initial_Wh"] = pwr_bat_init
                    MC_POWER["battery_degradation_per_year"] = pwr_bat_degr
                    MC_POWER["housekeeping_power_W"] = pwr_hk
                    MC_PROPULSION["power_W"] = prop_power
                    MC_PROPULSION["thrust_N"] = prop_thrust
                    MC_PROPULSION["isp_s"] = prop_isp
                    MC_PROPULSION["propellant_mass_kg"] = prop_mass
                    
                    power_model, (eff_mean, bat_cap_eom, thr_pwr) = build_power_model(sim_dur)
                    params["power_model"] = power_model
                    
                    results = atmospheric_model.run_simulation(
                        params,
                        model_type=sim_atm_key,
                        propulsion_model=ep_system,
                        compensation_mode=comp_mode_key
                    )
                    
                    print(f"Propellant used   : {ep_system.propellant_used:.3f} kg")
                    print(f"Total burn time   : {ep_system.burn_time / 3600:.2f} h")
                    print(f"Thruster cycles   : {ep_system.cycles}")
                    
                    # Generate plots
                    atmospheric_model.plot_results(results, h_min_km=h_min_km, h_max_km=h_max_km)
                    atmospheric_model.plot_drag_compensation_figures(results, ep_system=ep_system)
                
                # Show results in Streamlit
                log_area.code(logger.get_val())
                st.success("Simulation completed successfully!")
                
                # Render active matplotlib figures
                st.markdown("### 📊 Trajectory Plots")
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    st.pyplot(fig)
                    
            elif sim_mode == "Validation Simulation (Against TLE)":
                if sim_type == "Orbital Decay Only":
                    from validations.orbital_decay_validation import run_validation as run_dec_val
                    
                    sc_params_dict = {
                        "mass": val_sc_mass,
                        "cross_section": val_sc_area,
                        "drag_coeff_min": val_sc_cd_min,
                        "drag_coeff_max": val_sc_cd_max,
                        "time_step": val_time_step
                    }
                    
                    run_dec_val(
                        tle_file=tle_path,
                        model_name=selected_atm_key,
                        spacecraft_params=sc_params_dict
                    )
                    
                    log_area.code(logger.get_val())
                    st.success("Decay Validation simulation completed!")
                    
                    # Read files saved in results/results_decay_validations/
                    st.markdown("### 📊 Validation Plots")
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)
                        
                else:
                    from validations.drag_compensation_validation import run_dc_validation as run_comp_val
                    
                    sc_params_dict = {
                        "mass": val_sc_mass,
                        "cross_section": val_sc_area,
                        "drag_coeff": val_sc_cd,
                        "time_step": val_time_step
                    }
                    
                    th_params_dict = {
                        "thrust": val_thrust,
                        "isp": val_isp,
                        "propellant_mass": val_prop_mass,
                        "h_min_km": val_h_min_km,
                        "h_max_km": val_h_max_km,
                        "goal_altitude_km": val_goal_alt,
                        "goal_offset_km": val_goal_offset
                    }
                    
                    run_comp_val(
                        tle_file=tle_path,
                        model_name=selected_atm_key,
                        compensation_mode=val_comp_mode_key,
                        spacecraft_params=sc_params_dict,
                        thruster_params=th_params_dict
                    )
                    
                    log_area.code(logger.get_val())
                    st.success("Drag Compensation Validation simulation completed!")
                    
                    st.markdown("### 📊 Validation Plots")
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)
                        
            elif sim_mode == "Staged Mission Profile":
                # Staged mission validation GOCE
                import validations.staged_mission_validation as sm_val
                
                # Setup override values in sm_val config
                sm_val.THRUST_N = st_thrust
                sm_val.ISP_S = st_isp
                sm_val.PROPELLANT_MASS_KG = st_prop
                
                sm_val.run_staged_mission()
                
                log_area.code(logger.get_val())
                st.success("Staged Mission completed successfully!")
                
                st.markdown("### 📊 Staged Mission Plots")
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    st.pyplot(fig)
                    
        except Exception as sim_err:
            log_area.code(logger.get_val())
            st.error(f"An error occurred during simulation: {sim_err}")
            st.exception(sim_err)
