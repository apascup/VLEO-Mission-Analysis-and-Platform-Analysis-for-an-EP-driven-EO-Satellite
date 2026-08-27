"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          tradeoff_case_simulations.py
Description:
    Automated parametric trade-off simulation engine across altitude bands, thruster modes, and solar activity cycles.
===============================================================================
"""

import sys
import os
import argparse
import math
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Ensure matplotlib is non-interactive for batch plotting
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# ==============================================================================
# OREKIT INITIALIZATION
# ==============================================================================
import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception:
    pass

from orekit_jpype.pyhelpers import setup_orekit_data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
setup_orekit_data(filenames=str(PROJECT_ROOT / "orekit-data-main"), from_pip_library=False)

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from orbital_models import atmospheric_model
from orbital_models.electric_propulsion import ElectricPropulsionSystem
from orbital_models.power_subsystem import PowerSubsystem
import orbital_plot_style as _ops
from mission_config import get_thruster_config

# ==============================================================================
# ENGINEERING ASSUMPTIONS
# ==============================================================================
"""
1. Solar Array Orientation: Arrays are assumed to perfectly track the sun during 
   sunlight portions of the orbit (no cosine losses).
2. Aerodynamic Area: Solar array area is treated as edge-on for drag purposes
   unless 'projected_drag_area_m2' is specified > 0.
3. Payload Duty Cycle: Payload power is time-averaged over the orbit to create a constant
   average load, rather than discrete operations.
4. Battery Cycling Logic: Simple energy accounting with fixed charge/discharge 
   efficiencies. Cycle count is tracked as equivalent full cycles.
5. Missing Data Fallbacks: Platform base assumptions default to standard 
   mission_config values if missing.
"""

# ==============================================================================
# GLOBAL SIMULATION SETTINGS
# ==============================================================================
SIMULATION_START_YEAR = 2022
SIMULATION_START_MONTH = 1
SIMULATION_START_DAY = 1
SIMULATION_START_HOUR = 0
SIMULATION_START_MINUTE = 0
SIMULATION_START_SECOND = 0.0

SIMULATION_DURATION_DAYS = 4*365.0
OUTPUT_TIMESTEP_S = 3600.0
ATMOSPHERIC_MODEL = "nrlmsise00" # nrlmsise00, jb2008, dtm2000, harrispriester
MIN_DUTY_CYCLE_ALTITUDE_KM = 320.0
MAX_DUTY_CYCLE_ALTITUDE_KM = 380.0

INITIAL_ALTITUDE_KM = MAX_DUTY_CYCLE_ALTITUDE_KM
INITIAL_INCLINATION_DEG = 96.74
INITIAL_ECCENTRICITY = 1e-8
INITIAL_RAAN_DEG = 11.08
INITIAL_ARG_PERIGEE_DEG = 0.0
INITIAL_TRUE_ANOMALY_DEG = 0.0

SOLAR_PANEL_AREA_FACTOR = 0.1

OVERWRITE_EXISTING_RESULTS = True

# ==============================================================================
# CASE CONFIGURATION
# ==============================================================================
TRADEOFF_CASES = {
    "CASE_01": {
        "name": "16U CubeSat with HyperScape100",
        "enabled": True,
        "platform_id": "PLT-007",
        "platform_name": "EnduroSat 16U",
        "battery_id": "BAT-017",
        "battery_name": "EnduroSat 198 Wh battery configuration",
        "solar_array_id": "SOL-015",
        "solar_array_name": "1 x TSP-45W",
        "payload_id": "PAY-003",
        "payload_name": "Simera Sense HyperScape100",
        "battery_mass_included_in_platform": False,
        "solar_array_mass_included_in_platform": False,
        "platform": {
            "bus_dry_mass_kg": 12.3,
            "maximum_payload_mass_kg": 24.8,
            "avg_payload_power_w": 60.0,
            "peak_payload_power_w": None,
            "battery_capacity_included_wh": None,
            "solar_array_power_included_w": None,
            "dimensions_u_or_l": 8.0,
            "payload_volume_l": 8.0,
            "reference_drag_area_m2": 0.1,
            "drag_coefficient": 3.6,
            "pointing_accuracy_deg": None,
            "housekeeping_power_w": 30.0,
            "other_dry_mass_kg": 2.0,
            "battery_mass_included_in_platform": False,
            "solar_array_mass_included_in_platform": False,
        },
        "payload": {
            "mass_kg": 1.26,
            "envelope_volume_l": 1.71,
            "active_power_w": 7.75,
            "standby_power_w": 0.0,
            "readout_power_w": None,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 4.75,
            "gsd_model_alt_m": 2.85,
            "swath_ref_alt_km": 19.4,
            "swath_model_alt_km": 11.64,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 1,
            "nominal_capacity_wh": 198.0,
            "capacity_wh": 198.0,
            "capacity_per_module_wh": 198.0,
            "usable_energy_wh": 53.46,
            "mass_per_module_kg": None,
            "total_mass_kg": None,
            "voltage_v": None,
            "max_discharge_power_w": None,
            "max_continuous_power_w": None,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.30,
            "min_state_of_charge": 0.70,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 1,
            "bol_power_per_unit_w": 45.0,
            "bol_power_w": 45.0,
            "eol_power_w": 38.25,
            "mass_per_unit_kg": 0.885,
            "total_mass_kg": 0.885,
            "deployed_area_m2": 0.19344,
            "projected_drag_area_m2": 0.19344,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 5.0,
        },
    },
    "CASE_02": {
        "name": "SSTL-MICRO with MultiScape200 (external 400 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-017",
        "platform_name": "SSTL-MICRO",
        "battery_id": "BAT-020",
        "battery_name": "4 x GomSpace BPX 100 Wh",
        "solar_array_id": "SOL-017",
        "solar_array_name": "6 x TSP-45W",
        "payload_id": "PAY-005",
        "payload_name": "Simera Sense MultiScape200",
        "battery_mass_included_in_platform": False,
        "solar_array_mass_included_in_platform": False,
        "platform": {
            "bus_dry_mass_kg": 30.0,
            "maximum_payload_mass_kg": 65.0,
            "avg_payload_power_w": 63.0,
            "peak_payload_power_w": 200.0,
            "battery_capacity_included_wh": None,
            "solar_array_power_included_w": None,
            "dimensions_u_or_l": 65.0,
            "payload_volume_l": 65.0,
            "reference_drag_area_m2": 0.15,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": None,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 5.0,
            "battery_mass_included_in_platform": False,
            "solar_array_mass_included_in_platform": False,
        },
        "payload": {
            "mass_kg": 12.31,
            "envelope_volume_l": 15.0,
            "active_power_w": 45.0,
            "standby_power_w": 0.0,
            "readout_power_w": None,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 1.48,
            "gsd_model_alt_m": 0.89,
            "swath_ref_alt_km": 15.0,
            "swath_model_alt_km": 9.0,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 4,
            "nominal_capacity_wh": 400.0,
            "capacity_wh": 400.0,
            "capacity_per_module_wh": 100.0,
            "usable_energy_wh": 108.0,
            "mass_per_module_kg": 0.5,
            "total_mass_kg": 2.0,
            "voltage_v": 28.0,
            "max_discharge_power_w": 691.2,
            "max_continuous_power_w": 691.2,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 6,
            "bol_power_per_unit_w": 45.0,
            "bol_power_w": 270.0,
            "eol_power_w": 229.5,
            "mass_per_unit_kg": 0.885,
            "total_mass_kg": 5.31,
            "deployed_area_m2": 1.16064,
            "projected_drag_area_m2": 0.16064,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 10.0,
        },
    },
    "CASE_03": {
        "name": "SSTL-MICRO with DragonEye (external 800 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-017",
        "platform_name": "SSTL-MICRO",
        "battery_id": "BAT-021",
        "battery_name": "8 x GomSpace BPX 100 Wh",
        "solar_array_id": "SOL-018",
        "solar_array_name": "12 x TSP-45W",
        "payload_id": "PAY-011",
        "payload_name": "Dragonfly DragonEye",
        "battery_mass_included_in_platform": False,
        "solar_array_mass_included_in_platform": False,
        "platform": {
            "bus_dry_mass_kg": 30.0,
            "maximum_payload_mass_kg": 65.0,
            "avg_payload_power_w": 63.0,
            "peak_payload_power_w": 200.0,
            "battery_capacity_included_wh": None,
            "solar_array_power_included_w": None,
            "dimensions_u_or_l": 65.0,
            "payload_volume_l": 65.0,
            "reference_drag_area_m2": 0.15,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": None,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 5.0,
            "battery_mass_included_in_platform": False,
            "solar_array_mass_included_in_platform": False,
        },
        "payload": {
            "mass_kg": 18.0,
            "envelope_volume_l": None,
            "active_power_w": 45.0,
            "standby_power_w": 0.0,
            "readout_power_w": 15.0,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 0.75,
            "gsd_model_alt_m": 0.45,
            "swath_ref_alt_km": 10.0,
            "swath_model_alt_km": 6.0,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 8,
            "nominal_capacity_wh": 800.0,
            "capacity_wh": 800.0,
            "capacity_per_module_wh": 100.0,
            "usable_energy_wh": 216.0,
            "mass_per_module_kg": 0.5,
            "total_mass_kg": 4.0,
            "voltage_v": 28.0,
            "max_discharge_power_w": 1382.4,
            "max_continuous_power_w": 1382.4,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 12,
            "bol_power_per_unit_w": 45.0,
            "bol_power_w": 540.0,
            "eol_power_w": 459.0,
            "mass_per_unit_kg": 0.885,
            "total_mass_kg": 10.62,
            "deployed_area_m2": 2.32128,
            "projected_drag_area_m2": 2.32128,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 15.0,
        },
    },
    "CASE_04": {
        "name": "The Frame with HyperScape100 (integrated 1100 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-033",
        "platform_name": "EnduroSat The Frame",
        "battery_id": "Integrated",
        "battery_name": "Integrated 1100 Wh platform battery",
        "solar_array_id": "Integrated",
        "solar_array_name": "Integrated 600 W BOL array",
        "payload_id": "PAY-003",
        "payload_name": "Simera Sense HyperScape100",
        "battery_mass_included_in_platform": True,
        "solar_array_mass_included_in_platform": True,
        "platform": {
            "bus_dry_mass_kg": 100.0,
            "maximum_payload_mass_kg": 70.0,
            "avg_payload_power_w": 440.0,
            "peak_payload_power_w": 3400.0,
            "battery_capacity_included_wh": 1100.0,
            "solar_array_power_included_w": 600.0,
            "dimensions_u_or_l": 70.0,
            "payload_volume_l": 70.0,
            "reference_drag_area_m2": 0.5,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": 0.01,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 10.0,
            "battery_mass_included_in_platform": True,
            "solar_array_mass_included_in_platform": True,
        },
        "payload": {
            "mass_kg": 1.26,
            "envelope_volume_l": 1.71,
            "active_power_w": 7.75,
            "standby_power_w": 0.0,
            "readout_power_w": None,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 4.75,
            "gsd_model_alt_m": 2.85,
            "swath_ref_alt_km": 19.4,
            "swath_model_alt_km": 11.64,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 1,
            "nominal_capacity_wh": 1100.0,
            "capacity_wh": 1100.0,
            "capacity_per_module_wh": 1100.0,
            "usable_energy_wh": 297.0,
            "mass_per_module_kg": None,
            "total_mass_kg": None,
            "voltage_v": 28.0,
            "max_discharge_power_w": 3400.0,
            "max_continuous_power_w": 3400.0,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 1,
            "bol_power_per_unit_w": 600.0,
            "bol_power_w": 600.0,
            "eol_power_w": 510.0,
            "mass_per_unit_kg": None,
            "total_mass_kg": None,
            "deployed_area_m2": 2.0,
            "projected_drag_area_m2": 0.0,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 15.0,
        },
    },
    "CASE_05": {
        "name": "The Frame with MultiScape200 (integrated 1100 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-033",
        "platform_name": "EnduroSat The Frame",
        "battery_id": "Integrated",
        "battery_name": "Integrated 1100 Wh platform battery",
        "solar_array_id": "Integrated",
        "solar_array_name": "Integrated 600 W BOL array",
        "payload_id": "PAY-005",
        "payload_name": "Simera Sense MultiScape200",
        "battery_mass_included_in_platform": True,
        "solar_array_mass_included_in_platform": True,
        "platform": {
            "bus_dry_mass_kg": 100.0,
            "maximum_payload_mass_kg": 70.0,
            "avg_payload_power_w": 440.0,
            "peak_payload_power_w": 3400.0,
            "battery_capacity_included_wh": 1100.0,
            "solar_array_power_included_w": 600.0,
            "dimensions_u_or_l": 70.0,
            "payload_volume_l": 70.0,
            "reference_drag_area_m2": 0.5,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": 0.01,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 10.0,
            "battery_mass_included_in_platform": True,
            "solar_array_mass_included_in_platform": True,
        },
        "payload": {
            "mass_kg": 12.31,
            "envelope_volume_l": 15.0,
            "active_power_w": 7.0,
            "standby_power_w": 0.0,
            "readout_power_w": None,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 1.48,
            "gsd_model_alt_m": 0.89,
            "swath_ref_alt_km": 15.0,
            "swath_model_alt_km": 9.0,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 1,
            "nominal_capacity_wh": 1100.0,
            "capacity_wh": 1100.0,
            "capacity_per_module_wh": 1100.0,
            "usable_energy_wh": 297.0,
            "mass_per_module_kg": None,
            "total_mass_kg": None,
            "voltage_v": 28.0,
            "max_discharge_power_w": 3400.0,
            "max_continuous_power_w": 3400.0,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 1,
            "bol_power_per_unit_w": 600.0,
            "bol_power_w": 600.0,
            "eol_power_w": 510.0,
            "mass_per_unit_kg": None,
            "total_mass_kg": None,
            "deployed_area_m2": 2.0,
            "projected_drag_area_m2": 0.0,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 20.0,
        },
    },
    "CASE_06": {
        "name": "The Frame with Raptor (integrated 1100 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-033",
        "platform_name": "EnduroSat The Frame",
        "battery_id": "Integrated",
        "battery_name": "Integrated 1100 Wh platform battery",
        "solar_array_id": "Integrated",
        "solar_array_name": "Integrated 600 W BOL array",
        "payload_id": "PAY-012",
        "payload_name": "Dragonfly Raptor",
        "battery_mass_included_in_platform": True,
        "solar_array_mass_included_in_platform": True,
        "platform": {
            "bus_dry_mass_kg": 100.0,
            "maximum_payload_mass_kg": 70.0,
            "avg_payload_power_w": 440.0,
            "peak_payload_power_w": 3400.0,
            "battery_capacity_included_wh": 1100.0,
            "solar_array_power_included_w": 600.0,
            "dimensions_u_or_l": 70.0,
            "payload_volume_l": 70.0,
            "reference_drag_area_m2": 0.5,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": 0.01,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 10.0,
            "battery_mass_included_in_platform": True,
            "solar_array_mass_included_in_platform": True,
        },
        "payload": {
            "mass_kg": 60.0,
            "envelope_volume_l": None,
            "active_power_w": 45.0,
            "standby_power_w": 0.0,
            "readout_power_w": 20.0,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 0.30,
            "gsd_model_alt_m": 0.18,
            "swath_ref_alt_km": 8.0,
            "swath_model_alt_km": 4.8,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 1,
            "nominal_capacity_wh": 1100.0,
            "capacity_wh": 1100.0,
            "capacity_per_module_wh": 1100.0,
            "usable_energy_wh": 297.0,
            "mass_per_module_kg": None,
            "total_mass_kg": None,
            "voltage_v": 28.0,
            "max_discharge_power_w": 3400.0,
            "max_continuous_power_w": 3400.0,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 1,
            "bol_power_per_unit_w": 600.0,
            "bol_power_w": 600.0,
            "eol_power_w": 510.0,
            "mass_per_unit_kg": None,
            "total_mass_kg": None,
            "deployed_area_m2": 2.0,
            "projected_drag_area_m2": 0.0,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 20.0,
        },
    },
    "CASE_07": {
        "name": "MOOG Meteorite with Raptor (external 800 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-042",
        "platform_name": "MOOG Meteorite",
        "battery_id": "BAT-021",
        "battery_name": "8 x GomSpace BPX 100 Wh",
        "solar_array_id": "SOL-018",
        "solar_array_name": "12 x TSP-45W",
        "payload_id": "PAY-012",
        "payload_name": "Dragonfly Raptor",
        "battery_mass_included_in_platform": False,
        "solar_array_mass_included_in_platform": False,
        "platform": {
            "bus_dry_mass_kg": 120.0,
            "maximum_payload_mass_kg": 220.0,
            "avg_payload_power_w": 150.0,
            "peak_payload_power_w": 2000.0,
            "battery_capacity_included_wh": None,
            "solar_array_power_included_w": None,
            "dimensions_u_or_l": 220.0,
            "payload_volume_l": None,
            "reference_drag_area_m2": 0.6,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": 0.003,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 15.0,
            "battery_mass_included_in_platform": False,
            "solar_array_mass_included_in_platform": False,
        },
        "payload": {
            "mass_kg": 60.0,
            "envelope_volume_l": None,
            "active_power_w": 45.0,
            "standby_power_w": 0.0,
            "readout_power_w": 20.0,
            "peak_power_w": None,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 0.30,
            "gsd_model_alt_m": 0.18,
            "swath_ref_alt_km": 8.0,
            "swath_model_alt_km": 4.8,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 8,
            "nominal_capacity_wh": 800.0,
            "capacity_wh": 800.0,
            "capacity_per_module_wh": 100.0,
            "usable_energy_wh": 216.0,
            "mass_per_module_kg": 0.5,
            "total_mass_kg": 4.0,
            "voltage_v": 28.0,
            "max_discharge_power_w": 1382.4,
            "max_continuous_power_w": 1382.4,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 12,
            "bol_power_per_unit_w": 45.0,
            "bol_power_w": 540.0,
            "eol_power_w": 459.0,
            "mass_per_unit_kg": 0.885,
            "total_mass_kg": 10.62,
            "deployed_area_m2": 2.32128,
            "projected_drag_area_m2": 2.32128,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 30.0,
        },
    },
    "CASE_08": {
        "name": "MOOG Meteorite with SAR-C (external 1200 Wh EPS)",
        "enabled": True,
        "platform_id": "PLT-042",
        "platform_name": "MOOG Meteorite",
        "battery_id": "BAT-022",
        "battery_name": "12 x GomSpace BPX 100 Wh",
        "solar_array_id": "SOL-019",
        "solar_array_name": "20 x TSP-45W",
        "payload_id": "PAY-015",
        "payload_name": "Dragonfly SAR-C",
        "battery_mass_included_in_platform": False,
        "solar_array_mass_included_in_platform": False,
        "platform": {
            "bus_dry_mass_kg": 120.0,
            "maximum_payload_mass_kg": 220.0,
            "avg_payload_power_w": 150.0,
            "peak_payload_power_w": 2000.0,
            "battery_capacity_included_wh": None,
            "solar_array_power_included_w": None,
            "dimensions_u_or_l": 220.0,
            "payload_volume_l": None,
            "reference_drag_area_m2": 0.6,
            "drag_coefficient": 3.5,
            "pointing_accuracy_deg": 0.003,
            "housekeeping_power_w": 50.0,
            "other_dry_mass_kg": 15.0,
            "battery_mass_included_in_platform": False,
            "solar_array_mass_included_in_platform": False,
        },
        "payload": {
            "mass_kg": 176.0,
            "envelope_volume_l": None,
            "active_power_w": 250.0,
            "standby_power_w": 0.0,
            "readout_power_w": None,
            "peak_power_w": 4000.0,
            "duty_cycle": 0.1,
            "gsd_ref_alt_m": 1.0,
            "gsd_model_alt_m": 0.60,
            "swath_ref_alt_km": 10.0,
            "swath_model_alt_km": 6.0,
            "pointing_requirements_deg": None,
        },
        "battery": {
            "quantity": 12,
            "nominal_capacity_wh": 1200.0,
            "capacity_wh": 1200.0,
            "capacity_per_module_wh": 100.0,
            "usable_energy_wh": 324.0,
            "mass_per_module_kg": 0.5,
            "total_mass_kg": 6.0,
            "voltage_v": 28.0,
            "max_discharge_power_w": 2073.6,
            "max_continuous_power_w": 2073.6,
            "max_charge_power_w": None,
            "charge_efficiency": 0.90,
            "discharge_efficiency": 0.95,
            "usable_depth_of_discharge": 0.27,
            "min_state_of_charge": 0.73,
            "initial_state_of_charge": 1.0,
        },
        "solar_array": {
            "quantity": 20,
            "bol_power_per_unit_w": 45.0,
            "bol_power_w": 900.0,
            "eol_power_w": 765.0,
            "mass_per_unit_kg": 0.885,
            "total_mass_kg": 17.7,
            "deployed_area_m2": 3.8688,
            "projected_drag_area_m2": 3.8688,
            "degradation_rate_per_year": 0.15,
        },
        "propulsion": {
            "thruster_name": "RIT_10_EVO",
            "thruster_index": 0,
            "initial_propellant_mass_kg": 30.0,
        },
    },
}

def calculate_derived_quantities(case_data):
    """Calculate total case properties from selected component specifications."""
    plat = case_data.get("platform", {})
    pay = case_data.get("payload", {})
    bat = case_data.get("battery", {})
    sol = case_data.get("solar_array", {})
    prop = case_data.get("propulsion", {})
    
    bat_inc = case_data.get("battery_mass_included_in_platform", plat.get("battery_mass_included_in_platform", False))
    sol_inc = case_data.get("solar_array_mass_included_in_platform", plat.get("solar_array_mass_included_in_platform", False))
    
    # Battery capacity
    if bat.get("quantity") is not None and bat.get("capacity_per_module_wh") is not None:
        total_battery_capacity_wh = float(bat["quantity"] * bat["capacity_per_module_wh"])
    else:
        total_battery_capacity_wh = float(bat.get("capacity_wh", 0.0))
        
    # Battery mass
    if bat.get("quantity") is not None and bat.get("mass_per_module_kg") is not None:
        total_battery_mass_kg = float(bat["quantity"] * bat["mass_per_module_kg"])
    else:
        total_battery_mass_kg = bat.get("total_mass_kg", bat.get("mass_kg", None))
        
    # Solar BOL and EOL power
    if sol.get("quantity") is not None and sol.get("bol_power_per_unit_w") is not None:
        total_solar_bol_power_w = float(sol["quantity"] * sol["bol_power_per_unit_w"])
    else:
        total_solar_bol_power_w = float(sol.get("bol_power_w", 0.0))
        
    if sol.get("eol_power_w") is not None:
        total_solar_eol_power_w = float(sol["eol_power_w"])
    elif sol.get("degradation_rate_per_year") is not None:
        total_solar_eol_power_w = float(total_solar_bol_power_w * (1.0 - sol["degradation_rate_per_year"]))
    else:
        total_solar_eol_power_w = float(total_solar_bol_power_w)
        
    # Solar array mass
    if sol.get("quantity") is not None and sol.get("mass_per_unit_kg") is not None:
        total_solar_array_mass_kg = float(sol["quantity"] * sol["mass_per_unit_kg"])
    else:
        total_solar_array_mass_kg = sol.get("total_mass_kg", sol.get("mass_kg", None))
        
    # Solar array area
    if sol.get("quantity") is not None and sol.get("area_per_unit_m2") is not None:
        total_solar_array_area_m2 = float(sol["quantity"] * sol["area_per_unit_m2"])
    else:
        total_solar_array_area_m2 = float(sol.get("deployed_area_m2", 0.0))
        
    # Total drag area
    total_projected_drag_area_m2 = float(plat.get("reference_drag_area_m2", 0.0)) + (0.0 if sol_inc else float(total_solar_array_area_m2) * SOLAR_PANEL_AREA_FACTOR)
    
    # Total dry mass and wet mass
    bus_dry = float(plat.get("bus_dry_mass_kg", 0.0))
    other_dry = float(plat.get("other_dry_mass_kg", 0.0))
    pay_mass = float(pay.get("mass_kg", 0.0))
    ep_mass = float(prop.get("system_mass_kg", prop.get("mass_kg", 0.0)))
    
    bat_m = 0.0 if (bat_inc or total_battery_mass_kg is None) else float(total_battery_mass_kg)
    sol_m = 0.0 if (sol_inc or total_solar_array_mass_kg is None) else float(total_solar_array_mass_kg)
    
    total_dry_mass_kg = bus_dry + other_dry + pay_mass + ep_mass + bat_m + sol_m
    total_wet_mass_kg = total_dry_mass_kg + float(prop.get("initial_propellant_mass_kg", 0.0))
    
    # Peak power demand
    thruster_p = float(prop.get("power_W", 435.0))
    hk_p = float(plat.get("housekeeping_power_w", 50.0))
    pay_peak = float(pay.get("peak_power_w", 0.0)) if pay.get("peak_power_w") is not None else 0.0
    total_peak_power_demand_w = thruster_p + hk_p + pay_peak
    
    return {
        "total_battery_capacity_wh": total_battery_capacity_wh,
        "total_battery_mass_kg": total_battery_mass_kg,
        "total_solar_bol_power_w": total_solar_bol_power_w,
        "total_solar_eol_power_w": total_solar_eol_power_w,
        "total_solar_array_mass_kg": total_solar_array_mass_kg,
        "total_solar_array_area_m2": total_solar_array_area_m2,
        "total_projected_drag_area_m2": total_projected_drag_area_m2,
        "total_dry_mass_kg": total_dry_mass_kg,
        "total_wet_mass_kg": total_wet_mass_kg,
        "total_peak_power_demand_w": total_peak_power_demand_w,
    }

# ==============================================================================
# CUSTOM POWER SUBSYSTEM
# ==============================================================================
class TradeoffPowerSubsystem(PowerSubsystem):
    """
    Extends the existing PowerSubsystem to include:
    - Payload power (duty cycle weighted)
    - Battery charge / discharge efficiencies
    - DoD tracking and constraint
    """
    def __init__(self, bol_solar_power_w, battery_cap_wh, initial_soc,
                 charge_eff, discharge_eff, platform_hk_w, payload_active_w,
                 payload_standby_w, payload_dc, thruster_power_w, min_soc):
        
        super().__init__(
            solar_panel_area_m2=1.0,
            panel_efficiency=1.0,
            solar_flux_W_m2=bol_solar_power_w,
            battery_capacity_Wh=battery_cap_wh,
            battery_initial_Wh=battery_cap_wh * initial_soc,
            housekeeping_power_W=0.0, 
            thruster_power_W=thruster_power_w,
            panel_degradation_yr=0.0,
            battery_degradation_yr=0.0
        )
        
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.min_battery_Wh = battery_cap_wh * min_soc
        self.thruster_power_W = thruster_power_w
        
        avg_payload_load = payload_active_w * payload_dc + payload_standby_w * (1.0 - payload_dc)
        self.base_load = platform_hk_w + avg_payload_load

    def update(self, illumination_fraction, thruster_requesting, dt_s):
        P_gen = self.P_solar_max * float(illumination_fraction)
        P_cons_base = self.base_load
        
        if thruster_requesting:
            P_cons_full = P_cons_base + self.thruster_power_W
            net_with_thruster = P_gen - P_cons_full
            if self.battery_Wh <= self.min_battery_Wh and net_with_thruster < 0.0:
                thruster_allowed = False
                P_cons = P_cons_base
            else:
                thruster_allowed = True
                P_cons = P_cons_full
        else:
            thruster_allowed = True
            P_cons = P_cons_base
            
        net_power_W = P_gen - P_cons
        
        if net_power_W > 0:
            dE_Wh = (net_power_W * self.charge_eff) * float(dt_s) / 3600.0
        else:
            dE_Wh = (net_power_W / self.discharge_eff) * float(dt_s) / 3600.0
            
        self.battery_Wh = max(0.0, min(self.battery_capacity_Wh, self.battery_Wh + dE_Wh))
        
        return thruster_allowed, P_gen, P_cons

# ==============================================================================
# VALIDATION
# ==============================================================================
def validate_case(case_name, case_data):
    """
    Validate case completeness and static engineering constraints.
    Returns a dictionary reporting any missing parameters and failed constraint checks.
    Does NOT raise ValueError on engineering constraint violations so all cases can be compared.
    """
    required_sections = ["platform", "payload", "battery", "solar_array", "propulsion"]
    for sec in required_sections:
        if sec not in case_data:
            raise ValueError(f"Case '{case_name}' is missing required section '{sec}'.")
            
    missing_parameters = []
    for sec, params in case_data.items():
        if isinstance(params, dict):
            for k, v in params.items():
                if v is None:
                    missing_parameters.append(f"{sec}.{k}")
                    
    failed_checks = []
    plat = case_data["platform"]
    pay = case_data["payload"]
    bat = case_data["battery"]
    sol = case_data["solar_array"]
    
    # 1. Platform payload mass capacity check
    if pay.get("mass_kg") is not None and plat.get("maximum_payload_mass_kg") is not None:
        if float(pay["mass_kg"]) > float(plat["maximum_payload_mass_kg"]):
            failed_checks.append(f"Payload mass ({pay['mass_kg']} kg) exceeds platform max payload mass ({plat['maximum_payload_mass_kg']} kg)")
            
    # 2. Platform payload power availability checks
    if pay.get("active_power_w") is not None and plat.get("avg_payload_power_w") is not None:
        if float(pay["active_power_w"]) > float(plat["avg_payload_power_w"]):
            failed_checks.append(f"Payload active power ({pay['active_power_w']} W) exceeds platform avg payload power ({plat['avg_payload_power_w']} W)")
            
    if pay.get("peak_power_w") is not None and plat.get("peak_payload_power_w") is not None:
        if float(pay["peak_power_w"]) > float(plat["peak_payload_power_w"]):
            failed_checks.append(f"Payload peak power ({pay['peak_power_w']} W) exceeds platform peak payload power ({plat['peak_payload_power_w']} W)")
            
    if pay.get("peak_power_w") is not None and bat.get("max_continuous_power_w") is not None:
        if float(pay["peak_power_w"]) > float(bat["max_continuous_power_w"]):
            failed_checks.append(f"Payload peak power ({pay['peak_power_w']} W) exceeds battery max continuous power ({bat['max_continuous_power_w']} W)")
            
    # 3. Payload envelope volume capacity check
    if pay.get("envelope_volume_l") is not None and plat.get("payload_volume_l") is not None:
        if float(pay["envelope_volume_l"]) > float(plat["payload_volume_l"]):
            failed_checks.append(f"Payload envelope volume ({pay['envelope_volume_l']} L) exceeds platform payload volume ({plat['payload_volume_l']} L)")
            
    # 4. Battery SoC sanity check
    if bat.get("initial_state_of_charge") is not None:
        if float(bat["initial_state_of_charge"]) < 0.0 or float(bat["initial_state_of_charge"]) > 1.0:
            failed_checks.append("initial_state_of_charge must be between 0.0 and 1.0")
            
    # Also check global configuration
    if MIN_DUTY_CYCLE_ALTITUDE_KM >= MAX_DUTY_CYCLE_ALTITUDE_KM:
        raise ValueError("Global configuration: MIN_DUTY_CYCLE_ALTITUDE_KM >= MAX_DUTY_CYCLE_ALTITUDE_KM")
        
    return {
        "case_name": case_name,
        "missing_parameters": missing_parameters,
        "failed_checks": failed_checks,
        "feasible_preliminary": len(failed_checks) == 0
    }

# ==============================================================================
# EXECUTION
# ==============================================================================
def run_tradeoff_case(case_name, case_data, output_dir):
    """Run a single case and generate outputs without halting on constraint failures."""
    print(f"\n=======================================================================")
    print(f"--- Running {case_name}: {case_data.get('name', case_name)} ---")
    print(f"=======================================================================")
    
    val_report = validate_case(case_name, case_data)
    if val_report["missing_parameters"]:
        print(f"  [Notice] Missing parameters (fidelity impact): {', '.join(val_report['missing_parameters'])}")
    if val_report["failed_checks"]:
        print(f"  [Warning] Preliminary constraint checks failed: {'; '.join(val_report['failed_checks'])}")
    else:
        print("  [Pass] All preliminary engineering constraint checks passed.")
        
    prop = case_data.get("propulsion", {})
    if "thruster_name" in prop:
        thruster_config = get_thruster_config(prop["thruster_name"], prop.get("thruster_index", 0))
        if not thruster_config:
            raise ValueError(f"Case '{case_name}': Thruster '{prop['thruster_name']}' not found in mission_config.")
        for k, v in thruster_config.items():
            if k not in prop:
                prop[k] = v
        
    derived = calculate_derived_quantities(case_data)
    case_data["derived"] = derived
    
    plat = case_data["platform"]
    pay = case_data["payload"]
    bat = case_data["battery"]
    solar = case_data["solar_array"]
    prop = case_data["propulsion"]
        
    total_mass = derived["total_wet_mass_kg"]
    total_area = derived["total_projected_drag_area_m2"]
    
    utc = TimeScalesFactory.getUTC()
    start_date = AbsoluteDate(
        SIMULATION_START_YEAR, SIMULATION_START_MONTH, SIMULATION_START_DAY,
        SIMULATION_START_HOUR, SIMULATION_START_MINUTE, SIMULATION_START_SECOND, utc
    )
    
    prop_system = ElectricPropulsionSystem(
        thrust=prop.get("thrust_N"),
        isp=prop.get("isp_s"),
        initial_propellant_mass=prop["initial_propellant_mass_kg"],
        h_min=MIN_DUTY_CYCLE_ALTITUDE_KM * 1000.0,
        h_max=MAX_DUTY_CYCLE_ALTITUDE_KM * 1000.0,
        max_burn_time=prop.get("max_operating_time_s", prop.get("total_lifetime_h", 0) * 3600.0),
        max_cycles=prop.get("max_cycles", prop.get("max_operational_cycles", 0))
    )
    
    power_system = TradeoffPowerSubsystem(
        bol_solar_power_w=derived["total_solar_eol_power_w"],
        battery_cap_wh=derived["total_battery_capacity_wh"],
        initial_soc=bat["initial_state_of_charge"],
        charge_eff=bat["charge_efficiency"],
        discharge_eff=bat["discharge_efficiency"],
        platform_hk_w=plat["housekeeping_power_w"],
        payload_active_w=pay["active_power_w"],
        payload_standby_w=pay["standby_power_w"],
        payload_dc=pay["duty_cycle"],
        thruster_power_w=prop.get("power_W"),
        min_soc=bat.get("min_state_of_charge", 1.0 - bat["usable_depth_of_discharge"])
    )
    
    params = {
        "start_date": start_date,
        "altitude": INITIAL_ALTITUDE_KM * 1000.0,
        "inclination": INITIAL_INCLINATION_DEG,
        "eccentricity": INITIAL_ECCENTRICITY,
        "raan": INITIAL_RAAN_DEG,
        "arg_perigee": INITIAL_ARG_PERIGEE_DEG,
        "true_anomaly": INITIAL_TRUE_ANOMALY_DEG,
        "mass": total_mass,
        "cross_section": total_area,
        "drag_coeff": plat["drag_coefficient"],
        "duration": SIMULATION_DURATION_DAYS * 86400.0,
        "time_step": OUTPUT_TIMESTEP_S,
        "power_model": power_system
    }
    
    print(f"  Propagating for {SIMULATION_DURATION_DAYS} days with {OUTPUT_TIMESTEP_S}s step (Wet Mass: {total_mass:.2f} kg, Drag Area: {total_area:.4f} m2)...")
    results = atmospheric_model.run_simulation(
        params=params, 
        model_type=ATMOSPHERIC_MODEL,
        propulsion_model=prop_system,
        compensation_mode="duty_cycle"
    )
    
    if results.get("status") != "success":
        print(f"  [Error] Simulation {case_name} failed.")
        return None
        
    generate_outputs(case_name, case_data, params, prop_system, power_system, results, output_dir, val_report)
    summary = compile_summary(case_name, case_data, params, prop_system, power_system, results, val_report)
    print(f"  [Completed] Feasibility: {summary['feasibility_status']} | Min Altitude: {summary['minimum_altitude_km']:.2f} km | Prop Used: {summary['propellant_consumed_kg']:.3f} kg | Min SoC: {summary['minimum_battery_soc']*100:.1f}%")
    return summary

def generate_outputs(case_name, case_data, params, prop_model, power_model, results, out_dir, val_report=None):
    prop = case_data.get("propulsion", {})
    thruster_name = prop.get("name", prop.get("thruster_name", "Unknown")).replace(" ", "_")
    thruster_label = prop.get("label", str(prop.get("thruster_index", "0"))).replace(" ", "_")
    
    out_path = Path(out_dir) / case_name / f"{thruster_name}_{thruster_label}"
    out_path.mkdir(parents=True, exist_ok=True)
    
    if val_report is None:
        val_report = validate_case(case_name, case_data)
        
    # 1. Configuration JSON
    with open(out_path / "configuration.json", "w") as f:
        json.dump(case_data, f, indent=4)
        
    with open(out_path / "validation_report.json", "w") as f:
        json.dump(val_report, f, indent=4)
        
    # Extract times
    times = np.array(results["time"])
    days = times / 86400.0
    alts = np.array(results["altitude"])
    sma_km = np.array(results.get("sma", alts + 6371.0))
    bat_wh = np.array(results["battery_Wh"])
    p_gen = np.array(results["power_gen_W"])
    p_cons = np.array(results["power_cons_W"])
    thrust_on = np.array(results["thrust_on"])
    prop_rem = np.array(results["propellant_remaining"])
    illum = np.array(results["illumination"])
    
    # Generate CSVs
    _write_csv(out_path / "time_history.csv", ["Elapsed_s", "Altitude_km", "Density_kg_m3", "Inclination_deg", "Eccentricity", "RAAN_deg"], 
               times, alts, results["density"], results["inclination"], results["eccentricity"], results["raan"])
    _write_csv(out_path / "orbital_elements.csv", ["Elapsed_s", "SMA_km", "Eccentricity", "Inclination_deg", "RAAN_deg"], 
               times, sma_km, results["eccentricity"], results["inclination"], results["raan"])
    _write_csv(out_path / "power_budget.csv", ["Elapsed_s", "Generated_W", "Consumed_W", "Net_W", "Illumination"], 
               times, p_gen, p_cons, p_gen - p_cons, illum)
    _write_csv(out_path / "battery_history.csv", ["Elapsed_s", "Battery_Wh"], times, bat_wh)
    _write_csv(out_path / "thruster_history.csv", ["Elapsed_s", "Thrust_On", "Propellant_Remaining_kg", "Cycles"], 
               times, thrust_on, prop_rem, results["cycles_list"])
    
    with open(out_path / "eclipse_history.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Event_Midpoint_s", "Duration_s"])
        for ev in results.get("umbra_events", []):
            writer.writerow([ev[0], ev[1]])

    def _save_fig(fig, filename):
        fig.savefig(out_path / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    # Plot 1: Altitude Evolution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, alts, label="Instantaneous Altitude", color=_ops.COLORS["primary"])
    ax.axhline(MIN_DUTY_CYCLE_ALTITUDE_KM, color=_ops.COLORS["threshold_lower"], linestyle=":", label="Alt Lower Limit")
    ax.axhline(MAX_DUTY_CYCLE_ALTITUDE_KM, color=_ops.COLORS["threshold_upper"], linestyle=":", label="Alt Upper Limit")
    ax.set_title(f"Altitude Evolution - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Altitude (km)")
    ax.legend()
    _save_fig(fig, "altitude_evolution.png")
    
    # Plot 2: Orbital Parameters
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(days, results["inclination"], color=_ops.COLORS["secondary"])
    axs[0].set_ylabel("Inclination (deg)")
    axs[1].plot(days, results["eccentricity"], color=_ops.COLORS["validated"])
    axs[1].set_ylabel("Eccentricity")
    axs[2].plot(days, results["raan"], color=_ops.COLORS["reference"])
    axs[2].set_ylabel("RAAN (deg)")
    axs[2].set_xlabel("Time (days)")
    fig.suptitle(f"Orbital Parameters - {case_name}")
    _save_fig(fig, "orbital_parameters.png")
    
    # Plot 3: Propellant Mass
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, prop_rem, label="Propellant Remaining", color=_ops.COLORS["primary"])
    ax.set_title(f"Propellant Mass - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mass (kg)")
    ax.legend()
    _save_fig(fig, "propellant_mass.png")
    
    # Plot 4: Battery State of Charge
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, bat_wh, label="Battery SoC", color=_ops.COLORS["primary"])
    ax.set_title(f"Battery State of Charge - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Charge (Wh)")
    ax.legend()
    _save_fig(fig, "battery_soc.png")
    
    # Plot 5: Eclipse Evolution
    fig, ax = plt.subplots(figsize=(8, 4))
    if results.get("umbra_events"):
        ev_times = [ev[0]/86400.0 for ev in results["umbra_events"]]
        ev_durs = [ev[1]/60.0 for ev in results["umbra_events"]]
        ax.scatter(ev_times, ev_durs, color=_ops.COLORS["validated"], s=10)
    ax.set_title(f"Eclipse Duration per Orbit - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Duration (minutes)")
    _save_fig(fig, "eclipse_periods.png")
    
    # Plot 5: Power Budget
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, p_gen, label="Available Generated", color=_ops.COLORS["primary"], alpha=0.8)
    ax.plot(days, p_cons, label="Total Demand", color=_ops.COLORS["secondary"], alpha=0.8)
    ax.set_title(f"Power Budget - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    _save_fig(fig, "power_budget.png")
    
    # Plot 6: Power Generation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, p_gen, label="Instantaneous Generation", color=_ops.COLORS["primary"])
    ax.set_title(f"Power Generation - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    _save_fig(fig, "power_generation.png")
    
    # Plot 7: Battery Cycles
    fig, ax = plt.subplots(figsize=(8, 4))
    soc = bat_wh / power_model.battery_capacity_Wh
    ax.plot(days, soc * 100, label="State of Charge", color=_ops.COLORS["primary"])
    ax.axhline((1.0 - case_data["battery"]["usable_depth_of_discharge"])*100, color=_ops.COLORS["threshold_lower"], linestyle="--", label="Min Allowable SoC")
    ax.set_title(f"Battery State of Charge - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SoC (%)")
    ax.legend()
    _save_fig(fig, "battery_cycles.png")
    
    # Plot 8: Thruster Cycles
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(days, thrust_on, label="Thruster State", color=_ops.COLORS["reference"])
    ax.set_title(f"Thruster Operation - {case_name}")
    ax.set_xlabel("Time (days)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OFF", "ON"])
    ax.legend()
    _save_fig(fig, "thruster_cycles.png")
    
    # Write summary CSV and JSON
    summary = compile_summary(case_name, case_data, params, prop_model, power_model, results, val_report)
    with open(out_path / "simulation_summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for k, v in summary.items():
            writer.writerow([k, v])
            
    with open(out_path / "simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

def _write_csv(path, headers, *columns):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*columns):
            writer.writerow(row)

def compile_summary(case_name, case_data, params, prop_model, power_model, results, val_report=None):
    if val_report is None:
        val_report = validate_case(case_name, case_data)
        
    bat_wh = np.array(results["battery_Wh"])
    soc = bat_wh / power_model.battery_capacity_Wh
    min_soc_val = float(np.min(soc))
    min_allowable_soc = 1.0 - case_data["battery"]["usable_depth_of_discharge"]
    
    operational_fails = []
    if min_soc_val < min_allowable_soc:
        operational_fails.append(f"Battery SoC dropped below min limit ({min_allowable_soc*100:.1f}%) during cycle execution")
        
    all_fails = val_report.get("failed_checks", []) + operational_fails
    is_feasible = (len(all_fails) == 0)
    
    derived = case_data.get("derived", calculate_derived_quantities(case_data))
    
    summary = {
        "case_name": case_name,
        "case_title": case_data.get("name", case_name),
        "platform_id": case_data.get("platform_id", ""),
        "payload_id": case_data.get("payload_id", ""),
        "battery_id": case_data.get("battery_id", ""),
        "solar_array_id": case_data.get("solar_array_id", ""),
        "feasibility_status": "Feasible" if is_feasible else "Feasible: No",
        "mission_feasibility_status": "Success" if is_feasible else ("Failed - " + "; ".join(all_fails)),
        "failed_checks": "; ".join(all_fails) if all_fails else "None",
        "missing_parameters": "; ".join(val_report.get("missing_parameters", [])) if val_report.get("missing_parameters") else "None",
        "total_dry_mass_kg": derived["total_dry_mass_kg"],
        "initial_wet_mass_kg": params["mass"],
        "final_spacecraft_mass_kg": float(results["mass"][-1]),
        "propellant_consumed_kg": float(prop_model.propellant_used),
        "remaining_propellant_kg": float(prop_model.propellant_mass),
        "total_thruster_operating_time_s": float(prop_model.burn_time),
        "total_thruster_cycles": int(prop_model.cycles),
        "simulated_thruster_duty_cycle_pct": float(np.mean(results["thrust_on"]) * 100.0),
        "minimum_altitude_km": float(np.min(results["altitude"])),
        "maximum_altitude_km": float(np.max(results["altitude"])),
        "mean_altitude_km": float(np.mean(results["altitude"])),
        "minimum_battery_soc": min_soc_val,
        "termination_reason": prop_model.shutdown_reason if prop_model.shutdown_reason != "None" else "End of Simulation",
    }
    
    if summary["termination_reason"] == "Simulation Exception":
        summary["feasibility_status"] = "Feasible: No"
        summary["mission_feasibility_status"] = "Failed - Simulation Exception"
        
    return summary

def generate_global_comparison(summaries, output_dir):
    out_path = Path(output_dir) / "tradeoff_comparison_plots"
    out_path.mkdir(parents=True, exist_ok=True)
    
    if summaries:
        keys = summaries[0].keys()
        with open(Path(output_dir) / "all_cases_summary.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
            
        with open(Path(output_dir) / "all_cases_summary.json", "w") as f:
            json.dump(summaries, f, indent=4)
            
        # Write markdown feasibility report
        with open(Path(output_dir) / "all_cases_feasibility_report.md", "w") as f:
            f.write("# Spacecraft Trade-Off Cases Feasibility and Validation Report\n\n")
            f.write("| Case ID | Case Title | Platform | Payload | Dry Mass (kg) | Wet Mass (kg) | Feasible? | Failed Checks |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for s in summaries:
                f.write(f"| {s['case_name']} | {s['case_title']} | {s['platform_id']} | {s['payload_id']} | {s['total_dry_mass_kg']:.2f} | {s['initial_wet_mass_kg']:.2f} | **{s['feasibility_status']}** | {s['failed_checks']} |\n")
                
        # Chart 1: Propellant Consumption Comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        cases = [s["case_name"] for s in summaries]
        prop_used = [s["propellant_consumed_kg"] for s in summaries]
        bars = ax.bar(cases, prop_used, color=_ops.COLORS["primary"])
        ax.set_ylabel("Propellant Consumed (kg)")
        ax.set_title("Propellant Consumption Comparison Across All 8 Cases")
        plt.xticks(rotation=45)
        for bar, s in zip(bars, summaries):
            if s["feasibility_status"] != "Feasible":
                bar.set_color(_ops.COLORS.get("threshold_upper", "#d9534f"))
        fig.savefig(out_path / "propellant_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Chart 2: Total Dry vs Wet Mass Comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        dry_masses = [s["total_dry_mass_kg"] for s in summaries]
        wet_masses = [s["initial_wet_mass_kg"] for s in summaries]
        x = np.arange(len(cases))
        width = 0.35
        ax.bar(x - width/2, dry_masses, width, label="Dry Mass", color=_ops.COLORS["secondary"])
        ax.bar(x + width/2, wet_masses, width, label="Wet Mass", color=_ops.COLORS["primary"])
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Spacecraft Mass Comparison (Dry vs Wet)")
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=45)
        ax.legend()
        fig.savefig(out_path / "mass_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Chart 3: Simulated Thruster Duty Cycle Comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        dc_vals = [s.get("simulated_thruster_duty_cycle_pct", 0.0) for s in summaries]
        ax.bar(cases, dc_vals, color=_ops.COLORS["validated"])
        ax.set_ylabel("Simulated Thruster Duty Cycle (%)")
        ax.set_title("Simulated Thruster Duty Cycle (%) Across All 8 Cases")
        plt.xticks(rotation=45)
        fig.savefig(out_path / "thruster_duty_cycle_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run tradeoff case simulations.")
    parser.add_argument("--all", action="store_true", help="Run all enabled cases.")
    parser.add_argument("--case", type=str, help="Run a specific case by name.")
    parser.add_argument("--cases", type=str, nargs="+", help="Run a list of cases.")
    
    args = parser.parse_args()
    
    cases_to_run = []
    if args.all:
        cases_to_run = [k for k, v in TRADEOFF_CASES.items() if v.get("enabled", True)]
    elif args.case:
        if args.case in TRADEOFF_CASES:
            cases_to_run = [args.case]
        else:
            print(f"Error: Case {args.case} not found.")
            return
    elif args.cases:
        for c in args.cases:
            if c in TRADEOFF_CASES:
                cases_to_run.append(c)
            else:
                print(f"Error: Case {c} not found.")
                return
    else:
        print("\nSelect a trade-off case to run:")
        case_keys = list(TRADEOFF_CASES.keys())
        for i, k in enumerate(case_keys, 1):
            name = TRADEOFF_CASES[k].get("name", k)
            print(f"{i}. {k}: {name}")
        print(f"{len(case_keys) + 1}. All Cases (Run full trade-off suite and compare)")
        
        choice = input(f"\nEnter your choice (1-{len(case_keys) + 1}): ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(case_keys):
                selected_key = case_keys[choice_num - 1]
                cases_to_run = [selected_key]
            elif choice_num == len(case_keys) + 1:
                cases_to_run = [k for k, v in TRADEOFF_CASES.items() if v.get("enabled", True)]
            else:
                print("Invalid choice. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return
        
    out_dir = PROJECT_ROOT / "results_tradeoffs"
    if not OVERWRITE_EXISTING_RESULTS:
        out_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summaries = []
    for case_id in cases_to_run:
        summary = run_tradeoff_case(case_id, TRADEOFF_CASES[case_id], out_dir)
        if summary:
            summaries.append(summary)
            
    if summaries:
        generate_global_comparison(summaries, out_dir)
        print(f"\nAll requested cases complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
