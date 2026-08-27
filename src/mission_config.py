"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          mission_config.py
Description:
    Mission parameters, spacecraft physical constants, and ArianeGroup RIT thruster catalog database.
===============================================================================
"""

# ================================================================
# SPACECRAFT
# ================================================================
SPACECRAFT = {
    "name":                 "Generic LEO Satellite",

    # Physical properties
    "mass_kg":              500.0,        # [kg]   dry mass (without propellant)

    # Aerodynamic / SRP geometry
    "cross_section_m2":     0.5,          # [m²]   cross-sectional area for drag & SRP
    "drag_coeff":           3.6,          # [-]    aerodynamic drag coefficient Cd
    "reflectivity_coeff":   1.5,          # [-]    SRP reflectivity coefficient Cr
}

# ================================================================
# ORBIT  (initial conditions; circular orbit assumed)
# ================================================================
ORBIT = {
    # Altitude & shape
    "altitude_km":          320.0,        # [km]   initial altitude above WGS84 equatorial radius
    "eccentricity":         1e-8,         # [-]    near-circular

    # Orientation
    "inclination_deg":      96.67,         # [deg]  orbital inclination
    "raan_deg":             11.5,          # [deg]  right ascension of ascending node
    "arg_perigee_deg":      0.0,          # [deg]  argument of perigee
    "true_anomaly_deg":     0.0,          # [deg]  true anomaly at epoch

    # Epoch
    "start_year":           2026,
    "start_month":          1,
    "start_day":            1,
    "start_hour":           0,
    "start_minute":         0,
    "start_second":         0,
}

# ================================================================
# RIT THRUSTER FAMILY CATALOGUE  (Ariane Group / ECAPS)
# ================================================================
# Each thruster entry contains only parameters relevant to orbital
# simulation (thrust, Isp, power, mass, propellant, lifetime).
# Environmental / vibration / shock / RF data are intentionally omitted.
#
# Multi-point thrusters (RIT 10 EVO, RIT 2X) expose a list of
# operating points ordered from lowest to highest thrust level.
# Use get_thruster_config() below to select a specific point.
#
# Sources: Ariane Group RIT Thruster Family Performance Data sheet.
# ================================================================

RIT_THRUSTERS = {

    # ------------------------------------------------------------------
    # RIT µX  —  micro thruster for very small platforms
    # ------------------------------------------------------------------
    "RIT_uX": {
        "name":             "RIT µX",
        "description":      "Miniaturised RF ion thruster for micro/nano-satellites",

        # Single operating range (thrust is continuously throttleable)
        "operating_points": [
            {
                "label":        "nominal",
                "thrust_N":     250e-6,         # [N]   mid-range of 50–500 µN; use 250 µN as representative
                "thrust_min_N": 50e-6,          # [N]   minimum throttle point
                "thrust_max_N": 500e-6,         # [N]   maximum throttle point
                "isp_s":        1650,           # [s]   mid-range of 300–3000 s demonstrated
                "isp_min_s":    300,            # [s]   lower end of demonstrated Isp range
                "isp_max_s":    3000,           # [s]   upper end of demonstrated Isp range
                "power_W":      50.0,           # [W]   nominal power draw (< 50 W)
            },
        ],

        # Design / mechanical
        "mass_kg":              0.440,          # [kg]  thruster mass
        "diameter_m":           0.078,          # [m]   78 mm
        "height_m":             0.076,          # [m]   76 mm
        "propellant":           "Xenon",        # [-]   propellant type

        # Lifetime / reliability
        "total_impulse_Ns":     10e3,           # [N·s] > 10 kN·s  (up to 200 kN·s)
        "max_operational_cycles": 10000,        # [-]   > 10 000 cycles
        "total_lifetime_h":     20000,          # [h]   > 20 000 h

        # Technology
        "ionisation":           "RF-Principle",
        "acceleration":         "Electrostatic",
        "grid_system":          "2 Grids",
    },

    # ------------------------------------------------------------------
    # RIT 10 EVO  —  heritage mid-class thruster (3 operating points)
    # ------------------------------------------------------------------
    "RIT_10_EVO": {
        "name":             "RIT 10 EVO",
        "description":      "Heritage RF ion thruster; three discrete thrust levels",

        # Three certified operating points
        "operating_points": [
            {
                "label":    "5 mN",
                "thrust_N": 5e-3,               # [N]
                "isp_s":    1900,               # [s]   > 1900 s
                "power_W":  145.0,              # [W]
            },
            {
                "label":    "15 mN",
                "thrust_N": 15e-3,              # [N]
                "isp_s":    3000,               # [s]   > 3000 s
                "power_W":  435.0,              # [W]
            },
            {
                "label":    "25 mN",
                "thrust_N": 25e-3,              # [N]
                "isp_s":    3200,               # [s]   > 3200 s (max demonstrated 3400 s)
                "power_W":  760.0,              # [W]
            },
        ],

        # Design / mechanical
        "mass_kg":              1.8,            # [kg]
        "diameter_m":           0.186,          # [m]   186 mm
        "height_m":             0.134,          # [m]   134 mm
        "propellant":           "Xenon",

        # Lifetime / reliability
        "total_impulse_Ns":     1.1e6,          # [N·s] > 1.1 MN·s
        "max_operational_cycles": 10000,        # [-]   > 10 000 cycles
        "total_lifetime_h":     20000,          # [h]   > 20 000 h

        # Technology
        "ionisation":           "RF-Principle",
        "acceleration":         "Electrostatic",
        "grid_system":          "2 Grids",
    },

    # ------------------------------------------------------------------
    # RIT 2X  —  high-thrust class thruster (3 operating points)
    # ------------------------------------------------------------------
    "RIT_2X": {
        "name":             "RIT 2X",
        "description":      "High-thrust RF ion thruster for large GEO/science missions",

        # Three certified operating points
        "operating_points": [
            {
                "label":    "70-88 mN",
                "thrust_N": 79e-3,              # [N]   representative mid-point of 70–88 mN
                "thrust_min_N": 70e-3,          # [N]
                "thrust_max_N": 88e-3,          # [N]
                "isp_s":    3450,               # [s]   mid of 3400–3500 s range
                "isp_min_s": 3400,
                "isp_max_s": 3500,
                "power_W":  2250.0,             # [W]   mid of 2000–2500 W range
                "power_min_W": 2000.0,
                "power_max_W": 2500.0,
            },
            {
                "label":    "151-171 mN",
                "thrust_N": 161e-3,             # [N]   mid-point of 151–171 mN
                "thrust_min_N": 151e-3,
                "thrust_max_N": 171e-3,
                "isp_s":    3400,               # [s]   mid of 3300–3500 s range
                "isp_min_s": 3300,
                "isp_max_s": 3500,
                "power_W":  4250.0,             # [W]   mid of 4000–4500 W range
                "power_min_W": 4000.0,
                "power_max_W": 4500.0,
            },
            {
                "label":    "198-215 mN",
                "thrust_N": 206e-3,             # [N]   mid-point of 198–215 mN
                "thrust_min_N": 198e-3,
                "thrust_max_N": 215e-3,
                "isp_s":    2600,               # [s]   mid of 2450–2750 s range
                "isp_min_s": 2450,
                "isp_max_s": 2750,
                "power_W":  5050.0,             # [W]   mid of 4800–5300 W range
                "power_min_W": 4800.0,
                "power_max_W": 5300.0,
            },
        ],

        # Design / mechanical
        "mass_kg":              10.0,           # [kg]  < 10 kg
        "diameter_m":           0.330,          # [m]   < 330 mm
        "height_m":             0.220,          # [m]   < 220 mm
        "propellant":           "Xenon",

        # Lifetime / reliability
        "total_impulse_Ns":     10e6,           # [N·s] > 10 MN·s
        "max_operational_cycles": 10000,        # [-]   > 10 000 cycles
        "total_lifetime_h":     20000,          # [h]   > 20 000 h

        # Technology
        "ionisation":           "RF-Principle",
        "acceleration":         "Electrostatic",
        "grid_system":          "2 Grids",
    },
}


def get_thruster_config(thruster_key: str, operating_point_index: int = 0) -> dict:
    """
    Return a flat propulsion-parameter dict ready to feed into PROPULSION,
    for a given RIT thruster and operating point.

    Parameters
    ----------
    thruster_key : str
        Key in RIT_THRUSTERS, e.g. "RIT_uX", "RIT_10_EVO", "RIT_2X".
    operating_point_index : int, optional
        Index into the thruster's ``operating_points`` list (default 0 =
        lowest / nominal thrust level).

    Returns
    -------
    dict
        {"thrust_N", "isp_s", "power_W", "mass_kg", "propellant",
         "total_lifetime_h", "max_operational_cycles", "name", "label"}

    Example
    -------
    >>> cfg = get_thruster_config("RIT_10_EVO", operating_point_index=1)  # 15 mN point
    >>> PROPULSION["thrust_N"] = cfg["thrust_N"]
    >>> PROPULSION["isp_s"]    = cfg["isp_s"]
    >>> PROPULSION["power_W"]  = cfg["power_W"]
    """
    thruster = RIT_THRUSTERS[thruster_key]
    point = thruster["operating_points"][operating_point_index]
    return {
        "name":                     thruster["name"],
        "label":                    point["label"],
        "thrust_N":                 point["thrust_N"],
        "isp_s":                    point["isp_s"],
        "power_W":                  point["power_W"],
        "mass_kg":                  thruster["mass_kg"],
        "propellant":               thruster["propellant"],
        "total_lifetime_h":         thruster["total_lifetime_h"],
        "max_operational_cycles":   thruster["max_operational_cycles"],
        "total_impulse_Ns":         thruster["total_impulse_Ns"],
    }


# ================================================================
# PROPULSION SYSTEM
# ================================================================
# Select the active thruster here.  Choose any key from RIT_THRUSTERS
# and the desired operating-point index (0 = lowest thrust level).
#
#   "RIT_uX"      → index 0  (only one point: nominal 250 µN)
#   "RIT_10_EVO"  → index 0 / 1 / 2  (5 mN / 15 mN / 25 mN)
#   "RIT_2X"      → index 0 / 1 / 2  (79 mN / 161 mN / 206 mN)
# ================================================================
_ACTIVE_THRUSTER_KEY   = "RIT_uX"   # ← change to select thruster model
_ACTIVE_THRUSTER_POINT = 0              # ← change to select operating point

_thruster_cfg = get_thruster_config(_ACTIVE_THRUSTER_KEY, _ACTIVE_THRUSTER_POINT)

PROPULSION = {
    # --- Thruster identity (auto-populated from RIT_THRUSTERS) ---
    "thruster_name":        _thruster_cfg["name"],         # [-]    model name
    "thruster_label":       _thruster_cfg["label"],        # [-]    operating-point label

    # --- Performance (auto-populated from selected operating point) ---
    "thrust_N":             _thruster_cfg["thrust_N"],     # [N]    nominal thrust
    "isp_s":                _thruster_cfg["isp_s"],        # [s]    specific impulse
    "power_W":              _thruster_cfg["power_W"],      # [W]    electrical power draw

    # --- Propellant budget (set manually for the mission) ---
    "propellant_mass_kg":   5.0,                           # [kg]   initial propellant load

    # --- Station-keeping thresholds (used in duty-cycle mode) ---
    "h_min_km":             260.0,                         # [km]   fire below this altitude
    "h_max_km":             320.0,                         # [km]   stop firing above this altitude
}

# ================================================================
# POWER SUBSYSTEM
# ================================================================
POWER = {
    # Solar generation
    "solar_panel_area_m2":          5.0,    # [m²]   total solar panel area
    "panel_efficiency":             0.28,   # [-]    solar cell efficiency (BOL, beginning of life)
    "solar_flux_W_m2":              1361.0, # [W/m²] mean solar irradiance at 1 AU

    # Solar panel degradation
    # Typical GaAs triple-junction panels lose ~2-3 %/yr in LEO.
    # Set to 0.0 to disable degradation modelling.
    "solar_panel_degradation_per_year": 0.025,  # [-/yr]  fractional efficiency loss per year
                                                 #         e.g. 0.025 = 2.5 %/yr

    # Battery
    "battery_capacity_Wh":          300.0,  # [Wh]   total usable battery capacity
    "battery_initial_Wh":           300.0,  # [Wh]   initial state of charge (fully charged)

    # Battery degradation
    # Capacity fade is modelled as a linear loss over the mission lifetime.
    # Set to 0.0 to disable degradation modelling.
    "battery_degradation_per_year": 0.05,   # [-/yr]  fractional capacity loss per year
                                             #         e.g. 0.05 = 5 %/yr (10 % over 2 yr mission)

    # Platform loads (excluding propulsion)
    "housekeeping_power_W":         80.0,   # [W]    constant platform power draw
                                            #        (OBDH, comms, ADCS, thermal, etc.)
}

# ================================================================
# SIMULATION SETTINGS
# ================================================================
SIMULATION = {
    "duration_days":        4 * 365.0,          # [days] total simulation duration
    "time_step_s":          3600.0,       # [s]    output time step (1 orbit ≈ 5400 s at 400 km)
    "atmosphere_model":     "nrlmsise00", # density model: "nrlmsise00" | "jb2008" |
                                          #                "dtm2000"    | "harrispriester"
    "compensation_mode":    "duty_cycle", # thrust logic:  "duty_cycle" | "maintenance" |
                                          #                "goal"       | "staged_tracking"
}
