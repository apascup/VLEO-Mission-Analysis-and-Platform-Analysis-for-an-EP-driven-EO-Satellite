"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          staged_mission_validation.py
Description:
    Staged altitude profile tracking validation against historical GOCE mission telemetry and TLEs.
===============================================================================
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors
from datetime import datetime, timedelta

# Ensure src is in the path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import orbital_plot_style as _ops

import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception as e:
    pass

from orekit_jpype.pyhelpers import setup_orekit_data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
setup_orekit_data(filenames=os.path.join(project_root, "orekit-data-main"), from_pip_library=False)

from orbital_models.electric_propulsion import ElectricPropulsionSystem
from orbital_models.atmospheric_model import run_simulation
from validations.orbital_decay_validation import (
    parse_tle_file, 
    SPACECRAFT_REGISTRY
)
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants

def propagate_tle_at_epoch(name, line1, line2):
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    from org.orekit.orbits import KeplerianOrbit
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import Constants, IERSConventions
    import math

    inertial_frame = FramesFactory.getEME2000()
    tle = TLE(line1, line2)
    propagator = TLEPropagator.selectExtrapolator(tle)
    epoch = tle.getDate()
    state = propagator.propagate(epoch)

    kep = KeplerianOrbit(state.getOrbit())
    inc = kep.getI()
    raan = kep.getRightAscensionOfAscendingNode()
    arg_perigee = kep.getPerigeeArgument()
    true_anom = kep.getTrueAnomaly()

    mean_motion = tle.getMeanMotion()
    mean_a = math.pow(Constants.WGS84_EARTH_MU / (mean_motion * mean_motion), 1.0 / 3.0)
    
    altitude_km = (mean_a - Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 1000.0

    return {
        "epoch":            epoch,
        "altitude_km":      altitude_km,
        "sma_m":            kep.getA(),
        "mean_sma_m":       mean_a,
        "eccentricity":     kep.getE(),
        "inclination_deg":  math.degrees(inc),
        "raan_deg":         math.degrees(raan) % 360.0,
        "arg_perigee_deg":  math.degrees(arg_perigee) % 360.0,
        "true_anomaly_deg": math.degrees(true_anom) % 360.0,
    }

def get_truth_data(tle_entries):
    truth = []
    for name, l1, l2 in tle_entries:
        truth.append(propagate_tle_at_epoch(name, l1, l2))
    if not truth:
        return truth
    ref_epoch = truth[0]["epoch"]
    truth.sort(key=lambda d: d["epoch"].durationFrom(ref_epoch))
    return truth

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "jb2008"
SPACECRAFT = SPACECRAFT_REGISTRY["goce"]
THRUST_N = 0.02
ISP_S = 3000.0
PROPELLANT_MASS_KG = 50.0 # Enough for 1 year of drag compensation

# STAGE DEFINITIONS
# Format: (date_str, target_alt_km)
# If target_alt_km is "maintain", it holds the previous altitude.
# Note: Values are defined relative to the Earth's WGS84 equatorial radius (6378.137 km)
STAGES = [
    ("2012-07-30T00:00:00.000", "maintain"), # 1. Maintain altitude until 30th July
    ("2012-08-30T00:00:00.000", 251.0),      # 2. Decrease to 251km by 30th August
    ("2012-11-01T00:00:00.000", "maintain"), # 3. Maintain 251km until 1st Nov
    ("2012-12-01T00:00:00.000", 244.0),      # 4. Decrease to 244km by 1st Dec
    ("2013-02-02T00:00:00.000", "maintain"), # 5. Maintain 244km until 2nd Feb
    ("2013-02-13T00:00:00.000", 237.0),      # 6. Decrease to 237km by 13th Feb
    ("2013-05-15T00:00:00.000", "maintain"), # 7. Maintain 237km until 15th May
    ("2013-05-25T00:00:00.000", 229.0),      # 8. Decrease to 229km by 25th May
    (None, "maintain")                       # 9. Maintain 229km until the end of the simulation
]

def _epoch_to_datetime(orekit_epoch):
    return datetime.fromisoformat(str(orekit_epoch).replace("Z", "+00:00"))

def build_sma_profile(truth_data):
    """
    Builds the interpolation function `target_sma_func(date)`
    based on the STAGES definition and initial TLE state.
    """
    utc = TimeScalesFactory.getUTC()
    
    first = truth_data[0]
    last = truth_data[-1]
    
    initial_date = first["epoch"]
    initial_sma = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + first["altitude_km"] * 1000.0
    initial_alt = first["altitude_km"]
    
    waypoints = []
    
    # Starting point
    current_alt = initial_alt
    current_sma = initial_sma
    waypoints.append({
        "date": initial_date,
        "alt_km": current_alt,
        "sma_m": current_sma
    })
    
    # Process stages
    for date_str, target_alt in STAGES:
        if date_str is None:
            stage_date = last["epoch"]
        else:
            dt_obj = datetime.fromisoformat(date_str)
            stage_date = AbsoluteDate(dt_obj.year, dt_obj.month, dt_obj.day,
                                      dt_obj.hour, dt_obj.minute, float(dt_obj.second), utc)
        
        if target_alt == "maintain":
            target_alt = current_alt
            target_sma = current_sma
        else:
            target_alt = float(target_alt)
            # Directly target the absolute mean semi-major axis corresponding to the target altitude
            target_sma = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + target_alt * 1000.0
            
        waypoints.append({
            "date": stage_date,
            "alt_km": target_alt,
            "sma_m": target_sma
        })
        
        current_alt = target_alt
        current_sma = target_sma

    def target_sma_func(date):
        # Find the segment
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i+1]
            if date.compareTo(w1["date"]) >= 0 and date.compareTo(w2["date"]) <= 0:
                duration_total = w2["date"].durationFrom(w1["date"])
                if duration_total == 0:
                    return w1["sma_m"]
                
                duration_passed = date.durationFrom(w1["date"])
                fraction = duration_passed / duration_total
                
                # Linear interpolation of SMA
                sma = w1["sma_m"] + fraction * (w2["sma_m"] - w1["sma_m"])
                return sma
                
        # If past the last waypoint
        if date.compareTo(waypoints[-1]["date"]) > 0:
            return waypoints[-1]["sma_m"]
            
        # If before the first waypoint
        return waypoints[0]["sma_m"]
        
    return target_sma_func, waypoints


def run_staged_mission():
    tle_file = os.path.join(src_dir, "validations", "tle_data", "drag_compensation", "goce_20120701_20130701.txt")
    if not os.path.exists(tle_file):
        print(f"Error: Could not find TLE file {tle_file}")
        return
        
    print("====================================")
    print("  Staged Drag Compensation Mission")
    print("====================================\n")

    # 1. Parse TLE & Get Truth
    print("[1/3] Parsing TLE file and getting truth data...")
    tle_entries = parse_tle_file(tle_file)
    truth_data  = get_truth_data(tle_entries)
    first_epoch = truth_data[0]["epoch"]
    last_epoch  = truth_data[-1]["epoch"]
    print(f"      First epoch : {first_epoch}")
    print(f"      Last  epoch : {last_epoch}\n")

    # 2. Build Profile
    target_sma_func, waypoints = build_sma_profile(truth_data)
    
    print("--- Mission Profile ---")
    for w in waypoints:
        print(f"  {w['date']} -> Altitude: {w['alt_km']:.1f} km")
    print("-----------------------\n")

    # 3. Setup Simulation
    duration_s = last_epoch.durationFrom(first_epoch)
    first = truth_data[0]
    
    params = {
        "start_date":        first["epoch"],
        "sma_m":             first["sma_m"],
        "eccentricity":      first["eccentricity"],
        "inclination":       first["inclination_deg"],
        "raan":              first["raan_deg"],
        "arg_perigee":       first.get("arg_perigee_deg", 0.0),
        "true_anomaly":      first.get("true_anomaly_deg", 0.0),
        "altitude":          first["altitude_km"] * 1000.0,  # Legacy fallback
        "mass":              SPACECRAFT["mass"],
        "cross_section":     SPACECRAFT["cross_section"],
        "drag_coeff":        (SPACECRAFT["drag_coeff_min"] + SPACECRAFT["drag_coeff_max"]) / 2.0,
        "time_step":         SPACECRAFT["time_step"],
        "duration":          duration_s,
        "target_sma_func":   target_sma_func,
    }
    
    # h_min/h_max not used by staged_tracking mode, set extremely wide
    ep_system = ElectricPropulsionSystem(
        thrust=THRUST_N,
        isp=ISP_S,
        initial_propellant_mass=PROPELLANT_MASS_KG,
        h_min=1.0e9,
        h_max=2.0e9
    )
    
    print("[2/3] Running orbital model with 'staged_tracking' compensation mode...")
    results = run_simulation(params, model_type=MODEL_NAME, 
                             propulsion_model=ep_system, 
                             compensation_mode="staged_tracking")

    print(f"\n  Propellant used   : {ep_system.propellant_used:.3f} kg")
    print(f"  Total burn time   : {ep_system.burn_time / 3600:.2f} h")
    
    plot_staged_mission(truth_data, results, first_epoch, waypoints)


def plot_staged_mission(truth_data, results, first_epoch, waypoints):
    first_dt = _epoch_to_datetime(first_epoch)
    def secs_to_dt(s): return first_dt + timedelta(seconds=float(s))

    import numpy as np
    model_dates = [secs_to_dt(t) for t in results["time"]]
    delta_R = (Constants.WGS84_EARTH_EQUATORIAL_RADIUS - 6371000.0) / 1000.0  # 7.137 km
    model_alt   = [alt - delta_R for alt in results["altitude"]]

    # Smooth the model altitude with a moving average over 24 hours (24 steps)
    # to extract the true mean altitude between apogee and perigee, smoothing out short-periodic J2 oscillations.
    window_size = 24
    padded_alt = np.pad(model_alt, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
    model_alt_mean = np.convolve(padded_alt, np.ones(window_size) / window_size, 'valid')

    truth_dates = [_epoch_to_datetime(d["epoch"]) for d in truth_data]
    truth_alt   = [d["altitude_km"] for d in truth_data]

    # Map thrust levels
    thrust_on    = results.get("thrust_on", [False] * len(model_dates))
    thrust_level = [results.get("thrust_level", [0.0] * len(model_dates))[i]
                    if "thrust_level" in results
                    else (results.get("propulsion_model_thrust", 0.0)
                          if thrust_on[i] else 0.0)
                    for i in range(len(model_dates))]

    # Build target altitude line exactly as passed to the controller
    target_dates = []
    target_alts  = []
    for w in waypoints:
        dt_val = _epoch_to_datetime(w["date"])
        target_dates.append(dt_val)
        target_alts.append(w["alt_km"])

    date_fmt = mdates.DateFormatter("%b %Y")
    out_dir  = os.path.join(project_root, "results", "results_dc_validations")
    os.makedirs(out_dir, exist_ok=True)

    # =============================================================
    # Figure 1: Mean Altitude Tracking
    # =============================================================
    fig1, ax1 = _ops.make_figure("1x1", figsize=_ops.FIGURE_SIZES["wide"])
    fig1.suptitle("GOCE Staged Mission Profile \u2014 Mean Altitude Tracking")

    ax1.plot(target_dates, target_alts, **_ops.plot_kwargs("target_profile"))
    ax1.plot(model_dates,  model_alt_mean,
             color=_ops.COLORS["primary"],
             linewidth=_ops.LINE_WIDTHS["main"],
             label="Simulated mean altitude")
    ax1.plot(truth_dates,  truth_alt, **_ops.plot_kwargs("tle_truth", label="TLE Truth (Actual)"))
    ax1.set_ylabel("Mean altitude [km]")
    ax1.set_xlabel("Date")
    _ops.tidy_legend(ax1)
    ax1.xaxis.set_major_formatter(date_fmt)
    fig1.autofmt_xdate(rotation=30, ha="right")

    saved = _ops.save_figure(fig1, "goce_staged_tracking", output_dir=out_dir)
    print(f"Saved Mean Altitude plot to {saved[0]}")

    # =============================================================
    # Figure 2: Continuous Variable Thrust
    # =============================================================
    fig2, ax2 = _ops.make_figure("1x1", figsize=_ops.FIGURE_SIZES["wide"])
    fig2.suptitle("GOCE Staged Mission Profile \u2014 Continuous Variable Thrust")

    thrust_mN = [t * 1000.0 for t in thrust_level]
    ax2.plot(model_dates, thrust_mN,
             color=_ops.MODEL_STYLES["thrust"]["color"],
             linewidth=_ops.LINE_WIDTHS["main"],
             label="Average thrust")
    ax2.set_ylabel("Average thrust [mN]")
    ax2.set_xlabel("Date")
    _ops.tidy_legend(ax2)
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.set_ylim(bottom=0.0)
    fig2.autofmt_xdate(rotation=30, ha="right")

    saved2 = _ops.save_figure(fig2, "goce_staged_thrust", output_dir=out_dir)
    print(f"Saved Thrust plot to {saved2[0]}")

    mplcursors.cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    run_staged_mission()
