"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          drag_compensation_validation.py
Description:
    Active drag compensation validation comparing simulated station-keeping against historical satellite TLE data.
===============================================================================
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors
from datetime import datetime, timedelta

# ============================================================
# PATH SETUP
# ============================================================
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import orbital_plot_style as _ops

# ============================================================
# OREKIT INITIALIZATION
# ============================================================
import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception:
    pass  # Already running

from orekit_jpype.pyhelpers import setup_orekit_data
_project_root = os.path.dirname(_SRC_DIR)
_data_path = os.path.join(_project_root, "orekit-data-main")
setup_orekit_data(filenames=_data_path, from_pip_library=False)
# ============================================================

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.orbits import KeplerianOrbit
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.bodies import OneAxisEllipsoid

# ============================================================
# SPACECRAFT REGISTRY  (same as orbital_decay_validation.py)
# ============================================================
SPACECRAFT_REGISTRY = {
    "goce": {
        "mass":           977.0,
        "cross_section":  0.9,
        "drag_coeff":     3.8,          # nominal Cd for compensation
        "time_step":      3600.0,       # seconds between output points
    },
    "grace": {
        "mass":           398.0,
        "cross_section":  0.95,
        "drag_coeff":     4.2,
        "time_step":      3600.0,
    },
    "champ": {
        "mass":           492.0,
        "cross_section":  0.9,
        "drag_coeff":     2.6,
        "time_step":      3600.0,
    },
    "slats": {
        "mass":           339.0,
        "cross_section":  0.36,
        "drag_coeff":     5.0,
        "time_step":      3600.0,
    },
    "soar": {
        "mass":           2.88,
        "cross_section":  0.1125,
        "drag_coeff":     0.57,
        "time_step":      3600.0,
    },
}

# ============================================================
# USER CONFIGURATION – THRUSTER PARAMETERS
# ============================================================
# Edit the values below to change the electric propulsion system.
#
# MODE 1  – duty_cycle
#   Thruster fires at a fixed thrust level when altitude drops below h_min_km
#   and shuts off when it rises above h_max_km.
#   Both values are ABSOLUTE altitudes in km, independent of the initial orbit.
#
# MODE 2  – maintenance
#   Thrust VARIES each step to exactly cancel the instantaneous drag force,
#   keeping altitude constant at the first TLE value.  thrust is used as the
#   maximum available force; h_min_km / h_max_km are not used.
#
# MODE 3  – goal
#   Thruster fires at constant thrust until goal_altitude_km is reached, then
#   switches to a duty-cycle band ± goal_offset_km around the goal altitude.
#   If goal > initial the orbit is raised; if goal < initial the thruster is
#   silent during the descent phase (drag lowers the orbit naturally).
# ============================================================
DEFAULT_THRUSTER = {
    "thrust":            0.02,    # [N]  – e.g. 20 mN Hall thruster
    "isp":               2500.0,  # [s]
    "propellant_mass":   20.0,    # [kg]
    # Mode 1 – duty cycle thresholds (ABSOLUTE altitudes in km)
    "h_min_km":          250.0,   # thruster turns ON  below this altitude [km]
    "h_max_km":          300.0,   # thruster turns OFF above this altitude [km]
    # Mode 3 – goal altitude
    "goal_altitude_km":  271.5,   # target altitude to reach and maintain   [km]
    "goal_offset_km":    1.0,     # ± station-keeping band around goal      [km]
}


# ============================================================
# STEP 1 – TLE PARSING  (shared utility, identical to decay validation)
# ============================================================

def parse_tle_file(filepath):
    """
    Parse a TLE file containing multiple epochs for the same satellite.
    Returns a list of (name, line1, line2) tuples sorted oldest-first.
    """
    with open(filepath, "r") as f:
        raw = [line.rstrip() for line in f if line.strip() and not line.startswith("#")]

    entries = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if line.startswith("1 "):       # 2-line format
            entries.append(("UNKNOWN", raw[i], raw[i + 1]))
            i += 2
        else:                           # 3-line format
            entries.append((line.strip(), raw[i + 1], raw[i + 2]))
            i += 3
    return entries


# ============================================================
# STEP 2 – PROPAGATE TLEs → "TRUTH" ORBITAL ELEMENTS
# ============================================================

def propagate_tle_at_epoch(name, line1, line2):
    """
    Use Orekit's SGP4/SDP4 TLEPropagator to get Keplerian elements
    at the TLE's own epoch.
    """
    inertial_frame = FramesFactory.getEME2000()
    earth_frame    = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth_shape    = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame,
    )

    tle        = TLE(line1, line2)
    propagator = TLEPropagator.selectExtrapolator(tle)
    epoch      = tle.getDate()
    state      = propagator.propagate(epoch)

    kep         = KeplerianOrbit(state.getOrbit())
    inc         = kep.getI()
    raan        = kep.getRightAscensionOfAscendingNode()
    arg_perigee = kep.getPerigeeArgument()
    true_anom   = kep.getTrueAnomaly()

    pos      = state.getPVCoordinates().getPosition()
    geodetic = earth_shape.transform(pos, inertial_frame, epoch)
    alt_km   = geodetic.getAltitude() / 1000.0

    return {
        "epoch":            epoch,
        "altitude_km":      alt_km,
        "sma_m":            kep.getA(),
        "eccentricity":     kep.getE(),
        "inclination_deg":  math.degrees(inc),
        "raan_deg":         math.degrees(raan) % 360.0,
        "arg_perigee_deg":  math.degrees(arg_perigee) % 360.0,
        "true_anomaly_deg": math.degrees(true_anom) % 360.0,
    }


def get_truth_data(tle_entries):
    """Collect Keplerian elements at every TLE epoch, sorted oldest-first."""
    print(f"  Processing {len(tle_entries)} TLE entries...")
    truth = []
    for name, l1, l2 in tle_entries:
        truth.append(propagate_tle_at_epoch(name, l1, l2))

    if not truth:
        return truth
    ref_epoch = truth[0]["epoch"]
    truth.sort(key=lambda d: d["epoch"].durationFrom(ref_epoch))
    return truth

def get_truth_eclipses(tle_entries):
    from org.orekit.propagation.analytical.tle import TLEPropagator, TLE
    from org.orekit.propagation.events import EclipseDetector
    from org.orekit.propagation.events.handlers import RecordAndContinue
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import Constants, IERSConventions
    
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame
    )
    sun = CelestialBodyFactory.getSun()
    
    truth_umbra_handler = RecordAndContinue()
    truth_umbra_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withUmbra().withHandler(truth_umbra_handler)
    
    events_list = []
    for i in range(len(tle_entries)):
        _, line1, line2 = tle_entries[i]
        tle_current = TLE(line1, line2)
        
        prop = TLEPropagator.selectExtrapolator(tle_current)
        handler = RecordAndContinue()
        detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withUmbra().withHandler(handler)
        prop.addEventDetector(detector)
        
        start_prop = tle_current.getDate().shiftedBy(-6000.0)
        end_prop = tle_current.getDate().shiftedBy(6000.0)
        
        try:
            prop.propagate(start_prop, end_prop)
        except:
            continue
            
        start_date = TLE(tle_entries[0][1], tle_entries[0][2]).getDate()
        events = handler.getEvents()
        
        enter_t = None
        for e in events:
            t = e.getState().getDate()
            if not e.isIncreasing():
                enter_t = t
            else:
                if enter_t is not None:
                    duration = t.durationFrom(enter_t)
                    mid = enter_t.shiftedBy(duration / 2.0)
                    events_list.append((mid.durationFrom(start_date), duration))
                    break
                    
    events_list.sort(key=lambda x: x[0])
    return events_list


# ============================================================
# STEP 3 – RUN MODEL WITH DRAG COMPENSATION
# ============================================================

def run_model_with_compensation(truth_data, model_name, spacecraft_params,
                                thruster_params, compensation_mode):
    """
    Run the atmospheric model from the first TLE epoch with electric
    propulsion drag compensation.

    Parameters
    ----------
    compensation_mode : str
        "duty_cycle"  – fixed thrust ON when alt < h_min_km, OFF when alt > h_max_km
        "maintenance" – variable thrust that cancels drag to hold initial altitude
        "goal"        – constant thrust until goal_altitude_km reached, then duty-cycle
    """
    from orbital_models.atmospheric_model import run_simulation
    from orbital_models.electric_propulsion import ElectricPropulsionSystem

    first = truth_data[0]
    last  = truth_data[-1]
    # Simulation ends exactly at the last TLE epoch – no reentry overshoot.
    duration_s = last["epoch"].durationFrom(first["epoch"])
    h0_km = first["altitude_km"]

    if compensation_mode == "duty_cycle":
        # Absolute altitude thresholds – completely independent of the initial TLE.
        h_min_km = float(thruster_params["h_min_km"])
        h_max_km = float(thruster_params["h_max_km"])
        h_min = h_min_km * 1000.0
        h_max = h_max_km * 1000.0
        goal_altitude_km = None
        goal_offset_km   = None

    elif compensation_mode == "maintenance":
        # h_min / h_max not used; atmospheric_model handles variable thrust internally.
        h_min = 1.0e9
        h_max = 2.0e9
        goal_altitude_km = None
        goal_offset_km   = None

    else:  # "goal"
        goal_altitude_km = thruster_params.get("goal_altitude_km", h0_km)
        goal_offset_km   = thruster_params.get("goal_offset_km",   1.0)
        # Always set h_min very high so thruster is ON during Phase 1 (both raising & lowering)
        h_min = 1.0e9
        h_max = 2.0e9

    ep_system = ElectricPropulsionSystem(
        thrust                  = thruster_params["thrust"],
        isp                     = thruster_params["isp"],
        initial_propellant_mass = thruster_params["propellant_mass"],
        h_min                   = h_min,
        h_max                   = h_max,
    )

    params = {
        "start_date":        first["epoch"],
        "sma_m":             first["sma_m"],                 # Use exact SMA to match altitude
        "eccentricity":      first["eccentricity"],          # Use exact eccentricity to match altitude
        "inclination":       first["inclination_deg"],
        "raan":              first["raan_deg"],
        "arg_perigee":       first.get("arg_perigee_deg", 0.0),
        "true_anomaly":      first.get("true_anomaly_deg", 0.0),
        "altitude":          first["altitude_km"] * 1000.0,  # [m]
        "mass":              spacecraft_params["mass"],
        "cross_section":     spacecraft_params["cross_section"],
        "drag_coeff":        spacecraft_params["drag_coeff"],
        "time_step":         spacecraft_params["time_step"],
        "duration":          duration_s,
        # Goal-mode extras (ignored by other modes)
        "goal_altitude_km":  goal_altitude_km,
        "goal_offset_km":    goal_offset_km,
    }

    mode_labels = {
        "duty_cycle":   "duty-cycle thrust",
        "maintenance":  "altitude maintenance",
        "goal":         f"goal altitude ({goal_altitude_km:.1f} km)" if goal_altitude_km else "goal",
    }
    mode_label   = mode_labels.get(compensation_mode, compensation_mode)
    duration_days = duration_s / 86400.0
    print(f"  Running '{model_name}' with {mode_label} for "
          f"{duration_days:.1f} days "
          f"({first['epoch']} -> {last['epoch']})")
    print("  [Simulation stops at the last TLE epoch, not at reentry]")

    results = run_simulation(params, model_type=model_name,
                             propulsion_model=ep_system,
                             compensation_mode=compensation_mode)
    return results, ep_system


# ============================================================
# STEP 4 – PLOTTING
# ============================================================

def _epoch_to_datetime(orekit_epoch):
    """Convert an Orekit AbsoluteDate to a Python datetime (UTC)."""
    return datetime.fromisoformat(str(orekit_epoch).replace("Z", "+00:00"))


def plot_dc_validation(truth_data, results, first_epoch,
                       mission_name, model_name, compensation_mode,
                       thruster_params=None, output_dir=None, filename_prefix="dc_val",
                       truth_eclipses=None):
    """
    Three-figure output:
        Figure 1 – Mean altitude over time + thrust level
        Figure 2 – Eccentricity, Inclination and RAAN over time
        Figure 3 – Eclipse durations per orbit (model vs TLE truth)
    Truth data is overlaid on the altitude panel and orbital element panels.
    """
    first_dt = _epoch_to_datetime(first_epoch)

    def secs_to_dt(s):
        return first_dt + timedelta(seconds=float(s))

    # Build time axis (datetime objects)
    model_times  = results["time"]
    model_dates  = [secs_to_dt(t) for t in model_times]
    model_alt    = results["altitude"]
    model_ecc    = results.get("eccentricity", [])
    model_inc    = results.get("inclination", [])
    model_raan   = results.get("raan", [])

    thrust_on    = results.get("thrust_on", [False] * len(model_times))
    thrust_level = [results.get("thrust_level", [0.0] * len(model_times))[i]
                    if "thrust_level" in results
                    else (results.get("propulsion_model_thrust", 0.0)
                          if thrust_on[i] else 0.0)
                    for i in range(len(model_times))]

    # Truth dates and orbital elements
    truth_dates = [_epoch_to_datetime(d["epoch"]) for d in truth_data]
    truth_alt   = [d["altitude_km"]     for d in truth_data]
    truth_inc   = [d["inclination_deg"] for d in truth_data]
    truth_raan  = [d["raan_deg"]        for d in truth_data]
    truth_ecc   = [d["eccentricity"]    for d in truth_data]

    date_fmt = mdates.DateFormatter("%Y-%m-%d")
    mode_label_map = {
        "duty_cycle":   "Duty Cycle",
        "maintenance":  "Altitude Maintenance",
        "goal":         "Goal Altitude",
    }
    mode_label  = mode_label_map.get(compensation_mode, compensation_mode.title())
    goal_alt_km = (thruster_params or {}).get("goal_altitude_km")
    m_label     = _ops.model_label(model_name)
    kw_model    = _ops.plot_kwargs("model_thrust", label=f"{m_label} (with thrust)")
    kw_truth    = _ops.plot_kwargs("tle_truth", label="TLE Ground Truth")

    suptitle_base = (
        f"{mission_name} \u2014 Drag Compensation Validation\n"
        f"Model: {m_label} | Mode: {mode_label}"
    )

    # =============================================================
    # Figure 1: Altitude + Thrust
    # =============================================================
    fig1, axes1 = _ops.make_figure("2x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["wide_2panel"])
    fig1.suptitle(suptitle_base)

    # Panel 1a: altitude
    axes1[0].plot(model_dates, model_alt, **kw_model)
    axes1[0].plot(truth_dates, truth_alt, **kw_truth)
    if compensation_mode == "goal" and goal_alt_km is not None:
        axes1[0].axhline(
            goal_alt_km,
            color=_ops.COLORS["goal_line"],
            linewidth=_ops.LINE_WIDTHS["threshold"],
            linestyle=":",
            label=f"Goal altitude ({goal_alt_km:.0f} km)",
        )
    axes1[0].set_ylabel("Mean altitude [km]")
    _ops.tidy_legend(axes1[0])
    _ops.apply_panel_label(axes1[0], "a")

    # Panel 1b: thrust level (mN)
    thrust_mN = [t * 1000.0 for t in thrust_level]
    axes1[1].step(model_dates, thrust_mN,
                  color=_ops.MODEL_STYLES["thrust"]["color"],
                  where="post",
                  linewidth=_ops.LINE_WIDTHS["main"],
                  label="Thrust level")
    axes1[1].set_ylabel("Thrust level [mN]")
    axes1[1].set_xlabel("Date")
    axes1[1].xaxis.set_major_formatter(date_fmt)
    _ops.tidy_legend(axes1[1])
    _ops.apply_panel_label(axes1[1], "b")

    fig1.autofmt_xdate(rotation=30, ha="right")
    if output_dir:
        _ops.save_figure(fig1, f"{filename_prefix}_dc_altitude_thrust", output_dir=output_dir)

    # =============================================================
    # Figure 2: Orbital Parameters
    # =============================================================
    fig2, axes2 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig2.suptitle(suptitle_base)

    kw_truth_thin = _ops.plot_kwargs("tle_truth", label="TLE Ground Truth")

    # Eccentricity
    if model_ecc:
        axes2[0].plot(model_dates, model_ecc,
                      color=_ops.COLORS["secondary"],
                      linewidth=_ops.LINE_WIDTHS["main"],
                      label=f"{m_label}")
    axes2[0].plot(truth_dates, truth_ecc, **kw_truth_thin)
    axes2[0].set_ylabel("Eccentricity")
    _ops.tidy_legend(axes2[0])
    _ops.apply_panel_label(axes2[0], "a")

    # Inclination
    if model_inc:
        axes2[1].plot(model_dates, model_inc,
                      color=_ops.COLORS["reference"],
                      linewidth=_ops.LINE_WIDTHS["main"],
                      label=f"{m_label}")
    axes2[1].plot(truth_dates, truth_inc, **kw_truth_thin)
    axes2[1].set_ylabel("Inclination [deg]")
    _ops.tidy_legend(axes2[1])
    _ops.apply_panel_label(axes2[1], "b")

    # RAAN
    if model_raan:
        axes2[2].plot(model_dates, model_raan,
                      color=_ops.COLORS["primary"],
                      linewidth=_ops.LINE_WIDTHS["main"],
                      label=f"{m_label}")
    axes2[2].plot(truth_dates, truth_raan, **kw_truth_thin)
    axes2[2].set_ylabel("RAAN [deg]")
    axes2[2].set_xlabel("Date")
    axes2[2].xaxis.set_major_formatter(date_fmt)
    _ops.tidy_legend(axes2[2])
    _ops.apply_panel_label(axes2[2], "c")

    fig2.autofmt_xdate(rotation=30, ha="right")
    if output_dir:
        _ops.save_figure(fig2, f"{filename_prefix}_dc_orbital_params", output_dir=output_dir)

    # =============================================================
    # Figure 3: Eclipse Events
    # =============================================================
    fig3, ax3 = _ops.make_figure("1x1", figsize=_ops.FIGURE_SIZES["wide"])
    fig3.suptitle(
        f"{mission_name} \u2014 Eclipse Durations\n"
        f"Model: {m_label} | Mode: {mode_label}"
    )

    def pad_events(events, start_time, max_time):
        day_seconds = 86400.0
        padded = []
        current_marker = start_time
        event_idx = 0
        while current_marker <= max_time:
            next_marker = current_marker + day_seconds
            events_in_day = []
            while event_idx < len(events) and events[event_idx][0] < next_marker:
                if events[event_idx][0] >= current_marker:
                    events_in_day.append(events[event_idx])
                event_idx += 1
            if not events_in_day:
                padded.append((current_marker + day_seconds / 2, 0.0))
            else:
                for e in events_in_day:
                    padded.append(e)
            current_marker = next_marker
        return padded

    umbra_events = results.get("umbra_events", [])

    if umbra_events or truth_eclipses:
        start_dt = model_dates[0]
        max_time = results["time"][-1] if len(results["time"]) > 0 else 0

        u_padded = pad_events(umbra_events, 0.0, max_time)
        t_padded = pad_events(truth_eclipses, 0.0, max_time) if truth_eclipses else []

        if u_padded:
            u_dates = [start_dt + timedelta(seconds=e[0]) for e in u_padded]
            u_durs  = [e[1] for e in u_padded]
            ax3.plot(u_dates, u_durs,
                     marker=".", linestyle="none",
                     color=_ops.COLORS["primary"],
                     linewidth=_ops.LINE_WIDTHS["main"],
                     label="Model eclipses")

        if t_padded:
            t_dates = [start_dt + timedelta(seconds=e[0]) for e in t_padded]
            t_durs  = [e[1] for e in t_padded]
            ax3.plot(t_dates, t_durs,
                     marker=".", linestyle="none",
                     color=_ops.COLORS["truth"],
                     linewidth=_ops.LINE_WIDTHS["reference"],
                     label="TLE truth eclipses")

        ax3.set_ylabel("Eclipse duration [mm:ss]")
        _ops.tidy_legend(ax3)

        from matplotlib.ticker import FuncFormatter
        def format_mmss(y, pos):
            m = int(y // 60)
            s = int(y % 60)
            return f"{m:02d}:{s:02d}"
        ax3.yaxis.set_major_formatter(FuncFormatter(format_mmss))
    else:
        ax3.set_ylabel("Eclipse duration")

    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(date_fmt)
    fig3.autofmt_xdate(rotation=30, ha="right")

    if output_dir:
        _ops.save_figure(fig3, f"{filename_prefix}_dc_eclipse", output_dir=output_dir)

    mplcursors.cursor(hover=True)
    plt.show()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_dc_validation(tle_file, model_name, compensation_mode,
                      spacecraft_params=None, thruster_params=None):
    """
    Full drag-compensation validation pipeline.
    Called from main.py or run directly.

    Parameters
    ----------
    tle_file           : str  – path to TLE .txt file
    model_name         : str  – atmospheric model key (e.g. "nrlmsise00")
    compensation_mode  : str  – "constant" or "maintenance"
    spacecraft_params  : dict or None  (auto-detected from filename if None)
    thruster_params    : dict or None  (DEFAULT_THRUSTER used if None)
    """
    filename = os.path.basename(tle_file).lower()

    # --- Mission name & spacecraft params from filename ---
    mission_lookup = {
        "goce":  "GOCE",
        "grace": "GRACE",
        "champ": "CHAMP",
        "slats": "SLATS",
        "soar":  "SOAR",
    }
    mission_name = next(
        (v for k, v in mission_lookup.items() if filename.startswith(k)),
        "Unknown Mission",
    )

    if spacecraft_params is None:
        key = next(
            (k for k in mission_lookup if filename.startswith(k)),
            "goce",
        )
        spacecraft_params = SPACECRAFT_REGISTRY[key]

    if thruster_params is None:
        thruster_params = DEFAULT_THRUSTER.copy()

    print("\n====================================")
    print("  Drag Compensation Validation")
    print("====================================\n")

    # 1. Parse TLE
    print("[1/3] Parsing TLE file...")
    tle_entries = parse_tle_file(tle_file)
    print(f"      Found {len(tle_entries)} TLE entries.\n")

    # 2. Get truth
    print("[2/3] Propagating TLEs to get truth orbital elements...")
    truth_data  = get_truth_data(tle_entries)
    first_epoch = truth_data[0]["epoch"]
    print(f"      First epoch : {first_epoch}")
    print(f"      Last  epoch : {truth_data[-1]['epoch']}\n")

    # 3. Run model
    print("[3/3] Running orbital model with drag compensation...")
    results, ep_system = run_model_with_compensation(
        truth_data, model_name, spacecraft_params,
        thruster_params, compensation_mode,
    )
    
    # 4. Get truth eclipses
    print("      Extracting TLE truth eclipses...")
    truth_eclipses = get_truth_eclipses(tle_entries)

    # Summary
    print(f"\n  Propellant used   : {ep_system.propellant_used:.3f} kg")
    print(f"  Total burn time   : {ep_system.burn_time / 3600:.2f} h")
    print(f"  Number of cycles  : {ep_system.cycles}")
    print(f"  Shutdown reason   : {ep_system.shutdown_reason}")

    # 5. Plot
    results_dir = os.path.join(_project_root, "results", "results_dc_validations")
    os.makedirs(results_dir, exist_ok=True)
    tle_basename = os.path.splitext(os.path.basename(tle_file))[0]
    prefix = f"{tle_basename}_{model_name}_{compensation_mode}"

    plot_dc_validation(
        truth_data, results, first_epoch,
        mission_name, model_name, compensation_mode,
        thruster_params=thruster_params,
        output_dir=results_dir,
        filename_prefix=prefix,
        truth_eclipses=truth_eclipses
    )


    print("\nValidation complete.")
    return truth_data, results


if __name__ == "__main__":
    # Quick standalone test – replace paths and params as needed
    _tle_dir = os.path.join(os.path.dirname(__file__), "tle_data")
    _tle_file = os.path.join(_tle_dir, "goce_20130401_20131109.txt")
    run_dc_validation(_tle_file, model_name="nrlmsise00",
                      compensation_mode="constant")
