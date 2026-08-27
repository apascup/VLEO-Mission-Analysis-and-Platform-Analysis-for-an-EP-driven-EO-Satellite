"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          orbital_decay_validation.py
Description:
    Passive orbital decay validation comparing numerical propagation against real TLE data across 5 satellites.
===============================================================================
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# ============================================================
# PATH SETUP
# Allow this script to find the orbital_models package whether
# it is run standalone or from the project root.
# ============================================================
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import orbital_plot_style as _ops

# ============================================================
# OREKIT INITIALIZATION
# The JVM must be started before importing any 'org' packages.
# ============================================================
import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception:
    pass # Already running

from orekit_jpype.pyhelpers import setup_orekit_data
# _SRC_DIR is '.../src'
_project_root = os.path.dirname(_SRC_DIR) 
_data_path = os.path.join(_project_root, "orekit-data-main")
setup_orekit_data(filenames=_data_path, from_pip_library=False)
# ============================================================

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.orbits import KeplerianOrbit, OrbitType
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.bodies import OneAxisEllipsoid

# ============================================================
# USER CONFIGURATION
# ============================================================
TLE_FILE_PATH = os.path.join(os.path.dirname(__file__), "tle_data", "example_satellite.txt")

# Physical properties of the satellite (TLEs don't include these)
SPACECRAFT_REGISTRY = {
    "goce": {
        "mass":           977.0,      # kg
        "cross_section":  0.9,        # m^2
        "drag_coeff_min": 3.6,
        "drag_coeff_max": 4.0,
        "time_step":      3600.0,     # seconds between output points
    },
    "grace": {
        "mass":           398.0,      # kg
        "cross_section":  0.95,       # m^2
        "drag_coeff_min": 3.6,
        "drag_coeff_max": 4.8,
        "time_step":      3600.0,     # seconds between output points
    },
    "champ": {
        "mass":           492.0,      # kg
        "cross_section":  0.9,       # m^2
        "drag_coeff_min": 2.45,
        "drag_coeff_max": 2.7,
        "time_step":      3600.0,     # seconds between output points
    },
    "slats": {
        "mass":           339.0,      # kg
        "cross_section":  0.36,       # m^2
        "drag_coeff_min": 4.6,
        "drag_coeff_max": 5.4,
        "time_step":      3600.0,     # seconds between output points
    },
    "soar": {
        "mass":           2.88,      # kg
        "cross_section":  0.1125,       # m^2
        "drag_coeff_min": 0.32,
        "drag_coeff_max": 0.83,
        "time_step":      3600.0,     # seconds between output points
    }
}

# Which model to use for the simulation leg
# (Must match a key in orbital_models/__init__.py)
MODEL_NAME = "nrlmsise00"


# ============================================================
# STEP 1 – TLE PARSING
# ============================================================

def parse_tle_file(filepath):
    """
    Parse a TLE file that may contain multiple epochs for the same satellite.

    Accepted formats
    ----------------
    3-line (name + line1 + line2):
        SATELLITE NAME
        1 XXXXX...
        2 XXXXX...

    2-line (no name):
        1 XXXXX...
        2 XXXXX...

    Returns
    -------
    list of (name, line1, line2) tuples sorted by epoch (oldest first).
    """
    with open(filepath, "r") as f:
        raw = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]

    entries = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if line.startswith("1 "):          # 2-line format
            entries.append(("UNKNOWN", raw[i], raw[i + 1]))
            i += 2
        else:                              # 3-line format (name + two TLE lines)
            entries.append((line.strip(), raw[i + 1], raw[i + 2]))
            i += 3

    return entries


# ============================================================
# STEP 2 – PROPAGATE TLEs → "TRUTH" ORBITAL ELEMENTS
# ============================================================

def propagate_tle_at_epoch(name, line1, line2):
    """
    Use Orekit's SGP4/SDP4 TLEPropagator to get the Keplerian elements
    at the TLE's own epoch.

    Returns a dict with: epoch, altitude_km, inclination_deg, raan_deg,
    arg_perigee_deg, true_anomaly_deg.
    """
    inertial_frame = FramesFactory.getEME2000()
    earth_frame    = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth_shape    = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame
    )

    tle = TLE(line1, line2)
    propagator = TLEPropagator.selectExtrapolator(tle)

    epoch = tle.getDate()
    state = propagator.propagate(epoch)

    # Convert Cartesian state → Keplerian elements
    kep         = KeplerianOrbit(state.getOrbit())
    inc         = kep.getI()
    raan        = kep.getRightAscensionOfAscendingNode()
    arg_perigee = kep.getPerigeeArgument()
    true_anom   = kep.getTrueAnomaly()
    # Extract Mean Altitude from the TLE's Mean Motion
    # This prevents any short-periodic J2 oscillations from SGP4 aliasing into our plots.
    import math
    mean_motion = tle.getMeanMotion()
    mean_a = math.pow(Constants.WGS84_EARTH_MU / (mean_motion * mean_motion), 1.0 / 3.0)
    
    # Use Mean Earth Radius (6371.0 km) to get the "average between apoapsis and periapsis" 
    # of the geodetic altitude oscillation, rather than the minimum (Equatorial Radius).
    MEAN_EARTH_RADIUS = 6371000.0
    altitude_km = (mean_a - MEAN_EARTH_RADIUS) / 1000.0

    return {
        "epoch":            epoch,
        "altitude_km":      altitude_km,            # instantaneous geodetic altitude (for plots)
        "sma_m":            kep.getA(),              # semi-major axis [m] (for model initialisation)
        "eccentricity":     kep.getE(),              # real eccentricity from TLE
        "inclination_deg":  math.degrees(inc),
        "raan_deg":         math.degrees(raan)        % 360.0,
        "arg_perigee_deg":  math.degrees(arg_perigee) % 360.0,
        "true_anomaly_deg": math.degrees(true_anom)   % 360.0,
    }


def get_truth_data(tle_entries):
    """
    Iterate over all TLE entries and collect the orbital elements
    at each epoch.  Returns a list of result dicts (sorted by epoch).
    """
    print(f"  Processing {len(tle_entries)} TLE entries...")
    truth = []
    for name, l1, l2 in tle_entries:
        data = propagate_tle_at_epoch(name, l1, l2)
        truth.append(data)
    # Sort by epoch (oldest first).
    # IMPORTANT: capture the reference epoch BEFORE sorting.
    # Accessing truth[0] inside the lambda is unsafe because sort()
    # rearranges elements in-place and truth[0] changes mid-sort.
    if not truth:
        return truth
    ref_epoch = truth[0]["epoch"]
    truth.sort(key=lambda d: d["epoch"].durationFrom(ref_epoch))
    return truth


# ============================================================
# STEP 3 – RUN ORBITAL MODEL FROM FIRST TLE EPOCH
# ============================================================

def run_model_from_first_tle(truth_data, model_name, spacecraft_params, drag_coeff, force_reentry=False):
    """
    Build the params dict from the first TLE entry and run the selected model.
    The simulation duration is chosen to span the full range of the TLE dataset.
    """
    first = truth_data[0]
    last  = truth_data[-1]

    if force_reentry:
        # Give it a massive duration (10 years) to ensure it reaches reentry
        duration_s = 10.0 * 365.0 * 86400.0
    else:
        # Duration = time from first to last TLE epoch, with a small extra margin
        duration_s = last["epoch"].durationFrom(first["epoch"]) + spacecraft_params["time_step"]

    params = {
        "start_date":     first["epoch"],
        # Use the semi-major axis from the TLE (kep.getA()) so that the
        # model altitude at t=0 exactly matches the TLE truth.
        # Using geodetic altitude + R_equatorial would introduce a ~10-15 km
        # error at high latitudes because the Earth is not a sphere.
        "sma_m":          first["sma_m"],
        # Keep eccentricity small for numerical stability of the integrator.
        # The real osculating eccentricity can be large enough to cause
        # instability at GOCE's very low altitude.
        "eccentricity":   1e-4,
        "inclination":    first["inclination_deg"],
        "raan":           first["raan_deg"],
        "arg_perigee":    first.get("arg_perigee_deg",  0.0),
        "true_anomaly":   first.get("true_anomaly_deg", 0.0),
        "mass":           spacecraft_params["mass"],
        "cross_section":  spacecraft_params["cross_section"],
        "drag_coeff":     drag_coeff,
        "time_step":      spacecraft_params["time_step"],
        "duration":       duration_s,
    }

    if force_reentry:
        duration_str = "until reentry (max 10 years)"
    else:
        duration_str = f"for {duration_s/3600:.1f} hours"

    print(f"  Running model '{model_name}' {duration_str} "
          f"starting at {first['epoch']}...")

    # Import the run function from the unified atmospheric_model
    from orbital_models.atmospheric_model import run_simulation as _run
    results = _run(params, model_type=model_name)
    return results


# ============================================================
# STEP 4 – ALIGN MODEL OUTPUT WITH TLE TRUTH EPOCHS
# ============================================================

def align_model_to_truth(truth_data, model_results, first_epoch):
    """
    For each TLE truth epoch, find the closest time point in the model output
    and extract altitude, inclination, and RAAN for error calculation.
    If the model continues running after truth data ends, pad the remaining
    model time points assuming truth altitude is 0.0 km.
    """
    import math
    import numpy as np
    
    model_times_sec  = np.array(model_results["time"])        # seconds from start
    model_alt_km     = np.array(model_results["altitude"])
    model_inc_deg    = np.array(model_results.get("inclination", []))
    model_raan_deg   = np.array(model_results.get("raan", []))
    has_inc  = len(model_inc_deg)  > 0
    has_raan = len(model_raan_deg) > 0

    aligned = []
    
    last_truth_sec = truth_data[-1]["epoch"].durationFrom(first_epoch)
    last_truth_inc = truth_data[-1]["inclination_deg"]
    last_truth_raan = truth_data[-1]["raan_deg"]

    for entry in truth_data:
        t_sec = entry["epoch"].durationFrom(first_epoch)   # seconds

        # Stop aligning if the truth epoch is past the model's final simulation time (re-entry)
        if t_sec > model_times_sec[-1]:
            break

        # Find nearest index in model output
        idx = int(np.argmin(np.abs(model_times_sec - t_sec)))

        model_inc  = float(model_inc_deg[idx])  if has_inc  else float("nan")
        model_raan = float(model_raan_deg[idx]) if has_raan else float("nan")

        # RAAN error: wrap to [-180, 180] to handle 0°/360° discontinuity
        err_raan = model_raan - entry["raan_deg"] if has_raan else float("nan")
        if not math.isnan(err_raan):
            err_raan = (err_raan + 180.0) % 360.0 - 180.0

        aligned.append({
            "t_sec":      t_sec,
            "truth_alt":  entry["altitude_km"],
            "truth_inc":  entry["inclination_deg"],
            "truth_raan": entry["raan_deg"],
            "model_alt":  float(model_alt_km[idx]),
            "model_inc":  model_inc,
            "model_raan": model_raan,
            "err_alt":    float(model_alt_km[idx]) - entry["altitude_km"],
            "err_inc":    model_inc  - entry["inclination_deg"] if has_inc  else float("nan"),
            "err_raan":   err_raan,
        })
        
    # If the model continues after the last truth epoch, evaluate error at model steps
    if len(model_times_sec) > 0 and model_times_sec[-1] > last_truth_sec:
        for idx, t_sec in enumerate(model_times_sec):
            if t_sec > last_truth_sec:
                truth_alt = 0.0 # Truth re-entered
                truth_inc = last_truth_inc
                truth_raan = last_truth_raan
                
                model_inc  = float(model_inc_deg[idx])  if has_inc  else float("nan")
                model_raan = float(model_raan_deg[idx]) if has_raan else float("nan")

                err_raan = model_raan - truth_raan if has_raan else float("nan")
                if not math.isnan(err_raan):
                    err_raan = (err_raan + 180.0) % 360.0 - 180.0

                aligned.append({
                    "t_sec":      float(t_sec),
                    "truth_alt":  truth_alt,
                    "truth_inc":  truth_inc,
                    "truth_raan": truth_raan,
                    "model_alt":  float(model_alt_km[idx]),
                    "model_inc":  model_inc,
                    "model_raan": model_raan,
                    "err_alt":    float(model_alt_km[idx]) - truth_alt,
                    "err_inc":    model_inc  - truth_inc if has_inc  else float("nan"),
                    "err_raan":   err_raan,
                })
                
    return aligned


# ============================================================
# STEP 5 – PLOT RESULTS
# ============================================================

def plot_validation(truth_data, all_results, all_aligned, model_name, first_epoch, cd_min, cd_max, mission_name, output_dir=None, filename_prefix=""):
    """
    Produce two figures for a single model:
        Figure 1 – Orbital decay comparison (altitude, inclination, RAAN vs time)
        Figure 2 – Error in altitude, inclination, and RAAN vs TLE ground truth
    Uses fill_between to show the Cd_min / Cd_max envelope.
    """
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates
    import numpy as np

    first_dt = datetime.fromisoformat(str(first_epoch).replace("Z", "+00:00"))

    def secs_to_dt(seconds):
        return first_dt + timedelta(seconds=seconds)

    def pad_to_length(arr, target_length):
        if len(arr) == 0: return arr
        if len(arr) < target_length:
            return np.pad(arr, (0, target_length - len(arr)), 'edge')
        return arr

    # -- Build datetime axes --
    truth_dates = [secs_to_dt(entry["epoch"].durationFrom(first_epoch))
                   for entry in truth_data]
    truth_alt   = [d["altitude_km"]     for d in truth_data]
    truth_inc   = [d["inclination_deg"] for d in truth_data]
    truth_raan  = [d["raan_deg"]        for d in truth_data]

    res_min = all_results[model_name]["min"]
    res_max = all_results[model_name]["max"]

    max_len = max(len(res_min["time"]), len(res_max["time"]))
    common_time = res_min["time"] if len(res_min["time"]) > len(res_max["time"]) else res_max["time"]
    model_dates = [secs_to_dt(t) for t in common_time]

    alt_min = pad_to_length(res_min["altitude"], max_len)
    alt_max = pad_to_length(res_max["altitude"], max_len)

    has_model_inc = len(res_min.get("inclination", [])) > 0
    if has_model_inc:
        inc_min = pad_to_length(res_min["inclination"], max_len)
        inc_max = pad_to_length(res_max["inclination"], max_len)

    has_model_raan = len(res_min.get("raan", [])) > 0
    if has_model_raan:
        raan_min = pad_to_length(res_min["raan"], max_len)
        raan_max = pad_to_length(res_max["raan"], max_len)

    al_min = all_aligned[model_name]["min"]
    al_max = all_aligned[model_name]["max"]
    al_max_len = max(len(al_min), len(al_max))

    al_common_time = ([d["t_sec"] for d in al_min]
                      if len(al_min) > len(al_max)
                      else [d["t_sec"] for d in al_max])
    aligned_dates = [secs_to_dt(t) for t in al_common_time]

    err_alt_min  = pad_to_length([d["err_alt"]  for d in al_min], al_max_len)
    err_alt_max  = pad_to_length([d["err_alt"]  for d in al_max], al_max_len)
    err_inc_min  = pad_to_length([d["err_inc"]  for d in al_min], al_max_len)
    err_inc_max  = pad_to_length([d["err_inc"]  for d in al_max], al_max_len)
    err_raan_min = pad_to_length([d["err_raan"] for d in al_min], al_max_len)
    err_raan_max = pad_to_length([d["err_raan"] for d in al_max], al_max_len)

    date_fmt   = mdates.DateFormatter("%Y-%m-%d")
    m_label    = _ops.model_label(model_name)
    c_lo       = _ops.MODEL_STYLES.get(f"{model_name}_lo", _ops.MODEL_STYLES.get(model_name, {}))
    c_hi       = _ops.MODEL_STYLES.get(f"{model_name}_hi", c_lo)
    clr_lo     = c_lo.get("color", _ops.COLORS["primary"])
    clr_hi     = c_hi.get("color", _ops.COLORS["secondary"])
    clr_fill   = clr_lo
    kw_truth   = _ops.plot_kwargs("tle_truth", label="TLE Ground Truth")

    # ---------- Figure 1: Orbital Decay Comparison ----------
    fig1, axes1 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig1.suptitle(
        f"{mission_name.upper()} — Orbital Decay: {m_label} vs TLE Truth"
    )

    ax = axes1[0]
    _ops.plot_error_band(ax, model_dates, alt_min, alt_max, color=clr_fill)
    ax.plot(model_dates, alt_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
            label=f"{m_label} (Cd = {cd_min})")
    ax.plot(model_dates, alt_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
            label=f"{m_label} (Cd = {cd_max})")
    ax.plot(truth_dates, truth_alt, **kw_truth)
    ax.set_ylabel("Altitude [km]")
    _ops.tidy_legend(ax)
    _ops.apply_panel_label(ax, "a")

    ax = axes1[1]
    if has_model_inc:
        _ops.plot_error_band(ax, model_dates, inc_min, inc_max, color=clr_fill)
        ax.plot(model_dates, inc_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"{m_label} (Cd = {cd_min})")
        ax.plot(model_dates, inc_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"{m_label} (Cd = {cd_max})")
    ax.plot(truth_dates, truth_inc, **kw_truth)
    ax.set_ylabel("Inclination [deg]")
    _ops.tidy_legend(ax)
    _ops.apply_panel_label(ax, "b")

    ax = axes1[2]
    if has_model_raan:
        _ops.plot_error_band(ax, model_dates, raan_min, raan_max, color=clr_fill)
        ax.plot(model_dates, raan_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"{m_label} (Cd = {cd_min})")
        ax.plot(model_dates, raan_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"{m_label} (Cd = {cd_max})")
    ax.plot(truth_dates, truth_raan, **kw_truth)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.set_ylabel("RAAN [deg]")
    _ops.tidy_legend(ax)
    _ops.apply_panel_label(ax, "c")

    fig1.autofmt_xdate(rotation=30, ha="right")

    # ---------- Figure 2: Error ----------
    fig2, axes2 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig2.suptitle(
        f"{mission_name.upper()} — Model Error: {m_label} vs TLE Ground Truth"
    )

    ax = axes2[0]
    _ops.plot_error_band(ax, aligned_dates, err_alt_min, err_alt_max, color=clr_fill)
    ax.plot(aligned_dates, err_alt_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
            label=f"Error (Cd = {cd_min})")
    ax.plot(aligned_dates, err_alt_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
            label=f"Error (Cd = {cd_max})")
    _ops.add_zero_line(ax)
    _ops.add_error_stats(ax, list(err_alt_min) + list(err_alt_max))
    ax.set_ylabel("\u0394 Altitude [km]\n[Model − Truth]")
    _ops.tidy_legend(ax)
    _ops.apply_panel_label(ax, "a")

    ax = axes2[1]
    if has_model_inc:
        _ops.plot_error_band(ax, aligned_dates, err_inc_min, err_inc_max, color=clr_fill)
        ax.plot(aligned_dates, err_inc_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"Error (Cd = {cd_min})")
        ax.plot(aligned_dates, err_inc_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"Error (Cd = {cd_max})")
        _ops.add_zero_line(ax)
        ax.set_ylabel("\u0394 Inclination [deg]\n[Model − Truth]")
        _ops.tidy_legend(ax)
    else:
        ax.text(0.5, 0.5, "Inclination not available in model output",
                transform=ax.transAxes, ha="center", va="center",
                color=_ops.COLORS["gray_mid"])
        ax.set_ylabel("\u0394 Inclination [deg]")
    _ops.apply_panel_label(ax, "b")

    ax = axes2[2]
    ax.xaxis.set_major_formatter(date_fmt)
    if has_model_raan:
        _ops.plot_error_band(ax, aligned_dates, err_raan_min, err_raan_max, color=clr_fill)
        ax.plot(aligned_dates, err_raan_min, color=clr_lo, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"Error (Cd = {cd_min})")
        ax.plot(aligned_dates, err_raan_max, color=clr_hi, linewidth=_ops.LINE_WIDTHS["secondary"],
                label=f"Error (Cd = {cd_max})")
        _ops.add_zero_line(ax)
        ax.set_ylabel("\u0394 RAAN [deg]\n[Model − Truth]")
        _ops.tidy_legend(ax)
    else:
        ax.text(0.5, 0.5, "RAAN not available in model output",
                transform=ax.transAxes, ha="center", va="center",
                color=_ops.COLORS["gray_mid"])
        ax.set_ylabel("\u0394 RAAN [deg]")
    _ops.apply_panel_label(ax, "c")

    fig2.autofmt_xdate(rotation=30, ha="right")

    if output_dir:
        _ops.save_figure(fig1, f"{filename_prefix}_evolution", output_dir=output_dir)
        _ops.save_figure(fig2, f"{filename_prefix}_error",     output_dir=output_dir)

    mplcursors.cursor(hover=True)
    plt.show()

def plot_all_models_evolution(truth_data, all_results_dict, first_epoch, cd_min, cd_max, mission_name, output_dir=None, filename_prefix=""):
    """
    Single figure with 3 subplots comparing the absolute orbital evolution
    (Altitude, Inclination, RAAN) of all models against the TLE truth.
    Uses fill_between to show the Cd_min / Cd_max envelope for each model.
    """
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates
    import numpy as np

    first_dt = datetime.fromisoformat(str(first_epoch).replace("Z", "+00:00"))

    def secs_to_dt(seconds):
        return first_dt + timedelta(seconds=seconds)

    def pad_to_length(arr, target_length):
        if len(arr) == 0: return arr
        if len(arr) < target_length:
            return np.pad(arr, (0, target_length - len(arr)), 'edge')
        return arr

    truth_dates = [secs_to_dt(entry["epoch"].durationFrom(first_epoch)) for entry in truth_data]
    truth_alt   = [d["altitude_km"]     for d in truth_data]
    truth_inc   = [d["inclination_deg"] for d in truth_data]
    truth_raan  = [d["raan_deg"]        for d in truth_data]

    date_fmt = mdates.DateFormatter("%Y-%m-%d")

    fig, axes = _ops.make_figure("3x1", shared_x=True,
                                 figsize=_ops.FIGURE_SIZES["3panel"])
    fig.suptitle(
        f"{mission_name.upper()} — Orbital Decay: All Model Envelopes vs TLE Truth"
    )

    for model_name, res_dict in all_results_dict.items():
        res_min = res_dict["min"]
        res_max = res_dict["max"]

        max_len = max(len(res_min["time"]), len(res_max["time"]))
        common_time = (res_min["time"] if len(res_min["time"]) > len(res_max["time"])
                       else res_max["time"])
        model_dates = [secs_to_dt(t) for t in common_time]

        alt_min = pad_to_length(res_min["altitude"], max_len)
        alt_max = pad_to_length(res_max["altitude"], max_len)

        has_model_inc = len(res_min.get("inclination", [])) > 0
        if has_model_inc:
            inc_min = pad_to_length(res_min["inclination"], max_len)
            inc_max = pad_to_length(res_max["inclination"], max_len)

        has_model_raan = len(res_min.get("raan", [])) > 0
        if has_model_raan:
            raan_min = pad_to_length(res_min["raan"], max_len)
            raan_max = pad_to_length(res_max["raan"], max_len)

        s_lo  = _ops.MODEL_STYLES.get(f"{model_name}_lo", _ops.MODEL_STYLES.get(model_name, {}))
        s_hi  = _ops.MODEL_STYLES.get(f"{model_name}_hi", s_lo)
        clr   = s_lo.get("color", "gray")
        clr_d = s_hi.get("color", clr)
        lbl   = _ops.model_label(model_name)
        lw    = _ops.LINE_WIDTHS["secondary"]

        # Altitude
        _ops.plot_error_band(axes[0], model_dates, alt_min, alt_max, color=clr)
        axes[0].plot(model_dates, alt_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
        axes[0].plot(model_dates, alt_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

        # Inclination
        if has_model_inc:
            _ops.plot_error_band(axes[1], model_dates, inc_min, inc_max, color=clr)
            axes[1].plot(model_dates, inc_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
            axes[1].plot(model_dates, inc_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

        # RAAN
        if has_model_raan:
            _ops.plot_error_band(axes[2], model_dates, raan_min, raan_max, color=clr)
            axes[2].plot(model_dates, raan_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
            axes[2].plot(model_dates, raan_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

    # Ground truth overlay
    kw_truth = _ops.plot_kwargs("tle_truth", label="TLE Ground Truth")
    for ax in axes:
        ax.plot(truth_dates,
                truth_alt if ax is axes[0] else (truth_inc if ax is axes[1] else truth_raan),
                **kw_truth)

    axes[0].set_ylabel("Altitude [km]")
    axes[1].set_ylabel("Inclination [deg]")
    axes[2].set_ylabel("RAAN [deg]")
    axes[2].xaxis.set_major_formatter(date_fmt)

    panel_letters = ["a", "b", "c"]
    for i, ax in enumerate(axes):
        _ops.deduplicate_legend(ax, loc="upper right")
        _ops.apply_panel_label(ax, panel_letters[i])

    fig.autofmt_xdate(rotation=30, ha="right")

    if output_dir:
        _ops.save_figure(fig, f"{filename_prefix}_all_evolution", output_dir=output_dir)

    mplcursors.cursor(hover=True)
    plt.show()

def plot_all_models_errors(truth_data, all_aligned_dict, first_epoch, cd_min, cd_max, mission_name, output_dir=None, filename_prefix=""):
    """
    Single figure with 3 subplots comparing the signed error envelopes
    (Altitude, Inclination, RAAN) of all models against the TLE truth.
    Uses fill_between to show the Cd_min / Cd_max error envelope for each model.
    """
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates
    import numpy as np

    first_dt = datetime.fromisoformat(str(first_epoch).replace("Z", "+00:00"))

    def secs_to_dt(seconds):
        return first_dt + timedelta(seconds=seconds)

    def pad_to_length(arr, target_length):
        if len(arr) == 0: return arr
        if len(arr) < target_length:
            return np.pad(arr, (0, target_length - len(arr)), 'edge')
        return arr

    date_fmt = mdates.DateFormatter("%Y-%m-%d")

    fig, axes = _ops.make_figure("3x1", shared_x=True,
                                 figsize=_ops.FIGURE_SIZES["3panel"])
    fig.suptitle(
        f"{mission_name.upper()} — Model Error Envelopes vs TLE Ground Truth (All Models)"
    )

    for model_name, aligned_dict in all_aligned_dict.items():
        al_min = aligned_dict["min"]
        al_max = aligned_dict["max"]
        al_max_len = max(len(al_min), len(al_max))

        al_common_time = ([d["t_sec"] for d in al_min] if len(al_min) > len(al_max)
                          else [d["t_sec"] for d in al_max])
        aligned_dates = [secs_to_dt(t) for t in al_common_time]

        err_alt_min  = pad_to_length([d["err_alt"]  for d in al_min], al_max_len)
        err_alt_max  = pad_to_length([d["err_alt"]  for d in al_max], al_max_len)
        err_inc_min  = pad_to_length([d["err_inc"]  for d in al_min], al_max_len)
        err_inc_max  = pad_to_length([d["err_inc"]  for d in al_max], al_max_len)
        err_raan_min = pad_to_length([d["err_raan"] for d in al_min], al_max_len)
        err_raan_max = pad_to_length([d["err_raan"] for d in al_max], al_max_len)

        s_lo  = _ops.MODEL_STYLES.get(f"{model_name}_lo", _ops.MODEL_STYLES.get(model_name, {}))
        s_hi  = _ops.MODEL_STYLES.get(f"{model_name}_hi", s_lo)
        clr   = s_lo.get("color", "gray")
        clr_d = s_hi.get("color", clr)
        lbl   = _ops.model_label(model_name)
        lw    = _ops.LINE_WIDTHS["secondary"]

        # Altitude error
        _ops.plot_error_band(axes[0], aligned_dates, err_alt_min, err_alt_max, color=clr)
        axes[0].plot(aligned_dates, err_alt_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
        axes[0].plot(aligned_dates, err_alt_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

        # Inclination error
        if not math.isnan(err_inc_min[0]):
            _ops.plot_error_band(axes[1], aligned_dates, err_inc_min, err_inc_max, color=clr)
            axes[1].plot(aligned_dates, err_inc_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
            axes[1].plot(aligned_dates, err_inc_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

        # RAAN error
        if not math.isnan(err_raan_min[0]):
            _ops.plot_error_band(axes[2], aligned_dates, err_raan_min, err_raan_max, color=clr)
            axes[2].plot(aligned_dates, err_raan_min, color=clr,   linewidth=lw, label=f"{lbl} (Cd = {cd_min})")
            axes[2].plot(aligned_dates, err_raan_max, color=clr_d, linewidth=lw, label=f"{lbl} (Cd = {cd_max})")

    for i, ax in enumerate(axes):
        _ops.add_zero_line(ax)
        _ops.deduplicate_legend(ax, loc="upper right")
        _ops.apply_panel_label(ax, ["a", "b", "c"][i])

    axes[0].set_ylabel("\u0394 Altitude [km]\n[Model \u2212 Truth]")
    axes[1].set_ylabel("\u0394 Inclination [deg]\n[Model \u2212 Truth]")
    axes[2].set_ylabel("\u0394 RAAN [deg]\n[Model \u2212 Truth]")
    axes[2].xaxis.set_major_formatter(date_fmt)

    fig.autofmt_xdate(rotation=30, ha="right")

    if output_dir:
        _ops.save_figure(fig, f"{filename_prefix}_all_error", output_dir=output_dir)

    mplcursors.cursor(hover=True)
    plt.show()

# ============================================================
def run_validation(tle_file=TLE_FILE_PATH, model_name=MODEL_NAME,
                   spacecraft_params=None):
    """
    Full validation pipeline.  Can be called from main.py or run directly.
    """
    filename = os.path.basename(tle_file).lower()
    if filename.startswith("goce"):
        mission_name = "GOCE"
    elif filename.startswith("grace"):
        mission_name = "GRACE"
    elif filename.startswith("champ"):
        mission_name = "CHAMP"
    elif filename.startswith("slats"):
        mission_name = "SLATS"
    elif filename.startswith("soar"):
        mission_name = "SOAR"
    else:
        mission_name = "Unknown Mission"

    if spacecraft_params is None:
        if filename.startswith("goce"):
            spacecraft_params = SPACECRAFT_REGISTRY["goce"]
        elif filename.startswith("grace"):
            spacecraft_params = SPACECRAFT_REGISTRY["grace"]
        elif filename.startswith("champ"):
            spacecraft_params = SPACECRAFT_REGISTRY["champ"]
        elif filename.startswith("slats"):
            spacecraft_params = SPACECRAFT_REGISTRY["slats"]
        elif filename.startswith("soar"):
            spacecraft_params = SPACECRAFT_REGISTRY["soar"]
        else:
            spacecraft_params = SPACECRAFT_REGISTRY["goce"] # default fallback

    print("\n============================")
    print("  Orbital Decay Validation  ")
    print("============================\n")

    # 1. Parse TLE file
    print("[1/4] Parsing TLE file...")
    tle_entries = parse_tle_file(tle_file)
    print(f"      Found {len(tle_entries)} TLE entries.\n")

    # 2. Get truth data from TLEs
    print("[2/4] Propagating TLEs to get truth orbital elements...")
    truth_data = get_truth_data(tle_entries)
    first_epoch = truth_data[0]["epoch"]
    print(f"      First epoch : {first_epoch}")
    print(f"      Last  epoch : {truth_data[-1]['epoch']}\n")

    cd_min = spacecraft_params.get("drag_coeff_min", 2.2)
    cd_max = spacecraft_params.get("drag_coeff_max", 4.0)

    # 3. Run the orbital model(s)
    if model_name == "all":
        print("[3/4] Running ALL orbital models...")
        models_to_run = ["nrlmsise00", "jb2008", "dtm2000", "harrispriester"]
        all_aligned = {}
        all_results = {}
        for mod in models_to_run:
            print(f"\n  --- Running {mod} ---")
            try:
                res_min = run_model_from_first_tle(truth_data, mod, spacecraft_params, cd_min, force_reentry=True)
                al_min = align_model_to_truth(truth_data, res_min, first_epoch)
                
                res_max = run_model_from_first_tle(truth_data, mod, spacecraft_params, cd_max, force_reentry=True)
                al_max = align_model_to_truth(truth_data, res_max, first_epoch)
                
                all_aligned[mod] = {"min": al_min, "max": al_max}
                all_results[mod] = {"min": res_min, "max": res_max}
            except Exception as e:
                print(f"  [!] Failed to run {mod}: {e}")
        
        print("\n[4/4] Generating comparison plots and tables...")
        if all_aligned:
            import csv
            
            # Prepare CSV exporting
            results_dir = os.path.join(_project_root, "results", "results_decay_validations")
            os.makedirs(results_dir, exist_ok=True)
            tle_basename = os.path.splitext(os.path.basename(tle_file))[0]
            csv_path = os.path.join(results_dir, f"{tle_basename}_validation.csv")
            
            csv_rows = []
            csv_headers = ["Model (Bound)", "Model (d)", "Truth (d)", "Err (d)", "Err (%)", "Alt Err (km)", "Alt Err (%)"]

            print("\n" + "="*101)
            print(f"{'Model (Bound)':<17} | {'Model (d)':<10} | {'Truth (d)':<10} | {'Err (d)':<8} | {'Err (%)':<8} | {'Alt Err(km)':<11} | {'Alt Err(%)':<11}")
            print("-" * 101)
            
            truth_duration_days = truth_data[-1]["epoch"].durationFrom(first_epoch) / 86400.0
            
            name_map = {
                "nrlmsise00": "NRLMSISE-00",
                "jb2008": "JB2008",
                "dtm2000": "DTM2000",
                "harrispriester": "Harris-Priester"
            }

            for mod, aligned_dict in all_aligned.items():
                res_dict = all_results[mod]
                mod_label = name_map.get(mod, mod)
                
                for bound, bound_label in [("min", f"Cd {cd_min}"), ("max", f"Cd {cd_max}")]:
                    m_res = res_dict[bound]
                    aligned = aligned_dict[bound]
                    
                    model_duration_days = m_res["time"][-1] / 86400.0
                    
                    errs = [d["err_alt"] for d in aligned]
                    rel_errs = [(d["err_alt"] / d["truth_alt"]) * 100 for d in aligned if d["truth_alt"] != 0]
                    
                    avg_err = sum(errs) / len(errs) if errs else 0.0
                    avg_rel = sum(rel_errs) / len(rel_errs) if rel_errs else 0.0
                    
                    reentry_err_d = model_duration_days - truth_duration_days
                    reentry_err_rel = (reentry_err_d / truth_duration_days) * 100 if truth_duration_days != 0 else 0.0
                    
                    full_label = f"{mission_name.upper()} {mod_label} ({bound_label})"
                    print(f"{full_label:<17} | {model_duration_days:<10.2f} | {truth_duration_days:<10.2f} | {reentry_err_d:<8.2f} | {reentry_err_rel:<8.2f} | {avg_err:<11.2f} | {avg_rel:<11.2f}")
                    
                    csv_rows.append([
                        full_label,
                        f"{model_duration_days:.2f}",
                        f"{truth_duration_days:.2f}",
                        f"{reentry_err_d:.2f}",
                        f"{reentry_err_rel:.2f}",
                        f"{avg_err:.2f}",
                        f"{avg_rel:.2f}"
                    ])
                
            print("="*101 + "\n")
            
            # Write to CSV
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(csv_headers)
                    writer.writerows(csv_rows)
                print(f"      [CSV Output] Results exported to: {csv_path}")
            except Exception as e:
                print(f"      [!] Failed to export CSV: {e}")


            plot_all_models_evolution(truth_data, all_results, first_epoch, cd_min, cd_max, mission_name, output_dir=results_dir, filename_prefix=tle_basename)
            plot_all_models_errors(truth_data, all_aligned, first_epoch, cd_min, cd_max, mission_name, output_dir=results_dir, filename_prefix=tle_basename)
        else:
            print("No models succeeded, cannot plot.")
            
        print("\nValidation complete.")
        return truth_data, None, all_aligned

    else:
        print("[3/4] Running orbital model...")
        res_min = run_model_from_first_tle(truth_data, model_name, spacecraft_params, cd_min, force_reentry=True)
        res_max = run_model_from_first_tle(truth_data, model_name, spacecraft_params, cd_max, force_reentry=True)
        print()

        # 4. Align model output with truth epochs & compute error
        print("[4/4] Computing error and generating plots...")
        al_min = align_model_to_truth(truth_data, res_min, first_epoch)
        al_max = align_model_to_truth(truth_data, res_max, first_epoch)

        all_results = {model_name: {"min": res_min, "max": res_max}}
        all_aligned = {model_name: {"min": al_min, "max": al_max}}

        results_dir = os.path.join(_project_root, "results", "results_decay_validations")
        os.makedirs(results_dir, exist_ok=True)
        tle_basename = os.path.splitext(os.path.basename(tle_file))[0]

        # 5. Plot
        plot_validation(truth_data, all_results, all_aligned, model_name, first_epoch, cd_min, cd_max, mission_name, output_dir=results_dir, filename_prefix=f"{tle_basename}_{model_name}")

        print("\nValidation complete.")
        return truth_data, all_results, all_aligned

if __name__ == "__main__":
    run_validation()
