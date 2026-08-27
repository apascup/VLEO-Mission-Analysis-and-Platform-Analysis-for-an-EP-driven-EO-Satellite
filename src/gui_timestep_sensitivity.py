"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          gui_timestep_sensitivity.py
Description:
    Headless script for timestep convergence and sensitivity analysis across orbital, propulsion, and power models.
===============================================================================
"""

# ==============================================================================
# CONFIGURATION — edit everything here
# ==============================================================================

# -- Simulation type -----------------------------------------------------------
# "Orbital Decay Only"  |  "Orbital Decay + Drag Compensation"
SIM_TYPE = "Orbital Decay + Drag Compensation"

# -- Timesteps (seconds) — four distinct values --------------------------------
TIMESTEPS = [1.0, 30.0, 60.0, 300.0]   # smallest = reference

# -- Orbit parameters ----------------------------------------------------------
ORBIT_ALT_KM        = 320.0    # Initial altitude (km)
ORBIT_ECC           = 0.001    # Eccentricity
ORBIT_INC_DEG       = 96.67     # Inclination (deg)
ORBIT_RAAN_DEG      = 11.5      # RAAN (deg)
ORBIT_ARG_PERIGEE   = 0.0      # Argument of perigee (deg)
ORBIT_TRUE_ANOMALY  = 0.0      # True anomaly (deg)

# Epoch (UTC)
EPOCH_YEAR   = 2026
EPOCH_MONTH  = 1
EPOCH_DAY    = 1
EPOCH_HOUR   = 0
EPOCH_MIN    = 0
EPOCH_SEC    = 0.0

# -- Spacecraft ----------------------------------------------------------------
SC_MASS_KG          = 500.0      # Dry mass (kg)
SC_CROSS_SECTION_M2 = 0.5   # Cross-sectional area (m²)
SC_DRAG_COEFF       = 3.6      # Drag coefficient Cd
SC_REFL_COEFF       = 1.5      # Reflectivity coefficient Cr

# -- Propulsion (only used when SIM_TYPE includes drag compensation) -----------
# Thruster preset: "RIT_uX" | "RIT_10_EVO" | "RIT_2X" | "Custom"
THRUSTER_KEY        = "RIT_10_EVO"   # catalog key  (ignored when THRUSTER_KEY="Custom")
THRUSTER_OP_POINT   = 0          # operating-point index for multi-point thrusters

# Custom thruster values (only used when THRUSTER_KEY = "Custom")
CUSTOM_THRUST_N     = 0.02
CUSTOM_ISP_S        = 2500.0
CUSTOM_POWER_W      = 150.0

PROP_MASS_KG        = 5.0      # Propellant mass (kg)

# Drag compensation logic: "duty_cycle" | "maintenance" | "goal"
COMP_MODE           = "duty_cycle"
H_MIN_KM            = 260.0    # ON  altitude threshold  (duty_cycle mode)
H_MAX_KM            = 320.0    # OFF altitude threshold  (duty_cycle mode)
GOAL_ALT_KM         = 400.0    # Target altitude         (goal mode)
GOAL_OFFSET_KM      = 1.0      # Station-keeping band ±  (goal mode)

# -- Atmospheric model ---------------------------------------------------------
# "nrlmsise00" | "jb2008" | "dtm2000" | "harrispriester"
ATM_MODEL = "nrlmsise00"

# -- Simulation duration -------------------------------------------------------
SIM_DURATION_DAYS = 1460.0

# -- Power subsystem -----------------------------------------------------------
PWR_SOLAR_AREA_M2     = 5.0      # Solar panel area (m²)
PWR_PANEL_EFFICIENCY  = 0.28      # BOL efficiency  [0–1]
PWR_SOLAR_FLUX_W_M2   = 1361.0    # Solar flux (W/m²)
PWR_PANEL_DEGR_YR     = 0.025     # Panel degradation per year [0–1]
PWR_BAT_CAPACITY_WH   = 300.0      # Battery capacity (Wh)
PWR_BAT_INITIAL_WH    = 300.0       # Battery initial charge (Wh)
PWR_BAT_DEGR_YR       = 0.05      # Battery degradation per year [0–1]
PWR_HOUSEKEEPING_W    = 80.0       # Housekeeping power draw (W)

# -- Output --------------------------------------------------------------------
OUTPUT_DIR = "timestep_sensitivity_results"   # folder where plots are saved (relative to script)
SHOW_PLOTS = True   # set True to also display figures interactively

# ==============================================================================
# END OF CONFIGURATION — do not edit below unless you know what you are doing
# ==============================================================================

import os
import sys
import contextlib
import io

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Resolve paths and initialise Orekit
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception:
    pass

from orekit_jpype.pyhelpers import setup_orekit_data
PROJECT_ROOT = os.path.dirname(SRC_DIR)
setup_orekit_data(
    filenames=os.path.join(PROJECT_ROOT, "orekit-data-main"),
    from_pip_library=False,
)

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from orbital_models import atmospheric_model
from orbital_models.electric_propulsion import ElectricPropulsionSystem
from orbital_models.power_subsystem import PowerSubsystem
from mission_config import RIT_THRUSTERS, get_thruster_config

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
out_dir = os.path.join(PROJECT_ROOT, "results", OUTPUT_DIR)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = ["#00bcd4", "#ff7043", "#66bb6a", "#ab47bc"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class _StreamToLogger:
    """Capture stdout/stderr to a string buffer."""
    def __init__(self):
        self.buffer = io.StringIO()
    def write(self, msg):
        self.buffer.write(msg)
    def flush(self):
        pass
    def get_val(self):
        return self.buffer.getvalue()


def interp_onto_reference(ref_time, ref_vals, other_time, other_vals):
    return np.interp(ref_time, other_time, other_vals)


def reconstruct_illumination(time_s, umbra_events, penumbra_events):
    """Build 0/0.5/1.0 illumination signal from eclipse event lists."""
    illum = np.ones(len(time_s))
    for mid_t, dur in penumbra_events:
        mask = (time_s >= mid_t - dur / 2.0) & (time_s <= mid_t + dur / 2.0)
        illum[mask] = 0.5
    for mid_t, dur in umbra_events:
        mask = (time_s >= mid_t - dur / 2.0) & (time_s <= mid_t + dur / 2.0)
        illum[mask] = 0.0
    return illum


def simulate_power_posthoc(time_s, illumination, area, eff, flux,
                            bat_cap, bat_init, hk,
                            panel_degr=0.0, bat_degr=0.0):
    """Replay PowerSubsystem over a reconstructed illumination time-series."""
    pm = PowerSubsystem(
        solar_panel_area_m2=area, panel_efficiency=eff,
        solar_flux_W_m2=flux, battery_capacity_Wh=bat_cap,
        battery_initial_Wh=bat_init, housekeeping_power_W=hk,
        thruster_power_W=0.0,
        panel_degradation_yr=panel_degr, battery_degradation_yr=bat_degr,
    )
    n = len(time_s)
    P_gen = np.zeros(n); P_cons = np.zeros(n)
    bat   = np.zeros(n); soc    = np.zeros(n)
    bat[0] = bat_init; soc[0] = pm.state_of_charge
    for i in range(1, n):
        dt = max(float(time_s[i] - time_s[i - 1]), 1.0)
        _, pg, pc = pm.update(float(illumination[i]), False, dt)
        P_gen[i] = pg; P_cons[i] = pc
        bat[i]   = pm.battery_Wh; soc[i] = pm.state_of_charge
    P_gen[0] = pm.P_solar_max * float(illumination[0])
    P_cons[0] = hk
    return P_gen, P_cons, bat, soc


def _ax_style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#aaa")
    ax.grid(True, alpha=0.3, color="#ccc")


def _fig_style(fig):
    fig.patch.set_facecolor("white")


def savefig(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [saved] {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def print_table(rows, title=""):
    """Print a list-of-dicts as a plain-text table."""
    if not rows:
        return
    if title:
        print(f"\n{'='*60}\n{title}\n{'='*60}")
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(header); print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ==============================================================================
# VALIDATION
# ==============================================================================
ts_sorted = sorted(set(TIMESTEPS))
if len(ts_sorted) < 2:
    raise ValueError("Need at least 2 distinct TIMESTEPS.")

print("=" * 60)
print("LeoOrbSim — Timestep Sensitivity Analysis (headless)")
print("=" * 60)
print(f"Simulation type : {SIM_TYPE}")
print(f"Atmospheric model: {ATM_MODEL}")
print(f"Duration        : {SIM_DURATION_DAYS} days")
print(f"Timesteps (s)   : {ts_sorted}  (ref = {ts_sorted[0]} s)")
print(f"Output folder   : {out_dir}")
print()

# ==============================================================================
# BUILD SHARED PARAMETERS
# ==============================================================================
utc        = TimeScalesFactory.getUTC()
start_date = AbsoluteDate(
    int(EPOCH_YEAR), int(EPOCH_MONTH), int(EPOCH_DAY),
    int(EPOCH_HOUR), int(EPOCH_MIN), float(EPOCH_SEC), utc,
)

base_params = {
    "start_date":         start_date,
    "altitude":           float(ORBIT_ALT_KM) * 1000.0,
    "inclination":        float(ORBIT_INC_DEG),
    "eccentricity":       float(ORBIT_ECC),
    "raan":               float(ORBIT_RAAN_DEG),
    "arg_perigee":        float(ORBIT_ARG_PERIGEE),
    "true_anomaly":       float(ORBIT_TRUE_ANOMALY),
    "mass":               float(SC_MASS_KG),
    "cross_section":      float(SC_CROSS_SECTION_M2),
    "drag_coeff":         float(SC_DRAG_COEFF),
    "reflectivity_coeff": float(SC_REFL_COEFF),
    "duration":           float(SIM_DURATION_DAYS) * 86400.0,
}

if SIM_TYPE == "Orbital Decay + Drag Compensation" and COMP_MODE == "goal":
    base_params["goal_altitude_km"] = float(GOAL_ALT_KM)
    base_params["goal_offset_km"]   = float(GOAL_OFFSET_KM)

# Resolve thruster parameters
if SIM_TYPE == "Orbital Decay + Drag Compensation":
    if THRUSTER_KEY != "Custom":
        cfg         = get_thruster_config(THRUSTER_KEY, THRUSTER_OP_POINT)
        prop_thrust = cfg["thrust_N"]
        prop_isp    = float(cfg["isp_s"])
        prop_power  = float(cfg["power_W"])
        print(f"Thruster        : {cfg['name']} — {cfg['label']}")
    else:
        prop_thrust = float(CUSTOM_THRUST_N)
        prop_isp    = float(CUSTOM_ISP_S)
        prop_power  = float(CUSTOM_POWER_W)
        print(f"Thruster        : Custom  {prop_thrust:.6f} N  Isp={prop_isp:.0f} s  P={prop_power:.1f} W")

    if COMP_MODE == "duty_cycle":
        h_min = float(H_MIN_KM) * 1000.0
        h_max = float(H_MAX_KM) * 1000.0
    else:
        h_min = 1.0e12
        h_max = 2.0e12

# ==============================================================================
# RUN SIMULATIONS
# ==============================================================================
_t_start = time.perf_counter()   # ← timer starts here

all_results = {}
logger = _StreamToLogger()

with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
    for idx, ts in enumerate(ts_sorted):
        label = f"Δt = {ts:.0f} s"
        print(f"  Running simulation {idx+1}/{len(ts_sorted)}: {label}…", flush=True)
        params  = {**base_params, "time_step": float(ts)}
        ep_model = None
        prop_kw  = {}

        if SIM_TYPE == "Orbital Decay + Drag Compensation":
            ep_model = ElectricPropulsionSystem(
                thrust                  = float(prop_thrust),
                isp                     = float(prop_isp),
                initial_propellant_mass = float(PROP_MASS_KG),
                h_min                   = h_min,
                h_max                   = h_max,
            )
            prop_kw = {
                "propulsion_model":  ep_model,
                "compensation_mode": COMP_MODE,
            }

        try:
            result = atmospheric_model.run_simulation(
                params,
                model_type=ATM_MODEL,
                **prop_kw,
            )
            result["_ep_model"] = ep_model
            all_results[ts] = result
        except Exception as exc:
            print(f"  WARNING: {label} failed — {exc}")

# Echo captured log
log_text = logger.get_val().strip()
if log_text:
    print("\n--- Orekit log ---")
    print(log_text)
    print("--- end log ---\n")

if len(all_results) < 2:
    raise RuntimeError("At least 2 successful simulations are needed.")

# ==============================================================================
# REFERENCE EXTRACTION
# ==============================================================================
sorted_ts     = sorted(all_results.keys())
ref_ts        = sorted_ts[0]
ref           = all_results[ref_ts]
ref_time      = np.array(ref["time"])
ref_days      = ref_time / 86400.0
ref_alt       = np.array(ref["altitude"])
ref_ecc       = np.array(ref["eccentricity"])
ref_inc       = np.array(ref["inclination"])
comparison_ts = [ts for ts in sorted_ts if ts != ref_ts]

print(f"\n✅ {len(all_results)} simulations completed.  Reference: Δt = {ref_ts:.0f} s\n")

# ==============================================================================
# SECTION 1 — ORBITAL MECHANICS
# ==============================================================================
print("─── Section 1: Orbital Mechanics ───")

# Plot 1 — Altitude trajectories
fig1, ax1 = plt.subplots(figsize=(12, 5)); _fig_style(fig1); _ax_style(ax1)
for i, ts in enumerate(sorted_ts):
    r = all_results[ts]
    t = np.array(r["time"]) / 86400.0; a = np.array(r["altitude"])
    ax1.plot(t, a, color=COLORS[i], lw=2.2 if ts == ref_ts else 1.4,
             ls="-" if ts == ref_ts else "--", alpha=0.9,
             label=f"Δt={ts:.0f}s" + (" ← ref" if ts == ref_ts else ""))
ax1.set_title("Altitude vs Time — All Timesteps", fontsize=13)
ax1.set_xlabel("Time (days)"); ax1.set_ylabel("Altitude (km)")
ax1.legend(fontsize=9); plt.tight_layout()
savefig(fig1, "plot01_altitude_trajectories.png")

# Plot 2 — Altitude difference vs reference
fig2, ax2 = plt.subplots(figsize=(12, 5)); _fig_style(fig2); _ax_style(ax2)
for ts in comparison_ts:
    r = all_results[ts]; ot = np.array(r["time"]); oa = np.array(r["altitude"])
    d = interp_onto_reference(ref_time, ref_alt, ot, oa) - ref_alt
    c = COLORS[sorted_ts.index(ts)]
    ax2.plot(ref_days, d, color=c, lw=1.6,
             label=f"Δt={ts:.0f}s  RMS={np.sqrt(np.mean(d**2)):.4f} km")
ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax2.set_title(f"Altitude Difference vs Reference (Δt={ref_ts:.0f}s)", fontsize=13)
ax2.set_xlabel("Time (days)"); ax2.set_ylabel("Δ Altitude (km)")
ax2.legend(fontsize=9); plt.tight_layout()
savefig(fig2, "plot02_altitude_difference.png")

# Plot 3 — Eccentricity difference vs reference
fig3, ax3 = plt.subplots(figsize=(12, 4)); _fig_style(fig3); _ax_style(ax3)
for ts in comparison_ts:
    r = all_results[ts]; ot = np.array(r["time"]); oe = np.array(r["eccentricity"])
    d = interp_onto_reference(ref_time, ref_ecc, ot, oe) - ref_ecc
    c = COLORS[sorted_ts.index(ts)]
    ax3.plot(ref_days, d, color=c, lw=1.6,
             label=f"Δt={ts:.0f}s  RMS={np.sqrt(np.mean(d**2)):.2e}")
ax3.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax3.set_title(f"Eccentricity Difference vs Reference (Δt={ref_ts:.0f}s)", fontsize=13)
ax3.set_xlabel("Time (days)"); ax3.set_ylabel("Δ Eccentricity")
ax3.legend(fontsize=9); plt.tight_layout()
savefig(fig3, "plot03_eccentricity_difference.png")

# Plot 4 — Inclination difference vs reference
fig4, ax4 = plt.subplots(figsize=(12, 4)); _fig_style(fig4); _ax_style(ax4)
for ts in comparison_ts:
    r = all_results[ts]; ot = np.array(r["time"]); oi = np.array(r["inclination"])
    d = interp_onto_reference(ref_time, ref_inc, ot, oi) - ref_inc
    c = COLORS[sorted_ts.index(ts)]
    ax4.plot(ref_days, d, color=c, lw=1.6,
             label=f"Δt={ts:.0f}s  RMS={np.sqrt(np.mean(d**2)):.2e} °")
ax4.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax4.set_title(f"Inclination Difference vs Reference (Δt={ref_ts:.0f}s)", fontsize=13)
ax4.set_xlabel("Time (days)"); ax4.set_ylabel("Δ Inclination (°)")
ax4.legend(fontsize=9); plt.tight_layout()
savefig(fig4, "plot04_inclination_difference.png")

# Plot 5 — RMS bar chart
rms_alt, rms_ecc, rms_inc, bar_labels = [], [], [], []
for ts in comparison_ts:
    r = all_results[ts]; ot = np.array(r["time"])
    ia = interp_onto_reference(ref_time, ref_alt, ot, np.array(r["altitude"]))
    ie = interp_onto_reference(ref_time, ref_ecc, ot, np.array(r["eccentricity"]))
    ii = interp_onto_reference(ref_time, ref_inc, ot, np.array(r["inclination"]))
    rms_alt.append(np.sqrt(np.mean((ia - ref_alt)**2)))
    rms_ecc.append(np.sqrt(np.mean((ie - ref_ecc)**2)))
    rms_inc.append(np.sqrt(np.mean((ii - ref_inc)**2)))
    bar_labels.append(f"Δt={ts:.0f}s")
fig5, axes5 = plt.subplots(1, 3, figsize=(14, 5)); _fig_style(fig5)
fig5.suptitle(f"RMS Error vs Reference (Δt={ref_ts:.0f}s)", fontsize=13, fontweight="bold")
for ax, vals, title, unit, col in zip(
    axes5, [rms_alt, rms_ecc, rms_inc],
    ["Altitude RMS", "Eccentricity RMS", "Inclination RMS"],
    ["km", "–", "°"],
    ["#00bcd4", "#ff7043", "#66bb6a"],
):
    _ax_style(ax)
    bars = ax.bar(bar_labels, vals, color=col, alpha=0.85, width=0.5, edgecolor="#aaa")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_title(title, fontsize=10); ax.set_ylabel(f"RMS ({unit})")
plt.tight_layout()
savefig(fig5, "plot05_rms_summary.png")

# Orbital summary table
orb_rows = []
for ts in sorted_ts:
    r = all_results[ts]; t = np.array(r["time"]); a = np.array(r["altitude"])
    if ts == ref_ts:
        orb_rows.append({"Δt (s)": f"{ts:.0f} ← ref", "# Points": len(t),
                         "Alt Start (km)": f"{a[0]:.3f}", "Alt End (km)": f"{a[-1]:.3f}",
                         "Decay (km)": f"{a[0]-a[-1]:.3f}",
                         "RMS Δ Alt": "—", "RMS Δ Ecc": "—", "RMS Δ Inc": "—"})
    else:
        ot = np.array(r["time"])
        ia = interp_onto_reference(ref_time, ref_alt, ot, a)
        ie = interp_onto_reference(ref_time, ref_ecc, ot, np.array(r["eccentricity"]))
        ii = interp_onto_reference(ref_time, ref_inc, ot, np.array(r["inclination"]))
        orb_rows.append({"Δt (s)": f"{ts:.0f}", "# Points": len(t),
                         "Alt Start (km)": f"{a[0]:.3f}", "Alt End (km)": f"{a[-1]:.3f}",
                         "Decay (km)": f"{a[0]-a[-1]:.3f}",
                         "RMS Δ Alt": f"{np.sqrt(np.mean((ia-ref_alt)**2)):.6f} km",
                         "RMS Δ Ecc": f"{np.sqrt(np.mean((ie-ref_ecc)**2)):.3e}",
                         "RMS Δ Inc": f"{np.sqrt(np.mean((ii-ref_inc)**2)):.3e} °"})
print_table(orb_rows, "Orbital Numerical Summary")

# ==============================================================================
# SECTION 2 — ECLIPSE
# ==============================================================================
print("\n─── Section 2: Eclipse Analysis ───")

def ecl_scatter(events):
    if not events:
        return np.array([]), np.array([])
    return np.array([e[0] for e in events]) / 86400.0, np.array([e[1] for e in events]) / 60.0

# Plot 6 — Eclipse duration scatter
fig6, ax6 = plt.subplots(figsize=(12, 5)); _fig_style(fig6); _ax_style(ax6)
for i, ts in enumerate(sorted_ts):
    umbra_m, umbra_d = ecl_scatter(all_results[ts].get("umbra_events", []))
    pen_m, pen_d     = ecl_scatter(all_results[ts].get("penumbra_events", []))
    color = COLORS[i]
    lbl_base = f"Δt={ts:.0f}s" + (" ← ref" if ts == ref_ts else "")
    if len(umbra_m):
        ax6.scatter(umbra_m, umbra_d, color=color,
                    s=8 if ts == ref_ts else 4, alpha=0.85, marker="o",
                    zorder=3 if ts == ref_ts else 2,
                    label=f"{lbl_base} — umbra ({len(umbra_m)} ev)")
    if len(pen_m):
        ax6.scatter(pen_m, pen_d, color=color,
                    s=8 if ts == ref_ts else 4, alpha=0.45, marker="^",
                    zorder=3 if ts == ref_ts else 2,
                    label=f"{lbl_base} — penumbra ({len(pen_m)} ev)")
ax6.set_title("Eclipse Duration per Event — All Timesteps", fontsize=13)
ax6.set_xlabel("Time (days)"); ax6.set_ylabel("Eclipse Duration (min)")
ax6.legend(fontsize=8); plt.tight_layout()
savefig(fig6, "plot06_eclipse_scatter.png")

# Plot 8 — Cumulative umbra time
fig8, ax8 = plt.subplots(figsize=(12, 4)); _fig_style(fig8); _ax_style(ax8)
for i, ts in enumerate(sorted_ts):
    events = sorted(all_results[ts].get("umbra_events", []), key=lambda e: e[0])
    if events:
        cum_t   = [e[0] / 86400.0 for e in events]
        cum_dur = np.cumsum([e[1] / 3600.0 for e in events])
        ax8.plot(cum_t, cum_dur, color=COLORS[i], lw=2.2 if ts == ref_ts else 1.4,
                 ls="-" if ts == ref_ts else "--", alpha=0.9,
                 label=f"Δt={ts:.0f}s  total={cum_dur[-1]:.1f}h" + (" ← ref" if ts == ref_ts else ""))
ax8.set_title("Cumulative Umbra Time", fontsize=13)
ax8.set_xlabel("Time (days)"); ax8.set_ylabel("Cumulative Umbra (h)")
ax8.legend(fontsize=9); plt.tight_layout()
savefig(fig8, "plot08_cumulative_umbra.png")

# Eclipse statistics table
ref_umbra_total = sum(e[1] for e in ref.get("umbra_events", [])) / 3600.0
ecl_rows = []
for ts in sorted_ts:
    r = all_results[ts]
    u_ev = r.get("umbra_events", []); p_ev = r.get("penumbra_events", [])
    u_durs = [e[1] for e in u_ev]
    tot_u = sum(u_durs) / 3600.0
    ecl_rows.append({
        "Δt (s)":               f"{ts:.0f}" + (" ← ref" if ts == ref_ts else ""),
        "# Umbra":              len(u_ev),
        "# Penumbra":           len(p_ev),
        "Mean umbra (min)":     f"{np.mean(u_durs)/60:.2f}" if u_durs else "—",
        "Std umbra (min)":      f"{np.std(u_durs)/60:.2f}"  if u_durs else "—",
        "Total umbra (h)":      f"{tot_u:.2f}",
        "Δ Total vs ref":       "—" if ts == ref_ts else f"{tot_u - ref_umbra_total:+.3f} h",
    })
print_table(ecl_rows, "Eclipse Statistics Summary")

# ==============================================================================
# SECTION 3 — PROPULSION (only when drag compensation enabled)
# ==============================================================================
if SIM_TYPE == "Orbital Decay + Drag Compensation":
    print("\n─── Section 3: Propulsion Analysis ───")

    has_prop_data = all("propellant_remaining" in all_results[ts] for ts in sorted_ts)

    if not has_prop_data:
        print("  WARNING: propellant_remaining key not found in results — skipping propulsion plots.")
    else:
        ref_prop_rem = np.array(ref.get("propellant_remaining", []))
        ref_thrust   = np.array(ref.get("thrust_level", []))

        # Plot P1 — Propellant remaining
        figP1, axP1 = plt.subplots(figsize=(12, 5)); _fig_style(figP1); _ax_style(axP1)
        for i, ts in enumerate(sorted_ts):
            r  = all_results[ts]
            t  = np.array(r["time"]) / 86400.0
            pr = np.array(r.get("propellant_remaining", np.zeros(len(t))))
            axP1.plot(t, pr, color=COLORS[i], lw=2.2 if ts == ref_ts else 1.4,
                      ls="-" if ts == ref_ts else "--", alpha=0.9,
                      label=f"Δt={ts:.0f}s" + (" ← ref" if ts == ref_ts else ""))
        axP1.set_title("Propellant Remaining — All Timesteps", fontsize=13)
        axP1.set_xlabel("Time (days)"); axP1.set_ylabel("Propellant (kg)")
        axP1.legend(fontsize=9); plt.tight_layout()
        savefig(figP1, "plotP1_propellant_remaining.png")

        # Plot P2 — Propellant difference vs reference
        figP2, axP2 = plt.subplots(figsize=(12, 5)); _fig_style(figP2); _ax_style(axP2)
        for ts in comparison_ts:
            r  = all_results[ts]; ot = np.array(r["time"])
            pr = np.array(r.get("propellant_remaining", np.zeros(len(ot))))
            d  = interp_onto_reference(ref_time, ref_prop_rem, ot, pr) - ref_prop_rem
            c  = COLORS[sorted_ts.index(ts)]
            axP2.plot(ref_days, d, color=c, lw=1.6,
                      label=f"Δt={ts:.0f}s  RMS={np.sqrt(np.mean(d**2)):.6f} kg")
        axP2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        axP2.set_title(f"Propellant Difference vs Reference (Δt={ref_ts:.0f}s)", fontsize=13)
        axP2.set_xlabel("Time (days)"); axP2.set_ylabel("Δ Propellant (kg)")
        axP2.legend(fontsize=9); plt.tight_layout()
        savefig(figP2, "plotP2_propellant_difference.png")

        # Propulsion summary table
        prop_rows = []
        ref_used = float(PROP_MASS_KG) - float(ref_prop_rem[-1]) if len(ref_prop_rem) else 0.0
        for ts in sorted_ts:
            r     = all_results[ts]
            t_arr = np.array(r["time"])
            pr    = np.array(r.get("propellant_remaining", [float(PROP_MASS_KG)]))
            thr   = np.array(r.get("thrust_level", np.zeros(len(t_arr))))
            used  = float(PROP_MASS_KG) - float(pr[-1]) if len(pr) else 0.0
            duty  = float(np.mean(thr > 0)) * 100.0 if len(thr) else 0.0
            if ts == ref_ts:
                rms_str = "—"
            else:
                i_pr = interp_onto_reference(ref_time, ref_prop_rem, t_arr, pr)
                rms_str = f"{np.sqrt(np.mean((i_pr - ref_prop_rem)**2)):.6f} kg"
            prop_rows.append({
                "Δt (s)":              f"{ts:.0f}" + (" ← ref" if ts == ref_ts else ""),
                "Prop used (kg)":      f"{used:.4f}",
                "Prop remaining (kg)": f"{pr[-1]:.4f}" if len(pr) else "—",
                "Duty cycle (%)":      f"{duty:.2f}",
                "RMS Δ prop vs ref":   rms_str,
            })
        print_table(prop_rows, "Propulsion Statistics Summary")

# ==============================================================================
# SECTION 4 — POWER
# ==============================================================================
print("\n─── Section 4: Power Subsystem ───")

power_curves = {}
for ts in sorted_ts:
    r     = all_results[ts]
    t_arr = np.array(r["time"])
    illum = reconstruct_illumination(
        t_arr,
        r.get("umbra_events", []),
        r.get("penumbra_events", []),
    )
    Pg, Pc, bat, soc = simulate_power_posthoc(
        t_arr, illum,
        area       = float(PWR_SOLAR_AREA_M2),
        eff        = float(PWR_PANEL_EFFICIENCY),
        flux       = float(PWR_SOLAR_FLUX_W_M2),
        bat_cap    = float(PWR_BAT_CAPACITY_WH),
        bat_init   = float(PWR_BAT_INITIAL_WH),
        hk         = float(PWR_HOUSEKEEPING_W),
        panel_degr = float(PWR_PANEL_DEGR_YR),
        bat_degr   = float(PWR_BAT_DEGR_YR),
    )
    power_curves[ts] = {"time": t_arr, "P_gen": Pg, "P_cons": Pc, "bat": bat, "soc": soc}

ref_pwr  = power_curves[ref_ts]
ref_soc  = np.array(ref_pwr["soc"]) * 100.0
ref_pgen = np.array(ref_pwr["P_gen"])

# Plot 10 — Battery SoC
fig10, ax10 = plt.subplots(figsize=(12, 5)); _fig_style(fig10); _ax_style(ax10)
for i, ts in enumerate(sorted_ts):
    pc = power_curves[ts]
    ax10.plot(np.array(pc["time"]) / 86400.0, np.array(pc["soc"]) * 100.0,
              color=COLORS[i], lw=2.2 if ts == ref_ts else 1.4,
              ls="-" if ts == ref_ts else "--", alpha=0.9,
              label=f"Δt={ts:.0f}s" + (" ← ref" if ts == ref_ts else ""))
ax10.axhline(0,   color="tomato",    lw=1.0, ls=":", alpha=0.7)
ax10.axhline(100, color="limegreen", lw=1.0, ls=":", alpha=0.7)
ax10.set_title("Battery SoC — All Timesteps", fontsize=13)
ax10.set_xlabel("Time (days)"); ax10.set_ylabel("SoC (%)")
ax10.set_ylim(-5, 110); ax10.legend(fontsize=9); plt.tight_layout()
savefig(fig10, "plot10_battery_soc.png")

# Plot 11 — SoC difference vs reference
fig11, ax11 = plt.subplots(figsize=(12, 5)); _fig_style(fig11); _ax_style(ax11)
for ts in comparison_ts:
    pc   = power_curves[ts]; ot = np.array(pc["time"])
    isoc = interp_onto_reference(ref_time, ref_soc, ot, np.array(pc["soc"]) * 100.0)
    d    = isoc - ref_soc
    c    = COLORS[sorted_ts.index(ts)]
    ax11.plot(ref_days, d, color=c, lw=1.6,
              label=f"Δt={ts:.0f}s  RMS={np.sqrt(np.mean(d**2)):.3f} %")
ax11.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax11.set_title(f"Battery SoC Difference vs Reference (Δt={ref_ts:.0f}s)", fontsize=13)
ax11.set_xlabel("Time (days)"); ax11.set_ylabel("Δ SoC (%)")
ax11.legend(fontsize=9); plt.tight_layout()
savefig(fig11, "plot11_soc_difference.png")

# Plot 12 — Power generated (full simulation)
fig12, axes12 = plt.subplots(len(sorted_ts), 1, figsize=(12, 2.5 * len(sorted_ts)), sharex=True)
_fig_style(fig12)
if len(sorted_ts) == 1:
    axes12 = [axes12]
fig12.suptitle("Power Generated — Full Simulation", fontsize=13, fontweight="bold")
for ai, (ax12, ts) in enumerate(zip(axes12, sorted_ts)):
    pc = power_curves[ts]
    t  = np.array(pc["time"]) / 86400.0
    _ax_style(ax12)
    ax12.fill_between(t, np.array(pc["P_gen"]), step="post", alpha=0.55, color=COLORS[ai])
    ax12.step(t, np.array(pc["P_gen"]), where="post", color=COLORS[ai], lw=1.2)
    ax12.axhline(PWR_HOUSEKEEPING_W, color="tomato", lw=1.0, ls="--", alpha=0.7,
                 label=f"HK load {PWR_HOUSEKEEPING_W:.0f} W")
    ax12.set_ylabel(f"Δt={ts:.0f}s\nP (W)", fontsize=8)
    if ai == 0:
        ax12.legend(fontsize=8)
axes12[-1].set_xlabel("Time (days)"); plt.tight_layout()
savefig(fig12, "plot12_power_generated.png")

# Power summary table
pwr_rows = []
for ts in sorted_ts:
    pc       = power_curves[ts]
    t_arr    = np.array(pc["time"])
    soc_arr  = np.array(pc["soc"]) * 100.0
    pgen_arr = np.array(pc["P_gen"])
    if ts == ref_ts:
        rms_soc = "—"; rms_pg = "—"
    else:
        is_ = interp_onto_reference(ref_time, ref_soc,  t_arr, soc_arr)
        ip_ = interp_onto_reference(ref_time, ref_pgen, t_arr, pgen_arr)
        rms_soc = f"{np.sqrt(np.mean((is_ - ref_soc)**2)):.4f} %"
        rms_pg  = f"{np.sqrt(np.mean((ip_ - ref_pgen)**2)):.3f} W"
    pwr_rows.append({
        "Δt (s)":         f"{ts:.0f}" + (" ← ref" if ts == ref_ts else ""),
        "Mean SoC (%)":   f"{np.mean(soc_arr):.2f}",
        "Min SoC (%)":    f"{np.min(soc_arr):.2f}",
        "Final SoC (%)":  f"{soc_arr[-1]:.2f}",
        "Mean P_gen (W)": f"{np.mean(pgen_arr):.2f}",
        "RMS Δ SoC":      rms_soc,
        "RMS Δ P_gen":    rms_pg,
    })
print_table(pwr_rows, "Power Statistics Summary")

# ==============================================================================
_elapsed = time.perf_counter() - _t_start
_h  = int(_elapsed // 3600)
_m  = int((_elapsed % 3600) // 60)
_s  = _elapsed % 60
print(f"\n✅ All done.  Figures saved to: {out_dir}")
print(f"⏱  Total wall-clock time: {_h:02d}h {_m:02d}m {_s:05.2f}s  ({_elapsed:.1f} s)")
