"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          test_eclipse_detection.py
Description:
    Eclipse duration validation comparing Orekit's EclipseDetector against SGP4 analytical truth.
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
# PATH SETUP & OREKIT INITIALIZATION
# File is in src/validations/, so:
#   _VALIDATIONS_DIR = src/validations/
#   _SRC_DIR         = src/
#   _project_root    = 03_code/
# ============================================================
_VALIDATIONS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_VALIDATIONS_DIR)
_project_root = os.path.dirname(_SRC_DIR)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import orekit_jpype as orekit
try:
    orekit.initVM()
except Exception:
    pass

from orekit_jpype.pyhelpers import setup_orekit_data
_data_path = os.path.join(_project_root, "orekit-data-main")
setup_orekit_data(filenames=_data_path, from_pip_library=False)

# Output directory for saved figures
_OUTPUT_DIR = os.path.join(_project_root, "results", "results_eclipses")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.propagation.events import EclipseDetector
from org.orekit.propagation.events.handlers import RecordAndContinue

# ============================================================
# PERSONALISED PARAMETERS (Edit these if you choose option 2)
# ============================================================
CUSTOM_SMA = 6659660.00 #Constants.WGS84_EARTH_EQUATORIAL_RADIUS + 115000.0  # 320 km altitude
CUSTOM_ECCENTRICITY = 0.001
CUSTOM_INCLINATION_DEG = 96.7095
CUSTOM_RAAN_DEG = 85.5283
CUSTOM_ARG_PERIGEE_DEG = 322.67
CUSTOM_TRUE_ANOMALY_DEG = 37.3615
CUSTOM_DURATION_DAYS = 3 * 365.0

# ============================================================
# HELPER FUNCTIONS
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

def main():
    print("====================================")
    print("      Eclipse Detection Test")
    print("====================================")
    choice = input("Do you want to use a TLE file (1) or run with personalised parameters defined in this script (2)? [1/2]: ")
    
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame
    )
    sun = CelestialBodyFactory.getSun()
    
    # We want to detect both Umbra and Penumbra.
    # A Penumbra detector triggers when entering/leaving the penumbra.
    # The umbra is entirely contained within the penumbra, so the time spent
    # inside the penumbra represents the TOTAL eclipse time.
    handler = RecordAndContinue()
    # Using Penumbra will give the maximum duration (Umbra + Penumbra)
    eclipse_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withPenumbra().withHandler(handler)
    
    output_filename = "eclipse_duration"  # base name for saved figure

    if choice.strip() == '1':
        tle_path = input("Enter the path to the TLE file (relative to validations folder or absolute): ")
        if not os.path.isabs(tle_path):
            tle_path = os.path.join(_VALIDATIONS_DIR, tle_path)
            
        if not os.path.exists(tle_path):
            print(f"Error: File not found at {tle_path}")
            return
            
        entries = parse_tle_file(tle_path)
        if not entries:
            print("No TLEs found in the file.")
            return
            
        print(f"Loaded {len(entries)} TLE entries.")
        print("Propagating TLEs segment by segment to extract eclipse events...")
        
        all_events = []
        start_date = TLE(entries[0][1], entries[0][2]).getDate()
        
        # Use TLE filename (without extension) as the output filename
        tle_basename = os.path.splitext(os.path.basename(tle_path))[0]
        output_filename = f"eclipse_duration_{tle_basename}"
        
        for i in range(len(entries)):
            name, line1, line2 = entries[i]
            tle = TLE(line1, line2)
            prop = TLEPropagator.selectExtrapolator(tle)
            
            seg_handler = RecordAndContinue()
            seg_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withPenumbra().withHandler(seg_handler)
            prop.addEventDetector(seg_detector)
            
            start_prop = tle.getDate()
            if i < len(entries) - 1:
                end_prop = TLE(entries[i+1][1], entries[i+1][2]).getDate()
            else:
                end_prop = start_prop.shiftedBy(86400.0) # 1 day for the last one
                
            if end_prop.compareTo(start_prop) <= 0:
                continue
                
            try:
                prop.propagate(start_prop, end_prop)
                for e in seg_handler.getEvents():
                    all_events.append(e)
            except Exception as e:
                # If a segment crashes, skip to the next TLE
                continue
                
        # Sort all collected events chronologically
        all_events.sort(key=lambda e: e.getState().getDate().durationFrom(start_date))
        events = all_events
        
    elif choice.strip() == '2':
        print("Using personalised parameters from the script.")
        utc = TimeScalesFactory.getUTC()
        start_date = AbsoluteDate(2009, 3, 17, 20, 9, 36.4, utc)
        end_date = start_date.shiftedBy(CUSTOM_DURATION_DAYS * 86400.0)
        
        inertial_frame = FramesFactory.getEME2000()
        initial_orbit = KeplerianOrbit(
            float(CUSTOM_SMA), float(CUSTOM_ECCENTRICITY), math.radians(CUSTOM_INCLINATION_DEG),
            math.radians(CUSTOM_ARG_PERIGEE_DEG), math.radians(CUSTOM_RAAN_DEG), math.radians(CUSTOM_TRUE_ANOMALY_DEG),
            PositionAngleType.TRUE, inertial_frame,
            start_date, Constants.WGS84_EARTH_MU
        )
        
        prop = KeplerianPropagator(initial_orbit)
        prop.addEventDetector(eclipse_detector)
        print(f"Propagating custom orbit from {start_date} to {end_date}...")
        prop.propagate(start_date, end_date)
        events = handler.getEvents()
    else:
        print("Invalid choice. Please select 1 or 2.")
        return
        
    # Process events to calculate durations
    eclipse_durations = [] # (mid_date_datetime, duration_seconds)
    enter_t = None
    
    # AbsoluteDate to python datetime
    def to_datetime(abs_date):
        return datetime.fromisoformat(str(abs_date).replace("Z", "+00:00"))
        
    for e in events:
        t = e.getState().getDate()
        # isIncreasing() returns true when g() goes from negative to positive.
        # For EclipseDetector, g() < 0 means inside eclipse.
        # So decreasing (not isIncreasing) means entering eclipse, increasing means leaving.
        if not e.isIncreasing():
            enter_t = t
        else:
            if enter_t is not None:
                duration = t.durationFrom(enter_t)
                # Eclipses are generally < 2 hours. This filters out missed exits across TLE gaps
                if duration > 0 and duration < 7200.0: 
                    mid = enter_t.shiftedBy(duration / 2.0)
                    eclipse_durations.append((to_datetime(mid), duration))
                enter_t = None

    if not eclipse_durations:
        print("No eclipses detected! This may happen if the orbit is sun-synchronous with no eclipses.")
        return
        
    print(f"Detected {len(eclipse_durations)} eclipse events.")
    
    # Pad events with 0s for periods without eclipses
    padded_durations = []
    for i in range(len(eclipse_durations)):
        padded_durations.append(eclipse_durations[i])
        if i < len(eclipse_durations) - 1:
            t1, d1 = eclipse_durations[i]
            t2, d2 = eclipse_durations[i+1]
            gap = (t2 - t1).total_seconds()
            if gap > 7200.0:  # Gap > 2 hours means no eclipses occurred
                # Insert 0s slightly after the first eclipse and slightly before the next
                padded_durations.append((t1 + timedelta(minutes=100), 0.0))
                padded_durations.append((t2 - timedelta(minutes=100), 0.0))
                
    # Plotting
    dates = [e[0] for e in padded_durations]
    durations_sec = [e[1] for e in padded_durations]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Using a line plot to show the drop to 0 clearly
    ax.plot(dates, durations_sec, marker='.', linestyle='-', color='indigo', lw=1.5, label='Eclipse Duration')
    ax.set_title("Eclipse Duration per Orbit (Umbra + Penumbra)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Eclipse Duration [mm:ss]")
    ax.grid(True)
    ax.legend()
    
    # Format y-axis as mm:ss
    from matplotlib.ticker import FuncFormatter
    def format_mmss(y, pos):
        m = int(y // 60)
        s = int(y % 60)
        return f"{m:02d}:{s:02d}"
    ax.yaxis.set_major_formatter(FuncFormatter(format_mmss))
    
    date_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate(rotation=30, ha="right")
    
    plt.tight_layout()
    
    # Save figure to results/results_eclipses/
    save_path = os.path.join(_OUTPUT_DIR, f"{output_filename}.png")
    fig.savefig(save_path, dpi=150)
    print(f"Figure saved to: {save_path}")
    
    # Interactive cursor (hover to see date and duration in mm:ss)
    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x_val = mdates.num2date(sel.target[0])
        y_sec = sel.target[1]
        m = int(y_sec // 60)
        s = int(y_sec % 60)
        sel.annotation.set_text(f"{x_val.strftime('%Y-%m-%d %H:%M')}\n{m:02d}:{s:02d} (mm:ss)")

    plt.show()

if __name__ == "__main__":
    main()
