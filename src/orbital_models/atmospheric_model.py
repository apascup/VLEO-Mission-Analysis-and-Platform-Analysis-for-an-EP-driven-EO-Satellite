"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          atmospheric_model.py
Description:
    High-fidelity numerical orbit propagation, atmospheric drag models, SRP, and eclipse detection via Orekit.
===============================================================================
"""

import sys
import os
import math
import matplotlib.pyplot as plt
import mplcursors
import orbital_plot_style as _ops

# Basic Orekit imports
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# Imports for the environment (Earth, Sun, Atmosphere, Gravity)
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation, JB2008SpaceEnvironmentData, CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import NRLMSISE00, JB2008, DTM2000, HarrisPriester
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction

# Imports for the forces (Drag and SRP)
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.propagation.events import EclipseDetector, AltitudeDetector
from org.orekit.propagation.events.handlers import RecordAndContinue, StopOnDecreasing, StopOnIncreasing
from org.orekit.propagation.sampling import OrekitFixedStepHandler
import jpype

def run_simulation(params, model_type="nrlmsise00", propulsion_model=None, compensation_mode="constant"):
    """
    Numerical orbit propagation using Orekit with:
    - Holmes-Featherstone gravity & Third Body Attraction (Sun/Moon)
    - Selectable atmospheric density model
    - isotropic drag model
    """
    model_type = model_type.lower()
    print(f"Starting simulation with {model_type.upper()}...")
    
    # ==========================================
    # 1. TIME AND REFERENCE FRAMES CONFIGURATION
    # ==========================================
    if "start_date" in params:
        start_date = params["start_date"]
    else:
        utc = TimeScalesFactory.getUTC()
        start_date = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)
    
    inertial_frame = FramesFactory.getEME2000()
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # ==========================================
    # 2. ENVIRONMENT DEFINITION (EARTH, SUN, ATMOSPHERE)
    # ==========================================
    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame
    )
    
    sun = CelestialBodyFactory.getSun()
    
    if model_type == "nrlmsise00":
        space_weather_data = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
        )
        #space_weather_data = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
        atmosphere = NRLMSISE00(space_weather_data, sun, earth_shape)
    elif model_type == "jb2008":
        try:
            space_weather_data = JB2008SpaceEnvironmentData("SOLFSMY.TXT", "DTCFILE.TXT")
        except Exception as e:
            print(f"Error loading JB2008 data (ensure SOLFSMY.TXT and DTCFILE.TXT are in orekit-data): {e}")
            raise e
        atmosphere = JB2008(space_weather_data, sun, earth_shape)
    elif model_type == "dtm2000":
        space_weather_data = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
        )
        #space_weather_data = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
        atmosphere = DTM2000(space_weather_data, sun, earth_shape)
    elif model_type == "harrispriester":
        atmosphere = HarrisPriester(sun, earth_shape)
        space_weather_data = None
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # ==========================================
    # 3. INITIAL ORBIT DEFINITION
    # ==========================================
    sma_m = params.get("sma_m")
    if sma_m is None:
        altitude_m = params.get("altitude", 300000.0)
        sma_m = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + altitude_m
        
    eccentricity = params.get("eccentricity", 1e-8)
    inclination = math.radians(params.get("inclination", 51.6))
    raan = math.radians(params.get("raan", 0.0))
    arg_perigee = math.radians(params.get("arg_perigee", 0.0))
    true_anomaly = math.radians(params.get("true_anomaly", 0.0))

    initial_orbit = KeplerianOrbit(
        sma_m, eccentricity, inclination,
        arg_perigee, raan, true_anomaly,
        PositionAngleType.TRUE, inertial_frame,
        start_date, Constants.WGS84_EARTH_MU
    )
    
    mass = float(params.get("mass", 500.0))
    initial_state = SpacecraftState(initial_orbit, mass)

    # ==========================================
    # 4. PROPAGATOR SETUP
    # ==========================================
    min_step = 0.001
    max_step = 1000.0
    position_tolerance = 10.0
    
    tolerances = NumericalPropagator.tolerances(
        position_tolerance, initial_orbit, initial_orbit.getType()
    )
    
    area = float(params.get("cross_section", 2.0))
    cd = float(params.get("drag_coeff", 2.2))
    cr = float(params.get("reflectivity_coeff", 1.5))
    
    umbra_handler = RecordAndContinue()
    penumbra_handler = RecordAndContinue()
    umbra_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withUmbra().withHandler(umbra_handler)
    penumbra_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_shape).withPenumbra().withHandler(penumbra_handler)

    # Determine which propagators are needed.
    # maintenance / staged_tracking use a per-step Delta-V loop (single coast propagator).
    # duty_cycle / goal use ConstantThrustManeuver (continuous force) + event-based switching.
    _need_thrust_prop = (
        propulsion_model is not None
        and compensation_mode not in ["maintenance", "staged_tracking"]
    )

    def build_propagator(with_thrust, direction=1.0):
        integrator = DormandPrince853Integrator(
            min_step, max_step,
            tolerances[0], tolerances[1]
        )
        prop = NumericalPropagator(integrator)

        gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
        prop.addForceModel(HolmesFeatherstoneAttractionModel(earth_frame, gravity_provider))
        prop.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
        prop.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
        prop.addForceModel(DragForce(atmosphere, IsotropicDrag(area, cd)))

        srp = SolarRadiationPressure(sun, earth_shape, IsotropicRadiationSingleCoefficient(area, cr))
        prop.addForceModel(srp)
        prop.addEventDetector(umbra_detector)
        prop.addEventDetector(penumbra_detector)

        if with_thrust and _need_thrust_prop:
            from org.orekit.forces.maneuvers import ConstantThrustManeuver
            from org.orekit.frames import LOFType
            from org.orekit.attitudes import LofOffset
            from org.hipparchus.geometry.euclidean.threed import Vector3D

            attitude_provider = LofOffset(inertial_frame, LOFType.TNW)
            prop.setAttitudeProvider(attitude_provider)

            thrust_dir = Vector3D.PLUS_I if direction > 0 else Vector3D.MINUS_I
            thrust_force = ConstantThrustManeuver(
                start_date.shiftedBy(-86400.0),  # valid from before simulation start
                1e10,                            # effectively infinite duration
                float(propulsion_model.thrust),
                float(propulsion_model.isp),
                thrust_dir,
            )
            prop.addForceModel(thrust_force)

        return prop

    # Always build the coast propagator.
    propagator_coast = build_propagator(with_thrust=False)
    # Thrust propagators only needed for duty_cycle / goal modes.
    propagator_thrust       = build_propagator(with_thrust=True, direction= 1.0) if _need_thrust_prop else None
    propagator_thrust_retro = build_propagator(with_thrust=True, direction=-1.0) if _need_thrust_prop else None

    # ==========================================
    # 5. EXECUTION & DATA EXTRACTION
    # ==========================================
    duration = float(params.get("duration", 86400.0))
    target_date = start_date.shiftedBy(duration)
    time_step = float(params.get("time_step", 60.0))
    # Goal-mode parameters (read once before the loop)
    _goal_alt_km    = params.get("goal_altitude_km")   # None → not in goal mode
    _goal_offset_km = float(params.get("goal_offset_km") or 1.0)
    # Power subsystem model (optional; None = no power constraint)
    power_model = params.get("power_model", None)

    # Storage lists
    time_list = []
    altitude_list = []
    sma_list = []
    density_list = []
    inclination_list = []
    raan_list = []
    eccentricity_list = []
    f107_list = []

    # Extra storage for propulsion
    if propulsion_model is not None:
        thrust_on_list = []
        thrust_level_list = []   # actual thrust applied this step [N]
        propellant_rem_list = []
        mass_list = []
        burn_time_list = []
        propellant_used_list = []
        cycles_list = []

    # Extra storage for power subsystem
    if power_model is not None:
        battery_Wh_list    = []
        P_gen_list         = []
        P_cons_list        = []
        illumination_list  = []
    
    # (initial data point and propagation are handled in sections 6 & 7 below)

    # =========================================================================
    # 5b.  FIXED-STEP HANDLER  (used by Path A and Path C)
    # =========================================================================
    # Defined as a closure so it can read the output lists and shared state
    # (atmosphere, start_date, space_weather_data …) directly.
    # 'is_thrusting' and 'thrust_value' are set per-segment for Path C.
    MEAN_EARTH_RADIUS = 6371000.0

    @jpype.JImplements(OrekitFixedStepHandler)
    class _OrbitDataCollector:
        def __init__(self, is_thrusting=False, thrust_value=0.0):
            self._is_thrusting = bool(is_thrusting)
            self._thrust_value = float(thrust_value)

        @jpype.JOverride
        def init(self, s0, t, step):
            pass  # nothing to initialise; lists are pre-populated with t=0

        @jpype.JOverride
        def handleStep(self, state):
            """Called at each fixed output step during propagation."""
            current_date_ = state.getDate()
            pos_          = state.getPosition()

            kep_orbit_ = KeplerianOrbit(state.getOrbit())
            sma_km_    = kep_orbit_.getA() / 1000.0
            
            # Calculate geodetic altitude
            geodetic_point_ = earth_shape.transform(pos_, inertial_frame, current_date_)
            alt_km_ = geodetic_point_.getAltitude() / 1000.0
            
            rho_       = atmosphere.getDensity(current_date_, pos_, inertial_frame)
            inc_deg_   = math.degrees(kep_orbit_.getI())
            raan_deg_  = math.degrees(kep_orbit_.getRightAscensionOfAscendingNode()) % 360.0
            ecc_       = kep_orbit_.getE()

            time_list.append(current_date_.durationFrom(start_date))
            sma_list.append(sma_km_)
            altitude_list.append(alt_km_)
            density_list.append(rho_)
            inclination_list.append(inc_deg_)
            raan_list.append(raan_deg_)
            eccentricity_list.append(ecc_)

            # F10.7 solar flux
            f107_ = None
            if space_weather_data is not None:
                if hasattr(space_weather_data, "getF10"):
                    try: f107_ = space_weather_data.getF10(current_date_)
                    except: pass
                elif hasattr(space_weather_data, "getDailyFlux"):
                    try: f107_ = space_weather_data.getDailyFlux(current_date_)
                    except: pass
            f107_list.append(f107_)

            # Propulsion data (for Path C segments)
            if propulsion_model is not None:
                thrust_on_list.append(self._is_thrusting)
                thrust_level_list.append(self._thrust_value if self._is_thrusting else 0.0)
                # propellant_rem / burn_time are updated at segment boundaries;
                # within a segment they stay constant (ConstantThrustManeuver handles mass).
                propellant_rem_list.append(propulsion_model.propellant_mass)
                mass_list.append(float(state.getMass()))
                burn_time_list.append(propulsion_model.burn_time)
                propellant_used_list.append(propulsion_model.propellant_used)
                cycles_list.append(propulsion_model.cycles)

            # Power data (if power model is active) — per-step illumination
            if power_model is not None:
                try:
                    g_umbra_    = float(umbra_detector.g(state))
                    g_penumbra_ = float(penumbra_detector.g(state))
                    in_umbra_    = g_umbra_    < 0.0
                    in_penumbra_ = g_penumbra_ < 0.0 and not in_umbra_
                except Exception:
                    in_umbra_ = in_penumbra_ = False

                if in_umbra_:
                    illum_ = 0.0
                elif in_penumbra_:
                    illum_ = 0.5
                else:
                    illum_ = 1.0

                _pw_ok, _pg, _pc = power_model.update(illum_, self._is_thrusting, float(time_step))
                battery_Wh_list.append(power_model.battery_Wh)
                P_gen_list.append(_pg)
                P_cons_list.append(_pc)
                illumination_list.append(illum_)

        @jpype.JOverride
        def finish(self, finalState):
            pass

    # =========================================================================
    # 5c.  HELPER — collect one data point from a SpacecraftState object
    #       (used for t=0 initial snapshot and Path B per-step extraction)
    # =========================================================================
    def _collect_state(state, current_date_, rho_=None,
                       prop_applied_thrust=0.0,
                       illum_=1.0, P_gen_=0.0, P_cons_=0.0):
        """Append one row of data from a SpacecraftState to all output lists."""
        pos_       = state.getPosition()
        kep_orbit_ = KeplerianOrbit(state.getOrbit())
        sma_km_    = kep_orbit_.getA() / 1000.0
        geodetic_point_ = earth_shape.transform(pos_, inertial_frame, current_date_)
        alt_km_ = geodetic_point_.getAltitude() / 1000.0
        if rho_ is None:
            rho_ = atmosphere.getDensity(current_date_, pos_, inertial_frame)
        inc_deg_   = math.degrees(kep_orbit_.getI())
        raan_deg_  = math.degrees(kep_orbit_.getRightAscensionOfAscendingNode()) % 360.0
        ecc_       = kep_orbit_.getE()

        time_list.append(current_date_.durationFrom(start_date))
        sma_list.append(sma_km_)
        altitude_list.append(alt_km_)
        density_list.append(rho_)
        inclination_list.append(inc_deg_)
        raan_list.append(raan_deg_)
        eccentricity_list.append(ecc_)

        f107_ = None
        if space_weather_data is not None:
            if hasattr(space_weather_data, "getF10"):
                try: f107_ = space_weather_data.getF10(current_date_)
                except: pass
            elif hasattr(space_weather_data, "getDailyFlux"):
                try: f107_ = space_weather_data.getDailyFlux(current_date_)
                except: pass
        f107_list.append(f107_)

        if propulsion_model is not None:
            thrust_on_list.append(propulsion_model.is_on)
            thrust_level_list.append(prop_applied_thrust)
            propellant_rem_list.append(propulsion_model.propellant_mass)
            mass_list.append(float(state.getMass()))
            burn_time_list.append(propulsion_model.burn_time)
            propellant_used_list.append(propulsion_model.propellant_used)
            cycles_list.append(propulsion_model.cycles)

        if power_model is not None:
            battery_Wh_list.append(power_model.battery_Wh)
            P_gen_list.append(P_gen_)
            P_cons_list.append(P_cons_)
            illumination_list.append(illum_)

        return alt_km_, rho_

    # =========================================================================
    # 6.  INITIAL DATA POINT  (t = 0, shared by all paths)
    # =========================================================================
    print("Propagating orbit...")
    current_date  = start_date
    current_state = initial_state

    pos = current_state.getPosition()
    rho = atmosphere.getDensity(current_date, pos, inertial_frame)
    orbit_kep = KeplerianOrbit(current_state.getOrbit())
    alt_km = (orbit_kep.getA() - MEAN_EARTH_RADIUS) / 1000.0

    _p0_gen = power_model.P_solar_max if power_model is not None else 0.0
    _p0_hk  = power_model.housekeeping_power_W if power_model is not None else 0.0
    _collect_state(
        current_state, current_date, rho_=rho,
        illum_=1.0, P_gen_=_p0_gen, P_cons_=_p0_hk,
    )

    # =========================================================================
    # 7.  EXECUTION  — three paths
    # =========================================================================

    if propulsion_model is None:
        # ── PATH A: Orbital Decay Only ────────────────────────────────────────
        # Single propagate() call — DormandPrince853 runs uninterrupted.
        # setInitialState is called exactly once; zero mid-run resets.
        collector = _OrbitDataCollector()
        propagator_coast.setInitialState(initial_state)
        propagator_coast.getMultiplexer().clear()
        propagator_coast.getMultiplexer().add(float(time_step), collector)
        try:
            propagator_coast.propagate(target_date)
        except Exception as e:
            print(f"\n[!] Simulation stopped early: {e}")

    elif compensation_mode in ["maintenance", "staged_tracking"]:
        # ── PATH B: Variable-thrust drag compensation ─────────────────────────
        # Thrust magnitude is computed each step from live drag, so a state
        # modification (ΔV injection) is required before every next segment.
        # setInitialState is called once per output step — same as before,
        # but this is unavoidable for these modes.
        from org.orekit.utils import PVCoordinates as _PVC
        from org.orekit.orbits import CartesianOrbit as _CO
        g0  = 9.80665
        mu  = Constants.WGS84_EARTH_MU

        while current_date.compareTo(target_date) < 0:
            next_date = current_date.shiftedBy(time_step)
            if next_date.compareTo(target_date) > 0:
                next_date = target_date
            dt = float(next_date.durationFrom(current_date))

            # ── Duty cycle toggling ──
            if compensation_mode == "duty_cycle" and propulsion_model is not None:
                alt_m = alt_km * 1000.0
                if alt_m < propulsion_model.h_min and not getattr(propulsion_model, "is_on", False):
                    propulsion_model.is_on = True
                elif alt_m > propulsion_model.h_max and getattr(propulsion_model, "is_on", False):
                    propulsion_model.is_on = False

            # ── Power evaluation ──
            _power_thruster_allowed = True
            _illumination = 1.0
            _P_gen_step   = 0.0
            _P_cons_step  = 0.0

            if power_model is not None:
                try:
                    g_umbra_    = float(umbra_detector.g(current_state))
                    g_penumbra_ = float(penumbra_detector.g(current_state))
                    _in_umbra    = g_umbra_    < 0.0
                    _in_penumbra = g_penumbra_ < 0.0 and not _in_umbra
                except Exception:
                    _in_umbra = _in_penumbra = False

                _illumination = 0.0 if _in_umbra else (0.5 if _in_penumbra else 1.0)
                _thruster_requesting = False
                if propulsion_model.propellant_mass > 0:
                    if compensation_mode == "duty_cycle":
                        _thruster_requesting = getattr(propulsion_model, "is_on", False)
                    else:
                        _thruster_requesting = True
                _power_thruster_allowed, _P_gen_step, _P_cons_step = power_model.update(
                    _illumination, _thruster_requesting, dt
                )

            # ── Propagate one step (coast only — ΔV injected after) ──
            propagator_coast.setInitialState(current_state)
            try:
                current_state = propagator_coast.propagate(next_date)
            except Exception as e:
                print(f"\n[!] Simulation stopped early at {current_date}!")
                print(f"[!] {e}")
                propulsion_model.turn_off("Simulation Exception")
                break

            current_date = next_date

            # ── ΔV injection for drag compensation ──
            _applied_thrust = 0.0
            if propulsion_model.propellant_mass > 0:
                pv      = current_state.getPVCoordinates(inertial_frame)
                vel_vec = pv.getVelocity()
                v_mag   = float(vel_vec.getNorm())

                pv_earth = current_state.getPVCoordinates(earth_frame)
                v_rel    = float(pv_earth.getVelocity().getNorm())

                pos = current_state.getPosition()
                rho = atmosphere.getDensity(current_date, pos, inertial_frame)

                cur_mass = float(mass - propulsion_model.propellant_used)
                F_drag   = 0.5 * float(rho) * v_rel * v_rel * cd * area
                F_req    = 0.0

                if not _power_thruster_allowed:
                    F_req = 0.0
                elif compensation_mode == "maintenance":
                    F_req = F_drag
                elif compensation_mode == "staged_tracking":
                    target_sma_func = params.get("target_sma_func")
                    if target_sma_func:
                        a_target_current = target_sma_func(current_date)
                        a_target_next    = target_sma_func(current_date.shiftedBy(float(dt)))
                        a_dot_target     = (a_target_next - a_target_current) / dt
                        term2 = (cur_mass / v_mag) * (mu / (2.0 * a_target_current ** 2)) * a_dot_target
                        F_req = F_drag + term2
                elif compensation_mode == "duty_cycle":
                    if getattr(propulsion_model, "is_on", False):
                        F_req = float(propulsion_model.thrust)

                max_thr = float(propulsion_model.thrust)
                F_applied = max(-max_thr, min(max_thr, F_req))

                if abs(F_applied) > 1e-6:
                    if not getattr(propulsion_model, "_was_thrusting", False):
                        if propulsion_model.cycles < propulsion_model.max_cycles:
                            propulsion_model.cycles += 1
                            propulsion_model._was_thrusting = True
                        else:
                            F_applied = 0.0
                else:
                    propulsion_model._was_thrusting = False

                if abs(F_applied) > 1e-6 and v_mag > 0 and cur_mass > 0:
                    dv       = (F_applied / cur_mass) * dt
                    vel_unit = vel_vec.scalarMultiply(1.0 / v_mag)
                    new_vel  = vel_vec.add(vel_unit.scalarMultiply(dv))
                    new_pv   = _PVC(pv.getPosition(), new_vel)
                    new_orb  = _CO(new_pv, inertial_frame, current_date, mu)

                    mdot     = abs(F_applied) / (float(propulsion_model.isp) * g0)
                    consumed = mdot * dt
                    if consumed > propulsion_model.propellant_mass:
                        consumed  = propulsion_model.propellant_mass
                        sign      = 1.0 if F_applied >= 0 else -1.0
                        F_applied = sign * consumed * float(propulsion_model.isp) * g0 / dt
                        dv        = (F_applied / cur_mass) * dt
                        new_vel   = vel_vec.add(vel_unit.scalarMultiply(dv))
                        new_pv    = _PVC(pv.getPosition(), new_vel)
                        new_orb   = _CO(new_pv, inertial_frame, current_date, mu)

                    propulsion_model.propellant_used += consumed
                    propulsion_model.propellant_mass -= consumed
                    propulsion_model.burn_time       += dt
                    propulsion_model.is_on            = True
                    cur_mass -= consumed
                    current_state = SpacecraftState(
                        new_orb, current_state.getAttitude(), max(cur_mass, 1.0)
                    )
                    _applied_thrust = F_applied
                else:
                    new_mass      = max(mass - propulsion_model.propellant_used, 1.0)
                    current_state = SpacecraftState(
                        current_state.getOrbit(), current_state.getAttitude(), new_mass
                    )
            else:
                new_mass      = max(mass - propulsion_model.propellant_used, 1.0)
                current_state = SpacecraftState(
                    current_state.getOrbit(), current_state.getAttitude(), new_mass
                )

            pos = current_state.getPosition()
            rho = atmosphere.getDensity(current_date, pos, inertial_frame)
            _collect_state(
                current_state, current_date, rho_=rho,
                prop_applied_thrust=_applied_thrust,
                illum_=_illumination, P_gen_=_P_gen_step, P_cons_=_P_cons_step,
            )
            alt_km = altitude_list[-1]

    elif compensation_mode == "duty_cycle":
        # ── PATH B2: OPTIMIZED DUTY CYCLE ──────────────────────────────────────
        # Unlike standard Path B which resets the integrator every step to inject
        # a variable Delta V, this path evaluates logic every step but only restarts
        # the integrator when the thruster transitions between ON and OFF.
        # This reduces setInitialState calls from ~2.1 million to ~10,000.
        
        _is_thrusting = getattr(propulsion_model, "is_on", False)
        _active_propagator = propagator_thrust if _is_thrusting else propagator_coast
        _active_propagator.setInitialState(current_state)

        propagator_coast.getMultiplexer().clear()
        if propagator_thrust: propagator_thrust.getMultiplexer().clear()

        while current_date.compareTo(target_date) < 0:
            next_date = current_date.shiftedBy(time_step)
            if next_date.compareTo(target_date) > 0:
                next_date = target_date
            dt = float(next_date.durationFrom(current_date))

            # Duty cycle toggling based on current altitude
            if propulsion_model is not None:
                alt_m = alt_km * 1000.0
                if alt_m < propulsion_model.h_min and not getattr(propulsion_model, "is_on", False):
                    propulsion_model.is_on = True
                elif alt_m > propulsion_model.h_max and getattr(propulsion_model, "is_on", False):
                    propulsion_model.is_on = False

            # Power evaluation
            _power_thruster_allowed = True
            _illumination = 1.0
            _P_gen_step   = 0.0
            _P_cons_step  = 0.0

            if power_model is not None:
                try:
                    g_umbra_    = float(umbra_detector.g(current_state))
                    g_penumbra_ = float(penumbra_detector.g(current_state))
                    _in_umbra    = g_umbra_    < 0.0
                    _in_penumbra = g_penumbra_ < 0.0 and not _in_umbra
                except Exception:
                    _in_umbra = _in_penumbra = False

                _illumination = 0.0 if _in_umbra else (0.5 if _in_penumbra else 1.0)
                _thruster_requesting = getattr(propulsion_model, "is_on", False) if propulsion_model.propellant_mass > 0 else False
                _power_thruster_allowed, _P_gen_step, _P_cons_step = power_model.update(
                    _illumination, _thruster_requesting, dt
                )

            # Determine actual thrust state for this step
            should_thrust = False
            if propulsion_model.propellant_mass > 0 and getattr(propulsion_model, "is_on", False) and _power_thruster_allowed:
                should_thrust = True

            # Switch propagator if needed
            if should_thrust != _is_thrusting:
                if should_thrust and propulsion_model.cycles >= propulsion_model.max_cycles:
                    should_thrust = False # No more cycles allowed
                else:
                    if should_thrust:
                        propulsion_model.cycles += 1
                        _active_propagator = propagator_thrust
                    else:
                        _active_propagator = propagator_coast
                    _active_propagator.setInitialState(current_state)
                    _is_thrusting = should_thrust
            
            # Propagate! (NO setInitialState if we didn't switch)
            try:
                current_state = _active_propagator.propagate(next_date)
            except Exception as e:
                print(f"\n[!] Simulation stopped early at {current_date}!")
                print(f"[!] {e}")
                propulsion_model.turn_off("Simulation Exception")
                break

            current_date = next_date

            # Propellant mass tracking
            if _is_thrusting:
                g0    = 9.80665
                mdot  = float(propulsion_model.thrust) / (float(propulsion_model.isp) * g0)
                consumed = mdot * dt
                if consumed > propulsion_model.propellant_mass:
                    consumed = propulsion_model.propellant_mass
                propulsion_model.propellant_used += consumed
                propulsion_model.propellant_mass -= consumed
                propulsion_model.burn_time       += dt

            # Collect data
            pos = current_state.getPosition()
            rho = atmosphere.getDensity(current_date, pos, inertial_frame)
            _collect_state(
                current_state, current_date, rho_=rho,
                prop_applied_thrust=float(propulsion_model.thrust) if _is_thrusting else 0.0,
                illum_=_illumination, P_gen_=_P_gen_step, P_cons_=_P_cons_step,
            )
            alt_km = altitude_list[-1]

            if propulsion_model.propellant_mass <= 0 and _is_thrusting:
                _is_thrusting = False
                _active_propagator = propagator_coast
                _active_propagator.setInitialState(current_state)

    else:
        # ── PATH C: goal — event-based segment propagation ───────
        # AltitudeDetectors fire when the spacecraft crosses h_min or h_max.
        # Each call to propagate() runs until the next crossing (or end of sim),
        # so setInitialState is called only at thrust-switching boundaries.
        # setInitialState calls ≈ 2 × (number of thrust cycles)  instead of
        # ≈ duration / time_step.
        #
        # Within each segment a _OrbitDataCollector step-handler records output
        # at fixed time_step intervals — ConstantThrustManeuver handles orbit
        # dynamics continuously (no approximation).
        # Propellant consumed during a thrust segment is computed exactly from
        # mdot × segment_duration (same formula as ElectricPropulsionSystem.update).

        h_min_m  = float(propulsion_model.h_min)
        h_max_m  = float(propulsion_model.h_max)

        # AltitudeDetector.g(state) = altitude_above_ellipsoid - threshold
        # Coast propagator: stop when altitude decreases through h_min → thruster ON
        h_min_event = (AltitudeDetector(h_min_m, earth_shape)
                       .withMaxCheck(float(time_step))
                       .withThreshold(1.0)
                       .withHandler(StopOnDecreasing()))
        # Thrust propagator: stop when altitude increases through h_max → thruster OFF
        h_max_event = (AltitudeDetector(h_max_m, earth_shape)
                       .withMaxCheck(float(time_step))
                       .withThreshold(1.0)
                       .withHandler(StopOnIncreasing()))

        propagator_coast.addEventDetector(h_min_event)
        propagator_thrust.addEventDetector(h_max_event)

        # For goal mode we also need the retrograde propagator to have an h_min event
        # (stops lowering when below goal), but goal mode manages h_min/h_max itself.
        if compensation_mode == "goal" and _goal_alt_km is not None:
            if propagator_thrust_retro is not None:
                propagator_thrust_retro.addEventDetector(h_max_event)

        _is_thrusting      = False
        _active_propagator = propagator_coast
        _seg_start_date    = start_date

        while current_state.getDate().compareTo(target_date) < 0:
            # goal mode: re-evaluate which propagator to use at every segment start
            if compensation_mode == "goal" and _goal_alt_km is not None:
                if propulsion_model.h_min > 1.0e8:
                    # Phase 1 — seeking the goal altitude
                    _is_thrusting  = propulsion_model.propellant_mass > 0
                    is_lowering    = alt_km > _goal_alt_km
                    if not _is_thrusting:
                        _active_propagator = propagator_coast
                    elif is_lowering:
                        _active_propagator = propagator_thrust_retro
                    else:
                        _active_propagator = propagator_thrust
                    # Check if goal reached
                    if (abs(alt_km - _goal_alt_km) <= _goal_offset_km or
                            (is_lowering and alt_km < _goal_alt_km) or
                            (not is_lowering and alt_km > _goal_alt_km)):
                        propulsion_model.h_min = (_goal_alt_km - _goal_offset_km) * 1000.0
                        propulsion_model.h_max = (_goal_alt_km + _goal_offset_km) * 1000.0
                        h_min_m = propulsion_model.h_min
                        h_max_m = propulsion_model.h_max
                        # Rebuild events for new thresholds
                        propagator_coast.clearEventsDetectors()
                        propagator_coast.addEventDetector(umbra_detector)
                        propagator_coast.addEventDetector(penumbra_detector)
                        propagator_coast.addEventDetector(
                            AltitudeDetector(h_min_m, earth_shape)
                            .withMaxCheck(float(time_step)).withThreshold(1.0)
                            .withHandler(StopOnDecreasing()))
                        propagator_thrust.clearEventsDetectors()
                        propagator_thrust.addEventDetector(umbra_detector)
                        propagator_thrust.addEventDetector(penumbra_detector)
                        propagator_thrust.addEventDetector(
                            AltitudeDetector(h_max_m, earth_shape)
                            .withMaxCheck(float(time_step)).withThreshold(1.0)
                            .withHandler(StopOnIncreasing()))

            # Check propellant depletion
            if propulsion_model.propellant_mass <= 0:
                _is_thrusting      = False
                _active_propagator = propagator_coast

            _seg_start_date = current_state.getDate()
            collector = _OrbitDataCollector(
                is_thrusting=_is_thrusting,
                thrust_value=float(propulsion_model.thrust),
            )
            _active_propagator.setInitialState(current_state)
            _active_propagator.getMultiplexer().clear()
            _active_propagator.getMultiplexer().add(float(time_step), collector)

            try:
                final_state = _active_propagator.propagate(target_date)
            except Exception as e:
                print(f"\n[!] Simulation stopped early: {e}")
                propulsion_model.turn_off("Simulation Exception")
                break

            # If we stopped because of an altitude event (not end of simulation),
            # account for propellant burned during this thrust segment and switch.
            seg_end_date     = final_state.getDate()
            seg_duration     = float(seg_end_date.durationFrom(_seg_start_date))
            reached_end      = seg_end_date.compareTo(target_date) >= 0

            if _is_thrusting and seg_duration > 0:
                # Exact propellant accounting: mdot × segment_duration
                g0    = 9.80665
                mdot  = float(propulsion_model.thrust) / (float(propulsion_model.isp) * g0)
                consumed = mdot * seg_duration
                if consumed > propulsion_model.propellant_mass:
                    consumed = propulsion_model.propellant_mass
                propulsion_model.propellant_used += consumed
                propulsion_model.propellant_mass -= consumed
                propulsion_model.burn_time       += seg_duration

                # Retroactively correct the last recorded entry to show final propellant state.
                if propellant_rem_list:
                    propellant_rem_list[-1] = propulsion_model.propellant_mass
                    propellant_used_list[-1] = propulsion_model.propellant_used

                # Correct state mass to reflect Python-tracked propellant
                new_mass    = max(mass - propulsion_model.propellant_used, 1.0)
                final_state = SpacecraftState(
                    final_state.getOrbit(), final_state.getAttitude(), new_mass
                )

            current_state = final_state
            alt_km        = (KeplerianOrbit(current_state.getOrbit()).getA()
                             - MEAN_EARTH_RADIUS) / 1000.0

            if not reached_end:
                # Altitude event fired — switch thrust state
                if _is_thrusting:
                    # h_max crossed → thruster OFF
                    propulsion_model.turn_off("Reached Target Altitude")
                    _is_thrusting      = False
                    _active_propagator = propagator_coast
                else:
                    # h_min crossed → thruster ON
                    if (propulsion_model.propellant_mass > 0
                            and propulsion_model.cycles < propulsion_model.max_cycles):
                        propulsion_model.is_on = True
                        propulsion_model.cycles += 1
                        propulsion_model.shutdown_reason = "None"
                        _is_thrusting      = True
                        _active_propagator = propagator_thrust

    def process_events(handler):
        events_list = []
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
                    enter_t = None
        return events_list
        
    umbra_events = process_events(umbra_handler)
    penumbra_events = process_events(penumbra_handler)

    result_dict = {
        "status":        "success",
        "model":         model_type,
        "time":          time_list,
        "altitude":      altitude_list,
        "sma":           sma_list,
        "density":       density_list,
        "inclination":   inclination_list,   
        "raan":          raan_list,
        "eccentricity":  eccentricity_list,
        "f107":          f107_list,
        "umbra_events":  umbra_events,
        "penumbra_events": penumbra_events,
    }
    
    if propulsion_model is not None:
        result_dict.update({
            "thrust_on":            thrust_on_list,
            "thrust_level":         thrust_level_list,
            "propellant_remaining": propellant_rem_list,
            "mass":                 mass_list,
            "burn_time":            burn_time_list,
            "propellant_used":      propulsion_model.propellant_used,
            "propellant_used_list": propellant_used_list,
            "duty_cycle":           propulsion_model.duty_cycle,
            "number_of_cycles":     propulsion_model.cycles,
            "cycles_list":          cycles_list,
            "shutdown_reason":      propulsion_model.shutdown_reason,
        })

    if power_model is not None:
        result_dict.update({
            "battery_Wh":    battery_Wh_list,
            "power_gen_W":   P_gen_list,
            "power_cons_W":  P_cons_list,
            "illumination":  illumination_list,
        })

    return result_dict

def plot_results(results, h_min_km=None, h_max_km=None, output_dir=None):
    """
    Figure 1a – Altitude and Solar Activity (F10.7).
    Figure 1b – Other orbital elements (Inclination, RAAN, Eccentricity).

    Parameters
    ----------
    results    : simulation result dict from run_simulation().
    h_min_km   : optional lower altitude threshold to draw [km].
    h_max_km   : optional upper altitude threshold to draw [km].
    output_dir : optional directory for PDF+PNG export; no files saved if None.
    """
    time         = [t / 86400.0 for t in results["time"]]
    altitude     = results["altitude"]
    inclination  = results.get("inclination", [])
    raan         = results.get("raan", [])
    eccentricity = results.get("eccentricity", [])

    model_key   = results.get("model", "nrlmsise00")
    m_label     = _ops.model_label(model_key)
    m_color     = _ops.MODEL_STYLES.get(model_key, {}).get("color", _ops.COLORS["primary"])

    # ── Figure 1a: Altitude & Solar Activity ──────────────────────────
    fig1, axes1 = _ops.make_figure("2x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["wide_2panel"])
    fig1.suptitle(f"Altitude and Solar Activity — {m_label}")

    axes1[0].plot(time, altitude,
                  color=m_color,
                  linewidth=_ops.LINE_WIDTHS["main"],
                  label="Mean altitude")
    _ops.add_threshold_lines(
        axes1[0],
        lower=h_min_km,
        upper=h_max_km,
        lower_label=f"$h_{{\\mathrm{{min}}}}$ = {h_min_km:.0f} km" if h_min_km is not None else None,
        upper_label=f"$h_{{\\mathrm{{max}}}}$ = {h_max_km:.0f} km" if h_max_km is not None else None,
    )
    axes1[0].set_ylabel("Altitude [km]")
    _ops.tidy_legend(axes1[0])
    _ops.apply_panel_label(axes1[0], "a")

    f107     = results.get("f107", [])
    has_f107 = any(val is not None for val in f107)

    if has_f107:
        axes1[1].plot(time, f107, **_ops.plot_kwargs("solar_flux"))
        axes1[1].set_ylabel("F10.7 Solar Flux [sfu]")
        _ops.tidy_legend(axes1[1])
    else:
        axes1[1].text(
            0.5, 0.5,
            "Solar flux data not available\n(Harris-Priester model)",
            ha="center", va="center",
            transform=axes1[1].transAxes,
            color=_ops.COLORS["gray_mid"],
        )
        axes1[1].set_ylabel("F10.7 Solar Flux [sfu]")

    _ops.format_time_axis(axes1[1], unit="days")
    _ops.apply_panel_label(axes1[1], "b")

    if output_dir:
        _ops.save_figure(fig1, f"{model_key}_altitude_solar", output_dir=output_dir)
    mplcursors.cursor(hover=True)
    plt.show()

    # ── Figure 1b: Other Orbital Elements ────────────────────────────
    fig2, axes2 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig2.suptitle(f"Orbital Elements Evolution — {m_label}")

    if inclination:
        axes2[0].plot(time, inclination,
                      color=_ops.COLORS["reference"],
                      linewidth=_ops.LINE_WIDTHS["main"])
    axes2[0].set_ylabel("Inclination [deg]")
    _ops.apply_panel_label(axes2[0], "a")

    if raan:
        axes2[1].plot(time, raan,
                      color=_ops.COLORS["validated"],
                      linewidth=_ops.LINE_WIDTHS["main"])
    axes2[1].set_ylabel("RAAN [deg]")
    _ops.apply_panel_label(axes2[1], "b")

    if eccentricity:
        axes2[2].plot(time, eccentricity,
                      color=_ops.COLORS["secondary"],
                      linewidth=_ops.LINE_WIDTHS["main"])
    axes2[2].set_ylabel("Eccentricity")
    _ops.format_time_axis(axes2[2], unit="days")
    _ops.apply_panel_label(axes2[2], "c")

    if output_dir:
        _ops.save_figure(fig2, f"{model_key}_orbital_elements", output_dir=output_dir)
    mplcursors.cursor(hover=True)
    plt.show()



def plot_drag_compensation_figures(results, ep_system=None, output_dir=None):
    """
    Additional figures for Drag Compensation Standalone simulations.

    Figure 2 – Propulsion profile: thrust, propellant consumed, thruster cycles.
    Figure 3 – Eclipse timeline | Power generated | Battery state of charge.

    Parameters
    ----------
    results    : simulation result dict from run_simulation().
    ep_system  : optional ElectricPropulsionSystem instance (not plotted directly).
    output_dir : optional directory for PDF+PNG export; no files saved if None.
    """
    time_days  = [t / 86400.0 for t in results["time"]]
    model_key  = results.get("model", "")
    m_label    = _ops.model_label(model_key)

    # ── FIGURE 2 : PROPULSION PROFILE ────────────────────────────────
    thrust_level = results.get("thrust_level", [])
    prop_used    = results.get("propellant_used_list", [])
    cycles_list  = results.get("cycles_list", [])

    fig2, axes2 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig2.suptitle(f"Propulsion Profile — {m_label}")

    if thrust_level:
        thrust_mN = [abs(t) * 1e3 for t in thrust_level]   # N → mN
        axes2[0].plot(time_days, thrust_mN,
                      color=_ops.MODEL_STYLES["thrust"]["color"],
                      linewidth=_ops.LINE_WIDTHS["secondary"],
                      label="Instantaneous thrust")
        axes2[0].set_ylabel("Thrust [mN]")
        _ops.tidy_legend(axes2[0])

        if prop_used:
            axes2[1].plot(time_days, prop_used,
                          **_ops.plot_kwargs("propellant"))
            axes2[1].set_ylabel("Propellant consumed [kg]")
            _ops.tidy_legend(axes2[1])

        if cycles_list:
            axes2[2].plot(time_days, cycles_list,
                          color=_ops.COLORS["secondary"],
                          linewidth=_ops.LINE_WIDTHS["main"],
                          label="Thruster cycles")
            axes2[2].set_ylabel("Thruster cycles")
            _ops.tidy_legend(axes2[2])

        _ops.format_time_axis(axes2[2], unit="days")
    else:
        axes2[0].text(
            0.5, 0.5, "No thrust data available",
            ha="center", va="center",
            transform=axes2[0].transAxes,
            color=_ops.COLORS["gray_mid"],
        )

    _ops.apply_panel_label(axes2[0], "a")
    _ops.apply_panel_label(axes2[1], "b")
    _ops.apply_panel_label(axes2[2], "c")

    if output_dir:
        _ops.save_figure(fig2, f"{model_key}_propulsion_profile", output_dir=output_dir)
    mplcursors.cursor(hover=True)
    plt.show()

    # ── FIGURE 3 : ECLIPSES | POWER GENERATED | BATTERY ─────────────
    illumination    = results.get("illumination", [])
    power_gen       = results.get("power_gen_W",  [])
    battery_Wh      = results.get("battery_Wh",   [])
    umbra_events    = results.get("umbra_events",    [])

    fig3, axes3 = _ops.make_figure("3x1", shared_x=True,
                                   figsize=_ops.FIGURE_SIZES["3panel"])
    fig3.suptitle(f"Eclipse and Power Analysis — {m_label}")

    # Subplot 1: Eclipse timeline ────────────────────────────────────
    ax_ecl = axes3[0]
    if illumination:
        ax_ecl.step(
            time_days, illumination,
            where="post",
            color=_ops.MODEL_STYLES["illumination"]["color"],
            linewidth=_ops.MODEL_STYLES["illumination"]["linewidth"],
            label=_ops.MODEL_STYLES["illumination"]["label"],
        )
        ax_ecl.set_yticks([0.0, 0.5, 1.0])
        ax_ecl.set_yticklabels(["Umbra\n(0 %)", "Penumbra\n(50 %)", "Full sun\n(100 %)"])
        ax_ecl.set_ylabel("Illumination")
        ax_ecl.set_ylim(-0.05, 1.15)
        _ops.tidy_legend(ax_ecl)
    elif umbra_events:
        for (mid_s, dur_s) in umbra_events:
            start_day = (mid_s - dur_s / 2) / 86400.0
            end_day   = (mid_s + dur_s / 2) / 86400.0
            ax_ecl.axvspan(start_day, end_day,
                           color=_ops.COLORS["secondary"], alpha=0.25)
        ax_ecl.set_ylabel("Eclipse (umbra)")
        ax_ecl.set_yticks([])
    _ops.apply_panel_label(ax_ecl, "a")

    # Subplot 2: Power generated ─────────────────────────────────────
    ax_pwr = axes3[1]
    if power_gen:
        ax_pwr.plot(time_days, power_gen, **_ops.plot_kwargs("power_generated"))
        ax_pwr.set_ylabel("Power generated [W]")
        _ops.tidy_legend(ax_pwr)
    else:
        ax_pwr.text(
            0.5, 0.5, "No power generation data\n(pass power_model in params)",
            ha="center", va="center",
            transform=ax_pwr.transAxes,
            color=_ops.COLORS["gray_mid"],
        )
        ax_pwr.set_ylabel("Power generated [W]")
    _ops.apply_panel_label(ax_pwr, "b")

    # Subplot 3: Battery state of charge ─────────────────────────────
    ax_bat = axes3[2]
    if battery_Wh:
        ax_bat.plot(time_days, battery_Wh, **_ops.plot_kwargs("battery_soc"))
        ax_bat.set_ylabel("Battery SoC [Wh]")
        _ops.tidy_legend(ax_bat)
    else:
        ax_bat.text(
            0.5, 0.5, "No battery data\n(pass power_model in params)",
            ha="center", va="center",
            transform=ax_bat.transAxes,
            color=_ops.COLORS["gray_mid"],
        )
        ax_bat.set_ylabel("Battery SoC [Wh]")
    _ops.format_time_axis(ax_bat, unit="days")
    _ops.apply_panel_label(ax_bat, "c")

    if output_dir:
        _ops.save_figure(fig3, f"{model_key}_eclipse_power", output_dir=output_dir)
    mplcursors.cursor(hover=True)
    plt.show()

