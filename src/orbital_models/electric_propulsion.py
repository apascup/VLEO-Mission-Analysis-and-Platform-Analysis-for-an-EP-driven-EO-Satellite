"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          electric_propulsion.py
Description:
    Electric propulsion system model with throttle curves, propellant consumption, duty cycling, and cycle limits.
===============================================================================
"""

import math

class ElectricPropulsionSystem:
    """
    Simulates an electric propulsion system for drag compensation.
    Uses a simple ON/OFF duty cycle based on altitude thresholds.
    """
    def __init__(self, thrust, isp, initial_propellant_mass, 
                 h_min, h_max, max_burn_time=None, max_cycles=None):
        self.thrust = float(thrust)                      # [N]
        self.isp = float(isp)                            # [s]
        self.propellant_mass = float(initial_propellant_mass) # [kg]
        self.h_min = float(h_min)                        # [m]
        self.h_max = float(h_max)                        # [m]
        
        self.max_burn_time = float(max_burn_time) if max_burn_time is not None else float('inf')
        self.max_cycles = int(max_cycles) if max_cycles is not None else float('inf')
        
        # State variables
        self.is_on = False
        self.burn_time = 0.0                             # [s]
        self.propellant_used = 0.0                       # [kg]
        self.cycles = 0
        self.shutdown_reason = "None"
        
        # Earth gravity constant for mass flow calculations
        self.g0 = 9.80665

    def compute_mass_flow_rate(self):
        """Returns the mass flow rate [kg/s] when the thruster is ON."""
        return self.thrust / (self.isp * self.g0)

    def update(self, current_altitude_km, dt_sec, power_ok=True):
        """
        Evaluate if the thruster should be ON or OFF, and consume propellant.
        Updates internal state.
        
        Parameters
        ----------
        current_altitude_km : float
            Current spacecraft altitude [km].
        dt_sec : float
            Time step duration [s].
        power_ok : bool, optional
            If False (set by the power subsystem), the thruster is forced OFF
            regardless of altitude thresholds.  No propellant is consumed.
            Default is True (no power constraint).

        Returns
        -------
        bool
            True if the thruster is ON, False otherwise.
        """
        current_altitude_m = current_altitude_km * 1000.0

        # 0. Power override: power subsystem blocks thrust
        if not power_ok:
            if self.is_on:
                self.turn_off("Power Depleted")
            return False

        # 1. Check depletion or failure conditions
        if self.propellant_mass <= 0:
            self.turn_off("Propellant Depleted")
            return False
            
        if self.burn_time >= self.max_burn_time:
            self.turn_off("Max Burn Time Reached")
            return False
            
        if self.cycles >= self.max_cycles and not self.is_on:
            # If it's already ON, let it finish the cycle. But can't start a new one.
            self.turn_off("Max Cycles Reached")
            return False

        # 2. Logic for turning ON / OFF based on altitude
        if current_altitude_m < self.h_min and not self.is_on:
            if self.cycles < self.max_cycles:
                self.is_on = True
                self.cycles += 1
                self.shutdown_reason = "None"
                
        elif current_altitude_m > self.h_max and self.is_on:
            self.turn_off("Reached Target Altitude")
            
        # 3. If ON, consume propellant
        if self.is_on:
            mdot = self.compute_mass_flow_rate()
            consumed = mdot * dt_sec
            
            if consumed > self.propellant_mass:
                # Can only burn for a fraction of dt
                fraction = self.propellant_mass / consumed
                self.propellant_used += self.propellant_mass
                self.burn_time += dt_sec * fraction
                self.propellant_mass = 0.0
                self.turn_off("Propellant Depleted")
            else:
                self.propellant_used += consumed
                self.propellant_mass -= consumed
                self.burn_time += dt_sec

        return self.is_on

    def turn_off(self, reason):
        """Helper to turn off the thruster."""
        if self.is_on:
            self.is_on = False
            self.shutdown_reason = reason

    @property
    def duty_cycle(self):
        """Returns the duty cycle percentage (if we knew total time)."""
        # We can't return a true duty cycle without total mission time,
        # but returning burn_time is useful.
        return self.burn_time
