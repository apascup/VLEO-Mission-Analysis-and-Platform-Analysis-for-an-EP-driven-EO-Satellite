"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          power_subsystem.py
Description:
    Electrical power subsystem model with solar array generation, battery charge/discharge balance, and annual degradation.
===============================================================================
"""

class PowerSubsystem:
    """
    Electrical power subsystem with solar generation, battery storage,
    and thruster power-inhibit logic.

    Parameters
    ----------
    solar_panel_area_m2 : float
        Total solar panel area [m²].
    panel_efficiency : float
        Solar cell efficiency as a fraction (0–1).  Use beginning-of-life
        (BOL) or end-of-life (EOL) value as appropriate.
    solar_flux_W_m2 : float
        Mean solar irradiance at 1 AU [W/m²]. Typical value: 1361 W/m².
    battery_capacity_Wh : float
        Maximum usable battery energy [Wh].
    battery_initial_Wh : float
        Battery state of charge at t = 0 [Wh].
    housekeeping_power_W : float
        Constant platform power draw excluding propulsion [W].
        Covers OBDH, ADCS, comms, thermal control, etc.
    thruster_power_W : float
        Electrical power drawn by the thruster when firing [W].
        This is an explicit input parameter (not derived from thrust/Isp).
    """

    def __init__(
        self,
        solar_panel_area_m2,
        panel_efficiency,
        solar_flux_W_m2,
        battery_capacity_Wh,
        battery_initial_Wh,
        housekeeping_power_W,
        thruster_power_W,
        panel_degradation_yr=0.0,
        battery_degradation_yr=0.0,
    ):
        self.solar_panel_area_m2 = float(solar_panel_area_m2)
        self.panel_efficiency_bol = float(panel_efficiency)
        self.solar_flux_W_m2 = float(solar_flux_W_m2)
        self.panel_degradation_yr = float(panel_degradation_yr)

        # Maximum solar generation in full sunlight [W] (at BOL)
        self.P_solar_max = (
            self.solar_panel_area_m2
            * self.panel_efficiency_bol
            * self.solar_flux_W_m2
        )

        # Battery
        self.battery_capacity_bol_Wh = float(battery_capacity_Wh)
        self.battery_degradation_yr = float(battery_degradation_yr)

        self.battery_capacity_Wh = self.battery_capacity_bol_Wh
        self.battery_Wh          = min(float(battery_initial_Wh), self.battery_capacity_Wh)

        # Power loads
        self.housekeeping_power_W = float(housekeeping_power_W)
        self.thruster_power_W     = float(thruster_power_W)

        # Track elapsed time for degradation
        self.elapsed_time_s = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, illumination_fraction, thruster_requesting, dt_s):
        """
        Advance the power subsystem by one time step.

        Parameters
        ----------
        illumination_fraction : float
            Solar illumination level for this step:
              1.0 = full sun | 0.5 = penumbra | 0.0 = umbra.
        thruster_requesting : bool
            True if the propulsion controller wants to fire this step.
        dt_s : float
            Duration of the time step [s].

        Returns
        -------
        thruster_allowed : bool
            False if the battery is empty and solar cannot cover the
            combined housekeeping + thruster load.
        P_gen : float
            Solar power generated this step [W].
        P_cons : float
            Total power consumed this step [W]
            (housekeeping + thruster if allowed).
        """
        # Apply degradation based on elapsed time
        self.elapsed_time_s += float(dt_s)
        elapsed_years = self.elapsed_time_s / (365.25 * 86400.0)

        # Update solar panel efficiency and max power
        current_efficiency = self.panel_efficiency_bol * (1.0 - self.panel_degradation_yr * elapsed_years)
        current_efficiency = max(current_efficiency, 0.0)
        self.P_solar_max = self.solar_panel_area_m2 * current_efficiency * self.solar_flux_W_m2

        # Update battery capacity
        self.battery_capacity_Wh = self.battery_capacity_bol_Wh * (1.0 - self.battery_degradation_yr * elapsed_years)
        self.battery_capacity_Wh = max(self.battery_capacity_Wh, 1.0) # clamp to at least 1 Wh

        # Ensure battery charge doesn't exceed current capacity
        self.battery_Wh = min(self.battery_Wh, self.battery_capacity_Wh)

        # Solar generation scaled by illumination
        P_gen = self.P_solar_max * float(illumination_fraction)

        # Base load (always on)
        P_cons_base = self.housekeeping_power_W

        # Decide if thruster can fire
        if thruster_requesting:
            P_cons_full = P_cons_base + self.thruster_power_W
            net_with_thruster = P_gen - P_cons_full

            # Block thruster only when battery is empty AND net power is negative
            if self.battery_Wh <= 0.0 and net_with_thruster < 0.0:
                thruster_allowed = False
                P_cons = P_cons_base
            else:
                thruster_allowed = True
                P_cons = P_cons_full
        else:
            thruster_allowed = True   # not requesting → no issue
            P_cons = P_cons_base

        # Update battery state
        net_power_W = P_gen - P_cons
        dE_Wh       = net_power_W * float(dt_s) / 3600.0
        self.battery_Wh = max(
            0.0,
            min(self.battery_capacity_Wh, self.battery_Wh + dE_Wh)
        )

        return thruster_allowed, P_gen, P_cons

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def P_solar_max_W(self):
        """Maximum solar generation in full sun [W]."""
        return self.P_solar_max

    @property
    def state_of_charge(self):
        """Battery state of charge as a fraction (0–1)."""
        if self.battery_capacity_Wh > 0:
            return self.battery_Wh / self.battery_capacity_Wh
        return 0.0


# ================================================================
# FACTORY  –  build a PowerSubsystem from mission_config parameters
# ================================================================

def build_power_model(duration_days: float):
    """
    Construct and return a :class:`PowerSubsystem` instance configured
    from the ``POWER`` and ``PROPULSION`` dicts in ``mission_config.py``.

    Degradation is modelled linearly day-by-day inside the simulation.
    The values returned for logging (eff_mean, bat_cap_eom) reflect the 
    expected average and end-of-mission states for reference only.

    Set ``solar_panel_degradation_per_year`` or
    ``battery_degradation_per_year`` to ``0.0`` in ``POWER`` to disable
    degradation for that component.

    Parameters
    ----------
    duration_days : float
        Total mission duration in days (used to scale degradation).

    Returns
    -------
    power_model : PowerSubsystem
        Ready-to-use power subsystem instance.
    info : tuple[float, float, float]
        ``(eff_mean, bat_cap_eom, thruster_power_W)`` – the effective
        values used, convenient for terminal logging in ``main.py``.
    """
    # Import here to avoid a circular dependency at module load time
    from mission_config import POWER, PROPULSION

    duration_years = duration_days / 365.25

    # ── Solar panel efficiency (BOL) ──────────────────────────────────────────
    eff_bol  = float(POWER["panel_efficiency"])
    deg_rate = float(POWER.get("solar_panel_degradation_per_year", 0.0))

    # ── Battery capacity (BOL) ────────────────────────────────────────────────
    bat_cap_bol = float(POWER["battery_capacity_Wh"])
    bat_deg     = float(POWER.get("battery_degradation_per_year", 0.0))
    bat_init    = min(float(POWER["battery_initial_Wh"]), bat_cap_bol)

    # ── Thruster electrical power draw ────────────────────────────────────────
    thruster_power_W = float(PROPULSION.get("power_W", 0.0))

    power_model = PowerSubsystem(
        solar_panel_area_m2  = float(POWER["solar_panel_area_m2"]),
        panel_efficiency     = eff_bol,
        solar_flux_W_m2      = float(POWER["solar_flux_W_m2"]),
        battery_capacity_Wh  = bat_cap_bol,
        battery_initial_Wh   = bat_init,
        housekeeping_power_W = float(POWER["housekeeping_power_W"]),
        thruster_power_W     = thruster_power_W,
        panel_degradation_yr = deg_rate,
        battery_degradation_yr = bat_deg,
    )

    # Compute expected mean efficiency and EOL capacity for summary logs
    eff_mean = eff_bol * (1.0 - deg_rate * duration_years / 2.0)
    eff_mean = max(eff_mean, 0.0)
    
    bat_cap_eom = bat_cap_bol * (1.0 - bat_deg * duration_years)
    bat_cap_eom = max(bat_cap_eom, 1.0)

    return power_model, (eff_mean, bat_cap_eom, thruster_power_W)

