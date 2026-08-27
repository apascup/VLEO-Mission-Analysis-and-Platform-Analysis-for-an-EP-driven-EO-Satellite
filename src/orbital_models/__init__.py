"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          __init__.py
Description:
    Orbital mechanics, thermosphere, propulsion, and electrical power subsystem package.
===============================================================================
"""

# This file turns the 'orbital_models' folder into a Python package.
# Here models can be imported to make them easily accessible from main.py

from . import atmospheric_model
from . import electric_propulsion
