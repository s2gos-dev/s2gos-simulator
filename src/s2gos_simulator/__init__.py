"""S2GOS Simulator - Radiative transfer simulation components."""

# Configuration system
from .config import (
    SimulationConfig,
    SatelliteSensor, UAVSensor, GroundSensor,
    DirectionalIllumination, ConstantIllumination,
    AngularViewing, AngularFromOriginViewing, LookAtViewing, HemisphericalViewing,
    SpectralResponse, PlatformType, MeasurementType,
    UAVInstrumentType, GroundInstrumentType
)

# Import backends with optional dependencies
from .backends import EradiateSimulator, ERADIATE_AVAILABLE
from .backends.eradiate_backend import EradiateBackend

__version__ = "0.1.0"

__all__ = [
    # Configuration System
    "SimulationConfig",
    "SatelliteSensor",
    "UAVSensor", 
    "GroundSensor",
    "DirectionalIllumination",
    "ConstantIllumination",
    "AngularViewing",
    "AngularFromOriginViewing",
    "LookAtViewing", 
    "HemisphericalViewing",
    "SpectralResponse",
    "PlatformType",
    "MeasurementType",
    "UAVInstrumentType",
    "GroundInstrumentType",
    
    # Backends
    "EradiateSimulator",
    "EradiateBackend",
    "ERADIATE_AVAILABLE"
]