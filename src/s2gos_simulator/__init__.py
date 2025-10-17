"""S2GOS Simulator - Radiative transfer simulation components."""

# Configuration system
# Import backends with optional dependencies
from .backends import ERADIATE_AVAILABLE, EradiateSimulator
from .backends.eradiate_backend import EradiateBackend
from .config import (
    AngularFromOriginViewing,
    AngularViewing,
    ConstantIllumination,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HemisphericalViewing,
    LookAtViewing,
    MeasurementType,
    PlatformType,
    SatelliteSensor,
    SimulationConfig,
    SpectralResponse,
    UAVInstrumentType,
    UAVSensor,
)
from .hdrf_processor import HDRFProcessor

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
    "EradiateSimulator",
    "EradiateBackend",
    "ERADIATE_AVAILABLE",
    "HDRFProcessor",
]
