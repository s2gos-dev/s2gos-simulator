"""S2GOS Simulator - Radiative transfer simulation components."""

# Configuration system
# Import backends with optional dependencies
from .backends import ERADIATE_AVAILABLE, EradiateSimulator
from .backends.eradiate.backend import EradiateBackend
from .config import (
    AngularFromOriginViewing,
    AngularViewing,
    BHRConfig,
    BRFConfig,
    ConstantIllumination,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HCRFPostProcessingConfig,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    PlatformType,
    RadianceConfig,
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
    "HemisphericalMeasurementLocation",
    "SpectralResponse",
    "PlatformType",
    "UAVInstrumentType",
    "GroundInstrumentType",
    # Measurement Configs
    "IrradianceConfig",
    "HDRFConfig",
    "HCRFConfig",
    "HCRFPostProcessingConfig",
    "BRFConfig",
    "RadianceConfig",
    "BHRConfig",
    # Backends and Processors
    "EradiateSimulator",
    "EradiateBackend",
    "ERADIATE_AVAILABLE",
    "HDRFProcessor",
]
