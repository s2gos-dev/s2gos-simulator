"""S2GOS Simulator - Radiative transfer simulation components."""

# Configuration system
# Import backends with optional dependencies
from .backends import ERADIATE_AVAILABLE, EradiateSimulator
from .backends.eradiate.backend import EradiateBackend
from .bhr_processor import BHRProcessor
from .config import (
    CHIME_SPECTRAL_CONFIG,
    CHIME_SPECTRAL_REGIONS,
    AngularFromOriginViewing,
    AngularViewing,
    BHRConfig,
    BRFConfig,
    ConstantIllumination,
    DirectionalIllumination,
    DistantViewing,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HCRFPostProcessingConfig,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    PixelBHRConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
    PlatformType,
    RadianceConfig,
    RectangleTarget,
    SatelliteSensor,
    SimulationConfig,
    SpectralRegion,
    SpectralResponse,
    UAVInstrumentType,
    UAVSensor,
    WavelengthGrid,
    create_chime_sensor,
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
    "SpectralRegion",
    "WavelengthGrid",
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
    "PixelBHRConfig",
    "PixelBRFConfig",
    "PixelHDRFConfig",
    # CHIME Hyperspectral Support
    "CHIME_SPECTRAL_REGIONS",
    "CHIME_SPECTRAL_CONFIG",
    "SpectralRegion",
    "WavelengthGrid",
    "create_chime_sensor",
    # Viewing and Target Types
    "DistantViewing",
    "RectangleTarget",
    # Backends and Processors
    "EradiateSimulator",
    "EradiateBackend",
    "ERADIATE_AVAILABLE",
    "BHRProcessor",
    "HDRFProcessor",
]
