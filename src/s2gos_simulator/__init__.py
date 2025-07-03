"""S2GOS Simulator - Radiative transfer simulation components."""

# Legacy imports (for backward compatibility)
from .config import SimulationConfig as LegacySimulationConfig
from .sensors import (
    Sensor, PerspectiveSensor, DistantSensor, DistantMeasure, 
    MultiDistantMeasure, MultiPixelDistantMeasure, RadianceMeter, 
    MultiRadianceMeter
)
from .illumination import (
    Illumination, SpotIllumination, DirectionalIllumination,
    AstroObjectIllumination, ConstantIllumination
)
from .srf import (
    srf_delta, srf_uniform, srf_dataset,
    srf_rgb, srf_visible, srf_nir, srf_multispectral
)

# New config_v2 system (recommended)
from .config_v2 import (
    SimulationConfig,
    SatelliteSensor, UAVSensor, GroundSensor,
    DirectionalIllumination as DirectionalIlluminationV2,
    ConstantIllumination as ConstantIlluminationV2,
    AngularViewing, AngularFromOriginViewing, LookAtViewing, HemisphericalViewing,
    SpectralResponse, PlatformType, MeasurementType,
    UAVInstrumentType, GroundInstrumentType
)

# Import backends with optional dependencies
from .backends import EradiateSimulator, ERADIATE_AVAILABLE
from .backends.eradiate_backend_v2 import EradiateBackendV2

__version__ = "0.1.0"

__all__ = [
    # New Configuration System (Recommended)
    "SimulationConfig",
    "SatelliteSensor",
    "UAVSensor", 
    "GroundSensor",
    "DirectionalIlluminationV2",
    "ConstantIlluminationV2",
    "AngularViewing",
    "AngularFromOriginViewing",
    "LookAtViewing", 
    "HemisphericalViewing",
    "SpectralResponse",
    "PlatformType",
    "MeasurementType",
    "UAVInstrumentType",
    "GroundInstrumentType",
    
    # Legacy Configuration (Backward Compatibility)
    "LegacySimulationConfig",
    "Sensor",
    "PerspectiveSensor", 
    "DistantSensor",
    "DistantMeasure",
    "MultiDistantMeasure", 
    "MultiPixelDistantMeasure",
    "RadianceMeter",
    "MultiRadianceMeter",
    
    # Legacy Illumination
    "Illumination", 
    "SpotIllumination", 
    "DirectionalIllumination",
    "AstroObjectIllumination", 
    "ConstantIllumination",
    
    # Spectral Response Functions (Legacy)
    "srf_delta",
    "srf_uniform", 
    "srf_dataset",
    "srf_rgb",
    "srf_visible",
    "srf_nir",
    "srf_multispectral",
    
    # Backends
    "EradiateSimulator",
    "EradiateBackendV2",
    "ERADIATE_AVAILABLE"
]