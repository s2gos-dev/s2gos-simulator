"""S2GOS Simulator - Radiative transfer simulation components."""

from .config import SimulationConfig
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

# Import backends with optional dependencies
from .backends import EradiateSimulator, ERADIATE_AVAILABLE

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "SimulationConfig",
    
    # Sensors
    "Sensor",
    "PerspectiveSensor", 
    "DistantSensor",
    "DistantMeasure",
    "MultiDistantMeasure", 
    "MultiPixelDistantMeasure",
    "RadianceMeter",
    "MultiRadianceMeter",
    
    # Illumination
    "Illumination",
    "Illumination", 
    "SpotIllumination", 
    "DirectionalIllumination",
    "AstroObjectIllumination", 
    "ConstantIllumination"
    
    # Spectral Response Functions
    "srf_delta",
    "srf_uniform", 
    "srf_dataset",
    "srf_rgb",
    "srf_visible",
    "srf_nir",
    "srf_multispectral",
    
    # Backends
    "EradiateSimulator",
    "ERADIATE_AVAILABLE"
]