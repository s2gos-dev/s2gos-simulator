"""S2GOS Simulator - Radiative transfer simulation components."""

from .config import SimulationConfig, RenderConfig, create_default_render_config
from .sensors import Sensor, PerspectiveSensor, DistantSensor
from .illumination import IlluminationConfig, create_solar_illumination, create_constant_illumination

# Import backends with optional dependencies
from .backends import EradiateSimulator, ERADIATE_AVAILABLE

__version__ = "0.0.1"

__all__ = [
    "SimulationConfig",
    "RenderConfig", 
    "create_default_render_config",
    "Sensor",
    "PerspectiveSensor", 
    "DistantSensor",
    "IlluminationConfig",
    "create_solar_illumination",
    "create_constant_illumination",
    "EradiateSimulator",
    "ERADIATE_AVAILABLE"
]