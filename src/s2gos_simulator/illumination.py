from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod


@dataclass
class Illumination(ABC):
    """Abstract base class for all illumination configurations."""
    type: str
    id: str = "illumination"

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to an eradiate-compatible dictionary."""
        return {
            "type": self.type,
            "id": self.id,
        }


@dataclass
class DirectionalIllumination(Illumination):
    """
    Directional illumination, typically used to model direct solar lighting.
    Corresponds to `eradiate.scenes.illumination.DirectionalIllumination`.
    """
    type: str = "directional"
    zenith: float = 30.0
    azimuth: float = 180.0
    irradiance: Dict[str, Any] = field(default_factory=lambda: {
        "type": "solar_irradiance",
        "dataset": "thuillier_2003"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to an eradiate-compatible dictionary."""
        d = super().to_dict()
        d.update({
            "zenith": self.zenith,
            "azimuth": self.azimuth,
            "irradiance": self.irradiance
        })
        return d


@dataclass
class AstroObjectIllumination(Illumination):
    """
    Illumination from a distant astronomical object with a defined angular size, e.g., the Sun.
    Corresponds to `eradiate.scenes.illumination.AstroObjectIllumination`.
    """
    type: str = "astro_object"
    zenith: float = 30.0
    azimuth: float = 180.0
    angular_diameter: float = 0.5358  # Default from eradiate docs for the Sun
    irradiance: Dict[str, Any] = field(default_factory=lambda: {
        "type": "solar_irradiance",
        "dataset": "thuillier_2003"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to an eradiate-compatible dictionary."""
        d = super().to_dict()
        d.update({
            "zenith": self.zenith,
            "azimuth": self.azimuth,
            "angular_diameter": self.angular_diameter,
            "irradiance": self.irradiance
        })
        return d


@dataclass
class ConstantIllumination(Illumination):
    """
    Constant, uniform illumination from all directions.
    Corresponds to `eradiate.scenes.illumination.ConstantIllumination`.
    """
    type: str = "constant"
    radiance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to an eradiate-compatible dictionary."""
        d = super().to_dict()
        d.update({"radiance": self.radiance})
        return d


@dataclass
class SpotIllumination(Illumination):
    """
    Spotlight illumination from a specific origin towards a target.
    Corresponds to `eradiate.scenes.illumination.SpotIllumination`.
    """
    type: str = "spot"
    origin: List[float] = field(default_factory=lambda: [0, 0, 100e3])
    target: List[float] = field(default_factory=lambda: [0, 0, 0])
    beam_width: float = 10.0
    intensity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to an eradiate-compatible dictionary."""
        d = super().to_dict()
        d.update({
            "origin": self.origin,
            "target": self.target,
            "beam_width": self.beam_width,
            "intensity": self.intensity,
        })
        return d