from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

from .srf import srf_rgb


@dataclass
class Sensor(ABC):
    """Base sensor configuration."""
    type: str
    id: str = "sensor"
    srf: Optional[Union[Dict[str, Any], str]] = None
    spp: int = 128
    
    def __post_init__(self):
        if self.srf is None:
            self.srf = srf_rgb()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "srf": self.srf,
            "spp": self.spp,
            **self._specific_params()
        }
    
    @abstractmethod
    def _specific_params(self) -> Dict[str, Any]:
        """Sensor-specific parameters."""
        pass


@dataclass
class PerspectiveSensor(Sensor):
    """Perspective camera sensor for scene inspection."""
    type: str = "perspective"
    origin: List[float] = field(default_factory=lambda: [0, 0, 36612.90342332])
    target: List[float] = field(default_factory=lambda: [0, 0, 0])
    up: List[float] = field(default_factory=lambda: [0, 1, 0])
    fov: float = 70.0
    resolution: List[int] = field(default_factory=lambda: [1024, 768])
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "target": self.target, 
            "up": self.up,
            "fov": self.fov,
            "film_resolution": self.resolution
        }


@dataclass
class DistantMeasure(Sensor):
    """
    Single-pixel distant measure for a single viewing direction.
    This class acts as a wrapper for eradiate.scenes.measure.DistantMeasure,
    supporting its different construction methods.
    
    By default, it creates a sensor pointing nadir (zenith=0). Use the class
    methods `from_angles` or `from_direction` for explicit configuration.
    """
    type: str = "distant"

    # Stores constructor info. Can be None for direct construction.
    construct_method: Optional[str] = "from_angles"
    # Stores the parameters for construction.
    construct_params: Dict[str, Any] = field(default_factory=lambda: {
        "angles": [0.0, 0.0]
    })

    def _specific_params(self) -> Dict[str, Any]:
        """Returns the parameters required by the eradiate scene dictionary."""
        if self.construct_method:
            # Case 1: Use a class method constructor (e.g., from_angles)
            return {
                "construct": self.construct_method,
                **self.construct_params
            }
        else:
            # Case 2: Direct construction (e.g., from_direction)
            return self.construct_params

    @classmethod
    def from_angles(
        cls,
        zenith: float,
        azimuth: float,
        target: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "DistantMeasure":
        """
        Create a distant measure from a (zenith, azimuth) pair.

        Args:
            zenith (float): Zenith angle.
            azimuth (float): Azimuth angle.
            target (dict, optional): Target specification dictionary.
                Example: `{"type": "point", "position": [0, 0, 0]}`.
            **kwargs: Additional sensor parameters (id, spp, srf, ray_offset).

        Returns:
            DistantMeasure: An instance configured from the given angles.
        """
        params = {"angles": [zenith, azimuth]}
        if target:
            params["target"] = target
        
        # Pass remaining eradiate params if they exist in kwargs
        if "ray_offset" in kwargs:
            params["ray_offset"] = kwargs.pop("ray_offset")

        return cls(
            construct_method="from_angles",
            construct_params=params,
            **kwargs
        )

    @classmethod
    def from_direction(
        cls,
        direction: List[float],
        target: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "DistantMeasure":
        """
        Create a distant measure from a direction vector. This uses the
        direct constructor of the underlying eradiate class.

        Args:
            direction (List[float]): A 3-element direction vector [x, y, z].
            target (dict, optional): Target specification dictionary.
                Example: `{"type": "point", "position": [0, 0, 0]}`.
            **kwargs: Additional sensor parameters (id, spp, srf, ray_offset).

        Returns:
            DistantMeasure: An instance configured from the given direction.
        """
        params = {"direction": direction}
        if target:
            params["target"] = target

        if "ray_offset" in kwargs:
            params["ray_offset"] = kwargs.pop("ray_offset")

        return cls(
            construct_method=None,  # Indicates direct construction
            construct_params=params,
            **kwargs
        )


@dataclass
class MultiDistantMeasure(Sensor):
    """
    Multi-distant measurement sensor for multiple viewing angles.
    This class acts as a wrapper for eradiate.scenes.measure.MultiDistantMeasure,
    supporting its various construction methods. Use the class methods
    `hplane`, `grid`, `aring`, etc., to create specific configurations.
    """
    type: str = "mdistant"

    construct_method: str = "hplane"
    construct_params: Dict[str, Any] = field(default_factory=lambda: {
        "zeniths": [0.0, 10.0, 20.0, 30.0], "azimuth": 0.0
    })

    def _specific_params(self) -> Dict[str, Any]:
        """Returns the parameters required by the eradiate scene dictionary."""
        return {
            "construct": self.construct_method,
            **self.construct_params
        }

    @classmethod
    def hplane(cls, zeniths: List[float], azimuth: float = 0.0, **kwargs) -> "MultiDistantMeasure":
        """
        Create a hemispherical plane measurement configuration.

        Args:
            zeniths (List[float]): List of zenith angles.
            azimuth (float): Single azimuth angle for the plane.
            **kwargs: Additional sensor parameters (id, spp, srf).

        Returns:
            MultiDistantMeasure: An instance configured for hplane.
        """
        return cls(
            construct_method="hplane",
            construct_params={"zeniths": zeniths, "azimuth": azimuth},
            **kwargs
        )

    @classmethod
    def grid(cls, zeniths: List[float], azimuths: List[float], **kwargs) -> "MultiDistantMeasure":
        """
        Create a gridded (zenith x azimuth) measurement configuration.

        Args:
            zeniths (List[float]): List of zenith angles.
            azimuths (List[float]): List of azimuth angles.
            **kwargs: Additional sensor parameters (id, spp, srf).

        Returns:
            MultiDistantMeasure: An instance configured for a grid layout.
        """
        return cls(
            construct_method="grid",
            construct_params={"zeniths": zeniths, "azimuths": azimuths},
            **kwargs
        )

    @classmethod
    def aring(cls, zenith: float, azimuths: List[float], **kwargs) -> "MultiDistantMeasure":
        """
        Create an azimuth ring measurement configuration.

        Args:
            zenith (float): The single zenith angle for the ring.
            azimuths (List[float]): List of azimuth angles.
            **kwargs: Additional sensor parameters (id, spp, srf).

        Returns:
            MultiDistantMeasure: An instance configured for an azimuth ring.
        """
        return cls(
            construct_method="aring",
            construct_params={"zenith": zenith, "azimuths": azimuths},
            **kwargs
        )

    @classmethod
    def from_angles(cls, angles: List[Tuple[float, float]], **kwargs) -> "MultiDistantMeasure":
        """
        Create a measurement from explicit (zenith, azimuth) pairs.

        Args:
            angles (List[Tuple[float, float]]): A list of (zenith, azimuth) pairs.
            **kwargs: Additional sensor parameters (id, spp, srf).

        Returns:
            MultiDistantMeasure: An instance configured from explicit angles.
        """
        return cls(
            construct_method="from_angles",
            construct_params={"angles": angles},
            **kwargs
        )

    @classmethod
    def from_directions(cls, directions: List[List[float]], **kwargs) -> "MultiDistantMeasure":
        """
        Create a measurement from explicit direction vectors.

        Args:
            directions (List[List[float]]): A list of [x, y, z] direction vectors.
            **kwargs: Additional sensor parameters (id, spp, srf).

        Returns:
            MultiDistantMeasure: An instance configured from explicit directions.
        """
        return cls(
            construct_method="from_directions",
            construct_params={"directions": directions},
            **kwargs
        )


@dataclass
class MultiPixelDistantMeasure(Sensor):
    """Multi-pixel distant measurement for custom pixel arrangements."""
    type: str = "mpdistant"
    pixels: List[Tuple[float, float]] = field(default_factory=list)  # (zenith, azimuth) pairs
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "pixels": self.pixels
        }


@dataclass
class RadianceMeter(Sensor):
    """Radiance meter sensor for point measurements."""
    type: str = "radiancemeter"
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    direction: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "direction": self.direction
        }


@dataclass
class MultiRadianceMeter(Sensor):
    """Multiple radiance meter sensor."""
    type: str = "mradiancemeter"
    origins: List[List[float]] = field(default_factory=list)
    directions: List[List[float]] = field(default_factory=list)
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "origins": self.origins,
            "directions": self.directions
        }


@dataclass  
class DistantSensor(Sensor):
    """Legacy distant sensor - deprecated, use DistantMeasure instead."""
    type: str = "mpdistant"
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0])  # zenith, azimuth
    target_size: float = 50.0e3  # 50km square
    resolution: List[int] = field(default_factory=lambda: [256, 256])
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "construct": "from_angles",
            "angles": self.angles,
            "target": {
                "type": "rectangle",
                "xmin": -self.target_size, "xmax": self.target_size,
                "ymin": -self.target_size, "ymax": self.target_size
            },
            "film_resolution": self.resolution
        }


