from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from datetime import datetime

from .sensors import Sensor, PerspectiveSensor, DistantSensor
from .illumination import Illumination, DirectionalIllumination


@dataclass 
class SimulationConfig:
    """Rendering configuration for radiative transfer simulations.
    
    Contains only rendering-specific parameters: sensors and illumination.
    Atmosphere is now part of SceneConfig since it's tied to geographic location.
    
    Attributes:
        illumination: Light source configuration  
        sensors: List of sensor/camera configurations
        mode: Eradiate mode ('mono', 'ckd', 'mono_polarized', 'ckd_polarized')
        name: Optional identifier for this render configuration
    """
    
    illumination: Illumination = field(default_factory=DirectionalIllumination()) 
    sensors: List[Sensor] = field(default_factory=list)
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "illumination": self.illumination.to_dict(),
            "measures": [s.to_dict() for s in self.sensors]
        }
        
        if self.name:
            result["name"] = self.name
            
        return result
    
    def save_yaml(self, path: Path, include_comments: bool = True):
        """Save render configuration to YAML file.
        
        Args:
            path: Output file path
            include_comments: Whether to include header comments
        """
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if include_comments:
                f.write("# S2GOS Render Configuration (Sensors + Illumination)\n")
                f.write("# Atmosphere is part of scene configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
            
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, width=120)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SimulationConfig':
        """Load render configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            SimulationConfig instance loaded from file
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse illumination
        illum_data = data.get("illumination", {})
        illumination = Illumination(
            zenith=illum_data.get("zenith", 30.0),
            azimuth=illum_data.get("azimuth", 180.0),
            irradiance=illum_data.get("irradiance", {}).get("type", "solar_irradiance"),
            dataset=illum_data.get("irradiance", {}).get("dataset", "thuillier_2003")
        )
        
        # Parse sensors (simplified)
        sensors = []
        for sensor_data in data.get("measures", []):
            if sensor_data["type"] == "perspective":
                sensors.append(PerspectiveSensor(
                    id=sensor_data.get("id", "perspective"),
                    origin=sensor_data.get("origin", [0, 0, 98612.90342332]),
                    target=sensor_data.get("target", [0, 0, 0]),
                    fov=sensor_data.get("fov", 70),
                    resolution=sensor_data.get("film_resolution", [1024, 768]),
                    spp=sensor_data.get("spp", 128)
                ))
            # Add other sensor types as needed
        
        return cls(
            name=data.get("name"),
            illumination=illumination,
            sensors=sensors
        )
    