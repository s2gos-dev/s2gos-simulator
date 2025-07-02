from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

import xarray as xr


class SimulationBackend(ABC):
    """Abstract base class for radiative transfer simulation backends.
    
    Each backend is responsible for:
    - Converting scene configuration to backend-specific format
    - Handling backend-specific surface representations
    - Executing the simulation
    - Processing and saving results
    """
    
    def __init__(self, render_config):
        """Initialize backend with render configuration.
        
        Args:
            render_config: Rendering configuration (sensors, illumination)
        """
        self.render_config = render_config
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available.
        
        Returns:
            True if backend can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def run_simulation(self, scene_config, scene_dir: Path, 
                      output_dir: Optional[Path] = None, plot_image: bool=False,
                      id_to_plot: str="rgb_camera") -> xr.Dataset:
        """Run complete simulation pipeline.
        
        This is the main entry point for running simulations. Implementations
        should handle all aspects of the simulation internally:
        - Scene setup and surface creation
        - Backend-specific configuration
        - Simulation execution
        - Result processing and saving
        
        Args:
            scene_config: Scene configuration from s2gos_generator
            scene_dir: Directory containing scene assets (meshes, textures)
            output_dir: Directory for simulation outputs (optional)
            
        Returns:
            xarray.Dataset containing simulation results. If output_dir is provided,
            the dataset is also saved to a NetCDF file and the file path is stored
            in the dataset's 'saved_to' attribute.
        """
        pass
    
    def validate_scene(self, scene_config, scene_dir: Path) -> List[str]:
        """Validate scene configuration and assets.
        
        Optional method for backends to check scene validity before simulation.
        Default implementation returns no errors.
        
        Args:
            scene_config: Scene configuration to validate
            scene_dir: Directory containing scene assets
            
        Returns:
            List of validation error messages (empty if valid)
        """
        return []
    
    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.name}(available={self.is_available()})"