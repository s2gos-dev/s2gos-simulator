from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List


@dataclass
class SimulationResult:
    """Standard result format for all simulation backends.
    
    Attributes:
        success: Whether the simulation completed successfully
        raw_results: Path to raw simulation output file (e.g., NetCDF)
        rgb_image: Path to RGB visualization image (if created)
        output_dir: Directory containing all simulation outputs
        error: Error message if simulation failed
        metadata: Additional backend-specific metadata
    """
    success: bool
    raw_results: Optional[Path] = None
    rgb_image: Optional[Path] = None  
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                      output_dir: Optional[Path] = None) -> SimulationResult:
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
            SimulationResult with paths to outputs and status information
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