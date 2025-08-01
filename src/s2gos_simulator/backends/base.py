from abc import ABC, abstractmethod
from upath import UPath
from typing import Any, Dict, List, Optional

import xarray as xr
from s2gos_utils.typing import PathLike


class SimulationBackend(ABC):
    """Abstract base class for radiative transfer simulation backends.

    This class provides a generic interface for different radiative transfer models.
    Each backend handles:
    - Configuration translation from generic to backend-specific format
    - Scene setup and surface creation
    - Simulation execution
    - Result processing and standardization
    """

    def __init__(self, simulation_config):
        """Initialize backend with simulation configuration.

        Args:
            simulation_config: Complete simulation configuration (sensors, illumination, etc.)
        """
        self.simulation_config = simulation_config

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def run_simulation(
        self, scene_config, scene_dir: PathLike, output_dir: Optional[PathLike] = None, **kwargs
    ) -> xr.Dataset:
        """Run complete simulation pipeline.

        This is the main entry point for running simulations. Implementations
        should handle all aspects of the simulation internally:
        - Translation of generic config to backend-specific format
        - Scene setup and surface creation
        - Backend-specific configuration
        - Simulation execution
        - Result processing and standardization

        Args:
            scene_config: Scene configuration from s2gos_generator
            scene_dir: Directory containing scene assets (meshes, textures)
            output_dir: Directory for simulation outputs (optional)
            **kwargs: Additional backend-specific args

        Returns:
            xarray.Dataset containing simulation results following S2GOS standards
        """
        pass

    def validate_configuration(self) -> List[str]:
        """Validate simulation configuration for this backend.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Generic validation - can be overridden by specific backends
        if not self.simulation_config.sensors and not getattr(
            self.simulation_config, "radiative_quantities", []
        ):
            errors.append("No sensors or radiative quantities defined in configuration")

        # Check if backend supports all requested measurement types
        unsupported_measurements = self._get_unsupported_measurements()
        if unsupported_measurements:
            errors.append(f"Unsupported measurement types: {unsupported_measurements}")

        return errors

    def validate_scene(self, scene_config, scene_dir: PathLike) -> List[str]:
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

    def _get_unsupported_measurements(self) -> List[str]:
        """Get list of unsupported measurement types for this backend.

        Override in specific backends to define supported measurements.

        Returns:
            List of unsupported measurement type names
        """
        return []

    def _translate_illumination(self) -> Dict[str, Any]:
        """Translate generic illumination config to backend-specific format.

        Returns:
            Backend-specific illumination configuration
        """
        # Default implementation - override in specific backends
        return self.simulation_config.illumination.model_dump()

    def _translate_sensors(self) -> List[Dict[str, Any]]:
        """Translate generic sensor configs and radiative quantities to backend-specific format.

        Note: This method now handles both sensors and radiative quantities.
        Override in specific backends to provide proper translation.

        Returns:
            List of backend-specific measurement configurations
        """
        # Default implementation - override in specific backends
        measures = [sensor.model_dump() for sensor in self.simulation_config.sensors]

        # Add basic radiative quantity handling if present
        if hasattr(self.simulation_config, "radiative_quantities"):
            for rq in self.simulation_config.radiative_quantities:
                measures.append(
                    {
                        "id": f"{rq.quantity.value}_measure",
                        "type": "placeholder",
                        "radiative_quantity": rq.quantity.value,
                        "TODO": "Implement in specific backend",
                    }
                )

        return measures

    def _create_output_metadata(
        self, output_dir: Optional[PathLike] = None
    ) -> Dict[str, Any]:
        """Create standardized output metadata.

        Args:
            output_dir: Output directory path

        Returns:
            Dictionary of metadata for output datasets
        """
        metadata = {
            "simulation_name": self.simulation_config.name,
            "backend": self.name,
            "created_at": self.simulation_config.created_at.isoformat(),
            "sensor_count": len(self.simulation_config.sensors),
            "radiative_quantity_count": len(
                getattr(self.simulation_config, "radiative_quantities", [])
            ),
            "measurement_types": [
                mt.value for mt in self.simulation_config.output_quantities
            ]
            if hasattr(self.simulation_config, "output_quantities")
            else [],
            "wavelength_range": self.simulation_config.wavelength_range,
            "orthorectified": self.simulation_config.processing.orthorectified,
        }

        if output_dir:
            metadata["output_dir"] = str(output_dir)

        return metadata

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.__class__.__name__

    @property
    def supported_platforms(self) -> List[str]:
        """List of supported observation platforms.

        Override in specific backends to define platform support.

        Returns:
            List of platform names this backend supports
        """
        return ["satellite", "uav", "ground"]

    @property
    def supported_measurements(self) -> List[str]:
        """List of supported measurement types.

        Override in specific backends to define measurement support.

        Returns:
            List of measurement type names this backend supports
        """
        return ["radiance", "brf", "hdrf", "bhr"]

    def __str__(self) -> str:
        return f"{self.name}(available={self.is_available()})"
