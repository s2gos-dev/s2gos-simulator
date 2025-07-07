from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np
import xarray as xr
from PIL import Image

from .base import SimulationBackend
from ..config import (
    SimulationConfig, SatelliteSensor, UAVSensor, GroundSensor,
    DirectionalIllumination, ConstantIllumination, MeasurementType,
    PlatformType, AngularViewing, AngularFromOriginViewing, LookAtViewing, HemisphericalViewing,
    UAVInstrumentType, GroundInstrumentType, SatellitePlatform, SatelliteInstrument
)

try:
    import eradiate
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.units import unit_registry as ureg
    from eradiate.xarray.interp import dataarray_to_rgb
    import mitsuba as mi
    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class EradiateBackend(SimulationBackend):
    """
    Enhanced Eradiate backend for the new configuration system.
    """
    
    def __init__(self, simulation_config: SimulationConfig):
        """Initialize the Eradiate backend with new configuration system."""
        super().__init__(simulation_config)
        
        if ERADIATE_AVAILABLE:
            eradiate.set_mode("mono")
    
    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available."""
        return ERADIATE_AVAILABLE
    
    @property
    def supported_platforms(self) -> List[str]:
        """Eradiate supports all platform types."""
        return ["satellite", "uav", "ground"]
    
    @property
    def supported_measurements(self) -> List[str]:
        """Measurement types supported by Eradiate."""
        return ["radiance", "brf", "hdrf", "bhr", "bhr_iso", "fapar", "flux_3d", "irradiance", "dhp"]
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration for Eradiate backend."""
        errors = super().validate_configuration()
        
        if not self.is_available():
            errors.append("Eradiate is not available. Install with: pip install eradiate[kernel]")
        
        for sensor in self.simulation_config.sensors:
            if sensor.platform_type == PlatformType.SATELLITE:
                validation_error = self._validate_satellite_sensor(sensor)
                if validation_error:
                    errors.append(validation_error)
            elif sensor.platform_type == PlatformType.GROUND:
                if sensor.instrument not in [GroundInstrumentType.HYPSTAR, GroundInstrumentType.PERSPECTIVE_CAMERA, 
                                           GroundInstrumentType.PYRANOMETER, GroundInstrumentType.FLUX_METER, 
                                           GroundInstrumentType.DHP_CAMERA]:
                    errors.append(f"Ground sensor {sensor.id} instrument type {sensor.instrument} is not supported")
        
        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity.value not in self.supported_measurements:
                errors.append(f"Radiative quantity {rq.quantity} is not supported by Eradiate backend")
            
            if rq.quantity in [MeasurementType.BRF, MeasurementType.HDRF, MeasurementType.RADIANCE]:
                if rq.viewing_zenith is None or rq.viewing_azimuth is None:
                    errors.append(f"Radiative quantity {rq.quantity} requires viewing_zenith and viewing_azimuth")
        
        return errors
    
    def _validate_satellite_sensor(self, sensor) -> Optional[str]:
        """Validate satellite sensor platform/instrument/band combination using enum system."""
        from ..config import PLATFORM_INSTRUMENTS, INSTRUMENT_BANDS, SatellitePlatform, SatelliteInstrument
        
        if sensor.platform == SatellitePlatform.CUSTOM:
            if sensor.srf is None:
                return f"Custom platform requires explicit SRF configuration"
            return None
        
        try:
            platform_enum = SatellitePlatform(sensor.platform)
            instrument_enum = SatelliteInstrument(sensor.instrument)
        except ValueError as e:
            return f"Invalid platform or instrument: {e}"
        
        valid_instruments = PLATFORM_INSTRUMENTS.get(platform_enum, [])
        if instrument_enum not in valid_instruments:
            return f"Platform '{sensor.platform}' does not support instrument '{sensor.instrument}'. Valid instruments: {[inst.value for inst in valid_instruments]}"
        
        band_enum_class = INSTRUMENT_BANDS.get(instrument_enum)
        if band_enum_class is not None:
            try:
                band_enum_class(sensor.band)
            except ValueError:
                valid_bands = [band.value for band in band_enum_class]
                return f"Instrument '{sensor.platform}/{sensor.instrument}' does not support band '{sensor.band}'. Valid bands: {valid_bands}"
        
        return None
    
    def run_simulation(self, scene_config, scene_dir: Path, 
                      output_dir: Optional[Path] = None, 
                      **kwargs) -> xr.Dataset:
        """Run Eradiate simulation with new configuration system."""
        if not self.is_available():
            raise RuntimeError("Eradiate is not available")
        
        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Running Eradiate simulation: {self.simulation_config.name}")
        print(f"Sensors: {len(self.simulation_config.sensors)}")
        print(f"Radiative quantities: {len(self.simulation_config.radiative_quantities)}")
        print(f"Measurement types: {[mt.value for mt in self.simulation_config.output_quantities]}")
        
        
        experiment = self._create_experiment(scene_config, scene_dir)
        
        print("Executing Eradiate simulation...")
        eradiate.run(experiment)
        
        plot_image = kwargs.get('plot_image', False)
        if plot_image:
            self._create_rgb_visualization(experiment, output_dir, kwargs.get('id_to_plot', 'rgb_camera'))
        
        return self._process_results(experiment, output_dir)
    
    def _create_experiment(self, scene_config, scene_dir: Path):
        """Create Eradiate experiment from new configuration system."""
        try:
            from s2gos_generator.materials.registry import MaterialRegistry
            from s2gos_generator.assets.atmosphere import create_atmosphere
        except ImportError as e:
            raise ImportError(f"s2gos_generator is required: {e}")
        
        materials = scene_config.materials
        kdict, kpmap = MaterialRegistry.create_material_kdict_kpmap(materials)
        
        kdict.update(self._create_target_surface(scene_config, scene_dir))
        
        if scene_config.buffer:
            kdict.update(self._create_buffer_surface(scene_config, scene_dir))
        
        if scene_config.background:
            kdict.update(self._create_background_surface(scene_config, scene_dir))
        
        atmosphere = self._create_atmosphere_from_config(scene_config)
        
        illumination = self._translate_illumination()
        
        measures = self._translate_sensors()
        
        return AtmosphereExperiment(
            geometry={"type": "plane_parallel", "toa_altitude": 40.0 * ureg.km},
            atmosphere=atmosphere,
            surface=None,
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap
        )
    
    def _create_atmosphere_from_config(self, scene_config):
        """Create atmosphere based on the configuration mode."""
        try:
            from s2gos_generator.assets.atmosphere import create_atmosphere
            from s2gos_generator.core.config import AtmosphereMode
        except ImportError as e:
            raise ImportError(f"s2gos_generator atmosphere assets required: {e}")
        
        # Handle both legacy (dict) and new (Pydantic) atmosphere configs
        if hasattr(scene_config, 'atmosphere') and hasattr(scene_config.atmosphere, 'mode'):
            # New Pydantic configuration
            atm_config = scene_config.atmosphere
            
            if atm_config.mode == AtmosphereMode.PRESET:
                return create_atmosphere(
                    boa=atm_config.boa,
                    toa=atm_config.toa,
                    aerosol_ot=atm_config.aerosol_ot,
                    aerosol_scale=atm_config.aerosol_scale,
                    aerosol_ds=atm_config.aerosol_ds.value
                )
            
            elif atm_config.mode == AtmosphereMode.DATASET:
                # Use specific Eradiate datasets
                return self._create_atmosphere_from_datasets(atm_config)
            
            elif atm_config.mode == AtmosphereMode.CUSTOM:
                # Use custom molecular/particle layers
                return self._create_atmosphere_from_custom_layers(atm_config)
            
            elif atm_config.mode == AtmosphereMode.RAW:
                # Use raw Eradiate configuration
                return atm_config.raw_config
            
            else:
                raise ValueError(f"Unsupported atmosphere mode: {atm_config.mode}")
        
        else:
            # Legacy configuration (dict-based)
            atmosphere = scene_config.atmosphere if hasattr(scene_config, 'atmosphere') else {}
            return create_atmosphere(
                boa=atmosphere.get("boa", 0.0),
                toa=atmosphere.get("toa", 40.0e3),
                aerosol_ot=atmosphere.get("aerosol_ot", 0.1),
                aerosol_scale=atmosphere.get("aerosol_scale", 1e3),
                aerosol_ds=atmosphere.get("aerosol_ds", "sixsv-continental")
            )
    
    def _create_atmosphere_from_datasets(self, atm_config):
        """Create atmosphere using specific Eradiate datasets."""
        # This would be implemented based on Eradiate's dataset loading capabilities
        # For now, fall back to basic configuration
        aerosol_ds = atm_config.aerosol_dataset.value if atm_config.aerosol_dataset else "sixsv-continental"
        
        try:
            from s2gos_generator.assets.atmosphere import create_atmosphere
            return create_atmosphere(
                boa=atm_config.boa,
                toa=atm_config.toa,
                aerosol_ot=0.1,  # Default value
                aerosol_scale=1000.0,  # Default value
                aerosol_ds=aerosol_ds
            )
        except ImportError as e:
            raise ImportError(f"s2gos_generator atmosphere assets required: {e}")
    
    def _create_atmosphere_from_custom_layers(self, atm_config):
        """Create atmosphere using custom molecular/particle layers."""
        # This would be implemented to create custom Eradiate atmosphere
        # For now, fall back to basic configuration
        try:
            from s2gos_generator.assets.atmosphere import create_atmosphere
            return create_atmosphere(
                boa=atm_config.boa,
                toa=atm_config.toa,
                aerosol_ot=0.1,  # Default value
                aerosol_scale=1000.0,  # Default value  
                aerosol_ds="sixsv-continental"  # Default dataset
            )
        except ImportError as e:
            raise ImportError(f"s2gos_generator atmosphere assets required: {e}")
    
    def _translate_illumination(self) -> Dict[str, Any]:
        """Translate generic illumination to Eradiate format."""
        illumination = self.simulation_config.illumination
        
        if isinstance(illumination, DirectionalIllumination):
            return {
                "type": "directional",
                "id": illumination.id,
                "zenith": illumination.zenith * ureg.deg,
                "azimuth": illumination.azimuth * ureg.deg,
                "irradiance": {
                    "type": "solar_irradiance",
                    "dataset": illumination.irradiance_dataset
                }
            }
        elif isinstance(illumination, ConstantIllumination):
            return {
                "type": "constant",
                "id": illumination.id,
                "radiance": illumination.radiance
            }
        else:
            raise ValueError(f"Unsupported illumination type: {type(illumination)}")
    
    def _calculate_target_from_angles(self, view: AngularFromOriginViewing) -> tuple[list[float], list[float]]:
        """
        Calculates a target point and direction vector from an AngularFromOriginViewing object.
        
        Returns:
            A tuple containing (target_position, direction_vector).
        """
        zen_rad = np.deg2rad(view.zenith)
        az_rad = np.deg2rad(view.azimuth)
        
        direction = np.array([
            np.sin(zen_rad) * np.cos(az_rad),
            np.sin(zen_rad) * np.sin(az_rad),
            np.cos(zen_rad)
        ])
        
        origin_vec = np.array(view.origin)
        target_vec = origin_vec + direction
        
        return target_vec.tolist(), direction.tolist()

    def _translate_sensors(self) -> List[Dict[str, Any]]:
        """Translate generic sensors and radiative quantities to Eradiate measures."""
        measures = []
        
        # Translate sensor-based measures
        for sensor in self.simulation_config.sensors:
            if isinstance(sensor, SatelliteSensor):
                measures.append(self._translate_satellite_sensor(sensor))
            elif isinstance(sensor, UAVSensor):
                measures.append(self._translate_uav_sensor(sensor))
            elif isinstance(sensor, GroundSensor):
                measures.append(self._translate_ground_sensor(sensor))
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor)}")
        
        # Translate radiative quantity measures
        for rad_quantity in self.simulation_config.radiative_quantities:
            measures.append(self._translate_radiative_quantity(rad_quantity))
        
        return measures

    def _translate_satellite_sensor(self, sensor: SatelliteSensor) -> Dict[str, Any]:
        """Translate satellite sensor to Eradiate measure."""
        measure_config = {
            "type": "distant",
            "id": sensor.id,
            "spp": sensor.samples_per_pixel,
            "srf": self._translate_srf(sensor.srf),
            "construct": "from_angles",
            "angles": [sensor.viewing.zenith, sensor.viewing.azimuth]
        }
        
        if sensor.viewing.target:
            measure_config["target"] = sensor.viewing.target
        
        return measure_config

    def _translate_uav_sensor(self, sensor: UAVSensor) -> Dict[str, Any]:
        """Translate UAV sensor to Eradiate measure."""
        view = sensor.viewing

        base_config = {
            "id": sensor.id,
            "spp": sensor.samples_per_pixel,
            "srf": self._translate_srf(sensor.srf),
            "origin": view.origin,
        }

        if sensor.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            base_config["type"] = "perspective"
            base_config["fov"] = sensor.fov or 70.0
            base_config["film_resolution"] = sensor.resolution or [1024, 1024]
            
            if isinstance(view, LookAtViewing):
                base_config["target"] = view.target
                base_config["up"] = view.up or [0, 0, 1]
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_config["target"] = target
                base_config["up"] = view.up or [0, 0, 1]

        elif sensor.instrument == UAVInstrumentType.RADIANCEMETER:
            base_config["type"] = "radiancemeter"
            
            if isinstance(view, LookAtViewing):
                base_config["target"] = view.target
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_config["target"] = target
        
        else:
            raise ValueError(f"Unsupported UAV instrument type: {sensor.instrument}")

        return base_config

    def _translate_ground_sensor(self, sensor: GroundSensor) -> Dict[str, Any]:
        """Translate ground sensor to Eradiate measure."""
        view = sensor.viewing

        base_measure = {
            "id": sensor.id,
            "spp": sensor.samples_per_pixel,
            "srf": self._translate_srf(sensor.srf),
        }

        if isinstance(view, HemisphericalViewing):
            base_measure["type"] = "hdistant"
            base_measure["direction"] = [0, 0, 1] if view.upward_looking else [0, 0, -1]
            
        elif isinstance(view, (LookAtViewing, AngularFromOriginViewing)):
            base_measure["origin"] = view.origin

            if sensor.instrument in [GroundInstrumentType.PERSPECTIVE_CAMERA, GroundInstrumentType.DHP_CAMERA]:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = [1024, 1024]
                base_measure["fov"] = 180.0 if sensor.instrument == GroundInstrumentType.DHP_CAMERA else 70.0
                base_measure["up"] = view.up or [0, 0, 1]
            else:
                base_measure["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_measure["target"] = view.target
                
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_measure["target"] = target
        else:
            raise ValueError(f"Unsupported viewing type for ground sensor: {type(view)}")

        return base_measure
    
    def _translate_radiative_quantity(self, rad_quantity) -> Dict[str, Any]:
        """Translate radiative quantity configuration to placeholder measure.
        
        TODO: This is a placeholder implementation. Future versions will:
        1. Determine appropriate sensors needed for the radiative quantity
        2. Instantiate those sensors internally
        3. Calculate the requested quantity from sensor results
        """
        from ..config import MeasurementType
        
        quantity_id = f"{rad_quantity.quantity.value}_measure"
        
        # Log TODO message for user clarity
        print(f"TODO: {rad_quantity.quantity.value.upper()} calculation not yet implemented - generating placeholder")
        
        base_config = {
            "id": quantity_id,
            "type": "distant",
            "construct": "from_angles", 
            "angles": [0.0, 0.0],  # Nadir view as placeholder
            "spp": rad_quantity.samples_per_pixel
        }
        
        # Set basic wavelength configuration
        if rad_quantity.wavelengths:
            base_config["srf"] = {
                "type": "delta",
                "wavelengths": rad_quantity.wavelengths
            }
        elif rad_quantity.wavelength_range:
            base_config["srf"] = {
                "type": "uniform",
                "wmin": rad_quantity.wavelength_range[0],
                "wmax": rad_quantity.wavelength_range[1],
                "value": 1.0
            }
        else:
            # Default to visible spectrum
            base_config["srf"] = {
                "type": "uniform",
                "wmin": 400.0,
                "wmax": 700.0,
                "value": 1.0
            }
        
        return base_config
        
    def _translate_srf(self, srf) -> Union[Dict[str, Any], str]:
        """Translate generic SRF to Eradiate format."""
        if srf is None:
            return {
                "type": "uniform",
                "wmin": 400.0,
                "wmax": 700.0,
                "value": 1.0
            }
        elif isinstance(srf, str):
            return srf
        elif isinstance(srf, dict):
            return srf
        else:
            # From config
            if srf.type == "delta":
                return {
                    "type": "delta",
                    "wavelengths": srf.wavelengths
                }
            elif srf.type == "uniform":
                return {
                    "type": "uniform",
                    "wmin": srf.wmin,
                    "wmax": srf.wmax,
                    "value": 1.0
                }
            elif srf.type == "dataset":
                return srf.dataset_id
            elif srf.type == "custom":
                if srf.data and "wavelengths" in srf.data and "values" in srf.data:
                    return {
                        "type": "array",
                        "wavelengths": srf.data["wavelengths"],
                        "values": srf.data["values"]
                    }
                else:
                    raise ValueError("Custom SRF requires 'wavelengths' and 'values' in data")
            else:
                raise ValueError(f"Unsupported SRF type: {srf.type}")
    
    def _resolve_platform_srf(self, platform: str, instrument: str, band: str) -> str:
        """
        Resolve platform/instrument/band combination to Eradiate SRF identifier.
        
        This method converts platform identifiers to Eradiate dataset identifiers.
        """
        platform_norm = platform.lower().replace('-', '_')
        instrument_norm = instrument.lower()
        band_norm = band.lower()
        
        srf_id = f"{platform_norm}-{instrument_norm}-{band_norm}"
        
        return srf_id
    
    def _create_output_metadata(self, output_dir: Path) -> Dict[str, Any]:
        """Create standardized metadata for output files."""
        return {
            'simulation_name': self.simulation_config.name,
            'description': self.simulation_config.description,
            'created_at': self.simulation_config.created_at.isoformat(),
            'backend': 'eradiate',
            'output_dir': str(output_dir),
            'num_sensors': len(self.simulation_config.sensors),
            'num_radiative_quantities': len(self.simulation_config.radiative_quantities),
            'sensor_types': [s.platform_type.value for s in self.simulation_config.sensors],
            'radiative_quantities': [rq.quantity.value for rq in self.simulation_config.radiative_quantities],
            'illumination_type': self.simulation_config.illumination.type
        }
    
    def _create_target_surface(self, scene_config, scene_dir: Path) -> Dict[str, Any]:
        """Create target surface (reused from original backend)."""
        target_mesh_path = scene_dir / scene_config.target["mesh"]
        target_texture_path = scene_dir / scene_config.target["selection_texture"]
        
        texture_image = Image.open(target_texture_path)
        selection_texture_data = np.array(texture_image)
        selection_texture_data = np.atleast_3d(selection_texture_data)
        
        material_ids = [
            "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
            "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
            "_mat_wetland", "_mat_mangroves", "_mat_moss"
        ]
        
        return {
            "terrain_material": {
                "type": "selectbsdf",
                "id": "terrain_material",
                "indices": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": selection_texture_data,
                },
                **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
                for i, mat_id in enumerate(material_ids)}
            },
            "terrain": {
                "type": "ply",
                "filename": str(target_mesh_path),
                "bsdf": {"type": "ref", "id": "terrain_material"},
                "id": "terrain"
            }
        }
    
    def _create_buffer_surface(self, scene_config, scene_dir: Path) -> Dict[str, Any]:
        """Create buffer surface (reused from original backend)."""
        buffer_mesh_path = scene_dir / scene_config.buffer["mesh"]
        buffer_texture_path = scene_dir / scene_config.buffer["selection_texture"]
        mask_path = scene_dir / scene_config.buffer["mask_texture"] if "mask_texture" in scene_config.buffer else None
        
        buffer_texture_image = Image.open(buffer_texture_path)
        buffer_selection_texture_data = np.array(buffer_texture_image)
        buffer_selection_texture_data = np.atleast_3d(buffer_selection_texture_data)
        
        material_ids = [
            "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
            "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
            "_mat_wetland", "_mat_mangroves", "_mat_moss"
        ]
        
        result = {
            "buffer_material": {
                "type": "selectbsdf",
                "id": "buffer_material",
                "indices": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": buffer_selection_texture_data,
                },
                **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
                   for i, mat_id in enumerate(material_ids)}
            }
        }
        
        buffer_bsdf_id = "buffer_material"
        
        if mask_path and Path(mask_path).exists():
            mask_image = Image.open(mask_path)
            mask_data = np.array(mask_image) / 255.0
            mask_data = np.atleast_3d(mask_data)
            
            result["buffer_mask"] = {
                "type": "mask",
                "opacity": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": mask_data,
                },
                "material": {"type": "ref", "id": "buffer_material"}
            }
            buffer_bsdf_id = "buffer_mask"
        
        result["buffer_terrain"] = {
            "type": "ply",
            "filename": str(buffer_mesh_path),
            "bsdf": {"type": "ref", "id": buffer_bsdf_id},
            "id": "buffer_terrain"
        }
        
        return result
    
    def _create_background_surface(self, scene_config, scene_dir: Path) -> Dict[str, Any]:
        """Create background surface (reused from original backend)."""
        elevation = scene_config.background["elevation"]
        mask_texture_path = scene_dir / scene_config.background["mask_texture"]
        
        material_name = scene_config.background.get("material", "water")
        if not material_name.startswith("_mat_"):
            material_id = f"_mat_{material_name}"
        else:
            material_id = material_name
            
        mask_edge_length = scene_config.background.get("mask_edge_length", 100000.0)
        
        shape_size = 1e9
        scale = shape_size / mask_edge_length / 3.0
        offset = -0.002
        
        to_world = mi.ScalarTransform4f.translate(
            [0, 0, elevation + offset]
        ) @ mi.ScalarTransform4f.scale(0.5 * shape_size)
        
        to_uv = mi.ScalarTransform4f.scale(
            [scale, scale, 1]
        ) @ mi.ScalarTransform4f.translate(
            [0.5 * (1.0 / scale - 1.0), 0.5 * (1.0 / scale - 1.0), 0.0]
        )
        
        return {
            "background_surface": {
                "type": "rectangle",
                "to_world": to_world,
                "bsdf": {
                    "type": "mask",
                    "opacity": {
                        "type": "bitmap",
                        "filename": str(mask_texture_path),
                        "raw": True,
                        "filter_type": "nearest",
                        "wrap_mode": "clamp",
                        "to_uv": to_uv,
                    },
                    "material": {"type": "ref", "id": material_id},
                },
                "id": "background_surface",
            }
        }
    
    def _create_rgb_visualization(self, experiment, output_dir: Path, id_to_plot: str):
        """Create RGB visualization from camera results."""
        try:
            if id_to_plot not in experiment.results:
                print(f"Warning: Sensor '{id_to_plot}' not found in results")
                return
                
            sensor_data = experiment.results[id_to_plot]
            
            if "radiance" in sensor_data:
                radiance_data = sensor_data["radiance"]
                
                if "x_index" in radiance_data.dims and "y_index" in radiance_data.dims:
                    img = dataarray_to_rgb(
                        radiance_data,
                        channels=[("w", 660), ("w", 550), ("w", 440)],
                        normalize=False,
                    ) * 1.8
                    
                    img = np.clip(img, 0, 1)
                    
                    rgb_output = output_dir / f"{id_to_plot}_rgb.png"
                    plt_img = (img * 255).astype(np.uint8)
                    rgb_image = Image.fromarray(plt_img)
                    rgb_image.save(rgb_output)
                    
                    print(f"Camera RGB image saved to: {rgb_output}")
                    
                else:
                    spectral_output = output_dir / f"{id_to_plot}_spectrum.png"
                    self._plot_spectral_data(radiance_data, spectral_output)
                    print(f"Spectral data plot saved to: {spectral_output}")
                    
        except Exception as e:
            print(f"Warning: Could not create visualization for {id_to_plot}: {e}")
    
    def _plot_spectral_data(self, radiance_data, output_path: Path):
        """Plot spectral data for point sensors."""
        try:
            import matplotlib.pyplot as plt
            
            wavelengths = radiance_data.coords["w"].values
            radiance_values = radiance_data.values
            
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, radiance_values, 'b-', linewidth=2)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Radiance')
            plt.title('Spectral Radiance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available for spectral plotting")
        except Exception as e:
            print(f"Warning: Could not create spectral plot: {e}")
    
    def _process_results(self, experiment, output_dir: Path) -> xr.Dataset:
        """Process and save simulation results."""
        results = experiment.results
        
        if not results:
            raise ValueError("No results found in experiment")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = self._create_output_metadata(output_dir)
        
        # Process regular sensor results
        if isinstance(results, dict):
            for sensor_id, dataset in results.items():
                sensor_output = output_dir / f"{self.simulation_config.name}_{sensor_id}.zarr"
                
                dataset.attrs.update(metadata)
                dataset.attrs['sensor_id'] = sensor_id
                
                dataset.to_zarr(sensor_output, mode="w")
                print(f"Sensor '{sensor_id}' saved to {sensor_output}")
            
        else:
            results_ds = results
            single_output = output_dir / f"{self.simulation_config.name}_results.zarr"
            results_ds.attrs.update(metadata)
            results_ds.to_zarr(single_output, mode="w")
            print(f"Results saved to {single_output}")
        
        # Generate dummy results for radiative quantities (TODO placeholders)
        for rad_quantity in self.simulation_config.radiative_quantities:
            dummy_output = self._create_dummy_radiative_quantity_result(rad_quantity, output_dir, metadata)
            print(f"TODO: {rad_quantity.quantity.value.upper()} placeholder saved to {dummy_output}")
        
        return results
    
    def _create_dummy_radiative_quantity_result(self, rad_quantity, output_dir: Path, metadata: dict) -> Path:
        """Create dummy Zarr file for radiative quantity placeholder.
        
        TODO: This creates placeholder data. Future implementation will:
        1. Use results from appropriate sensors 
        2. Calculate the actual radiative quantity
        3. Return real calculated values
        """
        import numpy as np
        
        quantity_id = f"{rad_quantity.quantity.value}_measure"
        dummy_output = output_dir / f"{self.simulation_config.name}_{quantity_id}.zarr"
        
        # Create dummy data 
        dummy_data = np.ones((10, 10)) * 0.5

        # Get wavelengths for the quantity
        if rad_quantity.wavelengths:
            wavelengths = rad_quantity.wavelengths
        elif rad_quantity.wavelength_range:
            # Create 3 sample wavelengths in the range
            wmin, wmax = rad_quantity.wavelength_range
            wavelengths = [wmin, (wmin + wmax) / 2, wmax]
        else:
            wavelengths = [550.0]  # Default visible wavelength
        
        # Create xarray dataset with appropriate structure
        if len(wavelengths) > 1:
            # Multi-spectral data
            dummy_values = np.stack([dummy_data for _ in wavelengths], axis=0)
            coords = {
                'wavelength': ('wavelength', wavelengths),
                'x': ('x', np.arange(10)),
                'y': ('y', np.arange(10))
            }
            dims = ['wavelength', 'y', 'x']
        else:
            # Single wavelength
            dummy_values = dummy_data
            coords = {
                'x': ('x', np.arange(10)),
                'y': ('y', np.arange(10))
            }
            dims = ['y', 'x']
        
        # Create dataset
        dummy_dataset = xr.Dataset({
            rad_quantity.quantity.value: (dims, dummy_values)
        }, coords=coords)
        
        # Add metadata
        dummy_dataset.attrs.update(metadata)
        dummy_dataset.attrs.update({
            'radiative_quantity': rad_quantity.quantity.value,
            'status': 'TODO_PLACEHOLDER',
            'description': f'Placeholder data for {rad_quantity.quantity.value} calculation',
            'samples_per_pixel': rad_quantity.samples_per_pixel,
            'viewing_zenith': rad_quantity.viewing_zenith,
            'viewing_azimuth': rad_quantity.viewing_azimuth
        })
        
        # Save as Zarr
        dummy_dataset.to_zarr(dummy_output, mode="w")
        
        return dummy_output