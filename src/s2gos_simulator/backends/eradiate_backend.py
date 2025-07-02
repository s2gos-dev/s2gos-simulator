from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import xarray as xr
from PIL import Image

from .base import SimulationBackend

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
    """Unified Eradiate backend for radiative transfer simulations.
    
    This backend consolidates all Eradiate functionality into a single class,
    handling scene setup, surface creation, simulation execution, and result processing.
    """
    
    def __init__(self, render_config):
        """Initialize the Eradiate backend.
        
        Args:
            render_config: Rendering configuration (sensors, illumination)
        """
        super().__init__(render_config)
        
        if ERADIATE_AVAILABLE:
            eradiate.set_mode("mono")
    
    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available."""
        return ERADIATE_AVAILABLE
    
    def run_simulation(self, scene_config, scene_dir: Path, 
                      output_dir: Optional[Path] = None, plot_image: bool=False,
                      id_to_plot: str="rgb_camera") -> xr.Dataset:
        """Run complete Eradiate simulation pipeline.
        
        Args:
            scene_config: Scene configuration from s2gos_generator
            scene_dir: Directory containing scene assets
            output_dir: Directory for simulation outputs (defaults to scene_dir/eradiate_renders)
            
        Returns:
            xarray.Dataset containing simulation results. The dataset is also saved to NetCDF
            and the file path is stored in the 'saved_to' attribute.
        """
        if not self.is_available():
            raise RuntimeError("Eradiate is not available. Install with: pip install eradiate[kernel]")
        
        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Rendering scene '{scene_config.name}' with Eradiate...")
        print(f"Scene location: {scene_config.metadata.center_lat}, {scene_config.metadata.center_lon}")
        print(f"Target: {scene_config.target['mesh']} + {scene_config.target['selection_texture']}")
        if scene_config.buffer:
            print(f"Buffer: {scene_config.buffer['mesh']} + {scene_config.buffer['selection_texture']}")
        if scene_config.background:
            print(f"Background: {scene_config.background['material']} at {scene_config.background['elevation']:.1f}m")
        
        # Create the complete experiment
        experiment = self._create_experiment(scene_config, scene_dir)
        
        print("Running Eradiate simulation (this may take several minutes)...")
        eradiate.run(experiment)
        
        if plot_image:
            # Create RGB visualization
            img = dataarray_to_rgb(
                experiment.results[id_to_plot]["radiance"],
                channels=[("w", 660), ("w", 550), ("w", 440)],
                normalize=False,
            ) * 1.8
            
            img = np.clip(img, 0, 1)
            
            # Save RGB image
            rgb_output = output_dir / "eradiate_rgb.png"
            plt_img = (img * 255).astype(np.uint8)
            rgb_image = Image.fromarray(plt_img)
            rgb_image.save(rgb_output)
            
            print(f"Eradiate simulation complete!")
            print(f"RGB visualization: {rgb_output}")
        
        # Process and save results
        return self._process_results(experiment, output_dir)
    
    def validate_scene(self, scene_config, scene_dir: Path) -> List[str]:
        """Validate scene configuration for Eradiate backend."""
        errors = []
        
        # Check required files exist
        target_mesh = scene_dir / scene_config.target["mesh"]
        if not target_mesh.exists():
            errors.append(f"Target mesh not found: {target_mesh}")
        
        target_texture = scene_dir / scene_config.target["selection_texture"]
        if not target_texture.exists():
            errors.append(f"Target texture not found: {target_texture}")
        
        if scene_config.buffer:
            buffer_mesh = scene_dir / scene_config.buffer["mesh"]
            if not buffer_mesh.exists():
                errors.append(f"Buffer mesh not found: {buffer_mesh}")
                
            buffer_texture = scene_dir / scene_config.buffer["selection_texture"]
            if not buffer_texture.exists():
                errors.append(f"Buffer texture not found: {buffer_texture}")
        
        if scene_config.background and "mask_texture" in scene_config.background:
            bg_mask = scene_dir / scene_config.background["mask_texture"]
            if not bg_mask.exists():
                errors.append(f"Background mask not found: {bg_mask}")
        
        return errors
    
    def _create_experiment(self, scene_config, scene_dir: Path):
        """Create an Eradiate experiment from scene configuration."""
        # Import here to avoid circular imports
        try:
            from s2gos_generator.materials.registry import MaterialRegistry
            from s2gos_generator.assets.atmosphere import create_atmosphere
        except ImportError as e:
            raise ImportError(f"s2gos_generator is required for scene materials and atmosphere: {e}")
        
        # Create materials
        materials = scene_config.materials
        kdict, kpmap = MaterialRegistry.create_material_kdict_kpmap(materials)
        
        # Create surfaces
        kdict.update(self._create_target_surface(scene_config, scene_dir))
        
        if scene_config.buffer:
            kdict.update(self._create_buffer_surface(scene_config, scene_dir))
        
        if scene_config.background:
            kdict.update(self._create_background_surface(scene_config, scene_dir))
        
        # Create atmosphere from scene config
        atmosphere = create_atmosphere(
            boa=scene_config.atmosphere.get("boa", 0.0),
            toa=scene_config.atmosphere.get("toa", 40.0e3),
            aerosol_ot=scene_config.atmosphere.get("aerosol_ot", 0.1),
            aerosol_scale=scene_config.atmosphere.get("aerosol_scale", 1e3),
            aerosol_ds=scene_config.atmosphere.get("aerosol_ds", "sixsv-continental")
        )
        
        illumination = self._create_illumination_config()
        
        measures = []
        for sensor in self.render_config.sensors:
            measures.append(self._create_measure_config(sensor))
        print(measures)
        return AtmosphereExperiment(
            geometry={"type": "plane_parallel", "toa_altitude": 40.0 * ureg.km},
            atmosphere=atmosphere,
            surface=None,
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap
        )
    
    def _create_target_surface(self, scene_config, scene_dir: Path) -> Dict[str, Any]:
        """Create target surface with material selection."""
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
        """Create buffer surface with material selection and optional masking."""
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
        """Create background surface using dreams-scenes approach."""
        elevation = scene_config.background["elevation"]
        mask_texture_path = scene_dir / scene_config.background["mask_texture"]
        
        # Get material ID and ensure it has the proper "_mat_" prefix
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
        
        print(f"Added background surface at elevation {elevation:.1f}m")
        
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
    
    def _process_results(self, experiment, output_dir: Path) -> xr.Dataset:
        """
        Process simulation results, save each sensor to a separate netCDF file,
        and optionally create a combined Dataset.
        
        This method handles cases where results are a single Dataset or a 
        dictionary of Datasets keyed by sensor ID. When multiple sensors are
        present, each sensor's results are saved to individual netCDF files.
        """
        results = experiment.results

        if not results:
            raise ValueError("No results found in experiment.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []

        if isinstance(results, dict):
            print(f"Found a dictionary of results with {len(results)} sensors.")
            print("Saving each sensor to separate netCDF files...")
            
            datasets = list(results.values())
            sensor_ids = list(results.keys())
            
            # Save each sensor's results to separate files
            for sensor_id, dataset in results.items():
                sensor_output = output_dir / f"eradiate_results_{sensor_id}.nc"
                
                # Add metadata to individual dataset
                dataset.attrs['sensor_id'] = sensor_id
                dataset.attrs['saved_to'] = str(sensor_output)
                dataset.attrs['output_dir'] = str(output_dir)
                dataset.attrs['backend'] = 'eradiate'
                
                dataset.to_netcdf(sensor_output)
                saved_files.append(sensor_output)
                print(f"  Sensor '{sensor_id}' saved to {sensor_output}")
            
            # Create concatenated dataset for return value
            try:
                results_ds = xr.concat(datasets, dim="sensor_id", compat='override')
                results_ds = results_ds.assign_coords(sensor_id=sensor_ids)
                
                # Also save combined file for backward compatibility
                combined_output = output_dir / "eradiate_results_combined.nc"
                results_ds.to_netcdf(combined_output)
                saved_files.append(combined_output)
                print(f"  Combined results saved to {combined_output}")
            except Exception as e:
                print(f"  Warning: Could not create combined file due to incompatible coordinates: {e}")
                # Return the first dataset as fallback
                results_ds = datasets[0]
                results_ds.attrs['note'] = f"Combined file could not be created - using {sensor_ids[0]} dataset"

        elif isinstance(results, xr.Dataset):
            print("Found a single results Dataset.")
            results_ds = results
            
            # Save single dataset
            raw_output = output_dir / "eradiate_results.nc"
            results_ds.to_netcdf(raw_output)
            saved_files.append(raw_output)
            print(f"Results successfully saved to {raw_output}")
            
        else:
            raise TypeError(
                f"Unsupported experiment results type: {type(results)}. "
                "Expected xr.Dataset or Dict[str, xr.Dataset]."
            )
        
        # Update return dataset attributes
        results_ds.attrs['saved_files'] = [str(f) for f in saved_files]
        results_ds.attrs['output_dir'] = str(output_dir)
        results_ds.attrs['backend'] = 'eradiate'
        
        return results_ds
    
    def _create_illumination_config(self) -> Dict[str, Any]:
        """Create illumination configuration for Eradiate."""
        illumination = self.render_config.illumination
        config = illumination.to_dict()
        
        # Add units for angular parameters
        if "zenith" in config:
            config["zenith"] = config["zenith"] * ureg.deg
        if "azimuth" in config:
            config["azimuth"] = config["azimuth"] * ureg.deg
        
        return config
    
    def _create_measure_config(self, sensor) -> Dict[str, Any]:
        """Create measure configuration for Eradiate from sensor."""
        config = sensor.to_dict()
        
        return config