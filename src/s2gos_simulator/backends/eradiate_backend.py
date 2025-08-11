from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import open_file
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .base import SimulationBackend
from .eradiate_materials import EradiateMaterialAdapter
from ..config import (
    AngularFromOriginViewing,
    ConstantIllumination,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HemisphericalViewing,
    LookAtViewing,
    MeasurementType,
    PlatformType,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
    UAVInstrumentType,
    UAVSensor,
)

try:
    import eradiate
    import mitsuba as mi
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.scenes.atmosphere import (
        ExponentialParticleDistribution,
        GaussianParticleDistribution,
        HeterogeneousAtmosphere,
        HomogeneousAtmosphere,
        MolecularAtmosphere,
        ParticleLayer,
        UniformParticleDistribution,
    )
    from eradiate.units import unit_registry as ureg
    from eradiate.xarray.interp import dataarray_to_rgb

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

        eradiate_hints = simulation_config.backend_hints.get("eradiate", {})
        self._eradiate_mode = eradiate_hints.get("mode", "mono")

        if ERADIATE_AVAILABLE:
            try:
                eradiate.set_mode(self._eradiate_mode)
            except Exception as e:
                raise ValueError(
                    f"Failed to set Eradiate mode '{self._eradiate_mode}': {e}. "
                    f"Check if this mode is supported in your Eradiate installation."
                )

    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available."""
        return ERADIATE_AVAILABLE

    def _get_material_ids_from_scene(
        self, scene_description: SceneDescription
    ) -> List[str]:
        """Get material IDs from SceneDescription metadata.

        Args:
            scene_description: SceneDescription object

        Returns:
            List of material IDs with "_mat_" prefix in texture index order
        """
        if not scene_description.material_indices:
            raise ValueError("SceneDescription must contain 'material_indices'")

        material_indices = scene_description.material_indices

        # Generate material IDs with "_mat_" prefix, ordered by texture index
        material_ids = []
        for texture_index in sorted(material_indices.keys(), key=int):
            material_name = material_indices[texture_index]
            material_ids.append(f"_mat_{material_name}")

        return material_ids

    @property
    def supported_platforms(self) -> List[str]:
        """Eradiate supports all platform types."""
        return ["satellite", "uav", "ground"]

    @property
    def supported_measurements(self) -> List[str]:
        """Measurement types supported by Eradiate."""
        return [
            "radiance",
            "brf",
            "hdrf",
            "bhr",
            "bhr_iso",
            "fapar",
            "flux_3d",
            "irradiance",
            "dhp",
        ]

    def validate_configuration(self) -> List[str]:
        """Validate configuration for Eradiate backend."""
        errors = super().validate_configuration()

        if not self.is_available():
            errors.append(
                "Eradiate is not available. Install with: pip install eradiate[kernel]"
            )

        valid_modes = {
            "mono",
            "ckd",
            "mono_polarized",
            "ckd_polarized",
            "mono_single",
            "mono_polarized_single",
            "mono_double",
            "mono_polarized_double",
            "ckd_single",
            "ckd_polarized_single",
            "ckd_double",
            "ckd_polarized_double",
            "none",
        }
        if self._eradiate_mode not in valid_modes:
            errors.append(
                f"Invalid Eradiate mode '{self._eradiate_mode}'. Valid modes: {sorted(valid_modes)}"
            )

        for sensor in self.simulation_config.sensors:
            if sensor.platform_type == PlatformType.SATELLITE:
                validation_error = self._validate_satellite_sensor(sensor)
                if validation_error:
                    errors.append(validation_error)
            elif sensor.platform_type == PlatformType.GROUND:
                if sensor.instrument not in [
                    GroundInstrumentType.HYPSTAR,
                    GroundInstrumentType.PERSPECTIVE_CAMERA,
                    GroundInstrumentType.PYRANOMETER,
                    GroundInstrumentType.FLUX_METER,
                    GroundInstrumentType.DHP_CAMERA,
                ]:
                    errors.append(
                        f"Ground sensor {sensor.id} instrument type {sensor.instrument} is not supported"
                    )

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity.value not in self.supported_measurements:
                errors.append(
                    f"Radiative quantity {rq.quantity} is not supported by Eradiate backend"
                )

            if rq.quantity in [
                MeasurementType.BRF,
                MeasurementType.HDRF,
                MeasurementType.RADIANCE,
            ]:
                if rq.viewing_zenith is None or rq.viewing_azimuth is None:
                    errors.append(
                        f"Radiative quantity {rq.quantity} requires viewing_zenith and viewing_azimuth"
                    )

        return errors

    def _validate_satellite_sensor(self, sensor) -> Optional[str]:
        """Validate satellite sensor platform/instrument/band combination using enum system."""
        from ..config import INSTRUMENT_BANDS, PLATFORM_INSTRUMENTS

        if sensor.platform == SatellitePlatform.CUSTOM:
            if sensor.srf is None:
                return "Custom platform requires explicit SRF configuration"
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

    def run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: Optional[UPath] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run Eradiate simulation from scene description."""
        if not self.is_available():
            raise RuntimeError("Eradiate is not available")

        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        print(f"Running Eradiate simulation: {self.simulation_config.name}")
        print(f"Sensors: {len(self.simulation_config.sensors)}")
        print(
            f"Radiative quantities: {len(self.simulation_config.radiative_quantities)}"
        )
        print(
            f"Measurement types: {[mt.value for mt in self.simulation_config.output_quantities]}"
        )

        experiment = self._create_experiment(scene_description, scene_dir)

        print("Executing Eradiate simulation...")
        eradiate.run(experiment)

        plot_image = kwargs.get("plot_image", False)
        if plot_image:
            self._create_rgb_visualization(
                experiment, output_dir, kwargs.get("id_to_plot", "rgb_camera")
            )

        return self._process_results(experiment, output_dir)

    def _process_object_materials(self, scene_description: SceneDescription) -> None:
        """Process object materials that are dict definitions and add them to scene materials.
        
        This method extracts material definitions from objects and adds them to the 
        scene materials for unified processing, avoiding duplication.
        
        Args:
            scene_description: Scene description with objects that may contain material dicts
        """
        from s2gos_utils.scene.materials import Material
        
        if not scene_description.objects:
            return
            
        for obj in scene_description.objects:
            if "material" not in obj:
                continue
                
            material = obj["material"]
            if isinstance(material, dict):
                # Generate consistent material name
                import sys
                import os
                sys.path.append('/home/gonzalezm/s2gos/s2gos/experimenting')
                from mitsuba_xml_parser import generate_material_name
                material_name = generate_material_name(material)
                
                # Add to scene materials if not already present (deduplication)
                if material_name not in scene_description.materials:
                    # Convert dict to Material object using existing pattern
                    material_obj = Material.from_dict(material, id=material_name)
                    scene_description.materials[material_name] = material_obj
                
                # Update object to use material reference instead of dict
                obj["material"] = material_name

    def _create_experiment(self, scene_description: SceneDescription, scene_dir: UPath):
        """Create Eradiate experiment from scene description."""
        kdict = {}
        kpmap = {}
        adapter = EradiateMaterialAdapter()

        # Process object materials that are dict definitions and add them to scene materials
        self._process_object_materials(scene_description)

        hamster_data_dict = self._get_hamster_data_for_scene(scene_description, scene_dir)
        for mat_name, material in scene_description.materials.items():
            # Use adapter pattern to create Eradiate-specific dictionaries
            from s2gos_utils.scene.materials import (
                BilambertianMaterial,
                DiffuseMaterial,
                OceanLegacyMaterial,
                RPVMaterial,
            )

            if isinstance(material, DiffuseMaterial):
                mat_kdict = adapter.create_diffuse_kdict(material)
                mat_kpmap = adapter.create_diffuse_kpmap(material)
            elif isinstance(material, BilambertianMaterial):
                mat_kdict = adapter.create_bilambertian_kdict(material)
                mat_kpmap = adapter.create_bilambertian_kpmap(material)
            elif isinstance(material, RPVMaterial):
                mat_kdict = adapter.create_rpv_kdict(material)
                mat_kpmap = adapter.create_rpv_kpmap(material)
            elif isinstance(material, OceanLegacyMaterial):
                mat_kdict = adapter.create_ocean_kdict(material)
                mat_kpmap = adapter.create_ocean_kpmap(material)
            else:
                mat_kdict = adapter.create_diffuse_kdict(material)
                mat_kpmap = adapter.create_diffuse_kpmap(material)

            kdict.update(mat_kdict)
            kpmap.update(mat_kpmap)
        
        # Create surface-specific HAMSTER baresoil materials when available
        if hamster_data_dict is not None:
            baresoil_material = scene_description.materials.get("baresoil")
            if baresoil_material:
                for surface_name, hamster_data in hamster_data_dict.items():
                    hamster_material_id = f"_mat_baresoil_{surface_name}"
                    
                    hamster_kdict = adapter.create_hamster_kdict(
                        material_id=hamster_material_id,
                        albedo_data=hamster_data
                    )

                    hamster_kpmap = adapter.create_hamster_kpmap(
                        material_id=hamster_material_id,
                        albedo_data=hamster_data
                    )
                    
                    kdict.update(hamster_kdict)  
                    kpmap.update(hamster_kpmap)

        # Get surface configurations from SceneDescription fields
        buffer = scene_description.buffer
        background = scene_description.background

        # Create surfaces from scene description
        hamster_available = hamster_data_dict is not None
        kdict.update(self._create_target_surface(scene_description, scene_dir, hamster_available))

        if buffer:
            kdict.update(self._create_buffer_surface(scene_description, scene_dir, hamster_available))

        if background:
            kdict.update(self._create_background_surface(scene_description, scene_dir, hamster_available))

        # Process 3D objects from scene description
        if scene_description.objects:
            for obj in scene_description.objects:
                object_mesh_path = scene_dir / obj["mesh"]
                obj_dict = {
                    "type": "ply",
                    "filename": str(object_mesh_path),
                    "id": obj["id"]
                }
                
                # Add material assignment
                if "material" in obj:
                    material = obj["material"]
                    # At this point, all materials should be string references due to preprocessing
                    if isinstance(material, str):
                        # String reference - use as material ID with _mat_ prefix (following existing pattern)
                        obj_dict["bsdf"] = {"type": "ref", "id": f"_mat_{material}"}
                    else:
                        # Fallback for any remaining edge cases
                        obj_dict["bsdf"] = {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.5}}
                
                # Add transformation (position + rotation + scale)
                if "position" in obj and "scale" in obj:
                    x, y, z = obj["position"] 
                    scale = obj["scale"]
                    
                    # Build transform: translate @ rotate @ scale
                    to_world = mi.ScalarTransform4f.translate([x, y, z])
                    
                    # Apply rotations if present (in degrees)
                    if "rotation" in obj:
                        rx, ry, rz = obj["rotation"]
                        if rx != 0:
                            to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], rx)
                        if ry != 0:
                            to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], ry)
                        if rz != 0:
                            to_world = to_world @ mi.ScalarTransform4f.rotate([0, 0, 1], rz)
                    
                    to_world = to_world @ mi.ScalarTransform4f.scale(scale)
                    obj_dict["to_world"] = to_world
                
                kdict[obj["id"]] = obj_dict

        atmosphere = self._create_atmosphere_from_config(scene_description)

        illumination = self._translate_illumination()

        measures = self._translate_sensors()

        geometry = self._create_geometry_from_atmosphere(scene_description)

        return AtmosphereExperiment(
            geometry=geometry,
            atmosphere=atmosphere,
            surface=None,
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap,
        )

    def _create_geometry_from_atmosphere(self, scene_description: SceneDescription):
        """Create geometry with bounds matching the atmosphere configuration."""
        atmosphere = scene_description.atmosphere
        toa = (
            atmosphere["toa"] if "toa" in atmosphere else 40000.0
        )  # Top of atmosphere (meters)

        geometry = {
            "type": "plane_parallel",
            "toa_altitude": toa,
        }

        return geometry

    def _create_atmosphere_from_config(self, scene_description: SceneDescription):
        """Create atmosphere based on scene description format."""
        atmosphere = scene_description.atmosphere
        atmosphere_type = atmosphere["type"] if "type" in atmosphere else None

        if not atmosphere_type:
            raise ValueError("Atmosphere configuration must specify 'type' field")

        if atmosphere_type == "molecular":
            return self._create_molecular_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "homogeneous":
            return self._create_homogeneous_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "heterogeneous":
            return self._create_heterogeneous_atmosphere_from_scene(atmosphere)
        else:
            raise ValueError(f"Unknown atmosphere type: {atmosphere_type}")

    def _create_molecular_atmosphere_from_dict(self, mol_dict):
        """Create molecular atmosphere directly from scene description data."""
        thermoprops_id = mol_dict.get("thermoprops_identifier", "afgl_1986-us_standard")
        altitude_min = mol_dict.get("altitude_min", 0.0)
        altitude_max = mol_dict.get("altitude_max", 120000.0)
        num_steps = (
            (altitude_max - altitude_min) / mol_dict.get("altitude_step", 1000)
        ) + 1

        atmosphere = MolecularAtmosphere(
            thermoprops={
                "identifier": thermoprops_id,
                "z": np.linspace(altitude_min, altitude_max, int(num_steps)) * ureg.m,
            },
            has_absorption=mol_dict.get("has_absorption", True),
            has_scattering=mol_dict.get("has_scattering", True),
        )

        return atmosphere

    def _create_particle_layer_from_dict(self, layer_dict):
        """Create particle layer directly from scene description data."""
        dist_type = layer_dict.get("distribution_type", "exponential")
        if dist_type == "exponential":
            if "rate" in layer_dict.keys():
                if "scale" in layer_dict.keys():
                    print(
                        "WARNING: scale and rate should be mutually exclusive in exponential distribution, using rate"
                    )
                distribution = ExponentialParticleDistribution(
                    scale=layer_dict.get("rate", 5.0)
                )
            else:
                distribution = ExponentialParticleDistribution(
                    rate=layer_dict.get("scale", 0.2)
                )
        elif dist_type == "gaussian":
            distribution = GaussianParticleDistribution(
                mean=layer_dict.get("center_altitude", 0.5),
                std=layer_dict.get("width", 1 / 6),
            )
        else:
            distribution = UniformParticleDistribution(
                {"bounds": layer_dict.get("bounds", [0, 1])}
            )

        # Create particle layer using direct Eradiate API
        layer = ParticleLayer(
            dataset=layer_dict["aerosol_dataset"],
            tau_ref=layer_dict["optical_thickness"],
            w_ref=layer_dict.get("reference_wavelength", 550.0),
            bottom=layer_dict["altitude_bottom"],
            top=layer_dict["altitude_top"],
            distribution=distribution,
            has_absorption=layer_dict.get("has_absorption", True),
        )

        return layer

    def _create_molecular_atmosphere_from_scene(self, atmosphere_dict):
        """Create molecular atmosphere from scene description."""
        if "molecular_atmosphere" in atmosphere_dict:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            return self._create_molecular_atmosphere_from_dict(mol_dict)
        else:
            # Default molecular atmosphere
            return self._create_molecular_atmosphere_from_dict({})

    def _create_homogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create homogeneous atmosphere from scene description."""
        atmosphere = HomogeneousAtmosphere(
            boa=atmosphere_dict.get("boa", 0.0),
            toa=atmosphere_dict.get("toa", 40000.0),
            particle_layers=[
                ParticleLayer(
                    dataset=atmosphere_dict.get("aerosol_ds", "sixsv-continental"),
                    optical_thickness=atmosphere_dict.get("aerosol_ot", 0.1),
                    altitude_bottom=atmosphere_dict.get("boa", 0.0),
                    altitude_top=atmosphere_dict.get("toa", 40000.0),
                    reference_wavelength=550.0,
                )
            ],
        )

        return atmosphere

    def _create_heterogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create heterogeneous atmosphere from scene description."""
        has_molecular = (
            atmosphere_dict.get("has_molecular_atmosphere", False)
            or "molecular_atmosphere" in atmosphere_dict
        )
        has_particles = (
            atmosphere_dict.get("has_particle_layers", False)
            or "particle_layers" in atmosphere_dict
        )

        molecular_atmosphere = None
        particle_layers = []

        if has_molecular:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            molecular_atmosphere = self._create_molecular_atmosphere_from_dict(mol_dict)

        if has_particles:
            for layer_dict in atmosphere_dict["particle_layers"]:
                layer = self._create_particle_layer_from_dict(layer_dict)
                if layer:
                    particle_layers.append(layer)

        # Create heterogeneous atmosphere using direct Eradiate API
        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=molecular_atmosphere, particle_layers=particle_layers
        )

        return atmosphere

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
                    "dataset": illumination.irradiance_dataset,
                },
            }
        elif isinstance(illumination, ConstantIllumination):
            return {
                "type": "constant",
                "id": illumination.id,
                "radiance": illumination.radiance,
            }
        else:
            raise ValueError(f"Unsupported illumination type: {type(illumination)}")

    def _calculate_target_from_angles(
        self, view: AngularFromOriginViewing
    ) -> tuple[list[float], list[float]]:
        """
        Calculates a target point and direction vector from an AngularFromOriginViewing object.

        Returns:
            A tuple containing (target_position, direction_vector).
        """
        zen_rad = np.deg2rad(view.zenith)
        az_rad = np.deg2rad(view.azimuth)

        direction = np.array(
            [
                np.sin(zen_rad) * np.cos(az_rad),
                np.sin(zen_rad) * np.sin(az_rad),
                np.cos(zen_rad),
            ]
        )

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
            "angles": [sensor.viewing.zenith, sensor.viewing.azimuth],
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

            if sensor.instrument in [
                GroundInstrumentType.PERSPECTIVE_CAMERA,
                GroundInstrumentType.DHP_CAMERA,
            ]:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = [1024, 1024]
                base_measure["fov"] = (
                    180.0
                    if sensor.instrument == GroundInstrumentType.DHP_CAMERA
                    else 70.0
                )
                base_measure["up"] = view.up or [0, 0, 1]
            else:
                base_measure["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_measure["target"] = view.target

            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_measure["target"] = target
        else:
            raise ValueError(
                f"Unsupported viewing type for ground sensor: {type(view)}"
            )

        return base_measure

    def _translate_radiative_quantity(self, rad_quantity) -> Dict[str, Any]:
        """Translate radiative quantity configuration to placeholder measure.

        TODO: This is a placeholder implementation. Future versions will:
        1. Determine appropriate sensors needed for the radiative quantity
        2. Instantiate those sensors internally
        3. Calculate the requested quantity from sensor results
        """

        quantity_id = f"{rad_quantity.quantity.value}_measure"

        # Log TODO message for user clarity
        print(
            f"TODO: {rad_quantity.quantity.value.upper()} calculation not yet implemented - generating placeholder"
        )

        base_config = {
            "id": quantity_id,
            "type": "distant",
            "construct": "from_angles",
            "angles": [0.0, 0.0],  # Nadir view as placeholder
            "spp": rad_quantity.samples_per_pixel,
            "srf": self._translate_srf(rad_quantity.srf),
        }

        return base_config

    def _translate_srf(self, srf) -> Union[Dict[str, Any], str]:
        """Translate generic SRF to Eradiate format."""
        if srf is None:
            return {"type": "uniform", "wmin": 400.0, "wmax": 700.0, "value": 1.0}
        elif isinstance(srf, str):
            return srf
        elif isinstance(srf, dict):
            return srf
        else:
            # From config
            if srf.type == "delta":
                return {"type": "delta", "wavelengths": srf.wavelengths}
            elif srf.type == "uniform":
                return {
                    "type": "uniform",
                    "wmin": srf.wmin,
                    "wmax": srf.wmax,
                    "value": 1.0,
                }
            elif srf.type == "dataset":
                return srf.dataset_id
            elif srf.type == "custom":
                if srf.data and "wavelengths" in srf.data and "values" in srf.data:
                    return {
                        "type": "array",
                        "wavelengths": srf.data["wavelengths"],
                        "values": srf.data["values"],
                    }
                else:
                    raise ValueError(
                        "Custom SRF requires 'wavelengths' and 'values' in data"
                    )
            else:
                raise ValueError(f"Unsupported SRF type: {srf.type}")

    def _resolve_platform_srf(self, platform: str, instrument: str, band: str) -> str:
        """
        Resolve platform/instrument/band combination to Eradiate SRF identifier.

        This method converts platform identifiers to Eradiate dataset identifiers.
        """
        platform_norm = platform.lower().replace("-", "_")
        instrument_norm = instrument.lower()
        band_norm = band.lower()

        srf_id = f"{platform_norm}-{instrument_norm}-{band_norm}"

        return srf_id

    def _create_output_metadata(self, output_dir: UPath) -> Dict[str, Any]:
        """Create standardized metadata for output files."""
        return {
            "simulation_name": self.simulation_config.name,
            "description": self.simulation_config.description,
            "created_at": self.simulation_config.created_at.isoformat(),
            "backend": "eradiate",
            "output_dir": str(output_dir),
            "num_sensors": len(self.simulation_config.sensors),
            "num_radiative_quantities": len(
                self.simulation_config.radiative_quantities
            ),
            "sensor_types": [
                s.platform_type.value for s in self.simulation_config.sensors
            ],
            "radiative_quantities": [
                rq.quantity.value for rq in self.simulation_config.radiative_quantities
            ],
            "illumination_type": self.simulation_config.illumination.type,
        }


    def _create_target_surface(
        self, scene_description: SceneDescription, scene_dir: UPath, hamster_available: bool = False
    ) -> Dict[str, Any]:
        """Create target surface from SceneDescription."""
        target_config = scene_description.target
        target_mesh_path = scene_dir / target_config["mesh"]
        target_texture_path = scene_dir / target_config["selection_texture"]

        with open_file(target_texture_path, "rb") as f:
            texture_image = Image.open(f)
            texture_image.load()
        selection_texture_data = np.array(texture_image)
        selection_texture_data = np.atleast_3d(selection_texture_data)

        material_ids = self._get_material_ids_from_scene(scene_description)
        
        if hamster_available:
            material_ids = [
                "_mat_baresoil_target" if mat_id == "_mat_baresoil" else mat_id
                for mat_id in material_ids
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
                **{
                    f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
                    for i, mat_id in enumerate(material_ids)
                },
            },
            "terrain": {
                "type": "ply",
                "filename": str(target_mesh_path),
                "bsdf": {"type": "ref", "id": "terrain_material"},
                "id": "terrain",
            },
        }

    def _create_buffer_surface(
        self, scene_description: SceneDescription, scene_dir: UPath, hamster_available: bool = False
    ) -> Dict[str, Any]:
        """Create buffer surface from SceneDescription."""
        buffer_config = scene_description.buffer
        buffer_mesh_path = scene_dir / buffer_config["mesh"]
        buffer_texture_path = scene_dir / buffer_config["selection_texture"]
        mask_path = (
            scene_dir / buffer_config["mask_texture"]
            if "mask_texture" in buffer_config
            else None
        )

        with open_file(buffer_texture_path, "rb") as f:
            buffer_texture_image = Image.open(f)
            buffer_texture_image.load()
        buffer_selection_texture_data = np.array(buffer_texture_image)
        buffer_selection_texture_data = np.atleast_3d(buffer_selection_texture_data)

        material_ids = self._get_material_ids_from_scene(scene_description)
        
        if hamster_available:
            material_ids = [
                "_mat_baresoil_buffer" if mat_id == "_mat_baresoil" else mat_id
                for mat_id in material_ids
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
                **{
                    f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
                    for i, mat_id in enumerate(material_ids)
                },
            }
        }

        buffer_bsdf_id = "buffer_material"

        from s2gos_utils.io.paths import exists

        if mask_path and exists(mask_path):
            with open_file(mask_path, "rb") as f:
                mask_image = Image.open(f)
                mask_image.load()
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
                "material": {"type": "ref", "id": "buffer_material"},
            }
            buffer_bsdf_id = "buffer_mask"

        result["buffer_terrain"] = {
            "type": "ply",
            "filename": str(buffer_mesh_path),
            "bsdf": {"type": "ref", "id": buffer_bsdf_id},
            "id": "buffer_terrain",
        }

        return result

    def _create_background_surface(
        self, scene_description: SceneDescription, scene_dir: UPath, hamster_available: bool = False
    ) -> Dict[str, Any]:
        """Create background surface from SceneDescription."""
        background_config = scene_description.background
        elevation = background_config["elevation"]
        background_selection_texture_path = (
            scene_dir / background_config["selection_texture"]
        )
        background_size_km = background_config["size_km"]

        with open_file(background_selection_texture_path, "rb") as f:
            background_texture_image = Image.open(f)
            background_texture_image.load()
        background_selection_texture_data = np.array(background_texture_image)
        background_selection_texture_data = np.atleast_3d(
            background_selection_texture_data
        )

        material_ids = self._get_material_ids_from_scene(scene_description)
        
        if hamster_available:
            material_ids = [
                "_mat_baresoil_background" if mat_id == "_mat_baresoil" else mat_id
                for mat_id in material_ids
            ]

        result = {
            "background_material": {
                "type": "selectbsdf",
                "id": "background_material",
                "indices": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": background_selection_texture_data,
                },
                **{
                    f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
                    for i, mat_id in enumerate(material_ids)
                },
            }
        }

        scale_factor = (background_size_km * 1000) / 2.0

        to_world = mi.ScalarTransform4f.translate(
            [0, 0, elevation]
        ) @ mi.ScalarTransform4f.scale(scale_factor)

        result["background_surface"] = {
            "type": "rectangle",
            "to_world": to_world,
            "bsdf": {"type": "ref", "id": "background_material"},
            "id": "background_surface",
        }

        return result

    def _create_rgb_visualization(self, experiment, output_dir: UPath, id_to_plot: str):
        """Create RGB visualization from camera results."""
        try:
            if id_to_plot not in experiment.results:
                print(f"Warning: Sensor '{id_to_plot}' not found in results")
                return

            sensor_data = experiment.results[id_to_plot]

            if "radiance" in sensor_data:
                radiance_data = sensor_data["radiance"]

                if "x_index" in radiance_data.dims and "y_index" in radiance_data.dims:
                    img = (
                        dataarray_to_rgb(
                            radiance_data,
                            channels=[("w", 660), ("w", 550), ("w", 440)],
                            normalize=False,
                        )
                        * 1.8
                    )

                    img = np.clip(img, 0, 1)

                    rgb_output = output_dir / f"{id_to_plot}_rgb.png"
                    plt_img = (img * 255).astype(np.uint8)
                    rgb_image = Image.fromarray(plt_img)
                    with open_file(rgb_output, "wb") as f:
                        rgb_image.save(f, format="PNG")

                    print(f"Camera RGB image saved to: {rgb_output}")

                else:
                    spectral_output = output_dir / f"{id_to_plot}_spectrum.png"
                    self._plot_spectral_data(radiance_data, spectral_output)
                    print(f"Spectral data plot saved to: {spectral_output}")

        except Exception as e:
            print(f"Warning: Could not create visualization for {id_to_plot}: {e}")

    def _plot_spectral_data(self, radiance_data, output_path: UPath):
        """Plot spectral data for point sensors."""
        try:
            import matplotlib.pyplot as plt

            wavelengths = radiance_data.coords["w"].values
            radiance_values = radiance_data.values

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, radiance_values, "b-", linewidth=2)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Radiance")
            plt.title("Spectral Radiance")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            with open_file(output_path, "wb") as f:
                plt.savefig(f, format="png", dpi=150, bbox_inches="tight")
            plt.close()

        except ImportError:
            print("Warning: matplotlib not available for spectral plotting")
        except Exception as e:
            print(f"Warning: Could not create spectral plot: {e}")

    def _process_results(self, experiment, output_dir: UPath) -> xr.Dataset:
        """Process and save simulation results."""
        results = experiment.results

        if not results:
            raise ValueError("No results found in experiment")

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        metadata = self._create_output_metadata(output_dir)

        # Process regular sensor results
        if isinstance(results, dict):
            for sensor_id, dataset in results.items():
                sensor_output = (
                    output_dir / f"{self.simulation_config.name}_{sensor_id}.zarr"
                )

                dataset.attrs.update(metadata)
                dataset.attrs["sensor_id"] = sensor_id

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
            dummy_output = self._create_dummy_radiative_quantity_result(
                rad_quantity, output_dir, metadata
            )
            print(
                f"TODO: {rad_quantity.quantity.value.upper()} placeholder saved to {dummy_output}"
            )

        return results

    def _get_hamster_data_for_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Optional[dict]:
        """Load HAMSTER albedo data from zarr files referenced in scene description areas.
        
        Args:
            scene_description: Scene description with HAMSTER data file paths in area sections
            scene_dir: Directory containing the scene description file (for resolving relative paths)
            
        Returns:
            Dict with loaded HAMSTER albedo DataArrays for each surface area, or None
            Format: {'target': target_subset, 'buffer': buffer_subset, 'background': bg_subset}
        """
        try:
            hamster_data_files = {}
            
            if scene_description.target and 'hamster_data' in scene_description.target:
                hamster_data_files['target'] = scene_description.target['hamster_data']
                
            if scene_description.buffer and 'hamster_data' in scene_description.buffer:
                hamster_data_files['buffer'] = scene_description.buffer['hamster_data']
                
            if scene_description.background and 'hamster_data' in scene_description.background:
                hamster_data_files['background'] = scene_description.background['hamster_data']
                
            if not hamster_data_files:
                import logging
                logging.info("No HAMSTER data files found in scene description areas")
                return None
            
            hamster_data = {}
            base_path = scene_dir
            
            import logging
            import xarray as xr
            from s2gos_utils.io.paths import exists
            
            for area, relative_path in hamster_data_files.items():
                file_path = base_path / relative_path
                if not exists(file_path):
                    logging.warning(f"HAMSTER data file not found: {file_path}, skipping {area} area")
                    continue
                    
                try:
                    dataset = xr.open_zarr(file_path)
                    data_vars = list(dataset.data_vars.keys())
                    if not data_vars:
                        logging.warning(f"No data variables found in HAMSTER file: {file_path}")
                        continue
                        
                    albedo_data = dataset[data_vars[0]]
                    hamster_data[area] = albedo_data
                    logging.info(f"Loaded HAMSTER data for {area} area from {file_path}: {albedo_data.sizes}")
                    
                except Exception as e:
                    logging.warning(f"Failed to load HAMSTER data from {file_path}: {e}")
                    continue
            
            if hamster_data:
                logging.info(f"Successfully loaded HAMSTER data for {len(hamster_data)} surface areas")
                return hamster_data
            else:
                logging.warning("No HAMSTER data could be loaded from any files")
                return None
            
        except Exception as e:
            import logging
            logging.warning(f"Could not load HAMSTER data: {e}, falling back to standard baresoil")
            return None

    def _create_dummy_radiative_quantity_result(
        self, rad_quantity, output_dir: UPath, metadata: dict
    ) -> UPath:
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

        wavelengths = [550.0]  # Set a default value
        srf = rad_quantity.srf
        if srf.type == "delta" and srf.wavelengths:
            wavelengths = srf.wavelengths

        # Create xarray dataset with appropriate structure
        if len(wavelengths) > 1:
            # Multi-spectral data
            dummy_values = np.stack([dummy_data for _ in wavelengths], axis=0)
            coords = {
                "wavelength": ("wavelength", wavelengths),
                "x": ("x", np.arange(10)),
                "y": ("y", np.arange(10)),
            }
            dims = ["wavelength", "y", "x"]
        else:
            # Single wavelength
            dummy_values = dummy_data
            coords = {"x": ("x", np.arange(10)), "y": ("y", np.arange(10))}
            dims = ["y", "x"]

        # Create dataset
        dummy_dataset = xr.Dataset(
            {rad_quantity.quantity.value: (dims, dummy_values)}, coords=coords
        )

        # Add metadata
        dummy_dataset.attrs.update(metadata)
        dummy_dataset.attrs.update(
            {
                "radiative_quantity": rad_quantity.quantity.value,
                "status": "TODO_PLACEHOLDER",
                "description": f"Placeholder data for {rad_quantity.quantity.value} calculation",
                "samples_per_pixel": rad_quantity.samples_per_pixel,
                "viewing_zenith": rad_quantity.viewing_zenith,
                "viewing_azimuth": rad_quantity.viewing_azimuth,
            }
        )

        # Save as Zarr
        dummy_dataset.to_zarr(dummy_output, mode="w")

        return dummy_output
