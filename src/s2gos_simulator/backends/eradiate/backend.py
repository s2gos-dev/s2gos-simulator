"""Refactored Eradiate backend for S2GOS simulator.

This is the main orchestration layer that coordinates the specialized modules
for geometry, atmosphere, surfaces, sensors, and result processing.
"""

import logging
from typing import List, Optional

import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .atmosphere_builder import AtmosphereBuilder
from .constants import VALID_ERADIATE_MODES
from .geometry_utils import GeometryUtils
from .result_processor import ResultProcessor
from .sensor_translator import SensorTranslator
from .surface_builder import SurfaceBuilder
from ..base import SimulationBackend
from ...config import (
    GroundInstrumentType,
    HCRFConfig,
    HDRFConfig,
    PlatformType,
    SimulationConfig,
)

try:
    import eradiate
    from eradiate.experiments import AtmosphereExperiment

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class EradiateBackend(SimulationBackend):
    """Enhanced Eradiate backend with modular architecture.

    This backend orchestrates specialized modules for different aspects of
    the simulation: geometry, atmosphere, surfaces, sensors, and results.
    """

    def __init__(self, simulation_config: SimulationConfig):
        """Initialize the Eradiate backend with modular components.

        Args:
            simulation_config: Simulation configuration
        """
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

        self.geometry_utils = GeometryUtils()
        self.atmosphere_builder = AtmosphereBuilder()
        self.surface_builder = SurfaceBuilder()
        self.sensor_translator = SensorTranslator(
            simulation_config, self.geometry_utils
        )
        self.result_processor = ResultProcessor(simulation_config)

        self._current_scene_dir = None
        self._current_scene_description = None

    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available.

        Returns:
            True if Eradiate is available, False otherwise
        """
        return ERADIATE_AVAILABLE

    @property
    def supported_platforms(self) -> List[str]:
        """Get list of supported platform types.

        Returns:
            List of platform type strings
        """
        return ["satellite", "uav", "ground"]

    @property
    def supported_measurements(self) -> List[str]:
        """Get list of supported measurement types.

        Returns:
            List of measurement type strings
        """
        return [
            "radiance",
            "brf",
            "hdrf",
            "bhr_iso",
            "boa_irradiance",
        ]

    def validate_configuration(self) -> List[str]:
        """Validate configuration for Eradiate backend.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate_configuration()

        if not self.is_available():
            errors.append(
                "Eradiate is not available. Install with: pip install eradiate[kernel]"
            )

        if self._eradiate_mode not in VALID_ERADIATE_MODES:
            errors.append(
                f"Invalid Eradiate mode '{self._eradiate_mode}'. "
                f"Valid modes: {sorted(VALID_ERADIATE_MODES)}"
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
                    GroundInstrumentType.RADIANCEMETER,
                ]:
                    errors.append(
                        f"Ground sensor {sensor.id} instrument type {sensor.instrument} is not supported"
                    )

        for measurement in self.simulation_config.measurements:
            if measurement.type not in self.supported_measurements:
                errors.append(
                    f"Measurement type '{measurement.type}' is not supported by Eradiate backend"
                )

        return errors

    def _validate_satellite_sensor(self, sensor) -> Optional[str]:
        """Validate satellite sensor platform/instrument/band combination.

        Args:
            sensor: Satellite sensor configuration

        Returns:
            Error message if invalid, None if valid
        """
        from ...config import (
            INSTRUMENT_BANDS,
            PLATFORM_INSTRUMENTS,
            SatelliteInstrument,
            SatellitePlatform,
        )

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
            return (
                f"Platform '{sensor.platform}' does not support instrument '{sensor.instrument}'. "
                f"Valid instruments: {[inst.value for inst in valid_instruments]}"
            )

        band_enum_class = INSTRUMENT_BANDS.get(instrument_enum)
        if band_enum_class is not None:
            try:
                band_enum_class(sensor.band)
            except ValueError:
                valid_bands = [band.value for band in band_enum_class]
                return (
                    f"Instrument '{sensor.platform}/{sensor.instrument}' does not support band '{sensor.band}'. "
                    f"Valid bands: {valid_bands}"
                )

        return None

    def run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: Optional[UPath] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run Eradiate simulation from scene description.

        Automatically detects HDRF/HCRF measurements and executes appropriate workflow.

        Args:
            scene_description: Scene description from s2gos_generator
            scene_dir: Directory containing scene assets
            output_dir: Output directory (defaults to scene_dir/eradiate_renders)
            **kwargs: Additional options (plot_image, id_to_plot, etc.)

        Returns:
            xarray Dataset containing all simulation results
        """
        if not self.is_available():
            raise RuntimeError("Eradiate is not available")

        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        print(f"Eradiate simulation: {self.simulation_config.name}")
        print(f"  Sensors: {len(self.simulation_config.sensors)}")
        print(f"  Measurements: {len(self.simulation_config.measurements)}\n")

        from ...hdrf_processor import HDRFProcessor
        from ...irradiance_processor import IrradianceProcessor

        hdrf_proc = HDRFProcessor(self)
        irr_proc = IrradianceProcessor(self)
        requires_hdrf = hdrf_proc.requires_hdrf()
        requires_irr = irr_proc.requires_irradiance()

        if requires_hdrf or requires_irr:
            print("Workflow: Standard + BOA Irradiance")
            return self._run_combined_workflow(
                scene_description, scene_dir, output_dir, irr_proc, **kwargs
            )
        else:
            print("Workflow: Standard")
            return self._run_standard_simulation(
                scene_description, scene_dir, output_dir, **kwargs
            )

    def _run_standard_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        include_irradiance_measures: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Run standard simulation (BRF, radiance, etc.).

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory
            include_irradiance_measures: Whether to include irradiance measurements
            **kwargs: Additional options

        Returns:
            Simulation results dataset
        """
        experiment = self._create_experiment(
            scene_description, scene_dir, include_irradiance_measures
        )

        for i in range(len(experiment.measures)):
            measure_id = getattr(experiment.measures[i], "id", f"measure_{i}")
            print(f"  Measure {i + 1}/{len(experiment.measures)}: {measure_id}")
            eradiate.run(experiment, measures=i)

        if kwargs.get("plot_image", False):
            self.result_processor.create_rgb_visualization(
                experiment, output_dir, kwargs.get("id_to_plot", "rgb_camera")
            )

        return self.result_processor.process_results(experiment, output_dir)

    def _run_combined_workflow(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        irradiance_processor,
        **kwargs,
    ) -> xr.Dataset:
        """Execute combined workflow with dual simulation for HDRF/HCRF.

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory
            irradiance_processor: IrradianceProcessor instance
            **kwargs: Additional options

        Returns:
            Combined results dataset
        """
        print("\n[1/3] Radiance simulation (actual scene)...")
        actual_results = self._run_standard_simulation(
            scene_description,
            scene_dir,
            output_dir / "radiance",
            include_irradiance_measures=False,
            **kwargs,
        )

        print("\n[1.5/3] Applying post-processing to sensor results...")
        actual_dict = (
            actual_results
            if isinstance(actual_results, dict)
            else {"results": actual_results}
        )
        actual_dict = self.sensor_translator.apply_post_processing_to_sensors(
            actual_dict
        )

        print("\n[2/3] BOA irradiance measurements (white disk)...")
        irr_results = irradiance_processor.execute_irradiance_measurements(
            scene_description, scene_dir, output_dir / "boa_irradiance"
        )

        combined_results = {**actual_dict, **irr_results}

        derived_results = self.sensor_translator.compute_derived_measurements(
            combined_results
        )
        derived_output_dir = output_dir / "derived_results"
        for derived_name, dataset in derived_results.items():
            derived_output = (
                derived_output_dir
                / f"{self.simulation_config.name}_{derived_name}.zarr"
            )

            dataset.attrs["id"] = derived_name

            dataset.to_zarr(derived_output, mode="w")
            print(f"Derived measure '{derived_name}' saved to {derived_output}")

        postprocessed_results = {**combined_results, **derived_results}
        print(postprocessed_results.keys())
        print(
            f"\n✓ Combined workflow complete: {len(postprocessed_results)} total datasets"
        )
        return postprocessed_results

    def _create_experiment(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
    ):
        """Create Eradiate experiment from scene description.

        Args:
            scene_description: Scene description with all scene data
            scene_dir: Base directory for resolving file paths
            include_irradiance_measures: Whether to include irradiance measurements

        Returns:
            AtmosphereExperiment configured for the scene
        """
        self._current_scene_dir = scene_dir
        self._current_scene_description = scene_description

        # Auto-generate sensors for HDRF/HCRF if needed
        generated_sensor_ids = self._auto_generate_sensors_for_measurements()
        if generated_sensor_ids:
            print(f"  Auto-generated {len(generated_sensor_ids)} sensors:")
            for sensor_id in generated_sensor_ids:
                print(f"    - {sensor_id}")

        self._validate_object_materials(scene_description)
        kdict, kpmap = self.surface_builder.translate_materials(
            scene_description, scene_dir
        )

        hamster_data_dict = self._get_hamster_data_for_scene(
            scene_description, scene_dir
        )
        self.surface_builder.add_hamster_materials(
            kdict, kpmap, scene_description, hamster_data_dict
        )

        hamster_available = hamster_data_dict is not None
        kdict.update(
            self.surface_builder.create_target_surface(
                scene_description, scene_dir, hamster_available
            )
        )

        if scene_description.buffer:
            kdict.update(
                self.surface_builder.create_buffer_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        if scene_description.background:
            kdict.update(
                self.surface_builder.create_background_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        self.surface_builder.add_scene_objects(kdict, scene_description, scene_dir)

        atmosphere = self.atmosphere_builder.create_atmosphere_from_config(
            scene_description
        )

        illumination = self.sensor_translator.translate_illumination()

        measures = self.sensor_translator.translate_sensors(
            scene_description, scene_dir, include_irradiance_measures
        )

        geometry = self.atmosphere_builder.create_geometry_from_atmosphere(
            scene_description
        )

        self.surface_builder.validate_material_ids(kdict, scene_description)

        return AtmosphereExperiment(
            geometry=geometry,
            atmosphere=atmosphere,
            surface=None,  # We set this up through the expert interface (kdict, kpmap)
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap,
        )

    def _validate_object_materials(self, scene_description: SceneDescription) -> None:
        """Validate that all object materials use string references and exist.

        Args:
            scene_description: Scene description with objects

        Raises:
            ValueError: If any object has invalid material reference
        """
        if not scene_description.objects:
            return

        for i, obj in enumerate(scene_description.objects):
            if "material" not in obj:
                continue

            material = obj["material"]

            if not isinstance(material, str):
                raise ValueError(
                    f"Object {i} has invalid material type {type(material).__name__}. "
                    "Only string references are supported. "
                    "Define materials in the scene's material library."
                )

            if material not in scene_description.materials:
                available = list(scene_description.materials.keys())
                raise ValueError(
                    f"Object {i} references unknown material '{material}'. "
                    f"Available materials: {available}"
                )

    def _auto_generate_sensors_for_measurements(self) -> List[str]:
        """Auto-generate sensors for measurements that require them.

        Returns:
            List of generated sensor IDs
        """
        generated_sensor_ids = []

        for measurement in self.simulation_config.measurements:
            if (
                isinstance(measurement, HDRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                radiance_sensor = (
                    self.sensor_translator.generate_radiance_sensor_for_hdrf(
                        measurement
                    )
                )
                irradiance_measurement = (
                    self.sensor_translator.generate_irradiance_measurement_for_hdrf(
                        measurement
                    )
                )

                self.simulation_config.sensors.append(radiance_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = radiance_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(radiance_sensor.id)
                logging.info(
                    f"Auto-generated radiance sensor '{radiance_sensor.id}' and "
                    f"irradiance measurement '{irradiance_measurement.id}' for HDRF '{measurement.id}'"
                )

            elif (
                isinstance(measurement, HCRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                camera_sensor = self.sensor_translator.generate_camera_sensor_for_hcrf(
                    measurement
                )
                irradiance_measurement = (
                    self.sensor_translator.generate_irradiance_measurement_for_hcrf(
                        measurement
                    )
                )

                self.simulation_config.sensors.append(camera_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = camera_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(camera_sensor.id)
                logging.info(
                    f"Auto-generated camera sensor '{camera_sensor.id}' and "
                    f"irradiance measurement '{irradiance_measurement.id}' for HCRF '{measurement.id}'"
                )

        return generated_sensor_ids

    def _get_material_ids_from_scene(
        self, scene_description: SceneDescription
    ) -> List[str]:
        """Get material IDs from SceneDescription metadata.

        Args:
            scene_description: SceneDescription object

        Returns:
            List of material IDs in texture index order
        """
        material_indices = scene_description.material_indices

        material_ids = []
        for texture_index in sorted(material_indices.keys(), key=int):
            material_name = material_indices[texture_index]
            material_ids.append(material_name)

        return material_ids

    def _get_hamster_data_for_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Optional[dict]:
        """Load HAMSTER albedo data from zarr files.

        Args:
            scene_description: Scene description with HAMSTER data paths
            scene_dir: Directory containing scene files

        Returns:
            Dict with HAMSTER albedo DataArrays for each surface, or None
        """
        try:
            hamster_data_files = {}

            if scene_description.target and "hamster_data" in scene_description.target:
                hamster_data_files["target"] = scene_description.target["hamster_data"]

            if scene_description.buffer and "hamster_data" in scene_description.buffer:
                hamster_data_files["buffer"] = scene_description.buffer["hamster_data"]

            if (
                scene_description.background
                and "hamster_data" in scene_description.background
            ):
                hamster_data_files["background"] = scene_description.background[
                    "hamster_data"
                ]

            if not hamster_data_files:
                logging.info("No HAMSTER data files found in scene description")
                return None

            hamster_data = {}
            base_path = scene_dir

            import xarray as xr
            from s2gos_utils.io.paths import exists

            for area, relative_path in hamster_data_files.items():
                file_path = base_path / relative_path
                if not exists(file_path):
                    logging.warning(
                        f"HAMSTER data file not found: {file_path}, skipping {area}"
                    )
                    continue

                try:
                    dataset = xr.open_zarr(file_path)
                    data_vars = list(dataset.data_vars.keys())
                    if not data_vars:
                        logging.warning(
                            f"No data variables in HAMSTER file: {file_path}"
                        )
                        continue

                    albedo_data = dataset[data_vars[0]]
                    hamster_data[area] = albedo_data
                    logging.info(
                        f"Loaded HAMSTER data for {area} from {file_path}: {albedo_data.sizes}"
                    )

                except Exception as e:
                    logging.warning(
                        f"Failed to load HAMSTER data from {file_path}: {e}"
                    )
                    continue

            if hamster_data:
                logging.info(
                    f"Successfully loaded HAMSTER data for {len(hamster_data)} surface areas"
                )
                return hamster_data
            else:
                logging.warning("No HAMSTER data could be loaded")
                return None

        except Exception as e:
            logging.warning(f"Could not load HAMSTER data: {e}")
            return None
