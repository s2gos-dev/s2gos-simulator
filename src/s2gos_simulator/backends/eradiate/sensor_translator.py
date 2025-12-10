"""Sensor and measurement translation for Eradiate backend.

This module handles translation of generic sensor configurations to Eradiate-specific
measure definitions, including post-processing and derived measurement computation.
"""

from typing import Any, Dict, List, Union

import numpy as np
import xarray as xr
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .constants import IRRADIANCE_VARIABLE_NAMES, RADIANCE_VARIABLE_NAMES
from .geometry_utils import sanitize_sensor_id
from ...config import (
    AngularFromOriginViewing,
    ConstantIllumination,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HDRFConfig,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    SatelliteSensor,
    UAVInstrumentType,
    UAVSensor,
)

try:
    from eradiate.units import unit_registry as ureg

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class SensorTranslator:
    """Translator for sensors, measurements, and derived quantities."""

    def __init__(self, simulation_config, geometry_utils):
        """Initialize sensor translator.

        Args:
            simulation_config: SimulationConfig object
            geometry_utils: GeometryUtils instance for coordinate operations
        """
        self.simulation_config = simulation_config
        self.geometry_utils = geometry_utils
        self._current_scene_description = None
        self._current_scene_dir = None

    def translate_illumination(self) -> Dict[str, Any]:
        """Translate generic illumination to Eradiate format.

        Returns:
            Illumination configuration dictionary for Eradiate
        """
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

    def translate_sensors(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
    ) -> List[Dict[str, Any]]:
        """Translate generic sensors and measurements to Eradiate measures.

        Args:
            scene_description: Scene description with metadata
            scene_dir: Scene directory path
            include_irradiance_measures: Whether to include irradiance measurements

        Returns:
            List of Eradiate measure dictionaries
        """
        self._current_scene_description = scene_description
        self._current_scene_dir = scene_dir
        measures = []

        for sensor in self.simulation_config.sensors:
            if isinstance(sensor, SatelliteSensor):
                measures.append(
                    self.translate_satellite_sensor(sensor, scene_description)
                )
            elif isinstance(sensor, UAVSensor):
                measures.append(
                    self.translate_uav_sensor(sensor, scene_description, scene_dir)
                )
            elif isinstance(sensor, GroundSensor):
                measures.append(
                    self.translate_ground_sensor(sensor, scene_description, scene_dir)
                )
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor)}")

        if include_irradiance_measures:
            for measurement in self.simulation_config.measurements:
                if isinstance(measurement, IrradianceConfig):
                    measures.append(self.create_irradiance_measure(measurement))
                elif isinstance(measurement, (HDRFConfig, HCRFConfig)):
                    continue  # Handled in derived measurements
                else:
                    raise ValueError(
                        f"Unsupported measurement type: {type(measurement)}"
                    )

        return measures

    def translate_satellite_sensor(
        self, sensor: SatelliteSensor, scene_description: SceneDescription
    ) -> Dict[str, Any]:
        """Translate satellite sensor to Eradiate mpdistant measure.

        Args:
            sensor: Satellite sensor configuration
            scene_description: Scene description with location

        Returns:
            Eradiate measure dictionary
        """
        scene_location = scene_description.location
        scene_center_lat = scene_location.get("center_lat")
        scene_center_lon = scene_location.get("center_lon")

        if scene_center_lat is None or scene_center_lon is None:
            raise ValueError(
                "Scene description missing center_lat or center_lon in location"
            )

        coords = CoordinateSystem(scene_center_lat, scene_center_lon)

        if isinstance(sensor.target_size_km, (int, float)):
            width_km = height_km = sensor.target_size_km
        else:
            width_km, height_km = sensor.target_size_km

        target_bounds = coords.create_rectangle(
            sensor.target_center_lat, sensor.target_center_lon, width_km, height_km
        )

        measure_config = {
            "type": "mpdistant",
            "construct": "from_angles",
            "angles": [sensor.viewing.zenith, sensor.viewing.azimuth],
            "id": sanitize_sensor_id(sensor.id),
            "film_resolution": sensor.film_resolution,
            "target": {
                "type": "rectangle",
                "xmin": target_bounds["xmin"],
                "xmax": target_bounds["xmax"],
                "ymin": target_bounds["ymin"],
                "ymax": target_bounds["ymax"],
            },
            "srf": self.translate_srf(sensor.srf),
            "spp": sensor.samples_per_pixel,
        }

        return measure_config

    def translate_uav_sensor(
        self,
        sensor: UAVSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Dict[str, Any]:
        """Translate UAV sensor to Eradiate measure.

        Args:
            sensor: UAV sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path

        Returns:
            Eradiate measure dictionary
        """
        view = sensor.viewing

        origin, target = self.geometry_utils.adjust_origin_target_for_terrain(
            view, scene_description, scene_dir
        )

        base_config = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": self.translate_srf(sensor.srf),
            "origin": origin,
        }

        if sensor.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            base_config["type"] = "perspective"
            base_config["fov"] = sensor.fov or 70.0
            base_config["film_resolution"] = sensor.resolution or [1024, 1024]

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
                base_config["up"] = view.up or [0, 0, 1]
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self.geometry_utils.calculate_target_from_angles(view)
                base_config["target"] = target
                base_config["up"] = view.up or [0, 0, 1]

        elif sensor.instrument == UAVInstrumentType.RADIANCEMETER:
            base_config["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self.geometry_utils.calculate_target_from_angles(view)
                base_config["target"] = target

        else:
            raise ValueError(f"Unsupported UAV instrument type: {sensor.instrument}")

        return base_config

    def translate_ground_sensor(
        self,
        sensor: GroundSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Dict[str, Any]:
        """Translate ground sensor to Eradiate measure.

        Args:
            sensor: Ground sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path

        Returns:
            Eradiate measure dictionary
        """
        view = sensor.viewing

        # Handle HYPSTAR special case with post-processing
        if (
            sensor.instrument == GroundInstrumentType.HYPSTAR
            and sensor.hypstar_post_processing
            and sensor.hypstar_post_processing.apply_srf
        ):
            simulation_srf = {
                "type": "uniform",
                "wmin": 380.0,
                "wmax": 1680.0,
                "value": 1.0,
            }
        else:
            simulation_srf = self.translate_srf(sensor.srf)

        base_measure = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": simulation_srf,
        }

        if isinstance(view, HemisphericalViewing):
            base_measure["type"] = "hdistant"
            base_measure["direction"] = [0, 0, 1] if view.upward_looking else [0, 0, -1]

        elif isinstance(view, (LookAtViewing, AngularFromOriginViewing)):
            origin, target = self.geometry_utils.adjust_origin_target_for_terrain(
                view, scene_description, scene_dir
            )

            base_measure["origin"] = origin

            if sensor.instrument in [
                GroundInstrumentType.PERSPECTIVE_CAMERA,
                GroundInstrumentType.DHP_CAMERA,
            ]:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [1024, 1024]
                base_measure["fov"] = sensor.fov or (
                    180.0
                    if sensor.instrument == GroundInstrumentType.DHP_CAMERA
                    else 70.0
                )
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.HYPSTAR:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [5, 5]
                base_measure["fov"] = sensor.fov or 5.0
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.RADIANCEMETER:
                base_measure["type"] = "radiancemeter"

            else:
                base_measure["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_measure["target"] = target if target is not None else view.target

            elif isinstance(view, AngularFromOriginViewing):
                target, direction = self.geometry_utils.calculate_target_from_angles(
                    view
                )
                base_measure["target"] = target

        else:
            raise ValueError(
                f"Unsupported viewing type for ground sensor: {type(view)}"
            )

        return base_measure

    def create_irradiance_measure(self, irradiance_config) -> Dict[str, Any]:
        """Create BOA distant measure for irradiance measurement.

        Args:
            irradiance_config: IrradianceConfig object

        Returns:
            Eradiate measure dictionary
        """
        srf = None
        if irradiance_config.location and irradiance_config.location.srf:
            srf = self.translate_srf(irradiance_config.location.srf)
            spp = irradiance_config.location.samples_per_pixel

        return {
            "type": "hdistant",
            "id": sanitize_sensor_id(irradiance_config.id),
            "direction": [0, 0, 1],  # Upward looking
            "srf": srf,
            "spp": spp,
        }

    def create_brf_measure(self, brf_config) -> Dict[str, Any]:
        """Create BOA distant measure for BRF computation.

        Args:
            brf_config: BRFConfig object

        Returns:
            Eradiate measure dictionary
        """
        raise NotImplementedError("BRF measurements not yet implemented")

    def create_radiance_measure(self, radiance_config) -> Dict[str, Any]:
        """Create BOA radiance measure.

        Args:
            radiance_config: RadianceConfig object

        Returns:
            Eradiate measure dictionary
        """
        raise NotImplementedError("Direct radiance measurements handled via sensors")

    def create_bhr_measure(self, bhr_config) -> Dict[str, Any]:
        """Create hemispherical measure for BHR computation.

        Args:
            bhr_config: BHRConfig object

        Returns:
            Eradiate measure dictionary
        """
        raise NotImplementedError("BHR measurements not yet implemented")

    def translate_srf(self, srf) -> Union[Dict[str, Any], str]:
        """Translate generic SRF to Eradiate format.

        Args:
            srf: SRF configuration object

        Returns:
            Eradiate SRF dictionary or identifier string
        """
        if srf is None:
            return None

        if srf.type == "delta":
            return {
                "type": "multi_delta",
                "wavelengths": srf.wavelengths,
            }
        elif srf.type == "uniform":
            return {
                "type": "uniform",
                "wmin": srf.wmin,
                "wmax": srf.wmax,
            }
        elif srf.type == "dataset":
            return srf.dataset_id
        elif srf.type == "platform":
            return self.resolve_platform_srf(srf.platform, srf.instrument, srf.band)
        else:
            raise ValueError(f"Unsupported SRF type: {srf.type}")

    def resolve_platform_srf(self, platform: str, instrument: str, band: str) -> str:
        """Resolve platform/instrument/band combination to Eradiate SRF identifier.

        Args:
            platform: Platform name
            instrument: Instrument name
            band: Band name

        Returns:
            Eradiate SRF identifier string
        """
        srf_id = f"{platform.lower()}_{instrument.lower()}_{band.lower()}"
        return srf_id

    def generate_radiance_sensor_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> GroundSensor:
        """Generate radiance sensor for HDRF measurement.

        Args:
            hdrf_config: HDRF configuration

        Returns:
            Generated GroundSensor
        """
        sensor_id = f"hdrf_radiance_{hdrf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=hdrf_config.viewing,
            srf=hdrf_config.srf,
            samples_per_pixel=hdrf_config.samples_per_pixel or 1000,
        )

    def generate_irradiance_measurement_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HDRF.

        Args:
            hdrf_config: HDRF configuration

        Returns:
            Generated IrradianceConfig
        """
        irradiance_id = f"hdrf_irradiance_{hdrf_config.id}"

        # Update location with srf from hdrf_config
        location_with_srf = hdrf_config.location.model_copy(
            update={"srf": hdrf_config.srf}
        )

        return IrradianceConfig(
            id=irradiance_id,
            location=location_with_srf,
        )

    def generate_camera_sensor_for_hcrf(self, hcrf_config: HCRFConfig) -> GroundSensor:
        """Generate camera sensor for HCRF measurement.

        Args:
            hcrf_config: HCRF configuration

        Returns:
            Generated GroundSensor
        """
        sensor_id = f"hcrf_camera_{hcrf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            viewing=hcrf_config.viewing,
            fov=hcrf_config.fov,
            resolution=hcrf_config.film_resolution,
            srf=hcrf_config.srf,
            samples_per_pixel=hcrf_config.samples_per_pixel or 1000,
        )

    def generate_irradiance_measurement_for_hcrf(
        self, hcrf_config: HCRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HCRF.

        Args:
            hcrf_config: HCRF configuration

        Returns:
            Generated IrradianceConfig
        """
        irradiance_id = f"hcrf_irradiance_{hcrf_config.id}"

        # Update location with srf from hcrf_config
        location_with_srf = hcrf_config.location.model_copy(
            update={"srf": hcrf_config.srf}
        )

        return IrradianceConfig(
            id=irradiance_id,
            location=location_with_srf,
        )

    def get_sensor_by_id(self, sensor_id: str):
        """Find sensor config by ID.

        Args:
            sensor_id: Sensor ID to find

        Returns:
            Sensor configuration or None
        """
        for sensor in self.simulation_config.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def apply_post_processing_to_sensors(
        self, sensor_results: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Apply post-processing to sensor results.

        Args:
            sensor_results: Dictionary of sensor results

        Returns:
            Dictionary of post-processed results
        """
        from ...processors.post_processor import PostProcessor

        post_processor = PostProcessor(self.simulation_config)
        processed_results = {}

        for sensor_id, dataset in sensor_results.items():
            sensor = self.get_sensor_by_id(sensor_id)
            if sensor:
                processed_results[sensor_id] = post_processor.process_sensor_result(
                    dataset, sensor
                )
            else:
                processed_results[sensor_id] = dataset

        return processed_results

    def compute_derived_measurements(
        self, sensor_results: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Compute derived measurements (HDRF, HCRF) from sensor results.

        Args:
            sensor_results: Dictionary of sensor and measurement results

        Returns:
            Dictionary of derived measurement datasets
        """
        derived = {}

        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                if (
                    measurement.radiance_sensor_id
                    and measurement.irradiance_measurement_id
                ):
                    radiance_ds = sensor_results.get(measurement.radiance_sensor_id)
                    irradiance_ds = sensor_results.get(
                        measurement.irradiance_measurement_id
                    )

                    if radiance_ds is not None and irradiance_ds is not None:
                        hdrf_ds = self.compute_hdrf(
                            radiance_ds, irradiance_ds, measurement
                        )
                        derived[measurement.id] = hdrf_ds

            elif isinstance(measurement, HCRFConfig):
                if (
                    measurement.radiance_sensor_id
                    and measurement.irradiance_measurement_id
                ):
                    radiance_ds = sensor_results.get(measurement.radiance_sensor_id)
                    irradiance_ds = sensor_results.get(
                        measurement.irradiance_measurement_id
                    )

                    if radiance_ds is not None and irradiance_ds is not None:
                        hcrf_ds = self.compute_hcrf(
                            radiance_ds, irradiance_ds, measurement
                        )
                        derived[measurement.id] = hcrf_ds

        return derived

    def compute_hdrf(
        self,
        radiance_dataset: xr.Dataset,
        irradiance_dataset: xr.Dataset,
        hdrf_config: HDRFConfig,
    ) -> xr.Dataset:
        """Compute HDRF from radiance and irradiance datasets.

        Args:
            radiance_dataset: Radiance dataset
            irradiance_dataset: Irradiance dataset
            hdrf_config: HDRF configuration

        Returns:
            HDRF dataset
        """
        L_actual = self.extract_radiance_variable(radiance_dataset)
        E_reference = self.extract_irradiance_variable(irradiance_dataset)

        # Ensure wavelength alignment
        if "w" in L_actual.dims and "w" in E_reference.dims:
            if not np.array_equal(L_actual.w.values, E_reference.w.values):
                E_reference = E_reference.interp(w=L_actual.w, method="linear")

        # Broadcast if needed
        if set(L_actual.dims) - set(E_reference.dims):
            E_reference = E_reference.broadcast_like(L_actual)

        # Compute HDRF = π × L / E
        hdrf = (np.pi * L_actual) / E_reference

        hdrf_ds = xr.Dataset(
            {"hdrf": hdrf},
            attrs={
                "measurement_type": "hdrf",
                "measurement_id": hdrf_config.id,
                "units": "dimensionless",
            },
        )

        return hdrf_ds

    def compute_hcrf(
        self,
        radiance_dataset: xr.Dataset,
        irradiance_dataset: xr.Dataset,
        hcrf_config: HCRFConfig,
    ) -> xr.Dataset:
        """Compute HCRF from radiance and irradiance datasets.

        Args:
            radiance_dataset: Radiance dataset
            irradiance_dataset: Irradiance dataset
            hcrf_config: HCRF configuration

        Returns:
            HCRF dataset
        """
        L = self.extract_radiance_variable(radiance_dataset)
        E = self.extract_irradiance_variable(irradiance_dataset)

        # Ensure wavelength alignment
        if "w" in L.dims and "w" in E.dims:
            if not np.array_equal(L.w.values, E.w.values):
                E = E.interp(w=L.w, method="linear")

        # Broadcast if needed
        if set(L.dims) - set(E.dims):
            E = E.broadcast_like(L)

        # Compute HCRF = π × L / E (with epsilon for stability)
        epsilon = 1e-10
        hcrf = (np.pi * L) / (E + epsilon)

        hcrf_ds = xr.Dataset(
            {"hcrf": hcrf},
            attrs={
                "measurement_type": "hcrf",
                "measurement_id": hcrf_config.id,
                "units": "dimensionless",
            },
        )

        # Apply post-processing if configured
        if hcrf_config.post_processing:
            hcrf_ds = self.apply_hcrf_post_processing(
                hcrf_ds, hcrf_config.post_processing
            )

        return hcrf_ds

    def apply_hcrf_post_processing(
        self, dataset: xr.Dataset, post_processing_config
    ) -> xr.Dataset:
        """Apply post-processing to HCRF dataset.

        Args:
            dataset: HCRF dataset
            post_processing_config: Post-processing configuration

        Returns:
            Post-processed HCRF dataset
        """
        # Apply any configured post-processing steps
        # This is a placeholder for future implementation
        return dataset

    def extract_radiance_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        """Extract radiance variable from dataset.

        Args:
            dataset: Dataset containing radiance

        Returns:
            Radiance DataArray

        Raises:
            ValueError: If no radiance variable found
        """
        for var_name in RADIANCE_VARIABLE_NAMES:
            if var_name in dataset:
                return dataset[var_name]

        raise ValueError(
            f"No radiance variable found in dataset. Looked for: {RADIANCE_VARIABLE_NAMES}"
        )

    def extract_irradiance_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        """Extract irradiance variable from dataset.

        Args:
            dataset: Dataset containing irradiance

        Returns:
            Irradiance DataArray

        Raises:
            ValueError: If no irradiance variable found
        """
        for var_name in IRRADIANCE_VARIABLE_NAMES:
            if var_name in dataset:
                return dataset[var_name]

        raise ValueError(
            f"No irradiance variable found in dataset. Looked for: {IRRADIANCE_VARIABLE_NAMES}"
        )
