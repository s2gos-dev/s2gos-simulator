"""Sensor and measurement translation for Eradiate backend.

This module handles translation of generic sensor configurations to Eradiate-specific
measure definitions, including post-processing and derived measurement computation.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .geometry_utils import apply_asset_relative_transform, sanitize_sensor_id
from ...config import (
    AngularFromOriginViewing,
    BRFConfig,
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

logger = logging.getLogger(__name__)


class _AssetTransform(NamedTuple):
    """Internal: Transform information for a positioned asset.

    Used to resolve sensor positions specified relative to assets.
    Rotation contains USER-SPECIFIED rotations only, NOT including
    the internal Blender coordinate fix (90° X rotation).
    """

    object_id: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: float


class SensorTranslator:
    """Translator for sensors, measurements, and derived quantities."""

    def __init__(self, simulation_config, geometry_utils, backend=None):
        """Initialize sensor translator.

        Args:
            simulation_config: SimulationConfig object
            geometry_utils: GeometryUtils instance for coordinate operations
            backend: Optional backend instance for accessing shared state (e.g., disk_coords)
        """
        self.simulation_config = simulation_config
        self.geometry_utils = geometry_utils
        self.backend = backend
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

    def _extract_asset_transforms(
        self, scene_description: SceneDescription
    ) -> Dict[str, _AssetTransform]:
        """Extract asset transforms from scene description (backend-internal).

        Reads blender_fix metadata to reverse coordinate fix and get user rotations.

        Args:
            scene_description: SceneDescription with positioned assets

        Returns:
            Dict mapping object_id to _AssetTransform
        """
        asset_transforms = {}

        for obj in scene_description.objects:
            if not isinstance(obj, dict):
                continue
            if "position" not in obj or "rotation" not in obj or "scale" not in obj:
                continue
            if obj.get("type") in ["vegetation_collection", "shapegroup"]:
                continue

            final_rotation = obj["rotation"]
            blender_fix = obj.get("blender_fix", False)

            if blender_fix:
                user_rotation_x = final_rotation[0] - 90.0
                user_rotation_z = final_rotation[1]
                user_rotation_y = -final_rotation[2]
                user_rotation = (user_rotation_x, user_rotation_y, user_rotation_z)
            else:
                user_rotation = tuple(final_rotation)

            asset_transforms[obj["id"]] = _AssetTransform(
                object_id=obj["id"],
                position=tuple(obj["position"]),
                rotation=user_rotation,
                scale=obj.get("scale", 1.0),
            )

        return asset_transforms

    def _resolve_asset_reference(
        self, reference: str, asset_transforms: Dict[str, _AssetTransform]
    ) -> str:
        """Resolve user-friendly asset reference to actual object ID.

        Handles extensions and prefix matching for intuitive asset referencing.

        Args:
            reference: User reference like "tower_v0_1.xml", "my_asset.ply", or "exact_id"
            asset_transforms: Dict of available asset transforms

        Returns:
            Resolved object_id

        Raises:
            ValueError: If reference cannot be resolved
        """
        # 1. Try exact match first (for exact IDs without extensions)
        if reference in asset_transforms:
            return reference

        # 2. Strip extension if present
        if reference.endswith(".xml") or reference.endswith(".ply"):
            reference_stem = reference.rsplit(".", 1)[0]
        else:
            reference_stem = reference

        # 3. Try exact match with stem
        if reference_stem in asset_transforms:
            return reference_stem

        # 4. If original had .xml extension, find first asset with prefix
        if reference.endswith(".xml"):
            prefix = reference_stem + "_"
            matches = [oid for oid in asset_transforms if oid.startswith(prefix)]
            if matches:
                matches.sort()
                logger.info(
                    f"Resolved '{reference}' to '{matches[0]}' (first of {len(matches)} matches)"
                )
                return matches[0]

        # 5. Not found - provide error
        available = sorted(asset_transforms.keys())
        if reference.endswith(".xml"):
            prefix = reference_stem + "_"
            xml_assets = [a for a in available if a.startswith(prefix)]
            if xml_assets:
                raise ValueError(
                    f"Asset reference '{reference}' matches {len(xml_assets)} assets with prefix '{prefix}'. "
                    f"Matches: {xml_assets[:5]}... Use full object_id for specific asset."
                )
            else:
                raise ValueError(
                    f"No assets found for XML '{reference}'. "
                    f"Available assets: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
        else:
            raise ValueError(
                f"Asset '{reference}' not found. "
                f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
            )

    def translate_sensors(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
        sensor_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Translate generic sensors and measurements to Eradiate measures.

        Args:
            scene_description: Scene description with metadata
            scene_dir: Scene directory path
            include_irradiance_measures: Whether to include irradiance measurements
            sensor_ids: Optional set of sensor IDs to include. If None, all sensors
                are included. Use this to filter sensors for specific workflows
                (e.g., only BRF sensors for BRF workflow).

        Returns:
            List of Eradiate measure dictionaries
        """
        self._current_scene_description = scene_description
        self._current_scene_dir = scene_dir

        asset_transforms = self._extract_asset_transforms(scene_description)

        measures = []

        for sensor in self.simulation_config.sensors:
            if sensor_ids is not None and sensor.id not in sensor_ids:
                continue
            if isinstance(sensor, SatelliteSensor):
                measures.append(
                    self.translate_satellite_sensor(sensor, scene_description)
                )
            elif isinstance(sensor, UAVSensor):
                measures.append(
                    self.translate_uav_sensor(
                        sensor, scene_description, scene_dir, asset_transforms
                    )
                )
            elif isinstance(sensor, GroundSensor):
                measures.append(
                    self.translate_ground_sensor(
                        sensor, scene_description, scene_dir, asset_transforms
                    )
                )
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor)}")

        if include_irradiance_measures:
            for measurement in self.simulation_config.measurements:
                if isinstance(measurement, IrradianceConfig):
                    if self.backend is None or not hasattr(
                        self.backend, "irradiance_disk_coords"
                    ):
                        raise RuntimeError(
                            f"Cannot create irradiance measure '{measurement.id}': "
                            f"Backend irradiance_disk_coords not available. "
                            f"IrradianceProcessor must run before creating measures."
                        )

                    disk_coords = self.backend.irradiance_disk_coords.get(
                        measurement.id
                    )
                    if disk_coords is None:
                        raise RuntimeError(
                            f"Disk coordinates not found for irradiance measurement '{measurement.id}'. "
                            f"Available measurements: {list(self.backend.irradiance_disk_coords.keys())}"
                        )

                    measures.append(
                        self.create_irradiance_measure(measurement, disk_coords)
                    )
                elif isinstance(measurement, (HDRFConfig, HCRFConfig, BRFConfig)):
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
        asset_transforms: Dict[str, _AssetTransform],
    ) -> Dict[str, Any]:
        """Translate UAV sensor to Eradiate measure.

        Args:
            sensor: UAV sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path
            asset_transforms: Asset transforms for relative positioning (backend-internal)

        Returns:
            Eradiate measure dictionary
        """
        view = sensor.viewing

        # Handle asset-relative positioning
        if hasattr(view, "relative_to_asset") and view.relative_to_asset is not None:
            asset_name = self._resolve_asset_reference(
                view.relative_to_asset, asset_transforms
            )
            asset_transform = asset_transforms[asset_name]

            view.origin = apply_asset_relative_transform(view.origin, asset_transform)

            if isinstance(view, LookAtViewing):
                view.target = apply_asset_relative_transform(
                    view.target, asset_transform
                )

            # Avoid double application of terrain adjustment
            if view.terrain_relative_height:
                view.terrain_relative_height = False

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
        asset_transforms: Dict[str, _AssetTransform],
    ) -> Dict[str, Any]:
        """Translate ground sensor to Eradiate measure.

        Args:
            sensor: Ground sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path
            asset_transforms: Asset transforms for relative positioning (backend-internal)

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

        if hasattr(view, "relative_to_asset") and view.relative_to_asset is not None:
            asset_name = self._resolve_asset_reference(
                view.relative_to_asset, asset_transforms
            )
            asset_transform = asset_transforms[asset_name]

            if hasattr(view, "origin") and view.origin is not None:
                view.origin = apply_asset_relative_transform(
                    view.origin, asset_transform
                )

            if isinstance(view, LookAtViewing):
                view.target = apply_asset_relative_transform(
                    view.target, asset_transform
                )

            # Avoid double application of terrain adjustment
            if (
                hasattr(view, "terrain_relative_height")
                and view.terrain_relative_height
            ):
                view.terrain_relative_height = False

        if isinstance(view, HemisphericalViewing):
            base_measure["type"] = "hdistant"
            base_measure["direction"] = [0, 0, 1] if view.upward_looking else [0, 0, -1]

            if view.origin is not None:
                origin, _ = self.geometry_utils.adjust_origin_target_for_terrain(
                    view, scene_description, scene_dir
                )
                base_measure["target"] = origin
                logger.debug(
                    f"HemisphericalViewing sensor '{sensor.id}' target set to {origin}"
                )
            else:
                logger.warning(
                    f"HemisphericalViewing sensor '{sensor.id}' has no origin. "
                    f"Using scene center at ground level [0, 0, 0] as target. "
                    f"This may produce incorrect results."
                )
                base_measure["target"] = [0.0, 0.0, 0.0]

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

    def create_irradiance_measure(
        self, irradiance_config, disk_coords: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Create hdistant measure for BOA irradiance measurement.

        The target parameter is set to the white reference disk location.

        Args:
            irradiance_config: IrradianceConfig with measurement parameters
            disk_coords: Pre-computed disk position (x, y, z) in scene coordinates

        Returns:
            Eradiate hdistant measure dictionary with target set to disk location
        """
        if irradiance_config.location and irradiance_config.location.srf:
            srf = self.translate_srf(irradiance_config.location.srf)
            spp = irradiance_config.location.samples_per_pixel
        else:
            srf = {
                "type": "uniform",
                "wmin": 380.0,
                "wmax": 1680.0,
            }
            spp = irradiance_config.samples_per_pixel or 512

        x, y, z = disk_coords

        logger.debug(
            f"Creating hdistant measure '{irradiance_config.id}' with target at "
            f"({x:.2f}, {y:.2f}, {z:.2f}) m, spp={spp}, ray_offset=0.1m"
        )

        return {
            "type": "hdistant",
            "id": sanitize_sensor_id(irradiance_config.id),
            "target": [x, y, z],  # Point at disk location
            "direction": [0, 0, 1],  # Upward-looking hemisphere
            "ray_offset": 0.1,  # Small offset to prevent self-intersection
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
            srf: SRF configuration object or string identifier

        Returns:
            Eradiate SRF dictionary or identifier string
        """
        if srf is None:
            return None

        if isinstance(srf, str):
            return srf

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

    def generate_radiance_sensor_for_brf(self, brf_config: BRFConfig) -> GroundSensor:
        """Generate radiance sensor for BRF measurement (no atmosphere).

        Args:
            brf_config: BRF configuration

        Returns:
            Generated GroundSensor for radiance measurement
        """
        sensor_id = f"brf_radiance_{brf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=brf_config.viewing,
            srf=brf_config.srf,
            samples_per_pixel=brf_config.samples_per_pixel or 512,
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
        from ...processors.sensor_processor import SensorProcessor

        sensor_processor = SensorProcessor(self.simulation_config)
        processed_results = {}

        for sensor_id, dataset in sensor_results.items():
            logger.debug(f"Processing sensor: {sensor_id}")
            sensor = self.get_sensor_by_id(sensor_id)
            if sensor:
                logger.debug(f"Found sensor config: {sensor_id}")
                post_processed_dataset = sensor_processor.process_sensor_result(
                    dataset, sensor
                )
                # Only save raw version if post-processing changed the data
                if post_processed_dataset is not dataset:
                    logger.debug(f"Post-processing applied: {sensor_id}")
                    processed_results[f"{sensor_id}_raw_eradiate"] = dataset
                processed_results[sensor_id] = post_processed_dataset
            else:
                logger.debug(f"No sensor config found: {sensor_id}, passing through")
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
        L_actual = radiance_dataset["radiance"]
        E_reference = irradiance_dataset["boa_irradiance"]

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
        L = radiance_dataset["radiance"]
        E = irradiance_dataset["boa_irradiance"]

        # Ensure wavelength alignment
        if "w" in L.dims and "w" in E.dims:
            if not np.array_equal(L.w.values, E.w.values):
                E = E.interp(w=L.w, method="linear")

        # Broadcast if needed
        if set(L.dims) - set(E.dims):
            E = E.broadcast_like(L)

        hcrf = (np.pi * L) / E

        hcrf_ds = xr.Dataset(
            {"hcrf": hcrf},
            attrs={
                "measurement_type": "hcrf",
                "measurement_id": hcrf_config.id,
                "units": "dimensionless",
            },
        )

        return hcrf_ds

    def compute_brf(
        self,
        radiance_dataset: xr.Dataset,
        brf_config: BRFConfig,
    ) -> xr.Dataset:
        """Compute BRF from radiance dataset (no atmosphere).

        BRF = (π × L) / (E_toa × cos(SZA))

        Args:
            radiance_dataset: Radiance dataset with 'irradiance'
            brf_config: BRF configuration

        Returns:
            BRF dataset
        """
        L = radiance_dataset["radiance"]
        E_toa = radiance_dataset["irradiance"]

        illumination = self.simulation_config.illumination
        if not isinstance(illumination, DirectionalIllumination):
            raise ValueError(
                f"BRF requires DirectionalIllumination, got {type(illumination)}"
            )

        sza_rad = np.deg2rad(illumination.zenith)
        cos_sza = np.cos(sza_rad)

        if "w" in L.dims and "w" in E_toa.dims:
            if not np.array_equal(L.w.values, E_toa.w.values):
                E_toa = E_toa.interp(w=L.w, method="linear")

        if set(L.dims) - set(E_toa.dims):
            E_toa = E_toa.broadcast_like(L)

        brf = (np.pi * L) / (E_toa * cos_sza)

        return xr.Dataset(
            {"brf": brf},
            attrs={
                "measurement_type": "brf",
                "measurement_id": brf_config.id,
                "units": "dimensionless",
                "formula": "BRF = (π × L) / (E_toa × cos(SZA))",
                "solar_zenith_deg": illumination.zenith,
                "cos_sza": float(cos_sza),
                "atmosphere": "none",
            },
        )

    def compute_brf_measurements(
        self,
        sensor_results: Dict[str, xr.Dataset],
        brf_configs: List[BRFConfig],
    ) -> Dict[str, xr.Dataset]:
        """Compute BRF for all BRF configs.

        Args:
            sensor_results: Dictionary of sensor results
            brf_configs: List of BRFConfig measurements

        Returns:
            Dictionary of BRF measurement datasets
        """
        derived = {}
        for config in brf_configs:
            radiance_ds = sensor_results.get(config.radiance_sensor_id)
            if radiance_ds is not None:
                try:
                    derived[config.id] = self.compute_brf(radiance_ds, config)
                    logger.info(f"Computed BRF for '{config.id}'")
                except Exception as e:
                    logger.error(f"Failed to compute BRF for '{config.id}': {e}")
            else:
                logger.warning(
                    f"Radiance dataset not found for BRF '{config.id}' "
                    f"(sensor_id: {config.radiance_sensor_id})"
                )
        return derived
