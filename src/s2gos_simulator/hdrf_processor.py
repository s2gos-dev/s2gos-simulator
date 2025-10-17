import copy
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import xarray as xr
from s2gos_utils.scene import SceneDescription
from s2gos_utils.scene.materials import DiffuseMaterial
from upath import UPath

logger = logging.getLogger(__name__)


# HDRF measurement constants
REFERENCE_DISK_RADIUS_M = 0.01  # Fixed radius for white reference disk (meters)


class HDRFProcessor:
    """Processor for HDRF measurements using white reference technique.

    The HDRFProcessor handles all HDRF-specific logic:
    - Detecting HDRF measurement requests
    - Generating white reference object (perfect Lambertian at BOA)
    - Coordinating dual simulations
    - Computing HDRF from radiance ratios

    HDRF = L_actual / L_reference

    Where L_reference is radiance from a perfect white Lambertian surface (ρ=1.0)
    under identical illumination and atmospheric conditions.
    """

    def __init__(self, backend):
        """Initialize HDRF processor.

        Args:
            backend: Backend instance (EradiateBackend) that will execute simulations
        """
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def requires_hdrf(self) -> bool:
        """Check if any measurements require HDRF computation.

        Returns:
            True if HDRF computation is needed
        """
        from .config import MeasurementType

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces"):
                if MeasurementType.HDRF in sensor.produces:
                    return True

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                return True

        return False

    def get_hdrf_measure_ids(self) -> List[str]:
        """Get list of measure IDs that produce HDRF.

        Returns:
            List of measure IDs that require HDRF computation
        """
        from .config import MeasurementType

        hdrf_ids = []

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces"):
                if MeasurementType.HDRF in sensor.produces:
                    hdrf_ids.append(sensor.id)

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                if rq.id:
                    rq_id = rq.id
                else:
                    string_viewing_zenith = str(rq.viewing_zenith).replace(".", "_")
                    string_viewing_azimuth = str(rq.viewing_azimuth).replace(".", "_")
                    rq_id = f"hdrf_{string_viewing_zenith}__{string_viewing_azimuth}"
                hdrf_ids.append(rq_id)

        return [x.replace(".", "_") for x in hdrf_ids]

    def _get_hdrf_config_for_measure(
        self, measure_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """Get target location and height offset for an HDRF measure.

        Args:
            measure_id: HDRF measure ID

        Returns:
            Tuple of (target_lat, target_lon, height_offset_m) or None if not found
        """
        from .config import MeasurementType

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and MeasurementType.HDRF in sensor.produces:
                if sensor.id == measure_id or sensor.id.replace(".", "_") == measure_id:
                    if hasattr(sensor, "target_center_lat") and hasattr(
                        sensor, "reference_panel_offset_m"
                    ):
                        return (
                            sensor.target_center_lat,
                            sensor.target_center_lon,
                            sensor.reference_panel_offset_m,
                        )

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                if rq.id:
                    rq_id = rq.id
                else:
                    string_viewing_zenith = str(rq.viewing_zenith).replace(".", "_")
                    string_viewing_azimuth = str(rq.viewing_azimuth).replace(".", "_")
                    rq_id = f"hdrf_{string_viewing_zenith}__{string_viewing_azimuth}"

                if rq_id == measure_id:
                    return (
                        rq.target_lat,  # Will use scene center
                        rq.target_lon,
                        rq.reference_panel_offset_m,
                    )

        return None

    def _find_dem_file(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Optional[UPath]:
        """Find DEM file in scene data directory.

        Args:
            scene_description: Scene description
            scene_dir: Base scene directory

        Returns:
            Path to DEM file or None if not found
        """

        scene_name = scene_description.name
        resolution = scene_description.resolution_m

        dem_path = scene_dir / "data" / f"dem_{scene_name}_{resolution}m.zarr"

        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            return dem_path

        dem_path = scene_dir / "data" / f"dem_{scene_name}.zarr"
        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            return dem_path

        logger.warning(f"DEM not found. Tried: data/dem_{scene_name}_*m.zarr")
        return None

    def _query_terrain_elevation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        target_lat: float,
        target_lon: float,
    ) -> float:
        """Query DEM elevation at specific lat/lon.

        Args:
            scene_description: Scene description
            scene_dir: Base scene directory
            target_lat: Target latitude
            target_lon: Target longitude

        Returns:
            Terrain elevation in meters (0.0 if query fails)
        """
        dem_path = self._find_dem_file(scene_description, scene_dir)
        if not dem_path:
            logger.warning("DEM not found, using terrain elevation 0.0m")
            return 0.0

        try:
            from s2gos_utils.coordinates import CoordinateSystem

            center_lat = scene_description.location["center_lat"]
            center_lon = scene_description.location["center_lon"]
            coord_system = CoordinateSystem(center_lat, center_lon)
            x, y = coord_system.latlon_to_scene(target_lat, target_lon)

            logger.debug(
                f"Target location: ({target_lat:.6f}, {target_lon:.6f}) -> scene coords: ({x:.1f}m, {y:.1f}m)"
            )
        except Exception as e:
            logger.warning(f"Coordinate transformation failed: {e}")
            return 0.0

        try:
            from scipy.interpolate import RegularGridInterpolator

            with xr.open_zarr(dem_path) as dem_ds:
                dem_data = dem_ds["elevation"]
                interpolator = RegularGridInterpolator(
                    (dem_data.y.values, dem_data.x.values),
                    dem_data.values,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                elevation = float(interpolator([(y, x)])[0])
                logger.debug(f"DEM elevation at target: {elevation:.2f}m")
                return elevation
        except Exception as e:
            logger.warning(f"DEM query failed: {e}, using 0.0m")
            return 0.0

    def get_reference_panel_elevation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        target_lat: float,
        target_lon: float,
        height_offset_m: float,
    ) -> float:
        """Determine reference panel elevation at measurement location.

        Computes elevation as: terrain elevation + height offset.
        The height offset accounts for vegetation canopy height above terrain.

        Args:
            scene_description: Scene description
            scene_dir: Base scene directory
            target_lat: Target latitude in decimal degrees
            target_lon: Target longitude in decimal degrees
            height_offset_m: Height above terrain (accounts for vegetation)

        Returns:
            Reference panel elevation in meters (terrain + offset)
        """
        terrain_elevation = self._query_terrain_elevation(
            scene_description, scene_dir, target_lat, target_lon
        )

        boa_level = terrain_elevation + height_offset_m

        logger.info(f"BOA level at ({target_lat:.4f}, {target_lon:.4f}):")
        logger.info(f"  Terrain elevation: {terrain_elevation:.2f}m")
        logger.info(f"  Height offset: {height_offset_m:.2f}m")
        logger.info(f"  Total BOA level: {boa_level:.2f}m")

        return boa_level

    def create_white_reference_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Tuple[SceneDescription, Tuple[float, float, float]]:
        """Create white reference version of scene for HDRF computation.

        Creates a scene identical to the original, but adds a small fixed-size disk
        with perfect white Lambertian reflectance (ρ=1.0) at the measurement location.

        Reference disk:
        - Fixed radius (REFERENCE_DISK_RADIUS_M)
        - Positioned at terrain + height_offset elevation
        - Perfect Lambertian material for HDRF = L_actual / L_reference

        Args:
            scene_description: Original scene description
            scene_dir: Scene directory containing DEM for terrain elevation queries

        Returns:
            Tuple of (modified scene description, reference disk center coordinates (x, y, z))

        Raises:
            ValueError: If HDRF configuration cannot be determined
        """
        logger.info("Creating white reference scene for HDRF computation...")

        reference_scene = copy.deepcopy(scene_description)

        hdrf_measure_ids = self.get_hdrf_measure_ids()
        if not hdrf_measure_ids:
            raise ValueError("No HDRF measures found - cannot create reference scene")

        hdrf_config = self._get_hdrf_config_for_measure(hdrf_measure_ids[0])
        if hdrf_config is None:
            raise ValueError(
                f"HDRF configuration not found for measure '{hdrf_measure_ids[0]}'"
            )

        from s2gos_utils.coordinates import CoordinateSystem

        target_lat, target_lon, height_offset = hdrf_config

        if target_lat is None or target_lon is None:
            target_lat = scene_description.location["center_lat"]
            target_lon = scene_description.location["center_lon"]
            logger.info("Using scene center for HDRF reference disk location")

        scene_center_lat = scene_description.location["center_lat"]
        scene_center_lon = scene_description.location["center_lon"]
        coord_system = CoordinateSystem(scene_center_lat, scene_center_lon)
        target_x, target_y = coord_system.latlon_to_scene(target_lat, target_lon)

        panel_elevation = self.get_reference_panel_elevation(
            scene_description,
            scene_dir,
            target_lat,
            target_lon,
            height_offset,
        )

        white_material_id = "perfect_white_lambertian_hdrf"
        reference_scene.materials[white_material_id] = DiffuseMaterial(
            id=white_material_id,
            type="lambertian",
            reflectance={"type": "uniform", "value": 1.0},  # Perfect white
        )
        logger.info(f"Added white Lambertian material: {white_material_id}")

        white_disk = {
            "object_id": "white_reference_disk_hdrf",
            "type": "disk",
            "center": [target_x, target_y, panel_elevation],
            "radius": REFERENCE_DISK_RADIUS_M,
            "material": white_material_id,
        }

        if not hasattr(reference_scene, "objects") or reference_scene.objects is None:
            reference_scene.objects = []
        reference_scene.objects.insert(0, white_disk)

        logger.info(
            f"Added white reference disk: radius={REFERENCE_DISK_RADIUS_M}m "
            f"at ({target_x:.1f}, {target_y:.1f}, {panel_elevation:.2f})"
        )

        return reference_scene, (target_x, target_y, panel_elevation)

    def execute_dual_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Execute dual simulation workflow for HDRF.

        Runs two simulations:
        1. Actual scene (original surface properties)
        2. White reference scene (perfect Lambertian at BOA)

        Both simulations use identical:
        - Illumination geometry
        - Atmospheric conditions
        - Sensor/viewing configurations

        Args:
            scene_description: Scene description for actual simulation
            scene_dir: Scene directory containing assets
            output_dir: Output directory for results

        Returns:
            Tuple of (actual_results, reference_results) as xarray Datasets
        """
        logger.info("=" * 70)
        logger.info("HDRF Dual Simulation Workflow")
        logger.info("=" * 70)

        logger.info("\n[1/3] Creating white reference scene...")
        reference_scene, self.reference_panel_coords = (
            self.create_white_reference_scene(scene_description, scene_dir)
        )
        logger.info(f"✓ Reference panel coordinates: {self.reference_panel_coords}")

        self.backend.reference_panel_coords = self.reference_panel_coords

        logger.info("\n[2/3] Running ACTUAL surface simulation...")
        actual_output_dir = output_dir / "actual_simulation"
        actual_results = self._run_simulation(
            scene_description, scene_dir, actual_output_dir, simulation_type="actual"
        )
        logger.info(
            f"✓ Actual simulation complete. Results: {list(actual_results.keys())}"
        )

        logger.info("\n[3/3] Running WHITE REFERENCE simulation...")
        reference_output_dir = output_dir / "reference_simulation"
        reference_results = self._run_simulation(
            reference_scene,
            scene_dir,
            reference_output_dir,
            simulation_type="reference",
        )
        logger.info(
            f"✓ Reference simulation complete. Results: {list(reference_results.keys())}"
        )

        return actual_results, reference_results

    def _run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        simulation_type: str = "actual",
    ) -> xr.Dataset:
        """Run simulation for HDRF dual-simulation workflow.

        For actual simulations, runs all configured measures.
        For reference simulations, runs only HDRF measures (filtered by ID)
        to avoid unnecessary computation on the white reference scene.

        Args:
            scene_description: Scene description (actual or white reference)
            scene_dir: Scene directory containing DEM and scene assets
            output_dir: Output directory for simulation results
            simulation_type: "actual" or "reference" - controls measure filtering

        Returns:
            Simulation results as dictionary keyed by measure ID
        """
        import eradiate
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        logger.info(f"  Creating {simulation_type} experiment...")
        experiment = self.backend._create_experiment(scene_description, scene_dir)

        if simulation_type == "actual":
            logger.info(f"  Executing {len(experiment.measures)} measures...")
            for i_measure in range(len(experiment.measures)):
                measure_id = getattr(
                    experiment.measures[i_measure], "id", f"measure_{i_measure}"
                )
                logger.info(
                    f"    Measure {i_measure + 1}/{len(experiment.measures)}: {measure_id}"
                )
                eradiate.run(experiment, measures=i_measure)
        else:
            hdrf_measure_ids = self.get_hdrf_measure_ids()
            logger.info(
                f"  Executing {len(hdrf_measure_ids)} HDRF measures "
                f"(filtering from {len(experiment.measures)} total)..."
            )

            for i_measure in range(len(experiment.measures)):
                measure_id = getattr(
                    experiment.measures[i_measure], "id", f"measure_{i_measure}"
                )

                if measure_id in hdrf_measure_ids:
                    logger.info(
                        f"    Measure {i_measure + 1}/{len(experiment.measures)}: {measure_id}"
                    )
                    eradiate.run(experiment, measures=i_measure)

        logger.info(f"  Processing {simulation_type} results...")
        results = self.backend._process_results(experiment, output_dir)

        return results

    def compute_hdrf(
        self,
        actual_results: Dict[str, xr.Dataset],
        reference_results: Dict[str, xr.Dataset],
    ) -> Dict[str, xr.Dataset]:
        """Compute HDRF from actual and reference simulation results.

        HDRF = L_actual / L_reference

        Where:
        - L_actual: radiance from actual surface
        - L_reference: radiance from perfect white Lambertian (ρ=1.0)

        Both measured under identical conditions (illumination, atmosphere, viewing).

        Args:
            actual_results: Results from actual surface simulation
            reference_results: Results from white reference simulation

        Returns:
            Dictionary of HDRF datasets keyed by measure ID
        """
        logger.info("\n" + "=" * 70)
        logger.info("Computing HDRF from dual simulation results")
        logger.info("=" * 70)

        hdrf_datasets = {}
        hdrf_measure_ids = self.get_hdrf_measure_ids()
        for measure_id in hdrf_measure_ids:
            if measure_id not in actual_results:
                logger.warning(
                    f"Measure '{measure_id}' not found in actual results, skipping"
                )
                continue
            if measure_id not in reference_results:
                logger.warning(
                    f"Measure '{measure_id}' not found in reference results, skipping"
                )
                continue

            logger.info(f"\nProcessing measure: {measure_id}")

            actual_ds = actual_results[measure_id]
            reference_ds = reference_results[measure_id]

            if "radiance" not in actual_ds or "radiance" not in reference_ds:
                logger.warning(
                    f"Radiance not found in results for {measure_id}, skipping"
                )
                continue

            L_actual = actual_ds["radiance"].squeeze(drop=True)
            L_reference = reference_ds["radiance"].squeeze(drop=True)

            logger.info(f"  Actual radiance shape: {L_actual.shape}")
            logger.info(f"  Reference radiance shape: {L_reference.shape}")

            if L_actual.dims != L_reference.dims:
                logger.error(
                    f"Dimension mismatch: {L_actual.dims} != {L_reference.dims}"
                )
                continue

            # Compute HDRF, take mean to reduce noise
            hdrf = L_actual / L_reference.mean()

            logger.info(
                f"  HDRF computed: mean={float(hdrf.mean()):.4f}, "
                f"std={float(hdrf.std()):.4f}, "
                f"min={float(hdrf.min()):.4f}, "
                f"max={float(hdrf.max()):.4f}"
            )

            hdrf_dataset = xr.Dataset(
                {
                    "hdrf": hdrf,
                    "radiance_actual": L_actual,
                    "radiance_reference": L_reference,
                },
                coords=hdrf.coords,
            )

            hdrf_dataset.attrs.update(
                {
                    "measurement_type": "hdrf",
                    "measurement_level": "boa",
                    "quantity": "Hemispherical-Directional Reflectance Factor",
                    "quantity_definition": "Ratio of target radiance to perfect Lambertian reference radiance",
                    "units": "dimensionless",
                    "valid_range_min": 0.0,
                    "valid_range_max": 1.0,
                    "computation_method": "white_reference_dual_simulation",
                    "white_reference_reflectance": 1.0,
                    "white_reference_type": "perfect_lambertian",
                    "processing_timestamp": datetime.now().isoformat(),
                    "hdrf_mean": float(hdrf.mean()),
                    "hdrf_std": float(hdrf.std()),
                    "hdrf_min": float(hdrf.min()),
                    "hdrf_max": float(hdrf.max()),
                }
            )

            if hasattr(actual_ds, "attrs"):
                for key in ["sensor_id", "platform_type", "illumination_type"]:
                    if key in actual_ds.attrs:
                        hdrf_dataset.attrs[key] = actual_ds.attrs[key]

            hdrf_datasets[measure_id] = hdrf_dataset
            logger.info(f"  ✓ HDRF dataset created for {measure_id}")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"HDRF computation complete: {len(hdrf_datasets)} datasets")
        logger.info(f"{'=' * 70}\n")

        return hdrf_datasets

    def save_hdrf_results(
        self, hdrf_datasets: Dict[str, xr.Dataset], output_dir: UPath
    ) -> None:
        """Save HDRF results to disk.

        Args:
            hdrf_datasets: Dictionary of HDRF datasets
            output_dir: Output directory
        """
        from s2gos_utils.io.paths import mkdir

        hdrf_output_dir = output_dir / "hdrf_results"
        mkdir(hdrf_output_dir)

        logger.info(f"\nSaving HDRF results to: {hdrf_output_dir}")

        for measure_id, dataset in hdrf_datasets.items():
            output_file = hdrf_output_dir / f"{measure_id}_hdrf.zarr"
            dataset.to_zarr(output_file, mode="w")
            logger.info(f"  ✓ Saved: {output_file.name}")

        summary = {
            "hdrf_measures": [],
            "white_reference_info": {
                "reflectance": 1.0,
                "type": "perfect_lambertian",
            },
            "computation_timestamp": datetime.now().isoformat(),
        }

        for measure_id, dataset in hdrf_datasets.items():
            summary["hdrf_measures"].append(
                {
                    "measure_id": measure_id,
                    "mean_hdrf": float(dataset["hdrf"].mean()),
                    "std_hdrf": float(dataset["hdrf"].std()),
                    "min_hdrf": float(dataset["hdrf"].min()),
                    "max_hdrf": float(dataset["hdrf"].max()),
                    "physically_valid": float(dataset["hdrf"].min()) >= 0
                    and float(dataset["hdrf"].max()) <= 1.2,
                }
            )

        import json

        summary_file = hdrf_output_dir / "hdrf_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  ✓ Saved: {summary_file.name}\n")
